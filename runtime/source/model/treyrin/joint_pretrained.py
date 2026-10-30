"""Joint native/full-vocabulary training view of an existing SanSi wrapper.

Reuses its backbone and readout; adds no new memory or architectural novelty.
Requires a reviewed wrapper with _base/looped/_repeat/read and letter_ids.
"""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class JointPretrainedOutput:
    native_logits: tuple[torch.Tensor, ...]
    language_logits: torch.Tensor
    language_targets: torch.Tensor
    shared_decision_hidden: torch.Tensor


class JointPretrainedView(nn.Module):
    """One causal body computation supplies native and language training.

    Decision positions are the last PROMPT positions, even when teacher-forced
    answer tokens follow. Label construction belongs to a separate curriculum.
    Persistent experience state and cached decoding are not implemented here.
    The caller serializes execution; loop configuration is shared by the wrapper.
    """
    def __init__(self, native_wrapper: nn.Module):
        super().__init__()
        self.native = native_wrapper

    def _validate(self, input_ids, attention_mask, decision_positions, n_options,
                  language_prediction_mask, loops):
        if input_ids.ndim != 2 or input_ids.dtype != torch.long or min(input_ids.shape) < 1:
            raise ValueError('Expected a nonempty long-token matrix')
        batch, length = input_ids.shape
        if any(t.device != input_ids.device for t in
               (attention_mask, decision_positions, n_options, language_prediction_mask)):
            raise ValueError('All joint-batch tensors must share a device')
        if length < 2 or type(loops) is not int or loops < 1:
            raise ValueError('Joint training needs at least two tokens and one loop')
        if attention_mask.shape != input_ids.shape or not ((attention_mask == 0) | (attention_mask == 1)).all():
            raise ValueError('Binary token mask must align with input IDs')
        if decision_positions.shape != (batch,) or decision_positions.dtype != torch.long:
            raise ValueError('One last-prompt token position per batch row is required')
        if (decision_positions < 0).any() or (decision_positions >= length).any():
            raise ValueError('Decision position lies outside the token sequence')
        rows = torch.arange(batch, device=input_ids.device)
        if not attention_mask[rows, decision_positions].bool().all():
            raise ValueError('Decision positions cannot point into padding')
        if n_options.shape != (batch,) or n_options.dtype != torch.long:
            raise ValueError('One integer option count per row is required')
        if (n_options < 1).any() or (n_options > self.native.letter_ids.numel()).any():
            raise ValueError('Option count exceeds the native readout contract')
        if language_prediction_mask.shape != (batch, length - 1) or language_prediction_mask.dtype != torch.bool:
            raise ValueError('Language mask selects valid NEXT-token prediction positions')
        valid = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
        if not language_prediction_mask.any() or (language_prediction_mask & ~valid).any():
            raise ValueError('Select at least one language target, without padded positions')

    def _body_states(self, input_ids, attention_mask, loops, *, inputs_embeds=None):
        """Explicit embedding input also supports the recurrent-memory control.

        No hooks: checkpoint recomputation sees the same tensor input. Model loop
        configuration remains shared mutable state, so callers serialize calls.
        """
        base = self.native._base()
        if self.native.looped():
            base.model.total_ut_steps = base.config.total_ut_steps = base.model.config.total_ut_steps = loops
            kwargs = dict(input_ids=input_ids) if inputs_embeds is None else dict(inputs_embeds=inputs_embeds)
            _, states, _ = base.model(attention_mask=attention_mask, use_cache=False, **kwargs)
        elif inputs_embeds is None:
            states = self.native._repeat(base, input_ids, attention_mask, loops)
        else:
            states, hidden = [], inputs_embeds
            for _ in range(loops):
                hidden = base.model(inputs_embeds=hidden, attention_mask=attention_mask,
                                    use_cache=False).last_hidden_state
                states.append(hidden)
        if len(states) != loops:
            raise RuntimeError('Backbone did not return the requested loop states')
        return states

    def _read_states(self, states, input_ids, decision_positions, n_options,
                     language_prediction_mask):
        base = self.native._base()
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        native_logits = tuple(self.native.read(state[rows, decision_positions], t, n_options)
                              for t, state in enumerate(states))
        prediction_hidden = states[-1][:, :-1][language_prediction_mask]
        return JointPretrainedOutput(native_logits,
            base.lm_head(prediction_hidden), input_ids[:, 1:][language_prediction_mask],
            states[-1][rows, decision_positions])

    def forward(self, input_ids, attention_mask, decision_positions, n_options,
                language_prediction_mask, *, loops: int):
        self._validate(input_ids, attention_mask, decision_positions, n_options,
                       language_prediction_mask, loops)
        states = self._body_states(input_ids, attention_mask, loops)
        return self._read_states(states, input_ids, decision_positions, n_options,
                                 language_prediction_mask)
