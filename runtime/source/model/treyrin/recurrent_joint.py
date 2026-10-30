"""RMT-style persistent-state control for joint native/language computation.

This is a known-method baseline, not the selected Treyrin architecture. A causal
decoder writes observation segments into recurrent latent tokens; both output
heads later read the same committed tokens inside that decoder. Queries cannot
commit teacher-forced answers back into experience. No KV cache or concurrency.
"""
from dataclasses import dataclass
import os
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save
import torch
from torch import nn

from .joint_pretrained import JointPretrainedView


@dataclass(frozen=True)
class RecurrentLatentState:
    tokens: torch.Tensor
    writes: torch.Tensor
    binding: str

    def validate(self):
        if not isinstance(self.binding, str) or not self.binding:
            raise ValueError('State requires a verified checkpoint/tokenizer/mechanism identity')
        if self.tokens.ndim != 3 or min(self.tokens.shape) < 1:
            raise ValueError('State must be a nonempty batch of memory-token vectors')
        if not self.tokens.is_floating_point() or not torch.isfinite(self.tokens).all():
            raise ValueError('Latent state must contain finite floating-point values')
        if self.writes.shape != self.tokens.shape[:1] or self.writes.dtype != torch.long:
            raise ValueError('One int64 write counter per stream is required')
        if self.writes.device != self.tokens.device or (self.writes < 0).any():
            raise ValueError('Counters must be nonnegative and colocated with state')

    def detached(self):
        """Break BPTT at an explicit boundary; no tensor alias with old state."""
        return RecurrentLatentState(self.tokens.detach().clone(),
                                    self.writes.detach().clone(), self.binding)

    def save(self, path: Path):
        """Exclusive snapshot creation. Existing snapshots are never overwritten."""
        self.validate()
        data = save({'tokens': self.tokens.detach().cpu().contiguous(),
                     'writes': self.writes.detach().cpu().contiguous()},
                    metadata={'format': 'treyrin-recurrent-joint-v1', 'binding': self.binding})
        with Path(path).open('xb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

    @classmethod
    def load(cls, path: Path, *, binding: str):
        with safe_open(path, framework='pt', device='cpu') as handle:
            if handle.metadata() != {'format': 'treyrin-recurrent-joint-v1', 'binding': binding}:
                raise ValueError('Checkpoint/tokenizer/mechanism identity or state format differs')
            if set(handle.keys()) != {'tokens', 'writes'}:
                raise ValueError('Unexpected persistent state fields')
            state = cls(handle.get_tensor('tokens'), handle.get_tensor('writes'), binding)
        state.validate()
        return state


class RecurrentJointPretrainedView(JointPretrainedView):
    """Observation-only writes and read-only, shared-body joint queries.

    ``binding`` is supplied from verified checkpoint identities by the experiment,
    not guessed from tensor dimensions. Updating parameters requires a newly
    reviewed binding/re-encoding policy. Initial memory prefixes change inference
    before training; neither zero tokens nor a fresh prefix imply old-skill parity.
    """
    def __init__(self, native_wrapper, *, memory_tokens: int, binding: str,
                 context_limit: int):
        super().__init__(native_wrapper)
        if type(memory_tokens) is not int or memory_tokens < 1 or not isinstance(binding, str) or not binding:
            raise ValueError('Positive memory-token count and checkpoint binding required')
        embedding = self.native._base().get_input_embeddings().weight
        if (type(context_limit) is not int or context_limit <= 2*memory_tokens or
                context_limit > self.native._base().config.max_position_embeddings):
            raise ValueError('Choose an explicit read/write context budget within model capacity')
        self.memory_tokens, self.binding = memory_tokens, binding
        self.context_limit = context_limit
        self.initial_memory = nn.Parameter(embedding.new_empty(memory_tokens, embedding.shape[1]))
        self.write_positions = nn.Parameter(torch.empty_like(self.initial_memory))
        scale = getattr(self.native._base().config, 'initializer_range', .02)
        nn.init.normal_(self.initial_memory, std=scale)
        nn.init.normal_(self.write_positions, std=scale)

    def initial_state(self, batch_size: int):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('Positive number of independent experience streams required')
        return RecurrentLatentState(self.initial_memory.unsqueeze(0).expand(batch_size, -1, -1).clone(),
            torch.zeros(batch_size, device=self.initial_memory.device, dtype=torch.long), self.binding)

    def _check_state(self, state, batch_size, length):
        state.validate()
        if state.binding != self.binding:
            raise ValueError('Latent state belongs to another checkpoint/mechanism')
        if state.tokens.shape != (batch_size, self.memory_tokens, self.initial_memory.shape[1]):
            raise ValueError('State batch, token count or hidden dimension differs')
        if state.tokens.device != self.initial_memory.device or state.tokens.dtype != self.initial_memory.dtype:
            raise ValueError('State and model must share dtype and device; no implicit conversion')
        if length > self.context_limit:
            raise ValueError('Memory plus segment exceeds the explicitly chosen local context budget')

    def ingest(self, observation_ids, observation_mask, state, *, loops: int):
        """Return a proposed next state; only public observation IDs belong here.

        This API cannot know which IDs are evaluator labels. The data boundary
        must enforce that separation. Ragged observation segments are submitted
        separately; silent truncation or padding inside writes is not supported.
        """
        if observation_ids.ndim != 2 or observation_ids.dtype != torch.long or min(observation_ids.shape) < 1:
            raise ValueError('Submit a nonempty unpadded observation-token matrix')
        if (observation_mask.shape != observation_ids.shape or
                observation_mask.device != observation_ids.device or
                not (observation_mask == 1).all()):
            raise ValueError('Observation writes require an all-valid mask; submit ragged rows separately')
        if type(loops) is not int or loops < 1:
            raise ValueError('Positive loop count required')
        batch, length = observation_ids.shape
        self._check_state(state, batch, length + 2 * self.memory_tokens)
        if observation_ids.device != self.initial_memory.device:
            raise ValueError('Observation tokens must be on the model device')
        if (state.writes == torch.iinfo(torch.long).max).any():
            raise ValueError('Experience counter exhausted')
        embedding = self.native._base().get_input_embeddings()
        event = embedding(observation_ids)
        write = self.write_positions.unsqueeze(0).expand(batch, -1, -1)
        inputs = torch.cat((state.tokens, event, write), dim=1)
        mask = torch.ones(inputs.shape[:2], device=inputs.device, dtype=torch.long)
        hidden = self._body_states(None, mask, loops, inputs_embeds=inputs)[-1]
        next_state = RecurrentLatentState(hidden[:, -self.memory_tokens:].clone(),
                                          state.writes + 1, self.binding)
        next_state.validate()
        return next_state

    def forward(self, input_ids, attention_mask, decision_positions, n_options,
                language_prediction_mask, *, loops: int, state=None):
        self._validate(input_ids, attention_mask, decision_positions, n_options,
                       language_prediction_mask, loops)
        if state is None:
            if input_ids.shape[1] > self.context_limit:
                raise ValueError('Query exceeds the explicitly chosen local context budget')
            return super().forward(input_ids, attention_mask, decision_positions,
                                   n_options, language_prediction_mask, loops=loops)
        batch, length = input_ids.shape
        self._check_state(state, batch, length + self.memory_tokens)
        if input_ids.device != self.initial_memory.device:
            raise ValueError('Query tokens must be on the model device')
        embedding = self.native._base().get_input_embeddings()
        inputs = torch.cat((state.tokens, embedding(input_ids)), dim=1)
        prefix_mask = torch.ones((batch, self.memory_tokens), device=input_ids.device,
                                 dtype=attention_mask.dtype)
        states = self._body_states(None, torch.cat((prefix_mask, attention_mask), dim=1),
                                  loops, inputs_embeds=inputs)
        # Discard the read-only prefix from *both* outputs. Original token indices
        # and next-token targets remain aligned; no objective predicts memory.
        states = [hidden[:, self.memory_tokens:] for hidden in states]
        return self._read_states(states, input_ids, decision_positions, n_options,
                                 language_prediction_mask)
