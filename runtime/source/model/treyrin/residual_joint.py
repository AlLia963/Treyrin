"""Known gated latent-read control inside a reviewed causal decoder layer.

Reuses the historical RecurrentMemoryBridge; not a novel operator or the final
Treyrin architecture. Public observations alone update state. Both native and
full-vocabulary queries read that state through the same modified decoder.
No prefix positions, KV cache, concurrent calls or checkpoint recomputation.
"""
import threading

import torch
from torch import nn

from .joint_pretrained import JointPretrainedView
from .recurrent_joint import RecurrentLatentState
from .seed_core import RecurrentMemoryBridge, MemoryState


class ResidualMemoryLayer(nn.Module):
    """A real decoder layer followed by the existing gated memory read."""
    def __init__(self, decoder, memory):
        super().__init__()
        self.decoder, self.memory = decoder, memory
        # Reviewed OuroModel chooses its causal mask from this layer metadata.
        # Do not broadly proxy unknown decoder attributes across architectures.
        if hasattr(decoder, 'attention_type'):
            self.attention_type = decoder.attention_type
        self.active_state = None
        self.last_hidden = None
        self.calls = 0

    def forward(self, *args, **kwargs):
        if self.active_state is None:
            raise RuntimeError('Use the explicit observation/query state interface')
        hidden = self.decoder(*args, **kwargs)
        if not isinstance(hidden, torch.Tensor):
            raise TypeError('Only audited Tensor-returning decoder layers are supported')
        hidden = self.memory.read_hidden(hidden, self.active_state)
        self.last_hidden = hidden
        self.calls += 1
        return hidden


class ResidualJointPretrainedView(JointPretrainedView):
    """Separate write/read operations with one shared internally modified body.

    The zero gate gives exact *initial* identity, including populated memory.
    It also makes the writer gradient zero at that point. Learning must open the
    gate before meaningful writer learning; this is not evidence of retention
    with useful memory. ``binding`` still needs caller-verified parameter and
    tokenizer identities plus re-encoding after parameter changes.
    """
    def __init__(self, native_wrapper, *, layer_index: int, memory_tokens: int,
                 memory_dim: int, binding: str, context_limit: int):
        super().__init__(native_wrapper)
        base = self.native._base()
        if type(layer_index) is not int or not 0 <= layer_index < len(base.model.layers):
            raise ValueError('Choose an existing reviewed decoder layer')
        if type(memory_tokens) is not int or memory_tokens < 1 or type(memory_dim) is not int or memory_dim < 4 or memory_dim%4:
            raise ValueError('Positive slot count and a four-head-compatible memory width required')
        if not isinstance(binding, str) or not binding:
            raise ValueError('Verified parameter/tokenizer identity required')
        if type(context_limit) is not int or not 2 <= context_limit <= base.config.max_position_embeddings:
            raise ValueError('Explicit local context budget required')
        if getattr(base, 'is_gradient_checkpointing', False):
            raise ValueError('Ephemeral state is incompatible with checkpoint recomputation')
        layer = base.model.layers[layer_index]
        if isinstance(layer, ResidualMemoryLayer) or hasattr(layer, 'active_state'):
            raise ValueError('Decoder already has a stateful wrapper')
        reference = next(layer.parameters())
        bridge = RecurrentMemoryBridge(base.config.hidden_size, memory_dim, memory_tokens).to(
            device=reference.device, dtype=reference.dtype)
        base.model.layers[layer_index] = ResidualMemoryLayer(layer, bridge)
        self.layer_index, self.memory_tokens, self.memory_dim = layer_index, memory_tokens, memory_dim
        self.binding = binding + f'|residual-joint-v1|layer={layer_index}|slots={memory_tokens}|width={memory_dim}'
        self.context_limit = context_limit
        self._execution_lock = threading.Lock()

    @property
    def memory_layer(self):
        return self.native._base().model.layers[self.layer_index]

    def initial_state(self, batch_size: int):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('Positive stream count required')
        state = self.memory_layer.memory.initial_state(batch_size)
        return RecurrentLatentState(state.slots, state.steps, self.binding)

    def _check_state(self, state, batch, length):
        state.validate()
        reference = self.memory_layer.memory.gate
        if state.binding != self.binding or state.tokens.shape != (batch, self.memory_tokens, self.memory_dim):
            raise ValueError('State mechanism, checkpoint or stream shape differs')
        if state.tokens.device != reference.device or state.tokens.dtype != reference.dtype:
            raise ValueError('State and model dtype/device differ')
        if length > self.context_limit:
            raise ValueError('Segment exceeds the explicit local context budget')

    def _with_state(self, input_ids, attention_mask, loops, state):
        base = self.native._base()
        if getattr(base, 'is_gradient_checkpointing', False):
            raise ValueError('Checkpoint recomputation is not implemented for this control')
        if not self._execution_lock.acquire(blocking=False):
            raise RuntimeError('Concurrent shared-body execution is not supported')
        layer = self.memory_layer
        layer.active_state = MemoryState(state.tokens, state.writes)
        layer.calls = 0
        try:
            states = super()._body_states(input_ids, attention_mask, loops)
            if layer.calls != loops or layer.last_hidden is None:
                raise RuntimeError('Selected decoder layer did not execute once per loop')
            return states, layer.last_hidden
        finally:
            layer.active_state = None
            layer.last_hidden = None
            self._execution_lock.release()

    def ingest(self, observation_ids, observation_mask, state, *, loops: int):
        if observation_ids.ndim != 2 or observation_ids.dtype != torch.long or min(observation_ids.shape) < 1:
            raise ValueError('Nonempty unpadded public observation IDs required')
        if observation_mask.shape != observation_ids.shape or observation_mask.device != observation_ids.device or not (observation_mask == 1).all():
            raise ValueError('Submit ragged observations separately; no silent padding')
        if type(loops) is not int or loops < 1:
            raise ValueError('Positive loop count required')
        self._check_state(state, observation_ids.shape[0], observation_ids.shape[1])
        if observation_ids.device != state.tokens.device:
            raise ValueError('Observation and state devices differ')
        if (state.writes == torch.iinfo(torch.long).max).any():
            raise ValueError('Write counter exhausted')
        _, observation_hidden = self._with_state(observation_ids, observation_mask, loops, state)
        written = self.memory_layer.memory.write_state(observation_hidden, MemoryState(state.tokens, state.writes))
        result = RecurrentLatentState(written.slots, written.steps, self.binding)
        result.validate()
        return result

    def forward(self, input_ids, attention_mask, decision_positions, n_options,
                language_prediction_mask, *, loops: int, state=None):
        self._validate(input_ids, attention_mask, decision_positions, n_options,
                       language_prediction_mask, loops)
        state = self.initial_state(input_ids.shape[0]) if state is None else state
        self._check_state(state, input_ids.shape[0], input_ids.shape[1])
        if input_ids.device != state.tokens.device:
            raise ValueError('Query and state devices differ')
        states, _ = self._with_state(input_ids, attention_mask, loops, state)
        return self._read_states(states, input_ids, decision_positions, n_options, language_prediction_mask)
