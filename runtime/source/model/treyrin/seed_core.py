"""Experimental internal memory bridge for a trainable Qwen3-style decoder.

Unlike the GGUF readout, this modifies hidden states inside a decoder layer. Its
output is consumed by both the original LM head and a policy head. This is NOT
Titans/Hope, a biological circuit, or an adapter already trained on the local 4B.
Full-segment eager execution only: KV-cache generation and concurrent requests
must be implemented and verified before deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading

from safetensors import safe_open
from safetensors.torch import save_file
import torch
from torch import nn


@dataclass(frozen=True)
class MemoryState:
    slots: torch.Tensor
    steps: torch.Tensor

    def detached(self):
        return MemoryState(self.slots.detach(), self.steps.detach())

    def reset(self, mask: torch.Tensor):
        if mask.dtype != torch.bool or mask.shape != self.steps.shape:
            raise ValueError("Reset mask must select individual batch episodes")
        return MemoryState(self.slots.masked_fill(mask[:, None, None], 0),
                           self.steps.masked_fill(mask, 0))

    def save(self, path: Path, checkpoint_id: str):
        if not checkpoint_id:
            raise ValueError("State must be bound to a specific model checkpoint")
        save_file({"slots": self.slots.detach().cpu().contiguous(),
                   "steps": self.steps.detach().cpu().contiguous()}, str(path),
                  metadata={"format": "treyrin-seed-state-v1", "checkpoint_id": checkpoint_id})

    @classmethod
    def load(cls, path: Path, checkpoint_id: str):
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            if metadata != {"format": "treyrin-seed-state-v1", "checkpoint_id": checkpoint_id}:
                raise ValueError("Memory state belongs to another checkpoint or format")
            slots, steps = handle.get_tensor("slots"), handle.get_tensor("steps")
        if slots.ndim != 3 or steps.shape != (slots.shape[0],) or steps.dtype != torch.long:
            raise ValueError("Malformed memory state")
        if not slots.is_floating_point() or not torch.isfinite(slots).all() or (steps < 0).any():
            raise ValueError("Invalid memory content")
        return cls(slots, steps)


class RecurrentMemoryBridge(nn.Module):
    def __init__(self, hidden_size: int, memory_dim: int = 64, slots: int = 4):
        super().__init__()
        if memory_dim % 4:
            raise ValueError("Memory width must support four attention heads")
        self.hidden_size, self.memory_dim, self.slot_count = hidden_size, memory_dim, slots
        self.norm = nn.LayerNorm(hidden_size)
        self.query = nn.Linear(hidden_size, memory_dim)
        self.read = nn.MultiheadAttention(memory_dim, 4, batch_first=True)
        self.output = nn.Linear(memory_dim, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.zeros(hidden_size))
        self.addresses = nn.Parameter(torch.randn(slots, memory_dim) * .02)
        self.write_input = nn.Linear(hidden_size, memory_dim)
        self.write_attention = nn.MultiheadAttention(memory_dim, 4, batch_first=True)
        self.update = nn.GRUCell(memory_dim, memory_dim)

    def initial_state(self, batch_size: int, *, device=None, dtype=None):
        parameter = self.gate
        return MemoryState(torch.zeros(batch_size, self.slot_count, self.memory_dim,
                                       device=device or parameter.device, dtype=dtype or parameter.dtype),
                           torch.zeros(batch_size, dtype=torch.long, device=device or parameter.device))

    def read_hidden(self, hidden: torch.Tensor, state: MemoryState):
        query = self.query(self.norm(hidden))
        read, _ = self.read(query, state.slots, state.slots, need_weights=False)
        # With no prior episode, preserve the native path exactly even after
        # training biases. This is an empty-state invariant, not full retention.
        available = (state.steps > 0)[:, None, None]
        return hidden + torch.where(available, self.output(read) * self.gate.tanh(), 0.)

    def write_state(self, hidden: torch.Tensor, state: MemoryState):
        encoded = self.write_input(self.norm(hidden))
        query = state.slots + self.addresses[None, :, :]
        written, _ = self.write_attention(query, encoded, encoded, need_weights=False)
        updated = self.update(written.flatten(0, 1), state.slots.flatten(0, 1)).reshape_as(state.slots)
        return MemoryState(updated, state.steps + 1)


class MemoryDecoderLayer(nn.Module):
    def __init__(self, base: nn.Module, memory: RecurrentMemoryBridge):
        super().__init__()
        self.base, self.memory = base, memory
        self.active_state: MemoryState | None = None
        self.segment_hidden: torch.Tensor | None = None

    def forward(self, *args, **kwargs):
        if self.active_state is None:
            raise RuntimeError("Call SeedCore.forward_segment with explicit episode state")
        hidden = self.base(*args, **kwargs)
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("This prototype supports tensor-returning Qwen3 decoder layers")
        hidden = self.memory.read_hidden(hidden, self.active_state)
        self.segment_hidden = hidden
        return hidden


class SeedCore(nn.Module):
    """Insert a zero-gated recurrent bridge inside an existing decoder.

    Original layers are retained. A single fused hidden state feeds the native
    vocabulary logits and policy logits. No prompt or tool output is substituted.
    The caller owns episode state; layer-local references are cleared on return.
    """

    def __init__(self, backbone: nn.Module, layer_index: int = 0, memory_dim: int = 64,
                 slots: int = 4, actions: int = 8, freeze_backbone: bool = True):
        super().__init__()
        if freeze_backbone:
            backbone.requires_grad_(False)
        self.backbone, self.layer_index = backbone, layer_index
        hidden_size = backbone.config.hidden_size
        layer = backbone.model.layers[layer_index]
        if isinstance(layer, MemoryDecoderLayer):
            raise ValueError("Decoder layer already has a memory bridge")
        reference = next(layer.parameters())
        bridge = RecurrentMemoryBridge(hidden_size, memory_dim, slots).to(
            device=reference.device, dtype=reference.dtype)
        backbone.model.layers[layer_index] = MemoryDecoderLayer(layer, bridge)
        self.policy = nn.Linear(hidden_size, actions).to(device=reference.device, dtype=reference.dtype)
        self._execution_lock = threading.Lock()

    @property
    def layer(self) -> MemoryDecoderLayer:
        return self.backbone.model.layers[self.layer_index]

    def forward_segment(self, tokens: torch.Tensor, state: MemoryState | None = None):
        if tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] == 0:
            raise ValueError("A segment requires a non-empty unpadded token matrix")
        state = state if state is not None else self.layer.memory.initial_state(tokens.shape[0])
        if state.slots.shape != (tokens.shape[0], self.layer.memory.slot_count, self.layer.memory.memory_dim):
            raise ValueError("Memory layout does not match this model and episode batch")
        if state.steps.shape != (tokens.shape[0],) or state.steps.dtype != torch.long:
            raise ValueError("Invalid episode step counters")
        if not torch.isfinite(state.slots).all() or (state.steps < 0).any():
            raise ValueError("Invalid episode memory content")
        if state.slots.device != tokens.device or state.steps.device != tokens.device:
            raise ValueError("Tokens and persistent state must be on the same device")
        if not self._execution_lock.acquire(blocking=False):
            raise RuntimeError("Concurrent forwards are not supported by this research prototype")
        self.layer.active_state = state
        try:
            output = self.backbone.model(input_ids=tokens, use_cache=False)
            hidden = output.last_hidden_state
            next_state = self.layer.memory.write_state(self.layer.segment_hidden, state)
            return {"language_logits": self.backbone.lm_head(hidden),
                    "policy_logits": self.policy(hidden[:, -1]),
                    "shared_hidden": hidden[:, -1], "state": next_state}
        finally:
            self.layer.active_state = None
            self.layer.segment_hidden = None
            self._execution_lock.release()
