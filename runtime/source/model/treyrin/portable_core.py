"""Backbone-independent reference state rules, not a complete Treyrin model.

Inputs are explicit structured features. Learned hidden-space adapters and joint
language/world/policy training remain separate experiments. A state is bound to
the feature codec and kernel, even if its tensor shape happens to match another.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import save_file


@dataclass(frozen=True)
class CoreBinding:
    mechanism: str
    feature_codec: str
    key_width: int
    value_width: int
    rule_version: str = "reference-v1"

    def __post_init__(self):
        if not self.mechanism or not self.feature_codec or not self.rule_version:
            raise ValueError("Mechanism, codec and rule version must be explicit")
        if self.key_width <= 0 or self.value_width <= 0:
            raise ValueError("Feature widths must be positive")

    @property
    def identity(self):
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()


@dataclass(frozen=True)
class BoundState:
    memory: torch.Tensor
    updates: torch.Tensor
    binding_id: str

    def reset(self, selected: torch.Tensor):
        if selected.dtype != torch.bool or selected.shape != self.updates.shape:
            raise ValueError("Reset mask must select individual episodes")
        if selected.device != self.memory.device:
            raise ValueError("Reset mask and memory must use the same device")
        return BoundState(self.memory.masked_fill(selected[:, None, None], 0),
                          self.updates.masked_fill(selected, 0), self.binding_id)

    def permuted(self):
        return BoundState(self.memory.roll(1, 0), self.updates.roll(1, 0), self.binding_id)

    def save(self, path: Path):
        save_file({"memory": self.memory.detach().cpu().contiguous(),
                   "updates": self.updates.detach().cpu().contiguous()}, str(path),
                  metadata={"format": "treyrin-portable-state-v1", "binding_id": self.binding_id})


class AssociativeReference(nn.Module):
    """Explicit, untrained delta or additive kernel used to test state semantics.

    This is an algebraic reference with no trained parameters. A learned adapter
    must add its own checkpoint identity to feature_codec; dimensions alone are
    insufficient to reuse state. Updates are functional and batch-isolated.
    """
    def __init__(self, binding: CoreBinding):
        super().__init__()
        if binding.mechanism not in {"delta_reference", "additive_reference"}:
            raise ValueError("Unsupported reference rule")
        self.binding = binding

    def initial_state(self, batch_size: int, *, device="cpu", dtype=torch.float32):
        if batch_size <= 0:
            raise ValueError("Batch must contain an episode")
        return BoundState(torch.zeros(batch_size, self.binding.key_width,
                                      self.binding.value_width, device=device, dtype=dtype),
                          torch.zeros(batch_size, device=device, dtype=torch.long),
                          self.binding.identity)

    def validate(self, state: BoundState):
        if state.binding_id != self.binding.identity:
            raise ValueError("Explicit migration or event replay required: state binding differs")
        if state.memory.ndim != 3:
            raise ValueError("Memory must have batch/key/value axes")
        shape = (state.memory.shape[0], self.binding.key_width, self.binding.value_width)
        if state.memory.shape != shape:
            raise ValueError("Memory layout differs from bound contract")
        if state.updates.shape != (shape[0],) or state.updates.dtype != torch.long:
            raise ValueError("Invalid update counters")
        if state.memory.device != state.updates.device:
            raise ValueError("State tensors must share a device")
        if not state.memory.is_floating_point() or not torch.isfinite(state.memory).all():
            raise ValueError("Invalid memory values")
        if (state.updates < 0).any():
            raise ValueError("Negative update counters")

    def _keys(self, keys, state):
        self.validate(state)
        if keys.shape != (state.memory.shape[0], self.binding.key_width):
            raise ValueError("Key batch or width mismatch")
        if keys.device != state.memory.device or keys.dtype != state.memory.dtype:
            raise ValueError("Key device/dtype mismatch")
        if not torch.isfinite(keys).all():
            raise ValueError("Nonfinite key")
        norms = keys.square().sum(-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=2e-5, rtol=2e-5):
            raise ValueError("Reference update requires unit-length keys")

    def step(self, key: torch.Tensor, value: torch.Tensor, state: BoundState,
             write_mask: torch.Tensor | None = None):
        self._keys(key, state)
        if value.shape != (key.shape[0], self.binding.value_width):
            raise ValueError("Value width or batch mismatch")
        if value.device != key.device or value.dtype != key.dtype or not torch.isfinite(value).all():
            raise ValueError("Invalid value features")
        if write_mask is None:
            write_mask = torch.ones(key.shape[0], dtype=torch.bool, device=key.device)
        if write_mask.shape != state.updates.shape or write_mask.dtype != torch.bool:
            raise ValueError("Invalid write mask")
        if write_mask.device != key.device:
            raise ValueError("Write mask device mismatch")
        predicted = torch.einsum("bk,bkv->bv", key, state.memory)
        error = value - predicted if self.binding.mechanism == "delta_reference" else value
        change = key[:, :, None] * error[:, None, :] * write_mask[:, None, None]
        return BoundState(state.memory + change, state.updates + write_mask.long(), state.binding_id)

    def read(self, key: torch.Tensor, state: BoundState):
        self._keys(key, state)
        return torch.einsum("bk,bkv->bv", key, state.memory)

    def load_state(self, path: Path, *, device="cpu"):
        with safe_open(path, framework="pt", device="cpu") as handle:
            if handle.metadata() != {"format": "treyrin-portable-state-v1",
                                     "binding_id": self.binding.identity}:
                raise ValueError("Checkpoint/codec migration required before state loading")
            state = BoundState(handle.get_tensor("memory").to(device),
                               handle.get_tensor("updates").to(device), self.binding.identity)
        self.validate(state)
        return state
