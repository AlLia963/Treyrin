"""Standard configuration of the unified Treyrin model (R1, order 000064).

Backwards compatible with the v3 config.json (base_model, treyrin_checkpoint,
window, capacity, state_schema).  The additional fields bind the *actual*
module layout that R1 registers: the frozen local language body, the LoRA
adaptation, the output adapter and the full brain/graph recipe.

The authoritative graph recipe is the inherited candidate manifest; a config
that carries its own core_config is cross-checked against the manifest and the
load fails on any difference.
"""
from __future__ import annotations

import dataclasses
from typing import Any

from transformers import PretrainedConfig

from treyrin_paths import bootstrap

bootstrap()

from treyrin_recipe import (
    CHECKPOINT_LAYERS, CORE_CONFIG, LORA, OUTPUT_ADAPTER_RANK, STATE_SCHEMA,
    STANDARD_MANIFEST, STANDARD_RESIDENT, STANDARD_TRAINABLE,
)


class TreyrinConfig(PretrainedConfig):
    """Configuration for one complete Treyrin model (language body + graph)."""

    model_type = "treyrin"

    def __init__(
        self,
        base_model: str = "Qwen/Qwen3.5-4B",
        treyrin_checkpoint: str = "model-ready",
        window: int = 8192,
        capacity: int = 2048,
        state_schema: str = STATE_SCHEMA,
        *,
        base_body: str = "assets/body",
        tokenizer: str = "assets/tokenizer",
        core_config: dict | None = None,
        lora: dict | None = None,
        output_adapter_rank: int = OUTPUT_ADAPTER_RANK,
        checkpoint_layers: bool = CHECKPOINT_LAYERS,
        body_dtype: str = "bfloat16",
        strict_identity: bool = True,
        manifest_name: str = STANDARD_MANIFEST,
        trainable_name: str = STANDARD_TRAINABLE,
        resident_name: str = STANDARD_RESIDENT,
        **kwargs: Any,
    ) -> None:
        self.base_model = base_model
        self.treyrin_checkpoint = treyrin_checkpoint
        self.window = int(window)
        self.capacity = int(capacity)
        self.state_schema = state_schema
        self.base_body = base_body
        self.tokenizer = tokenizer
        self.core_config = dict(core_config) if core_config is not None else dict(CORE_CONFIG)
        self.lora = dict(lora) if lora is not None else dict(LORA)
        self.output_adapter_rank = int(output_adapter_rank)
        self.checkpoint_layers = bool(checkpoint_layers)
        self.body_dtype = body_dtype
        self.strict_identity = bool(strict_identity)
        self.manifest_name = manifest_name
        self.trainable_name = trainable_name
        self.resident_name = resident_name
        super().__init__(**kwargs)

    # -- recipe helpers -----------------------------------------------------
    def core_config_dict(self) -> dict:
        """Full CoreConfig dict, defaults included (what the binding hashes).

        Values are normalised through JSON so tuples and lists compare equal
        against the inherited candidate manifest.
        """
        import json

        from treyrin.structural_core import CoreConfig

        raw = dataclasses.asdict(CoreConfig(**self.core_config))
        return json.loads(json.dumps(raw))

    def lora_config(self):
        from peft import LoraConfig

        return LoraConfig(**self.lora)

    def resolved_core_config(self, declared: dict | None = None) -> dict:
        """Cross-check the config recipe against the checkpoint manifest."""
        own = self.core_config_dict()
        if declared is not None:
            declared = dict(declared)
            if declared != own:
                differing = sorted(
                    key for key in set(own) | set(declared) if own.get(key) != declared.get(key)
                )
                raise ValueError(
                    "TreyrinConfig.core_config disagrees with the checkpoint manifest on "
                    + ", ".join(differing)
                )
        return own
