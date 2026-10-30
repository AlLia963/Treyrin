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

CHECKPOINT_LAYERS = True
CORE_CONFIG = {'abstain_read': True, 'adaptive_retrieval_k': True, 'amygdala': True, 'archive_slots': 8192, 'arousal_gating': True, 'bg_thalamic_loop': True, 'body_drift_coupling': True, 'body_rounds': 1, 'cerebellar_thalamic_gain': True, 'cerebellum': True, 'code_attention_hops': 2, 'cognitive_clock': 'event', 'cognitive_workspace': True, 'competitive_retrieval': True, 'contextual_goal_control': True, 'cortical_control_calibration': True, 'cortical_goal_control': True, 'cortical_streams': True, 'coupling_mode': 'layer_interleaved', 'deep_native_field': True, 'dopamine_mb_target': True, 'dopamine_plasticity': True, 'dopamine_write_gate': True, 'drives': True, 'dual_condition_write': True, 'echo_slots': 127, 'ema_anchor_consolidation': True, 'episode_slots': 8, 'episodic_policy': True, 'evict_consolidate': True, 'goal_gated_retrieval': True, 'goal_joint_control': True, 'goal_joint_hidden': 16, 'hippocampal_bg_loop': True, 'hippocampal_circuit': True, 'hypothesis_mode': 'adaptive', 'judgment_mode': 'separated', 'language_prior_gain': 1.0, 'latent_candidates': 4, 'layer_sites': [7, 15, 23, 31], 'mental_archive_slots': 16384, 'native_body_bridge': True, 'native_single_pass': True, 'neuromodulation': True, 'per_episode_gate': True, 'perception_memory': True, 'preserve_free_capacity': True, 'regional_circuit': True, 'response_cognitive_stride': 4, 'semantic_slots': 4, 'shared_choice_readout': True, 'thalamic_gate': True, 'token_code_attention': True, 'valence_dopamine': True, 'width': 32, 'workspace_slots': 8}
LORA = {'lora_alpha': 4, 'lora_dropout': 0.0, 'r': 2, 'target_modules': ['out_proj', 'gate_proj', 'up_proj', 'down_proj'], 'task_type': 'CAUSAL_LM'}
OUTPUT_ADAPTER_RANK = 2
STATE_SCHEMA = 'treyrin-state-v3'
STANDARD_MANIFEST = None
STANDARD_RESIDENT = None
STANDARD_TRAINABLE = 'model.safetensors'


class TreyrinConfig(PretrainedConfig):
    """Configuration for one complete Treyrin model (language body + graph)."""

    model_type = "treyrin"

    def __init__(
        self,
        base_model: str = "Qwen/Qwen3.5-4B",
        treyrin_checkpoint: str = "model-ready",
        window: int | None = None,
        capacity: int | None = None,
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
        self.window = int(window) if window is not None else None
        self.capacity = int(capacity) if capacity is not None else None
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
        if isinstance(kwargs.get("text_config"), dict):
            from transformers import Qwen3_5TextConfig
            kwargs["text_config"] = Qwen3_5TextConfig(**kwargs["text_config"])
        super().__init__(**kwargs)

    # -- recipe helpers -----------------------------------------------------
    def core_config_dict(self) -> dict:
        """Full CoreConfig dict, defaults included (what the binding hashes).

        Values are normalised through JSON so tuples and lists compare equal
        against the inherited candidate manifest.
        """
        import json

        CoreConfig = __import__("importlib").import_module("treyrin.structural_core").CoreConfig

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
