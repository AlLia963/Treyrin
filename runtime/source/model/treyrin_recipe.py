"""Inherited recipes and identities of the v3 candidate (order 000063/000064).

Values here are *copied facts about the inherited candidate*, not new design
choices: they are the constants candidate.py used to build the graph that
produced model-ready (binding c39fa658..., contract b4151558...).  R1 keeps
them byte-exact so that a GPU load with the same code path still reproduces the
inherited binding; the CPU reference path records its own binding explicitly.
"""
from __future__ import annotations

import json
from pathlib import Path

from treyrin_paths import WORKSPACE, sha256_file

# --- inherited parameter / contract identity -------------------------------
INHERITED_BINDING = "c39fa658d5c15376c42b7cb2c5478f46b588a23e934cef277af7a1830ac41e1e"
INHERITED_CONTRACT = "b41515582dcfe4499b450eda0fcc9fd2921401f4df775bf1dbc01226f6605d7d"
INHERITED_OPTIMIZER_STEPS = 2351
INHERITED_TRAINABLE_TENSORS = 501
INHERITED_TRAINABLE_ELEMENTS = 3447187
INHERITED_RESIDENT_TENSORS = 76
INHERITED_INDEX_KEYS = 1239
INHERITED_FORMAL_RELEASE = False

# --- weight payload hashes (WEIGHT-CONTRACT.json / candidate manifest) -----
TRAINABLE_PAYLOAD_SHA256 = "c38ee19bbb40f7a58fac256d4df24d2ae588edaccb423e7e76918ca0bc4f5038"
RESIDENT_PAYLOAD_SHA256 = "463400734bc092e087c642c4122b901be91a5a3141e9d1d1fb1d17f09ab94633"
BASE_SHARD_SHA256 = {
    "model.safetensors-00001-of-00002.safetensors":
        "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
    "model.safetensors-00002-of-00002.safetensors":
        "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
}
BASE_INDEX_SHA256 = "cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d"
ASSET_MANIFEST_SHA256 = "5c5228a69c431f41c992cea1d79faa0390983f383653ef3256ab5a04f1c7e445"
ENCODING_SHA256 = "1bcad349c6b09516dd5d080c456743b5fef70c8af52f980ad49b5eb6ffcf5009"

# --- inherited graph construction constants (candidate.py) -----------------
CORE_CONFIG = {
    "width": 32, "episode_slots": 8, "semantic_slots": 4, "latent_candidates": 4,
    "cognitive_clock": "event", "language_prior_gain": 1., "perception_memory": True,
    "response_cognitive_stride": 4, "judgment_mode": "separated",
    "coupling_mode": "layer_interleaved", "body_rounds": 1,
    "layer_sites": (7, 15, 23, 31), "regional_circuit": True, "hypothesis_mode": "adaptive",
    "hippocampal_circuit": True, "thalamic_gate": True, "cortical_streams": True,
    "dopamine_plasticity": True, "cerebellum": True, "neuromodulation": True,
    "amygdala": True, "drives": True, "competitive_retrieval": True,
    "deep_native_field": True, "native_body_bridge": True, "dopamine_mb_target": True,
    "dual_condition_write": True, "bg_thalamic_loop": True, "valence_dopamine": True,
    "arousal_gating": True, "ema_anchor_consolidation": True, "body_drift_coupling": True,
    "hippocampal_bg_loop": True, "dopamine_write_gate": True, "cerebellar_thalamic_gain": True,
    "adaptive_retrieval_k": True, "archive_slots": 8192, "echo_slots": 127,
    "evict_consolidate": True, "per_episode_gate": True, "goal_gated_retrieval": True,
    "abstain_read": True, "episodic_policy": True, "cognitive_workspace": True,
    "workspace_slots": 8, "mental_archive_slots": 16384, "preserve_free_capacity": True,
    "token_code_attention": True, "native_single_pass": True, "goal_joint_control": True,
    "goal_joint_hidden": 16, "contextual_goal_control": True, "code_attention_hops": 2,
    "shared_choice_readout": True, "cortical_goal_control": True,
    "cortical_control_calibration": True,
}

LORA = {
    "r": 2,
    "lora_alpha": 4,
    "lora_dropout": 0.0,
    "target_modules": ["out_proj", "gate_proj", "up_proj", "down_proj"],
    "task_type": "CAUSAL_LM",
}
OUTPUT_ADAPTER_RANK = 2
CHECKPOINT_LAYERS = True

# Neutral retention probes used by the inherited learning contract.
PROBES = (
    "请简洁回答：一加一等于几？",
    "阅读记录时，应当区分已知事实和待验证的假设。",
)

STATE_SCHEMA = "treyrin-state-v3"
CANDIDATE_SCHEMA = "treyrin-formal-v1"
STANDARD_SCHEMA = "treyrin-standard-model-v1"
STANDARD_MANIFEST = "treyrin-manifest.json"
STANDARD_TRAINABLE = "treyrin-trainable.safetensors"
STANDARD_RESIDENT = "treyrin-resident.safetensors"

RUNTIME_PROTOCOL = "role-correct-single-user-v1"


def workspace_identity(workspace=None) -> dict:
    """Declared source identity of the inherited graph (fixes included)."""
    root = Path(workspace) if workspace is not None else WORKSPACE
    return {
        "native_dialogue_protocol": RUNTIME_PROTOCOL,
        "native_dialogue_source_sha256": sha256_file(root / "role_conversation.py"),
        "generation_transport_source_sha256": sha256_file(root / "generation_result.py"),
    }


def candidate_manifest(checkpoint_dir) -> dict:
    return json.loads((Path(checkpoint_dir) / "candidate-manifest.json").read_text("utf-8"))
