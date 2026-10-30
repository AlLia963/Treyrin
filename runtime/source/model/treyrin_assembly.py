"""Strict assembly of the standard Treyrin model from the inherited candidate.

R1 of order 000064.  This module is the single place that materialises the real
module tree of TreyrinForCausalLM:

    language body (local Qwen3.5-4B text decoder, frozen)
      + always-on LoRA factors (peft, inherited r=2/alpha=4 recipe)
      + output adaptation (readout-only rank-2 adapter)
      + bridge (body hidden <-> core width, scalar feedback gain)
      + brain core (StructuralCore from the inherited recipe)

and it loads the inherited payload (model-ready) under strict identity checks:

* declared workspace sources unchanged for every source that participates in the
  parameter binding / learning contract,
* frozen asset manifest hash, tokenizer encoding identity and per-file tokenizer
  hashes,
* every tensor of the language body present in the real local shards with the
  exact shape/dtype and no unexpected text tensor,
* the 501 trainable tensors and the 76 resident state tensors loaded by exact
  key set, shape, dtype and finiteness; payload files hash-verified,
* the resulting parameter binding compared to the inherited binding when the
  execution context can reproduce it (CUDA + NF4), and recorded with an explicit
  reason when it cannot (CPU bf16 reference path).

No training, no GPU kernels beyond the inherited execution path, no mutation of
the inherited sources.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from treyrin_paths import (
    MODEL_DIR, WORKSPACE, asset_identity, bootstrap, canonical, sha256_canonical,
    sha256_file, tokenizer_file_hashes, tokenizer_identity,
)

bootstrap()

from configuration_treyrin import TreyrinConfig
from treyrin_recipe import (
    ASSET_MANIFEST_SHA256, BASE_INDEX_SHA256, BASE_SHARD_SHA256, CANDIDATE_SCHEMA,
    CHECKPOINT_LAYERS, CORE_CONFIG, ENCODING_SHA256, INHERITED_BINDING,
    INHERITED_CONTRACT, INHERITED_INDEX_KEYS, INHERITED_OPTIMIZER_STEPS,
    INHERITED_RESIDENT_TENSORS, INHERITED_TRAINABLE_ELEMENTS,
    INHERITED_TRAINABLE_TENSORS, LORA, OUTPUT_ADAPTER_RANK, PROBES,
    RESIDENT_PAYLOAD_SHA256, STATE_SCHEMA, STANDARD_MANIFEST, STANDARD_RESIDENT,
    STANDARD_SCHEMA, STANDARD_TRAINABLE, TRAINABLE_PAYLOAD_SHA256,
    workspace_identity,
)

# Sources whose bytes decide the parameter binding and the learning contract
# (seal_parameters / _execution_configuration / StructuralLearningSession).
BINDING_CRITICAL_SOURCES = (
    "modeling_treyrin.py",
    "native_port.py",
    "public_labels.py",
    "role_conversation.py",
    "generation_result.py",
    "treyrin/structural_model.py",
    "treyrin/structural_core.py",
    "treyrin/decision_field.py",
    "treyrin/interleaved_body.py",
    "treyrin/regional_circuit.py",
    "treyrin/hypothesis_population.py",
    "treyrin/hippocampal_circuit.py",
    "treyrin/r8_circuits.py",
    "treyrin/qwen_interleaved.py",
    "treyrin/episodic_policy.py",
    "treyrin/cognitive_workspace.py",
    "treyrin/output_adaptation.py",
    "treyrin/learning_checkpoint.py",
    "treyrin/retention_reference.py",
)


class TreyrinStrictLoadError(RuntimeError):
    """Raised when the inherited model identity or payload does not match."""


class _Checks:
    """Collect named pass/fail checks and fail loudly at the end."""

    def __init__(self) -> None:
        self.items: list[dict] = []

    def check(self, name: str, ok: bool, detail=None) -> bool:
        self.items.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            raise TreyrinStrictLoadError(f"{name} failed: {detail}")
        return True

    def observe(self, name: str, ok: bool, detail=None) -> bool:
        self.items.append({"check": name, "ok": bool(ok), "detail": detail, "informational": True})
        return ok

    def as_list(self) -> list[dict]:
        return list(self.items)


def _require(condition, message: str) -> None:
    if not condition:
        raise TreyrinStrictLoadError(message)


def read_safetensors_header(path) -> dict[str, tuple[tuple[int, ...], str]]:
    """(shape, dtype) per tensor from the safetensors header, no data read."""
    with Path(path).open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length).decode("utf-8"))
    return {
        key: (tuple(value["shape"]), value["dtype"])
        for key, value in header.items()
        if key != "__metadata__"
    }


_DTYPE_NAMES = {
    torch.bfloat16: "BF16",
    torch.float16: "F16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}


def _dtype_matches(tensor_dtype, header_dtype: str) -> bool:
    name = _DTYPE_NAMES.get(tensor_dtype)
    if name is None:
        return True
    return name == header_dtype


def base_body_index(base_body_dir) -> dict:
    """The real local shard layout of the frozen language body."""
    root = Path(base_body_dir)
    index_path = root / "model.safetensors.index.json"
    _require(index_path.is_file(), f"missing base body index: {index_path}")
    index = json.loads(index_path.read_text("utf-8"))
    weight_map = index["weight_map"]
    shards = sorted({Path(value).name for value in weight_map.values()})
    headers = {shard: read_safetensors_header(root / shard) for shard in shards}
    layout = {}
    for key, shard in weight_map.items():
        shape, dtype = headers[Path(shard).name][key]
        layout[key] = {"shard": Path(shard).name, "shape": shape, "dtype": dtype}
    return {"path": str(root), "index_sha256": sha256_file(index_path), "shards": shards,
            "files": {name: sha256_file(root / name) for name in
                      [index_path.name, *shards]},
            "weights": layout}


def map_text_checkpoint_keys(base_layout: dict) -> dict:
    """Documented ConditionalGeneration -> text decoder mapping.

    Only the text decoder tensors of the local checkpoint are consumed; the
    vision tower and the multi-token-prediction head are not part of the
    reviewed text body and are reported, never silently ignored as unknown.
    """
    mapped: dict[str, str] = {}
    skipped: dict[str, str] = {}
    unknown: dict[str, str] = {}
    for key in base_layout["weights"]:
        if key.startswith("model.language_model."):
            mapped["model." + key[len("model.language_model."):]] = key
        elif key.startswith("model.visual."):
            skipped[key] = "vision tower (not in the reviewed text body)"
        elif key.startswith("mtp."):
            skipped[key] = "multi-token-prediction head (not in the reviewed text body)"
        else:
            unknown[key] = "unmapped checkpoint tensor"
    return {"mapped": mapped, "skipped": skipped, "unknown": unknown}


# ---------------------------------------------------------------------------
# Inherited identity verification
# ---------------------------------------------------------------------------
def verify_assets(assets_dir, *, mode: str = "full") -> dict:
    """Verify the frozen language assets against their own manifest.

    mode='full' hashes every asset file (the inherited candidate.py behaviour;
    it streams ~9.3 GB and is the R1 proof).  mode='manifest' verifies only the
    manifest file itself, which is what repeated training loads need.
    """
    assets = Path(assets_dir)
    manifest_path = assets / "asset-manifest.json"
    _require(manifest_path.is_file(), f"missing asset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    _require(manifest.get("schema") == "treyrin-local-body-assets-v1",
             f"unexpected asset manifest schema: {manifest.get('schema')}")
    manifest_sha = sha256_file(manifest_path)
    _require(manifest_sha == ASSET_MANIFEST_SHA256,
             f"asset manifest hash {manifest_sha} != inherited {ASSET_MANIFEST_SHA256}")
    files = {}
    if mode == "full":
        for name, expected in manifest["files"].items():
            path = assets / name
            _require(path.is_file(), f"missing asset file: {name}")
            actual = sha256_file(path)
            _require(actual == expected, f"asset file hash differs: {name}")
            files[name] = {"sha256": actual, "bytes": path.stat().st_size}
    elif mode != "manifest":
        raise ValueError(f"unknown asset verification mode: {mode}")
    return {"mode": mode, "manifest_sha256": manifest_sha, "commit": manifest.get("commit"),
            "upstream": manifest.get("upstream"), "declared_files": len(manifest["files"]),
            "verified_files": files}


def verify_sources(root, declared: dict, *, strict: bool = True) -> dict:
    """Compare the actual inherited sources with the candidate declaration."""
    root = Path(root)
    changed, missing = [], []
    for name, expected in declared.items():
        path = root / name
        if not path.is_file():
            missing.append(name)
        elif sha256_file(path) != expected:
            changed.append(name)
    critical_missing = [n for n in BINDING_CRITICAL_SOURCES if n in missing]
    critical_changed = [n for n in BINDING_CRITICAL_SOURCES if n in changed]
    declared_critical = [n for n in BINDING_CRITICAL_SOURCES if n in declared]
    result = {
        "declared_files": len(declared),
        "missing": sorted(missing),
        "changed": sorted(changed),
        "binding_critical_declared": len(declared_critical),
        "binding_critical_missing": critical_missing,
        "binding_critical_changed": critical_changed,
        "declared_sources_intact": not missing and not changed,
    }
    if strict:
        _require(not critical_missing and not critical_changed,
                 "binding-critical inherited source changed: "
                 + ", ".join(sorted(critical_missing + critical_changed)))
        _require(not critical_missing,
                 "binding-critical inherited source missing: " + ", ".join(critical_missing))
    return result


def verify_candidate(checkpoint_dir, *, assets_dir=None, tokenizer_dir=None,
                     verify_assets_mode: str = "full", strict: bool = True,
                     expect_inherited: bool = True) -> dict:
    """Verify one inherited model-ready payload and its declared identity."""
    checkpoint = Path(checkpoint_dir)
    manifest_path = checkpoint / "candidate-manifest.json"
    _require(manifest_path.is_file(), f"missing candidate manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    root = checkpoint.parent
    checks = _Checks()

    checks.check("candidate.schema", manifest.get("schema") == CANDIDATE_SCHEMA,
                 manifest.get("schema"))
    if expect_inherited:
        checks.check("candidate.parameter_binding",
                     manifest.get("parameter_binding") == INHERITED_BINDING,
                     manifest.get("parameter_binding"))
        checks.check("candidate.contract_identity",
                     manifest.get("contract_identity") == INHERITED_CONTRACT,
                     manifest.get("contract_identity"))
        checks.check("candidate.optimizer_steps",
                     manifest.get("optimizer_steps") == INHERITED_OPTIMIZER_STEPS,
                     manifest.get("optimizer_steps"))
    checks.check("candidate.formal_release_false",
                 manifest.get("capability_accepted") is False,
                 manifest.get("capability_accepted"))

    assets = Path(assets_dir) if assets_dir is not None else root / "assets"
    tokenizer = Path(tokenizer_dir) if tokenizer_dir is not None else assets / "tokenizer"
    asset_report = verify_assets(assets, mode=verify_assets_mode)
    checks.check("assets.identity", asset_report["manifest_sha256"] == manifest["assets_sha256"],
                 {"actual": asset_report["manifest_sha256"],
                  "declared": manifest["assets_sha256"]})
    encoding = tokenizer_identity(assets)
    checks.check("tokenizer.encoding_identity", encoding == manifest["encoding_sha256"],
                 {"actual": encoding, "declared": manifest["encoding_sha256"]})
    if expect_inherited:
        checks.check("tokenizer.inherited_identity", encoding == ENCODING_SHA256, encoding)
    declared_tokenizer = json.loads((assets / "asset-manifest.json").read_text("utf-8"))["files"]
    for name, actual in tokenizer_file_hashes(assets).items():
        checks.check(f"tokenizer.file:{name}", declared_tokenizer.get(name) == actual,
                     {"actual": actual, "declared": declared_tokenizer.get(name)})

    payload = {}
    for name, expected in manifest["files"].items():
        path = checkpoint / name
        checks.check(f"payload.present:{name}", path.is_file(), str(path))
        actual = sha256_file(path)
        checks.check(f"payload.sha256:{name}", actual == expected,
                     {"actual": actual, "declared": expected})
        payload[name] = {"sha256": actual, "bytes": path.stat().st_size}
    checks.check("payload.trainable_contract",
                 payload["session/learning/trainable.safetensors"]["sha256"] == TRAINABLE_PAYLOAD_SHA256,
                 payload["session/learning/trainable.safetensors"]["sha256"])
    checks.check("payload.resident_contract",
                 payload["resident.safetensors"]["sha256"] == RESIDENT_PAYLOAD_SHA256,
                 payload["resident.safetensors"]["sha256"])

    sources = verify_sources(root, manifest.get("code", {}), strict=strict)
    base = base_body_index(assets / "body")
    mapped = map_text_checkpoint_keys(base)
    checks.check("base_body.index_sha256", base["index_sha256"] == BASE_INDEX_SHA256,
                 base["index_sha256"])
    for shard, expected in BASE_SHARD_SHA256.items():
        checks.check(f"base_body.shard:{shard}", base["files"].get(shard) == expected,
                     base["files"].get(shard))
    checks.check("base_body.only_documented_skips", not mapped["unknown"],
                 sorted(mapped["unknown"])[:8])
    checks.check("base_body.text_tensors", len(mapped["mapped"]) == 426,
                 len(mapped["mapped"]))
    return {
        "schema": "treyrin-inherited-identity-v1",
        "checkpoint": str(checkpoint),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest,
        "checks": checks.as_list(),
        "assets": asset_report,
        "sources": sources,
        "payload": payload,
        "base_body": {"index_sha256": base["index_sha256"], "shards": base["shards"],
                      "text_tensors": len(mapped["mapped"]),
                      "skipped_tensors": len(mapped["skipped"]),
                      "skipped_reasons": sorted(set(mapped["skipped"].values()))},
        "identity": {
            "parameter_binding": manifest["parameter_binding"],
            "contract_identity": manifest["contract_identity"],
            "optimizer_steps": manifest["optimizer_steps"],
            "encoding_sha256": manifest["encoding_sha256"],
            "assets_sha256": manifest["assets_sha256"],
            "trainable_sha256": payload["session/learning/trainable.safetensors"]["sha256"],
            "resident_sha256": payload["resident.safetensors"]["sha256"],
            "declared_sources": len(manifest.get("code", {})),
        },
    }


# ---------------------------------------------------------------------------
# Real module construction
# ---------------------------------------------------------------------------
def _torch_dtype(name: str):
    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if name not in mapping:
        raise ValueError(f"unsupported body dtype: {name}")
    return mapping[name]


def build_language_body(base_body_dir, *, device: str = "cpu", dtype="bfloat16",
                        quantize: str | None = None, asset_commit: str | None = None) -> tuple[Any, dict]:
    """Load the local language body as a real nn.Module.

    device='cpu' or quantize=None keeps the frozen bf16 weights file-mapped
    (no RAM copy); device='cuda' requires quantize='nf4' (the inherited
    deployment representation, 4.7 GB).
    """
    from transformers import AutoConfig, GenerationConfig, Qwen3_5ForCausalLM

    root = Path(base_body_dir)
    body_dtype = _torch_dtype(dtype)
    if device == "cuda" and quantize != "nf4":
        raise ValueError("CUDA body loading requires the inherited NF4 representation")
    if quantize not in (None, "nf4"):
        raise ValueError(f"unsupported quantization: {quantize}")
    config = AutoConfig.from_pretrained(root, local_files_only=True).text_config
    config._attn_implementation = "eager"
    generation = GenerationConfig.from_pretrained(root, local_files_only=True)
    if quantize == "nf4":
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16)
        base, info = Qwen3_5ForCausalLM.from_pretrained(
            root, config=config, generation_config=generation, dtype=torch.bfloat16,
            quantization_config=quant, device_map={"": "cuda:0"},
            local_files_only=True, output_loading_info=True)
    else:
        base, info = Qwen3_5ForCausalLM.from_pretrained(
            root, config=config, generation_config=generation, dtype=body_dtype,
            low_cpu_mem_usage=True, local_files_only=True, output_loading_info=True)
        if device != "cpu":
            base = base.to(device)
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        values = info.get(key) or []
        if values:
            sample = sorted(map(str, values))[:6]
            raise TreyrinStrictLoadError(f"language body load reported {key}: {sample}")
    commit = asset_commit or ""
    base.config._name_or_path = "treyrin-local-qwen3.5-4b@" + str(commit)
    base.eval().requires_grad_(False)
    tensors = sum(1 for _ in base.parameters())
    elements = sum(p.numel() for p in base.parameters())
    first = next(base.parameters())
    return base, {
        "class": type(base).__name__,
        "decoder_class": type(base.base_model).__name__,
        "layers": len(base.base_model.layers),
        "dtype": str(first.dtype),
        "device": str(first.device),
        "parameters": tensors,
        "elements": elements,
        "quantization": quantize,
        "loading_info_empty": True,
        "layers_file_mapped": quantize is None,
    }


def choice_label_tokens(tokenizer):
    """One-token public choice alphabet, as the inherited body adapter expects."""
    from public_labels import ALPHABET

    encoded = [tokenizer(label, add_special_tokens=False)["input_ids"] for label in ALPHABET]
    bad = [label for label, ids in zip(ALPHABET, encoded) if len(ids) != 1]
    if bad:
        raise TreyrinStrictLoadError(f"public label alphabet is not one token per label: {bad}")
    return [ids[0] for ids in encoded]


def inject_lora(base, lora_config: dict):
    """Always-on LoRA on the reviewed module list (inherited recipe)."""
    from peft import LoraConfig, get_peft_model

    adapted = get_peft_model(base, LoraConfig(**lora_config))
    adapted.eval()
    trainable = [name for name, p in adapted.named_parameters() if p.requires_grad]
    leaked = [name for name in trainable if ".lora_" not in name]
    if leaked:
        raise TreyrinStrictLoadError(f"non-LoRA language parameters became trainable: {leaked[:6]}")
    return adapted, {"trainable_tensors": len(trainable), "config": dict(lora_config)}


def build_retention_reference(body, *, tokenizer, asset_identity_value: str, device: str):
    """Frozen loss-side reference built BEFORE adapter injection.

    The inherited candidate cached the two neutral probe projections of the
    unadapted local body before get_peft_model, and the cached identity includes
    exactly that parameter tree.  Building it here keeps the inherited learning
    contract reproducible on the execution path that produced it.
    """
    from treyrin.retention_reference import CachedRetentionReference
    from treyrin.structural_model import HFBodyPort

    labels = choice_label_tokens(tokenizer)
    port = HFBodyPort(body, asset_identity_value, choice_label_token_ids=labels).eval()
    probes = [torch.tensor([tokenizer(text, add_special_tokens=False)["input_ids"]],
                           device=device) for text in PROBES]
    reference = CachedRetentionReference(port, [(ids, torch.ones_like(ids)) for ids in probes])
    return reference, port


def build_graph(adapted_body, *, tokenizer, core_config: dict, asset_identity_value: str,
                checkpoint_layers: bool = True, output_adapter_rank: int = OUTPUT_ADAPTER_RANK,
                workspace=None, device: str | None = None):
    """Build the inherited structural graph around the real language body."""
    from native_port import NativeBodyPort
    from treyrin.structural_core import CoreConfig
    from treyrin.structural_model import TreyrinStructuralModel

    labels = choice_label_tokens(tokenizer)
    port = NativeBodyPort(adapted_body, asset_identity_value,
                          checkpoint_layers=checkpoint_layers,
                          choice_label_token_ids=labels,
                          output_adapter_rank=output_adapter_rank)
    graph = TreyrinStructuralModel(port, CoreConfig(**core_config), freeze_body=None)
    graph = graph.eval()
    if device is not None:
        target = torch.device(device)
        if next(graph.core.parameters()).device != target:
            graph = graph.to(target).eval()
    graph.body.runtime_identity.update(workspace_identity(workspace))
    return graph, {"choice_labels": len(labels), "core_config": graph.config,
                   "output_adapter_rank": output_adapter_rank,
                   "checkpoint_layers": checkpoint_layers}


def trainable_parameters(graph) -> dict:
    return {name: parameter for name, parameter in graph.named_parameters()
            if parameter.requires_grad}


def load_trainable_strict(graph, path, *, expected_sha256: str | None = None,
                          device: str | None = None) -> dict:
    """Load the exact trainable tensor set; no key, shape or dtype may differ."""
    from safetensors.torch import load_file

    source = Path(path)
    _require(source.is_file(), f"missing trainable payload: {source}")
    actual_sha = sha256_file(source)
    if expected_sha256 is not None:
        _require(actual_sha == expected_sha256,
                 f"trainable payload hash {actual_sha} != {expected_sha256}")
    saved = load_file(source, device="cpu")
    expected = trainable_parameters(graph)
    missing = sorted(set(expected) - set(saved))
    unexpected = sorted(set(saved) - set(expected))
    _require(not missing, f"missing trainable tensors: {missing[:6]}")
    _require(not unexpected, f"unexpected trainable tensors: {unexpected[:6]}")
    for name, parameter in expected.items():
        value = saved[name]
        _require(tuple(value.shape) == tuple(parameter.shape),
                 f"shape differs for {name}: {tuple(value.shape)} != {tuple(parameter.shape)}")
        _require(value.dtype == parameter.dtype,
                 f"dtype differs for {name}: {value.dtype} != {parameter.dtype}")
        _require(bool(torch.isfinite(value).all()), f"nonfinite trainable tensor: {name}")
    if device is not None:
        target = torch.device(device)
        for parameter in expected.values():
            _require(parameter.device.type == target.type,
                     f"trainable tensor device differs: {parameter.device}")
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(saved[name])
    elements = int(sum(value.numel() for value in saved.values()))
    return {
        "path": str(source),
        "sha256": actual_sha,
        "tensors": len(saved),
        "elements": elements,
        "keyset_sha256": sha256_canonical(sorted(expected)),
        "values_sha256": sha256_canonical(
            {name: [list(value.shape), str(value.dtype)] for name, value in sorted(saved.items())}),
    }


def load_resident_strict(graph, path, *, binding: str, expected_sha256: str | None = None) -> tuple[Any, dict]:
    """Load the explicit neural state object (never model weights)."""
    from safetensors.torch import load_file
    from treyrin.structural_core import ExperienceState

    source = Path(path)
    _require(source.is_file(), f"missing resident payload: {source}")
    actual_sha = sha256_file(source)
    if expected_sha256 is not None:
        _require(actual_sha == expected_sha256,
                 f"resident payload hash {actual_sha} != {expected_sha256}")
    # The declared schema is the model's own initial state: exactly the fields
    # this configuration materialises.  A saved state must cover all of them and
    # must not smuggle in anything else.
    template = graph.core.initial_state(1, binding).tensors()
    fields = set(template)
    saved = load_file(source, device="cpu")
    missing = sorted(fields - set(saved))
    unexpected = sorted(set(saved) - fields)
    _require(not missing, f"missing resident tensors: {missing[:6]}")
    _require(not unexpected, f"unexpected resident tensors: {unexpected[:6]}")
    device = next(graph.core.parameters()).device
    tensors = {}
    for name, value in saved.items():
        expected = template[name]
        _require(tuple(value.shape) == tuple(expected.shape),
                 f"resident shape differs for {name}: {tuple(value.shape)} != {tuple(expected.shape)}")
        _require(value.dtype == expected.dtype,
                 f"resident dtype differs for {name}: {value.dtype} != {expected.dtype}")
        tensors[name] = value.to(device)
    state = ExperienceState(**tensors, binding=binding)
    graph.core.validate_state(state, 1)
    return state, {
        "path": str(source),
        "sha256": actual_sha,
        "tensors": len(saved),
        "elements": int(sum(value.numel() for value in saved.values())),
        "binding": binding,
        "schema": STATE_SCHEMA,
        "validated": True,
    }


# ---------------------------------------------------------------------------
# Parameter / state key inventory
# ---------------------------------------------------------------------------
def classify_parameter(name: str) -> str:
    if name.startswith("body.model.") and ".lora_" in name:
        return "language_lora"
    if name.startswith("body.model."):
        return "language_body"
    if name.startswith("body.output_adapter."):
        return "output_adapter"
    if name.startswith("body."):
        return "body_port"
    if name in ("to_core.weight", "to_body.weight", "feedback_gain"):
        return "bridge"
    if name.startswith("core."):
        return "core"
    return "other"


def key_report(graph, resident_state, *, base_layout: dict | None = None) -> dict:
    """Exact inventory of what the standard class owns."""
    groups: dict[str, dict] = {}
    named = []
    for name, parameter in graph.named_parameters():
        group = classify_parameter(name)
        entry = groups.setdefault(group, {"tensors": 0, "elements": 0,
                                          "trainable_tensors": 0, "trainable_elements": 0,
                                          "frozen_tensors": 0, "frozen_elements": 0})
        entry["tensors"] += 1
        entry["elements"] += parameter.numel()
        key = "trainable" if parameter.requires_grad else "frozen"
        entry[key + "_tensors"] += 1
        entry[key + "_elements"] += parameter.numel()
        named.append({"name": name, "group": group,
                      "shape": list(parameter.shape), "dtype": str(parameter.dtype),
                      "requires_grad": bool(parameter.requires_grad)})
    state_keys = list(graph.state_dict().keys())
    buffers = [name for name, _ in graph.named_buffers()]
    parameters = list(graph.named_parameters())
    trainable = trainable_parameters(graph)
    report = {
        "schema": "treyrin-standard-keys-v1",
        "named_parameters": {
            "tensors": len(parameters),
            "elements": int(sum(p.numel() for _, p in parameters)),
            "trainable_tensors": len(trainable),
            "trainable_elements": int(sum(p.numel() for p in trainable.values())),
            "frozen_tensors": len(parameters) - len(trainable),
        },
        "state_dict": {
            "keys": len(state_keys),
            "parameter_keys": len(parameters),
            "buffer_keys": buffers,
        },
        "by_group": groups,
        "parameters": named,
        "trainable_keys": sorted(trainable),
        "state_dict_keys": sorted(state_keys),
        "buffers": [{"name": name, "shape": list(buffer.shape), "dtype": str(buffer.dtype)}
                    for name, buffer in graph.named_buffers()],
        "resident_state": {
            "tensors": len(resident_state.tensors()),
            "keys": sorted(resident_state.tensors()),
            "schema": STATE_SCHEMA,
            "binding": resident_state.binding,
        },
        "module_tree": {
            "language_body": "body.model (peft-wrapped Qwen3_5ForCausalLM, frozen) -> body.model.base_model.model.model (Qwen3_5TextModel)",
            "lora": "body.model.base_model.model.model.layers.<i>.<proj>.lora_{A,B}.default.weight",
            "output_adapter": "body.output_adapter.{A,B}",
            "bridge": "to_core.weight, to_body.weight, feedback_gain",
            "core": "core.* (StructuralCore)",
            "buffer": "body.choice_label_tokens (public one-token label alphabet)",
        },
    }
    if base_layout is not None:
        mapped = map_text_checkpoint_keys(base_layout)
        report["reconciliation"] = {
            "local_checkpoint_tensors": len(base_layout["weights"]),
            "text_tensors": len(mapped["mapped"]),
            "skipped_tensors": len(mapped["skipped"]),
            "trainable_tensors": len(trainable),
            "inherited_index_keys": INHERITED_INDEX_KEYS,
            "checkpoint_plus_trainable": len(base_layout["weights"]) + len(trainable),
            "tied_output_embedding_counts_once_in_parameters": True,
            "module_state_dict_keys": len(state_keys),
            "explanation": ("The inherited 1239-key index is the 738-tensor local checkpoint "
                            "(426 text + 312 vision/MTP) plus the 501 trainable tensors. The "
                            "standard class consumes the 426 text tensors (the MTP head is tied "
                            "to the shared embedding and the vision tower is not part of the "
                            "reviewed text body), so its own state_dict is smaller and exact."),
        }
    return report


# ---------------------------------------------------------------------------
# Standard package save / load
# ---------------------------------------------------------------------------
def _package_config_dict(config) -> dict:
    value = config.to_dict()
    value["architectures"] = ["TreyrinForCausalLM"]
    value["auto_map"] = {
        "AutoConfig": "configuration_treyrin.TreyrinConfig",
        "AutoModelForCausalLM": "modeling_treyrin.TreyrinForCausalLM",
    }
    return value


def save_standard_package(model, directory, *, export_base: bool = False, note: str | None = None) -> dict:
    """Write a strict, self-describing package of the real model.

    The frozen language body is *referenced* by identity unless export_base is
    True (then it is hardlinked/copied for a self-contained delivery).  The
    trainable payload and the resident state are always written as separate
    files, never mixed.
    """
    import shutil
    from datetime import datetime, timezone

    from safetensors.torch import save_file

    directory = Path(directory)
    _require(not directory.exists(), f"refusing to overwrite package: {directory}")
    directory.mkdir(parents=True)
    graph = model.graph
    trainable = {name: parameter.detach().cpu().clone().contiguous()
                 for name, parameter in trainable_parameters(graph).items()}
    resident = {name: value.detach().cpu().clone().contiguous()
                for name, value in model.resident_state.tensors().items()}
    trainable_path = directory / STANDARD_TRAINABLE
    resident_path = directory / STANDARD_RESIDENT
    save_file(trainable, trainable_path)
    save_file(resident, resident_path)

    base_body = Path(model.base_body_dir)
    base_layout = base_body_index(base_body)
    if export_base:
        shutil.copyfile(base_body.parent / "asset-manifest.json", directory / "asset-manifest.json")
        shutil.copytree(base_body.parent / "tokenizer", directory / "tokenizer")
        target = directory / "base_body"
        target.mkdir()
        for name in ("config.json", "generation_config.json", "model.safetensors.index.json"):
            shutil.copyfile(base_body / name, target / name)
        for shard in base_layout["shards"]:
            try:
                (target / shard).hardlink_to(base_body / shard)
                linked = True
            except OSError:
                shutil.copyfile(base_body / shard, target / shard)
                linked = False
        exported_layout = base_body_index(target)
        _require(exported_layout["files"] == base_layout["files"],
                 "exported base body differs from the frozen assets")
        base_reference = {"exported": True, "relative_path": "base_body",
                          "assets_root": ".", "hardlinked": linked}
    else:
        base_reference = {"exported": False, "resolved_path": str(base_body),
                          "assets_root": str(base_body.parent)}

    config = model.config
    config_dict = _package_config_dict(config)
    (directory / "config.json").write_text(json.dumps(config_dict, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")

    def tensor_entries(payload):
        entries = []
        for name, value in sorted(payload.items()):
            raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
            entries.append({"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
                            "sha256": hashlib.sha256(raw).hexdigest()})
        return {
            "count": len(payload),
            "elements": int(sum(value.numel() for value in payload.values())),
            "keys": entries,
        }

    files = {
        "config.json": sha256_file(directory / "config.json"),
        STANDARD_TRAINABLE: sha256_file(trainable_path),
        STANDARD_RESIDENT: sha256_file(resident_path),
    }
    manifest = {
        "schema": STANDARD_SCHEMA,
        "order": "000064",
        "stage": "R1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state_schema": config.state_schema,
        "binding": model.binding,
        "binding_context": {
            "device": model.load_report["context"]["device"],
            "dtype": model.load_report["context"]["dtype"],
            "quantize": model.load_report["context"]["quantize"],
            "torch": torch.__version__,
        },
        "inherited": {
            "source": model.load_report["inherited"]["checkpoint"],
            "candidate_manifest_sha256": model.load_report["inherited"]["manifest_sha256"],
            "parameter_binding": model.load_report["inherited"]["parameter_binding"],
            "contract_identity": model.load_report["inherited"]["contract_identity"],
            "optimizer_steps": model.load_report["inherited"]["optimizer_steps"],
            "trainable_sha256": model.load_report["inherited"]["trainable_sha256"],
            "resident_sha256": model.load_report["inherited"]["resident_sha256"],
            "binding_preserved_in_this_context": model.load_report["inherited"]["binding_preserved"],
            "binding_note": model.load_report["inherited"]["binding_note"],
        },
        "core_config": config.core_config_dict(),
        "lora": dict(config.lora),
        "output_adapter_rank": config.output_adapter_rank,
        "checkpoint_layers": config.checkpoint_layers,
        "base_body": {
            **base_reference,
            "asset_manifest_sha256": asset_identity(base_body.parent),
            "index_sha256": base_layout["index_sha256"],
            "shards": {name: base_layout["files"][name] for name in base_layout["shards"]},
            "text_tensors": len(map_text_checkpoint_keys(base_layout)["mapped"]),
            "commit": json.loads((base_body.parent / "asset-manifest.json").read_text("utf-8")).get("commit"),
            "encoding_sha256": tokenizer_identity(base_body.parent),
        },
        "tensors": {
            "trainable": tensor_entries(trainable),
            "resident": tensor_entries(resident),
        },
        "files": files,
        "note": note,
    }
    manifest_path = directory / STANDARD_MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_standard_package(directory, *, device: str = "cpu", dtype: str | None = None,
                          quantize: str | None = None, require_binding: bool = True,
                          asset_verification: str = "manifest",
                          retention_reference: bool = False) -> "TreyrinAssembly":
    """Strictly reload a package written by save_standard_package."""
    directory = Path(directory)
    manifest_path = directory / STANDARD_MANIFEST
    _require(manifest_path.is_file(), f"missing standard manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    _require(manifest.get("schema") == STANDARD_SCHEMA, manifest.get("schema"))
    config = TreyrinConfig.from_pretrained(directory, local_files_only=True)
    core_config = config.resolved_core_config(manifest["core_config"])
    configured_dtype = dtype or config.body_dtype
    context = {"device": device, "dtype": configured_dtype,
               "quantize": quantize if quantize is not None else (manifest["binding_context"]["quantize"])}
    recorded = {key: manifest["binding_context"][key] for key in ("device", "dtype", "quantize")}
    if require_binding:
        _require(context == recorded,
                 f"binding comparison across execution contexts is not meaningful: {context} != {recorded}")
    reference = manifest["base_body"]
    if reference.get("exported"):
        assets_root = directory
        base_body = directory / reference["relative_path"]
    else:
        assets_root = Path(reference["assets_root"])
        base_body = Path(reference["resolved_path"])
    actual_asset = asset_identity(assets_root)
    _require(actual_asset == reference["asset_manifest_sha256"],
             f"base body identity {actual_asset} != recorded {reference['asset_manifest_sha256']}")
    if asset_verification == "full":
        verify_assets(assets_root, mode="full")
    layout = base_body_index(base_body)
    _require(layout["index_sha256"] == reference["index_sha256"],
             "base body index differs from the recorded package")
    for shard, expected in reference["shards"].items():
        _require(layout["files"][shard] == expected, f"base body shard differs: {shard}")

    tokenizer = _load_tokenizer(assets_root / "tokenizer")
    _require(tokenizer_identity(assets_root) == reference["encoding_sha256"],
             "tokenizer encoding identity differs from the recorded package")
    body, body_report = build_language_body(base_body, device=device, dtype=configured_dtype,
                                            quantize=quantize,
                                            asset_commit=manifest["base_body"].get("commit"))
    retention = None
    if retention_reference:
        retention, _reference_port = build_retention_reference(
            body, tokenizer=tokenizer,
            asset_identity_value=reference["asset_manifest_sha256"], device=device)
    adapted, lora_report = inject_lora(body, config.lora)
    graph, graph_report = build_graph(
        adapted, tokenizer=tokenizer, core_config=core_config,
        asset_identity_value=reference["asset_manifest_sha256"],
        checkpoint_layers=config.checkpoint_layers,
        output_adapter_rank=config.output_adapter_rank, device=device)

    expected_keys = {entry["name"]: entry for entry in manifest["tensors"]["trainable"]["keys"]}
    actual_keys = trainable_parameters(graph)
    missing = sorted(set(expected_keys) - set(actual_keys))
    unexpected = sorted(set(actual_keys) - set(expected_keys))
    _require(not missing and not unexpected,
             f"module layout differs from the package: missing={missing[:4]} unexpected={unexpected[:4]}")
    for name, entry in expected_keys.items():
        parameter = actual_keys[name]
        _require(list(parameter.shape) == entry["shape"],
                 f"shape differs from package for {name}")
        _require(str(parameter.dtype) == entry["dtype"], f"dtype differs from package for {name}")

    trainable_report = load_trainable_strict(
        graph, directory / STANDARD_TRAINABLE,
        expected_sha256=manifest["files"][STANDARD_TRAINABLE], device=device)
    resident_state, resident_report = load_resident_strict(
        graph, directory / STANDARD_RESIDENT, binding=manifest["binding"],
        expected_sha256=manifest["files"][STANDARD_RESIDENT])
    _require(resident_report["tensors"] == manifest["tensors"]["resident"]["count"],
             "resident tensor count differs from the package")
    binding = graph.seal_parameters()
    if require_binding:
        _require(binding == manifest["binding"],
                 f"reloaded binding {binding} != saved binding {manifest['binding']}")
    assembly = TreyrinAssembly(
        config=config, body=adapted, graph=graph, resident_state=resident_state,
        tokenizer=tokenizer, binding=binding, base_body_dir=base_body,
        retention_reference=retention,
        source={"kind": "standard-package", "directory": str(directory),
                "manifest": manifest, "manifest_sha256": sha256_file(manifest_path)},
        load_report={
            "schema": "treyrin-standard-load-v1",
            "source": "standard-package",
            "directory": str(directory),
            "context": context,
            "encoding_sha256": reference["encoding_sha256"],
            "body": body_report,
            "language_lora": lora_report,
            "graph": {key: value for key, value in graph_report.items() if key != "core_config"},
            "trainable": trainable_report,
            "resident": resident_report,
            "binding": binding,
            "binding_expected": manifest["binding"],
            "binding_preserved": binding == manifest["binding"],
            "inherited": {
                "checkpoint": manifest["inherited"]["source"],
                "manifest_sha256": manifest["inherited"]["candidate_manifest_sha256"],
                "parameter_binding": manifest["inherited"]["parameter_binding"],
                "contract_identity": manifest["inherited"]["contract_identity"],
                "optimizer_steps": manifest["inherited"]["optimizer_steps"],
                "trainable_sha256": manifest["inherited"]["trainable_sha256"],
                "resident_sha256": manifest["inherited"]["resident_sha256"],
                "binding_preserved": False,
                "binding_note": "standard package: the inherited binding is carried as lineage",
            },
        },
    )
    return assembly


# ---------------------------------------------------------------------------
# Assembly object and the inherited workspace loader
# ---------------------------------------------------------------------------
@dataclass
class TreyrinAssembly:
    """One materialised standard Treyrin model and the evidence around it."""

    config: Any
    body: Any
    graph: Any
    resident_state: Any
    tokenizer: Any
    binding: str
    base_body_dir: Path
    source: dict
    load_report: dict = field(default_factory=dict)
    retention_reference: Any | None = None

    def key_report(self) -> dict:
        return key_report(self.graph, self.resident_state,
                          base_layout=base_body_index(self.base_body_dir))


def _load_tokenizer(tokenizer_dir):
    from transformers import AutoTokenizer

    path = Path(tokenizer_dir)
    _require(path.is_dir(), f"missing tokenizer directory: {path}")
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def binding_context_note(context: dict, inherited_binding: str, binding: str) -> str | None:
    if binding == inherited_binding:
        return None
    if context["quantize"] is None and context["device"] == "cpu":
        return ("CPU bf16 reference path: the inherited binding was computed over the "
                "NF4-quantized CUDA language body, whose state dict stores quantized uint8 "
                "weights and absmax buffers instead of bf16 tensors. The inherited parameter "
                "values are preserved byte-exactly in the verified trainable and resident "
                "payloads; only the container representation, and therefore the seal hash, "
                "differs on this path.")
    return ("binding differs from the inherited value in this execution context "
            f"(device={context['device']}, dtype={context['dtype']}, "
            f"quantize={context['quantize']})")


def assemble_workspace(root, *, device: str = "cpu", dtype: str | None = None,
                       quantize: str | None = None, verify_assets_mode: str = "full",
                       strict_identity: bool = True, expect_inherited: bool = True,
                       require_binding: bool | None = None, seal: bool = True,
                       tokenizer=None, retention_reference: bool = False) -> TreyrinAssembly:
    """Build the real standard model from the inherited workspace layout."""
    root = Path(root)
    config = TreyrinConfig.from_pretrained(root, local_files_only=True)
    checkpoint = root / config.treyrin_checkpoint
    identity = verify_candidate(
        checkpoint, assets_dir=root / "assets", tokenizer_dir=root / "assets" / "tokenizer",
        verify_assets_mode=verify_assets_mode, strict=strict_identity,
        expect_inherited=expect_inherited)
    manifest = identity["manifest"]
    core_config = config.resolved_core_config(manifest["core_config"])
    configured_dtype = dtype or config.body_dtype
    base_body = root / config.base_body
    if tokenizer is None:
        tokenizer = _load_tokenizer(root / config.tokenizer)
    context = {"device": device, "dtype": configured_dtype, "quantize": quantize}
    if require_binding is None:
        require_binding = bool(device == "cuda" and quantize == "nf4")
    if device == "cuda" and quantize != "nf4":
        raise ValueError("the inherited CUDA representation is NF4; pass quantize='nf4'")

    body, body_report = build_language_body(
        base_body, device=device, dtype=configured_dtype, quantize=quantize,
        asset_commit=identity["assets"].get("commit"))
    retention = None
    if retention_reference:
        retention, _reference_port = build_retention_reference(
            body, tokenizer=tokenizer,
            asset_identity_value=identity["identity"]["assets_sha256"], device=device)
    adapted, lora_report = inject_lora(body, config.lora)
    graph, graph_report = build_graph(
        adapted, tokenizer=tokenizer, core_config=core_config,
        asset_identity_value=identity["identity"]["assets_sha256"],
        checkpoint_layers=config.checkpoint_layers,
        output_adapter_rank=config.output_adapter_rank, device=device)

    trainable_report = load_trainable_strict(
        graph, checkpoint / "session" / "learning" / "trainable.safetensors",
        expected_sha256=identity["payload"]["session/learning/trainable.safetensors"]["sha256"],
        device=device)
    resident_state, resident_report = load_resident_strict(
        graph, checkpoint / "resident.safetensors",
        binding=manifest["parameter_binding"],
        expected_sha256=identity["payload"]["resident.safetensors"]["sha256"])

    if expect_inherited:
        _require(trainable_report["tensors"] == INHERITED_TRAINABLE_TENSORS,
                 f"trainable tensor count {trainable_report['tensors']} != {INHERITED_TRAINABLE_TENSORS}")
        _require(trainable_report["elements"] == INHERITED_TRAINABLE_ELEMENTS,
                 f"trainable element count {trainable_report['elements']} != {INHERITED_TRAINABLE_ELEMENTS}")
        _require(resident_report["tensors"] == INHERITED_RESIDENT_TENSORS,
                 f"resident tensor count {resident_report['tensors']} != {INHERITED_RESIDENT_TENSORS}")

    binding = graph.seal_parameters() if seal else graph._binding
    inherited_binding = manifest["parameter_binding"]
    binding_preserved = binding == inherited_binding
    if require_binding:
        _require(binding_preserved,
                 f"parameters sealed to {binding}, inherited binding is {inherited_binding}")
    note = binding_context_note(context, inherited_binding, binding)
    body_layout = base_body_index(base_body)
    assembly = TreyrinAssembly(
        config=config, body=adapted, graph=graph, resident_state=resident_state,
        tokenizer=tokenizer, binding=binding, base_body_dir=base_body,
        retention_reference=retention,
        source={"kind": "inherited-workspace", "root": str(root),
                "checkpoint": str(checkpoint),
                "manifest_sha256": identity["manifest_sha256"]},
        load_report={
            "schema": "treyrin-standard-load-v1",
            "source": "inherited-workspace",
            "root": str(root),
            "checkpoint": str(checkpoint),
            "context": context,
            "encoding_sha256": identity["identity"]["encoding_sha256"],
            "body": body_report,
            "language_lora": lora_report,
            "graph": {key: value for key, value in graph_report.items() if key != "core_config"},
            "core_config": core_config,
            "trainable": trainable_report,
            "resident": resident_report,
            "binding": binding,
            "binding_expected": inherited_binding,
            "binding_preserved": binding_preserved,
            "binding_note": note,
            "inherited": {
                "checkpoint": str(checkpoint),
                "manifest_sha256": identity["manifest_sha256"],
                "parameter_binding": inherited_binding,
                "contract_identity": manifest["contract_identity"],
                "optimizer_steps": manifest["optimizer_steps"],
                "trainable_sha256": identity["identity"]["trainable_sha256"],
                "resident_sha256": identity["identity"]["resident_sha256"],
                "binding_preserved": binding_preserved,
                "binding_note": note,
            },
            "identity": identity,
            "base_body": {
                "path": str(base_body),
                "index_sha256": body_layout["index_sha256"],
                "shards": body_layout["shards"],
                "tensors": len(body_layout["weights"]),
                "text_tensors": len(map_text_checkpoint_keys(body_layout)["mapped"]),
            },
        },
    )
    return assembly


def open_model(directory, **kwargs) -> TreyrinAssembly:
    """Load either a standard R1 package or the inherited workspace layout."""
    path = Path(directory)
    _require(path.is_dir(), f"model directory does not exist: {path}")
    if (path / STANDARD_MANIFEST).is_file():
        mode = kwargs.pop("verify_assets_mode", None)
        # Workspace-only policies: a package is self-describing and must be
        # sealed to verify the binding it was saved with.
        kwargs.pop("strict_identity", None)
        kwargs.pop("expect_inherited", None)
        kwargs.pop("tokenizer", None)
        if kwargs.pop("seal", True) is False:
            raise TreyrinStrictLoadError(
                "a standard package must be sealed to verify its saved binding")
        if mode is not None:
            kwargs["asset_verification"] = mode
        return load_standard_package(path, **kwargs)
    if (path / "config.json").is_file():
        return assemble_workspace(path, **kwargs)
    raise TreyrinStrictLoadError(
        f"{path} is neither a standard package ({STANDARD_MANIFEST}) nor a workspace root (config.json)")
