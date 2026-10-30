"""Path bootstrap and content-hash helpers for the standard Treyrin model.

R1 of order 000064.  The standard model must execute the *inherited* research
modules unchanged: they carry the repaired behaviours (F01-F08 and the later
fixes) and their bytes are part of the inherited candidate identity that is
recorded in model-ready/candidate-manifest.json.  Nothing in this package
mutates them; it only imports them.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent


def _find_workspace() -> Path:
    """Locate the research workspace that owns treyrin/ and model-ready/.

    An explicit TREYRIN_WORKSPACE wins (a future self-contained export can
    point it at its own tree).  Otherwise walk up from the model directory and
    take the first ancestor that contains the inherited structural model.
    """
    declared = os.environ.get("TREYRIN_WORKSPACE")
    if declared:
        candidate = Path(declared).expanduser().resolve()
        if (candidate / "treyrin" / "structural_model.py").is_file():
            return candidate
        raise RuntimeError(f"TREYRIN_WORKSPACE does not contain the Treyrin graph: {candidate}")
    for parent in (MODEL_DIR, *MODEL_DIR.parents):
        if (parent / "treyrin" / "structural_model.py").is_file() and (
            parent / "model-ready" / "candidate-manifest.json"
        ).is_file():
            return parent
        if (parent / "treyrin" / "structural_model.py").is_file() and (
            parent / "assets" / "asset-manifest.json"
        ).is_file():
            return parent
    return MODEL_DIR.parent


WORKSPACE = MODEL_DIR


def bootstrap() -> tuple[Path, Path]:
    """Make the inherited modules and this package importable by file name.

    The order matters: WORKSPACE first so ``treyrin``, ``native_port`` and the
    root-level protocol modules resolve to the inherited sources, then
    MODEL_DIR for the R1 modules themselves.
    """
    # Remove first so the order is deterministic: MODEL_DIR must shadow the
    # legacy draft modules of the same name that still live at the workspace
    # root, and WORKSPACE must provide treyrin/, native_port, generation_result.
    for path in (str(WORKSPACE), str(MODEL_DIR)):
        while path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, str(WORKSPACE))
    sys.path.insert(0, str(MODEL_DIR))
    return WORKSPACE, MODEL_DIR


def sha256_file(path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Whole-file SHA256, streamed (the base body runs to several GB)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value) -> str:
    """The exact canonical JSON encoding the inherited candidate.py uses."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_canonical(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def asset_identity(assets_dir=None) -> str:
    """Identity of the frozen language assets (hash of asset-manifest.json)."""
    assets = Path(assets_dir) if assets_dir is not None else WORKSPACE / "assets"
    return sha256_file(assets / "asset-manifest.json")


def tokenizer_identity(assets_dir=None) -> str:
    """Encoding identity declared by the asset manifest, in candidate order.

    candidate.py derives its encoding identity from the *declared* tokenizer
    entries of the already-hash-verified asset manifest; recomputing it this
    way keeps the inherited value (1bcad349...) reproducible without trusting
    the tokenizer directory a second time.
    """
    assets = Path(assets_dir) if assets_dir is not None else WORKSPACE / "assets"
    manifest = json.loads((assets / "asset-manifest.json").read_text("utf-8"))
    files = {name: value for name, value in manifest["files"].items() if name.startswith("tokenizer/")}
    return sha256_canonical(files)


def tokenizer_file_hashes(assets_dir=None) -> dict:
    """Actual on-disk tokenizer hashes, for the strict asset cross-check."""
    assets = Path(assets_dir) if assets_dir is not None else WORKSPACE / "assets"
    root = assets / "tokenizer"
    return {
        "tokenizer/" + path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
