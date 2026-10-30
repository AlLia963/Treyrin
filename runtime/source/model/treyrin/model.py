"""A trained decision readout of the existing language backbone.

This is an early model expansion experiment. It does not claim deep fusion or
learned open-world dynamics. Native language requests bypass no original layers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import torch
from torch import nn

from .backbone import NativeBackbone
from .decision import Problem


class DecisionReadout(nn.Module):
    def __init__(self, representation_dim: int = 2560, numeric_dim: int = 37):
        super().__init__()
        self.language = nn.Sequential(nn.LayerNorm(representation_dim), nn.Linear(representation_dim, 24), nn.Tanh())
        self.value = nn.Sequential(nn.Linear(numeric_dim + 24, 96), nn.SiLU(),
                                   nn.Linear(96, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, representation: torch.Tensor, numeric: torch.Tensor):
        return self.value(torch.cat((self.language(representation), numeric), dim=-1)).squeeze(-1)


class RepresentationCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, data BLOB NOT NULL)")

    def close(self):
        self.db.close()

    def encode(self, native: NativeBackbone, texts: list[str], batch_size: int = 8) -> np.ndarray:
        prefix = native.model_digest + ":llama-b10360:pooling-last:encoding-v1:"
        keys = [hashlib.sha256((prefix + text).encode()).hexdigest() for text in texts]
        vectors, missing = {}, {}
        for key, text in zip(keys, texts):
            row = self.db.execute("SELECT data FROM vectors WHERE key=?", (key,)).fetchone()
            if row:
                vectors[key] = np.frombuffer(row[0], dtype=np.float32).copy()
            else:
                missing[key] = text
        missing_items = list(missing.items())
        for start in range(0, len(missing_items), batch_size):
            chunk = missing_items[start:start + batch_size]
            encoded = native.encode([text for _, text in chunk])
            for (key, _), vector in zip(chunk, encoded):
                value = np.asarray(vector, dtype=np.float32)
                if value.shape != (2560,) or not np.isfinite(value).all():
                    raise ValueError("Unexpected backbone representation")
                vectors[key] = value
                self.db.execute("INSERT OR REPLACE INTO vectors VALUES (?,?)", (key, value.tobytes()))
            self.db.commit()
            if len(missing_items) > batch_size and start % 400 == 0:
                print(json.dumps({"encoded": min(start + batch_size, len(missing_items)),
                                  "uncached_texts": len(missing_items)}), flush=True)
        return np.stack([vectors[key] for key in keys])


class TreyrinModel:
    def __init__(self, backbone: NativeBackbone, readout: DecisionReadout,
                 cache: RepresentationCache | None = None, use_language: bool = True,
                 architecture: str = "v1"):
        self.backbone, self.readout, self.cache = backbone, readout.eval(), cache
        self.use_language = use_language
        self.architecture = architecture

    def chat(self, messages: list[dict], **parameters) -> dict:
        return self.backbone.chat(messages, **parameters)

    @torch.inference_mode()
    def decide(self, problem: Problem) -> dict:
        started = time.perf_counter()
        encoded = [problem.encode_action(action) for action in problem.actions]
        numeric = np.asarray([x[1] for x in encoded], dtype=np.float32)
        if self.use_language:
            context = [problem.encode_context()]
            state = (self.cache.encode(self.backbone, context) if self.cache else
                     np.asarray(self.backbone.encode(context), dtype=np.float32))
            vectors = np.repeat(state, len(encoded), axis=0)
        else:
            vectors = np.zeros((len(encoded), 2560), dtype=np.float32)
        values = self.readout(torch.from_numpy(vectors), torch.from_numpy(numeric)).numpy() * 40
        feasible = np.asarray([problem.feasible(action) for action in problem.actions])
        selected = int(np.argmax(np.where(feasible, values, -np.inf)))
        return {"action": problem.actions[selected], "predicted_values": dict(zip(problem.actions, map(float, values))),
                "feasible_actions": [action for action, valid in zip(problem.actions, feasible) if valid],
                "seconds": time.perf_counter() - started,
                "provenance": {"problem_id": problem.id, "backbone_digest": self.backbone.model_digest,
                               "decision_readout": self.architecture, "world_model": "explicit_supplied_two_state",
                               "used_exact_oracle_at_inference": False}}
