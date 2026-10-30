"""Versioned structured-event curriculum, separate from any model features.

The diagnostic labels are obtained from observed event history, not fed to a
memory kernel. Surface domains share one transition algebra by design.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import random


@dataclass(frozen=True)
class Event:
    sequence: int
    entity: int
    operation: str
    value: int | None
    source_id: str


@dataclass(frozen=True)
class Query:
    entity: int
    at_sequence: int | None = None


@dataclass(frozen=True)
class Episode:
    episode_id: str
    split: str
    seed: int
    case: str
    domain: str
    events: tuple[Event, ...]
    query: Query
    target: int

    def semantic_hash(self):
        # IDs, labels, seeds and provenance nonces cannot disguise duplicate tasks.
        payload = {"domain": self.domain,
                   "events": [[e.sequence, e.entity, e.operation, e.value] for e in self.events],
                   "query": asdict(self.query)}
        return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",", ":")).encode()).hexdigest()

    def record(self):
        return {"schema": "treyrin-structured-episode-v1", **asdict(self),
                "semantic_sha256": self.semantic_hash()}

    @classmethod
    def from_record(cls, record):
        if record.get("schema") != "treyrin-structured-episode-v1":
            raise ValueError("Unknown episode schema")
        episode = cls(**{k:record[k] for k in
                         ("episode_id","split","seed","case","domain","target")},
                      events=tuple(Event(**e) for e in record["events"]),
                      query=Query(**record["query"]))
        if episode.semantic_hash() != record["semantic_sha256"]:
            raise ValueError("Episode input content hash mismatch")
        if replay_value(episode.events, episode.query) != episode.target:
            raise ValueError("Episode target inconsistent with observed history")
        return episode


def replay_value(events: tuple[Event, ...], query: Query, unknown: int = 8):
    """Observed-history positive control. Handles temporal queries explicitly."""
    value = unknown
    last_sequence = -1
    for event in events:
        if event.sequence <= last_sequence:
            raise ValueError("Events must have a strictly increasing sequence")
        last_sequence = event.sequence
        if query.at_sequence is not None and event.sequence > query.at_sequence:
            continue
        if event.operation not in {"set", "cancel"}:
            raise ValueError("Unsupported event operation")
        if event.entity == query.entity:
            if event.operation == "cancel":
                value = unknown
            elif event.value is not None and 0 <= event.value < unknown:
                value = event.value
            else:
                raise ValueError("Invalid observed categorical value")
    return value


def generate_episodes(split: str, seed: int, case: str, domain: str, count: int,
                      interference_events: int = 48):
    if case not in {"multi_binding","overwrite","cancel","unknown","interference","historical"}:
        raise ValueError("Unregistered case")
    if domain not in {"research", "environment"}:
        raise ValueError("Unregistered surface domain")
    rng = random.Random(f"treyrin-p1-v1:{seed}:{case}:{domain}")
    episodes = []
    for i in range(count):
        ident = f"{split}:{seed}:{case}:{domain}:{i}"
        entities = rng.sample(range(16),16)
        initial = [rng.randrange(8) for _ in range(4)]
        raw = [(entities[j], "set", initial[j]) for j in range(4)]
        selected = rng.randrange(4)
        query = Query(entities[selected])
        if case == "overwrite":
            new = (initial[selected] + rng.randrange(1,8)) % 8
            raw.append((entities[selected], "set", new))
        elif case == "cancel":
            raw.append((entities[selected], "cancel", None))
        elif case == "unknown":
            query = Query(entities[4])
        elif case == "interference":
            for _ in range(interference_events):
                raw.append((rng.choice(entities[4:]), "set", rng.randrange(8)))
        elif case == "historical":
            # First observation is guaranteed to have a different later revision.
            query = Query(entities[0], at_sequence=0)
            raw.append((entities[0], "set", (initial[0]+rng.randrange(1,8))%8))
        events = tuple(Event(t,k,op,v,f"{ident}:observation:{t}") for t,(k,op,v) in enumerate(raw))
        target = replay_value(events, query)
        episodes.append(Episode(ident,split,seed,case,domain,events,query,target))
    return episodes
