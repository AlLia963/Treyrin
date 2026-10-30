"""P3 (order 000062): growable task library, physical slots and action receipts.

What this module is
-------------------
A library of complete delegations that keeps growing, plus a projection of the
current active set onto the existing D06 GoalState rows (the same physical slots
and the same 29/9 feature formulas that the learned goal/action head reads).
Three identities are kept apart on purpose:

* task identity - task_id (the same-body proposal's goal-... record id).
  Stable for the whole life of the task, including across revisions, completion
  and re-projection.
* physical slot - a row of the current GoalState projection.  A task may be
  unbound when the active set is larger than the fixed slot capacity.  Rows are
  rebuilt from the active set, so a slot is never a permanent identity.
* action identity - (task_id, subtask_id, attempt), recorded on every progress
  event and tool receipt.  It never points at a slot.

What is stored where
--------------------
* tasks.jsonl - append-only, hash-chained journal of post-image task records and
  receipts (fsync before the in-memory view changes).
* blobs.jsonl - growable, content-addressed references: the original delegation
  full text, the complete constraint/deliverable/evidence lists and long tool
  results.  A task record keeps only ids, so a long delegation and a growing
  evidence list do not inflate every event.
* task-checkpoint.json - byte offset, sequence, chain and state digest of a
  verified prefix; TaskStore.open re-verifies it during replay.

Engine separation: this file never loads a model, never touches CUDA and never
writes a proposal.  The proposal is produced by the same-body generator
(treyrin.intake.intake_proposal / intent_lifecycle.IntentLifecycle); this store
accepts the validated GoalProposal and keeps the raw full text.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import torch

from .goal_control import (ACTIONS, CTX_NAMES, PER_GOAL_NAMES, context_features,
                           goal_feature_rows, legality_mask)
from .goal_state import GoalRecord, GoalState, GoalStateError
from .intake import GoalProposal, RECORD_ID, parse_goal_proposal
from .joint_context import resource_legal

SCHEMA_JOURNAL = 'p3-task-journal-v1'
SCHEMA_BLOB = 'p3-task-blob-v1'
SCHEMA_CHECKPOINT = 'p3-task-checkpoint-v1'
ZERO_CHAIN = '0' * 64

TASK_STATUSES = ('proposed', 'active', 'completed', 'cancelled', 'superseded')
SUBTASK_STATUSES = ('pending', 'active', 'done', 'failed', 'skipped')
PROGRESS_STATUSES = ('not_started', 'running', 'blocked', 'done', 'cancelled')
PRECONDITION_STATUSES = ('pending', 'satisfied', 'failed', 'waived')
RECEIPT_KINDS = ('submit', 'accept', 'start', 'progress', 'tool', 'revise',
                 'complete', 'cancel', 'precondition', 'resume')
INLINE_LIST_MAX = 8


class TaskStoreError(ValueError):
    """Typed rejection: .code is a stable token, never a message."""

    def __init__(self, code, message='', **detail):
        super().__init__('%s: %s' % (code, message))
        self.code = code
        self.detail = detail


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'))


def _sha_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _utcnow():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def _epoch(value):
    if isinstance(value, (int, float)):
        return float(value)
    return datetime.fromisoformat(str(value)).timestamp()


def _atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex[:10])
    with open(temporary, 'w', encoding='utf-8') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return _sha_text(payload)


# ---------------------------------------------------------------------------
# journal + growable references
# ---------------------------------------------------------------------------
class TaskJournal:
    """Hash-chained append-only journal; one canonical ASCII line per event."""

    def __init__(self, path, *, sync=True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sync = bool(sync)
        self._handle = None
        self._seq = 0
        self._chain = ZERO_CHAIN
        self._offset = 0

    def open(self):
        if self._handle is not None:
            return self
        if not self.path.exists():
            with open(self.path, 'wb'):
                pass
        self._handle = open(self.path, 'r+b')
        self._handle.seek(0, os.SEEK_END)
        self._offset = self._handle.tell()
        return self

    @property
    def next_seq(self):
        return self._seq + 1

    @property
    def seq(self):
        return self._seq

    @property
    def chain(self):
        return self._chain

    @property
    def offset(self):
        return self._offset

    def append(self, kind, payload, *, ts):
        if self._handle is None:
            raise TaskStoreError('ts.journal-closed', 'journal must be open')
        entry = {'schema': SCHEMA_JOURNAL, 'seq': self._seq + 1,
                 'entry_id': uuid.uuid4().hex, 'ts': ts, 'kind': kind,
                 'payload': payload, 'prev_chain': self._chain}
        entry['payload_sha256'] = _sha_text(canonical(payload))
        digest = _sha_text(canonical(entry))
        entry['entry_sha256'] = digest
        entry['chain_sha256'] = _sha_text(self._chain + digest)
        line = canonical(entry).encode('ascii') + b'\n'
        self._handle.write(line)
        self._handle.flush()
        if self.sync:
            os.fsync(self._handle.fileno())
        self._offset += len(line)
        self._seq += 1
        self._chain = entry['chain_sha256']
        return entry

    def iter_entries(self):
        chain = ZERO_CHAIN
        expected = 1
        with open(self.path, 'rb') as handle:
            position = 0
            for raw_line in handle:
                if not raw_line.endswith(b'\n'):
                    raise TaskStoreError('ts.journal-torn',
                                         'unterminated line at byte %d' % position)
                try:
                    entry = json.loads(raw_line.decode('ascii'))
                except Exception as exc:                       # noqa: BLE001
                    raise TaskStoreError('ts.journal-corrupt',
                                         'byte %d: %s' % (position, exc))
                if entry.get('schema') != SCHEMA_JOURNAL:
                    raise TaskStoreError('ts.journal-corrupt', 'schema differs')
                if entry.get('seq') != expected:
                    raise TaskStoreError('ts.journal-gap', str(entry.get('seq')))
                if entry.get('prev_chain') != chain:
                    raise TaskStoreError('ts.journal-chain', 'byte %d' % position)
                body = {k: v for k, v in entry.items()
                        if k not in ('entry_sha256', 'chain_sha256')}
                if entry.get('payload_sha256') != _sha_text(canonical(body['payload'])):
                    raise TaskStoreError('ts.journal-payload', 'seq %d' % expected)
                if entry.get('entry_sha256') != _sha_text(canonical(body)):
                    raise TaskStoreError('ts.journal-digest', 'seq %d' % expected)
                new_chain = _sha_text(chain + entry['entry_sha256'])
                if entry.get('chain_sha256') != new_chain:
                    raise TaskStoreError('ts.journal-chain', 'seq %d' % expected)
                yield entry
                chain = new_chain
                expected += 1
                position += len(raw_line)

    def close(self):
        if self._handle is not None:
            try:
                self._handle.close()
            finally:
                self._handle = None

    def adopt(self, seq, chain, offset):
        self._seq = int(seq)
        self._chain = chain
        self._offset = int(offset)
        return self


class BlobLog:
    """Content-addressed growable references (delegation text, long lists)."""

    def __init__(self, path, *, sync=True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sync = bool(sync)
        self._entries = {}
        self._handle = None
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        with open(self.path, 'rb') as handle:
            for position, raw_line in enumerate(handle):
                if not raw_line.endswith(b'\n'):
                    raise TaskStoreError('ts.blob-torn', 'line %d' % position)
                entry = json.loads(raw_line.decode('ascii'))
                if entry.get('schema') != SCHEMA_BLOB:
                    raise TaskStoreError('ts.blob-corrupt', 'line %d' % position)
                if entry.get('sha256') != _sha_text(entry['text']):
                    raise TaskStoreError('ts.blob-digest', entry.get('blob_id'))
                self._entries[entry['blob_id']] = entry

    def put(self, text, *, kind, source=''):
        if not isinstance(text, str) or not text:
            raise TaskStoreError('ts.blob-empty', 'non-empty text required')
        blob_id = 'blob-' + hashlib.sha256((kind + '\0' + text).encode('utf-8')).hexdigest()[:24]
        if blob_id in self._entries:
            return blob_id
        if self._handle is None:
            self._handle = open(self.path, 'ab')
        entry = {'schema': SCHEMA_BLOB, 'blob_id': blob_id, 'kind': kind,
                 'source': source, 'chars': len(text), 'sha256': _sha_text(text),
                 'text': text}
        line = canonical(entry).encode('ascii') + b'\n'
        self._handle.write(line)
        self._handle.flush()
        if self.sync:
            os.fsync(self._handle.fileno())
        self._entries[blob_id] = entry
        return blob_id

    def get(self, blob_id):
        entry = self._entries.get(blob_id)
        if entry is None:
            raise TaskStoreError('ts.blob-missing', str(blob_id))
        if _sha_text(entry['text']) != entry['sha256']:
            raise TaskStoreError('ts.blob-digest', str(blob_id))
        return entry['text']

    def chars(self, blob_id):
        return self._entries[blob_id]['chars']

    def close(self):
        if self._handle is not None:
            try:
                self._handle.close()
            finally:
                self._handle = None

    def __len__(self):
        return len(self._entries)


# ---------------------------------------------------------------------------
# typed task payload
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Subtask:
    subtask_id: str
    text: str
    status: str = 'pending'

    def __post_init__(self):
        if not self.subtask_id or not isinstance(self.subtask_id, str):
            raise TaskStoreError('ts.subtask-id', 'subtask_id required')
        if self.status not in SUBTASK_STATUSES:
            raise TaskStoreError('ts.subtask-status', repr(self.status))

    def to_payload(self):
        return dict(subtask_id=self.subtask_id, text=self.text, status=self.status)

    @classmethod
    def from_payload(cls, payload):
        return cls(payload['subtask_id'], payload['text'], payload.get('status', 'pending'))


@dataclass(frozen=True)
class Precondition:
    precondition_id: str
    kind: str
    description: str
    status: str = 'pending'
    blocking: bool = True
    detail_ref: str | None = None
    checked_at: str | None = None

    def __post_init__(self):
        if not self.precondition_id or not self.kind:
            raise TaskStoreError('ts.precondition-id', 'id and kind required')
        if self.status not in PRECONDITION_STATUSES:
            raise TaskStoreError('ts.precondition-status', repr(self.status))

    def to_payload(self):
        return dict(precondition_id=self.precondition_id, kind=self.kind,
                    description=self.description, status=self.status,
                    blocking=bool(self.blocking), detail_ref=self.detail_ref,
                    checked_at=self.checked_at)

    @classmethod
    def from_payload(cls, payload):
        return cls(payload['precondition_id'], payload['kind'], payload['description'],
                   payload.get('status', 'pending'), bool(payload.get('blocking', True)),
                   payload.get('detail_ref'), payload.get('checked_at'))


@dataclass(frozen=True)
class Revision:
    revision_id: str
    at: str
    author: str
    note: str
    parent_task_id: str
    changed: tuple = ()

    def to_payload(self):
        return dict(revision_id=self.revision_id, at=self.at, author=self.author,
                    note=self.note, parent_task_id=self.parent_task_id,
                    changed=list(self.changed))

    @classmethod
    def from_payload(cls, payload):
        return cls(payload['revision_id'], payload['at'], payload['author'],
                   payload['note'], payload['parent_task_id'],
                   tuple(payload.get('changed', ())))


@dataclass(frozen=True)
class Progress:
    status: str = 'not_started'
    current_subtask: str | None = None
    done: tuple = ()
    failed: tuple = ()
    note: str = ''
    updated_at: str = ''

    def __post_init__(self):
        if self.status not in PROGRESS_STATUSES:
            raise TaskStoreError('ts.progress-status', repr(self.status))

    def to_payload(self):
        return dict(status=self.status, current_subtask=self.current_subtask,
                    done=list(self.done), failed=list(self.failed),
                    note=self.note, updated_at=self.updated_at)

    @classmethod
    def from_payload(cls, payload):
        return cls(payload.get('status', 'not_started'),
                   payload.get('current_subtask'), tuple(payload.get('done', ())),
                   tuple(payload.get('failed', ())), payload.get('note', ''),
                   payload.get('updated_at', ''))


@dataclass(frozen=True)
class Receipt:
    receipt_id: str
    kind: str
    at: str
    task_id: str
    action: str
    status: str
    attempt: int
    subtask_id: str | None = None
    request_id: str | None = None
    tool: str | None = None
    detail_ref: str | None = None
    detail: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in RECEIPT_KINDS:
            raise TaskStoreError('ts.receipt-kind', repr(self.kind))
        if not self.receipt_id or not self.task_id or not self.action:
            raise TaskStoreError('ts.receipt-fields', 'id/task/action required')

    def to_payload(self):
        return dict(receipt_id=self.receipt_id, kind=self.kind, at=self.at,
                    task_id=self.task_id, action=self.action, status=self.status,
                    attempt=int(self.attempt), subtask_id=self.subtask_id,
                    request_id=self.request_id, tool=self.tool,
                    detail_ref=self.detail_ref, detail=dict(self.detail))

    @classmethod
    def from_payload(cls, payload):
        return cls(payload['receipt_id'], payload['kind'], payload['at'],
                   payload['task_id'], payload['action'], payload['status'],
                   int(payload['attempt']), payload.get('subtask_id'),
                   payload.get('request_id'), payload.get('tool'),
                   payload.get('detail_ref'), payload.get('detail') or {})


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    kind: str
    objective: str
    delegation_ref: str
    source_kind: str
    acceptance: str
    autonomy: str
    priority: float
    interruption_cost: float
    deadline_at: float | None
    evidence_ids: tuple
    evidence_ref: str | None
    constraints: tuple
    constraint_ref: str | None
    deliverables: tuple
    deliverable_ref: str | None
    subtasks: tuple
    preconditions: tuple
    revisions: tuple
    progress: Progress
    status: str
    version: int
    lineage: str
    attempts: int
    last_action: float
    created_at: str
    updated_at: str
    submitted_seq: int

    # -- queries ---------------------------------------------------------
    def current_subtask(self):
        if not self.subtasks:
            return None
        for subtask in self.subtasks:
            if subtask.status in ('active', 'pending'):
                return subtask
        return None

    def remaining_subtasks(self):
        return tuple(s.subtask_id for s in self.subtasks
                     if s.status not in ('done', 'skipped'))

    def blocking_preconditions(self):
        return tuple(p for p in self.preconditions if p.blocking)

    def open_preconditions(self):
        return tuple(p for p in self.preconditions
                     if p.blocking and p.status in ('pending', 'failed'))

    def as_dict(self):
        return self.to_payload()

    # -- persistence -----------------------------------------------------
    def to_payload(self):
        return dict(task_id=self.task_id, kind=self.kind, objective=self.objective,
                    delegation_ref=self.delegation_ref, source_kind=self.source_kind,
                    acceptance=self.acceptance, autonomy=self.autonomy,
                    priority=float(self.priority),
                    interruption_cost=float(self.interruption_cost),
                    deadline_at=self.deadline_at, evidence_ids=list(self.evidence_ids),
                    evidence_ref=self.evidence_ref, constraints=list(self.constraints),
                    constraint_ref=self.constraint_ref,
                    deliverables=list(self.deliverables),
                    deliverable_ref=self.deliverable_ref,
                    subtasks=[s.to_payload() for s in self.subtasks],
                    preconditions=[p.to_payload() for p in self.preconditions],
                    revisions=[r.to_payload() for r in self.revisions],
                    progress=self.progress.to_payload(), status=self.status,
                    version=int(self.version), lineage=self.lineage,
                    attempts=int(self.attempts), last_action=float(self.last_action),
                    created_at=self.created_at, updated_at=self.updated_at,
                    submitted_seq=int(self.submitted_seq))

    @classmethod
    def from_payload(cls, payload):
        return cls(task_id=payload['task_id'], kind=payload['kind'],
                   objective=payload['objective'],
                   delegation_ref=payload['delegation_ref'],
                   source_kind=payload['source_kind'],
                   acceptance=payload['acceptance'], autonomy=payload['autonomy'],
                   priority=float(payload['priority']),
                   interruption_cost=float(payload['interruption_cost']),
                   deadline_at=payload.get('deadline_at'),
                   evidence_ids=tuple(payload.get('evidence_ids', ())),
                   evidence_ref=payload.get('evidence_ref'),
                   constraints=tuple(payload.get('constraints', ())),
                   constraint_ref=payload.get('constraint_ref'),
                   deliverables=tuple(payload.get('deliverables', ())),
                   deliverable_ref=payload.get('deliverable_ref'),
                   subtasks=tuple(Subtask.from_payload(x) for x in payload.get('subtasks', ())),
                   preconditions=tuple(Precondition.from_payload(x)
                                       for x in payload.get('preconditions', ())),
                   revisions=tuple(Revision.from_payload(x)
                                   for x in payload.get('revisions', ())),
                   progress=Progress.from_payload(payload.get('progress') or {}),
                   status=payload['status'], version=int(payload['version']),
                   lineage=payload.get('lineage', ''),
                   attempts=int(payload.get('attempts', 0)),
                   last_action=float(payload.get('last_action', 0.0)),
                   created_at=payload['created_at'], updated_at=payload['updated_at'],
                   submitted_seq=int(payload['submitted_seq']))


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------
class TaskStore:
    """Growable task library + active-set projection; CPU only."""

    SCHEMA = 'p3-task-store'

    def __init__(self, directory, *, capacity=8, evidence_slots=16, sync=True,
                 clock=None, store_id=None):
        if not 1 <= int(capacity):
            raise TaskStoreError('ts.capacity', 'capacity must be >= 1')
        if not 1 <= int(evidence_slots):
            raise TaskStoreError('ts.evidence-capacity', 'evidence_slots must be >= 1')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.capacity = int(capacity)
        self.evidence_slots = int(evidence_slots)
        self.sync = bool(sync)
        self.clock = clock or _utcnow
        self.store_id = store_id or self.directory.name
        self.journal = TaskJournal(self.directory / 'tasks.jsonl', sync=sync)
        self.blobs = BlobLog(self.directory / 'blobs.jsonl', sync=sync)
        self._tasks = {}
        self._receipts = []
        self._requests = {}
        self._bindings = []
        self._projection = None
        self._submitted = 0
        self._checkpoint_verified = None
        try:
            self._replay()
        except Exception:
            self.close()
            raise

    # -- lifecycle -------------------------------------------------------
    @classmethod
    def open(cls, directory, *, checkpoint_path=None, verify_checkpoint=True, **kwargs):
        store = cls(directory, **kwargs)
        if verify_checkpoint:
            path = (Path(checkpoint_path) if checkpoint_path
                    else Path(directory) / 'task-checkpoint.json')
            if path.exists():
                try:
                    store._verify_checkpoint_prefix(path)
                except Exception:
                    store.close()
                    raise
        return store

    def _replay(self):
        self.journal.open()
        last = None
        for entry in self.journal.iter_entries():
            self._apply(entry)
            last = entry
        if last is not None:
            self.journal.adopt(last['seq'], last['chain_sha256'],
                               self.journal.path.stat().st_size)
        self._projection = None

    def _apply(self, entry):
        payload = entry.get('payload') or {}
        for task_id, record in (payload.get('records') or {}).items():
            self._tasks[task_id] = TaskRecord.from_payload(record)
            self._submitted = max(self._submitted, int(record.get('submitted_seq', 0)))
        receipt = payload.get('receipt')
        if receipt:
            parsed = Receipt.from_payload(receipt)
            self._receipts.append(parsed)
            if parsed.request_id:
                self._requests[parsed.request_id] = parsed
        binding = payload.get('binding')
        if binding:
            self._bindings.append(binding)

    def _verify_checkpoint_prefix(self, path):
        checkpoint = json.loads(Path(path).read_text(encoding='utf-8'))
        if checkpoint.get('schema') != SCHEMA_CHECKPOINT:
            raise TaskStoreError('ts.checkpoint-schema', str(path))
        target = int(checkpoint['journal_seq'])
        if self.journal.seq < target:
            raise TaskStoreError('ts.checkpoint-missing-prefix',
                                 'journal has %d < checkpoint %d'
                                 % (self.journal.seq, target))
        replayed = TaskStore.__new__(TaskStore)          # replay-only clone
        replayed.journal = self.journal
        replayed.blobs = self.blobs
        replayed._tasks = {}
        replayed._receipts = []
        replayed._requests = {}
        replayed._bindings = []
        replayed._submitted = 0
        for entry in self.journal.iter_entries():
            replayed._apply(entry)
            if entry['seq'] == target:
                break
        digest = replayed.state_digest()
        if digest != checkpoint['state_sha256']:
            raise TaskStoreError('ts.checkpoint-diverged',
                                 'checkpoint %s != replay %s'
                                 % (checkpoint['state_sha256'], digest))
        self._checkpoint_verified = checkpoint
        return checkpoint

    def close(self):
        self.journal.close()
        self.blobs.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- helpers ---------------------------------------------------------
    def now(self):
        value = self.clock()
        return value if isinstance(value, str) else value.isoformat(timespec='milliseconds')

    def _task(self, task_id):
        if task_id not in self._tasks:
            raise TaskStoreError('ts.not-found', str(task_id))
        return self._tasks[task_id]

    def _new_receipt(self, task, *, kind, action, status, subtask_id=None,
                     request_id=None, tool=None, detail_ref=None, detail=None,
                     attempt=None, at=None):
        resolved = int(task.attempts if (attempt is None and task is not None)
                       else (attempt or 0))
        return Receipt(receipt_id='rcpt-' + uuid.uuid4().hex[:16], kind=kind,
                       at=at or self.now(),
                       task_id=task.task_id if task is not None else '',
                       action=action, status=status, attempt=resolved,
                       subtask_id=subtask_id, request_id=request_id, tool=tool,
                       detail_ref=detail_ref, detail=dict(detail or {}))

    def _existing_request(self, request_id):
        if not request_id:
            return None
        return self._requests.get(request_id)

    def _request_conflict(self, prior, *, kind, task_id, action):
        if prior is None:
            return False
        return not (prior.kind == kind and prior.task_id == task_id
                    and prior.action == action)

    def _append(self, kind, *, records=(), receipt=None, binding=None):
        payload = {}
        if records:
            payload['records'] = {record.task_id: record.to_payload()
                                  for record in records}
        if receipt is not None:
            payload['receipt'] = receipt.to_payload()
        if binding is not None:
            payload['binding'] = binding
        entry = self.journal.append(kind, payload, ts=self.now())
        self._apply(entry)
        if kind != 'binding':
            self._projection = None
        return entry

    def put_detail(self, payload, *, kind='detail', source=''):
        """Store an arbitrary JSON payload as a growable reference."""
        text = payload if isinstance(payload, str) else canonical(payload)
        return self.blobs.put(text, kind=kind, source=source)

    def detail(self, blob_id):
        return self.blobs.get(blob_id)

    # -- submission ------------------------------------------------------
    def submit(self, delegation, proposal, *, subtasks=(), constraints=(),
               deliverables=(), preconditions=(), evidence_ids=(),
               known_evidence=None, request_id=None, author='user', now=None):
        """Accept one validated proposal and keep the original delegation text.

        proposal may be a GoalProposal (the real intake path), a mapping or raw
        JSON text (validated again here against known_evidence).
        """
        prior = self._existing_request(request_id)
        if prior is not None:
            if self._request_conflict(prior, kind='submit', task_id=prior.task_id,
                                      action='submit'):
                raise TaskStoreError('ts.request-conflict', str(request_id))
            return dict(task=self._task(prior.task_id), receipt=prior, duplicate=True)
        if not isinstance(delegation, str) or not delegation.strip():
            raise TaskStoreError('ts.delegation-empty', 'original delegation text required')
        proposal = self._coerce_proposal(proposal, known_evidence)
        task_id = proposal.record_id
        if task_id in self._tasks:
            raise TaskStoreError('ts.duplicate-identity', task_id)
        if proposal.acceptance == 'declined':
            raise TaskStoreError('ts.decline-needs-cancel',
                                 'a declined proposal cancels an existing task')
        evidence = tuple(dict.fromkeys(tuple(proposal.evidence_ids)
                                       + tuple(evidence_ids or ())))
        if known_evidence is not None:
            unknown = [e for e in evidence if e not in set(known_evidence)]
            if unknown:
                raise TaskStoreError('ts.evidence-unknown', ','.join(unknown))
        stamps = now or self.now()
        delegation_ref = self.blobs.put(delegation, kind='delegation', source=task_id)
        blocks = self._lists(constraints, deliverables, evidence, task_id)
        subtasks = self._subtasks(subtasks)
        preconditions = self._preconditions(preconditions)
        status = 'proposed' if proposal.acceptance == 'pending' else 'active'
        deadline = None if proposal.deadline_at is None else _epoch(proposal.deadline_at)
        self._submitted += 1
        record = TaskRecord(
            task_id=task_id, kind=proposal.kind, objective=proposal.text,
            delegation_ref=delegation_ref, source_kind=proposal.source_kind,
            acceptance=proposal.acceptance, autonomy=proposal.autonomy,
            priority=float(proposal.priority),
            interruption_cost=float(proposal.interruption_cost), deadline_at=deadline,
            evidence_ids=evidence[:INLINE_LIST_MAX],
            evidence_ref=blocks['evidence_ref'],
            constraints=tuple(constraints[:INLINE_LIST_MAX]),
            constraint_ref=blocks['constraint_ref'],
            deliverables=tuple(deliverables[:INLINE_LIST_MAX]),
            deliverable_ref=blocks['deliverable_ref'],
            subtasks=subtasks, preconditions=preconditions, revisions=(),
            progress=Progress(updated_at=stamps), status=status, version=0,
            lineage='', attempts=0, last_action=0.0, created_at=stamps,
            updated_at=stamps, submitted_seq=self._submitted)
        receipt = self._new_receipt(record, kind='submit', action='submit',
                                    status='accepted', request_id=request_id,
                                    at=stamps,
                                    detail={'delegation_ref': delegation_ref,
                                            'delegation_chars': len(delegation),
                                            'author': author,
                                            'proposal_acceptance': proposal.acceptance})
        self._append('submit', records=(record,), receipt=receipt)
        return dict(task=record, receipt=receipt, duplicate=False)

    def submit_raw(self, raw, delegation, *, known_evidence, **kwargs):
        """Feed the same-body generator's raw JSON through the intake validator."""
        proposal = parse_goal_proposal(
            raw, known_evidence=known_evidence, known_records=tuple(self._tasks),
            now=datetime.fromisoformat(self.now()))
        return self.submit(delegation, proposal, known_evidence=known_evidence, **kwargs)

    def _coerce_proposal(self, proposal, known_evidence):
        if isinstance(proposal, GoalProposal):
            return proposal
        if isinstance(proposal, Mapping):
            proposal = canonical(dict(proposal))
        if isinstance(proposal, str):
            if known_evidence is None:
                raise TaskStoreError('ts.known-evidence',
                                     'raw proposals need the actual evidence identities')
            return parse_goal_proposal(proposal, known_evidence=set(known_evidence),
                                       known_records=tuple(self._tasks),
                                       now=datetime.fromisoformat(self.now()))
        raise TaskStoreError('ts.proposal-type', type(proposal).__name__)

    def _lists(self, constraints, deliverables, evidence, task_id):
        out = dict(constraint_ref=None, deliverable_ref=None, evidence_ref=None)
        if constraints:
            out['constraint_ref'] = self.blobs.put(canonical(list(constraints)),
                                                   kind='constraints', source=task_id)
        if deliverables:
            out['deliverable_ref'] = self.blobs.put(canonical(list(deliverables)),
                                                    kind='deliverables', source=task_id)
        if evidence:
            out['evidence_ref'] = self.blobs.put(canonical(list(evidence)),
                                                 kind='evidence', source=task_id)
        return out

    def _subtasks(self, items):
        out = []
        for index, item in enumerate(items, 1):
            if isinstance(item, Subtask):
                out.append(item)
                continue
            if isinstance(item, str):
                out.append(Subtask('sub-%d' % index, item))
                continue
            if isinstance(item, Mapping):
                out.append(Subtask(item.get('subtask_id') or 'sub-%d' % index,
                                   item['text'], item.get('status', 'pending')))
                continue
            raise TaskStoreError('ts.subtask-type', repr(type(item)))
        seen = set()
        for subtask in out:
            if subtask.subtask_id in seen:
                raise TaskStoreError('ts.subtask-duplicate', subtask.subtask_id)
            seen.add(subtask.subtask_id)
        return tuple(out)

    def _preconditions(self, items):
        out = []
        for index, item in enumerate(items, 1):
            if isinstance(item, Precondition):
                out.append(item)
                continue
            if isinstance(item, Mapping):
                out.append(Precondition(item.get('precondition_id') or 'pre-%d' % index,
                                        item['kind'], item.get('description', ''),
                                        item.get('status', 'pending'),
                                        bool(item.get('blocking', True)),
                                        item.get('detail_ref'), item.get('checked_at')))
                continue
            raise TaskStoreError('ts.precondition-type', repr(type(item)))
        return tuple(out)

    # -- re-read at an execution step ------------------------------------
    def read(self, task_id):
        """Re-read the current record (never a cached parse from an earlier step)."""
        return self._task(task_id)

    def delegation_text(self, task_id):
        return self.blobs.get(self._task(task_id).delegation_ref)

    def full_constraints(self, task_id):
        task = self._task(task_id)
        if task.constraint_ref is None:
            return list(task.constraints)
        return json.loads(self.blobs.get(task.constraint_ref))

    def full_deliverables(self, task_id):
        task = self._task(task_id)
        if task.deliverable_ref is None:
            return list(task.deliverables)
        return json.loads(self.blobs.get(task.deliverable_ref))

    def full_evidence(self, task_id):
        task = self._task(task_id)
        if task.evidence_ref is None:
            return list(task.evidence_ids)
        return json.loads(self.blobs.get(task.evidence_ref))

    def context(self, task_id, *, step=None, include_delegation=True):
        """Everything an execution step must re-read, read fresh from the store."""
        task = self._task(task_id)
        if step is not None and step not in {s.subtask_id for s in task.subtasks}:
            raise TaskStoreError('ts.subtask-unknown', str(step))
        delegation = self.delegation_text(task_id)
        first = task.current_subtask()
        payload = dict(task_id=task.task_id, objective=task.objective,
                       kind=task.kind, status=task.status, version=task.version,
                       lineage=task.lineage, acceptance=task.acceptance,
                       autonomy=task.autonomy, priority=task.priority,
                       deadline_at=task.deadline_at,
                       constraints=self.full_constraints(task_id),
                       deliverables=self.full_deliverables(task_id),
                       evidence_ids=self.full_evidence(task_id),
                       subtasks=[s.to_payload() for s in task.subtasks],
                       preconditions=[p.to_payload() for p in task.preconditions],
                       progress=task.progress.to_payload(),
                       revisions=[r.to_payload() for r in task.revisions],
                       remaining_subtasks=list(task.remaining_subtasks()),
                       current_subtask=first.subtask_id if first else None,
                       attempts=task.attempts,
                       delegation_ref=task.delegation_ref,
                       delegation_chars=len(delegation),
                       delegation_sha256=_sha_text(delegation),
                       receipts=[r.to_payload() for r in self.receipts_for(task_id)],
                       step=step)
        payload['delegation_text'] = delegation if include_delegation else None
        return payload

    def receipts_for(self, task_id):
        return [r for r in self._receipts if r.task_id == task_id]

    # -- lifecycle operations -------------------------------------------
    def accept(self, task_id, *, request_id=None, now=None):
        task = self._task(task_id)
        prior = self._existing_request(request_id)
        if prior is not None:
            if self._request_conflict(prior, kind='accept', task_id=task_id,
                                      action='accept'):
                raise TaskStoreError('ts.request-conflict', str(request_id))
            return dict(task=self._task(task_id), receipt=prior, duplicate=True)
        if task.status != 'proposed' or task.acceptance != 'pending':
            raise TaskStoreError('ts.status', 'only a pending proposal can be accepted')
        stamps = now or self.now()
        record = replace(task, acceptance='accepted', status='active',
                         updated_at=stamps, attempts=task.attempts + 1,
                         last_action=_epoch(stamps))
        receipt = self._new_receipt(record, kind='accept', action='accept',
                                    status='accepted', request_id=request_id,
                                    at=stamps, detail={'authority': 'user'})
        self._append('accept', records=(record,), receipt=receipt)
        return dict(task=record, receipt=receipt, duplicate=False)

    def readiness(self, task_id, *, resolvers=None):
        task = self._task(task_id)
        resolvers = resolvers or {}
        open_items = []
        for precondition in task.open_preconditions():
            status = precondition.status
            if precondition.kind in resolvers:
                outcome = resolvers[precondition.kind](precondition, task)
                status = outcome.get('status', status)
            if status in ('pending', 'failed'):
                open_items.append(dict(precondition_id=precondition.precondition_id,
                                       kind=precondition.kind, status=status,
                                       description=precondition.description))
        return dict(ready=not open_items, blocking=open_items)

    def start(self, task_id, *, request_id=None, resolvers=None, now=None):
        task = self._task(task_id)
        prior = self._existing_request(request_id)
        if prior is not None:
            if self._request_conflict(prior, kind='start', task_id=task_id,
                                      action='start'):
                raise TaskStoreError('ts.request-conflict', str(request_id))
            return dict(task=self._task(task_id), receipt=prior, duplicate=True)
        if task.status != 'active':
            raise TaskStoreError('ts.status', 'only an accepted task can start')
        resolvers = resolvers or {}
        stamps = now or self.now()
        for precondition in list(task.open_preconditions()):
            if precondition.kind not in resolvers:
                continue
            outcome = resolvers[precondition.kind](precondition, self._task(task_id))
            if (not isinstance(outcome, Mapping)
                    or outcome.get('status') not in PRECONDITION_STATUSES):
                raise TaskStoreError('ts.resolver-invalid', precondition.kind)
            task = self._set_precondition(
                task_id, precondition.precondition_id, status=outcome['status'],
                detail_ref=outcome.get('detail_ref'), detail=outcome.get('detail'),
                now=stamps)
        blockers = [dict(precondition_id=p.precondition_id, kind=p.kind,
                         status=p.status, description=p.description)
                    for p in task.open_preconditions()]
        if blockers:
            receipt = self._new_receipt(task, kind='start', action='start',
                                        status='refused', request_id=None, at=stamps,
                                        detail={'blocking': blockers})
            self._append('receipt', receipt=receipt)
            error = TaskStoreError('ts.precondition-pending',
                                   canonical(blockers)[:1000])
            error.blocking = blockers
            raise error
        first = task.current_subtask()
        progress = Progress(status='running',
                            current_subtask=first.subtask_id if first else None,
                            done=task.progress.done, failed=task.progress.failed,
                            note=task.progress.note, updated_at=stamps)
        subtasks = tuple(replace(s, status='active')
                         if first is not None and s.subtask_id == first.subtask_id
                         else s for s in task.subtasks)
        record = replace(task, progress=progress, subtasks=subtasks,
                         attempts=task.attempts + 1, last_action=_epoch(stamps),
                         updated_at=stamps)
        receipt = self._new_receipt(record, kind='start', action='start',
                                    status='running', request_id=request_id,
                                    subtask_id=progress.current_subtask, at=stamps,
                                    detail={'preconditions_checked':
                                            [p.precondition_id for p in record.preconditions]})
        self._append('start', records=(record,), receipt=receipt)
        return dict(task=record, receipt=receipt, duplicate=False)

    def _set_precondition(self, task_id, precondition_id, *, status, detail_ref=None,
                          detail=None, now=None):
        task = self._task(task_id)
        stamps = now or self.now()
        found = False
        items = []
        for precondition in task.preconditions:
            if precondition.precondition_id != precondition_id:
                items.append(precondition)
                continue
            found = True
            items.append(replace(precondition, status=status,
                                 detail_ref=detail_ref or precondition.detail_ref,
                                 checked_at=stamps))
        if not found:
            raise TaskStoreError('ts.precondition-unknown', str(precondition_id))
        record = replace(task, preconditions=tuple(items), updated_at=stamps)
        receipt = self._new_receipt(record, kind='precondition',
                                    action='check:' + str(status), status=status,
                                    at=stamps, detail_ref=detail_ref,
                                    detail={'precondition_id': precondition_id,
                                            'kind': next(p.kind for p in items
                                                         if p.precondition_id == precondition_id),
                                            'detail': detail or {}})
        self._append('precondition', records=(record,), receipt=receipt)
        return record

    def set_precondition(self, task_id, precondition_id, *, status, detail_ref=None,
                         detail=None, request_id=None, now=None):
        prior = self._existing_request(request_id)
        if prior is not None and prior.kind == 'precondition':
            return dict(task=self._task(task_id), receipt=prior, duplicate=True)
        record = self._set_precondition(task_id, precondition_id, status=status,
                                        detail_ref=detail_ref, detail=detail, now=now)
        receipt = self.receipts_for(task_id)[-1]
        if request_id:
            self._requests[request_id] = receipt
        return dict(task=record, receipt=receipt, duplicate=False)

    def progress(self, task_id, *, subtask=None, status, note='', request_id=None,
                 now=None):
        task = self._task(task_id)
        prior = self._existing_request(request_id)
        if prior is not None:
            if self._request_conflict(prior, kind='progress', task_id=task_id,
                                      action='progress:' + status):
                raise TaskStoreError('ts.request-conflict', str(request_id))
            return dict(task=self._task(task_id), receipt=prior, duplicate=True)
        if task.status != 'active':
            raise TaskStoreError('ts.status', 'progress needs an active task')
        if status not in ('started', 'done', 'failed', 'skipped', 'note'):
            raise TaskStoreError('ts.progress-status', repr(status))
        stamps = now or self.now()
        done = list(task.progress.done)
        failed = list(task.progress.failed)
        items = list(task.subtasks)
        current = task.progress.current_subtask
        if subtask is not None:
            if subtask not in {s.subtask_id for s in items}:
                raise TaskStoreError('ts.subtask-unknown', str(subtask))
            if status in ('started', 'done', 'failed', 'skipped'):
                target = {'started': 'active', 'done': 'done', 'failed': 'failed',
                          'skipped': 'skipped'}[status]
                items = [replace(s, status=target) if s.subtask_id == subtask
                         else s for s in items]
            if status == 'done' and subtask not in done:
                done.append(subtask)
            if status in ('done', 'started', 'skipped') and subtask in failed:
                failed.remove(subtask)
            if status == 'failed' and subtask not in failed:
                failed.append(subtask)
            if status == 'started':
                current = subtask
        remaining = [s.subtask_id for s in items if s.status not in ('done', 'skipped')]
        if status == 'failed':
            progress_status = 'blocked'
        elif not remaining:
            progress_status = 'done'
            current = None
        elif task.progress.status == 'not_started':
            progress_status = 'not_started'
        else:
            progress_status = 'running'
            current = remaining[0]
        progress = Progress(status=progress_status, current_subtask=current,
                            done=tuple(done), failed=tuple(failed),
                            note=note or task.progress.note, updated_at=stamps)
        record = replace(task, subtasks=tuple(items), progress=progress,
                         attempts=task.attempts + 1, last_action=_epoch(stamps),
                         updated_at=stamps)
        receipt = self._new_receipt(record, kind='progress',
                                    action='progress:' + status,
                                    status=status, subtask_id=subtask,
                                    request_id=request_id, at=stamps,
                                    detail={'note': note,
                                            'progress_status': progress_status,
                                            'remaining': remaining})
        self._append('progress', records=(record,), receipt=receipt)
        return dict(task=record, receipt=receipt, duplicate=False)

    def tool_receipt(self, task_id, *, tool, action, status, request_id,
                     subtask=None, result_ref=None, detail=None, now=None):
        if not request_id:
            raise TaskStoreError('ts.request-required',
                                 'tool receipts need a request id')
        task = self._task(task_id)
        prior = self._existing_request(request_id)
        if prior is not None:
            if self._request_conflict(prior, kind='tool', task_id=task_id,
                                      action=action):
                raise TaskStoreError('ts.request-conflict', str(request_id))
            return dict(task=self._task(task_id), receipt=prior, duplicate=True)
        if status not in ('ok', 'failed', 'denied'):
            raise TaskStoreError('ts.tool-status', repr(status))
        if task.status != 'active':
            raise TaskStoreError('ts.status', 'a tool receipt needs a live task')
        stamps = now or self.now()
        record = replace(task, attempts=task.attempts + 1,
                         last_action=_epoch(stamps), updated_at=stamps)
        receipt = self._new_receipt(record, kind='tool', action=action, status=status,
                                    subtask_id=subtask, request_id=request_id,
                                    tool=tool, detail_ref=result_ref,
                                    detail=detail or {}, at=stamps)
        self._append('tool', records=(record,), receipt=receipt)
        return dict(task=record, receipt=receipt, duplicate=False)

    def revise(self, parent_id, *, changes, author='user', note='', request_id=None,
               task_id=None, now=None):
        parent = self._task(parent_id)
        prior = self._existing_request(request_id)
        if prior is not None:
            if self._request_conflict(prior, kind='revise', task_id=parent_id,
                                      action='revise'):
                raise TaskStoreError('ts.request-conflict', str(request_id))
            child = next((t for t in self._tasks.values()
                          if t.lineage == parent_id and t.status == 'active'), None)
            if child is None:
                raise TaskStoreError('ts.not-found', 'revised child of ' + parent_id)
            return dict(task=child, parent=self._task(parent_id), receipt=prior,
                        duplicate=True)
        if parent.status not in ('active', 'proposed'):
            raise TaskStoreError('ts.status', 'only a live branch can be revised')
        stamps = now or self.now()
        child_id = task_id or changes.get('task_id')
        if not child_id:
            child_id = 'goal-' + uuid.uuid4().hex[:16]
        if not RECORD_ID.match(child_id):
            raise TaskStoreError('ts.task-id', child_id)
        if child_id in self._tasks:
            raise TaskStoreError('ts.duplicate-identity', child_id)
        unknown = sorted(set(changes) - {'objective', 'subtasks', 'constraints',
                                         'deliverables', 'preconditions', 'priority',
                                         'deadline_at', 'autonomy', 'evidence_ids',
                                         'acceptance', 'kind', 'task_id'})
        if unknown:
            raise TaskStoreError('ts.revise-field', ','.join(unknown))
        objective = changes.get('objective', parent.objective)
        subtasks = (self._subtasks(changes['subtasks'])
                    if 'subtasks' in changes else parent.subtasks)
        constraints = tuple(changes.get('constraints', parent.constraints))
        deliverables = tuple(changes.get('deliverables', parent.deliverables))
        preconditions = (self._preconditions(changes['preconditions'])
                         if 'preconditions' in changes else parent.preconditions)
        evidence = tuple(changes.get('evidence_ids', parent.evidence_ids))
        priority = float(changes.get('priority', parent.priority))
        deadline = changes.get('deadline_at', parent.deadline_at)
        if deadline is not None and not isinstance(deadline, (int, float)):
            deadline = _epoch(deadline)
        acceptance = changes.get('acceptance', parent.acceptance)
        if acceptance == 'declined':
            raise TaskStoreError('ts.decline-needs-cancel', 'use cancel()')
        status = 'proposed' if acceptance == 'pending' else 'active'
        constraint_ref = (self.blobs.put(canonical(list(constraints)),
                                         kind='constraints', source=child_id)
                          if constraints else None)
        deliverable_ref = (self.blobs.put(canonical(list(deliverables)),
                                          kind='deliverables', source=child_id)
                           if deliverables else None)
        evidence_ref = (self.blobs.put(canonical(list(evidence)),
                                       kind='evidence', source=child_id)
                        if evidence else None)
        done = tuple(parent.progress.done)
        carried = Progress(status='not_started'
                           if parent.progress.status == 'not_started' else 'running',
                           current_subtask=None, done=done,
                           failed=parent.progress.failed,
                           note=parent.progress.note, updated_at=stamps)
        remaining = [s for s in subtasks if s.subtask_id not in set(done)
                     and s.status not in ('done', 'skipped')]
        carried = replace(carried,
                          current_subtask=remaining[0].subtask_id if remaining else None)
        revision = Revision('rev-' + uuid.uuid4().hex[:12], stamps, author, note,
                            parent_id, tuple(sorted(changes)))
        self._submitted += 1
        child = TaskRecord(
            task_id=child_id, kind=changes.get('kind', parent.kind),
            objective=objective, delegation_ref=parent.delegation_ref,
            source_kind=parent.source_kind, acceptance=acceptance,
            autonomy=changes.get('autonomy', parent.autonomy), priority=priority,
            interruption_cost=parent.interruption_cost, deadline_at=deadline,
            evidence_ids=evidence[:INLINE_LIST_MAX], evidence_ref=evidence_ref,
            constraints=constraints[:INLINE_LIST_MAX],
            constraint_ref=constraint_ref,
            deliverables=deliverables[:INLINE_LIST_MAX],
            deliverable_ref=deliverable_ref, subtasks=subtasks,
            preconditions=preconditions,
            revisions=parent.revisions + (revision,), progress=carried,
            status=status, version=parent.version + 1, lineage=parent_id,
            attempts=parent.attempts, last_action=parent.last_action,
            created_at=stamps, updated_at=stamps, submitted_seq=self._submitted)
        parent_after = replace(parent, status='superseded', updated_at=stamps)
        receipt = self._new_receipt(child, kind='revise', action='revise',
                                    status='accepted', request_id=request_id,
                                    at=stamps,
                                    detail={'parent_task_id': parent_id,
                                            'changed': sorted(changes),
                                            'carried_done': list(done),
                                            'note': note, 'author': author})
        self._append('revise', records=(parent_after, child), receipt=receipt)
        return dict(task=child, parent=parent_after, receipt=receipt, duplicate=False)

    def complete(self, task_id, *, request_id=None, note='', force=False, now=None):
        task = self._task(task_id)
        prior = self._existing_request(request_id)
        if prior is not None:
            if self._request_conflict(prior, kind='complete', task_id=task_id,
                                      action='complete'):
                raise TaskStoreError('ts.request-conflict', str(request_id))
            return dict(task=self._task(task_id), receipt=prior, duplicate=True)
        if task.status != 'active':
            raise TaskStoreError('ts.status', 'only an active task can complete')
        remaining = task.remaining_subtasks()
        if remaining and not force:
            error = TaskStoreError('ts.subtasks-open', ','.join(remaining))
            error.remaining = remaining
            raise error
        stamps = now or self.now()
        subtasks = tuple(replace(s, status='skipped') if s.subtask_id in set(remaining)
                         else s for s in task.subtasks)
        progress = Progress(status='done', current_subtask=None,
                            done=task.progress.done, failed=task.progress.failed,
                            note=note or task.progress.note, updated_at=stamps)
        record = replace(task, status='completed', subtasks=subtasks,
                         progress=progress, attempts=task.attempts + 1,
                         last_action=_epoch(stamps), updated_at=stamps)
        receipt = self._new_receipt(record, kind='complete', action='complete',
                                    status='completed', request_id=request_id,
                                    at=stamps,
                                    detail={'note': note, 'forced': bool(force),
                                            'skipped_on_complete': list(remaining)})
        self._append('complete', records=(record,), receipt=receipt)
        return dict(task=record, receipt=receipt, duplicate=False)

    def cancel(self, task_id, *, reason, request_id=None, now=None):
        task = self._task(task_id)
        prior = self._existing_request(request_id)
        if prior is not None:
            if self._request_conflict(prior, kind='cancel', task_id=task_id,
                                      action='cancel'):
                raise TaskStoreError('ts.request-conflict', str(request_id))
            return dict(task=self._task(task_id), receipt=prior, duplicate=True)
        if task.status not in ('active', 'proposed'):
            raise TaskStoreError('ts.status', 'only a live task can be cancelled')
        if not reason:
            raise TaskStoreError('ts.cancel-reason', 'explicit reason required')
        stamps = now or self.now()
        progress = replace(task.progress, status='cancelled', updated_at=stamps)
        record = replace(task, status='cancelled', progress=progress,
                         attempts=task.attempts + 1, last_action=_epoch(stamps),
                         updated_at=stamps)
        receipt = self._new_receipt(record, kind='cancel', action='cancel',
                                    status='cancelled', request_id=request_id,
                                    at=stamps, detail={'reason': reason})
        self._append('cancel', records=(record,), receipt=receipt)
        return dict(task=record, receipt=receipt, duplicate=False)

    # -- active set projection ------------------------------------------
    def schedule(self, *, log=True):
        """Bind the active set onto the fixed physical slot table.

        Order is submission order: this store never invents a preference rule.
        A task beyond capacity stays in the library and is reported as unbound;
        it is never silently evicted, and completion of a bound task binds the
        next one on the following schedule.
        """
        active = [t for t in self._tasks.values() if t.status == 'active']
        active.sort(key=lambda t: (t.submitted_seq, t.task_id))
        bound = active[:self.capacity]
        unbound = active[self.capacity:]
        table = GoalState.empty(self.capacity, self.evidence_slots)
        unique_evidence = {e for t in bound for e in t.evidence_ids}
        if len(unique_evidence) > self.evidence_slots:
            raise TaskStoreError('ts.evidence-capacity',
                                 '%d cited identities > %d slots'
                                 % (len(unique_evidence), self.evidence_slots))
        order = []
        for task in bound:
            record = GoalRecord(
                record_id=task.task_id, status='active',
                source_kind=task.source_kind,
                acceptance='accepted' if task.acceptance == 'accepted' else 'pending',
                priority=task.priority, interruption_cost=task.interruption_cost,
                deadline_at=task.deadline_at, can_act=task.autonomy == 'can_act',
                evidence_ids=tuple(task.evidence_ids),
                evidence_ok=0, evidence_open=0, version=task.version, lineage=0)
            try:
                slot = table.install(record)
            except GoalStateError as exc:
                raise TaskStoreError('ts.projection-' + exc.code.replace('gs.', ''),
                                     str(exc))
            table['attempts'][0, slot] = int(task.attempts)
            table['last_action'][0, slot] = float(task.last_action)
            order.append((task.task_id, int(table['ids'][0, slot]), slot))
        table.validate()
        generation = (self._projection['generation'] + 1) if self._projection else 1
        signature = [(task_id, slot) for task_id, _, slot in order]
        if log and (self._projection is None
                    or self._projection['signature'] != signature):
            binding = dict(generation=generation, capacity=self.capacity,
                           order=[task_id for task_id, _, _ in order],
                           bindings=[dict(task_id=task_id, row_identity=identity,
                                          slot=slot)
                                     for task_id, identity, slot in order],
                           unbound=[t.task_id for t in unbound], at=self.now())
            self._append('binding', binding=binding)
        self._projection = dict(generation=generation, table=table, order=order,
                                unbound=tuple(t.task_id for t in unbound),
                                signature=signature, at=self.now())
        return self.projection_report()

    def projection_report(self):
        if self._projection is None:
            self.schedule(log=False)
        projection = self._projection
        return dict(generation=projection['generation'], capacity=self.capacity,
                    order=[task_id for task_id, _, _ in projection['order']],
                    bindings=[dict(task_id=task_id, row_identity=identity, slot=slot)
                              for task_id, identity, slot in projection['order']],
                    unbound=list(projection['unbound']),
                    free_slots=self.capacity - len(projection['order']),
                    signature=[list(pair) for pair in projection['signature']])

    def slot_of(self, task_id):
        if self._projection is None:
            self.schedule(log=False)
        for bound_id, _, slot in self._projection['order']:
            if bound_id == task_id:
                return slot
        raise TaskStoreError('ts.unbound', str(task_id))

    def is_bound(self, task_id):
        if self._projection is None:
            self.schedule(log=False)
        return any(bound == task_id for bound, _, _ in self._projection['order'])

    def projection_table(self):
        if self._projection is None:
            self.schedule(log=False)
        return self._projection['table']

    def full_feature_block(self, *, now=None, elapsed=0.0, total_elapsed=0.0,
                           ctx_cost=0.0, resource=1.0):
        """The fixed-slot D06 feature matrix (capacity + 1 rows, null last)."""
        if self._projection is None:
            self.schedule(log=False)
        table = self._projection['table']
        now = float(time.time() if now is None else now)
        slots = self.capacity
        t = table
        revision = (t['version'][0] > 0).to(torch.int64)
        per_goal = goal_feature_rows(t['ids'][0], t['status'][0], t['accept'][0],
                                     t['source'][0], t['priority'][0], t['icost'][0],
                                     t['deadline_at'][0], t['deadline_mask'][0],
                                     t['attempts'][0], t['last_action'][0],
                                     t['evidence'][0], t['ev_ok'][0], t['ev_open'][0],
                                     revision, t['can_act'][0], now, slots=slots)
        context = context_features(t['ids'][0], t['status'][0], t['accept'][0],
                                   t['priority'][0], t['deadline_at'][0],
                                   t['deadline_mask'][0], now, elapsed,
                                   total_elapsed, ctx_cost, resource, slots=slots)
        occupied = torch.cat((t['ids'][0] > 0,
                              torch.ones(1, dtype=torch.bool)))[None]
        can_act = torch.cat((t['can_act'][0].bool(),
                             torch.zeros(1, dtype=torch.bool)))[None]
        allowed = legality_mask(occupied, can_act=can_act, resource=resource)
        allowed = resource_legal(allowed, float(resource))
        return dict(per_goal=per_goal, context=context, allowed=allowed,
                    capacity=slots, generation=self._projection['generation'],
                    per_goal_names=PER_GOAL_NAMES, context_names=CTX_NAMES,
                    actions=ACTIONS, resource=float(resource))

    def feature_block(self, *, now=None, elapsed=0.0, total_elapsed=0.0,
                      ctx_cost=0.0, resource=1.0):
        """Dynamic-row feature block: bound rows compacted, null row last.

        Rows are gathered from the full D06 feature matrix, so a task's values
        are bit-identical to the fixed-slot table and the empty rows are gone;
        the context vector keeps the physical slot count as denominator.
        """
        full = self.full_feature_block(now=now, elapsed=elapsed,
                                       total_elapsed=total_elapsed,
                                       ctx_cost=ctx_cost, resource=resource)
        slots = full['capacity']
        keep = [slot for _, _, slot in self._projection['order']] + [slots]
        rows = [task_id for task_id, _, _ in self._projection['order']] + [None]
        return dict(per_goal=full['per_goal'][keep], context=full['context'],
                    allowed=full['allowed'][:, keep, :], rows=rows,
                    null_row=len(rows) - 1, capacity=slots,
                    generation=full['generation'],
                    per_goal_names=full['per_goal_names'],
                    context_names=full['context_names'], actions=full['actions'],
                    resource=full['resource'])

    # -- checkpoint / resume --------------------------------------------
    def state_payload(self):
        return dict(tasks={task_id: self._tasks[task_id].to_payload()
                           for task_id in sorted(self._tasks)},
                    receipts=[receipt.to_payload() for receipt in self._receipts],
                    bindings=list(self._bindings))

    def state_digest(self):
        return hashlib.sha256(canonical(self.state_payload()).encode('utf-8')).hexdigest()

    def checkpoint(self, path=None):
        if self.journal._handle is not None:
            self.journal._handle.flush()
        payload = dict(schema=SCHEMA_CHECKPOINT, version=1,
                       store_id=self.store_id, at=self.now(),
                       journal_bytes=self.journal.offset,
                       journal_seq=self.journal.seq, chain=self.journal.chain,
                       state_sha256=self.state_digest(),
                       counts=dict(tasks=len(self._tasks),
                                   receipts=len(self._receipts),
                                   blobs=len(self.blobs),
                                   bindings=len(self._bindings)),
                       capacity=self.capacity,
                       evidence_slots=self.evidence_slots)
        target = Path(path) if path else self.directory / 'task-checkpoint.json'
        _atomic_write(target, canonical(payload) + '\n')
        payload['path'] = str(target)
        return payload

    def load_checkpoint(self, path=None):
        target = Path(path) if path else self.directory / 'task-checkpoint.json'
        if not target.exists():
            raise TaskStoreError('ts.checkpoint-missing', str(target))
        payload = json.loads(target.read_text(encoding='utf-8'))
        if payload.get('schema') != SCHEMA_CHECKPOINT:
            raise TaskStoreError('ts.checkpoint-schema', str(target))
        return payload

    def resume(self, *, task_id=None, request_id=None, now=None):
        """Continue: verify the checkpoint, then report live work and receipts."""
        checkpoint = self.load_checkpoint()
        if self.journal.seq < checkpoint['journal_seq']:
            raise TaskStoreError('ts.checkpoint-missing-prefix',
                                 'journal %d < checkpoint %d'
                                 % (self.journal.seq, checkpoint['journal_seq']))
        if checkpoint['journal_seq'] == self.journal.seq:
            digest = self.state_digest()
            if digest != checkpoint['state_sha256']:
                raise TaskStoreError('ts.checkpoint-diverged',
                                     'runtime %s != checkpoint %s'
                                     % (digest, checkpoint['state_sha256']))
        stamps = now or self.now()
        receipt = None
        if request_id:
            prior = self._existing_request(request_id)
            if prior is not None:
                receipt = prior
            else:
                anchor = self._task(task_id) if task_id else next(
                    (t for t in self._tasks.values() if t.status == 'active'),
                    next(iter(self._tasks.values()), None))
                if anchor is None:
                    raise TaskStoreError('ts.not-found', 'no task to resume')
                receipt = self._new_receipt(anchor, kind='resume', action='resume',
                                            status='resumed', request_id=request_id,
                                            at=stamps,
                                            detail={'checkpoint': checkpoint['state_sha256'],
                                                    'journal_seq': self.journal.seq})
                self._append('receipt', receipt=receipt)
        current = self._task(task_id) if task_id else None
        return dict(checkpoint=checkpoint, current_task=current,
                    current=(self.context(task_id, include_delegation=False)
                             if current is not None else None),
                    active=[t.task_id for t in self._tasks.values()
                            if t.status == 'active'],
                    completed=[t.task_id for t in self._tasks.values()
                               if t.status == 'completed'],
                    receipts=len(self._receipts), receipt=receipt,
                    state_sha256=self.state_digest())

    # -- reporting -------------------------------------------------------
    def report(self):
        tasks = list(self._tasks.values())
        return dict(schema=self.SCHEMA, store_id=self.store_id,
                    directory=str(self.directory), capacity=self.capacity,
                    evidence_slots=self.evidence_slots,
                    counts=dict(tasks=len(tasks),
                                proposed=sum(t.status == 'proposed' for t in tasks),
                                active=sum(t.status == 'active' for t in tasks),
                                completed=sum(t.status == 'completed' for t in tasks),
                                cancelled=sum(t.status == 'cancelled' for t in tasks),
                                superseded=sum(t.status == 'superseded' for t in tasks),
                                receipts=len(self._receipts), blobs=len(self.blobs),
                                bindings=len(self._bindings)),
                    tasks=[dict(task_id=t.task_id, status=t.status, version=t.version,
                                lineage=t.lineage, acceptance=t.acceptance,
                                remaining=list(t.remaining_subtasks()),
                                progress=t.progress.status,
                                revisions=len(t.revisions),
                                delegation_chars=self.blobs.chars(t.delegation_ref))
                           for t in tasks],
                    journal=dict(seq=self.journal.seq, bytes=self.journal.offset,
                                 chain=self.journal.chain),
                    state_sha256=self.state_digest(),
                    projection=self.projection_report())

    def verify(self):
        rebuilt = TaskStore.__new__(TaskStore)
        rebuilt.journal = self.journal
        rebuilt.blobs = self.blobs
        rebuilt._tasks = {}
        rebuilt._receipts = []
        rebuilt._requests = {}
        rebuilt._bindings = []
        rebuilt._submitted = 0
        for entry in self.journal.iter_entries():
            rebuilt._apply(entry)
        digest = rebuilt.state_digest()
        if digest != self.state_digest():
            raise TaskStoreError('ts.verify-diverged', digest)
        return dict(ok=True, seq=self.journal.seq, state_sha256=digest,
                    tasks=len(self._tasks), receipts=len(self._receipts))
