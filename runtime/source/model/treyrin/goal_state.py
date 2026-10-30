"""D06 integration: per-goal state as tensors, with explicit schema migration.

Design contract
---------------
* One row per goal slot.  Every field that a control head may read is a tensor;
  the only non-tensor payload is the string<->int identity table (a tensor
  cannot hold a string, and pretending otherwise would hide the table).
* An *empty* old state (no identities, zero counters) initialises directly.
* Acceptance is orthogonal to lifecycle in stored tensors: "pending" means
  "not established".  A migrated v6 row may be pending with any status because
  v6 never recorded acceptance; only "declined" asserts an act and therefore
  requires retracted.  A freshly authored GoalRecord keeps the stricter rule.
* A *non-empty* old state is NEVER rebound field-by-field.  It must come with an
  explicit MigrationPlan naming every legacy field it maps, every field it
  drops and every slot it writes.  Anything else raises GoalStateError with a
  stable code, so a silent partial restore is impossible by construction.
* dtypes are part of the contract: journal identities/status/source/acceptance/
  versions are int64, priority/interruption cost are float32, epoch seconds are
  float64 (float32 at 1.7e9 resolves to about 128 s, which cannot represent a
  one-second deadline), evidence references are a bool bitmap.

B = 1 mutations are implemented (the reference integration is single-stream);
multi-batch tensors are accepted and validated, mutations index batch 0.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import torch

SCHEMA = 'd06-goal-state'
SCHEMA_VERSION = 1

STATUSES = ('active', 'superseded', 'retracted', 'completed')
ACCEPTANCE = ('pending', 'accepted', 'declined')
SOURCE_KINDS = ('user_message', 'device_log', 'document', 'tool_result', 'model_output', 'schedule')
AUTONOMY = ('notify_only', 'can_act')

INT_FIELDS = ('ids', 'status', 'source', 'accept', 'version', 'lineage',
              'deadline_mask', 'attempts', 'ev_ok', 'ev_open', 'can_act')
FLOAT_FIELDS = ('priority', 'icost')
EPOCH_FIELDS = ('deadline_at', 'last_action')
BITMAP_FIELDS = ('evidence',)
FIELDS = INT_FIELDS + FLOAT_FIELDS + EPOCH_FIELDS + BITMAP_FIELDS


class GoalStateError(ValueError):
    """Typed rejection: .code is a stable token, never a free-form message."""

    def __init__(self, code, message=''):
        super().__init__('%s: %s' % (code, message))
        self.code = code


@dataclass(frozen=True)
class GoalRecord:
    """One typed goal row as it enters the tensor state."""
    record_id: str
    status: str = 'active'
    source_kind: str = 'user_message'
    acceptance: str = 'accepted'
    priority: float = 0.5
    interruption_cost: float = 0.0
    deadline_at: float | None = None
    can_act: bool = False
    evidence_ids: tuple = ()
    evidence_ok: int = 0
    evidence_open: int = 0
    version: int = 0
    lineage: int = 0

    def __post_init__(self):
        if not isinstance(self.record_id, str) or not self.record_id:
            raise GoalStateError('gs.record-id', 'record_id must be a nonempty string')
        if self.status not in STATUSES:
            raise GoalStateError('gs.status', repr(self.status))
        if self.source_kind not in SOURCE_KINDS:
            raise GoalStateError('gs.source', repr(self.source_kind))
        if self.acceptance not in ACCEPTANCE:
            raise GoalStateError('gs.acceptance', repr(self.acceptance))
        if self.acceptance == 'declined' and self.status != 'retracted':
            raise GoalStateError('gs.acceptance-basis', 'declined requires status=retracted')
        if self.acceptance == 'pending' and self.status != 'active':
            raise GoalStateError('gs.acceptance-basis', 'pending requires status=active')
        for name in ('priority', 'interruption_cost'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise GoalStateError('gs.field-invalid', name)
        if not 0.0 <= float(self.priority) <= 1.0:
            raise GoalStateError('gs.field-invalid', 'priority outside [0,1]')
        if float(self.interruption_cost) < 0.0:
            raise GoalStateError('gs.field-invalid', 'negative interruption cost')
        if self.deadline_at is not None:
            if isinstance(self.deadline_at, bool) or not isinstance(self.deadline_at, (int, float)):
                raise GoalStateError('gs.field-invalid', 'deadline_at')
            if not math.isfinite(self.deadline_at) or float(self.deadline_at) <= 0.0:
                raise GoalStateError('gs.field-invalid', 'deadline_at must be a positive epoch second')
        if isinstance(self.evidence_ids, str) or any(not isinstance(e, str) for e in self.evidence_ids):
            raise GoalStateError('gs.field-invalid', 'evidence_ids must be strings')
        if type(self.can_act) is not bool:
            raise GoalStateError('gs.field-invalid', 'can_act must be an explicit bool')
        if type(self.evidence_open) is not int or self.evidence_open < 0:
            raise GoalStateError('gs.field-invalid', 'evidence_open must be a nonnegative int')
        if type(self.evidence_ok) is not int or self.evidence_ok < 0:
            raise GoalStateError('gs.field-invalid', 'evidence_ok must be a nonnegative int')
        if self.evidence_ok > len(self.evidence_ids):
            raise GoalStateError('gs.field-invalid', 'evidence_ok cannot exceed the cited identities')
        if type(self.version) is not int or self.version < 0:
            raise GoalStateError('gs.field-invalid', 'version')
        if type(self.lineage) is not int or self.lineage < 0:
            raise GoalStateError('gs.field-invalid', 'lineage')


@dataclass(frozen=True)
class MigrationPlan:
    """Explicit, reviewable statement of what a non-empty restore does."""
    source_schema: str
    fields: dict                      # legacy name -> target name, or None to drop
    accept_nonempty: bool = False
    slot_map: tuple = ()              # ((source_slot, target_slot), ...)
    reason: str = ''
    drop_reasons: dict = field(default_factory=dict)
    sentinels: dict = field(default_factory=dict)   # target field -> (from, to)

    def __post_init__(self):
        if not isinstance(self.source_schema, str) or not self.source_schema:
            raise GoalStateError('gs.plan', 'source_schema required')
        if not isinstance(self.fields, dict) or not self.fields:
            raise GoalStateError('gs.plan', 'field mapping required')
        if not isinstance(self.accept_nonempty, bool):
            raise GoalStateError('gs.plan', 'accept_nonempty must be an explicit bool')
        for source, target in self.fields.items():
            if target is not None and (not isinstance(target, str) or target not in FIELDS):
                raise GoalStateError('gs.plan', 'unknown target field %r' % (target,))
            if target is None and source not in self.drop_reasons:
                raise GoalStateError('gs.plan', 'dropping %r needs a stated reason' % (source,))
        if not self.reason:
            raise GoalStateError('gs.plan', 'migration reason required')
        for name, pair in self.sentinels.items():
            if name not in FIELDS:
                raise GoalStateError('gs.plan', 'sentinel on unknown field %r' % (name,))
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise GoalStateError('gs.plan', 'sentinel must be a (from, to) pair')


def _index(value):
    return STATUSES.index(value)


class GoalState:
    """Per-slot goal tensors plus the identity tables needed to restore names."""

    def __init__(self, slots, evidence_slots, batch, dtype, device, arrays=None,
                 goal_names=None, evidence_names=None, next_identity=1, schema=SCHEMA,
                 version=SCHEMA_VERSION):
        self.slots = int(slots)
        self.evidence_slots = int(evidence_slots)
        self.batch = int(batch)
        self.dtype = dtype
        self.device = device
        self.schema = schema
        self.version = int(version)
        self.goal_names = dict(goal_names or {})
        self.evidence_names = dict(evidence_names or {})
        self.next_identity = int(next_identity)
        shape = dict(ids=(batch, slots), status=(batch, slots), source=(batch, slots),
                     accept=(batch, slots), version=(batch, slots), lineage=(batch, slots),
                     deadline_mask=(batch, slots), attempts=(batch, slots),
                     ev_ok=(batch, slots), ev_open=(batch, slots), can_act=(batch, slots),
                     priority=(batch, slots), icost=(batch, slots),
                     deadline_at=(batch, slots), last_action=(batch, slots),
                     evidence=(batch, slots, self.evidence_slots))
        self._t = {}
        for name in FIELDS:
            if arrays is not None and name in arrays:
                # dtype must be given here: torch.as_tensor([1.7e9]) is float32
                # and would silently round a one-second deadline away.
                value = torch.as_tensor(arrays[name], device=device, dtype=self._dtype(name))
            else:
                value = torch.zeros(shape[name], device=device, dtype=self._dtype(name))
            if tuple(value.shape) != shape[name]:
                raise GoalStateError('gs.shape-mismatch', name)
            self._t[name] = value
        self.validate()

    # -- construction -----------------------------------------------------
    @staticmethod
    def _dtype(name):
        if name in BITMAP_FIELDS:
            return torch.bool
        if name in EPOCH_FIELDS:
            return torch.float64
        if name in FLOAT_FIELDS:
            return torch.float32
        return torch.int64

    @classmethod
    def empty(cls, slots=8, evidence_slots=8, batch=1, *, device='cpu', dtype=torch.float32):
        return cls(slots, evidence_slots, batch, dtype, device)

    # -- tensor access ----------------------------------------------------
    def __getitem__(self, name):
        if name not in self._t:
            raise KeyError(name)
        return self._t[name]

    def tensors(self):
        return {name: value for name, value in self._t.items()}

    def clone(self):
        return GoalState(self.slots, self.evidence_slots, self.batch, self.dtype, self.device,
                         {k: v.clone() for k, v in self._t.items()}, self.goal_names,
                         self.evidence_names, self.next_identity, self.schema, self.version)

    def to(self, device=None, dtype=None):
        out = self.clone()
        out.device = self.device if device is None else device
        out.dtype = self.dtype if dtype is None else dtype
        for name in FIELDS:
            target = GoalState._dtype(name)
            if name in FLOAT_FIELDS and dtype is not None:
                target = dtype
            out._t[name] = self._t[name].to(out.device, target)
        return out

    # -- queries ----------------------------------------------------------
    def occupied(self):
        return self['ids'][0] > 0

    def slot_of(self, identity):
        hit = (self['ids'][0] == int(identity)).nonzero().flatten()
        if not hit.numel():
            raise GoalStateError('gs.unknown-identity', str(identity))
        return int(hit[0])

    def num_of(self, record_id):
        if record_id not in self.goal_names:
            raise GoalStateError('gs.unknown-record', record_id)
        return int(self.goal_names[record_id])

    def active_slots(self):
        return [int(s) for s in range(self.slots)
                if int(self['ids'][0, s]) > 0 and int(self['status'][0, s]) == _index('active')]

    def remaining(self, now):
        mask = self['deadline_mask'][0].bool()
        return torch.where(mask, self['deadline_at'][0] - float(now),
                           torch.full_like(self['deadline_at'][0], float('nan')))

    def evidence_gap(self):
        need = self['evidence'][0].sum(-1) > 0
        return need & ~self['evidence'][0].all(-1)

    # -- mutation (batch 0) ----------------------------------------------
    def _slot_for_write(self, slot):
        if slot is not None:
            if not 0 <= int(slot) < self.slots:
                raise GoalStateError('gs.slot-range', str(slot))
            if int(self['ids'][0, slot]) != 0:
                raise GoalStateError('gs.slot-occupied', str(slot))
            return int(slot)
        free = (self['ids'][0] == 0).nonzero().flatten()
        if not free.numel():
            raise GoalStateError('gs.capacity', 'accepted goals are never silently evicted')
        return int(free[0])

    def install(self, record: GoalRecord, *, slot=None):
        if record.record_id in self.goal_names:
            raise GoalStateError('gs.duplicate-identity', record.record_id)
        slot = self._slot_for_write(slot)
        identity = self.next_identity
        self.next_identity += 1
        self.goal_names[record.record_id] = identity
        self['ids'][0, slot] = identity
        self['status'][0, slot] = _index(record.status)
        self['source'][0, slot] = SOURCE_KINDS.index(record.source_kind)
        self['accept'][0, slot] = ACCEPTANCE.index(record.acceptance)
        self['version'][0, slot] = int(record.version)
        self['lineage'][0, slot] = int(record.lineage)
        self['priority'][0, slot] = float(record.priority)
        self['icost'][0, slot] = float(record.interruption_cost)
        self['can_act'][0, slot] = int(record.can_act)
        self['ev_open'][0, slot] = int(record.evidence_open)
        self['ev_ok'][0, slot] = int(record.evidence_ok)
        if record.deadline_at is None:
            self['deadline_at'][0, slot] = 0.0
            self['deadline_mask'][0, slot] = 0
        else:
            self['deadline_at'][0, slot] = float(record.deadline_at)
            self['deadline_mask'][0, slot] = 1
        for evidence in record.evidence_ids:
            self.attach_evidence(int(identity), evidence)
        return slot

    def attach_evidence(self, identity, evidence_id):
        if evidence_id not in self.evidence_names:
            if len(self.evidence_names) >= self.evidence_slots:
                raise GoalStateError('gs.evidence-capacity', evidence_id)
            self.evidence_names[evidence_id] = len(self.evidence_names)
        slot = self.slot_of(identity)
        self['evidence'][0, slot, self.evidence_names[evidence_id]] = True

    def set_status(self, identity, status, *, acceptance=None):
        if status not in STATUSES:
            raise GoalStateError('gs.status', repr(status))
        slot = self.slot_of(identity)
        self['status'][0, slot] = _index(status)
        if acceptance is not None:
            if acceptance not in ACCEPTANCE:
                raise GoalStateError('gs.acceptance', repr(acceptance))
            self['accept'][0, slot] = ACCEPTANCE.index(acceptance)
        return slot

    def cancel(self, identity):
        slot = self.slot_of(identity)
        if int(self['status'][0, slot]) == _index('completed'):
            raise GoalStateError('gs.terminal', 'a completed goal cannot be cancelled')
        return self.set_status(identity, 'retracted', acceptance='declined')

    def complete(self, identity):
        return self.set_status(identity, 'completed')

    def revise(self, parent_identity, record: GoalRecord):
        """New identity supersedes the parent and inherits its interruption lineage."""
        parent = self.slot_of(parent_identity)
        if int(self['status'][0, parent]) != _index('active'):
            raise GoalStateError('gs.revision-branch', 'only the live branch can be revised')
        if record.record_id in self.goal_names:
            raise GoalStateError('gs.duplicate-identity', record.record_id)
        child = GoalRecord(**{**record.__dict__,
                               'version': int(self['version'][0, parent]) + 1,
                               'lineage': int(parent_identity)})
        self.set_status(parent_identity, 'superseded')
        slot = self.install(child)
        self['attempts'][0, slot] = self['attempts'][0, parent]
        self['last_action'][0, slot] = self['last_action'][0, parent]
        return slot

    def record_attempt(self, identity, now):
        slot = self.slot_of(identity)
        self['attempts'][0, slot] += 1
        self['last_action'][0, slot] = float(now)
        return slot

    # -- invariants -------------------------------------------------------
    def validate(self):
        ids = self['ids']
        if (ids < 0).any():
            raise GoalStateError('gs.identity', 'negative identity')
        positive = ids[ids > 0]
        if positive.numel() and int(positive.max()) >= self.next_identity:
            raise GoalStateError('gs.identity', 'identity beyond the allocating counter')
        if positive.numel() and int(torch.unique(positive).numel()) != int(positive.numel()):
            raise GoalStateError('gs.duplicate-identity', 'two slots share an identity')
        if bool((self['status'] < 0).any()) or bool((self['status'] >= len(STATUSES)).any()):
            raise GoalStateError('gs.status', 'out of range')
        if bool((self['accept'] < 0).any()) or bool((self['accept'] >= len(ACCEPTANCE)).any()):
            raise GoalStateError('gs.acceptance', 'out of range')
        if bool((self['source'] < 0).any()) or bool((self['source'] >= len(SOURCE_KINDS)).any()):
            raise GoalStateError('gs.source', 'out of range')
        for name in ('priority', 'icost', 'deadline_at', 'last_action'):
            if not bool(torch.isfinite(self[name]).all()):
                raise GoalStateError('gs.nonfinite', name)
        if (self['ev_open'] < 0).any() or (self['ev_ok'] < 0).any():
            raise GoalStateError('gs.counter', 'negative evidence counter')
        if (self['attempts'] < 0).any() or (self['version'] < 0).any() or (self['lineage'] < 0).any():
            raise GoalStateError('gs.counter', 'negative counter')
        mask = self['deadline_mask']
        if int(((mask != 0) & (mask != 1)).sum()):
            raise GoalStateError('gs.deadline-mask', 'must be 0/1')
        if bool((mask.bool() & (self['deadline_at'] <= 0)).any()):
            raise GoalStateError('gs.deadline-mask', 'masked deadline needs a positive epoch')
        if bool((~mask.bool() & (self['deadline_at'] != 0)).any()):
            raise GoalStateError('gs.deadline-mask', 'unmasked deadline must be zero')
        declined = self['accept'] == ACCEPTANCE.index('declined')
        if bool((declined & (self['status'] != _index('retracted'))).any()):
            raise GoalStateError('gs.acceptance-basis', 'declined requires retracted')
        # pending means "acceptance not established".  A migrated v6 row never
        # had an acceptance field, so pending may coexist with any lifecycle
        # there; only a declined acceptance asserts an act and needs retracted.
        # Fresh typed records keep the stricter GoalRecord rule.
        empty = ids == 0
        for name in FIELDS:
            if name == 'ids':
                continue
            if bool(self[name][empty].any()):
                raise GoalStateError('gs.empty-slot', 'empty slots must be all-zero: ' + name)
        if bool(((self['can_act'] != 0) & (self['can_act'] != 1)).any()):
            raise GoalStateError('gs.can-act', 'must be 0/1')
        return self

    # -- persistence ------------------------------------------------------
    def payload(self):
        return dict(schema=self.schema, version=self.version, slots=self.slots,
                    evidence_slots=self.evidence_slots, batch=self.batch,
                    dtype=str(self.dtype).replace('torch.', ''),
                    goal_names=dict(sorted(self.goal_names.items())),
                    evidence_names=dict(sorted(self.evidence_names.items())),
                    next_identity=self.next_identity,
                    arrays={name: self[name].detach().cpu().tolist() for name in FIELDS})

    def digest(self):
        blob = json.dumps(self.payload(), sort_keys=True, ensure_ascii=True,
                          separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(blob).hexdigest()

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(self.payload(), handle, sort_keys=True, ensure_ascii=True,
                      separators=(',', ':'))
        return self.digest()

    @classmethod
    def load(cls, path, *, device='cpu'):
        with open(path, 'r', encoding='utf-8') as handle:
            return cls.restore(json.load(handle), device=device)

    @classmethod
    def restore(cls, payload, *, device='cpu', dtype=torch.float32):
        if payload.get('schema') != SCHEMA:
            raise GoalStateError('gs.schema', 'restore needs the %r schema' % SCHEMA)
        if int(payload.get('version', 0)) > SCHEMA_VERSION:
            raise GoalStateError('gs.schema-newer',
                                 'stored %r newer than %r' % (payload.get('version'), SCHEMA_VERSION))
        return cls(payload['slots'], payload['evidence_slots'], payload['batch'], dtype, device,
                   payload['arrays'], payload['goal_names'], payload['evidence_names'],
                   payload['next_identity'], payload['schema'], payload['version'])

    # -- migration --------------------------------------------------------
    @classmethod
    def migrate(cls, payload, *, plan: MigrationPlan | None = None, slots=None,
                evidence_slots=8, batch=1, device='cpu', dtype=torch.float32):
        """Empty/absent old state initialises; anything non-empty needs a plan.

        Returns (state, report).  The report is what a reviewer reads: which
        legacy fields were mapped, which were dropped and why, how many slots
        were written, and whether the source was empty.
        """
        if payload is None or payload.get('empty') is True:
            state = cls.empty(slots or 8, evidence_slots, batch, device=device, dtype=dtype)
            return state, dict(migrated='empty-init', source_schema=None, mapped={},
                               dropped={}, slots_written=0, nonempty_source=False)
        if not isinstance(payload, dict):
            raise GoalStateError('gs.payload', 'migration payload must be a mapping')
        source_schema = payload.get('schema')
        nonempty = cls._is_nonempty(payload)
        if plan is None:
            if source_schema == SCHEMA and not nonempty and payload.get('version', 0) <= SCHEMA_VERSION:
                state = cls.restore(payload, device=device, dtype=dtype)
                return state, dict(migrated='same-schema-empty', source_schema=source_schema,
                                   mapped={}, dropped={}, slots_written=0, nonempty_source=False)
            raise GoalStateError('gs.nonempty-without-migration' if nonempty else 'gs.migration-plan-required',
                                 'source schema %r cannot be rebound implicitly' % (source_schema,))
        if source_schema != plan.source_schema:
            raise GoalStateError('gs.plan', 'plan source %r != payload %r'
                                 % (plan.source_schema, source_schema))
        if nonempty and not plan.accept_nonempty:
            raise GoalStateError('gs.nonempty-without-migration',
                                 'plan must state accept_nonempty=True for a populated source')
        known = set(payload.get('arrays', {}) or {})
        unmapped = sorted(known - set(plan.fields))
        if unmapped:
            raise GoalStateError('gs.unmapped-field', ','.join(unmapped))
        width = slots or cls._source_width(payload) or len(plan.slot_map) or 8
        target = cls.empty(width, evidence_slots, batch, device=device, dtype=dtype)
        mapped, dropped = {}, {}
        for source, name in plan.fields.items():
            if name is None:
                dropped[source] = plan.drop_reasons[source]
                continue
            if source not in known:
                raise GoalStateError('gs.missing-field', source)
            target._t[name] = torch.as_tensor(payload['arrays'][source], device=device,
                                              dtype=cls._dtype(name))
            if name in plan.sentinels:
                source_value, target_value = plan.sentinels[name]
                target._t[name] = torch.where(target._t[name] == source_value,
                                              torch.full_like(target._t[name], target_value),
                                              target._t[name])
            mapped[source] = name
        positive = target._t['ids'][target._t['ids'] > 0]
        target.next_identity = int(positive.max()) + 1 if positive.numel() else 1
        written = 0
        for source_slot, target_slot in plan.slot_map:
            for source, name in mapped.items():
                target._t[name][0, int(target_slot)] = target._t[name][0, int(source_slot)]
            written += 1
        target.validate()
        return target, dict(migrated='explicit-plan', source_schema=source_schema,
                            mapped=mapped, dropped=dropped, slots_written=written,
                            nonempty_source=nonempty, reason=plan.reason)

    @staticmethod
    def _source_width(payload):
        arrays = payload.get('arrays') or {}
        for name in ('ids', 'goal_ids'):
            rows = arrays.get(name)
            if isinstance(rows, list) and rows and isinstance(rows[0], list):
                return len(rows[0])
        return 0

    @staticmethod
    def _is_nonempty(payload):
        arrays = payload.get('arrays') or {}
        for name in ('ids', 'workspace_seen', 'goal_ids'):
            for row in arrays.get(name, []) or []:
                values = row if isinstance(row, list) else [row]
                if any(isinstance(v, (int, float)) and float(v) > 0 for v in values):
                    return True
        return bool(payload.get('goal_names'))


def v6_workspace_plan(*, accept_nonempty=False, slot_map=()):
    """The only sanctioned v6 -> d06 mapping, with its losses named.

    v6 keeps goal_ids/goal_statuses per slot and a learned goal_sources vector.
    A learned source vector is not a source enum, so the enum column is dropped
    and stays at its zero value until a typed record supplies one; this is a
    data loss, not a reinterpretation.
    """
    return MigrationPlan(
        source_schema='treyrin.cognitive_workspace.v6',
        fields={'goal_ids': 'ids', 'goal_statuses': 'status', 'goal_sources': None},
        accept_nonempty=accept_nonempty,
        slot_map=slot_map,
        reason='typed goal rows adopted from the frozen v6 workspace slots',
        drop_reasons={'goal_sources': 'v6 stores a learned provenance vector, not a source enum'},
        sentinels={'ids': (-1, 0)})
