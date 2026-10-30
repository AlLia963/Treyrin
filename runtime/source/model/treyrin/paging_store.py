"""Variable-length paged experience store (internal transport pages).

One experience is one logical record: a stable identity, its total length, the
ordered list of pages it occupies with offsets and checksums, and its provenance.
A long experience occupies several pages; a short one occupies part of one page.
Nothing is truncated, summarised or re-labelled on the way in or out.

Format (treyrin-paged-store-v1)
-------------------------------
A store root contains::

    STORE.json                    schema + creation-time page/cache/shard scale
    shards/000000/data.bin        append-only page payload bytes
    shards/000000/pages.jsonl     append-only page index (page -> offset/bytes/units)
    shards/000000/manifest.json   sealed shard: digests + counts
    records/<xx>/<exp_id>.json    one immutable experience record (small)
    index/segment-000000.jsonl    append-only hash-chained index entries
    index/base.json               checkpoint: index rows + counters (no payloads)
    index/SNAPSHOT.json           cites base and folded segment count; atomic pointer

One page is page_units payload symbols (default 2048: 2048 characters of text,
2048 token ids, or 2048 bytes).  A page is stored as its encoded payload bytes,
addressed by the SHA-256 of those bytes, so identical pages are stored once.
PageRef keeps ordinal, symbol offset, symbol length, byte length and that
checksum.  Experience.payload_sha256 covers the whole payload;
Experience.record_sha256 covers every metadata field.

Scale knobs are independent: long-term storage is unbounded; the active set is
resident_capacity experiences; one computation reads batch_size (max batch_max)
experiences; cache_units bounds the hot page payload cache; shard_units bounds
one shard file.  2048 is the transport page, 8192 the default cache scale, 16384
the default shard scale (assembly list 000062).

Identity and lineage
--------------------
experience_id is derived (never random) from payload digest, payload kind,
source kind, source, parent identity/ordinal, known-at time, evidence class and
the prior revision id.  The same content admitted twice under the same
provenance is therefore one experience (idempotent write), while equal text from
different sources stays distinct.  Page layout is NOT part of the identity: the
same record re-paged with another page size keeps its id.

Revisions and retractions are index state, never record rewrites: a revision
supersedes its prior through a status entry and links by revision_of.  Eviction
removes payload from the hot cache and the active set only; the record stays
readable by its stable id (payload reloads it from the shards).

This module is pure standard-library Python.  It performs no model call, no
scoring of content and no GPU work: the caller supplies the score used by the
admission/replace policy, so the existing content scoring path still decides
what stays resident.  It is durability and addressing bookkeeping, not thinking.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

SCHEMA = 'treyrin-paged-experience-v1'
STORE_SCHEMA = 'treyrin-paged-store-v1'
BASE_SCHEMA = 'treyrin-paged-base-v1'
SNAPSHOT_SCHEMA = 'treyrin-paged-snapshot-v1'
SHARD_SCHEMA = 'treyrin-paged-shard-v1'
INDEX_ENTRY_SCHEMA = 'treyrin-paged-index-entry-v1'
ZERO_CHAIN = '0' * 64
NL = chr(10)

PAGE_UNITS = 2048
CACHE_UNITS = 8192
SHARD_UNITS = 16384
RESIDENT_CAPACITY = 8192
BATCH_SIZE = 4
BATCH_MAX = 8

PAYLOAD_KINDS = ('text', 'token_ids', 'bytes')
SOURCE_KINDS = ('file', 'user_input', 'model_output', 'internal_record',
                'tool_result', 'external_document', 'device_log', 'schedule')
PARENT_KINDS = ('document', 'session', 'task', 'goal', 'event', '')
STATUSES = ('active', 'superseded', 'retracted', 'completed')
TRANSITIONS = {'active': ('superseded', 'retracted', 'completed'),
               'superseded': ('retracted',), 'retracted': (), 'completed': ()}
# Mirrors tools/records.py EPISTEMIC: an internal hypothesis may never be filed
# as a user assertion, and model output may never become external evidence.
EPISTEMIC_SOURCES = {
    'user_assertion': ('user_input',),
    'external_source': ('file', 'external_document', 'tool_result', 'device_log', 'schedule'),
    'model_output': ('model_output',),
    'internal_hypothesis': ('internal_record',),
}
MENTAL_KINDS = ('hypothesis', 'belief', 'assistant_output', 'task', 'goal', 'action')

HEX64 = re.compile(r'^[0-9a-f]{64}$')
ID_RE = re.compile(r'^exp-[0-9a-f]{32}$')
_RECORD_FIELDS = ('schema', 'experience_id', 'payload_kind', 'total_units', 'page_units',
                  'pages', 'payload_sha256', 'source_kind', 'source', 'parent_kind',
                  'parent_id', 'parent_ordinal', 'known_at', 'recorded_at', 'sequence',
                  'status', 'epistemic', 'revision', 'revision_of', 'meta', 'record_sha256')


class PagingError(ValueError):
    """A store invariant was violated; the code names the invariant."""

    def __init__(self, code, message=''):
        super().__init__('%s: %s' % (code, message) if message else str(code))
        self.code = code


def _check(condition, code, message=''):
    if not condition:
        raise PagingError(code, message)


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':'))


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00', 'Z')


def _sha256_text(text):
    return sha256_bytes(text.encode('utf-8'))


def _int(value, code, name, minimum=0):
    _check(type(value) is int and value >= minimum, code,
           '%s must be an integer >= %d' % (name, minimum))
    return value


def _hex64(value, code, name):
    _check(isinstance(value, str) and bool(HEX64.match(value)), code,
           '%s must be a 64-character lowercase sha256 hex digest' % name)
    return value


def _token_ids(values):
    seq = tuple(values)
    for value in seq:
        if type(value) is not int or value < 0 or value > 0x7fffffff:
            raise PagingError('payload.token-id', 'token ids must be nonnegative int32 values')
    return seq


def encode_payload(payload, payload_kind):
    """Exact durable bytes of one whole payload."""
    if payload_kind == 'text':
        _check(isinstance(payload, str), 'payload.type', 'text payload must be str')
        return payload.encode('utf-8')
    if payload_kind == 'token_ids':
        values = _token_ids(payload)
        return struct.pack('<%di' % len(values), *values)
    if payload_kind == 'bytes':
        _check(isinstance(payload, (bytes, bytearray, memoryview)), 'payload.type',
               'byte payload must be bytes-like')
        return bytes(payload)
    raise PagingError('payload.kind', repr(payload_kind))


def decode_payload(data, payload_kind):
    if payload_kind == 'text':
        try:
            return data.decode('utf-8')
        except UnicodeDecodeError:
            raise PagingError('payload.decode', 'page bytes are not UTF-8 text') from None
    if payload_kind == 'token_ids':
        _check(len(data) % 4 == 0, 'payload.decode', 'token payload is not int32 aligned')
        return struct.unpack('<%di' % (len(data) // 4), data)
    return data


def payload_units(payload, payload_kind):
    if payload_kind == 'text':
        _check(isinstance(payload, str), 'payload.type', 'text payload must be str')
        return len(payload)
    if payload_kind == 'token_ids':
        return len(_token_ids(payload))
    return len(encode_payload(payload, 'bytes'))


def slice_encoded(payload, payload_kind, start, units):
    if payload_kind == 'text':
        return payload[start:start + units].encode('utf-8')
    if payload_kind == 'token_ids':
        return struct.pack('<%di' % units, *payload[start:start + units])
    return bytes(payload[start:start + units])


def experience_identity(*, payload_kind, payload_sha256, source_kind, source, parent_kind,
                        parent_id, parent_ordinal, known_at, epistemic, revision_of):
    body = canonical({'payload_kind': payload_kind, 'payload_sha256': payload_sha256,
                      'source_kind': source_kind, 'source': source,
                      'parent_kind': parent_kind, 'parent_id': parent_id,
                      'parent_ordinal': parent_ordinal, 'known_at': known_at,
                      'epistemic': epistemic, 'revision_of': revision_of})
    return 'exp-' + sha256_bytes(body.encode('utf-8'))[:32]


def record_digest(data):
    body = {key: value for key, value in data.items() if key != 'record_sha256'}
    return _sha256_text(canonical(body))


@dataclass(frozen=True)
class PageRef:
    """One page of one experience: where it sits, how long it is, its checksum."""

    ordinal: int
    offset: int
    units: int
    nbytes: int
    sha256: str

    def __post_init__(self):
        _int(self.ordinal, 'page.ordinal', 'page ordinal')
        _int(self.offset, 'page.offset', 'page offset')
        _int(self.units, 'page.units', 'page units', minimum=1)
        _int(self.nbytes, 'page.bytes', 'page byte length', minimum=1)
        _hex64(self.sha256, 'page.checksum', 'page checksum')

    def as_dict(self):
        return {'ordinal': self.ordinal, 'offset': self.offset, 'units': self.units,
                'nbytes': self.nbytes, 'sha256': self.sha256}

    @classmethod
    def from_dict(cls, data):
        _check(isinstance(data, Mapping) and set(data) == {'ordinal', 'offset', 'units',
                                                          'nbytes', 'sha256'},
               'page.keys', 'page reference fields differ')
        return cls(ordinal=data['ordinal'], offset=data['offset'], units=data['units'],
                   nbytes=data['nbytes'], sha256=data['sha256'])


@dataclass(frozen=True)
class Experience:
    """One immutable experience record.  Total length = sum of its page lengths."""

    schema: str
    experience_id: str
    payload_kind: str
    total_units: int
    page_units: int
    pages: tuple
    payload_sha256: str
    source_kind: str
    source: str
    parent_kind: str
    parent_id: str
    parent_ordinal: int
    known_at: str
    recorded_at: str
    sequence: int
    status: str
    epistemic: str
    revision: int
    revision_of: str
    meta: str
    record_sha256: str = ''

    def __post_init__(self):
        _check(self.schema == SCHEMA, 'record.schema', repr(self.schema))
        _check(isinstance(self.experience_id, str) and bool(ID_RE.match(self.experience_id)),
               'record.identity', 'stable experience id is required')
        _check(self.payload_kind in PAYLOAD_KINDS, 'record.payload-kind', repr(self.payload_kind))
        _int(self.total_units, 'record.length', 'total length')
        _int(self.page_units, 'record.page-units', 'page size', minimum=1)
        _check(isinstance(self.pages, tuple), 'record.pages', 'pages must be a tuple')
        for index, page in enumerate(self.pages):
            _check(isinstance(page, PageRef), 'record.pages', 'page references required')
            _check(page.ordinal == index, 'record.page-order',
                   'page ordinals must be 0..n-1 in order')
            _check(page.offset == sum(item.units for item in self.pages[:index]),
                   'record.page-offset', 'page offsets must be contiguous')
            expected = min(self.page_units, self.total_units - page.offset)
            _check(page.units == expected, 'record.page-length',
                   'page length differs from the declared total length')
            if index < len(self.pages) - 1:
                _check(page.units == self.page_units, 'record.page-length',
                       'only the last page may be shorter than the page size')
            _check(page.offset + page.units <= self.total_units, 'record.page-range',
                   'page exceeds the experience length')
        if self.total_units:
            _check(bool(self.pages), 'record.pages', 'a nonempty experience needs pages')
            _check(self.pages[-1].offset + self.pages[-1].units == self.total_units,
                   'record.page-coverage', 'pages must cover the whole payload')
        else:
            _check(not self.pages, 'record.pages', 'an empty experience has no pages')
        _hex64(self.payload_sha256, 'record.payload-digest', 'payload digest')
        _check(self.source_kind in SOURCE_KINDS, 'record.source-kind', repr(self.source_kind))
        _check(isinstance(self.source, str) and bool(self.source.strip()), 'record.source',
               'explicit source identity is required')
        _check(self.parent_kind in PARENT_KINDS, 'record.parent-kind', repr(self.parent_kind))
        _check(isinstance(self.parent_id, str), 'record.parent', 'parent id must be text')
        _check(bool(self.parent_id) == bool(self.parent_kind), 'record.parent',
               'parent identity and parent kind must be present together')
        _int(self.parent_ordinal, 'record.parent-ordinal', 'parent ordinal')
        _check(isinstance(self.meta, str), 'record.meta', 'meta must be canonical JSON text')
        try:
            parsed_meta = json.loads(self.meta)
        except json.JSONDecodeError:
            raise PagingError('record.meta', 'meta is not JSON') from None
        _check(isinstance(parsed_meta, dict), 'record.meta', 'meta must be a JSON object')
        _check(isinstance(self.known_at, str), 'record.time', 'known_at must be text')
        if not self.known_at:
            _check(parsed_meta.get('time_unknown') is True, 'record.time',
                   'empty known_at requires an explicit time_unknown marker')
        _check(isinstance(self.recorded_at, str) and bool(self.recorded_at),
               'record.time', 'recorded_at is required')
        _int(self.sequence, 'record.sequence', 'admission sequence', minimum=1)
        _check(self.status in STATUSES, 'record.status', repr(self.status))
        _check(self.epistemic == '' or self.epistemic in EPISTEMIC_SOURCES, 'record.epistemic',
               repr(self.epistemic))
        if self.epistemic:
            _check(self.source_kind in EPISTEMIC_SOURCES[self.epistemic],
                   'record.epistemic',
                   '%s may not be filed under source kind %s' % (self.epistemic, self.source_kind))
        _int(self.revision, 'record.revision', 'revision number', minimum=1)
        if self.revision_of:
            _check(isinstance(self.revision_of, str) and bool(ID_RE.match(self.revision_of)),
                   'record.revision', 'revision target must be a stable experience id')
            _check(self.revision >= 2, 'record.revision', 'a revision starts above revision 1')
        else:
            _check(self.revision == 1, 'record.revision', 'an unrevised record is revision 1')
        expected_identity = experience_identity(
            payload_kind=self.payload_kind, payload_sha256=self.payload_sha256,
            source_kind=self.source_kind, source=self.source, parent_kind=self.parent_kind,
            parent_id=self.parent_id, parent_ordinal=self.parent_ordinal,
            known_at=self.known_at, epistemic=self.epistemic,
            revision_of=self.revision_of)
        _check(expected_identity == self.experience_id, 'record.identity',
               'stable id does not match this experience identity')
        digest = record_digest(self.as_dict())
        if self.record_sha256:
            _check(self.record_sha256 == digest, 'record.digest',
                   'record metadata digest differs')
        else:
            object.__setattr__(self, 'record_sha256', digest)

    def metadata(self):
        return json.loads(self.meta)

    def as_dict(self):
        return {'schema': self.schema, 'experience_id': self.experience_id,
                'payload_kind': self.payload_kind, 'total_units': self.total_units,
                'page_units': self.page_units,
                'pages': [page.as_dict() for page in self.pages],
                'payload_sha256': self.payload_sha256, 'source_kind': self.source_kind,
                'source': self.source, 'parent_kind': self.parent_kind,
                'parent_id': self.parent_id, 'parent_ordinal': self.parent_ordinal,
                'known_at': self.known_at, 'recorded_at': self.recorded_at,
                'sequence': self.sequence, 'status': self.status,
                'epistemic': self.epistemic, 'revision': self.revision,
                'revision_of': self.revision_of, 'meta': self.meta,
                'record_sha256': self.record_sha256}

    @classmethod
    def from_dict(cls, data):
        _check(isinstance(data, Mapping) and set(data) == set(_RECORD_FIELDS),
               'record.keys', 'experience record fields differ')
        return cls(schema=data['schema'], experience_id=data['experience_id'],
                   payload_kind=data['payload_kind'], total_units=data['total_units'],
                   page_units=data['page_units'],
                   pages=tuple(PageRef.from_dict(page) for page in data['pages']),
                   payload_sha256=data['payload_sha256'], source_kind=data['source_kind'],
                   source=data['source'], parent_kind=data['parent_kind'],
                   parent_id=data['parent_id'], parent_ordinal=data['parent_ordinal'],
                   known_at=data['known_at'], recorded_at=data['recorded_at'],
                   sequence=data['sequence'], status=data['status'],
                   epistemic=data['epistemic'], revision=data['revision'],
                   revision_of=data['revision_of'], meta=data['meta'],
                   record_sha256=data['record_sha256'])

    def row(self):
        """Compact index row: locating metadata only, never payload bytes."""
        return {'id': self.experience_id, 'record_sha256': self.record_sha256,
                'payload_sha256': self.payload_sha256, 'payload_kind': self.payload_kind,
                'units': self.total_units, 'pages': len(self.pages),
                'page_units': self.page_units, 'sequence': self.sequence,
                'status': self.status, 'source_kind': self.source_kind,
                'source': self.source, 'parent_kind': self.parent_kind,
                'parent_id': self.parent_id, 'parent_ordinal': self.parent_ordinal,
                'known_at': self.known_at, 'epistemic': self.epistemic,
                'revision': self.revision, 'revision_of': self.revision_of}


@dataclass(frozen=True)
class StoreConfig:
    """Independent scale knobs.  Every field may be set on its own."""

    page_units: int = PAGE_UNITS
    cache_units: int = CACHE_UNITS
    shard_units: int = SHARD_UNITS
    resident_capacity: int = RESIDENT_CAPACITY
    batch_size: int = BATCH_SIZE
    batch_max: int = BATCH_MAX

    def __post_init__(self):
        _int(self.page_units, 'config.page', 'page size', minimum=1)
        _int(self.cache_units, 'config.cache', 'cache size', minimum=0)
        _int(self.shard_units, 'config.shard', 'shard size', minimum=1)
        _int(self.resident_capacity, 'config.resident', 'resident capacity', minimum=0)
        _int(self.batch_size, 'config.batch', 'batch size', minimum=1)
        _int(self.batch_max, 'config.batch', 'batch maximum', minimum=1)
        _check(self.batch_max <= BATCH_MAX, 'config.batch',
               'the bounded batch maximum is %d' % BATCH_MAX)
        _check(self.batch_size <= self.batch_max, 'config.batch',
               'batch size may not exceed the batch maximum')

    def as_dict(self):
        return {'page_units': self.page_units, 'cache_units': self.cache_units,
                'shard_units': self.shard_units,
                'resident_capacity': self.resident_capacity,
                'batch_size': self.batch_size, 'batch_max': self.batch_max}


class PageCache:
    """LRU page payload cache, bounded in payload units (default 8192 = 4 pages)."""

    def __init__(self, max_units):
        self.max_units = int(max_units)
        self._items = OrderedDict()
        self._units = {}
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.inserts = 0

    def get(self, page_id):
        data = self._items.get(page_id)
        if data is None:
            self.misses += 1
            return None
        self.hits += 1
        self._items.move_to_end(page_id)
        return data

    def put(self, page_id, data, units):
        if units > self.max_units:
            return False
        if page_id in self._items:
            self._items[page_id] = data
            self._units[page_id] = units
            self._items.move_to_end(page_id)
            return True
        while self._items and sum(self._units.values()) + units > self.max_units:
            old, _ = self._items.popitem(last=False)
            self._units.pop(old, None)
            self.evictions += 1
        self._items[page_id] = data
        self._units[page_id] = units
        self.inserts += 1
        return True

    def drop(self, page_id):
        if page_id in self._items:
            del self._items[page_id]
            self._units.pop(page_id, None)
            return True
        return False

    def has(self, page_id):
        return page_id in self._items

    def page_ids(self):
        return tuple(self._items)

    @property
    def used_units(self):
        return sum(self._units.values())

    def stats(self):
        return {'max_units': self.max_units, 'used_units': self.used_units,
                'pages': len(self._items), 'hits': self.hits, 'misses': self.misses,
                'evictions': self.evictions, 'inserts': self.inserts}


@dataclass(frozen=True)
class AdmissionDecision:
    """What the caller's admission/replace policy decided for one experience."""

    action: str
    victim: str = ''

    def __post_init__(self):
        _check(self.action in ('activate', 'replace', 'keep'), 'residence.action',
               repr(self.action))
        if self.action == 'replace':
            _check(bool(self.victim), 'residence.action', 'a replace needs a victim id')
        else:
            _check(not self.victim, 'residence.action', 'only replace carries a victim')

    def as_dict(self):
        return {'action': self.action, 'victim': self.victim}


class LowestScorePolicy:
    """Deterministic default: free seat, else replace the lowest-scored resident.

    The score is supplied by the caller (the existing content-scoring path); this
    policy only turns scores into a residence decision.  A tie keeps the resident
    set stable instead of churning on every admission.
    """

    def __call__(self, *, new_id, new_score, resident, scores, capacity):
        if capacity <= 0:
            return AdmissionDecision('keep')
        if len(resident) < capacity:
            return AdmissionDecision('activate')
        victim = min(resident, key=lambda rid: (scores.get(rid, 0.0), resident.index(rid)))
        if new_score > scores.get(victim, 0.0):
            return AdmissionDecision('replace', victim)
        return AdmissionDecision('keep')


@dataclass(frozen=True)
class AdmitResult:
    experience: Experience
    duplicate: bool
    pages_written: int
    pages_deduped: int
    decision: AdmissionDecision
    resident: bool

    def as_dict(self):
        return {'experience_id': self.experience.experience_id,
                'sequence': self.experience.sequence,
                'payload_kind': self.experience.payload_kind,
                'total_units': self.experience.total_units,
                'pages': len(self.experience.pages),
                'payload_sha256': self.experience.payload_sha256,
                'record_sha256': self.experience.record_sha256,
                'duplicate': self.duplicate, 'pages_written': self.pages_written,
                'pages_deduped': self.pages_deduped,
                'decision': self.decision.as_dict(), 'resident': self.resident}


@dataclass(frozen=True)
class BatchRead:
    """One bounded read batch: default 4 records, never more than 8."""

    batch_index: int
    experience_ids: tuple
    payloads: tuple
    evidence: tuple
    next_index: int
    remaining: int
    complete: bool
    read_at: str

    def as_dict(self, include_payloads=False):
        data = {'batch_index': self.batch_index,
                'experience_ids': list(self.experience_ids),
                'evidence': [dict(item) for item in self.evidence],
                'next_index': self.next_index, 'remaining': self.remaining,
                'complete': self.complete, 'read_at': self.read_at}
        if include_payloads:
            data['payloads'] = list(self.payloads)
        return data


class BatchCursor:
    """Consecutive bounded batches with retained read evidence and state."""

    def __init__(self, store, experience_ids, *, size=None):
        self.store = store
        self.ids = tuple(experience_ids)
        self.size = store.config.batch_size if size is None else size
        self.index = 0
        self.batches = 0
        self.evidence = ()
        self.units = 0
        self.complete = not self.ids

    def next(self):
        if self.complete:
            return None
        batch = self.store.read_batch(self.ids, start=self.index, size=self.size)
        self.index = batch.next_index
        self.batches += 1
        self.evidence = self.evidence + batch.evidence
        self.units += sum(item['units'] for item in batch.evidence)
        self.complete = batch.complete
        return batch

    def summary(self):
        return {'batch_size': self.size, 'batches': self.batches,
                'records_read': len(self.evidence), 'units_read': self.units,
                'next_index': self.index, 'complete': self.complete,
                'page_ids': [page for item in self.evidence for page in item['page_ids']]}


class PagedExperienceStore:
    """Growable experience library with paged storage and evictable residence."""

    def __init__(self, root, *, config=None, clock=None, policy=None, create=True,
                 repair=False):
        self.root = Path(root).resolve()
        self.config = StoreConfig() if config is None else config
        _check(isinstance(self.config, StoreConfig), 'config.type', 'StoreConfig required')
        self.clock = utc_now if clock is None else clock
        self.policy = LowestScorePolicy() if policy is None else policy
        self._lock = threading.RLock()
        self._cache = PageCache(self.config.cache_units)
        self._page_loc = {}
        self._shard_stats = {}
        self._shard = None
        self._index = {}
        self._order = []
        self._parents = {}
        self._resident = []
        self._scores = {}
        self._revised_by = {}
        self._sequence = 0
        self._index_seq = 0
        self._chain_head = ZERO_CHAIN
        self._counters = {'experiences': 0, 'pages': 0, 'units': 0, 'dedup_units': 0,
                          'reads': 0, 'read_records': 0, 'evictions': 0}
        self._writable = None
        self._generation = 0
        if create:
            self._initialize()
        else:
            _check((self.root / 'STORE.json').exists(), 'store.missing',
                   'no paged experience store at %s' % self.root)
        self._open(repair=repair)

    # -- layout ---------------------------------------------------------
    def _initialize(self):
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ('shards', 'records', 'index'):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        store_file = self.root / 'STORE.json'
        if not store_file.exists():
            payload = {'schema': STORE_SCHEMA, 'created_at': self.clock(),
                       'config': self.config.as_dict()}
            _atomic_write_text(store_file, json.dumps(payload, sort_keys=True, indent=2) + NL)

    def _shard_dirs(self):
        root = self.root / 'shards'
        if not root.exists():
            return []
        return sorted(path for path in root.iterdir() if path.is_dir())

    def _shard_dir(self, number):
        return self.root / 'shards' / ('%06d' % number)

    def _load_shards(self):
        for directory in self._shard_dirs():
            number = int(directory.name)
            data_path = directory / 'data.bin'
            file_bytes = data_path.stat().st_size if data_path.exists() else 0
            index_path = directory / 'pages.jsonl'
            page_bytes = 0
            pages = 0
            units = 0
            if index_path.exists():
                for line in index_path.read_text('utf-8').splitlines():
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        # A torn trailing line stays in place for audit; its page is
                        # simply not addressable and a record pointing at it fails
                        # loudly on read rather than being silently patched.
                        break
                    page_id = str(entry.get('page_id'))
                    _check(bool(HEX64.match(page_id)), 'shard.index',
                           'page address differs')
                    location = (number, int(entry['offset']), int(entry['bytes']),
                                int(entry['units']))
                    previous = self._page_loc.get(page_id)
                    if previous is not None:
                        _check(previous[2] == location[2] and previous[3] == location[3],
                               'shard.index', 'duplicate page address with another length')
                    else:
                        self._page_loc[page_id] = location
                    page_bytes = max(page_bytes, location[1] + location[2])
                    pages += 1
                    units += location[3]
            manifest_path = directory / 'manifest.json'
            sealed = manifest_path.exists()
            if sealed:
                metadata = json.loads(manifest_path.read_text('utf-8'))
                _check(metadata.get('schema') == SHARD_SCHEMA, 'shard.manifest',
                       'shard manifest schema differs')
                _check(int(metadata['data_bytes']) == file_bytes, 'shard.manifest',
                       'sealed shard byte count differs from its data file')
                _check(int(metadata['pages']) == pages, 'shard.manifest',
                       'sealed shard page count differs from its index')
            else:
                page_bytes = max(page_bytes, file_bytes)
            self._shard_stats[number] = {'pages': pages, 'units': units,
                                         'data_bytes': page_bytes, 'sealed': sealed}

    def _current_shard(self):
        if self._shard is not None:
            return self._shard
        numbers = sorted(self._shard_stats)
        if numbers and not self._shard_stats[numbers[-1]]['sealed']:
            number = numbers[-1]
        else:
            number = (numbers[-1] + 1) if numbers else 0
        directory = self._shard_dir(number)
        directory.mkdir(parents=True, exist_ok=True)
        data_path = directory / 'data.bin'
        index_path = directory / 'pages.jsonl'
        stats = self._shard_stats.get(number, {'pages': 0, 'units': 0, 'data_bytes': 0,
                                               'sealed': False})
        _check(not stats['sealed'], 'shard.sealed', 'a sealed shard is not writable')
        data_bytes = stats['data_bytes']
        if data_path.exists():
            data_bytes = max(data_bytes, data_path.stat().st_size)
        self._shard = {'number': number, 'dir': directory, 'data': data_path,
                       'index': index_path,
                       'data_handle': open(data_path, 'ab'),
                       'index_handle': open(index_path, 'a', encoding='utf-8', newline=NL),
                       'data_bytes': data_bytes, 'pages': stats['pages'],
                       'units': stats['units']}
        stats['data_bytes'] = data_bytes
        self._shard_stats[number] = stats
        self._shard['stats'] = stats
        return self._shard

    def _seal_shard(self):
        shard = self._shard
        if shard is None:
            return None
        shard['data_handle'].flush()
        os.fsync(shard['data_handle'].fileno())
        shard['index_handle'].flush()
        os.fsync(shard['index_handle'].fileno())
        shard['data_handle'].close()
        shard['index_handle'].close()
        manifest = {'schema': SHARD_SCHEMA, 'number': shard['number'],
                    'pages': shard['pages'], 'units': shard['units'],
                    'data_bytes': shard['data_bytes'],
                    'data_sha256': _sha256_file(shard['data']),
                    'index_sha256': _sha256_file(shard['index']),
                    'sealed_at': self.clock()}
        _atomic_write_text(shard['dir'] / 'manifest.json',
                           json.dumps(manifest, sort_keys=True, indent=2) + NL)
        stats = self._shard_stats[shard['number']]
        stats['sealed'] = True
        stats['data_bytes'] = shard['data_bytes']
        stats['pages'] = shard['pages']
        stats['units'] = shard['units']
        self._shard = None
        return manifest

    # -- index segments -------------------------------------------------
    def _segment_files(self):
        directory = self.root / 'index'
        if not directory.exists():
            return []
        return sorted(directory.glob('segment-*.jsonl'))

    def _next_segment_number(self):
        numbers = []
        for path in self._segment_files():
            try:
                numbers.append(int(path.stem.split('-')[-1]))
            except ValueError:
                continue
        return (max(numbers) + 1) if numbers else 0

    def _append_entry(self, op):
        with self._lock:
            if self._writable is None:
                number = self._next_segment_number()
                path = self.root / 'index' / ('segment-%06d.jsonl' % number)
                path.touch()
                self._writable = path
            body = {'schema': INDEX_ENTRY_SCHEMA, 'seq': self._index_seq + 1,
                    'op': op, 'at': self.clock(), 'prev_chain': self._chain_head}
            entry_sha = _sha256_text(canonical(body))
            body['entry_sha256'] = entry_sha
            body['chain_sha256'] = _sha256_text(self._chain_head + entry_sha)
            with self._writable.open('a', encoding='utf-8', newline=NL) as handle:
                handle.write(canonical(body) + NL)
                handle.flush()
                os.fsync(handle.fileno())
            self._index_seq += 1
            self._chain_head = body['chain_sha256']
            return body

    @staticmethod
    def _verify_entry(entry, *, expected_seq, prev_chain):
        _check(isinstance(entry, dict) and entry.get('schema') == INDEX_ENTRY_SCHEMA,
               'index.entry', 'index entry schema differs')
        _check(entry.get('seq') == expected_seq, 'index.sequence',
               'index sequence gap at %r' % (entry.get('seq'),))
        _check(entry.get('prev_chain') == prev_chain, 'index.chain',
               'index chain break at seq %r' % (entry.get('seq'),))
        body = {key: value for key, value in entry.items()
                if key not in ('entry_sha256', 'chain_sha256')}
        expected = _sha256_text(canonical(body))
        _check(entry.get('entry_sha256') == expected, 'index.digest',
               'index entry digest differs at seq %r' % (entry.get('seq'),))
        _check(entry.get('chain_sha256') == _sha256_text(prev_chain + expected),
               'index.chain', 'index chain digest differs at seq %r' % (entry.get('seq'),))
        return entry['chain_sha256']

    def _replay_segment(self, path):
        expected = self._index_seq + 1
        chain = self._chain_head
        for line in path.read_text('utf-8').splitlines():
            if not line:
                continue
            entry = json.loads(line)
            chain = self._verify_entry(entry, expected_seq=expected, prev_chain=chain)
            expected += 1
            self._apply_entry(entry['op'])
            self._index_seq = entry['seq']
            self._chain_head = chain

    def _apply_entry(self, op):
        kind = op.get('op')
        if kind == 'admit':
            self._apply_admit(op['row'])
        elif kind == 'status':
            self._apply_status(op['id'], op['status'], by=op.get('by', ''),
                               reason=op.get('reason', ''))
        elif kind == 'residence':
            self._apply_residence(op)
        else:
            raise PagingError('index.op', repr(kind))

    def _apply_admit(self, row):
        _check(isinstance(row, dict) and set(row) == set(
            ('id', 'record_sha256', 'payload_sha256', 'payload_kind', 'units', 'pages',
             'page_units', 'sequence', 'status', 'source_kind', 'source', 'parent_kind',
             'parent_id', 'parent_ordinal', 'known_at', 'epistemic', 'revision',
             'revision_of')), 'index.row', 'index row fields differ')
        _check(row['id'] not in self._index, 'index.duplicate',
               'experience %s was indexed twice' % row['id'])
        self._index[row['id']] = dict(row)
        self._order.append(row['id'])
        self._parents.setdefault(row['parent_id'], []).append(row['id'])
        self._sequence = max(self._sequence, int(row['sequence']))
        self._counters['experiences'] += 1
        self._counters['pages'] += int(row['pages'])
        self._counters['units'] += int(row['units'])
        if row['revision_of']:
            self._revised_by[row['revision_of']] = row['id']

    def _apply_status(self, experience_id, status, *, by='', reason=''):
        row = self._index.get(experience_id)
        _check(row is not None, 'index.missing', 'unknown experience %s' % experience_id)
        _check(status in STATUSES, 'index.status', repr(status))
        row['status'] = status
        if by:
            self._revised_by[experience_id] = by

    def _apply_residence(self, op):
        action = op.get('action')
        experience_id = op.get('id')
        if action == 'evict':
            if experience_id in self._resident:
                self._resident.remove(experience_id)
            self._counters['evictions'] += 1
        elif action == 'activate':
            _check(experience_id in self._index, 'index.missing',
                   'resident experience %s is not indexed' % experience_id)
            if experience_id not in self._resident:
                self._resident.append(experience_id)
            self._scores[experience_id] = float(op.get('score', 0.0))
        elif action == 'replace':
            victim = op.get('victim')
            _check(victim in self._resident, 'index.missing',
                   'replacement victim %s is not resident' % victim)
            self._resident.remove(victim)
            self._counters['evictions'] += 1
            if experience_id not in self._resident:
                self._resident.append(experience_id)
            self._scores[experience_id] = float(op.get('score', 0.0))
        else:
            raise PagingError('index.residence', repr(action))

    # -- open -----------------------------------------------------------
    def _open(self, *, repair):
        store_file = self.root / 'STORE.json'
        metadata = json.loads(store_file.read_text('utf-8'))
        _check(metadata.get('schema') == STORE_SCHEMA, 'store.schema',
               'store metadata schema differs')
        self._load_shards()
        folded = 0
        snapshot_path = self.root / 'index' / 'SNAPSHOT.json'
        if snapshot_path.exists():
            snapshot = json.loads(snapshot_path.read_text('utf-8'))
            _check(snapshot.get('schema') == SNAPSHOT_SCHEMA, 'index.snapshot',
                   'snapshot schema differs')
            base_path = self.root / 'index' / snapshot['base']
            _check(base_path.exists(), 'index.snapshot', 'snapshot base is missing')
            _check(_sha256_file(base_path) == snapshot['base_sha256'], 'index.snapshot',
                   'snapshot base digest differs')
            base = json.loads(base_path.read_text('utf-8'))
            _check(base.get('schema') == BASE_SCHEMA, 'index.base', 'base schema differs')
            self._load_base(base)
            folded = int(snapshot['segments_folded'])
            _check(folded <= len(self._segment_files()), 'index.snapshot',
                   'snapshot folds more segments than exist')
        for path in self._segment_files()[folded:]:
            self._replay_segment(path)
        self._verify_index(repair=repair)

    def _load_base(self, base):
        _check(base.get('schema') == BASE_SCHEMA, 'index.base', 'base schema differs')
        self._generation = int(base['generation'])
        self._index = {}
        self._order = []
        self._parents = {}
        self._revised_by = {str(key): str(value) for key, value in base['revised_by'].items()}
        self._resident = list(base['resident'])
        self._scores = {str(key): float(value) for key, value in base['scores'].items()}
        self._sequence = int(base['last_sequence'])
        self._index_seq = int(base['index_seq'])
        self._chain_head = base['chain_head']
        self._counters = {key: int(value) for key, value in base['counters'].items()}
        for row in base['rows']:
            _check(row['id'] not in self._index, 'index.base', 'duplicate base row')
            self._index[row['id']] = dict(row)
            self._order.append(row['id'])
        for parent, ids in base['parents'].items():
            self._parents[str(parent)] = list(ids)
        _hex64(self._chain_head, 'index.base', 'index chain head')

    def _verify_index(self, *, repair):
        disk = {path.stem for path in (self.root / 'records').glob('*/*.json')}
        if disk != set(self._index):
            if repair:
                self.rebuild_index()
                return
            raise PagingError('index.missing-row',
                              'index has %d rows but %d record files exist; '
                              'open with repair=True to rebuild the index'
                              % (len(self._index), len(disk)))
        _check(len(self._index) == self._counters['experiences'], 'index.counters',
               'experience counter differs from the index')
        _check(self._sequence == max([0] + [row['sequence']
                                           for row in self._index.values()]),
               'index.counters', 'sequence counter differs from the index')
        for rid in self._resident:
            _check(rid in self._index, 'index.resident', 'resident id is not indexed')
            _check(self._index[rid]['status'] == 'active', 'index.resident',
                   'a non-active experience is resident')

    # -- write path -----------------------------------------------------
    def _record_path(self, experience_id):
        return self.root / 'records' / experience_id[4:6] / (experience_id + '.json')

    def _append_page(self, page_id, data, units, payload_kind):
        shard = self._current_shard()
        offset = shard['data_bytes']
        shard['data_handle'].write(data)
        shard['data_handle'].flush()
        os.fsync(shard['data_handle'].fileno())
        entry = {'page_id': page_id, 'offset': offset, 'bytes': len(data),
                 'units': units, 'kind': payload_kind}
        shard['index_handle'].write(canonical(entry) + NL)
        shard['index_handle'].flush()
        os.fsync(shard['index_handle'].fileno())
        shard['data_bytes'] += len(data)
        shard['pages'] += 1
        shard['units'] += units
        stats = shard['stats']
        stats['data_bytes'] = shard['data_bytes']
        stats['pages'] = shard['pages']
        stats['units'] = shard['units']
        self._page_loc[page_id] = (shard['number'], offset, len(data), units)
        if shard['units'] >= self.config.shard_units:
            self._seal_shard()
        return self._page_loc[page_id]

    def _store_pages(self, payload, payload_kind, page_units):
        units_total = payload_units(payload, payload_kind)
        pages = []
        written = deduped = deduped_units = 0
        for ordinal, offset in enumerate(range(0, units_total, page_units)):
            units = min(page_units, units_total - offset)
            data = slice_encoded(payload, payload_kind, offset, units)
            page_id = sha256_bytes(data)
            if page_id in self._page_loc:
                deduped += 1
                deduped_units += units
            else:
                self._append_page(page_id, data, units, payload_kind)
                written += 1
            pages.append(PageRef(ordinal=ordinal, offset=offset, units=units,
                                 nbytes=len(data), sha256=page_id))
        self._counters['dedup_units'] += deduped_units
        return tuple(pages), written, deduped

    def _make_record(self, *, payload, payload_kind, payload_sha256, pages, page_units,
                     source_kind, source, parent_kind, parent_id, parent_ordinal,
                     known_at, epistemic, meta, status, revision_of, revision, sequence):
        identity = experience_identity(
            payload_kind=payload_kind, payload_sha256=payload_sha256,
            source_kind=source_kind, source=source, parent_kind=parent_kind,
            parent_id=parent_id, parent_ordinal=parent_ordinal, known_at=known_at,
            epistemic=epistemic, revision_of=revision_of)
        return Experience(schema=SCHEMA, experience_id=identity, payload_kind=payload_kind,
                          total_units=payload_units(payload, payload_kind),
                          page_units=page_units, pages=pages,
                          payload_sha256=payload_sha256, source_kind=source_kind,
                          source=source, parent_kind=parent_kind, parent_id=parent_id,
                          parent_ordinal=parent_ordinal, known_at=known_at,
                          recorded_at=self.clock(), sequence=sequence, status=status,
                          epistemic=epistemic, revision=revision,
                          revision_of=revision_of, meta=_meta_text(meta))

    def admit(self, payload, *, payload_kind, source_kind, source, known_at=None,
              parent_kind='', parent_id='', parent_ordinal=0, epistemic='', meta=None,
              status='active', revision_of='', page_units=None, score=0.0,
              policy=None):
        """Accept one experience into the library and decide its residence.

        Every accepted experience is persisted, whether or not it becomes
        resident.  The caller's score comes from the existing content-scoring
        path; the policy defaults to the store policy (lowest score replaced).
        """
        with self._lock:
            _check(payload_kind in PAYLOAD_KINDS, 'payload.kind', repr(payload_kind))
            _check(source_kind in SOURCE_KINDS, 'provenance.source-kind', repr(source_kind))
            _check(isinstance(source, str) and bool(source.strip()), 'provenance.source',
                   'explicit source identity is required')
            _check(parent_kind in PARENT_KINDS, 'provenance.parent-kind', repr(parent_kind))
            _check(isinstance(parent_id, str), 'provenance.parent', 'parent id must be text')
            _check(bool(parent_id) == bool(parent_kind), 'provenance.parent',
                   'parent identity and parent kind must be present together')
            _int(parent_ordinal, 'provenance.parent-ordinal', 'parent ordinal')
            _check(epistemic == '' or epistemic in EPISTEMIC_SOURCES,
                   'provenance.epistemic', repr(epistemic))
            if epistemic:
                _check(source_kind in EPISTEMIC_SOURCES[epistemic], 'provenance.epistemic',
                       '%s may not be filed under source kind %s'
                       % (epistemic, source_kind))
            _check(status in STATUSES, 'provenance.status', repr(status))
            _check(revision_of == '' or bool(ID_RE.match(str(revision_of))),
                   'provenance.revision', 'revision target must be a stable experience id')
            if revision_of:
                prior = self.record(revision_of)
                _check(self.status(revision_of) == 'active', 'provenance.revision',
                       'only an active experience may be revised')
                revision = prior.revision + 1
            else:
                revision = 1
            if known_at is None:
                known_at = self.clock()
            _check(isinstance(known_at, str), 'provenance.time', 'known_at must be text')
            units = payload_units(payload, payload_kind)
            payload_sha256 = sha256_bytes(encode_payload(payload, payload_kind))
            effective_page = self.config.page_units if page_units is None else page_units
            _int(effective_page, 'provenance.page', 'page size', minimum=1)
            sequence = self._sequence + 1
            pages, written, deduped = self._store_pages(payload, payload_kind,
                                                        effective_page)
            record = self._make_record(
                payload=payload, payload_kind=payload_kind,
                payload_sha256=payload_sha256, pages=pages, page_units=effective_page,
                source_kind=source_kind, source=source, parent_kind=parent_kind,
                parent_id=parent_id, parent_ordinal=parent_ordinal, known_at=known_at,
                epistemic=epistemic, meta=meta, status=status,
                revision_of=revision_of, revision=revision, sequence=sequence)
            _check(record.total_units == units, 'record.length',
                   'declared payload length differs from the stored pages')
            path = self._record_path(record.experience_id)
            if path.exists():
                existing = self.record(record.experience_id)
                _check(existing.payload_sha256 == payload_sha256, 'record.duplicate',
                       'the same identity already holds different content')
                return AdmitResult(existing, True, 0, 0, AdmissionDecision('keep'),
                                   existing.experience_id in self._resident)
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(path, json.dumps(record.as_dict(), sort_keys=True,
                                                indent=2) + NL)
            self._apply_admit(record.row())
            self._append_entry({'op': 'admit', 'row': record.row()})
            decision = AdmissionDecision('keep')
            resident = False
            if status == 'active':
                decision = self._residence_decision(record, score=score, policy=policy)
                if decision.action == 'activate':
                    self._apply_residence({'action': 'activate',
                                           'id': record.experience_id, 'score': score})
                    self._append_entry({'op': 'residence', 'action': 'activate',
                                        'id': record.experience_id, 'score': float(score)})
                    resident = True
                elif decision.action == 'replace':
                    self._apply_residence({'action': 'replace',
                                           'id': record.experience_id,
                                           'victim': decision.victim, 'score': score})
                    self._append_entry({'op': 'residence', 'action': 'replace',
                                        'id': record.experience_id,
                                        'victim': decision.victim, 'score': float(score)})
                    self._drop_pages(decision.victim)
                    resident = True
            return AdmitResult(record, False, written, deduped, decision, resident)

    def _residence_decision(self, record, *, score, policy):
        chooser = self.policy if policy is None else policy
        return chooser(new_id=record.experience_id, new_score=float(score),
                       resident=tuple(self._resident), scores=dict(self._scores),
                       capacity=self.config.resident_capacity)

    # -- convenience ingestions ----------------------------------------
    def admit_text(self, text, **kwargs):
        return self.admit(text, payload_kind='text', **kwargs)

    def admit_tokens(self, token_ids, **kwargs):
        return self.admit(tuple(token_ids), payload_kind='token_ids', **kwargs)

    def admit_bytes(self, data, **kwargs):
        return self.admit(bytes(data), payload_kind='bytes', **kwargs)

    def admit_input(self, text, *, source='user_input', known_at=None, **kwargs):
        _check(isinstance(text, str), 'payload.type', 'user input must be text')
        return self.admit_text(text, source_kind='user_input', source=source,
                               known_at=known_at, epistemic='user_assertion', **kwargs)

    def admit_output(self, text, *, source='model_generation', known_at=None, **kwargs):
        _check(isinstance(text, str), 'payload.type', 'model output must be text')
        return self.admit_text(text, source_kind='model_output', source=source,
                               known_at=known_at, epistemic='model_output', **kwargs)

    def admit_internal(self, text, *, identity, kind, known_at=None, status='active',
                       revises='', **kwargs):
        _check(isinstance(text, str) and bool(text), 'payload.type',
               'an internal record needs its full text')
        _check(isinstance(identity, str) and bool(identity), 'provenance.internal',
               'an internal record keeps its own identity')
        _check(kind in MENTAL_KINDS, 'provenance.internal-kind', repr(kind))
        _check(status in STATUSES, 'provenance.status', repr(status))
        payload_meta = dict(kwargs.pop('meta', None) or {})
        payload_meta.update({'record_id': identity, 'kind': kind,
                             'record_status': status})
        result = self.admit_text(text, source_kind='internal_record',
                                 source='mental-record:' + identity, known_at=known_at,
                                 epistemic='internal_hypothesis', revision_of=revises,
                                 meta=payload_meta, **kwargs)
        if status != 'active' and not result.duplicate:
            self.set_status(result.experience.experience_id, status,
                            reason='internal record status')
        return result

    def admit_file(self, path, *, source=None, known_at=None, record_mtime=False, **kwargs):
        path = Path(path)
        data = path.read_bytes()
        meta = dict(kwargs.pop('meta', None) or {})
        meta.setdefault('file_name', path.name)
        meta.setdefault('file_bytes', len(data))
        if record_mtime:
            meta.setdefault('fs_mtime', datetime.fromtimestamp(
                path.stat().st_mtime, timezone.utc).isoformat(timespec='microseconds'))
        parent_kind = kwargs.pop('parent_kind', 'document')
        return self.admit_bytes(data, source_kind='file',
                                source=source or path.name, known_at=known_at,
                                parent_kind=parent_kind, epistemic='external_source',
                                meta=meta, **kwargs)

    def admit_observation(self, token_ids, *, source='device-log', known_at=None, **kwargs):
        return self.admit_tokens(token_ids, source_kind='device_log', source=source,
                                 known_at=known_at,
                                 parent_kind=kwargs.pop('parent_kind', 'session'),
                                 **kwargs)

    # -- read path ------------------------------------------------------
    def has(self, experience_id):
        """True when this stable id already holds a committed record."""
        return self._record_path(experience_id).exists()

    def record(self, experience_id):
        path = self._record_path(experience_id)
        _check(path.exists(), 'record.missing', 'unknown experience %s' % experience_id)
        record = Experience.from_dict(json.loads(path.read_text('utf-8')))
        _check(record.experience_id == experience_id, 'record.identity',
               'record file holds another identity')
        return record

    def _read_page_bytes(self, page_id):
        location = self._page_loc.get(page_id)
        _check(location is not None, 'page.missing',
               'page %s is not in any shard' % page_id)
        number, offset, nbytes, _units = location
        data_path = self._shard_dir(number) / 'data.bin'
        _check(data_path.exists(), 'shard.missing', 'shard data file is missing')
        with open(data_path, 'rb') as handle:
            handle.seek(offset)
            data = handle.read(nbytes)
        _check(len(data) == nbytes, 'page.short', 'page payload is shorter than recorded')
        _check(sha256_bytes(data) == page_id, 'page.corrupt',
               'page payload digest differs from its address')
        return data

    def _page_bytes(self, page):
        data = self._cache.get(page.sha256)
        if data is None:
            data = self._read_page_bytes(page.sha256)
            self._cache.put(page.sha256, data, page.units)
        else:
            _check(sha256_bytes(data) == page.sha256, 'page.corrupt',
                   'cached page digest differs')
        return data

    def _load_chunks(self, record):
        chunks = []
        for page in record.pages:
            data = self._page_bytes(page)
            _check(len(data) == page.nbytes, 'page.short', 'page byte length differs')
            chunks.append(data)
        return chunks

    def payload(self, experience_id):
        """Full decoded payload, verified against the record checksum."""
        with self._lock:
            record = self.record(experience_id)
            chunks = self._load_chunks(record)
            encoded = b''.join(chunks)
            _check(sha256_bytes(encoded) == record.payload_sha256, 'page.payload-digest',
                   'reassembled payload digest differs')
            if record.payload_kind == 'text':
                value = ''.join(decode_payload(chunk, 'text') for chunk in chunks)
            elif record.payload_kind == 'token_ids':
                value = tuple(item for chunk in chunks
                              for item in decode_payload(chunk, 'token_ids'))
            else:
                value = encoded
            self._counters['reads'] += 1
            self._counters['read_records'] += 1
            return value

    def read_text(self, experience_id):
        record = self.record(experience_id)
        _check(record.payload_kind == 'text', 'read.kind',
               'experience %s is not text' % experience_id)
        return self.payload(experience_id)

    def read_tokens(self, experience_id):
        record = self.record(experience_id)
        _check(record.payload_kind == 'token_ids', 'read.kind',
               'experience %s is not a token stream' % experience_id)
        return self.payload(experience_id)

    def read_bytes(self, experience_id):
        record = self.record(experience_id)
        _check(record.payload_kind == 'bytes', 'read.kind',
               'experience %s is not stored as bytes' % experience_id)
        return self.payload(experience_id)

    def page_bytes(self, experience_id, ordinal):
        record = self.record(experience_id)
        _check(0 <= ordinal < len(record.pages), 'read.page', 'page ordinal out of range')
        page = record.pages[ordinal]
        data = self._page_bytes(page)
        _check(len(data) == page.nbytes, 'page.short', 'page byte length differs')
        return data

    def iter_pages(self, experience_id):
        record = self.record(experience_id)
        for page in record.pages:
            yield page, self._page_bytes(page)

    def read_slice(self, experience_id, start, units):
        """Read part of an experience, loading only the pages it touches."""
        _int(start, 'read.range', 'slice start')
        _int(units, 'read.range', 'slice length')
        with self._lock:
            record = self.record(experience_id)
            _check(start + units <= record.total_units, 'read.range',
                   'slice exceeds the experience length')
            end = start + units
            pieces = []
            for page in record.pages:
                page_start, page_end = page.offset, page.offset + page.units
                if page_end <= start or page_start >= end:
                    continue
                data = self._page_bytes(page)
                _check(sha256_bytes(data) == page.sha256, 'page.corrupt',
                       'page digest differs')
                local_start = max(start - page_start, 0)
                local_end = min(end - page_start, page.units)
                if record.payload_kind == 'text':
                    pieces.append(decode_payload(data, 'text')[local_start:local_end])
                elif record.payload_kind == 'token_ids':
                    pieces.append(decode_payload(data[local_start * 4:local_end * 4],
                                                 'token_ids'))
                else:
                    pieces.append(data[local_start:local_end])
            if record.payload_kind == 'text':
                return ''.join(pieces)
            if record.payload_kind == 'token_ids':
                return tuple(item for piece in pieces for item in piece)
            return b''.join(pieces)

    # -- residence ------------------------------------------------------
    def _drop_pages(self, experience_id):
        try:
            record = self.record(experience_id)
        except PagingError:
            return 0
        dropped = 0
        for page in record.pages:
            if self._cache.drop(page.sha256):
                dropped += 1
        return dropped

    def evict(self, experience_id, *, reason=''):
        """Leave the active set and the hot cache; the record remains readable."""
        with self._lock:
            record = self.record(experience_id)
            was_resident = experience_id in self._resident
            dropped = self._drop_pages(experience_id)
            if was_resident:
                self._apply_residence({'action': 'evict', 'id': experience_id})
                self._append_entry({'op': 'residence', 'action': 'evict',
                                    'id': experience_id, 'reason': reason})
            return {'experience_id': experience_id, 'was_resident': was_resident,
                    'pages_dropped': dropped, 'still_readable': True,
                    'record_units': record.total_units}

    def reload(self, experience_id, *, score=0.0, policy=None):
        """Load every page from the shards again and re-activate by policy."""
        with self._lock:
            record = self.record(experience_id)
            loaded = 0
            for page in record.pages:
                data = self._read_page_bytes(page.sha256)
                _check(sha256_bytes(data) == page.sha256, 'page.corrupt',
                       'page digest differs on reload')
                if self._cache.put(page.sha256, data, page.units):
                    loaded += 1
            decision = AdmissionDecision('keep')
            if self.status(experience_id) == 'active':
                if experience_id in self._resident:
                    decision = AdmissionDecision('activate')
                else:
                    decision = self._residence_decision(record, score=score,
                                                        policy=policy)
                    if decision.action == 'activate':
                        self._apply_residence({'action': 'activate', 'id': experience_id,
                                               'score': score})
                        self._append_entry({'op': 'residence', 'action': 'activate',
                                            'id': experience_id, 'score': float(score)})
                    elif decision.action == 'replace':
                        self._apply_residence({'action': 'replace', 'id': experience_id,
                                               'victim': decision.victim, 'score': score})
                        self._append_entry({'op': 'residence', 'action': 'replace',
                                            'id': experience_id,
                                            'victim': decision.victim,
                                            'score': float(score)})
                        self._drop_pages(decision.victim)
            return {'experience_id': experience_id, 'pages_loaded': loaded,
                    'pages': len(record.pages), 'decision': decision.as_dict(),
                    'resident': experience_id in self._resident}

    def active(self):
        return tuple(self._resident)

    def status(self, experience_id):
        row = self._index.get(experience_id)
        _check(row is not None, 'index.missing', 'unknown experience %s' % experience_id)
        return row['status']

    def set_status(self, experience_id, status, *, by='', reason=''):
        with self._lock:
            _check(status in STATUSES, 'status.value', repr(status))
            current = self.status(experience_id)
            _check(status == current or status in TRANSITIONS[current],
                   'status.transition', '%s may not become %s' % (current, status))
            if status != current:
                self._apply_status(experience_id, status, by=by, reason=reason)
                self._append_entry({'op': 'status', 'id': experience_id,
                                    'status': status, 'by': by, 'reason': reason})
                if status != 'active' and experience_id in self._resident:
                    self._apply_residence({'action': 'evict', 'id': experience_id})
                    self._append_entry({'op': 'residence', 'action': 'evict',
                                        'id': experience_id, 'reason': status})
                    self._drop_pages(experience_id)
            return self.status(experience_id)

    def revise(self, prior_id, payload, *, payload_kind=None, known_at=None, score=0.0,
               **kwargs):
        """Admit a revision and supersede the prior record (its text is kept)."""
        with self._lock:
            prior = self.record(prior_id)
            _check(self.status(prior_id) == 'active', 'revision.prior',
                   'only an active experience may be revised')
            fields = dict(source_kind=prior.source_kind, source=prior.source,
                          parent_kind=prior.parent_kind, parent_id=prior.parent_id,
                          parent_ordinal=prior.parent_ordinal,
                          epistemic=prior.epistemic, meta=prior.metadata(),
                          revision_of=prior_id)
            fields.update(kwargs)
            result = self.admit(payload,
                                payload_kind=prior.payload_kind if payload_kind is None
                                else payload_kind,
                                known_at=self.clock() if known_at is None else known_at,
                                score=score, **fields)
            if not result.duplicate:
                self.set_status(prior_id, 'superseded',
                                by=result.experience.experience_id, reason='revision')
            return result

    def retract(self, experience_id, *, reason='withdrawn'):
        return self.set_status(experience_id, 'retracted', reason=reason)

    def revised_by(self, experience_id):
        return self._revised_by.get(experience_id, '')

    def history(self, experience_id):
        chain = []
        current = experience_id
        while current:
            row = self._index[current]
            chain.append(dict(row))
            current = row['revision_of']
        chain.reverse()
        return {'lineage': chain, 'revised_by': self.revised_by(experience_id)}

    def neighbours(self, experience_id):
        """Previous/next within the same parent identity, else global order."""
        row = self._index.get(experience_id)
        _check(row is not None, 'index.missing', 'unknown experience %s' % experience_id)
        siblings = sorted((self._index[rid] for rid in self._parents.get(row['parent_id'], ())),
                          key=lambda item: (item['parent_ordinal'], item['sequence']))
        ids = [item['id'] for item in siblings]
        position = ids.index(experience_id)
        return {'parent_id': row['parent_id'],
                'previous': ids[position - 1] if position > 0 else '',
                'next': ids[position + 1] if position + 1 < len(ids) else '',
                'ordinal_in_parent': position}

    # -- bounded batches ------------------------------------------------
    def read_batch(self, experience_ids, *, start=0, size=None):
        """Read at most batch_max (8) experiences in one batch."""
        with self._lock:
            ids = tuple(experience_ids)
            _int(start, 'read.batch', 'batch start')
            _check(0 <= start <= len(ids), 'read.batch', 'batch start out of range')
            size = self.config.batch_size if size is None else size
            _int(size, 'read.batch-size', 'batch size', minimum=1)
            _check(size <= self.config.batch_max, 'read.batch-size',
                   'one batch reads at most %d experiences' % self.config.batch_max)
            chosen = ids[start:start + size]
            payloads = []
            evidence = []
            for experience_id in chosen:
                record = self.record(experience_id)
                payloads.append(self.payload(experience_id))
                row = self._index[experience_id]
                evidence.append({'experience_id': experience_id,
                                 'record_sha256': record.record_sha256,
                                 'payload_sha256': record.payload_sha256,
                                 'payload_kind': record.payload_kind,
                                 'units': record.total_units,
                                 'pages': len(record.pages),
                                 'page_ids': [page.sha256 for page in record.pages],
                                 'source_kind': record.source_kind,
                                 'source': record.source,
                                 'parent_kind': record.parent_kind,
                                 'parent_id': record.parent_id,
                                 'known_at': record.known_at,
                                 'status': row['status'],
                                 'revision': record.revision,
                                 'revision_of': record.revision_of})
            next_index = start + len(chosen)
            return BatchRead(batch_index=start // size, experience_ids=chosen,
                             payloads=tuple(payloads), evidence=tuple(evidence),
                             next_index=next_index, remaining=len(ids) - next_index,
                             complete=next_index >= len(ids), read_at=self.clock())

    def batch_cursor(self, experience_ids, *, size=None):
        return BatchCursor(self, experience_ids, size=size)

    # -- enumeration / statistics / maintenance -------------------------
    def records(self):
        for experience_id in self._order:
            yield self.record(experience_id)

    def index_rows(self):
        return tuple(dict(self._index[rid]) for rid in self._order)

    def shard_stats(self):
        return {number: dict(stats) for number, stats in sorted(self._shard_stats.items())}

    def stats(self):
        data_bytes = sum(stats['data_bytes'] for stats in self._shard_stats.values())
        segments = self._segment_files()
        return {'schema': STORE_SCHEMA, 'root': str(self.root),
                'config': self.config.as_dict(),
                'experiences': len(self._index), 'sequence': self._sequence,
                'pages': self._counters['pages'], 'units': self._counters['units'],
                'resident': list(self._resident),
                'resident_capacity': self.config.resident_capacity,
                'cache': self._cache.stats(),
                'shards': self.shard_stats(), 'shard_data_bytes': data_bytes,
                'segments': [path.name for path in segments],
                'segment_entries': self._index_seq, 'generation': self._generation,
                'counters': dict(self._counters)}

    def check(self, *, verify_payloads=True, deep=False):
        """Verify records, pages, checksums, index rows and lineage links."""
        errors = []
        records = {}
        for path in sorted((self.root / 'records').glob('*/*.json')):
            try:
                record = Experience.from_dict(json.loads(path.read_text('utf-8')))
                records[record.experience_id] = record
            except Exception as error:  # report the invariant, not a traceback
                errors.append({'code': 'record.unreadable', 'path': path.name,
                               'detail': str(error)})
        for experience_id, record in records.items():
            row = self._index.get(experience_id)
            if row is None:
                errors.append({'code': 'index.missing-row', 'id': experience_id})
                continue
            if row['record_sha256'] != record.record_sha256:
                errors.append({'code': 'index.digest', 'id': experience_id})
            try:
                if verify_payloads:
                    self.payload(experience_id)
                for page in record.pages:
                    if page.sha256 not in self._page_loc:
                        errors.append({'code': 'page.missing', 'id': experience_id,
                                       'page': page.ordinal})
                if record.revision_of and record.revision_of not in self._index:
                    errors.append({'code': 'revision.missing', 'id': experience_id})
            except PagingError as error:
                errors.append({'code': error.code, 'id': experience_id,
                               'detail': str(error)})
        for experience_id in set(self._index) - set(records):
            errors.append({'code': 'record.missing', 'id': experience_id})
        for experience_id in self._resident:
            if self._index.get(experience_id, {}).get('status') != 'active':
                errors.append({'code': 'residence.status', 'id': experience_id})
        if deep:
            for number in self._shard_stats:
                manifest_path = self._shard_dir(number) / 'manifest.json'
                if not manifest_path.exists():
                    continue
                manifest = json.loads(manifest_path.read_text('utf-8'))
                if _sha256_file(self._shard_dir(number) / 'data.bin') != manifest['data_sha256']:
                    errors.append({'code': 'shard.digest', 'shard': number})
                if _sha256_file(self._shard_dir(number) / 'pages.jsonl') != manifest['index_sha256']:
                    errors.append({'code': 'shard.index-digest', 'shard': number})
        try:
            expected_seq, chain = 1, ZERO_CHAIN
            for path in self._segment_files():
                for line in path.read_text('utf-8').splitlines():
                    if not line:
                        continue
                    chain = self._verify_entry(json.loads(line), expected_seq=expected_seq,
                                               prev_chain=chain)
                    expected_seq += 1
            if (expected_seq - 1) != self._index_seq or chain != self._chain_head:
                errors.append({'code': 'index.chain-head'})
        except PagingError as error:
            errors.append({'code': error.code, 'detail': str(error)})
        return {'ok': not errors, 'records': len(records),
                'pages': sum(len(record.pages) for record in records.values()),
                'units': sum(record.total_units for record in records.values()),
                'segments': len(self._segment_files()), 'errors': errors}

    def snapshot(self):
        """Checkpoint the index (no payload copy) and cite folded segments."""
        with self._lock:
            return self._write_snapshot(reason='snapshot')

    def _shard_descriptors(self):
        descriptors = []
        for number, stats in sorted(self._shard_stats.items()):
            manifest_path = self._shard_dir(number) / 'manifest.json'
            descriptor = {'number': number, 'sealed': bool(stats['sealed']),
                          'pages': stats['pages'], 'units': stats['units'],
                          'data_bytes': stats['data_bytes']}
            if manifest_path.exists():
                descriptor['manifest_sha256'] = _sha256_file(manifest_path)
            descriptors.append(descriptor)
        return descriptors

    def _write_snapshot(self, *, reason):
        self._generation += 1
        segments = self._segment_files()
        base = {'schema': BASE_SCHEMA, 'generation': self._generation,
                'last_sequence': self._sequence, 'index_seq': self._index_seq,
                'chain_head': self._chain_head,
                'counters': {key: int(value) for key, value in self._counters.items()},
                'rows': [self._index[rid] for rid in self._order],
                'parents': {key: list(ids) for key, ids in self._parents.items()},
                'resident': list(self._resident),
                'scores': {key: float(value) for key, value in self._scores.items()},
                'revised_by': dict(self._revised_by),
                'shards': self._shard_descriptors(),
                'reason': reason, 'written_at': self.clock()}
        base_path = self.root / 'index' / 'base.json'
        _atomic_write_text(base_path, json.dumps(base, sort_keys=True, indent=2) + NL)
        snapshot = {'schema': SNAPSHOT_SCHEMA, 'generation': self._generation,
                    'base': 'base.json', 'base_sha256': _sha256_file(base_path),
                    'segments_folded': len(segments),
                    'segments': [{'name': path.name, 'sha256': _sha256_file(path),
                                  'entries': sum(1 for line in
                                                 path.read_text('utf-8').splitlines()
                                                 if line)}
                                 for path in segments],
                    'counts': {'experiences': len(self._index),
                               'pages': self._counters['pages'],
                               'units': self._counters['units']},
                    'shards': self._shard_descriptors(),
                    'written_at': self.clock()}
        snapshot_path = self.root / 'index' / 'SNAPSHOT.json'
        _atomic_write_text(snapshot_path,
                           json.dumps(snapshot, sort_keys=True, indent=2) + NL)
        self._writable = None
        return {'generation': self._generation, 'segments_folded': len(segments),
                'base_bytes': base_path.stat().st_size,
                'snapshot_bytes': snapshot_path.stat().st_size,
                'counts': snapshot['counts']}

    def rebuild_index(self):
        """Repair: rebuild index rows and counters from the record files."""
        with self._lock:
            prior_status = {rid: row['status'] for rid, row in self._index.items()}
            prior_resident = list(self._resident)
            prior_scores = dict(self._scores)
            experiences = []
            for path in sorted((self.root / 'records').glob('*/*.json')):
                experiences.append(Experience.from_dict(json.loads(path.read_text('utf-8'))))
            experiences.sort(key=lambda record: record.sequence)
            self._index = {}
            self._order = []
            self._parents = {}
            self._resident = []
            self._scores = {}
            self._revised_by = {}
            self._sequence = 0
            self._counters = {'experiences': 0, 'pages': 0, 'units': 0,
                              'dedup_units': self._counters['dedup_units'],
                              'reads': self._counters['reads'],
                              'read_records': self._counters['read_records'],
                              'evictions': self._counters['evictions']}
            for record in experiences:
                row = record.row()
                row['status'] = prior_status.get(record.experience_id, record.status)
                self._apply_admit(row)
            for rid in prior_resident:
                if rid in self._index and self._index[rid]['status'] == 'active':
                    self._resident.append(rid)
                    self._scores[rid] = prior_scores.get(rid, 0.0)
            lost = [rid for rid in prior_status if rid not in self._index]
            self._write_snapshot(reason='rebuild')
            return {'records': len(experiences), 'resident': len(self._resident),
                    'status_history_kept': True, 'dropped_ids': len(lost)}

    def close(self):
        with self._lock:
            if self._shard is not None:
                self._shard['data_handle'].flush()
                os.fsync(self._shard['data_handle'].fileno())
                self._shard['index_handle'].flush()
                os.fsync(self._shard['index_handle'].fileno())
                self._shard['data_handle'].close()
                self._shard['index_handle'].close()
                self._shard = None

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()
        return False


def _atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex[:12])
    with open(temporary, 'w', encoding='utf-8', newline=NL) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return _sha256_text(text)


def _sha256_file(path, chunk=1 << 20):
    hasher = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _meta_text(meta):
    if meta is None:
        return '{}'
    if isinstance(meta, str):
        try:
            value = json.loads(meta)
        except json.JSONDecodeError:
            raise PagingError('provenance.meta', 'meta text is not JSON') from None
    else:
        _check(isinstance(meta, Mapping), 'provenance.meta', 'meta must be a mapping')
        value = dict(meta)
    _check(isinstance(value, dict), 'provenance.meta', 'meta must be a JSON object')
    for key in value:
        _check(isinstance(key, str), 'provenance.meta', 'meta keys must be text')
    try:
        return canonical(value)
    except (TypeError, ValueError) as error:
        raise PagingError('provenance.meta',
                          'meta is not JSON serialisable: %s' % error) from None


__all__ = ['PagingError', 'PageRef', 'Experience', 'StoreConfig', 'PageCache',
           'AdmissionDecision', 'LowestScorePolicy', 'AdmitResult', 'BatchRead',
           'BatchCursor', 'PagedExperienceStore', 'experience_identity',
           'encode_payload', 'decode_payload', 'payload_units', 'record_digest',
           'canonical', 'sha256_bytes', 'utc_now', 'SCHEMA', 'STORE_SCHEMA',
           'PAGE_UNITS', 'CACHE_UNITS', 'SHARD_UNITS', 'RESIDENT_CAPACITY',
           'BATCH_SIZE', 'BATCH_MAX', 'SOURCE_KINDS', 'PAYLOAD_KINDS', 'STATUSES']
