"""P3 (order 000062): material directory manifest, typed parsing, streaming
segmented ingestion, progress/checkpoint/resume and inspection states.

What this module is
-------------------
The intake side of a real delegation: a directory is listed with content
identity, each file is parsed by its type, its text is streamed into ordered
segments with exact source locations (byte/char/line ranges), and the session
keeps a checkpoint so an interrupted ingestion resumes at the exact next
segment instead of starting over.

States are the ones the duty requires, plus two explicit ones that must not be
hidden inside them:

* unread       - listed, content identity known, no segment written yet.
* read         - every segment written and the typed parse succeeded.
* parse_failed - the typed parse (or UTF-8 decode) failed; the error keeps the
                 line/column/byte location.  Segments written before the failure
                 stay for audit; the entry blocks "inspect all before start".
* changed      - the file's sha256 differs from the ingested/listed one.  A
                 previously read entry goes back to blocking until re-ingested.
* unsupported  - no parser for this suffix (e.g. a binary blob).  Explicit, not
                 silently counted as text.
* ignored      - the operator explicitly waived one entry with a stated reason.
                 The waiver is journaled, never inferred.

Parsers are the standard ones already used across this source tree: json /
json.loads per JSONL line (as in the audit scripts) and the csv module.  No new
third-party parser is introduced, and no file content is rewritten.

This module never loads a model and never touches CUDA.
"""
from __future__ import annotations

import codecs
import csv
import hashlib
import io
import json
import os
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_MANIFEST = 'p3-material-manifest-v1'
SCHEMA_SEGMENT = 'p3-material-segment-v1'
SCHEMA_PROGRESS = 'p3-material-progress-v1'
SCHEMA_REPORT = 'p3-material-inspection-v1'
SCHEMA_EVENT = 'p3-material-event-v1'

SUFFIX_KINDS = {'.md': 'markdown', '.markdown': 'markdown', '.txt': 'text',
                '.log': 'log', '.json': 'json', '.jsonl': 'jsonl',
                '.ndjson': 'jsonl', '.csv': 'csv', '.tsv': 'tsv'}
LINE_KINDS = ('text', 'log', 'markdown', 'jsonl', 'csv', 'tsv')
STATES = ('unread', 'read', 'parse_failed', 'changed', 'unsupported')
BLOCKING_STATES = ('unread', 'parse_failed', 'changed', 'unsupported')


class MaterialError(ValueError):
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


def file_sha256(path, *, chunk=1 << 20):
    hasher = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def entry_identity(rel_path):
    return 'mat-' + hashlib.sha256(rel_path.encode('utf-8')).hexdigest()[:16]


def kind_of(path):
    return SUFFIX_KINDS.get(Path(path).suffix.lower(), 'unsupported')


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MaterialEntry:
    entry_id: str
    rel_path: str
    kind: str
    size: int
    mtime_ns: int
    sha256: str
    state: str = 'unread'
    segments: int = 0
    records: int = 0
    parser: str | None = None
    error_code: str | None = None
    error: str | None = None
    error_detail: dict | None = None
    attempt: int = 0
    ignored: bool = False
    ignored_reason: str | None = None

    def __post_init__(self):
        if self.state not in STATES:
            raise MaterialError('mi.state', repr(self.state))
        if self.kind != 'unsupported' and self.state == 'unsupported':
            raise MaterialError('mi.state', 'unsupported state needs unsupported kind')
        if self.kind == 'unsupported' and self.state not in ('unsupported', 'changed'):
            raise MaterialError('mi.state', 'unsupported kind stays unsupported')

    def blocking(self):
        return self.state in BLOCKING_STATES and not self.ignored

    def to_payload(self):
        return dict(entry_id=self.entry_id, rel_path=self.rel_path, kind=self.kind,
                    size=int(self.size), mtime_ns=int(self.mtime_ns),
                    sha256=self.sha256, state=self.state, segments=int(self.segments),
                    records=int(self.records), parser=self.parser,
                    error_code=self.error_code, error=self.error,
                    error_detail=self.error_detail, attempt=int(self.attempt),
                    ignored=bool(self.ignored), ignored_reason=self.ignored_reason)

    @classmethod
    def from_payload(cls, payload):
        return cls(entry_id=payload['entry_id'], rel_path=payload['rel_path'],
                   kind=payload['kind'], size=int(payload['size']),
                   mtime_ns=int(payload['mtime_ns']), sha256=payload['sha256'],
                   state=payload.get('state', 'unread'),
                   segments=int(payload.get('segments', 0)),
                   records=int(payload.get('records', 0)),
                   parser=payload.get('parser'), error_code=payload.get('error_code'),
                   error=payload.get('error'), error_detail=payload.get('error_detail'),
                   attempt=int(payload.get('attempt', 0)),
                   ignored=bool(payload.get('ignored', False)),
                   ignored_reason=payload.get('ignored_reason'))


@dataclass(frozen=True)
class MaterialManifest:
    root: str
    entries: tuple
    generated_at: str = ''
    schema: str = SCHEMA_MANIFEST
    version: int = 1

    def entry(self, entry_id):
        for item in self.entries:
            if item.entry_id == entry_id:
                return item
        raise MaterialError('mi.entry-unknown', str(entry_id))

    def by_path(self, rel_path):
        for item in self.entries:
            if item.rel_path == rel_path:
                return item
        raise MaterialError('mi.entry-unknown', str(rel_path))

    def with_entries(self, entries):
        return replace(self, entries=tuple(entries))

    def replace_entry(self, entry):
        return self.with_entries([entry if item.entry_id == entry.entry_id else item
                                  for item in self.entries])

    def counts(self):
        out = {state: 0 for state in STATES}
        for item in self.entries:
            out[item.state] += 1
        out['ignored'] = sum(1 for item in self.entries if item.ignored)
        out['total'] = len(self.entries)
        return out

    def digest(self):
        body = dict(schema=self.schema, version=self.version, root=self.root,
                    entries=[item.to_payload() for item in self.entries])
        return hashlib.sha256(canonical(body).encode('utf-8')).hexdigest()

    def as_dict(self):
        return dict(schema=self.schema, version=self.version, root=self.root,
                    generated_at=self.generated_at,
                    entries=[item.to_payload() for item in self.entries],
                    counts=self.counts(), digest=self.digest())

    def save(self, path):
        _atomic_write(path, canonical(self.as_dict()) + '\n')
        return self.digest()

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        if payload.get('schema') != SCHEMA_MANIFEST:
            raise MaterialError('mi.manifest-schema', str(path))
        return cls(root=payload['root'],
                   entries=tuple(MaterialEntry.from_payload(x)
                                 for x in payload['entries']),
                   generated_at=payload.get('generated_at', ''),
                   version=int(payload.get('version', 1)))


def scan_directory(root, *, ignore_names=('__pycache__', '.git', 'node_modules'),
                   ignore_suffixes=('.pyc',), clock=None):
    """List the directory with content identity; nothing is read beyond hashing."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise MaterialError('mi.not-directory', str(root))
    stamps = (clock or _utcnow)()
    entries = []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in ignore_names for part in relative.parts):
            continue
        if path.suffix.lower() in ignore_suffixes:
            continue
        rel_path = relative.as_posix()
        kind = kind_of(path)
        stat = path.stat()
        entries.append(MaterialEntry(
            entry_id=entry_identity(rel_path), rel_path=rel_path, kind=kind,
            size=int(stat.st_size), mtime_ns=int(stat.st_mtime_ns),
            sha256=file_sha256(path),
            state='unsupported' if kind == 'unsupported' else 'unread',
            error_code='mi.unsupported-type' if kind == 'unsupported' else None,
            error=('no parser for ' + path.suffix.lower()) if kind == 'unsupported' else None))
    return MaterialManifest(root=str(root), entries=tuple(entries),
                            generated_at=stamps)


# ---------------------------------------------------------------------------
# segments
# ---------------------------------------------------------------------------
class SegmentStore:
    """Append-only ordered segments with verified text and exact locations."""

    def __init__(self, path, *, sync=True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sync = bool(sync)
        self._entries = {}
        self._by_entry = {}
        self._handle = None
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        with open(self.path, 'rb') as handle:
            for position, raw_line in enumerate(handle):
                if not raw_line.endswith(b'\n'):
                    raise MaterialError('mi.segment-torn', 'line %d' % position)
                entry = json.loads(raw_line.decode('ascii'))
                if entry.get('schema') != SCHEMA_SEGMENT:
                    raise MaterialError('mi.segment-schema', 'line %d' % position)
                if _sha_text(entry['text']) != entry['sha256']:
                    raise MaterialError('mi.segment-digest', entry.get('segment_id'))
                segment = dict(entry)
                segment.pop('schema', None)
                self._entries[segment['segment_id']] = segment
                self._by_entry.setdefault((segment['entry_id'], segment['attempt']),
                                          []).append(segment)

    def append(self, segment):
        for name in ('segment_id', 'entry_id', 'attempt', 'index', 'char_start',
                     'char_end', 'byte_start', 'byte_end', 'line_start', 'line_end',
                     'text'):
            if name not in segment:
                raise MaterialError('mi.segment-field', name)
        if _sha_text(segment['text']) != segment['sha256']:
            raise MaterialError('mi.segment-digest', segment['segment_id'])
        if segment['segment_id'] in self._entries:
            raise MaterialError('mi.segment-duplicate', segment['segment_id'])
        if self._handle is None:
            self._handle = open(self.path, 'ab')
        entry = dict(schema=SCHEMA_SEGMENT, **segment)
        line = canonical(entry).encode('ascii') + b'\n'
        self._handle.write(line)
        self._handle.flush()
        if self.sync:
            os.fsync(self._handle.fileno())
        self._entries[segment['segment_id']] = segment
        self._by_entry.setdefault((segment['entry_id'], segment['attempt']),
                                  []).append(segment)

    def segments(self, entry_id, attempt=None):
        if attempt is None:
            attempt = self.latest_attempt(entry_id)
        return list(self._by_entry.get((entry_id, attempt), ()))

    def latest_attempt(self, entry_id):
        found = [key[1] for key in self._by_entry if key[0] == entry_id]
        return max(found) if found else 0

    def read(self, segment_id):
        segment = self._entries.get(segment_id)
        if segment is None:
            raise MaterialError('mi.segment-missing', str(segment_id))
        if _sha_text(segment['text']) != segment['sha256']:
            raise MaterialError('mi.segment-digest', str(segment_id))
        return segment

    def text(self, entry_id, attempt=None):
        return ''.join(segment['text'] for segment in self.segments(entry_id, attempt))

    def locate(self, segment_id):
        segment = self.read(segment_id)
        return dict(segment_id=segment_id, entry_id=segment['entry_id'],
                    attempt=segment['attempt'], source_ref=segment['source_ref'],
                    char_start=segment['char_start'], char_end=segment['char_end'],
                    byte_start=segment['byte_start'], byte_end=segment['byte_end'],
                    line_start=segment['line_start'], line_end=segment['line_end'],
                    chars=len(segment['text']))

    def verify(self):
        entries = 0
        for segment in self._entries.values():
            if _sha_text(segment['text']) != segment['sha256']:
                raise MaterialError('mi.segment-digest', segment['segment_id'])
            entries += 1
        return dict(ok=True, segments=entries, bytes=self.path.stat().st_size
                    if self.path.exists() else 0)

    def close(self):
        if self._handle is not None:
            try:
                self._handle.close()
            finally:
                self._handle = None

    def __len__(self):
        return len(self._entries)


# ---------------------------------------------------------------------------
# streaming segments
# ---------------------------------------------------------------------------
def segment_cut(buffer, target, line_mode):
    if len(buffer) <= target:
        return len(buffer)
    if line_mode:
        position = buffer.rfind('\n', 0, target)
        if position >= 0:
            return position + 1
    return target


def iter_text_segments(path, *, rel_path, start_byte=0, start_char=0, start_line=1,
                       start_index=0, chunk_bytes=4096, segment_chars=192,
                       line_mode=True):
    """Stream one UTF-8 file into ordered segments with exact locations.

    Byte offsets are exact: the buffer's start is (bytes decoded so far minus
    the incremental decoder's pending bytes) minus the buffer's own UTF-8 size.
    A checkpoint taken between segments can therefore seek back to byte_offset
    and continue without re-emitting text.  A line larger than segment_chars is
    cut at the character target; the line numbers still point at the real line.
    """
    if segment_chars < 8:
        raise MaterialError('mi.segment-size', segment_chars)
    if chunk_bytes < 1:
        raise MaterialError('mi.chunk-size', chunk_bytes)
    decoder = codecs.getincrementaldecoder('utf-8')('strict')
    handle = open(path, 'rb')
    try:
        handle.seek(int(start_byte))
        raw_read = int(start_byte)
        decoded_end = int(start_byte)
        buffer = ''
        chars = int(start_char)
        line = int(start_line)
        index = int(start_index)
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            chunk_start = raw_read
            try:
                piece = decoder.decode(chunk)
            except UnicodeDecodeError as exc:
                raise MaterialError('mi.decode', str(exc),
                                    byte=chunk_start + exc.start)
            raw_read += len(chunk)
            decoded_end = raw_read - len(decoder.getstate()[0])
            buffer += piece
            while len(buffer) > segment_chars:
                cut = segment_cut(buffer, segment_chars, line_mode)
                text = buffer[:cut]
                bytes_start = decoded_end - len(buffer.encode('utf-8'))
                yield _make_segment(rel_path, index, chars, line, text,
                                    bytes_start, buffer)
                chars += len(text)
                line += text.count('\n')
                index += 1
                buffer = buffer[cut:]
        try:
            tail = decoder.decode(b'', final=True)
        except UnicodeDecodeError as exc:
            raise MaterialError('mi.decode', str(exc), byte=raw_read)
        decoded_end = raw_read
        buffer += tail
        while len(buffer) > segment_chars:
            cut = segment_cut(buffer, segment_chars, line_mode)
            text = buffer[:cut]
            bytes_start = decoded_end - len(buffer.encode('utf-8'))
            yield _make_segment(rel_path, index, chars, line, text, bytes_start, buffer)
            chars += len(text)
            line += text.count('\n')
            index += 1
            buffer = buffer[cut:]
        if buffer:
            bytes_start = decoded_end - len(buffer.encode('utf-8'))
            yield _make_segment(rel_path, index, chars, line, buffer, bytes_start, buffer)
    finally:
        handle.close()


def _make_segment(rel_path, index, chars, line, text, bytes_start, buffer):
    bytes_end = bytes_start + len(text.encode('utf-8'))
    line_end = line + text.count('\n') - (1 if text.endswith('\n') else 0)
    next_line = line + text.count('\n')
    return dict(index=index, char_start=chars, char_end=chars + len(text),
                byte_start=bytes_start, byte_end=bytes_end,
                line_start=line, line_end=line_end, next_line=next_line,
                text=text, chars=len(text), sha256=_sha_text(text),
                source_ref='%s:L%d-L%d' % (rel_path, line, line_end))


# ---------------------------------------------------------------------------
# typed parsers
# ---------------------------------------------------------------------------
def validate_document(kind, text, *, delimiter=','):
    """Return (records, detail, error) for one fully streamed document."""
    if kind == 'json':
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            return 0, None, dict(code='mi.json-parse', message=exc.msg,
                                 line=exc.lineno, column=exc.colno, char=exc.pos)
        return 1, dict(top_level=type(value).__name__, chars=len(text),
                       parser='json'), None
    if kind == 'jsonl':
        count = 0
        offset = 0
        number = 0
        for number, line_text in enumerate(text.splitlines(keepends=True), 1):
            stripped = line_text.strip()
            if stripped:
                try:
                    json.loads(stripped)
                except json.JSONDecodeError as exc:
                    return count, None, dict(code='mi.jsonl-parse',
                                             message=exc.msg, line=number,
                                             column=exc.colno,
                                             char=offset + exc.pos)
                count += 1
            offset += len(line_text)
        return count, dict(records=count, lines=number, parser='jsonl'), None
    if kind in ('csv', 'tsv'):
        try:
            rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
        except csv.Error as exc:
            return 0, None, dict(code='mi.csv-parse', message=str(exc),
                                 line=getattr(exc, 'line_num', None))
        return len(rows), dict(rows=len(rows), header=rows[0] if rows else [],
                               delimiter=delimiter, parser=kind), None
    if kind in ('text', 'log', 'markdown'):
        lines = text.splitlines()
        return len(lines), dict(lines=len(lines), parser=kind), None
    return 0, None, dict(code='mi.unsupported-type', message=kind)


def inspection_precondition(report, *, kind='materials_inspected'):
    """Turn an inspection report into the task precondition outcome."""
    blocking = {name: value for name, value in report.get('blocking', {}).items()
                if value}
    status = 'satisfied' if report.get('ok') else 'pending'
    return dict(kind=kind, status=status,
                summary=('%d read / %d blocking' % (report.get('counts', {}).get('read', 0),
                                                    sum(len(v) for v in blocking.values()))),
                detail=dict(digest=report.get('digest'), blocking=blocking,
                            counts=report.get('counts'), at=report.get('at')))


# ---------------------------------------------------------------------------
# ingestion session
# ---------------------------------------------------------------------------
class IntakeSession:
    """Directory intake with durable segments, progress and resume."""

    def __init__(self, root, workdir, *, segment_chars=192, chunk_bytes=4096,
                 checkpoint_every_segments=4, clock=None, sync=True):
        self.root = Path(root).resolve()
        self.directory = Path(workdir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.segment_chars = int(segment_chars)
        self.chunk_bytes = int(chunk_bytes)
        self.checkpoint_every_segments = max(1, int(checkpoint_every_segments))
        self.clock = clock or _utcnow
        self.sync = bool(sync)
        self.segment_store = SegmentStore(self.directory / 'segments.jsonl',
                                          sync=sync)
        manifest_path = self.directory / 'manifest.json'
        if manifest_path.exists():
            loaded = MaterialManifest.load(manifest_path)
            if Path(loaded.root) == self.root:
                self.manifest = loaded
            else:
                self.manifest = scan_directory(self.root, clock=self.clock)
        else:
            self.manifest = scan_directory(self.root, clock=self.clock)
        self.progress = self._load_progress()
        self.events = []

    # -- helpers ---------------------------------------------------------
    def now(self):
        value = self.clock()
        return value if isinstance(value, str) else value.isoformat(timespec='milliseconds')

    def _load_progress(self):
        path = self.directory / 'progress.json'
        if path.exists():
            payload = json.loads(path.read_text(encoding='utf-8'))
            if payload.get('schema') != SCHEMA_PROGRESS:
                raise MaterialError('mi.progress-schema', str(path))
            return payload
        return dict(schema=SCHEMA_PROGRESS, version=1, root=str(self.root),
                    manifest_digest='', completed=[], current=None, paused={},
                    segments_total=0, updated_at='')

    def _save_progress(self):
        self.progress['manifest_digest'] = self.manifest.digest()
        self.progress['segments_total'] = len(self.segment_store)
        self.progress['updated_at'] = self.now()
        _atomic_write(self.directory / 'progress.json',
                      canonical(self.progress) + '\n')

    def _event(self, kind, **payload):
        entry = dict(schema=SCHEMA_EVENT, version=1, at=self.now(), kind=kind,
                     **payload)
        with open(self.directory / 'intake-events.jsonl', 'ab') as handle:
            handle.write(canonical(entry).encode('ascii') + b'\n')
            handle.flush()
            if self.sync:
                os.fsync(handle.fileno())
        self.events.append(entry)
        return entry

    def save_manifest(self):
        digest = self.manifest.save(self.directory / 'manifest.json')
        self._save_progress()
        return digest

    def checkpoint(self):
        self.save_manifest()
        return dict(manifest_digest=self.manifest.digest(),
                    segments=len(self.segment_store),
                    current=self.progress.get('current'),
                    path=str(self.directory / 'progress.json'))

    def close(self):
        self.segment_store.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- scanning --------------------------------------------------------
    def scan(self):
        fresh = scan_directory(self.root, clock=self.clock)
        previous = {entry.entry_id: entry for entry in self.manifest.entries}
        merged = []
        for entry in fresh.entries:
            prev = previous.get(entry.entry_id)
            if prev is None:
                merged.append(entry)
                continue
            same = prev.sha256 == entry.sha256
            if same:
                merged.append(replace(
                    entry, state=prev.state, segments=prev.segments,
                    records=prev.records, parser=prev.parser,
                    error_code=prev.error_code, error=prev.error,
                    error_detail=prev.error_detail, attempt=prev.attempt,
                    ignored=prev.ignored, ignored_reason=prev.ignored_reason))
                continue
            if entry.kind == 'unsupported':
                state = 'unsupported'
                code = 'mi.unsupported-type'
                message = 'no parser for ' + Path(entry.rel_path).suffix.lower()
            elif prev.state in ('read', 'parse_failed', 'changed'):
                state = 'changed'
                code = None
                message = None
            else:
                state = 'unread'
                code = None
                message = None
            merged.append(replace(entry, state=state, segments=0, records=0,
                                  parser=None, error_code=code, error=message,
                                  error_detail=None, attempt=prev.attempt,
                                  ignored=prev.ignored,
                                  ignored_reason=prev.ignored_reason))
        current_ids = {entry.entry_id for entry in fresh.entries}
        removed = [entry.rel_path for entry in self.manifest.entries
                   if entry.entry_id not in current_ids]
        self.manifest = self.manifest.with_entries(merged)
        self.manifest = replace(self.manifest, generated_at=fresh.generated_at)
        paused = self.progress.get('paused') or {}
        valid = {entry.entry_id: entry.sha256 for entry in merged}
        for entry_id in list(paused):
            if (entry_id not in valid
                    or paused[entry_id].get('file_sha256') != valid[entry_id]):
                paused.pop(entry_id, None)
        self.save_manifest()
        return dict(manifest=self.manifest, removed=removed)

    # -- resume point ----------------------------------------------------
    def _resume_point(self, entry):
        paused = self.progress.get('paused') or {}
        current = paused.get(entry.entry_id) or self.progress.get('current')
        if not current or current.get('entry_id') != entry.entry_id:
            return None
        if current.get('file_sha256') != entry.sha256:
            return None
        segments = self.segment_store.segments(entry.entry_id, current['attempt'])
        if len(segments) != int(current.get('segment_index', -1)):
            return None
        if segments:
            last = segments[-1]
            if (last['char_end'] != int(current.get('char_offset', -1))
                    or last['byte_end'] != int(current.get('byte_offset', -1))
                    or last['index'] != len(segments) - 1):
                return None
        return current

    # -- ingestion -------------------------------------------------------
    def ingest(self, entry_ids=None, *, stop_after_segments=None,
               retry_failed=True, now=None):
        """Stream unread/changed (and optionally failed) entries in order.

        stop_after_segments leaves a durable checkpoint mid-entry; the next
        call (a new process or a new session object) resumes at that segment.
        """
        self.scan()
        requested = set(entry_ids) if entry_ids is not None else None
        targets = []
        for entry in self.manifest.entries:
            if requested is not None and entry.entry_id not in requested \
                    and entry.rel_path not in requested:
                continue
            if entry.ignored or entry.kind == 'unsupported':
                continue
            if entry.state in ('unread', 'changed') or (
                    retry_failed and entry.state == 'parse_failed'):
                targets.append(entry)
        budget = None if stop_after_segments is None else int(stop_after_segments)
        interrupted = False
        done = []
        failed = []
        for entry in targets:
            if budget is not None and budget <= 0:
                interrupted = True
                break
            current = self.manifest.entry(entry.entry_id)
            path = self.root / current.rel_path
            if not path.exists():
                self.scan()
                current = self.manifest.entry(entry.entry_id)
                path = self.root / current.rel_path
            result = self._ingest_entry(current, stop_after_segments=budget)
            if budget is not None:
                budget -= result['segments_written']
            if result.get('interrupted'):
                interrupted = True
                break
            if result['status'] == 'read':
                done.append(current.rel_path)
            else:
                failed.append(dict(rel_path=current.rel_path,
                                   error_code=result.get('error_code'),
                                   error=result.get('error')))
        self.checkpoint()
        return dict(interrupted=interrupted, read=done, failed=failed,
                    remaining=self._pending()[:20],
                    segments=len(self.segment_store),
                    manifest_digest=self.manifest.digest(),
                    counts=self.manifest.counts())

    def _pending(self):
        return [entry.rel_path for entry in self.manifest.entries
                if entry.blocking()]

    def _ingest_entry(self, entry, *, stop_after_segments=None):
        path = self.root / entry.rel_path
        point = self._resume_point(entry)
        if point is not None:
            attempt = int(point['attempt'])
            start_byte = int(point['byte_offset'])
            start_char = int(point['char_offset'])
            start_line = int(point.get('line_offset', 1))
            start_index = int(point['segment_index'])
            prefix = self.segment_store.text(entry.entry_id, attempt)
        else:
            attempt = self.segment_store.latest_attempt(entry.entry_id) + 1
            start_byte = 0
            start_char = 0
            start_line = 1
            start_index = 0
            prefix = ''
        existing = self.segment_store.segments(entry.entry_id, attempt)
        written = 0
        self._event('ingest-start', entry_id=entry.entry_id, rel_path=entry.rel_path,
                    attempt=attempt, start_byte=start_byte,
                    start_index=start_index)
        try:
            for piece in iter_text_segments(
                    path, rel_path=entry.rel_path, start_byte=start_byte,
                    start_char=start_char, start_line=start_line,
                    start_index=start_index, chunk_bytes=self.chunk_bytes,
                    segment_chars=self.segment_chars,
                    line_mode=entry.kind in LINE_KINDS):
                segment = dict(piece)
                segment.update(entry_id=entry.entry_id, attempt=attempt,
                               segment_id='seg-%s-a%d-%05d'
                               % (entry.entry_id[4:], attempt, piece['index']))
                self.segment_store.append(segment)
                written += 1
                self.progress['current'] = dict(
                    entry_id=entry.entry_id, attempt=attempt,
                    file_sha256=entry.sha256, byte_offset=piece['byte_end'],
                    char_offset=piece['char_end'], line_offset=piece['next_line'],
                    segment_index=piece['index'] + 1,
                    segments_written=len(existing) + written,
                    chars_written=piece['char_end'])
                if written % self.checkpoint_every_segments == 0:
                    self.checkpoint()
                if stop_after_segments is not None and written >= stop_after_segments:
                    self.progress.setdefault('paused', {})[entry.entry_id] = dict(
                        self.progress['current'])
                    self.checkpoint()
                    self._event('ingest-interrupt', entry_id=entry.entry_id,
                                attempt=attempt, segments=written,
                                byte_offset=piece['byte_end'])
                    return dict(status='interrupted', interrupted=True,
                                segments_written=written,
                                entry_id=entry.entry_id)
        except MaterialError as exc:
            self._mark_failed(entry, attempt, existing, written, exc.code,
                              str(exc), exc.detail)
            return dict(status='parse_failed', segments_written=written,
                        error_code=exc.code, error=str(exc))
        full_text = self.segment_store.text(entry.entry_id, attempt)
        delimiter = '\t' if entry.kind == 'tsv' else ','
        records, detail, error = validate_document(entry.kind, full_text,
                                                   delimiter=delimiter)
        if error is not None:
            self._mark_failed(entry, attempt, existing, written, error['code'],
                              error.get('message', ''), error)
            return dict(status='parse_failed', segments_written=written,
                        error_code=error['code'], error=error.get('message'))
        parser = (detail or {}).get('parser', entry.kind)
        updated = replace(entry, state='read', segments=len(existing) + written,
                          records=int(records), parser=parser,
                          error_code=None, error=None, error_detail=None,
                          attempt=attempt)
        self.manifest = self.manifest.replace_entry(updated)
        self.progress['current'] = None
        (self.progress.setdefault('paused', {}) or {}).pop(entry.entry_id, None)
        self.save_manifest()
        self._event('ingest-read', entry_id=entry.entry_id, attempt=attempt,
                    segments=updated.segments, records=updated.records,
                    chars=len(full_text), sha256=_sha_text(full_text))
        return dict(status='read', segments_written=written, records=records,
                    chars=len(full_text), segments=updated.segments)

    def _mark_failed(self, entry, attempt, existing, written, code, message, detail):
        updated = replace(entry, state='parse_failed', segments=len(existing) + written,
                          records=0, parser=entry.kind, error_code=code,
                          error=message, error_detail=detail, attempt=attempt)
        self.manifest = self.manifest.replace_entry(updated)
        self.progress['current'] = None
        (self.progress.setdefault('paused', {}) or {}).pop(entry.entry_id, None)
        self.save_manifest()
        self._event('ingest-failed', entry_id=entry.entry_id, attempt=attempt,
                    error_code=code, error=message, detail=detail)
        return updated

    # -- inspection ------------------------------------------------------
    def inspection_report(self):
        entries = self.manifest.entries
        read = [entry for entry in entries if entry.state == 'read']
        blocked = {state: [entry for entry in entries
                           if entry.state == state and not entry.ignored]
                   for state in BLOCKING_STATES}
        ignored = [entry for entry in entries if entry.ignored]
        blocking = {state: [entry.rel_path for entry in items]
                    for state, items in blocked.items()}
        report = dict(
            schema=SCHEMA_REPORT, version=1, root=self.manifest.root,
            at=self.now(), manifest_digest=self.manifest.digest(),
            ok=not any(blocking.values()), total=len(entries),
            counts=self.manifest.counts(),
            read=[dict(entry_id=entry.entry_id, rel_path=entry.rel_path,
                       kind=entry.kind, segments=entry.segments,
                       records=entry.records, chars=self._read_chars(entry),
                       sha256=entry.sha256) for entry in read],
            unread=[dict(entry_id=entry.entry_id, rel_path=entry.rel_path)
                    for entry in blocked['unread']],
            parse_failed=[dict(entry_id=entry.entry_id, rel_path=entry.rel_path,
                               error_code=entry.error_code, error=entry.error,
                               detail=entry.error_detail)
                          for entry in blocked['parse_failed']],
            changed=[dict(entry_id=entry.entry_id, rel_path=entry.rel_path)
                     for entry in blocked['changed']],
            unsupported=[dict(entry_id=entry.entry_id, rel_path=entry.rel_path,
                              error_code=entry.error_code)
                         for entry in blocked['unsupported']],
            ignored=[dict(entry_id=entry.entry_id, rel_path=entry.rel_path,
                          reason=entry.ignored_reason) for entry in ignored],
            blocking=blocking,
            segments=sum(entry.segments for entry in entries),
            records=sum(entry.records for entry in entries),
            read_bytes=sum(entry.size for entry in read))
        report['digest'] = hashlib.sha256(canonical(report).encode('utf-8')).hexdigest()
        return report

    def _read_chars(self, entry):
        if entry.state != 'read':
            return 0
        try:
            return len(self.segment_store.text(entry.entry_id, entry.attempt or None))
        except MaterialError:
            return 0

    def mark_ignored(self, entry_id, reason, *, now=None):
        if not reason or not str(reason).strip():
            raise MaterialError('mi.ignore-reason', 'explicit reason required')
        entry = self.manifest.entry(entry_id)
        if entry.state not in BLOCKING_STATES:
            raise MaterialError('mi.ignore-state', entry.state)
        updated = replace(entry, ignored=True, ignored_reason=str(reason))
        self.manifest = self.manifest.replace_entry(updated)
        self.save_manifest()
        self._event('ignore', entry_id=entry_id, rel_path=entry.rel_path,
                    reason=str(reason))
        return updated

    def unignore(self, entry_id, *, now=None):
        entry = self.manifest.entry(entry_id)
        updated = replace(entry, ignored=False, ignored_reason=None)
        self.manifest = self.manifest.replace_entry(updated)
        self.save_manifest()
        self._event('unignore', entry_id=entry_id, rel_path=entry.rel_path)
        return updated

    # -- reads -----------------------------------------------------------
    def read_segment(self, segment_id):
        return self.segment_store.read(segment_id)['text']

    def locate(self, segment_id):
        return self.segment_store.locate(segment_id)

    def read_entry(self, entry_id, *, verify=True):
        entry = self.manifest.entry(entry_id)
        text = self.segment_store.text(entry_id, entry.attempt or None)
        if verify and entry.state == 'read' and _sha_text(text) != entry.sha256:
            raise MaterialError('mi.text-digest', entry.rel_path)
        return text

    # -- status / verification ------------------------------------------
    def resume(self):
        self.scan()
        paused = dict(self.progress.get('paused') or {})
        current = (self.progress.get('current')
                   or (next(iter(paused.values())) if paused else None))
        return dict(root=str(self.root), interrupted=bool(current or paused),
                    current=current, paused=paused, pending=self._pending(),
                    manifest_digest=self.manifest.digest(),
                    counts=self.manifest.counts(),
                    segments=len(self.segment_store))

    def status(self):
        return dict(root=str(self.root), counts=self.manifest.counts(),
                    manifest_digest=self.manifest.digest(),
                    current=self.progress.get('current'),
                    segments=len(self.segment_store),
                    report_digest=self.inspection_report()['digest'])

    def verify(self):
        segments = self.segment_store.verify()
        checked = 0
        for entry in self.manifest.entries:
            if entry.state != 'read':
                continue
            text = self.segment_store.text(entry.entry_id, entry.attempt or None)
            if _sha_text(text) != entry.sha256:
                raise MaterialError('mi.text-digest', entry.rel_path)
            segments_here = self.segment_store.segments(entry.entry_id,
                                                        entry.attempt or None)
            for expected, segment in enumerate(segments_here):
                if segment['index'] != expected:
                    raise MaterialError('mi.segment-order', entry.rel_path)
                if segment['entry_id'] != entry.entry_id:
                    raise MaterialError('mi.segment-entry', segment['segment_id'])
            checked += 1
        return dict(ok=True, entries=checked, segments=segments['segments'],
                    manifest_digest=self.manifest.digest())
