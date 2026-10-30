"""Durable preassembly session layout: banks, page index, request WAL, one pointer.

The frozen runtime already commits four neural banks (core experience state,
KV/GDN native capsule, episodic codes, mental codes) plus the output profile as
one resident generation (resident_store.ResidentStore.save_generation).  The
preassembly candidate additionally has to survive three things that generation
did not cover:

1. a request that has been accepted but not yet answered -- its identity, the
   output produced so far and the identities of actions already executed,
2. the experience address index that makes one record readable from its page
   without loading the whole journal,
3. the task/material store, when the product layer installs one.

This module adds those components *around* the existing generation writer and
publishes them behind a single atomic pointer, so a reader sees either the
previous complete generation or the next complete generation, never a mix.  It
never rewrites a committed generation and never writes to model-ready.

On-disk layout (relative to one store root):

    CURRENT.json                      atomic pointer, replaced last
    generations/<seq>-<hex>/
        generation.json               inventory: every file + component manifests
        banks/                        ResidentStore generation (four banks, profile,
                                       profile moments, trainable + optimizer state)
        experience/INDEX.json         address -> page -> body hash map
        experience/pages/<p>.json     record bodies, one page at a time
        request/                      snapshot of the live request WAL
        tasks/TASKS.json              only when a provider is installed
        diagnostics/STARTUP.json      startup view captured at commit time
    request/                          live write-ahead request (fsync per step)
    requests/<request_id>/            requests that completed (audit trail)
    staging/<uuid>/                   unpublished work; invisible to readers

Durability contract
-------------------
RequestJournal.begin writes INTENT.json and fsyncs it *before* any token is
generated.  Produced output and action receipts are appended and fsynced before
the banks are committed, so a crash leaves either (a) the previous generation
plus a resumable live request, or (b) a generation whose request snapshot already
contains the produced output.  Request identity (request_id, event_id) and
action identity (operation_id) are stable across a resume, therefore a replayed
request cannot become a second observation and an executed action cannot run
twice.

Nothing here loads the 4B body: torch is imported only inside the resident-store
provider, so the layout is importable and testable on CPU.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

LAYOUT_SCHEMA = 'treyrin-preassembly-session-v1'
GENERATION_SCHEMA = 'treyrin-preassembly-generation-v1'
POINTER_SCHEMA = 'treyrin-preassembly-pointer-v1'
REQUEST_SCHEMA = 'treyrin-request-journal-v1'
REQUEST_SNAPSHOT_SCHEMA = 'treyrin-request-snapshot-v1'
INDEX_SCHEMA = 'treyrin-experience-index-v1'
DIAGNOSTICS_SCHEMA = 'treyrin-startup-diagnostics-v1'
MIGRATION_SCHEMA = 'treyrin-preassembly-migration-v1'
BANKS_SCHEMA = 'treyrin-resident-generation-v1'
STATE_SCHEMA = 'treyrin-structural-learning-v3'
LIVE_KIND = 'live-role-conversation'
LEGACY_KIND = 'legacy-snapshot'
LAYOUT_VERSION = 1
DEFAULT_STORE_NAME = 'resident-session'
DEFAULT_MODEL_NAME = 'model-ready'
MODEL_POINTER = 'CURRENT-MODEL.json'
REQUEST_STATES = ('pending', 'generating', 'generated', 'completed', 'cancelled', 'interrupted', 'failed')
UNFINISHED_STATES = ('pending', 'generating', 'generated')
RESUMABLE_STATES = ('pending', 'generating', 'generated', 'cancelled', 'interrupted', 'failed')
PAGE_SIZE = 64

# Files whose bytes define whether a saved generation may be resumed by this
# source tree.  Every generation records them, so a later difference is evidence
# rather than a guess.  Missing files are recorded as absent, never as equal.
SOURCE_FILES = (
    'preassembly_api.py','interaction_state.py','verify_revise.py','live_memory.py',
    'individual_learning.py','fixed_output_transaction.py','treyrin_app.py',
    'candidate.py', 'role_conversation.py', 'native_session.py', 'native_stream.py',
    'replay_session.py', 'resident_store.py', 'episode_codes.py', 'mental_codes.py',
    'native_decode.py', 'decode_runtime.py', 'generation_result.py', 'message_stream.py',
    'public_labels.py', 'self_description.py', 'runtime_self_info.py', 'native_cognition.py',
    'treyrin/preassembly_session.py', 'treyrin/structural_learning.py',
    'treyrin/structural_core.py', 'treyrin/structural_model.py',
    'treyrin/cognitive_workspace.py', 'treyrin/learning_checkpoint.py',
    'tools/preassembly_cli.py', 'tools/migrate_preassembly_layout.py',
)


class PreassemblyError(RuntimeError):
    """A durable-layout invariant was violated; the store was left untouched."""


# --------------------------------------------------------------------------- #
# small durable-file helpers
# --------------------------------------------------------------------------- #

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def sha256_value(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat()


def _fsync(handle):
    handle.flush()
    os.fsync(handle.fileno())


def write_json(path, value, *, indent=2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, sort_keys=True, ensure_ascii=False, indent=indent)
        handle.write('\n')
        _fsync(handle)
    return path


def write_json_atomic(path, value):
    """Replace one small file such that a reader sees old or new, never partial."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temporary.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, sort_keys=True, ensure_ascii=False, indent=2)
        handle.write('\n')
        _fsync(handle)
    os.replace(temporary, path)
    return path


def append_jsonl(path, value):
    """Append one durable record; used for output and executed-action receipts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + '\n')
        _fsync(handle)
    return path


def read_json(path, *, default=None):
    path = Path(path)
    if not path.exists():
        if default is not None:
            return default
        raise PreassemblyError('missing durable file: ' + str(path))
    return json.loads(path.read_text('utf-8'))


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text('utf-8').splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def walk_files(directory):
    directory = Path(directory)
    return sorted(path.relative_to(directory).as_posix() for path in directory.rglob('*') if path.is_file())


def inventory(directory):
    directory = Path(directory)
    rows = {}
    for name in walk_files(directory):
        path = directory / name
        rows[name] = {'sha256': sha256_file(path), 'bytes': path.stat().st_size}
    return rows


def unique_bytes(directory):
    """Storage actually used: hardlinked payloads are counted once."""
    return storage_report(directory)['unique_bytes']


def storage_report(directory):
    """Apparent size, bytes actually held, and bytes shared through hardlinks."""
    directory = Path(directory)
    state = dict(path=str(directory), apparent_bytes=0, unique_bytes=0,
                 shared_with_other_paths_bytes=0, files=0, shared_files=0)
    if not directory.exists():
        state['state'] = 'absent'
        return state
    seen = set()
    for path in directory.rglob('*'):
        if not path.is_file():
            continue
        stat = path.stat()
        state['files'] += 1
        state['apparent_bytes'] += stat.st_size
        if stat.st_nlink > 1:
            state['shared_files'] += 1
        key = (stat.st_dev, stat.st_ino)
        if key in seen:
            continue
        seen.add(key)
        state['unique_bytes'] += stat.st_size
        if stat.st_nlink > 1:
            state['shared_with_other_paths_bytes'] += stat.st_size
    state['state'] = 'measured'
    return state


# --------------------------------------------------------------------------- #
# request journal (write-ahead)
# --------------------------------------------------------------------------- #

class RequestJournal:
    """One live request: intent, produced output, executed-action identities.

    Every mutation is fsynced before the caller may start generating.  A journal
    is created exclusively (a second live request is refused, never merged), so
    the store never carries two competing current requests.
    """

    def __init__(self, directory):
        self.directory = Path(directory)

    # -- reading -----------------------------------------------------------
    @property
    def exists(self):
        return (self.directory / 'INTENT.json').is_file()

    @property
    def intent(self):
        return read_json(self.directory / 'INTENT.json')

    @property
    def state(self):
        return read_json(self.directory / 'STATE.json', default={
            'schema': REQUEST_SCHEMA, 'status': 'missing', 'updated_at': None})

    @property
    def status(self):
        return self.state['status']

    def resumable(self):
        return self.status in RESUMABLE_STATES

    # -- creating ----------------------------------------------------------
    @classmethod
    def begin(cls, directory, *, request_id, text, event_id, known_at, source='human_user',
              frame=None, banks_sequence=None, conversation_id=None, meta=None):
        directory = Path(directory)
        if (directory / 'INTENT.json').exists():
            raise PreassemblyError('a live request already exists; resume or cancel it before a new one')
        if not isinstance(text, str) or not text:
            raise PreassemblyError('nonempty request text required')
        if not request_id or not event_id:
            raise PreassemblyError('request/event identity required')
        directory.mkdir(parents=True, exist_ok=False)
        intent = dict(schema=REQUEST_SCHEMA, request_id=request_id, text=text, event_id=event_id,
                      source=source, known_at=known_at, created_at=now_iso(), frame=frame,
                      banks_sequence_at_begin=banks_sequence, conversation_id=conversation_id,
                      meta=dict(meta or {}))
        write_json(directory / 'INTENT.json', intent)
        write_json(directory / 'STATE.json', dict(schema=REQUEST_SCHEMA, request_id=request_id,
                                                  status='pending', updated_at=now_iso(),
                                                  attempts=0, resumed=0))
        return intent

    @classmethod
    def load(cls, directory):
        journal = cls(directory)
        if not journal.exists:
            raise PreassemblyError('no live request journal in ' + str(directory))
        return journal

    # -- mutating ----------------------------------------------------------
    def set_status(self, status, **extra):
        if status not in REQUEST_STATES:
            raise PreassemblyError('unknown request status: ' + str(status))
        state = dict(self.state)
        state.update(extra)
        state.update(schema=REQUEST_SCHEMA, request_id=self.intent['request_id'],
                     status=status, updated_at=now_iso())
        write_json_atomic(self.directory / 'STATE.json', state)
        return state

    def _attempt(self):
        return int(self.state.get('attempts') or 0)

    def append_output(self, text, *, token_ids=None, finish_reason=None, delivered=False,
                      supersedes_attempt=None, note=None):
        """Durably record produced output.  Called before the banks are committed."""
        if not isinstance(text, str):
            raise PreassemblyError('output text must be a string')
        rows = read_jsonl(self.directory / 'OUTPUT.jsonl')
        row = dict(index=len(rows), at=now_iso(), attempt=self._attempt(), text=text,
                   characters=len(text),
                   token_ids=[int(x) for x in token_ids] if token_ids is not None else None,
                   token_count=len(token_ids) if token_ids is not None else None,
                   finish_reason=finish_reason, delivered=bool(delivered),
                   supersedes_attempt=supersedes_attempt, note=note)
        append_jsonl(self.directory / 'OUTPUT.jsonl', row)
        return row

    def output(self):
        rows = read_jsonl(self.directory / 'OUTPUT.jsonl')
        delivered = [row for row in rows if row.get('delivered')]
        return dict(chunks=len(rows), characters=sum(row['characters'] for row in rows),
                    text=''.join(row['text'] for row in rows),
                    delivered_text=''.join(row['text'] for row in delivered),
                    delivered_characters=sum(row['characters'] for row in delivered),
                    superseded=[row for row in rows if row.get('supersedes_attempt') is not None],
                    rows=[dict(index=row['index'], attempt=row['attempt'], characters=row['characters'],
                               delivered=row['delivered'], finish_reason=row['finish_reason'])
                          for row in rows])

    def action(self, action_id):
        for row in reversed(read_jsonl(self.directory / 'ACTIONS.jsonl')):
            if row['action_id'] == action_id:
                return row
        return None

    def actions(self):
        return read_jsonl(self.directory / 'ACTIONS.jsonl')

    def executed_actions(self):
        return [row for row in self.actions() if row.get('state') == 'executed']

    def begin_action(self, action_id, kind, *, payload_sha256=None):
        prior = self.action(action_id)
        if prior is not None and prior.get('state') == 'executed':
            return prior
        row = dict(action_id=action_id, kind=kind, state='executing', at=now_iso(),
                   attempt=self._attempt(), payload_sha256=payload_sha256, receipt_sha256=None)
        append_jsonl(self.directory / 'ACTIONS.jsonl', row)
        return row

    def record_action(self, action_id, kind, *, receipt=None, receipt_sha256=None, value_sha256=None):
        prior = self.action(action_id)
        if prior is not None and prior.get('state') == 'executed':
            if value_sha256 is not None and prior.get('payload_sha256') != value_sha256:
                raise PreassemblyError('action identity reused with a different payload: ' + action_id)
            return prior
        digest = receipt_sha256 or (sha256_value(receipt) if receipt is not None else None)
        row = dict(action_id=action_id, kind=kind, state='executed', at=now_iso(),
                   attempt=self._attempt(), payload_sha256=(prior or {}).get('payload_sha256'),
                   receipt_sha256=digest)
        if receipt is not None and len(canonical(receipt)) <= 4096:
            row['receipt'] = receipt
        append_jsonl(self.directory / 'ACTIONS.jsonl', row)
        return row

    def fail_action(self, action_id, error):
        row = dict(action_id=action_id, kind=(self.action(action_id) or {}).get('kind'), state='failed',
                   at=now_iso(), attempt=self._attempt(), error=str(error)[:2000])
        append_jsonl(self.directory / 'ACTIONS.jsonl', row)
        return row

    # -- moving parts around ----------------------------------------------
    def snapshot(self, target):
        """Copy the live journal into a generation; the live journal keeps running."""
        target = Path(target)
        target.mkdir(parents=True, exist_ok=False)
        files = {}
        for name in ('INTENT.json', 'STATE.json', 'OUTPUT.jsonl', 'ACTIONS.jsonl','TOKENS.jsonl'):
            source = self.directory / name
            if not source.exists():
                continue
            data = source.read_bytes()
            (target / name).write_bytes(data)
            # Windows FlushFileBuffers needs a writable descriptor.
            with (target / name).open('r+b') as handle:
                os.fsync(handle.fileno())
            files[name] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
        snapshot = dict(schema=REQUEST_SNAPSHOT_SCHEMA, request_id=self.intent['request_id'],
                        status=self.status, files=files, output=self.output(),
                        executed_actions=[dict(action_id=row['action_id'], kind=row['kind'],
                                               receipt_sha256=row.get('receipt_sha256'))
                                          for row in self.executed_actions()],
                        snapshot_sha256=sha256_value(files))
        write_json(target / 'SNAPSHOT.json', snapshot)
        return snapshot

    def archive(self, requests_dir):
        """Move a finished live journal into the permanent per-request history."""
        requests_dir = Path(requests_dir)
        requests_dir.mkdir(parents=True, exist_ok=True)
        intent, state = self.intent, self.state
        request_id = intent['request_id']
        target = requests_dir / request_id
        if target.exists():
            raise PreassemblyError('request archive already exists: ' + request_id)
        shutil.move(str(self.directory), str(target))
        record = dict(request_id=request_id, directory=target.name, status=state['status'],
                      archived_at=now_iso(), text=intent['text'][:400],
                      event_id=intent['event_id'], known_at=intent['known_at'],
                      created_at=intent['created_at'])
        write_json(requests_dir / (request_id + '.json'), record)
        return record

    def plan(self, *, current_sequence=None):
        """What a resuming process may do, from durable facts only."""
        intent, state = self.intent, self.state
        output = self.output()
        executed = self.executed_actions()
        resumable = self.resumable()
        return dict(resumable=resumable, request_id=intent['request_id'], event_id=intent['event_id'],
                    text=intent['text'], known_at=intent['known_at'], source=intent['source'],
                    status=state['status'], attempts=int(state.get('attempts') or 0),
                    resumed=int(state.get('resumed') or 0),
                    output_recorded=output['characters'] > 0,
                    emitted_text=output['text'], emitted_characters=output['characters'],
                    delivered_text=output['delivered_text'],
                    delivered_characters=output['delivered_characters'],
                    superseded_attempts=len(output['superseded']),
                    executed_actions=[dict(action_id=row['action_id'], kind=row['kind'])
                                      for row in executed],
                    banks_sequence_at_begin=intent.get('banks_sequence_at_begin'),
                    banks_committed_since_begin=(current_sequence is not None
                        and intent.get('banks_sequence_at_begin') is not None
                        and current_sequence > intent['banks_sequence_at_begin']),
                    next_step=('regenerate' if resumable else 'none'))


# --------------------------------------------------------------------------- #
# experience address index + pages
# --------------------------------------------------------------------------- #

def _observation_record(ordinal, observation):
    text = observation['text']
    return dict(address='observation:%d' % ordinal, kind='observation', ordinal=ordinal,
                identity=observation['event_id'], body=text, body_sha256=sha256_text(text),
                characters=len(text), source=observation.get('source', ''), status='observed',
                known_at=observation.get('known_at', ''), revises='', evidence_ids=[],
                occurred_at=observation.get('occurred_at', ''), scope=observation.get('scope', ''))


def _mental_record(ordinal, record):
    text = record['text']
    return dict(address='mental:%d' % ordinal, kind='mental', ordinal=ordinal,
                identity=record['record_id'], body=text, body_sha256=sha256_text(text),
                characters=len(text), source=record.get('source', ''),
                status=record.get('status', ''), known_at=record.get('known_at', ''),
                revises=record.get('revises') or '',
                evidence_ids=list(record.get('evidence_ids') or ()),
                record_kind=record.get('kind', ''), goal_action_of=record.get('goal_action_of', ''),
                action_name=record.get('action_name', ''),
                goal_metadata=record.get('goal_metadata', ''))


def index_records(journal):
    """Address every record exactly the way the live replay path addresses it."""
    rows = []
    for ordinal, observation in enumerate(journal.get('observations') or (),1):
        rows.append(_observation_record(ordinal, observation))
    for ordinal, record in enumerate(journal.get('mental_records') or (),1):
        rows.append(_mental_record(ordinal, record))
    return rows


def build_index(journal, *, page_size=PAGE_SIZE, journal_sha256=None, created_at=None):
    """Build the paged address index; record bodies live in pages, not in the index."""
    if type(page_size) is not int or page_size < 1:
        raise PreassemblyError('page size must be a positive int')
    rows = index_records(journal)
    created_at = created_at or now_iso()
    pages, entries = [], []
    for start in range(0, len(rows), page_size):
        chunk = rows[start:start + page_size]
        page_index = len(pages)
        pages.append(dict(schema=INDEX_SCHEMA, page=page_index,
                          addresses=[row['address'] for row in chunk],
                          bodies={row['address']: row['body'] for row in chunk},
                          created_at=created_at))
        for row in chunk:
            entry = dict(address=row['address'], kind=row['kind'], ordinal=row['ordinal'],
                         identity=row['identity'], body_sha256=row['body_sha256'],
                         characters=row['characters'], source=row['source'],
                         status=row['status'], known_at=row['known_at'], revises=row['revises'],
                         evidence_ids=row['evidence_ids'], page=page_index)
            for extra in ('record_kind', 'goal_action_of', 'action_name', 'goal_metadata',
                          'occurred_at', 'scope'):
                if row.get(extra):
                    entry[extra] = row[extra]
            entries.append(entry)
    revision_links = {}
    for row in rows:
        if row['kind'] == 'mental' and row['revises']:
            revision_links.setdefault(row['revises'], []).append(row['identity'])
    index = dict(schema=INDEX_SCHEMA, created_at=created_at, page_size=page_size,
                 journal_sha256=journal_sha256,
                 counts=dict(observations=len(journal.get('observations') or ()),
                             mental_records=len(journal.get('mental_records') or ()),
                             records=len(rows)),
                 pages=[dict(page=page['page'], records=len(page['addresses']),
                             addresses=page['addresses'],
                             bodies_sha256=sha256_value(page['bodies'])) for page in pages],
                 records=entries, revision_links=revision_links)
    return index, pages


def write_index(directory, index, pages):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'pages').mkdir()
    for page in pages:
        write_json(directory / 'pages' / ('%06d.json' % page['page']), page)
    write_json(directory / 'INDEX.json', index)
    return dict(index_sha256=sha256_file(directory / 'INDEX.json'), pages=len(pages),
                records=index['counts']['records'],
                page_files={name: sha256_file(directory / 'pages' / name)
                            for name in walk_files(directory / 'pages')})


def load_index(directory):
    directory = Path(directory)
    index = read_json(directory / 'INDEX.json')
    if index.get('schema') != INDEX_SCHEMA:
        raise PreassemblyError('unknown experience index schema')
    return index


def read_record(directory, address):
    """Page in exactly one page and return one record body (on-demand paging)."""
    directory = Path(directory)
    index = load_index(directory)
    entry = next((row for row in index['records'] if row['address'] == address), None)
    if entry is None:
        raise PreassemblyError('unknown experience address: ' + str(address))
    page = read_json(directory / 'pages' / ('%06d.json' % entry['page']))
    body = page['bodies'].get(address)
    if body is None:
        raise PreassemblyError('address missing from its page: ' + str(address))
    if sha256_text(body) != entry['body_sha256']:
        raise PreassemblyError('paged body differs from the indexed hash: ' + str(address))
    return dict(entry, body=body, page_loaded=entry['page'])


def verify_index(directory, journal, *, sample=16):
    """Check body/address/source/status/revision per record against the journal.

    The journal is the authority: the index is verified value by value, never by
    hash equality of a derived blob.  Any difference is a hard failure.
    """
    directory = Path(directory)
    index = load_index(directory)
    expected = index_records(journal)
    if index['counts']['records'] != len(expected):
        raise PreassemblyError('index record count differs from the journal')
    page_cache, mismatches, checked, samples = {}, [], 0, []
    for row in expected:
        entry = next((item for item in index['records'] if item['address'] == row['address']), None)
        if entry is None:
            mismatches.append(dict(address=row['address'], field='address', detail='missing'))
            continue
        if entry['page'] not in page_cache:
            page_cache[entry['page']] = read_json(directory / 'pages' / ('%06d.json' % entry['page']))
        body = page_cache[entry['page']]['bodies'].get(row['address'])
        fields = {'body_sha256': (entry['body_sha256'], row['body_sha256']),
                  'body': (sha256_text(body) if body is not None else None, row['body_sha256']),
                  'identity': (entry['identity'], row['identity']),
                  'source': (entry['source'], row['source']),
                  'status': (entry['status'], row['status']),
                  'revises': (entry['revises'], row['revises']),
                  'known_at': (entry['known_at'], row['known_at']),
                  'characters': (entry['characters'], row['characters']),
                  'evidence_ids': (entry['evidence_ids'], row['evidence_ids'])}
        checked += 1
        for name, (stored, authoritative) in fields.items():
            if stored != authoritative:
                mismatches.append(dict(address=row['address'], field=name, index=stored,
                                       journal=authoritative))
        if len(samples) < sample:
            samples.append(dict(address=row['address'], identity=row['identity'],
                                source=entry['source'], status=entry['status'],
                                revises=entry['revises'], evidence_ids=entry['evidence_ids'],
                                body_sha256=entry['body_sha256'], characters=entry['characters'],
                                body_head=row['body'][:80]))
    links = {}
    for row in expected:
        if row['kind'] == 'mental' and row['revises']:
            links.setdefault(row['revises'], []).append(row['identity'])
    if links != index.get('revision_links'):
        mismatches.append(dict(address='*', field='revision_links', index=index.get('revision_links'),
                               journal=links))
    return dict(checked=checked, records=len(expected), mismatches=mismatches, sample=samples,
                revision_links=links, verified=not mismatches)


# --------------------------------------------------------------------------- #
# generation + layout
# --------------------------------------------------------------------------- #

class Staging:
    """Unpublished generation content; removed on failure, renamed on publish."""

    def __init__(self, layout):
        self.layout = layout
        self.path = layout.staging_dir / uuid.uuid4().hex
        self.path.mkdir(parents=True, exist_ok=False)
        self.published = False

    def component(self, name):
        path = self.path / name
        path.mkdir(parents=True, exist_ok=False)
        return path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if not self.published:
            shutil.rmtree(self.path, ignore_errors=True)
        return False


class Generation:
    """One committed generation, read through its inventory."""

    def __init__(self, layout, directory, manifest):
        self.layout = layout
        self.directory = Path(directory)
        self.manifest = manifest

    # -- plain accessors ---------------------------------------------------
    @property
    def sequence(self): return int(self.manifest['sequence'])
    @property
    def kind(self): return self.manifest['kind']
    @property
    def components(self): return self.manifest['components']
    @property
    def versions(self): return self.manifest.get('versions', {})
    @property
    def banks(self): return self.directory / 'banks'
    @property
    def experience(self): return self.directory / 'experience'
    @property
    def request(self): return self.directory / 'request'
    @property
    def tasks(self): return self.directory / 'tasks'

    def banks_manifest(self):
        return read_json(self.banks / 'manifest.json')

    def session_manifest(self):
        return read_json(self.banks / 'conversation' / 'core-native' / 'core' / 'session' / 'manifest.json')

    def journal(self):
        return read_json(self.banks / 'conversation' / 'core-native' / 'core' / 'session' / 'journal.json')

    def index(self):
        return load_index(self.experience)

    def request_snapshot(self):
        return read_json(self.request / 'SNAPSHOT.json')

    def record(self, address):
        return read_record(self.experience, address)

    # -- verification ------------------------------------------------------
    def verify(self, *, deep=True):
        """Require every listed file to be present and unchanged."""
        manifest = self.manifest
        if manifest.get('schema') != GENERATION_SCHEMA:
            raise PreassemblyError('unknown generation schema')
        if manifest.get('layout_version') != LAYOUT_VERSION:
            raise PreassemblyError('layout version differs: %r' % (manifest.get('layout_version'),))
        problems, checked = [], 0
        listed = manifest['files']
        for name, row in sorted(listed.items()):
            path = self.directory / name
            if not path.is_file():
                problems.append(dict(file=name, problem='missing'))
                continue
            size = path.stat().st_size
            if size != row['bytes']:
                problems.append(dict(file=name, problem='size', stored=row['bytes'], actual=size))
                continue
            if deep and sha256_file(path) != row['sha256']:
                problems.append(dict(file=name, problem='sha256'))
                continue
            checked += 1
        unlisted = [name for name in walk_files(self.directory)
                    if name not in listed and name != 'generation.json']
        if unlisted:
            problems.append(dict(files=unlisted, problem='unlisted'))
        cross = self._cross_checks()
        problems.extend(cross.pop('problems'))
        return dict(sequence=self.sequence, kind=self.kind, files=len(listed), checked=checked,
                    deep=deep, problems=problems, cross_checks=cross, verified=not problems)

    def _cross_checks(self):
        """One generation must agree with itself; a mixed generation is refused."""
        problems, result = [], {}
        banks = self.banks_manifest()
        session = self.session_manifest()
        index = self.index()
        conversation = read_json(self.banks / 'conversation' / 'manifest.json')
        result.update(banks_schema=banks.get('schema'), banks_sequence=banks.get('sequence'),
                      banks_conversation_id=banks.get('conversation_id'),
                      optimizer_steps=banks.get('optimizer_steps'),
                      parameter_binding=banks.get('parameter_binding'),
                      four_banks=banks.get('four_banks') or {})
        if banks.get('schema') != BANKS_SCHEMA:
            problems.append(dict(component='banks', problem='schema', value=banks.get('schema')))
        if banks.get('conversation_kind') != 'role-conversation':
            problems.append(dict(component='banks', problem='conversation_kind',
                                 value=banks.get('conversation_kind')))
        entry = self.components.get('banks') or {}
        if entry.get('sequence') != banks.get('sequence'):
            problems.append(dict(component='banks', problem='sequence drifted from the inventory'))
        four = banks.get('four_banks') or {}
        if index['counts']['observations'] != four.get('observations'):
            problems.append(dict(component='experience', problem='observations differ from the banks',
                                 index=index['counts']['observations'], banks=four.get('observations')))
        if index['counts']['mental_records'] != four.get('mental_records'):
            problems.append(dict(component='experience', problem='mental records differ from the banks',
                                 index=index['counts']['mental_records'],
                                 banks=four.get('mental_records')))
        if index.get('journal_sha256') != session.get('journal_sha256'):
            problems.append(dict(component='experience', problem='journal hash differs from the banks',
                                 index=index.get('journal_sha256'),
                                 banks=session.get('journal_sha256')))
        snapshot = self.request_snapshot()
        request_entry = self.components.get('request') or {}
        if snapshot.get('request_id') != request_entry.get('request_id'):
            problems.append(dict(component='request', problem='request identity drifted'))
        result.update(journal_sha256=index.get('journal_sha256'),
                      request_id=snapshot.get('request_id'),
                      request_status=snapshot.get('status'),
                      task_state=(self.components.get('tasks') or {}).get('state'),
                      profile_sha256=banks.get('profile_sha256'),
                      profile_moments_sha256=banks.get('profile_moments_sha256'),
                      contract_identity=session.get('contract_identity'),
                      conversation_source_sha256=conversation.get('source_sha256'),
                      problems=problems)
        return result

    def compatibility(self, *, root=None, candidate=None, strict=False):
        """Explicit encoding / parameter / state version relation for this generation."""
        versions = self.versions
        saved_sources = versions.get('sources', {})
        # Only names the snapshot actually declared can be compared.  A module
        # that appeared afterwards does not change the saved banks; it does
        # change candidate.code_hashes(), which is why the weights gate below is
        # reported separately from generation compatibility.
        current_sources = ({name: sha256_file(Path(root) / name) for name in saved_sources
                            if (Path(root) / name).is_file()} if root is not None else {})
        binding_now = getattr(getattr(candidate, 'model', None), '_binding', None)
        differing = sorted(name for name in saved_sources
                           if name in current_sources and saved_sources[name] != current_sources[name])
        missing = sorted(name for name in saved_sources if name not in current_sources)
        added = sorted(name for name in source_versions(root) if name not in saved_sources) if root is not None else []
        report = dict(
            saved=dict(layout_version=versions.get('layout_version'),
                       parameter_binding=versions.get('parameter_binding'),
                       encoding_sha256=versions.get('encoding_sha256'),
                       state_schema=versions.get('state_schema'),
                       contract_identity=versions.get('contract_identity'),
                       sources=saved_sources),
            current=(dict(parameter_binding=binding_now, source_files=len(current_sources))
                     if root is not None else None),
            sources=dict(differing=differing, missing=missing, newer_modules_not_declared=added),
            # The parameter binding is NOT decided here: a fresh candidate has
            # its own random initial binding until the banks are restored, and
            # ResidentStore.check_generation is the single authority that
            # compares the saved profile against the restored parameters.
            checks=dict(layout_version=versions.get('layout_version') == LAYOUT_VERSION,
                        state_schema=versions.get('state_schema') == STATE_SCHEMA,
                        parameter_binding=None,
                        source_identity=((not differing and not missing)
                                         if root is not None else None),
                        newer_modules_not_declared=(bool(added) if root is not None else None)),
            binding_authority='ResidentStore.check_generation during banks restore')
        blocking = [name for name in ('layout_version', 'state_schema', 'source_identity')
                    if report['checks'][name] is False]
        report['blocking'] = blocking
        report['compatible'] = not blocking
        if strict and blocking:
            raise PreassemblyError('generation is not compatible with this source: '
                                   + ', '.join(blocking))
        return report


class Layout:
    """Store root: pointer, generations, staging, live request and its archive."""

    def __init__(self, root):
        self.root = Path(root)

    # -- paths -------------------------------------------------------------
    @property
    def pointer_path(self): return self.root / 'CURRENT.json'
    @property
    def generations_dir(self): return self.root / 'generations'
    @property
    def staging_dir(self): return self.root / 'staging'
    @property
    def request_path(self): return self.root / 'request'
    @property
    def requests_dir(self): return self.root / 'requests'

    def prepare(self):
        for path in (self.root, self.generations_dir, self.staging_dir, self.requests_dir):
            path.mkdir(parents=True, exist_ok=True)
        return self

    # -- pointer -----------------------------------------------------------
    def pointer(self):
        if not self.pointer_path.exists():
            return None
        record = read_json(self.pointer_path)
        if record.get('schema') != POINTER_SCHEMA:
            raise PreassemblyError('unknown pointer schema: %r' % (record.get('schema'),))
        target = (self.root / record['directory']).resolve()
        if not target.is_relative_to(self.root.resolve()):
            raise PreassemblyError('pointer escapes its store')
        generation = target / 'generation.json'
        if not generation.is_file():
            raise PreassemblyError('pointer targets a generation without an inventory')
        if sha256_file(generation) != record['generation_sha256']:
            raise PreassemblyError('pointer inventory hash differs; the store was altered or torn')
        inventory = read_json(generation)
        if inventory.get('sequence') != record['sequence']:
            raise PreassemblyError('pointer sequence differs from its generation inventory')
        return record

    def current(self):
        record = self.pointer()
        if record is None:
            return None
        directory = self.root / record['directory']
        return Generation(self, directory, read_json(directory / 'generation.json'))

    def sequence(self):
        record = self.pointer()
        return int(record['sequence']) if record else 0

    def generations(self):
        rows = []
        if not self.generations_dir.exists():
            return rows
        for directory in sorted(self.generations_dir.iterdir()):
            if directory.is_dir() and (directory / 'generation.json').is_file():
                rows.append(Generation(self, directory, read_json(directory / 'generation.json')))
        return rows

    # -- live request ------------------------------------------------------
    def live_request(self):
        journal = RequestJournal(self.request_path)
        return journal if journal.exists else None

    def archived_requests(self):
        rows = []
        if not self.requests_dir.exists():
            return rows
        for path in sorted(self.requests_dir.glob('*.json')):
            try:
                rows.append(read_json(path))
            except ValueError:
                continue
        return rows

    # -- staging / publish -------------------------------------------------
    def staging(self):
        self.prepare()
        return Staging(self)

    def publish(self, staging, meta):
        """Complete the inventory, rename into place, then flip the pointer."""
        path = Path(staging.path)
        banks_manifest = read_json(path / 'banks' / 'manifest.json')
        files = inventory(path)
        self.prepare()
        sequence = self.sequence() + 1
        name = '%012d-%s' % (sequence, uuid.uuid4().hex[:8])
        generation_meta = dict(schema=GENERATION_SCHEMA, layout_version=LAYOUT_VERSION,
                               sequence=sequence, kind=meta.get('kind', LIVE_KIND),
                               created_at=now_iso(),
                               conversation_id=banks_manifest.get('conversation_id'),
                               reason=meta.get('reason'), components=meta['components'],
                               versions=meta['versions'], files=files,
                               request=meta.get('request'), counters=meta.get('counters', {}),
                               migration=meta.get('migration'))
        write_json(path / 'generation.json', generation_meta)
        target = self.generations_dir / name
        if target.exists():
            raise PreassemblyError('generation directory already exists: ' + name)
        os.replace(path, target)
        staging.published = True
        self._dedupe_against_previous(target, sequence, files)
        record = dict(schema=POINTER_SCHEMA, layout_version=LAYOUT_VERSION, sequence=sequence,
                      directory='generations/' + name,
                      generation_sha256=sha256_file(target / 'generation.json'),
                      kind=generation_meta['kind'], published_at=now_iso(),
                      conversation_id=generation_meta['conversation_id'],
                      parameter_binding=banks_manifest.get('parameter_binding'),
                      optimizer_steps=banks_manifest.get('optimizer_steps'),
                      request_id=_snapshot_request_id(generation_meta))
        write_json_atomic(self.pointer_path, record)
        return Generation(self, target, generation_meta)

    def _dedupe_against_previous(self, directory, sequence, files):
        """Share byte-identical payloads with the previous generation (never the pointer)."""
        if sequence <= 1:
            return
        previous = self.pointer()
        if previous is None:
            return
        source = self.root / previous['directory']
        if not source.is_dir():
            return
        old_manifest = read_json(source / 'generation.json')
        for name, row in files.items():
            old = old_manifest['files'].get(name)
            if old is None or old['sha256'] != row['sha256'] or old['bytes'] != row['bytes']:
                continue
            new, previous_file = directory / name, source / name
            if not previous_file.is_file():
                continue
            new.unlink()
            os.link(previous_file, new)

    def recover(self, *, max_staging_age=3600.0):
        """Remove abandoned staging work and stale temporary pointer files."""
        report = dict(removed_staging=[], kept_staging=[], removed_temporaries=[], pointer=None)
        if self.staging_dir.exists():
            now = time.time()
            for path in sorted(self.staging_dir.iterdir()):
                if not path.is_dir():
                    continue
                if now - path.stat().st_mtime < max_staging_age:
                    report['kept_staging'].append(path.name)
                    continue
                shutil.rmtree(path, ignore_errors=True)
                report['removed_staging'].append(path.name)
        for path in sorted(self.root.glob('CURRENT.json.tmp-*')):
            path.unlink(missing_ok=True)
            report['removed_temporaries'].append(path.name)
        try:
            report['pointer'] = self.pointer()
        except PreassemblyError as exc:
            report['pointer_error'] = str(exc)
        return report

    # -- summary -----------------------------------------------------------
    def summary(self):
        record = self.pointer()
        current = None
        if record is not None:
            try:
                generation = self.current()
                current = dict(sequence=generation.sequence, kind=generation.kind,
                               created_at=generation.manifest.get('created_at'),
                               directory=generation.directory.name)
            except PreassemblyError as exc:
                current = dict(error=str(exc))
        generations = []
        if self.generations_dir.exists():
            for directory in sorted(self.generations_dir.iterdir()):
                if (directory / 'generation.json').is_file():
                    generations.append(dict(directory=directory.name,
                                            storage=storage_report(directory)))
        live = self.live_request()
        return dict(schema=LAYOUT_SCHEMA, layout_version=LAYOUT_VERSION, root=str(self.root),
                    pointer=record, current=current, generations=generations,
                    live_request=(None if live is None
                                  else live.plan(current_sequence=self.sequence())),
                    archived_requests=len(self.archived_requests()),
                    storage=storage_report(self.root))


def _snapshot_request_id(generation_meta):
    request = generation_meta.get('request')
    return request.get('request_id') if isinstance(request, dict) else None


# --------------------------------------------------------------------------- #
# source / default resolution
# --------------------------------------------------------------------------- #

def source_root():
    """The candidate source tree that owns this module."""
    return Path(__file__).resolve().parents[1]


def source_versions(root=None):
    root = Path(root or source_root())
    return {name: sha256_file(root / name) for name in SOURCE_FILES if (root / name).is_file()}


def model_directory(root=None, *, model=None):
    root = Path(root or source_root())
    if model is not None:
        return Path(model)
    pointer = root / MODEL_POINTER
    if pointer.is_file():
        record = read_json(pointer)
        return (root / record['directory']).resolve()
    return root / DEFAULT_MODEL_NAME


def model_manifest(directory):
    return read_json(Path(directory) / 'candidate-manifest.json')


def weights_compatibility(directory=None, *, root=None):
    """Does the saved weight manifest still describe this exact source tree?

    Candidate.restore requires manifest['code'] == code_hashes().  That gate is
    reproduced here without importing the 4B body, so a drifted tree is reported
    as a concrete file list instead of an opaque restore failure.
    """
    root = Path(root or source_root())
    directory = Path(directory or model_directory(root))
    try:
        manifest = model_manifest(directory)
    except (OSError, PreassemblyError, ValueError) as exc:
        return dict(state='unavailable', model_directory=str(directory), reason=str(exc),restore_gate='would_refuse')
    declared = manifest.get('code') or {}
    current = source_versions(root)
    missing = sorted(name for name in declared if not (root / name).is_file())
    differing = sorted(name for name in declared
                       if (root / name).is_file() and sha256_file(root / name) != declared[name])
    added = sorted(name for name in current if name not in declared)
    asset_file = root / 'assets' / 'asset-manifest.json'
    assets_now = sha256_file(asset_file) if asset_file.is_file() else None
    equal = not missing and not differing and not added and manifest.get('assets_sha256') == assets_now
    return dict(state='equal' if equal else 'differs', model_directory=str(directory),
                schema=manifest.get('schema'), parameter_binding=manifest.get('parameter_binding'),
                optimizer_steps=manifest.get('optimizer_steps'),
                encoding_sha256=manifest.get('encoding_sha256'),
                declared_code=len(declared), present_code=len(current),
                missing_from_tree=missing,
                differing=[dict(file=name, declared=declared[name],
                                present=sha256_file(root / name)) for name in differing],
                added_to_tree=added, assets_sha256=manifest.get('assets_sha256'),
                assets_now=assets_now, restore_gate='equal' if equal else 'would_refuse',
                note=('Candidate.restore accepts this manifest' if equal else
                      'Candidate.restore compares manifest[code] to code_hashes(); the tree must '
                      'be re-saved under the assembled source before the default entry can restore'))


def default_paths(root=None, *, model=None, store=None):
    """Resolve the default entry: this source tree, its current weights, one store."""
    root = Path(root or source_root())
    model_dir = model_directory(root, model=model)
    store_dir = Path(store) if store else root / DEFAULT_STORE_NAME
    return dict(source_root=str(root), model=str(model_dir), store=str(store_dir),
                model_exists=model_dir.is_dir(),
                model_manifest=(model_dir / 'candidate-manifest.json').is_file(),
                store_exists=store_dir.is_dir(),
                weights=weights_compatibility(model_dir, root=root),
                source=source_versions(root))


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #

def _process_memory():
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                        ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        if not ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            raise OSError('GetProcessMemoryInfo failed')
        return dict(state='measured', working_set_bytes=int(counters.WorkingSetSize),
                    peak_working_set_bytes=int(counters.PeakWorkingSetSize),
                    pagefile_bytes=int(counters.PagefileUsage))
    except Exception as exc:  # noqa: BLE001 - a platform probe reports, never raises
        return dict(state='unavailable', reason='%s: %s' % (type(exc).__name__, exc))


def resource_state(root, *, model_loaded=False):
    state = dict(process=_process_memory())
    store_dir = Path(root)
    existing = store_dir
    while not existing.exists() and existing.parent != existing:
        existing = existing.parent
    try:
        usage = shutil.disk_usage(str(existing))
        state['disk'] = dict(state='measured', probe_path=str(existing), total_bytes=usage.total,
                             free_bytes=usage.free, used_bytes=usage.used)
    except OSError as exc:
        state['disk'] = dict(state='unavailable', reason=str(exc))
    generations = store_dir / 'generations'
    state['store'] = dict(storage_report(store_dir),
                          generations=(len(list(generations.iterdir()))
                                       if generations.exists() else 0))
    if model_loaded:
        try:
            import torch
            available = torch.cuda.is_available()
            state['gpu'] = dict(state='loaded', device='cuda' if available else 'cpu',
                                allocated_bytes=int(torch.cuda.memory_allocated()) if available else None,
                                reserved_bytes=int(torch.cuda.memory_reserved()) if available else None)
        except Exception as exc:  # noqa: BLE001
            state['gpu'] = dict(state='unavailable', reason=str(exc))
    else:
        state['gpu'] = dict(state='not_loaded', reason='diagnostics ran without loading the body')
    return state


def task_progress(providers):
    provider = (providers or {}).get('tasks')
    if provider is None:
        return dict(state='absent',
                    reason='no task-store provider configured; pass providers={"tasks": store} '
                           '(the product task/material layer installs it)')
    try:
        progress = provider.progress() if hasattr(provider, 'progress') else None
    except Exception as exc:  # noqa: BLE001
        return dict(state='error', reason='%s: %s' % (type(exc).__name__, exc))
    if progress is None:
        return dict(state='present', detail='provider exposes no progress() summary')
    return dict(state='present', progress=progress)


def paging_progress(providers):
    provider = (providers or {}).get('paging')
    if provider is None:
        return dict(state='absent',
                    reason='no paged-experience store configured; pass providers={"paging": store}')
    try:
        progress = provider.progress() if hasattr(provider, 'progress') else None
    except Exception as exc:  # noqa: BLE001
        return dict(state='error', reason='%s: %s' % (type(exc).__name__, exc))
    return dict(state='present', progress=progress)


def material_progress(providers):
    provider = (providers or {}).get('materials')
    if provider is None:
        return dict(state='absent',
                    reason='no material provider configured; pass providers={"materials": store}')
    try:
        progress = provider.progress() if hasattr(provider, 'progress') else None
    except Exception as exc:  # noqa: BLE001
        return dict(state='error', reason='%s: %s' % (type(exc).__name__, exc))
    return dict(state='present', progress=progress)


def startup_diagnostics(layout, *, providers=None, generation=None, live_journal=None,
                        model=None, model_loaded=False, compatibility=None, extra=None):
    """Everything the product shows at startup, computed from durable state."""
    report = dict(schema=DIAGNOSTICS_SCHEMA, at=now_iso(), state='ok')
    try:
        current = generation if generation is not None else layout.current()
    except PreassemblyError as exc:
        return dict(schema=DIAGNOSTICS_SCHEMA, at=now_iso(), state='broken', error=str(exc),
                    store=layout.summary(), resources=resource_state(layout.root))
    report['store'] = dict(root=str(layout.root), pointer=layout.pointer(),
                           archived_requests=len(layout.archived_requests()))
    report['model'] = (dict(path=str(model), loaded=bool(model_loaded)) if model is not None
                       else dict(path=None, loaded=False,
                                 reason='no model directory supplied to diagnostics'))
    report['task_progress'] = task_progress(providers)
    report['material_progress'] = material_progress(providers)
    report['paged_experience'] = paging_progress(providers)
    report['resources'] = resource_state(layout.root, model_loaded=model_loaded)
    if current is None:
        report.update(generation=None, active_memory=dict(state='empty'),
                      persistent_experiences=dict(records=0, pages=0),
                      request=dict(state='idle', resumable=False),
                      versions=dict(sources=len(source_versions()), layout_version=LAYOUT_VERSION,
                                    compatibility=compatibility))
        if extra:
            report['extra'] = extra
        return report
    banks = current.banks_manifest()
    index = current.index()
    four = banks.get('four_banks') or {}
    report['generation'] = dict(sequence=current.sequence, kind=current.kind,
                                created_at=current.manifest.get('created_at'),
                                conversation_id=banks.get('conversation_id'),
                                parameter_binding=banks.get('parameter_binding'),
                                optimizer_steps=banks.get('optimizer_steps'),
                                files=len(current.manifest['files']),
                                storage=storage_report(current.directory))
    report['active_memory'] = dict(native_tokens=four.get('native_tokens'),
                                   window=four.get('window'), capacity=four.get('capacity'),
                                   observations=four.get('observations'),
                                   mental_records=four.get('mental_records'),
                                   cognitive_ticks=four.get('cognitive_ticks'),
                                   writes=four.get('writes'),
                                   episode_code_tensors=four.get('episode_code_tensors'),
                                   mental_code_tensors=four.get('mental_code_tensors'))
    report['persistent_experiences'] = dict(
        records=index['counts']['records'], pages=len(index['pages']),
        observations=index['counts']['observations'],
        mental_records=index['counts']['mental_records'], page_size=index['page_size'],
        journal_sha256=index.get('journal_sha256'),
        archived_requests=len(layout.archived_requests()),
        request_records=current.components.get('request', {}).get('records'))
    journal = live_journal if live_journal is not None else layout.live_request()
    if journal is None:
        report['request'] = dict(state='idle', resumable=False)
    else:
        plan = journal.plan(current_sequence=layout.sequence())
        report['request'] = dict(state='resumable' if plan['resumable'] else 'finished',
                                 request_id=plan['request_id'], status=plan['status'],
                                 emitted_characters=plan['emitted_characters'],
                                 delivered_characters=plan['delivered_characters'],
                                 executed_actions=len(plan['executed_actions']),
                                 next_step=plan['next_step'])
    report['versions'] = dict(layout_version=LAYOUT_VERSION, saved=current.versions,
                              current_sources=len(source_versions()),
                              compatibility=compatibility)
    if extra:
        report['extra'] = extra
    return report


# --------------------------------------------------------------------------- #
# banks provider (the real save path)
# --------------------------------------------------------------------------- #

class PagingStoreComponent:
    """Reference a paged experience store in the same generation.

    The store (treyrin.paging_store) keeps its own shards and incremental base;
    the generation records the store's snapshot descriptor, its verified counts
    and the store root, so a reader can tell exactly which paged state the
    generation belongs to without copying the whole library on every save.
    """

    name = 'paging'
    schema = 'treyrin-paged-store-v1'

    def __init__(self, store):
        self.store = store

    def save_component(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        stats = self.store.stats()
        check = self.store.check(verify_payloads=False, deep=False)
        try:
            snapshot = self.store.snapshot()
            snapshot_path = 'snapshot'
        except Exception as exc:  # noqa: BLE001 - reported, never hidden
            snapshot = dict(state='unavailable', reason='%s: %s' % (type(exc).__name__, exc))
            snapshot_path = 'snapshot-error'
        reference = dict(schema=self.schema, root=str(stats.get('root') or getattr(self.store, 'root', '')),
                         at=now_iso(), stats=stats, check=check, snapshot=snapshot,
                         snapshot_source=snapshot_path)
        write_json(path / 'REFERENCE.json', reference)
        if not check.get('ok', False):
            raise PreassemblyError('paged experience store failed its own check: '
                                   + canonical(check.get('errors'))[:500])
        return dict(schema=self.schema, state='present', root=str(reference['root']),
                    records=check.get('records'), pages=check.get('pages'),
                    units=check.get('units'), segments=check.get('segments'),
                    snapshot_generation=(snapshot.get('generation') if isinstance(snapshot, dict) else None),
                    snapshot=snapshot_path, reference='paging/REFERENCE.json',
                    verified=True, file='paging/REFERENCE.json')

    def progress(self):
        stats = self.store.stats()
        return dict(schema=stats.get('schema'), experiences=stats.get('experiences'),
                    pages=stats.get('pages'), units=stats.get('units'),
                    sequence=stats.get('sequence'),
                    resident=len(stats.get('resident') or ()), config=stats.get('config'))


class TaskStoreComponent:
    """Checkpoint the product task store into the same generation."""

    name = 'tasks'
    schema = 'treyrin-task-store-checkpoint-v1'

    def __init__(self, store):
        self.store = store

    def save_component(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        target = path / 'task-store.checkpoint.json'
        result = self.store.checkpoint(target)
        if not target.is_file():
            raise PreassemblyError('task store checkpoint did not produce a file')
        verification = self.store.verify() if hasattr(self.store, 'verify') else None
        if isinstance(verification, dict) and verification.get('ok') is False:
            raise PreassemblyError('task store failed verification before the commit')
        return dict(schema=self.schema, state='present',
                    checkpoint=target.name, checkpoint_sha256=sha256_file(target),
                    state_sha256=(result or {}).get('state_sha256'),
                    journal_seq=(result or {}).get('journal_seq'),
                    chain=(result or {}).get('chain'),
                    counts=(result or {}).get('counts'),
                    verified=(None if verification is None else verification.get('ok')))

    def progress(self):
        report = self.store.report()
        return dict(schema=report.get('schema'), counts=report.get('counts'),
                    capacity=report.get('capacity'), tasks=len(report.get('tasks') or ()),
                    state_sha256=report.get('state_sha256'),
                    projection=report.get('projection'))


class MaterialIntakeComponent:
    """Record material-intake progress and checkpoint in the same generation."""

    name = 'materials'
    schema = 'treyrin-material-intake-v1'

    def __init__(self, session):
        self.session = session

    def save_component(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        status = self.session.status()
        inspection = (self.session.inspection_report()
                      if hasattr(self.session, 'inspection_report') else None)
        record = dict(schema=self.schema, at=now_iso(), status=status, inspection=inspection)
        checkpoint = self.session.checkpoint() if hasattr(self.session, 'checkpoint') else None
        record['checkpoint'] = checkpoint if isinstance(checkpoint, (dict, str, int, float, bool,
                                                                    type(None))) else str(checkpoint)
        source = None
        if isinstance(checkpoint, dict):
            source = checkpoint.get('path') or checkpoint.get('checkpoint')
        if isinstance(source, str) and Path(source).is_file():
            destination = path / 'intake-checkpoint.json'
            shutil.copyfile(source, destination)
            record['checkpoint_file'] = destination.name
            record['checkpoint_sha256'] = sha256_file(destination)
        write_json(path / 'STATUS.json', record)
        counts = (status or {}).get('counts') if isinstance(status, dict) else None
        return dict(schema=self.schema, state='present', counts=counts,
                    checkpoint_file=record.get('checkpoint_file'),
                    checkpoint_sha256=record.get('checkpoint_sha256'),
                    status='materials/STATUS.json')

    def progress(self):
        status = self.session.status()
        return dict(schema=status.get('schema'), counts=status.get('counts'),
                    directory=status.get('root') or status.get('directory'),
                    entries=len(status.get('entries') or ()) if isinstance(status, dict) else None)


class ResidentBanksProvider:
    """Banks/profile/optimizer component, written by the existing resident writer.

    ResidentStore.save_generation is the frozen single-generation save: four
    banks, output profile and profile moments, with cross-generation and
    cross-candidate checks.  This provider writes one such generation into
    staging/banks and adopts it; it deliberately does not reimplement any of its
    payload formats or checks.
    """

    name = 'banks'
    schema = BANKS_SCHEMA

    def save(self, staging_path, candidate, conversation):
        import resident_store
        staging_path = Path(staging_path)
        store = resident_store.ResidentStore(staging_path)
        record = store.save_generation(candidate, conversation)
        produced = staging_path / record['directory']
        target = staging_path / 'banks'
        os.replace(produced, target)
        for leftover in ('generations', 'CURRENT-CONVERSATION.json'):
            path = staging_path / leftover
            if path.is_dir():
                shutil.rmtree(path)
            elif path.is_file():
                path.unlink()
        manifest = read_json(target / 'manifest.json')
        entry = dict(schema=BANKS_SCHEMA, directory='banks', manifest='banks/manifest.json',
                     manifest_sha256=sha256_file(target / 'manifest.json'),
                     sequence=manifest.get('sequence'),
                     conversation_id=manifest.get('conversation_id'),
                     conversation_kind=manifest.get('conversation_kind'),
                     parameter_binding=manifest.get('parameter_binding'),
                     optimizer_steps=manifest.get('optimizer_steps'),
                     four_banks=manifest.get('four_banks'),
                     files=len(manifest.get('files') or {}),
                     role_state=manifest.get('role_state_sha256') is not None)
        return dict(entry=entry, manifest=manifest, directory=target)

    def restore(self, generation, candidate, *, window=8192, capacity=2048):
        import resident_store
        from role_conversation import RoleConversationSession
        banks = generation.banks
        manifest = read_json(banks / 'manifest.json')
        if manifest.get('schema') != BANKS_SCHEMA:
            raise PreassemblyError('banks component has an unknown schema')
        conversation = RoleConversationSession.restore(banks / 'conversation', candidate)
        if manifest.get('parameter_binding') != candidate.model._binding:
            raise PreassemblyError('banks belong to different restored parameters')
        resident_store.ResidentStore(banks).check_generation(
            dict(schema=BANKS_SCHEMA, sequence=manifest['sequence'],
                 conversation_id=manifest['conversation_id']),
            banks, candidate, conversation)
        return conversation


# --------------------------------------------------------------------------- #
# the session
# --------------------------------------------------------------------------- #

class PreassemblySession:
    """One durable conversation: banks + pages + request WAL + one pointer.

    open restores the last committed generation (or starts a new one over the
    already restored base weights) and reports whether a live request is
    resumable.  turn writes the request intent before generating, records the
    produced output, and only then commits the next generation.
    """

    def __init__(self, layout, *, candidate=None, conversation=None, provider=None,
                 providers=None, root=None):
        self.layout = layout
        self.candidate = candidate
        self.conversation = conversation
        self.provider = provider or ResidentBanksProvider()
        self.providers = dict(providers or {})
        self.root = Path(root) if root is not None else source_root()
        self.generation = None
        self.restored = False
        self.compatibility = None

    # -- opening -----------------------------------------------------------
    @classmethod
    def open(cls, root, *, candidate=None, create=True, providers=None, window=8192,
             capacity=2048, self_info=True, strict=True, provider=None, source=None):
        layout = Layout(root)
        layout.prepare()
        session = cls(layout, candidate=candidate, provider=provider, providers=providers,
                      root=source)
        current = layout.current()
        if current is None:
            if not create:
                raise PreassemblyError('store has no committed generation: ' + str(layout.root))
            if candidate is None:
                raise PreassemblyError('starting a session requires the restored candidate')
            session.conversation = _new_conversation(candidate, window, capacity, self_info)
            session._reconcile()
            return session
        # Full bank hashing is done by the resident store check_generation during
        # restore; here sizes, membership and cross-generation agreement are
        # checked first so a torn store fails before the body is touched.
        current.verify(deep=False)
        session.compatibility = current.compatibility(root=session.root, candidate=candidate,
                                                      strict=bool(strict and candidate is not None))
        if current.kind != LIVE_KIND:
            raise PreassemblyError('generation kind %r cannot be resumed as a live conversation'
                                   % (current.kind,))
        if candidate is not None:
            session.conversation = session.provider.restore(current, candidate,
                                                           window=window, capacity=capacity)
            # The restore itself compares the saved profile with the restored
            # parameters, so a successful restore is the binding verification.
            if session.compatibility is not None:
                session.compatibility['checks']['parameter_binding'] = True
        session.generation = current
        session.restored = True
        session._reconcile()
        return session

    def _reconcile(self):
        """Archive a request that completed but whose process died before archiving."""
        journal = self.layout.live_request()
        if journal is not None and journal.status == 'completed':
            journal.archive(self.layout.requests_dir)

    # -- request lifecycle -------------------------------------------------
    def begin_request(self, text, *, request_id=None, event_id=None, known_at=None,
                      source='human_user'):
        """Durably accept a request before anything is generated."""
        if self.conversation is None:
            raise PreassemblyError('no conversation is loaded')
        existing = self.layout.live_request()
        if existing is not None:
            raise PreassemblyError('a live request already exists (%s); resume or cancel it first'
                                   % existing.intent['request_id'])
        request_id = request_id or 'request-' + uuid.uuid4().hex
        event_id = event_id or 'user-' + uuid.uuid4().hex
        known_at = known_at or now_iso()
        conversation_id = (self.generation.manifest.get('conversation_id')
                           if self.generation is not None else None)
        RequestJournal.begin(self.layout.request_path, request_id=request_id, text=text,
                             event_id=event_id, known_at=known_at, source=source,
                             banks_sequence=self.layout.sequence(),
                             conversation_id=conversation_id)
        journal = RequestJournal(self.layout.request_path)
        # The user turn is committed in memory only: the WAL above is what makes
        # the request durable.  A crash here is repaired by replaying the same
        # event_id, which the observation journal treats as an exact replay.
        self.conversation.user_turn(text, event_id=event_id, known_at=known_at)
        return dict(request_id=request_id, event_id=event_id, known_at=known_at,
                    live_directory=str(journal.directory), status='pending')

    def journal(self):
        journal = self.layout.live_request()
        if journal is None:
            raise PreassemblyError('no live request journal')
        return journal

    def emit(self, text, *, token_id=None, delivered=True, finish_reason=None):
        """Record output already produced (or already shown to the user)."""
        journal = self.journal()
        journal.append_output(text, token_ids=None if token_id is None else [token_id],
                              delivered=delivered, finish_reason=finish_reason)
        return journal.output()

    def respond(self, *, max_new_tokens=None, replay_budget=4, delivered=True, note=None,
                supersedes_attempt=None,on_token=None,should_cancel=None):
        journal = self.journal()
        attempt = int(journal.state.get('attempts') or 0) + 1
        journal.set_status('generating', attempts=attempt)
        started = time.monotonic()
        first_token=None
        def progress(token,tokens):
            nonlocal first_token
            elapsed=time.monotonic()-started
            if first_token is None:first_token=elapsed
            append_jsonl(journal.directory/'TOKENS.jsonl',dict(index=len(tokens)-1,token_id=int(token),
                attempt=attempt,seconds=elapsed))
            if on_token is not None:on_token(token,tokens)
        import inspect
        options=dict(max_new_tokens=max_new_tokens,replay_budget=replay_budget)
        if 'on_token' in inspect.signature(self.conversation.respond).parameters:
            options.update(on_token=progress,should_cancel=should_cancel or (lambda:journal.status=='cancelled'))
        result = self.conversation.respond(**options)
        journal.append_output(result['answer'], token_ids=result.get('answer_token_ids'),
                              finish_reason=result.get('finish_reason'), delivered=delivered,
                              note=note, supersedes_attempt=supersedes_attempt)
        interrupted=result.get('finish_reason')=='cancelled'
        journal.set_status('cancelled' if interrupted else 'generated', last_finish_reason=result.get('finish_reason'),
                           seconds=round(time.monotonic() - started, 3))
        request_id = journal.intent['request_id']
        committed = self.commit(reason='interrupted' if interrupted else 'answered')
        if not interrupted:
            journal.set_status('completed')
            journal.archive(self.layout.requests_dir)
        return dict(answer=result['answer'], finish_reason=result.get('finish_reason'),
                    diagnostics=result.get('diagnostics'), request_id=request_id,
                    generation=committed.sequence, status='interrupted' if interrupted else 'completed',
                    timing=dict(first_token_seconds=first_token,completion_seconds=time.monotonic()-started))

    def turn(self, text, *, max_new_tokens=None, replay_budget=4, request_id=None):
        self.begin_request(text, request_id=request_id)
        return self.respond(max_new_tokens=max_new_tokens, replay_budget=replay_budget)

    def cancel_request(self, reason='cancelled by user'):
        journal = self.journal()
        journal.set_status('cancelled', reason=reason)
        return journal.plan(current_sequence=self.layout.sequence())

    def discard_request(self, reason='discarded without resuming'):
        """Put a resumable request aside: archived with its output, never resumed."""
        journal = self.journal()
        journal.set_status('failed', reason=reason, abandoned=True)
        return journal.archive(self.layout.requests_dir)

    def close(self, reason='process exit'):
        """Leave an in-flight request resumable instead of silently dropping it."""
        journal = self.layout.live_request()
        if journal is None:
            return dict(state='idle', resumable=False)
        if journal.status in UNFINISHED_STATES:
            journal.set_status('interrupted', reason=reason)
        return journal.plan(current_sequence=self.layout.sequence())

    def resume_plan(self):
        journal = self.layout.live_request()
        if journal is None:
            return dict(resumable=False, state='idle')
        plan = journal.plan(current_sequence=self.layout.sequence())
        plan['state'] = 'resumable' if plan['resumable'] else 'finished'
        plan['requires_regeneration'] = plan['resumable']
        plan['supersedes'] = plan['emitted_text'] if plan['output_recorded'] else None
        return plan

    def resume(self, *, max_new_tokens=None, replay_budget=4,on_token=None):
        """Continue the same request identity; an earlier attempt is recorded, not hidden."""
        journal = self.journal()
        plan = journal.plan(current_sequence=self.layout.sequence())
        if not plan['resumable']:
            raise PreassemblyError('the live request is not resumable')
        intent = journal.intent
        previous_attempt = int(journal.state.get('attempts') or 0)
        resumed = int(journal.state.get('resumed') or 0) + 1
        journal.set_status('pending', resumed=resumed, attempts=previous_attempt)
        pending=getattr(self.conversation,'pending_user',None)
        if pending is None:
            self.conversation.user_turn(intent['text'], event_id=intent['event_id'],known_at=intent['known_at'])
        elif pending['event_id']!=intent['event_id']:raise PreassemblyError('Pending role request identity differs')
        sampled=read_jsonl(journal.directory/'TOKENS.jsonl')
        if sampled and getattr(self.conversation,'pending_user',None) is not None:
            by_index={int(row['index']):int(row['token_id']) for row in sampled}
            if sorted(by_index)!=list(range(len(by_index))):raise PreassemblyError('Incomplete saved output prefix')
            self.conversation.pending_user['generated_tokens']=[by_index[i] for i in range(len(by_index))]
        note = None
        if plan['output_recorded']:
            note = ('an earlier attempt recorded %d characters before interruption; it was not '
                    'committed to the banks' % plan['emitted_characters'])
        result = self.respond(max_new_tokens=max_new_tokens, replay_budget=replay_budget,
                              note=note,on_token=on_token,
                              supersedes_attempt=(previous_attempt if plan['output_recorded']
                                                  else None))
        result.update(resumed=resumed, previous_attempts=previous_attempt,
                      superseded_characters=plan['emitted_characters'],
                      request_event_id=intent['event_id'])
        return result

    # -- guarded side effects ---------------------------------------------
    def execute(self, action_id, kind, operation, *, payload=None):
        """Run one side effect at most once per action identity."""
        journal = self.layout.live_request()
        payload_sha = sha256_value(payload) if payload is not None else None
        if journal is not None:
            prior = journal.action(action_id)
            if prior is not None and prior.get('state') == 'executed':
                return dict(reused=True, action_id=action_id, kind=kind,
                            receipt_sha256=prior.get('receipt_sha256'),
                            result=prior.get('receipt'))
            journal.begin_action(action_id, kind, payload_sha256=payload_sha)
        try:
            result = operation()
        except Exception as exc:
            if journal is not None:
                journal.fail_action(action_id, exc)
            raise
        if journal is not None:
            journal.record_action(action_id, kind, receipt=result, value_sha256=payload_sha)
        return dict(reused=False, action_id=action_id, kind=kind, result=result)

    def tick(self, payload):
        """Cognitive opportunity with the action identity bound to operation_id."""
        operation_id = payload.get('operation_id')
        if not operation_id:
            raise PreassemblyError('a cognitive tick needs an operation_id identity')
        outcome = self.execute(operation_id, 'cognitive-tick',
                               lambda: self.conversation.tick(payload), payload=payload)
        if outcome['reused']:
            return dict(state='already_executed', action_id=operation_id,
                        receipt_sha256=outcome['receipt_sha256'], result=outcome['result'])
        return outcome['result']

    # -- committing --------------------------------------------------------
    def commit(self, reason='manual'):
        """Write the next generation: banks, pages, request snapshot, one pointer."""
        if self.conversation is None:
            raise PreassemblyError('no conversation to commit')
        staging = self.layout.staging()
        with staging:
            banks = self.provider.save(staging.path, self.candidate, self.conversation)
            journal = self.layout.live_request()
            request_entry = self._write_request(staging, journal)
            index_entry = self._write_experience(staging, banks)
            paging_entry = self._write_paging(staging)
            tasks_entry = self._write_tasks(staging)
            materials_entry = self._write_materials(staging)
            diagnostics_entry = self._write_diagnostics(staging, banks, index_entry)
            components = dict(
                banks=banks['entry'],
                parameters=dict(schema='treyrin-output-profile-v1',
                                directory='banks',
                                profile_sha256=banks['manifest'].get('profile_sha256'),
                                moments_sha256=banks['manifest'].get('profile_moments_sha256'),
                                tensors=list(banks['manifest'].get('profile_tensors') or ()),
                                files=['profile.safetensors', 'profile-moments.pt']),
                optimizer=dict(schema='treyrin-learning-checkpoint-v2', directory='banks',
                               optimizer_steps=banks['manifest'].get('optimizer_steps'),
                               training_state='banks/conversation/core-native/core/session/'
                                              'learning/local-training-state.pt',
                               trainable='banks/conversation/core-native/core/session/'
                                         'learning/trainable.safetensors'),
                experience=index_entry, paging=paging_entry, request=request_entry,
                tasks=tasks_entry, materials=materials_entry,
                diagnostics=diagnostics_entry)
            generation = self.layout.publish(staging, dict(
                kind=LIVE_KIND, reason=reason, components=components,
                versions=self._versions(banks), request=request_entry,
                counters=dict(four_banks=banks['manifest'].get('four_banks'),
                              optimizer_steps=banks['manifest'].get('optimizer_steps'))))
        self.generation = generation
        return generation

    def _write_request(self, staging, journal):
        target = staging.path / 'request'
        if journal is not None:
            snapshot = journal.snapshot(target)
            return dict(schema=REQUEST_SNAPSHOT_SCHEMA, directory='request', state='present',
                        request_id=snapshot['request_id'], status=snapshot['status'],
                        records=snapshot['output']['characters'],
                        executed_actions=len(snapshot['executed_actions']),
                        snapshot_sha256=snapshot['snapshot_sha256'], files=len(snapshot['files']))
        target.mkdir(parents=True, exist_ok=False)
        placeholder = dict(schema=REQUEST_SNAPSHOT_SCHEMA, directory='request', state='idle',
                           request_id=None, status='idle', records=0, executed_actions=0,
                           snapshot_sha256=None, files=0)
        write_json(target / 'SNAPSHOT.json', placeholder)
        return placeholder

    def _write_experience(self, staging, banks):
        session = self.conversation.session
        document = dict(observations=[asdict(record) for record in session.observations],
                        mental_records=[asdict(record) for record in session.mental_records],
                        cognitive_ticks=[asdict(record) for record in session.cognitive_ticks],
                        operations=[list(item) for item in session.operations])
        session_manifest = read_json(banks['directory'] / 'conversation' / 'core-native'
                                     / 'core' / 'session' / 'manifest.json')
        index, pages = build_index(document, page_size=PAGE_SIZE,
                                   journal_sha256=session_manifest.get('journal_sha256'))
        written = write_index(staging.path / 'experience', index, pages)
        return dict(schema=INDEX_SCHEMA, directory='experience', index='experience/INDEX.json',
                    index_sha256=written['index_sha256'], records=written['records'],
                    pages=written['pages'], page_size=PAGE_SIZE,
                    journal_sha256=index.get('journal_sha256'), counts=index['counts'])

    def _write_tasks(self, staging):
        provider = self.providers.get('tasks')
        if provider is None:
            return dict(schema='treyrin-task-store-v1', directory=None, state='absent',
                        reason='no task-store provider configured; pass providers={"tasks": ...}')
        target = staging.path / 'tasks'
        target.mkdir(parents=True, exist_ok=False)
        if hasattr(provider, 'save_component'):
            detail = provider.save_component(target) or {}
        elif hasattr(provider, 'save'):
            detail = provider.save(target) or {}
        else:
            raise PreassemblyError('task provider must expose save_component(path) or save(path)')
        entry = dict(detail, schema=detail.get('schema', 'treyrin-task-store-v1'),
                     directory='tasks', state='present', files=len(walk_files(target)),
                     manifest_sha256=sha256_value(inventory(target)),
                     progress=(provider.progress() if hasattr(provider, 'progress') else None))
        entry.pop('schema_of_store', None)
        return entry

    def _write_paging(self, staging):
        provider = self.providers.get('paging')
        if provider is None:
            return dict(schema='treyrin-paged-store-v1', directory=None, state='absent',
                        reason='no paged-experience store configured; pass providers={"paging": store}')
        target = staging.path / 'paging'
        detail = provider.save_component(target)
        return dict(detail, directory='paging')

    def _write_materials(self, staging):
        provider = self.providers.get('materials')
        if provider is None:
            return dict(schema='treyrin-material-intake-v1', directory=None, state='absent',
                        reason='no material intake configured; pass providers={"materials": session}')
        target = staging.path / 'materials'
        detail = provider.save_component(target)
        return dict(detail, directory='materials')

    def _write_diagnostics(self, staging, banks, index_entry):
        target = staging.path / 'diagnostics'
        target.mkdir(parents=True, exist_ok=False)
        four = banks['manifest'].get('four_banks') or {}
        report = dict(schema=DIAGNOSTICS_SCHEMA, at=now_iso(), state='commit-time',
                      generation=dict(sequence=banks['manifest'].get('sequence'),
                                      conversation_id=banks['manifest'].get('conversation_id'),
                                      parameter_binding=banks['manifest'].get('parameter_binding'),
                                      optimizer_steps=banks['manifest'].get('optimizer_steps')),
                      active_memory=dict(native_tokens=four.get('native_tokens'),
                                         observations=four.get('observations'),
                                         mental_records=four.get('mental_records'),
                                         cognitive_ticks=four.get('cognitive_ticks'),
                                         writes=four.get('writes')),
                      persistent_experiences=index_entry.get('counts'),
                      task_progress=task_progress(self.providers),
                      material_progress=material_progress(self.providers),
                      paged_experience=paging_progress(self.providers),
                      resources=resource_state(self.layout.root))
        write_json(target / 'STARTUP.json', report)
        return dict(schema=DIAGNOSTICS_SCHEMA, directory='diagnostics',
                    file='diagnostics/STARTUP.json', state='present')

    def _versions(self, banks):
        manifest = banks['manifest']
        session_manifest = read_json(banks['directory'] / 'conversation' / 'core-native'
                                     / 'core' / 'session' / 'manifest.json')
        candidate = (self.candidate if self.candidate is not None
                     else getattr(self.conversation, 'candidate', None))
        return dict(layout_version=LAYOUT_VERSION, saved_at=now_iso(),
                    parameter_binding=manifest.get('parameter_binding'),
                    encoding_sha256=getattr(candidate, 'encoding_sha', None),
                    state_schema=session_manifest.get('schema'),
                    contract_identity=session_manifest.get('contract_identity'),
                    optimizer_steps=manifest.get('optimizer_steps'),
                    sources=source_versions(self.root))

    # -- reading -----------------------------------------------------------
    def status(self):
        return self.layout.summary()

    def verify(self, *, deep=True):
        if self.generation is None:
            return dict(state='empty', verified=True, problems=[])
        return self.generation.verify(deep=deep)

    def inspect_record(self, address):
        if self.generation is None:
            raise PreassemblyError('no committed generation to read from')
        return self.generation.record(address)

    def diagnostics(self, *, model_loaded=False):
        return startup_diagnostics(self.layout, providers=self.providers,
                                   generation=self.generation,
                                   model=(self.candidate is not None),
                                   model_loaded=model_loaded,
                                   compatibility=self.compatibility)


def _new_conversation(candidate, window, capacity, self_info):
    from role_conversation import RoleConversationSession
    return RoleConversationSession(candidate, window=window, capacity=capacity, self_info=self_info)


# --------------------------------------------------------------------------- #
# API entry point
# --------------------------------------------------------------------------- #

def open_default_session(*, root=None, model=None, store=None, candidate=None, window=8192,
                         capacity=2048, self_info=True, providers=None, strict=True,
                         create=True):
    """The default entry: this source tree, its current weights, one durable store.

    The caller may pass an already restored candidate; otherwise the base
    weights are restored here through the real Candidate.restore path.
    """
    paths = default_paths(root, model=model, store=store)
    if candidate is None:
        from candidate import Candidate
        candidate = Candidate()
        candidate.restore(Path(paths['model']))
    session = PreassemblySession.open(paths['store'], candidate=candidate, window=window,
                                      capacity=capacity, self_info=self_info,
                                      providers=providers, strict=strict, create=create,
                                      source=paths['source_root'])
    return session, paths
