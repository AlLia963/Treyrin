"""D06 integration: same-body natural text -> typed JSON goal proposal.

The 4B body writes the text; this module never does.  Generation is injected as
a callable (prompt -> text), so the exact same validation runs on a CPU stub
during tests and on the 4B body once a GPU order exists.

What the validator does: identity, field types and ranges, evidence existence,
provenance namespace, deadline parsing and revision existence.  What it must
not do: decide what the user wants.  There is no keyword scan, no urgency
inference and no reminder rule; a proposal that says priority 0.05 stays at
0.05 even when its text says "立刻".

Provenance is typed, not asserted: every source kind has its own evidence
namespace (ev-user-, ev-device-, ev-doc-, ev-tool-, ev-sched-, ev-model-).  A
proposal may only claim a source kind if it cites at least one identity from
that namespace.  A model-authored item (source_kind=model_output) must carry
ev-model- evidence and must use pending acceptance, so assistant output can
never be recorded as a user fact.

Every attempt is kept: prompt text, prompt sha256, prompt token count, raw
text, parse error and accepted record all stay in the append-only journal.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime

from .goal_state import ACCEPTANCE, AUTONOMY, SOURCE_KINDS

GOAL_KINDS = ('goal', 'task')
REQUIRED = ('record_id', 'kind', 'text', 'source_kind', 'evidence_ids')
OPTIONAL = ('priority', 'interruption_cost', 'deadline_at', 'confidence',
            'acceptance', 'autonomy', 'revision_of')
EVIDENCE_PREFIX = dict(user_message='ev-user-', device_log='ev-device-',
                       document='ev-doc-', tool_result='ev-tool-',
                       schedule='ev-sched-', model_output='ev-model-')
RECORD_ID = re.compile(r'^goal-[A-Za-z0-9][A-Za-z0-9._:-]{0,58}$')
EVIDENCE_ID = re.compile(r'^ev-[a-z]+-[A-Za-z0-9][A-Za-z0-9._:-]{0,58}$')


class IntakeError(ValueError):
    """Stable rejection code plus the raw payload for the journal."""

    def __init__(self, code, message='', raw=None):
        super().__init__('%s: %s' % (code, message))
        self.code = code
        self.raw = raw


@dataclass(frozen=True)
class GoalProposal:
    record_id: str
    kind: str
    text: str
    source_kind: str
    evidence_ids: tuple
    priority: float = 0.5
    interruption_cost: float = 0.1
    deadline_at: str | None = None
    confidence: float = 0.5
    acceptance: str = 'accepted'
    autonomy: str = 'notify_only'
    revision_of: str | None = None

    def as_dict(self):
        return dict(record_id=self.record_id, kind=self.kind, text=self.text,
                    source_kind=self.source_kind, evidence_ids=list(self.evidence_ids),
                    priority=self.priority, interruption_cost=self.interruption_cost,
                    deadline_at=self.deadline_at, confidence=self.confidence,
                    acceptance=self.acceptance, autonomy=self.autonomy,
                    revision_of=self.revision_of)


def _number(data, name, default, raw):
    value = data.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IntakeError('p.field-invalid', '%s must be a number' % name, raw)
    if not 0.0 <= float(value) <= 1.0:
        raise IntakeError('p.field-invalid', '%s must be in [0,1]' % name, raw)
    return float(value)


def parse_goal_proposal(raw, *, known_evidence, known_records=(), now=None):
    """Strict typed parse of one model-authored JSON object."""
    if not isinstance(raw, str) or not raw.strip():
        raise IntakeError('p.empty', 'empty generation', raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise IntakeError('p.not-json', str(exc), raw)
    if not isinstance(data, dict):
        raise IntakeError('p.not-object', 'proposal must be one JSON object', raw)
    unknown = sorted(set(data) - set(REQUIRED) - set(OPTIONAL))
    if unknown:
        raise IntakeError('p.unknown-field', ','.join(unknown), raw)
    missing = [name for name in REQUIRED if name not in data]
    if missing:
        raise IntakeError('p.missing-field', ','.join(missing), raw)
    record_id = data['record_id']
    if not isinstance(record_id, str) or not RECORD_ID.match(record_id):
        raise IntakeError('p.field-invalid', 'record_id must be goal-<id>', raw)
    if data['kind'] not in GOAL_KINDS:
        raise IntakeError('p.field-invalid', 'kind must be one of %s' % (GOAL_KINDS,), raw)
    text = data['text']
    if not isinstance(text, str) or not text.strip():
        raise IntakeError('p.field-invalid', 'text must be a nonempty string', raw)
    source_kind = data['source_kind']
    if source_kind not in SOURCE_KINDS:
        raise IntakeError('p.field-invalid', 'unknown source_kind %r' % (source_kind,), raw)
    evidence = data['evidence_ids']
    if not isinstance(evidence, list) or any(not isinstance(e, str) for e in evidence) or not evidence:
        raise IntakeError('p.field-invalid', 'evidence_ids must be a nonempty list of strings', raw)
    if len(set(evidence)) != len(evidence):
        raise IntakeError('p.duplicate-evidence', 'duplicate evidence identity', raw)
    for identity in evidence:
        if not EVIDENCE_ID.match(identity):
            raise IntakeError('p.field-invalid', 'malformed evidence identity %r' % (identity,), raw)
    known = set(known_evidence)
    unknown_evidence = [e for e in evidence if e not in known]
    if unknown_evidence:
        raise IntakeError('p.evidence-unknown', ','.join(unknown_evidence), raw)
    prefix = EVIDENCE_PREFIX[source_kind]
    if not any(e.startswith(prefix) for e in evidence):
        raise IntakeError('p.provenance-mismatch',
                          'source %r needs at least one %s* identity' % (source_kind, prefix), raw)
    priority = _number(data, 'priority', 0.5, raw)
    cost = _number(data, 'interruption_cost', 0.1, raw)
    confidence = _number(data, 'confidence', 0.5, raw)
    acceptance = data.get('acceptance', 'pending' if source_kind == 'model_output' else 'accepted')
    if acceptance not in ACCEPTANCE:
        raise IntakeError('p.field-invalid', 'unknown acceptance', raw)
    if source_kind == 'model_output' and acceptance != 'pending':
        raise IntakeError('p.source-not-fact',
                          'model-authored items need explicit pending acceptance', raw)
    autonomy = data.get('autonomy', 'notify_only')
    if autonomy not in AUTONOMY:
        raise IntakeError('p.field-invalid', 'unknown autonomy', raw)
    revision_of = data.get('revision_of')
    if revision_of is not None:
        if not isinstance(revision_of, str) or not RECORD_ID.match(revision_of):
            raise IntakeError('p.field-invalid', 'revision_of must be a goal record id', raw)
        if revision_of not in set(known_records):
            raise IntakeError('p.revision-unknown', revision_of, raw)
    if acceptance == 'declined' and revision_of is None:
        raise IntakeError('p.acceptance-lifecycle', 'a cancellation must identify an existing commitment', raw)
    deadline = data.get('deadline_at')
    if deadline is not None:
        if not isinstance(deadline, str):
            raise IntakeError('p.field-invalid', 'deadline_at must be an ISO string or null', raw)
        try:
            parsed = datetime.fromisoformat(deadline)
        except ValueError as exc:
            raise IntakeError('p.deadline-format', str(exc), raw)
        if parsed.tzinfo is None:
            raise IntakeError('p.deadline-tz', 'deadline_at needs an explicit UTC offset', raw)
        if now is not None and parsed < now:
            raise IntakeError('p.deadline-past', deadline, raw)
    return GoalProposal(record_id=record_id, kind=data['kind'], text=text, source_kind=source_kind,
                        evidence_ids=tuple(evidence), priority=priority,
                        interruption_cost=cost, deadline_at=deadline, confidence=confidence,
                        acceptance=acceptance, autonomy=autonomy, revision_of=revision_of)


SCHEMA_HINT = ('{"record_id": "goal-<id>", "kind": "goal|task", "text": "...", '
               '"source_kind": "user_message|device_log|document|tool_result|model_output|schedule", '
               '"evidence_ids": ["ev-user-..."], "priority": 0.0-1.0, '
               '"interruption_cost": 0.0-1.0, "deadline_at": "ISO-8601 with offset or null", '
               '"confidence": 0.0-1.0, "acceptance": "pending|accepted", '
               '"autonomy": "notify_only|can_act", "revision_of": "goal-<id> or null"}')


def proposal_prompt(tokenizer, *, system, context_lines, known_evidence, max_new_tokens=None):
    """Real-tokenizer prompt contract; the body must answer with one JSON object."""
    content = ('可用证据身份：%s\n上下文：\n%s\n请只输出一个 JSON 对象，不要解释。字段：%s'
               % (sorted(known_evidence), '\n'.join(context_lines), SCHEMA_HINT))
    prompt = tokenizer.apply_chat_template(
        [dict(role='system', content=system), dict(role='user', content=content)],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    return dict(prompt=prompt, input_ids=ids, tokens=len(ids),
                prompt_sha256=hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                max_new_tokens=max_new_tokens, template='qwen2-chat/think-off')


class IntakeJournal:
    """Append-only record of every generation, including every failure."""

    def __init__(self):
        self.entries = []

    def append(self, entry):
        encoded = json.dumps(entry, sort_keys=True, ensure_ascii=True, default=str)
        entry = dict(entry)
        entry['entry_sha256'] = hashlib.sha256(encoded.encode('utf-8')).hexdigest()
        self.entries.append(entry)
        return entry

    def payload(self):
        return dict(schema='d06-intake-journal', version=1, entries=list(self.entries))

    def digest(self):
        return hashlib.sha256(json.dumps(self.payload(), sort_keys=True,
                                         ensure_ascii=True).encode('utf-8')).hexdigest()

    def accepted(self):
        return [e for e in self.entries if e.get('status') == 'accepted']

    def revisions(self):
        children = {}
        for entry in self.accepted():
            record = entry.get('proposal') or {}
            parent = record.get('revision_of')
            if parent:
                children.setdefault(parent, []).append(record.get('record_id'))
        return {parent: tuple(ids) for parent, ids in children.items()}

    def counts(self):
        rejected = [e for e in self.entries if e.get('status') == 'rejected']
        errors = [e for e in self.entries if e.get('status') == 'generator-error']
        return dict(attempts=len(self.entries), accepted=len(self.accepted()),
                    rejected=len(rejected), generator_errors=len(errors),
                    codes=sorted({e.get('error_code') for e in rejected}))


def intake_proposal(generate, tokenizer, *, system, context_lines, known_evidence,
                    now=None, known_records=(), journal=None, attempts=2,
                    max_new_tokens=None):
    """Call the injected same-body generator, validate, keep everything.

    generate(prompt_text) -> raw text.  Any exception from the generator is
    recorded as generator-error; it never becomes a proposal.
    """
    journal = IntakeJournal() if journal is None else journal
    prompt = proposal_prompt(tokenizer, system=system, context_lines=context_lines,
                             known_evidence=known_evidence, max_new_tokens=max_new_tokens)
    errors = []
    for attempt in range(1, int(attempts) + 1):
        base = dict(attempt=attempt, prompt_sha256=prompt['prompt_sha256'],
                    prompt_tokens=prompt['tokens'], prompt=prompt['prompt'],
                    max_new_tokens=max_new_tokens)
        try:
            raw = generate(prompt['prompt'])
        except Exception as exc:                     # noqa: BLE001 - kept verbatim
            entry = journal.append(dict(base, status='generator-error',
                                        error_code='p.generator-error',
                                        error='%s: %s' % (type(exc).__name__, exc), raw=None))
            errors.append(entry)
            continue
        try:
            proposal = parse_goal_proposal(raw, known_evidence=known_evidence,
                                           known_records=known_records, now=now)
        except IntakeError as exc:
            entry = journal.append(dict(base, status='rejected', error_code=exc.code,
                                        error=str(exc), raw=str(raw)))
            errors.append(entry)
            continue
        entry = journal.append(dict(base, status='accepted', error_code=None, error=None,
                                    raw=str(raw), proposal=proposal.as_dict()))
        return dict(status='accepted', attempt=attempt, proposal=proposal,
                    prompt=prompt, errors=errors, journal=journal)
    return dict(status='rejected', attempt=int(attempts), proposal=None,
                prompt=prompt, errors=errors, journal=journal)


TOKENIZER_DIR = (r'D:\MAT\Tryline\research\treyrin-native-language-closure-2026-10-25'
                 r'\assets\tokenizer')


def load_tokenizer(path=TOKENIZER_DIR):
    """Load ONLY the pinned tokenizer (no body, no weights, no network)."""
    import os
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path)