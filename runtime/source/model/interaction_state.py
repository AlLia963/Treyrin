"""Durable per-request state for one live role turn: submit, cancel, fault, resume.

A normal assistant turn is one request with one identity. The request is written
to an append-only, fsync'd journal BEFORE the decoder starts, so a user interrupt,
a process exit or an engine failure cannot lose the fact that the user asked
something. Nothing here infers intent from the question text: the record keeps
the exact submitted text, the user event identity and the resident generation
sequence it was committed onto.

State machine (an event log, not a mutable row):

    submitted -> generating -> committed
                    |  -> cancelled   (user stop / Ctrl-C / SIGINT)
                    |  -> failed      (resource, engine or decode failure)
                    -> cancelled / failed   (interrupt during prefill)

A delivered answer is final. A cancelled or failed attempt keeps the same
request_id, so continuing it answers the SAME user event instead of submitting
a new one; the attempt counter then increments.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path

SCHEMA = 'treyrin-request-state-v1'
STATES = ('submitted', 'generating', 'committed', 'cancelled', 'failed')
TERMINAL = ('committed', 'cancelled', 'failed')
TRANSITIONS = {
    'submitted': ('generating', 'cancelled', 'failed'),
    'generating': ('committed', 'cancelled', 'failed'),
    # a delivered answer is final; an interrupted attempt may be continued
    # under the SAME request identity so the user event is answered once
    'committed': (),
    'cancelled': ('generating',),
    'failed': ('generating',),
}
# finish reasons that mean "the model finished on its own terms"
COMPLETE_FINISH = ('stop', 'eos')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def _nonempty(name, value):
    if not isinstance(value, str) or not value:
        raise ValueError('Request state requires a nonempty %s' % name)
    return value


@dataclass(frozen=True)
class RequestRecord:
    request_id: str
    conversation_id: str
    state: str
    attempt: int
    user_text: str
    user_event_id: str
    known_at: str
    admitted: bool = False
    generation_sequence: int | None = None
    partial_token_ids: tuple | None = None
    answer: str | None = None
    finish_reason: str | None = None
    error: str | None = None
    previous_sha256: str | None = None
    schema: str = SCHEMA

    def body(self):
        values = asdict(self)
        values.pop('previous_sha256', None)
        return values

    def sealed(self):
        values = self.body()
        return dict(values, previous_sha256=self.previous_sha256, record_sha256=_digest(values))


class RequestJournal:
    """Append-only request log. One live turn => at most one open request."""

    def __init__(self, path):
        self.path = Path(path)

    # -- reading -----------------------------------------------------------
    def entries(self):
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text('utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            body = {k: v for k, v in row.items() if k not in ('previous_sha256', 'record_sha256')}
            if row.get('record_sha256') != _digest(body):
                raise ValueError('Request journal record digest differs at request ' + str(row.get('request_id')))
            rows.append(row)
        previous = None
        for row in rows:
            if row.get('previous_sha256') != previous:
                raise ValueError('Request journal chain is broken at request ' + str(row.get('request_id')))
            previous = row['record_sha256']
        return rows

    def history(self, request_id):
        return [row for row in self.entries() if row['request_id'] == request_id]

    def state_of(self, request_id):
        rows = self.history(request_id)
        if not rows:
            raise KeyError('Unknown request identity ' + str(request_id))
        return rows[-1]

    def open_requests(self):
        """Requests already in flight (submitted or generating)."""
        last = {}
        for row in self.entries():
            last[row['request_id']] = row
        return [row for row in last.values() if row['state'] not in TERMINAL]

    def unanswered_requests(self):
        """Requests the user asked and no answer was delivered for.

        A cancelled or failed attempt still leaves the user's turn unanswered:
        the role session keeps its pending_user, so the same request_id is the
        only legal continuation. Nothing is re-submitted under a new identity.
        """
        last = {}
        for row in self.entries():
            last[row['request_id']] = row
        return [row for row in last.values() if row['state'] != 'committed']

    def resume_plan(self):
        """The one request a restarted process may continue, or None."""
        rows = self.unanswered_requests()
        if len(rows) > 1:
            raise ValueError('More than one unanswered request; the live turn is ambiguous')
        if not rows:
            return None
        row = rows[0]
        return dict(request_id=row['request_id'], user_text=row['user_text'],
                    user_event_id=row['user_event_id'], known_at=row['known_at'],
                    state=row['state'], attempt=row['attempt'], unanswered=True,
                    generation_sequence=row.get('generation_sequence'),
                    partial_token_ids=row.get('partial_token_ids'),
                    admitted=bool(row.get('admitted')),
                    resume_action=('respond' if row.get('admitted') else 'submit_user_turn_then_respond'),
                    answers_the_same_user_event=True,
                    needs_user_confirmation=row['state'] in ('cancelled', 'failed'),
                    reason=(row.get('finish_reason') or row.get('error')))

    # -- writing -----------------------------------------------------------
    @staticmethod
    def _project(row, **overrides):
        """Rebuild a record from a journal row, keeping only declared fields."""
        fields = {name: row.get(name) for name in
                  ('request_id', 'conversation_id', 'state', 'attempt', 'user_text',
                   'user_event_id', 'known_at', 'admitted', 'generation_sequence',
                   'partial_token_ids', 'answer', 'finish_reason', 'error')}
        fields.update(overrides)
        fields.pop('schema', None)
        return RequestRecord(**fields)

    def _previous(self):
        rows = self.entries()
        return rows[-1]['record_sha256'] if rows else None

    def _append(self, record, state):
        if state not in STATES:
            raise ValueError('Unknown request state ' + str(state))
        if not isinstance(record, RequestRecord):
            raise ValueError('Re-validate a RequestRecord before appending its next state')
        if record.attempt < 1:
            raise ValueError('Request attempt starts at 1')
        rows = self.history(record.request_id)
        if rows:
            # identity first: a request may only ever answer the turn it was
            # submitted for, whatever transition is being attempted
            if rows[0]['user_text'] != record.user_text:
                raise ValueError('Resumed request does not answer the submitted text')
            if rows[0]['user_event_id'] != record.user_event_id:
                raise ValueError('Resumed request does not answer the submitted user event')
            current = rows[-1]['state']
            if state not in TRANSITIONS[current]:
                raise ValueError('Illegal request transition %s -> %s' % (current, state))
            attempt = rows[-1]['attempt'] + int(state == 'generating' and current in ('cancelled', 'failed'))
        else:
            if state != 'submitted':
                raise ValueError('A request must be submitted before it is ' + state)
            attempt = record.attempt
        if state == 'committed' and not isinstance(record.answer, str):
            raise ValueError('A committed request must carry the delivered answer text')
        if state == 'committed' and not record.admitted:
            raise ValueError('Cannot deliver an answer for a request the session never admitted')
        sealed = RequestRecord(request_id=record.request_id, conversation_id=record.conversation_id,
                               state=state, attempt=attempt, user_text=record.user_text,
                               user_event_id=record.user_event_id, known_at=record.known_at,
                               admitted=bool(record.admitted),
                               generation_sequence=record.generation_sequence,
                               partial_token_ids=(None if record.partial_token_ids is None
                                                  else tuple(int(x) for x in record.partial_token_ids)),
                               answer=record.answer, finish_reason=record.finish_reason,
                               error=record.error, previous_sha256=self._previous()).sealed()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(canonical(sealed) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        return sealed

    def submit(self, *, request_id, conversation_id, user_text, user_event_id, known_at,
               attempt=1, generation_sequence=None, admitted=False):
        _nonempty('request_id', request_id)
        _nonempty('conversation_id', conversation_id)
        _nonempty('user_text', user_text)
        _nonempty('user_event_id', user_event_id)
        _nonempty('known_at', known_at)
        record = RequestRecord(request_id=request_id, conversation_id=conversation_id,
                               state='submitted', attempt=int(attempt), user_text=user_text,
                               user_event_id=user_event_id, known_at=known_at, admitted=bool(admitted),
                               generation_sequence=generation_sequence)
        return self._append(record, 'submitted')

    def begin(self, request_id, *, generation_sequence=None, admitted=True):
        """The role session now holds this user turn as its pending request."""
        row = self.state_of(request_id)
        record = self._project(row, admitted=bool(admitted),
                               generation_sequence=(generation_sequence if generation_sequence is not None
                                                    else row.get('generation_sequence')))
        return self._append(record, 'generating')

    def commit(self, request_id, *, answer, finish_reason, token_ids=None, generation_sequence=None):
        if not isinstance(answer, str):
            raise ValueError('Committed answer text is required')
        _nonempty('finish_reason', finish_reason)
        row = self.state_of(request_id)
        record = self._project(row, answer=answer, finish_reason=finish_reason, admitted=True,
                                  partial_token_ids=token_ids,
                                  generation_sequence=(generation_sequence if generation_sequence is not None
                                                       else row.get('generation_sequence')))
        return self._append(record, 'committed')

    def cancel(self, request_id, *, finish_reason='cancelled', partial_token_ids=None, note=None,
               admitted=None):
        row = self.state_of(request_id)
        record = self._project(row, finish_reason=finish_reason,
                               admitted=(row.get('admitted') if admitted is None else bool(admitted)),
                               partial_token_ids=partial_token_ids, error=note)
        return self._append(record, 'cancelled')

    def fail(self, request_id, *, error, partial_token_ids=None, admitted=None):
        _nonempty('error', error)
        row = self.state_of(request_id)
        record = self._project(row, error=error, partial_token_ids=partial_token_ids,
                               admitted=(row.get('admitted') if admitted is None else bool(admitted)))
        return self._append(record, 'failed')
