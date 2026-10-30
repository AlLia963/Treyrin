"""Base-independent observation input, separated from evaluation-only labels.

This is an ingestion boundary, not a learned memory implementation. Historical
roles describe data; callers must not replay them as live system instructions.
"""
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import hmac
import re
from typing import Mapping


@dataclass(frozen=True)
class ExperienceMessage:
    event_id: str
    session_id: str
    session_timestamp: str
    role: str
    content: str


@dataclass(frozen=True)
class ExperienceQuery:
    invocation_id: str
    text: str
    timestamp: str


@dataclass(frozen=True)
class ExperienceInput:
    messages: tuple[ExperienceMessage, ...]
    query: ExperienceQuery

    def as_model_data(self) -> dict:
        """Only public observations and the query; no evaluator object attached."""
        return asdict(self)


def source_timestamp(text: str) -> datetime:
    """Parse released LME timestamps without inventing a timezone or precision."""
    match = re.fullmatch(r'(\d{4})/(\d{2})/(\d{2}) \([^)]*\) (\d{2}):(\d{2})', text)
    if not match:
        raise ValueError(f'Unsupported source timestamp: {text!r}')
    return datetime(*map(int, match.groups()))


def longmemeval_input(item: Mapping, *, nonce: bytes) -> ExperienceInput:
    """Compile one released evaluation history, with stable local replay IDs.

    Raw session IDs may contain ``answer_`` and raw question IDs may contain
    ``_abs``. They are not model input. Nonce is run-local, at least 16 bytes,
    and belongs to evaluator bookkeeping. Labels embedded in turns are stripped
    by a whitelist, while original role/content remain verbatim.

    Oracle history is not necessarily timestamp sorted. All provided sessions
    are retained and stably sorted; this function does not silently apply a
    query-time cutoff or turn oracle retrieval into an unprivileged experiment.
    """
    if not isinstance(nonce, bytes) or len(nonce) < 16:
        raise ValueError('Use an evaluator-owned nonce of at least 16 bytes')

    def opaque(namespace: str, value: str) -> str:
        return namespace + '_' + hmac.new(nonce, value.encode('utf-8'), hashlib.sha256).hexdigest()[:32]

    sessions = item['haystack_sessions']
    dates = item['haystack_dates']
    ids = item['haystack_session_ids']
    if not len(sessions) == len(dates) == len(ids):
        raise ValueError('Sessions, timestamps and session IDs must align')
    ordered = sorted(range(len(sessions)), key=lambda i: (source_timestamp(dates[i]), i))
    messages = []
    for index in ordered:
        session_id = ids[index]
        if not isinstance(session_id, str):
            raise ValueError('Session IDs must be strings')
        for turn_index, turn in enumerate(sessions[index]):
            role, content = turn['role'], turn['content']
            if role not in ('user', 'assistant') or not isinstance(content, str):
                raise ValueError('Expected historical user/assistant text turns')
            messages.append(ExperienceMessage(
                event_id=opaque('event', session_id + '\0' + str(turn_index)),
                session_id=opaque('session', session_id), session_timestamp=dates[index],
                role=role, content=content))
    question, date = item['question'], item['question_date']
    if not isinstance(question, str):
        raise ValueError('Question must be text')
    source_timestamp(date)
    return ExperienceInput(tuple(messages), ExperienceQuery(opaque('query', 'invocation'), question, date))
