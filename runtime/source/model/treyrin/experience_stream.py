"""Lossless, bounded observation segments for persistent-state experiments.

Only historical message data cross this boundary. Questions, teacher answers and
scoring labels are not arguments. Splitting occurs on the token IDs of one JSON
record, never by independently decoding partial Unicode tokens. A segment may
contain a JSON fragment; the consumer processes a stream, not separate records.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Callable, Iterable, Iterator, Sequence

from .experience_io import ExperienceMessage


@dataclass(frozen=True)
class ObservationSegment:
    input_ids: tuple[int, ...]
    event_id: str
    event_ordinal: int
    part_index: int
    part_count: int
    token_offset: int
    record_sha256: str


def render_observation(message: ExperienceMessage) -> str:
    """Historical role is a data field, not a live chat/system role."""
    if not isinstance(message, ExperienceMessage):
        raise ValueError('Only validated historical ExperienceMessage records may be written')
    if message.role not in ('user', 'assistant'):
        raise ValueError('Historical messages must use the reviewed data-role vocabulary')
    if any(not isinstance(value, str) for value in asdict(message).values()):
        raise ValueError('Observation fields must be strings')
    if not all((message.event_id, message.session_id, message.session_timestamp)):
        raise ValueError('Explicit event, session and source time are required')
    return json.dumps({'record_type': 'historical_observation', **asdict(message)},
                      ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n'


def observation_segments(messages: Iterable[ExperienceMessage],
                         encode: Callable[[str], Sequence[int]], *,
                         max_tokens: int) -> Iterator[ObservationSegment]:
    """Preserve supplied order and every token, including replayed event IDs.

    ``max_tokens`` is the OBSERVATION budget after the caller reserves read/write
    memory positions and model limits. Metadata in ObservationSegment is local
    bookkeeping; the model receives input_ids only. No record is silently cut,
    deduplicated, timestamp-filtered, summarised or relabelled here.
    """
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError('A positive observation-token budget is required')
    for ordinal, message in enumerate(messages):
        record = render_observation(message)
        ids = tuple(encode(record))
        if not ids or any(type(value) is not int or value < 0 for value in ids):
            raise ValueError('Encoder must return nonempty nonnegative integer token IDs')
        parts = (len(ids) + max_tokens - 1) // max_tokens
        digest = hashlib.sha256(record.encode('utf-8')).hexdigest()
        for part, offset in enumerate(range(0, len(ids), max_tokens)):
            yield ObservationSegment(ids[offset:offset + max_tokens], message.event_id,
                                     ordinal, part, parts, offset, digest)
