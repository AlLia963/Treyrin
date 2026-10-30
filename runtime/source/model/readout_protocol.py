"""Registered order-controlled readout over two counterbalanced model calls.

The registered protocol asks the same question twice with the option order swapped,
maps both probability vectors back to semantic option indices, and decides on the
mean. A declared unknown band turns sub-threshold margins into an explicit
unknown choice instead of a forced letter, so an item whose content margin is smaller
than its option-order term is no longer answered by order luck.

Pure standard-library Python: no model call, no GPU work. The caller supplies the
two probability vectors; replay_session.query(order_control=True) performs the
two calls and this module performs the mapping and decision.
"""
from __future__ import annotations

PROTOCOL = 'order_controlled_v1'
ORDERS = ('original', 'swapped')
UNKNOWN_MARGIN = 0.05


def registration():
    """The declared protocol record; persisted with every order-controlled read."""
    return dict(protocol=PROTOCOL, orders=list(ORDERS),
                mapping='semantic mean: (forward[i] + reverse[n-1-i]) / 2',
                decision='argmax over the semantic mean; unknown when top-two margin < threshold',
                unknown_margin=UNKNOWN_MARGIN,
                calibration=('answer-side band calibrated on the frozen ten cells: (0.0224, 0.3207] '
                             'has 0 false-unknown and 0 false-answer (R5-1M-ACCEPTANCE.json); the '
                             'neural replay-mass gate remains declared'))


def semantic_probabilities(forward, reverse):
    """Map a swapped second call back onto the first call's option indices."""
    if len(forward) != len(reverse) or not forward:
        raise ValueError('Counterbalanced probability vectors must have equal nonzero length')
    if len(set(forward)) == 0 or len(set(reverse)) == 0:
        raise ValueError('Probability vectors must be finite')
    return [(float(forward[i]) + float(reverse[len(forward) - 1 - i])) / 2 for i in range(len(forward))]


def decide(probabilities, *, unknown_margin=None):
    """Return the semantic choice, the top-two margin and whether it is unknown."""
    if not probabilities:
        raise ValueError('A nonempty semantic probability vector is required')
    order = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)
    top = probabilities[order[0]]
    second = probabilities[order[1]] if len(order) > 1 else 0.0
    margin = float(top - second)
    limit = UNKNOWN_MARGIN if unknown_margin is None else float(unknown_margin)
    unknown = len(order) > 1 and margin < limit
    return dict(choice=None if unknown else order[0], unknown=bool(unknown),
                margin=margin, order=order, unknown_margin=limit)


def counterbalanced(forward, reverse, *, unknown_margin=None):
    """Full protocol result: semantic probabilities plus the decision."""
    semantic = semantic_probabilities(forward, reverse)
    decision = decide(semantic, unknown_margin=unknown_margin)
    return dict(registration=registration(), forward=list(forward), reverse=list(reverse),
                semantic_probabilities=semantic, **decision)
