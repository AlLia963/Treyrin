"""Untrained multiplicity-aware descendant, separate from frozen v0.8 study.

Normalized content alone cannot express the amount of matching observations.
This adds a learnable unnormalized content integral and a count direction.
Different IDs do NOT prove statistical independence. The branch makes amount
representable; it does not prescribe confidence or solve source correlation.
"""
import torch
from torch import nn

from treyrin.evidence_coupling import EvidenceCoupling


class MeasuredEvidenceCoupling(EvidenceCoupling):
    def __init__(self, width=48):
        super().__init__(width)
        self.sum_projection = nn.Linear(width, width, bias=False)
        self.amount_direction = nn.Parameter(torch.zeros(width))
        # Zero initialization permits exact functional inheritance from the
        # earlier operator. These parameters can receive gradients at zero.
        nn.init.zeros_(self.sum_projection.weight)

    def initial_state(self, observed):
        weight = self.replay_weights(observed)
        integral = (observed.values * weight[..., None]).sum(1)
        amount = weight.sum(1, keepdim=True)
        return (super().initial_state(observed) + self.sum_projection(integral)
                + amount.log1p() * self.amount_direction)

    def inherit_operator(self, state_dict):
        result = self.load_state_dict(state_dict, strict=False)
        expected = {'sum_projection.weight', 'amount_direction'}
        if result.unexpected_keys or set(result.missing_keys) != expected:
            raise ValueError('Expected an exact prior evidence operator checkpoint')
        return result
