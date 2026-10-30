"""Experimental, model-independent internal operator for variable hypotheses.

Native probabilities can route reads of observations instead of being written
as new facts. This is an UNTRAINED candidate, not an independent complete model,
an established efficiency gain, a brain reproduction, or a novel method claim.
Probabilities are conditional on the supplied candidate set, not open-world
truth probabilities. Candidate encoding/coverage must be learned and tested.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Observations:
    values: torch.Tensor       # B,N,D: observed event encodings, not predictions
    valid: torch.Tensor        # B,N
    event_ids: torch.Tensor    # B,N: explicit event identity, not publisher ID


@dataclass(frozen=True)
class Candidates:
    values: torch.Tensor       # B,K,D: content encodings, no fixed category head
    valid: torch.Tensor        # B,K
    exhaustive: bool = False   # Metadata only; softmax never estimates coverage.


@dataclass(frozen=True)
class CouplingResult:
    state: torch.Tensor
    conditional_probability: torch.Tensor
    observation_attention: torch.Tensor


MODES = ('none', 'content', 'evidence', 'detach_evidence', 'uniform_evidence',
         'misaligned_evidence', 'state_evidence', 'mean_query_evidence')


class EvidenceCoupling(nn.Module):
    def __init__(self, width=48):
        super().__init__()
        self.width = width
        self.prior = nn.Parameter(torch.zeros(width))
        self.score_query = nn.Linear(width, width, bias=False)
        self.score_key = nn.Linear(width, width, bias=False)
        self.read_query = nn.Linear(width, width, bias=False)
        self.hypothesis_query = nn.Linear(width, width, bias=False)
        # Independent state-feature router: same functional family as native
        # scoring, without tying routing weights to native judgment supervision.
        self.router_query = nn.Linear(width, width, bias=False)
        self.router_key = nn.Linear(width, width, bias=False)
        self.read_key = nn.Linear(width, width, bias=False)
        self.read_value = nn.Linear(width, width, bias=False)
        self.refine = nn.Sequential(nn.LayerNorm(3 * width), nn.Linear(3 * width, 2 * width),
                                    nn.GELU(), nn.Linear(2 * width, width))
        self.norm = nn.LayerNorm(width)

    def validate(self, observed, candidates):
        if observed.values.ndim != 3 or candidates.values.ndim != 3:
            raise ValueError('Observation and candidate banks must be B,N,D tensors')
        batch, count, width = observed.values.shape
        if width != self.width or candidates.values.shape[0] != batch or candidates.values.shape[2] != width:
            raise ValueError('Bank batch/width mismatch')
        if min(batch, count, candidates.values.shape[1]) < 1:
            raise ValueError('Use masked null observations for episodes without evidence')
        if observed.valid.shape != (batch, count) or observed.event_ids.shape != (batch, count):
            raise ValueError('Observation metadata mismatch')
        if candidates.valid.shape != candidates.values.shape[:2] or not candidates.valid.any(-1).all():
            raise ValueError('Every episode needs a nonempty candidate set')
        if observed.valid.dtype != torch.bool or candidates.valid.dtype != torch.bool:
            raise ValueError('Validity masks must be boolean')
        if not torch.isfinite(observed.values).all() or not torch.isfinite(candidates.values).all():
            raise ValueError('Nonfinite bank values')

    @staticmethod
    def replay_weights(observed):
        same = observed.event_ids[:, :, None].eq(observed.event_ids[:, None, :])
        counts = (same & observed.valid[:, None, :]).sum(-1).clamp_min(1)
        # Exact replay has unit total weight per event. Identity is supplied by
        # provenance, not discovered by this operator. Conflicting encodings or
        # unknown duplicate identity are not semantically resolved here.
        return observed.valid.to(observed.values.dtype) / counts

    def initial_state(self, observed):
        weights = self.replay_weights(observed)
        mean = (observed.values * weights[..., None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(1)
        return mean + self.prior

    def decision(self, state, candidates):
        logits = torch.einsum('bd,bkd->bk', self.score_query(state), self.score_key(candidates.values))
        logits = logits / math.sqrt(self.width)
        return logits.masked_fill(~candidates.valid, -torch.inf).softmax(-1)

    def _attend(self, queries, observed):
        weight = self.replay_weights(observed)
        score = torch.einsum('bqd,bnd->bqn', queries, self.read_key(observed.values)) / math.sqrt(self.width)
        score = score + weight.clamp_min(1e-30).log()[:, None, :]
        score = score.masked_fill(~observed.valid[:, None, :], torch.finfo(score.dtype).min)
        attention = score.softmax(-1) * observed.valid[:, None, :]
        # All-masked observation banks yield a zero read and retain the learned
        # prior. No invented observation is inserted to make softmax work.
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-30)
        read = torch.einsum('bqn,bnd->bqd', attention, self.read_value(observed.values))
        return read, attention

    def forward(self, observed, candidates, *, mode='evidence', steps=2, state=None,
                probability_override=None):
        self.validate(observed, candidates)
        if mode not in MODES or steps < 1:
            raise ValueError('Invalid mode or inference depth')
        state = self.initial_state(observed) if state is None else state
        if state.shape != (observed.values.shape[0], self.width) or not torch.isfinite(state).all():
            raise ValueError('Invalid working state')
        if probability_override is not None:
            if probability_override.shape != candidates.valid.shape:
                raise ValueError('Intervention shape mismatch')
            if (not torch.isfinite(probability_override).all() or (probability_override < 0).any()
                    or (probability_override.masked_select(~candidates.valid) != 0).any()
                    or not torch.allclose(probability_override.sum(-1),
                                          torch.ones_like(probability_override[:, 0]), atol=1e-6)):
                raise ValueError('Intervention must be a valid candidate-conditional distribution')
        attention = None
        for _ in range(steps):
            probability = self.decision(state, candidates)
            routed = probability if probability_override is None else probability_override
            if mode == 'state_evidence' and probability_override is None:
                route_score = torch.einsum('bd,bkd->bk', self.router_query(state),
                                          self.router_key(candidates.values)) / math.sqrt(self.width)
                routed = route_score.masked_fill(~candidates.valid, -torch.inf).softmax(-1)
            if mode == 'detach_evidence':
                routed = routed.detach()
            if mode == 'uniform_evidence':
                routed = candidates.valid.to(state.dtype) / candidates.valid.sum(-1, keepdim=True)
            if mode == 'misaligned_evidence':
                # Cyclic shift among VALID candidates only; padding gets no mass.
                routed = torch.stack([torch.zeros_like(row).scatter(0, valid.nonzero()[:, 0],
                              row[valid].roll(1)) for row, valid in zip(routed, candidates.valid)])
            base_query = self.read_query(state)[:, None]
            content = (candidates.values * routed[..., None]).sum(1)
            if mode in ('none', 'content'):
                ordinary_read, ordinary_attention = self._attend(base_query, observed)
                read, attention = ordinary_read[:, 0], ordinary_attention[:, 0]
            elif mode == 'mean_query_evidence':
                query = base_query + self.hypothesis_query(content)[:, None]
                mean_read, mean_attention = self._attend(query, observed)
                read, attention = mean_read[:, 0], mean_attention[:, 0]
            else:
                candidate_queries = base_query + self.hypothesis_query(candidates.values)
                candidate_reads, candidate_attention = self._attend(candidate_queries, observed)
                # Average AFTER nonlinear candidate-conditioned reads. This is
                # compared with the explicit mean-query control above.
                read = (candidate_reads * routed[..., None]).sum(1)
                attention = (candidate_attention * routed[..., None]).sum(1)
            if mode != 'content':
                content = content * 0
            state = self.norm(state + self.refine(torch.cat((state, read, content), -1)))
        return CouplingResult(state, self.decision(state, candidates), attention)
