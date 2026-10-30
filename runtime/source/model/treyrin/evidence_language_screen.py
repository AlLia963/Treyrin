"""Bounded-language adapter for the experimental evidence-coupling operator.

Reuses the same small encoder and causal decoder in every control. Event
boundaries/identity are explicit input metadata, not learned provenance.
This adapter is not an open-language model or a cross-pretrained transfer proof.
"""
import torch

from treyrin.coupled_screen import CoupledScreen
from treyrin.evidence_coupling import Candidates, EvidenceCoupling, Observations


class EvidenceLanguageScreen(CoupledScreen):
    def __init__(self, width=48):
        super().__init__(width)
        self.coupling = EvidenceCoupling(width)

    def encode_bank(self, tokens, valid):
        if tokens.ndim != 3 or valid.shape != tokens.shape[:2]:
            raise ValueError('Bank token/mask shape mismatch')
        batch, count, length = tokens.shape
        flat_valid = valid.flatten()
        if (tokens.flatten(0, 1)[flat_valid].ne(0).sum(-1) == 0).any():
            raise ValueError('A valid bank entry needs observed content')
        result = self.embedding.weight.new_zeros(batch * count, self.width)
        indices = flat_valid.nonzero()[:, 0]
        if indices.numel():
            encoded = self.encode(tokens.flatten(0, 1)[indices])
            result = result.index_copy(0, indices, encoded)
        return result.reshape(batch, count, self.width)

    def coupled_state(self, event_tokens, event_valid, event_ids, candidate_tokens,
                      candidate_valid, *, mode='evidence', depth=2, probability_override=None):
        observed = Observations(self.encode_bank(event_tokens, event_valid), event_valid, event_ids)
        candidates = Candidates(self.encode_bank(candidate_tokens, candidate_valid), candidate_valid,
                                exhaustive=False)
        return self.coupling(observed, candidates, mode=mode, steps=depth,
                             probability_override=probability_override)

    def load_small_pretraining(self, weights):
        # The old module remains intact so historical checkpoints/experiments
        # retain their meaning. New operator initialization is seed-controlled.
        result = self.load_state_dict(weights, strict=False)
        if result.unexpected_keys or any(not name.startswith('coupling.') for name in result.missing_keys):
            raise ValueError('Incompatible small-model pretraining checkpoint')
        return result
