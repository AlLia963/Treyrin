"""Bounded, oracle-supervised mechanism screen; not a pretrained language model.

One latent recurrent operator uses native distributions and causal token states.
The finite, tied compatibility map is NOT an Energy-Based Transformer replica.
Observed evidence is encoded once and remains an anchor during internal loops.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


WORDS = (['pad', 'bos', 'eos', 'source'] + [f's{i}' for i in range(8)]
         + ['sensor'] + [f'f{i}' for i in range(8)]
         + ['is', 'not', 'high', 'low', 'reads', 'likely']
         + [f'c{i}' for i in range(4)]
         + ['inspect', 'uncertainty', 'low_u', 'mid_u', 'high_u', 'copy']
         + ['alpha', 'beta', 'gamma', 'delta', ';', '.'])
VOCAB = {word: index for index, word in enumerate(WORDS)}
CAUSE_IDS = [VOCAB[f'c{i}'] for i in range(4)]
MODES = ('shared', 'soft', 'detach', 'hard', 'gated', 'independent')


class CoupledScreen(nn.Module):
    def __init__(self, width=48):
        super().__init__()
        self.width = width
        self.embedding = nn.Embedding(len(WORDS), width, padding_idx=0)
        self.position = nn.Embedding(80, width)
        layer = nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=0.,
                                          batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.anchor_norm = nn.LayerNorm(width)
        self.core = nn.Sequential(nn.Linear(3 * width, 2 * width), nn.GELU(),
                                  nn.Linear(2 * width, width))
        self.state_norm = nn.LayerNorm(width)
        self.compatibility = nn.Sequential(nn.LayerNorm(width),
                                           nn.Linear(width, width, bias=False))
        # All arms allocate the same parameters; inactive parameters are reported.
        self.independent_decision = nn.Linear(width, 4)
        self.gate = nn.Sequential(nn.Linear(width + 2, 24), nn.GELU(), nn.Linear(24, 1))
        self.decoder = nn.GRU(width, width, batch_first=True)

    def encode(self, tokens):
        keep = tokens.ne(0)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.embedding(tokens) + self.position(positions)[None]
        hidden = self.encoder(hidden, src_key_padding_mask=~keep)
        return self.anchor_norm((hidden * keep[..., None]).sum(1)
                                / keep.sum(1, keepdim=True))

    def token_logits(self, state):
        return F.linear(self.compatibility(state), self.embedding.weight) / math.sqrt(self.width)

    def decision_logits(self, state, mode):
        # The tied arms use exactly the same compatibility map as causal tokens.
        tied = self.token_logits(state)[..., CAUSE_IDS]
        independent = self.independent_decision(state)
        return independent if mode == 'independent' else tied

    def think(self, anchor, mode, depth, *, override=None, feedback_off=False):
        if mode not in MODES or depth < 1:
            raise ValueError('Unknown arm or invalid recurrence depth')
        state = anchor
        previous = None
        gates = []
        for _ in range(depth):
            probability = self.decision_logits(state, mode).softmax(-1)
            routed = probability if override is None else override
            if mode == 'detach':
                routed = routed.detach()
            if mode == 'hard':
                routed = F.one_hot(routed.argmax(-1), 4).to(routed.dtype)
            message = routed @ self.embedding.weight[CAUSE_IDS]
            entropy = -(probability * probability.clamp_min(1e-8).log()).sum(-1) / math.log(4)
            disagreement = (torch.zeros_like(entropy) if previous is None else
                            (probability - previous).abs().sum(-1) / 2)
            gate = self.gate(torch.cat((state, entropy[:, None], disagreement[:, None]), -1)).sigmoid()
            if mode == 'gated':
                message = message * gate
            if mode == 'shared' or feedback_off:
                message = message * 0
            # External anchor is re-injected, but internal guesses do not become
            # external observations. This does not guarantee calibrated beliefs.
            state = self.state_norm(state + self.core(torch.cat((state, anchor, message), -1)))
            previous = probability
            gates.append(gate[:, 0])
        return state, self.decision_logits(state, mode), torch.stack(gates, 1)

    def language(self, state, observed_prefix):
        # GRU is causal. Only BOS and earlier target tokens are supplied at each
        # position. Decision loss is computed before any target token is read.
        hidden, _ = self.decoder(self.embedding(observed_prefix), state[None])
        return self.token_logits(hidden)

    def generate(self, state, count=10):
        token = torch.full((state.shape[0], 1), VOCAB['bos'], dtype=torch.long,
                           device=state.device)
        hidden = state[None]
        outputs = []
        for _ in range(count):
            out, hidden = self.decoder(self.embedding(token), hidden)
            token = self.token_logits(out[:, -1]).argmax(-1)[:, None]
            outputs.append(token[:, 0])
        return torch.stack(outputs, 1)
