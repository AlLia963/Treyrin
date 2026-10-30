"""r8 one-shot brain-structure integration modules.

Thalamic gating, dual cortical streams, dopaminergic basal-ganglia closure,
competitive hippocampal retrieval, cerebellar fast prediction, global
neuromodulation, amygdalar valence and hypothalamic drives are real
computations with explicit intervention handles. Their names are engineering
correspondences to verified circuit literature (PLAN.md section 4), NOT claims
of anatomical fidelity. Every structure can be ablated at call time so paired
causal tests stay possible. No teacher signal enters any of these modules:
dopamine is an internal temporal-difference on predicted versus observed
encodings, never a loss-side label.
"""
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .regional_circuit import PositiveLinear


@dataclass
class ThalamicResult:
    routed: Tensor            # B,D; observation content that reaches cortex
    populations: Tensor       # B,3,D; relay / attention / inhibition
    gate: Tensor              # B,3; per-region throughput


class ThalamicGate(nn.Module):
    """A gated relay into the regional cortex; other core pathways remain.

    Three populations: a content relay (specific nuclei analogue), an
    attention gate on the pulvinar analogue conditioned on goal and state
    statistics, and a reticular-nucleus analogue providing inhibitory
    suppression. Eradath 2021-style causal tests are the paired ablations
    no_relay / no_gate / open_gate.
    """
    populations = ('sensory_relay', 'pulvinar_attention', 'reticular_inhibition')
    modes = ('normal', 'no_relay', 'no_gate', 'open_gate')

    def __init__(self, width, regions: int = 3):
        super().__init__()
        self.regions = regions
        self.relay = nn.Sequential(nn.Linear(2 * width, width), nn.Tanh())
        self.pulvinar = nn.Sequential(nn.Linear(2 * width + 4, width), nn.SiLU(),
                                      nn.Linear(width, regions))
        self.reticular = PositiveLinear(width)
        self.bias = nn.Parameter(torch.zeros(regions))
        # r8.2 loops (caller flags gate their use): basal-ganglia output
        # disinhibits the thalamic gate (motor-thalamus closure); arousal
        # (drive curiosity EMA) widens it (exploration mode).
        self.bg_gain = nn.Parameter(torch.tensor(0.))
        self.arousal_gain = nn.Parameter(torch.tensor(0.))
        self.cb_thalamic_gain = nn.Parameter(torch.tensor(0.))

    def forward(self, x: Tensor, goal: Tensor, stats: Tensor, *, mode='normal',
                selection_output: Tensor | None = None, arousal: Tensor | None = None,
                surprise: Tensor | None = None):
        b, d = x.shape
        if goal.shape != x.shape or stats.shape != (b, 4):
            raise ValueError('Goal and four state statistics required')
        if mode not in self.modes:
            raise ValueError('Unknown thalamic intervention')
        relayed = self.relay(torch.cat((x, goal), -1))
        suppression = self.reticular(x).sigmoid().mean(-1)
        gate = (self.pulvinar(torch.cat((x, goal, stats), -1)).sigmoid()
                * (1 - suppression[:, None]) + self.bias.sigmoid())
        if selection_output is not None:
            # striatum->GPi->thalamus disinhibition: strong BG output closes
            # the gate; its release lets observation content through.
            # A bounded positive gain with a nonzero derivative at the default
            # initialization. abs(tanh(0)) made this connection unlearnable.
            gate = gate * (1 - self.bg_gain.sigmoid() * selection_output[:, None])
        if arousal is not None:
            gate = gate * (1 + self.arousal_gain.tanh() * arousal[:, None])
        if surprise is not None:
            # L3 (r8.4): cerebellar prediction error opens the gate — predictable
            # content passes attenuated, surprising content boosted.
            gate = gate * (1 + self.cb_thalamic_gain.tanh() * surprise[:, None])
        if mode == 'no_relay':
            relayed = torch.zeros_like(relayed)
        if mode == 'no_gate':
            gate = torch.ones_like(gate)
        if mode == 'open_gate':
            gate = torch.ones_like(gate)
            relayed = x
        throughput = gate.mean(-1, keepdim=True)
        populations = torch.stack((relayed, throughput.expand_as(relayed),
                                   suppression[:, None].expand_as(relayed)), 1)
        return ThalamicResult(relayed * throughput, populations, gate)


class DualStreams(nn.Module):
    """Per-region L5 intratelencephalic / pyramidal-tract analogue streams.

    The IT stream stays associative (feeds recurrent cortex); the PT stream is
    the output-driving population that projects into striatal selection.
    Goldberg 2004's cell-type separation is the correspondence; the claim is
    architectural, not histological.
    """
    def __init__(self, width):
        super().__init__()
        self.it = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, width))
        self.pt = nn.Sequential(nn.Linear(3 * width, width), nn.SiLU(), nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)

    def forward(self, top: Tensor, selected: Tensor, goal: Tensor):
        if selected.shape != top.shape or goal.shape != top.shape:
            raise ValueError('Aligned top/selection/goal required')
        it = self.norm(self.it(torch.cat((top, selected), -1)))
        pt = self.pt(torch.cat((top, selected, goal), -1)).tanh()
        return it, pt


class DopamineSystem(nn.Module):
    """Internal temporal-difference neuromodulation; teacher-free by contract.

    The prediction head estimates the next observation encoding from the
    current judgment context; realized signal is the actual next encoded
    observation at the following scan position. r_t is the negative mismatch;
    da combines r_t with the value change. Kravitz 2012's dMSN-persistent /
    iMSN-transient asymmetry is implemented as asymmetric plasticity gains on
    the direct/indirect selection populations during commit.

    MB-in-target mode (Priestley & Akam 2026, eLife 111458): the model-based
    value enters ONLY the TD target, never the prediction — the normative
    account of dopamine ramps. target = gamma*V_MB(s') + r, rpe = target - V_cached(s).
    """
    def __init__(self, width):
        super().__init__()
        self.critic = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, 1))
        self.mb_critic = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
        self.gamma = nn.Parameter(torch.tensor(1.7))  # sigmoid ≈ 0.85
        self.decay = nn.Parameter(torch.tensor(2.))

    def value(self, top: Tensor, selection: Tensor) -> Tensor:
        return self.critic(torch.cat((top, selection), -1)).squeeze(-1)

    def mb_value(self, inferred_state: Tensor) -> Tensor:
        """Value of the model-based (hippocampal/world-model) inferred state."""
        return self.mb_critic(inferred_state).squeeze(-1)

    def forward(self, value_now: Tensor, realized: Tensor, da_prev: Tensor,
                mb_value: Tensor | None = None) -> Tensor:
        if realized.shape != value_now.shape or da_prev.shape != value_now.shape:
            raise ValueError('Aligned value/realized/previous-dopamine required')
        target = realized if mb_value is None else \
            self.gamma.sigmoid() * mb_value + realized
        rpe = target - value_now
        da = torch.tanh(rpe) + self.decay.sigmoid() * da_prev * (1 - torch.tanh(rpe).abs())
        return da, rpe


class Cerebellum(nn.Module):
    """Fast supervised-error predictor parallel to slow cortex.

    A fixed random granule expansion (Marr-Albus analogue) feeds a learned
    Purkinje predictor of the NEXT observation encoding; prediction error is
    the deep-nucleus output injected as anticipatory context into the
    following position (Giovannucci 2017: anticipatory granule activity).
    Learning is self-supervised on the next encoded observation, teacher-free.
    """
    def __init__(self, width, expansion: int = 4):
        super().__init__()
        self.width = width
        self.expansion = expansion
        rng = torch.Generator(device='cpu').manual_seed(5150 + width)
        self.register_buffer('granule_projection',
                             torch.randn(width, width * expansion, generator=rng) / width ** .5)
        self.purkinje = nn.Sequential(nn.Linear(width * expansion + width, width), nn.SiLU(),
                                      nn.Linear(width, width))
        self.inhibitory = PositiveLinear(width)
        self.context_gain = nn.Parameter(torch.tensor(0.0))

    def forward(self, encoded_now: Tensor, state_prediction: Tensor | None,
                *, mode='normal'):
        """state_prediction: previous Purkinje output made for THIS position."""
        if mode not in ('normal', 'lesion', 'no_context'):
            raise ValueError('Unknown cerebellar intervention')
        granule = torch.matmul(encoded_now, self.granule_projection)
        prediction = self.purkinje(torch.cat((granule, encoded_now), -1))
        if mode == 'lesion':
            prediction = torch.zeros_like(prediction)
        error = (None if state_prediction is None else state_prediction - encoded_now)
        if error is None or mode == 'no_context':
            context = torch.zeros_like(encoded_now)
        else:
            context = -error * (1 - self.inhibitory(encoded_now).sigmoid())
        return prediction, error, self.context_gain.tanh() * context


class Neuromodulators(nn.Module):
    """Global DA / ACh / NE scalars modulating plasticity and throughput.

    DA arrives from DopamineSystem. ACh (learning-versus-recall mode, and the
    Hagger-Vaughan 2024 plateau correspondence for apical-side plasticity)
    rises with observation novelty/mismatch; NE rises with posterior
    uncertainty. Consumers: write gate, association rate, consolidation rate,
    read temperature, cortical update rates. All effects are multiplicative
    gates around 1 so ablation ('off') reproduces ancestor behavior exactly.
    """
    names = ('da', 'ach', 'ne')

    def __init__(self, width, use_drift: bool = False):
        super().__init__()
        if type(use_drift) is not bool:
            raise ValueError('Explicit drift-channel policy required')
        self.use_drift = use_drift
        self.head = nn.Sequential(
            nn.Linear(width + (4 if use_drift else 3), width), nn.SiLU(), nn.Linear(width, 3))
        self.sensitivity = nn.Parameter(torch.zeros(3))

    def forward(self, pooled: Tensor, mismatch: Tensor, uncertainty: Tensor, da: Tensor,
                drift: Tensor | None = None) -> dict:
        if mismatch.ndim != 1 or uncertainty.ndim != 1 or da.ndim != 1:
            raise ValueError('One scalar per stream required for mismatch/uncertainty/da')
        if mismatch.shape != pooled.shape[:1] or uncertainty.shape != mismatch.shape or da.shape != mismatch.shape:
            raise ValueError('Aligned per-stream scalar statistics required')
        if self.use_drift:
            if drift is None or drift.shape != mismatch.shape:
                raise ValueError('Drift channel enabled: one aligned scalar required')
            stats = torch.stack((mismatch, uncertainty, da, drift), -1)
        else:
            if drift is not None:
                raise ValueError('Drift channel disabled')
            stats = torch.stack((mismatch, uncertainty, da), -1)
        raw = torch.tanh(self.head(torch.cat((pooled, stats), -1))) * self.sensitivity.exp()
        return dict(da=da, ach=raw[:, 1] * .5, ne=raw[:, 2] * .5)


class Amygdala(nn.Module):
    """Fast valence tagging that biases selection before slow evaluation.

    positive/negative valence scalars from the raw observation asymmetrically
    bias direct (approach) versus indirect (avoid) striatal populations.
    """
    def __init__(self, width):
        super().__init__()
        self.valence = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2))
        self.gain = nn.Parameter(torch.tensor(0.02))
        # r8.2: valence biases the dopamine reward signal (amygdala->VTA path)
        self.reward_gain = nn.Parameter(torch.tensor(0.))

    def forward(self, x: Tensor, *, mode='normal'):
        if mode not in ('normal', 'off'):
            raise ValueError('Unknown amygdala intervention')
        v = torch.tanh(self.valence(x)) * self.gain.tanh()
        if mode == 'off':
            v = torch.zeros_like(v)
        return v

    def reward_bias(self, v):
        if v is None:
            return None
        return self.reward_gain.tanh() * (v[:, 0] - v[:, 1])


class HypothalamicDrives(nn.Module):
    """Homeostatic drive scalars producing self-initiation pressure.

    curiosity (EMA of posterior entropy), error_pressure (EMA of prediction
    mismatch), idle (normalized writes since last query) form an internal goal
    bias and a self-initiate logit appended to the control head. This is the
    r9 autonomous-speech substrate; it never fabricates observations.
    """
    def __init__(self, width):
        super().__init__()
        self.drive = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        self.initiate = nn.Linear(3, 1)

    def forward(self, drives: Tensor) -> Tensor:
        if drives.shape[-1] != 3:
            raise ValueError('Three drive scalars required')
        return self.drive(drives).tanh(), self.initiate(drives).squeeze(-1)


def sparse_attention(scores: Tensor, mask: Tensor, inverse_temperature: Tensor,
                     top_fraction: float) -> Tensor:
    """CA3-style competitive retrieval: temperature sharpening + WTA top-k.

    Hard selection is nondifferentiable. Gradients flow through the selected
    softmax scores; this is not a straight-through estimator for unselected
    addresses. Tensor fractions select a separate budget for every stream.
    """
    sharpened = scores * inverse_temperature
    sharpened = sharpened.masked_fill(~mask, torch.finfo(sharpened.dtype).min)
    if isinstance(top_fraction, Tensor):
        if top_fraction.shape not in ((scores.shape[0],), (scores.shape[0], 1)):
            raise ValueError('One retrieval fraction per stream required')
        counts = (scores.shape[-1] * top_fraction.detach().reshape(-1)).round().long()
        counts = counts.clamp(min=1, max=scores.shape[-1])
        keep = int(counts.max().item())
        top = sharpened.topk(keep, dim=-1).indices
        selected = (torch.arange(keep, device=scores.device)[None, None, :]
                    < counts[:, None, None]).expand_as(top)
        hard_mask = torch.zeros_like(sharpened, dtype=torch.bool).scatter(-1, top, selected) & mask
    else:
        keep = min(scores.shape[-1], max(1, round(scores.shape[-1] * top_fraction)))
        top = sharpened.topk(keep, dim=-1).indices
        hard_mask = torch.zeros_like(sharpened, dtype=torch.bool).scatter(-1, top, True) & mask
    hard_weights = hard_mask.to(sharpened.dtype)
    weights = sharpened.masked_fill(~hard_mask, torch.finfo(sharpened.dtype).min)
    attention = weights.softmax(-1) * hard_weights
    attention = attention / attention.sum(-1, keepdim=True).clamp_min(torch.finfo(sharpened.dtype).tiny)
    return attention
