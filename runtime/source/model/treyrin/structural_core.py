"""Treyrin candidate core: explicit state, recurrent judgment/read coupling.

This is a trainable architectural hypothesis, not a claim of biological fidelity
or demonstrated language/decision skill. No oracle, external model or retrieval
service participates in the transition. Observations and hypotheses use distinct
interfaces; query-time transitions never commit factual experience.
"""
from dataclasses import asdict, dataclass, fields
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .decision_field import NativeDecisionField
from .regional_circuit import RegionalCircuit
from .hypothesis_population import AdaptiveHypotheses,HypothesisPopulation
from .episodic_policy import EpisodicPolicy
from .code_attention import TokenCodeAttention
from .cognitive_workspace import (CognitiveWorkspace,KINDS,STATUSES,WORKSPACE_FIELDS,
    ACCEPTANCE,SOURCE_KINDS,ACTIONS,GOAL_FIELDS)
from .hippocampal_circuit import HippocampalCircuit
from .r8_circuits import (Amygdala,Cerebellum,DopamineSystem,DualStreams,
    HypothalamicDrives,Neuromodulators,ThalamicGate,sparse_attention)


@dataclass(frozen=True)
class CoreConfig:
    width: int = 64
    levels: int = 3
    episode_slots: int = 32
    semantic_slots: int = 8
    latent_candidates: int = 4
    inner_rounds: int = 2
    body_rounds: int = 2
    description_encoder: str = 'causal_body'
    cognitive_clock: str = 'token'
    language_prior_gain: float = 0.
    shared_choice_readout: bool = False
    perception_memory: bool = False
    response_cognitive_stride: int = 1
    sparse_key_fraction: float = 0.25
    association_decay: float = 0.995
    consolidation_rate: float = 0.05
    judgment_mode: str = 'legacy'
    coupling_mode: str = 'embedding_rounds'
    layer_sites: tuple[int, ...] = ()
    regional_circuit: bool = False
    hypothesis_mode: str = 'fixed'
    hippocampal_circuit: bool = False
    # r8 one-shot integration flags. All-False reproduces the r7 ancestor
    # computation exactly; any r8 flag requires the regional architecture.
    thalamic_gate: bool = False
    cortical_streams: bool = False
    dopamine_plasticity: bool = False
    cerebellum: bool = False
    neuromodulation: bool = False
    amygdala: bool = False
    drives: bool = False
    competitive_retrieval: bool = False
    retrieval_top_fraction: float = 0.25
    adaptive_retrieval_k: bool = False
    deep_native_field: bool = False
    native_body_bridge: bool = False
    # r8.1 frontier adoptions (frontier-2026-adoption.md P0 items)
    dopamine_mb_target: bool = False
    dual_condition_write: bool = False
    bg_thalamic_loop: bool = False
    valence_dopamine: bool = False
    arousal_gating: bool = False
    ema_anchor_consolidation: bool = False
    body_drift_coupling: bool = False
    hippocampal_bg_loop: bool = False
    dopamine_write_gate: bool = False
    cerebellar_thalamic_gain: bool = False
    goal_gated_retrieval: bool = False
    abstain_read: bool = False
    per_episode_gate: bool = False
    evict_consolidate: bool = False
    archive_slots: int = 0
    echo_slots: int = 0
    echo_ngram: int = 2
    episodic_policy: bool = False
    preserve_free_capacity: bool = False
    token_code_attention: bool = False
    code_attention_hops: int = 1
    native_single_pass: bool = False
    cognitive_workspace: bool = False
    workspace_slots: int = 8
    mental_archive_slots: int = 0
    # D06 goal-joint control: ONE learned head chooses (goal slot, action).
    # Default False keeps the v6 ancestor computation and every existing named
    # parameter tensor unchanged; the head only exists when this flag is True.
    goal_joint_control: bool = False
    cortical_goal_control: bool = False
    cortical_control_calibration: bool = False
    contextual_goal_control: bool = False
    goal_joint_hidden: int = 16
    goal_evidence_slots: int = 8
    goal_act_resource_floor: float = 0.0

    @property
    def r8_active(self) -> bool:
        return any((self.thalamic_gate, self.cortical_streams, self.dopamine_plasticity,
                    self.cerebellum, self.neuromodulation, self.amygdala, self.drives,
                    self.competitive_retrieval, self.deep_native_field, self.native_body_bridge,
                    self.dopamine_mb_target, self.dual_condition_write,
                    self.bg_thalamic_loop, self.valence_dopamine, self.arousal_gating,
                    self.ema_anchor_consolidation, self.body_drift_coupling,
                    self.hippocampal_bg_loop, self.dopamine_write_gate,
                    self.cerebellar_thalamic_gain, self.adaptive_retrieval_k,
                    self.goal_gated_retrieval, self.abstain_read,
                    self.per_episode_gate, self.evict_consolidate,
                    self.archive_slots > 0, self.echo_slots > 0,self.episodic_policy,self.cognitive_workspace))

    def __post_init__(self):
        if self.cognitive_clock not in ('token','event'):raise ValueError('Explicit token/event cognitive clock required')
        if type(self.language_prior_gain)not in(float,int)or not 0<=self.language_prior_gain<=4:
            raise ValueError('Finite nonnegative language choice prior gain <=4 required')
        if type(self.perception_memory)is not bool or type(self.response_cognitive_stride)is not int or not 1<=self.response_cognitive_stride<=16:
            raise ValueError('Explicit perception memory and bounded positive cognitive stride required')
        for key in ('width', 'levels', 'episode_slots', 'semantic_slots',
                    'latent_candidates', 'inner_rounds', 'body_rounds'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        if self.width < 4 or self.levels < 2:
            raise ValueError('At least width four and two working levels required')
        if self.description_encoder not in ('causal_body', 'embedding_mean'):
            raise ValueError('Unknown public-description encoder')
        if self.judgment_mode not in ('legacy', 'separated'):
            raise ValueError('Unknown judgment/utility organization')
        if type(self.regional_circuit) is not bool:
            raise ValueError('Explicit regional-circuit flag required')
        if self.regional_circuit and (self.levels!=3 or self.judgment_mode!='separated'):
            raise ValueError('Regional circuit requires three compartments and separated judgments')
        if self.hypothesis_mode not in ('fixed','adaptive'):
            raise ValueError('Unknown internal hypothesis organization')
        if self.hypothesis_mode=='adaptive' and (self.judgment_mode!='separated' or self.latent_candidates<2):
            raise ValueError('Adaptive hypotheses require separated judgments and capacity >=2')
        if type(self.hippocampal_circuit) is not bool or (self.hippocampal_circuit and not self.regional_circuit):
            raise ValueError('Hippocampal circuit requires the explicit regional architecture')
        for flag in ('thalamic_gate','cortical_streams','dopamine_plasticity','cerebellum',
                     'neuromodulation','amygdala','drives','competitive_retrieval','deep_native_field',
                     'dopamine_mb_target','dual_condition_write','bg_thalamic_loop',
                     'valence_dopamine','arousal_gating','ema_anchor_consolidation',
                     'body_drift_coupling','hippocampal_bg_loop','dopamine_write_gate',
                     'cerebellar_thalamic_gain','adaptive_retrieval_k','native_body_bridge',
                     'goal_gated_retrieval','abstain_read','per_episode_gate',
                     'evict_consolidate','episodic_policy','cognitive_workspace',
                     'preserve_free_capacity','token_code_attention','native_single_pass','goal_joint_control','contextual_goal_control','shared_choice_readout'):
            if type(getattr(self, flag)) is not bool:
                raise ValueError(f'{flag} must be an explicit bool')
        if self.r8_active and not self.regional_circuit:
            raise ValueError('r8 structures require the explicit regional architecture')
        if type(self.workspace_slots) is not int or self.workspace_slots<2:
            raise ValueError('At least two typed workspace slots required')
        if type(self.mental_archive_slots)is not int or not 0<=self.mental_archive_slots<=32768:raise ValueError('Bounded mental archive slots required')
        if self.mental_archive_slots and not self.cognitive_workspace:raise ValueError('Non-fact archive requires typed workspace')
        if self.goal_joint_control and not self.cognitive_workspace:
            raise ValueError('Goal-joint control requires the cognitive workspace')
        if self.contextual_goal_control and not self.goal_joint_control:raise ValueError('Contextual cognition requires the joint neural head')
        if type(self.cortical_goal_control) is not bool:raise ValueError('Explicit same-body cortical control flag required')
        if self.cortical_goal_control and not (self.contextual_goal_control and self.shared_choice_readout):raise ValueError('Cortical control requires contextual joint policy and shared cortical choice readout')
        if type(self.cortical_control_calibration) is not bool or (self.cortical_control_calibration and not self.cortical_goal_control):raise ValueError('Cortical calibration requires the same-body cortical controller')
        if type(self.code_attention_hops)is not int or self.code_attention_hops not in(1,2):raise ValueError('Bounded code attention hops 1 or 2 required')
        if type(self.goal_joint_hidden) is not int or self.goal_joint_hidden < 1:
            raise ValueError('goal_joint_hidden must be a positive integer')
        if type(self.goal_evidence_slots) is not int or self.goal_evidence_slots < 1:
            raise ValueError('goal_evidence_slots must be a positive integer')
        if type(self.goal_act_resource_floor) is not float or not 0 <= self.goal_act_resource_floor <= 1:
            raise ValueError('goal_act_resource_floor must be a float in [0,1]')
        if self.cognitive_workspace and not (self.episodic_policy and self.drives and self.hypothesis_mode=='adaptive'):
            raise ValueError('Persistent cognition requires episodic policy, drives and adaptive hypotheses')
        if (self.preserve_free_capacity or self.token_code_attention)and not self.episodic_policy:
            raise ValueError('Capacity preservation/code attention requires physical episodic identities')
        if self.native_single_pass and (self.cognitive_clock!='event'or not self.perception_memory):
            raise ValueError('Native single-pass requires final-position perception')
        if type(self.retrieval_top_fraction) is not float or not 0 < self.retrieval_top_fraction <= 1:
            raise ValueError('retrieval_top_fraction must be a float in (0,1]')
        if type(self.archive_slots) is not int or self.archive_slots < 0:
            raise ValueError('archive_slots must be a nonnegative integer')
        if self.archive_slots > 0 and self.episode_slots >= self.archive_slots:
            raise ValueError('archive must exceed the hot ring to add capacity')
        if type(self.echo_slots) is not int or self.echo_slots < 0:
            raise ValueError('echo_slots must be a nonnegative integer')
        if type(self.echo_ngram) is not int or self.echo_ngram < 1:
            raise ValueError('echo_ngram must be a positive integer')
        if self.dopamine_plasticity and not (self.cerebellum or self.hippocampal_circuit):
            raise ValueError('Internal dopamine needs a prediction-error source (cerebellum or hippocampus)')
        if self.dopamine_mb_target and not self.dopamine_plasticity:
            raise ValueError('MB-in-target dopamine requires the dopamine system')
        if self.bg_thalamic_loop and not (self.thalamic_gate and self.regional_circuit):
            raise ValueError('BG-thalamic loop requires thalamic gate and regional circuit')
        if self.valence_dopamine and not (self.amygdala and self.dopamine_plasticity):
            raise ValueError('Valence-dopamine requires amygdala and dopamine system')
        if self.arousal_gating and not (self.thalamic_gate and self.drives):
            raise ValueError('Arousal gating requires thalamic gate and drives')
        if self.body_drift_coupling and not self.neuromodulation:
            raise ValueError('Body-drift coupling rides the neuromodulator channel')
        if self.hippocampal_bg_loop and not (self.regional_circuit and self.hippocampal_circuit):
            raise ValueError('Hippocampal-BG loop requires regional circuit and hippocampus')
        if self.dopamine_write_gate and not (self.dopamine_plasticity and self.hippocampal_circuit):
            raise ValueError('DA write gate requires dopamine and hippocampal fast weights')
        if self.cerebellar_thalamic_gain and not (self.cerebellum and self.thalamic_gate):
            raise ValueError('Cerebellar-thalamic gain requires cerebellum and thalamic gate')
        if self.goal_gated_retrieval and not self.competitive_retrieval:
            raise ValueError('Goal-gated retrieval rides the competitive retrieval path')
        if self.abstain_read and self.judgment_mode != 'separated':
            raise ValueError('Abstain read requires separated judgments')
        if self.per_episode_gate and not self.competitive_retrieval:
            raise ValueError('Per-episode gating rides the competitive retrieval path')
        if self.coupling_mode not in ('embedding_rounds', 'layer_interleaved'):
            raise ValueError('Unknown language/core coupling')
        if self.coupling_mode == 'layer_interleaved':
            if self.body_rounds != 1 or not self.layer_sites:
                raise ValueError('Layer-interleaved execution needs one body pass and explicit visits')
            if tuple(self.layer_sites) != tuple(sorted(set(self.layer_sites))) or any(type(i) is not int or i < 0 for i in self.layer_sites):
                raise ValueError('Distinct increasing nonnegative layer sites required')
        elif self.layer_sites:
            raise ValueError('Layer sites have no meaning in embedding-round control')
        if not 0 < self.sparse_key_fraction <= 1:
            raise ValueError('Invalid sparse key fraction')
        if not 0 < self.association_decay <= 1 or not 0 < self.consolidation_rate <= 1:
            raise ValueError('Invalid state time scales')


@dataclass
class ExperienceState:
    working: Tensor             # B,L,D; last committed observation context
    keys: Tensor                # B,E,D; sparse episode addresses
    values: Tensor              # B,E,D; observed-event content
    sources: Tensor             # B,E,D; public provenance representation
    occupied: Tensor            # B,E bool
    association: Tensor         # B,D,D; fast learned delta-update state
    semantic: Tensor            # B,S,D; explicit, slower consolidation state
    writes: Tensor              # B long; physical events, not capacity claims
    consolidations: Tensor      # B long
    binding: str
    cortex: Tensor | None = None
    relay: Tensor | None = None
    selection: Tensor | None = None
    hippocampal_population: Tensor | None = None
    hippocampal_fast: Tensor | None = None
    hippocampal_eligibility: Tensor | None = None
    hippocampal_updates: Tensor | None = None
    hippocampal_events: Tensor | None = None
    hippocampal_provenance: Tensor | None = None
    # r8 persistent circuit state (None in the all-False ancestor config).
    thalamic: Tensor | None = None            # B,3,D populations
    cerebellar_prediction: Tensor | None = None  # B,D; Purkinje output for next observation
    dopamine: Tensor | None = None            # B; EMA level at last commit
    neuromod_last: Tensor | None = None       # B,3; da/ach/ne scalars at last commit
    drive_state: Tensor | None = None         # B,3; curiosity/error/idle EMAs
    write_threshold: Tensor | None = None     # B,1; EMA of recent write-gate levels
    semantic_anchor: Tensor | None = None     # B,S,D; streaming-EMA slow-state anchor
    archive_keys: Tensor | None = None        # B,A,D; cold episodic addresses
    archive_values: Tensor | None = None      # B,A,D; cold episodic content
    archive_sources: Tensor | None = None     # B,A,D; cold episodic provenance
    archive_occupied: Tensor | None = None    # B,A bool
    archive_writes: Tensor | None = None      # B long
    echo_keys: Tensor | None = None           # B,C long; hashed n-gram slots (-1 empty)
    echo_values: Tensor | None = None         # B,C,D; detached content vectors, not verbatim text
    echo_count: Tensor | None = None          # B long
    episode_admissions: Tensor | None = None # B long; actual hot writes, distinct from observed events
    episode_event_ids: Tensor | None = None  # B,E long; one-based public journal positions, -1 empty
    archive_event_ids: Tensor | None = None  # B,A long; identities survive promotion
    admission_distribution: Tensor | None = None # B,E+1; skip/physical slot policy, not a belief
    workspace_vectors: Tensor | None = None
    workspace_sources: Tensor | None = None
    workspace_ids: Tensor | None = None
    workspace_kinds: Tensor | None = None
    workspace_statuses: Tensor | None = None
    workspace_seen: Tensor | None = None
    workspace_admissions: Tensor | None = None
    goal_vectors: Tensor | None = None
    goal_sources: Tensor | None = None
    goal_ids: Tensor | None = None
    goal_statuses: Tensor | None = None
    cognitive_ticks: Tensor | None = None
    deliberations: Tensor | None = None
    elapsed_seconds: Tensor | None = None
    deadline_seconds: Tensor | None = None
    cognitive_context: Tensor | None = None
    cognitive_distribution: Tensor | None = None
    mental_archive_vectors: Tensor | None = None
    mental_archive_sources: Tensor | None = None
    mental_archive_ids: Tensor | None = None
    mental_archive_kinds: Tensor | None = None
    mental_archive_statuses: Tensor | None = None
    mental_archive_admissions: Tensor | None = None
    # D06 typed goal rows, initialised only when config.goal_joint_control.
    # Epoch seconds stay float64: float32 at 1.7e9 resolves to about 128 s.
    goal_priority: Tensor | None = None        # B,G float32
    goal_icost: Tensor | None = None           # B,G float32
    goal_deadline_at: Tensor | None = None     # B,G float64
    goal_deadline_mask: Tensor | None = None   # B,G long
    goal_accept: Tensor | None = None          # B,G long acceptance code
    goal_sources_typed: Tensor | None = None   # B,G long source-kind code
    goal_evidence: Tensor | None = None        # B,G,E bool referenced identities
    goal_ev_ok: Tensor | None = None           # B,G long satisfied citations
    goal_ev_open: Tensor | None = None         # B,G long declared gaps
    goal_attempts: Tensor | None = None        # B,G long
    goal_last_action: Tensor | None = None     # B,G float64
    goal_revision: Tensor | None = None        # B,G long revision depth
    goal_can_act: Tensor | None = None         # B,G long autonomy permission
    cognitive_joint: Tensor | None = None      # B,R,A joint (goal, action) allocation
    cognitive_goal: Tensor | None = None       # B long selected goal row
    cognitive_goal_identity: Tensor | None = None  # B selected record identity, safe across slot reuse

    def tensors(self):
        return {f.name: getattr(self, f.name) for f in fields(self)
                if f.name != 'binding' and getattr(self,f.name) is not None}

    def detached(self):
        return ExperienceState(**{k: v.detach().clone() for k, v in self.tensors().items()},
                               binding=self.binding)


@dataclass
class CoreStep:
    working: Tensor
    shared: Tensor
    native_logits: Tensor
    control_logits: Tensor       # continue / wait / request-evidence, not calibrated
    predicted_outcomes: Tensor   # B,K,D; needs result-grounded supervision
    candidate_reads: Tensor      # B,K,E
    route_weights: Tensor        # B,6
    predicate_logits: Tensor | None = None
    action_values: Tensor | None = None
    read_allocation: Tensor | None = None
    cortex: Tensor | None = None
    relay: Tensor | None = None
    selection: Tensor | None = None
    regional_edges: Tensor | None = None
    hypotheses: HypothesisPopulation | None = None
    hippocampal_population: Tensor | None = None
    hippocampal_mismatch: Tensor | None = None
    # r8 per-step scratch outputs
    thalamic: Tensor | None = None            # B,3,D
    cerebellar_prediction: Tensor | None = None  # B,D; prediction for the NEXT position
    cerebellar_error: Tensor | None = None    # B,D; realized prediction error
    dopamine: Tensor | None = None            # B; current level
    neuromod: Tensor | None = None            # B,3; da/ach/ne
    drive_vector: Tensor | None = None        # B,D
    initiate_logit: Tensor | None = None      # B; self-initiation pressure
    cognitive_control_logits: Tensor | None = None # B,5; wait/think/speak/ask/act
    soft_candidate_reads: Tensor | None = None # loss-only dense addresses before hard competition


def poly_hash(ids: Tensor, slots: int, ngram: int, prime: int = 1000003) -> Tensor:
    """Deterministic rolling poly-hash of the last ngram token ids -> slot id.
    Pure torch; identical across processes (no Python hash)."""
    tail = ids[:, -ngram:] if ids.shape[1] >= ngram else ids
    h = torch.zeros(ids.shape[0], dtype=torch.long, device=ids.device)
    for column in range(tail.shape[1]):
        h = (h * prime + tail[:, column].long()) % slots
    return h


def causal_ngram_fingerprints(ids: Tensor, mask: Tensor, ngram: int,
                              prime: int = 1000003) -> Tensor:
    """Per-prefix fingerprints of the last n valid tokens, never future ids.

    Fingerprints and table addresses are separate. This detects bucket
    collisions; a finite fingerprint is not a promise of verbatim storage.
    Masked tokens do not enter the stream, including left/right padding.
    """
    if ids.ndim != 2 or mask.shape != ids.shape or ngram < 1:
        raise ValueError('Aligned token/mask matrices and positive ngram required')
    ranks = mask.long().cumsum(-1) - 1
    packed = torch.zeros_like(ids)
    packed.scatter_add_(1, ranks.clamp_min(0), (ids + 1) * mask.long())
    hashes = torch.zeros_like(ids)
    for lag in range(ngram - 1, -1, -1):
        index = ranks - lag
        token = packed.gather(1, index.clamp_min(0))
        token = torch.where(index >= 0, token, torch.zeros_like(token))
        hashes = (hashes * prime + token) % 4294967291
    return hashes


def masked_distribution(scores: Tensor, mask: Tensor) -> Tensor:
    """An empty memory produces exactly zero weights, never a NaN/uniform fact."""
    mask = mask.expand_as(scores)
    safe = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(safe, dim=-1) * mask.to(scores.dtype)
    return weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(scores.dtype).tiny)


class StructuralCore(nn.Module):
    def __init__(self, config: CoreConfig):
        super().__init__()
        self.config = config
        if config.token_code_attention:self.code_attention=TokenCodeAttention(hops=config.code_attention_hops)
        d = config.width
        self.key = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d, bias=False)
        self.source = nn.Linear(d, d, bias=False)
        self.read_query = nn.Linear(d * 2, d, bias=False)
        self.context = nn.Linear(d * 2, d)
        self.route = nn.Sequential(nn.Linear(d * 2 + 2, d), nn.SiLU(), nn.Linear(d, 6))
        self.proposals = nn.ModuleList(nn.Linear(4 * d, d) for _ in range(config.levels))
        self.update_gates = nn.ModuleList(nn.Linear(4 * d, d) for _ in range(config.levels))
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(config.levels))
        self.outcome = nn.Sequential(nn.Linear(3 * d, d), nn.Tanh())
        if config.judgment_mode == 'separated':
            # The frozen r2 legacy control keeps its original parameter set.
            # These fields participate INSIDE each read/work/language cycle.
            self.native_field = NativeDecisionField(d, deep=config.deep_native_field,
                abstain=config.abstain_read,body_bridge=config.native_body_bridge)
        else:
            self.judge = nn.Sequential(nn.Linear(4 * d, d), nn.SiLU(), nn.Linear(d, 1))
        if config.goal_gated_retrieval:
            # Created AFTER the judgment branch so the flag-off parameter
            # stream is bit-identical to the ancestor (an extra module would
            # consume RNG draws and silently shift every later init).
            self.goal_retrieval_gate = nn.Linear(2 * d, 1)
        if config.per_episode_gate:
            # Explicit slot-query interaction can change which slots pass.
            # Concatenating a query into a single linear gate only adds the
            # same query offset to every slot's pre-sigmoid score.
            self.episode_gate = nn.Linear(4 * d, 1)
        if config.echo_slots > 0:
            # Bounded observation-echo gate; not the static parameter lookup
            # mechanism of the Engram paper.
            self.echo_gain = nn.Parameter(torch.tensor(0.))
        self.control = nn.Linear(d, 3)
        self.write_gate = nn.Linear(2 * d, 1)
        self.semantic_queries = nn.Parameter(torch.randn(config.semantic_slots, d) * 0.02)
        self.latent_candidates = nn.Parameter(torch.randn(config.latent_candidates, d) * 0.02)
        self.register_buffer('level_rates', torch.linspace(1.0, 0.25, config.levels))
        if config.regional_circuit:
            self.regional = RegionalCircuit(d, context_bias=config.hippocampal_bg_loop)
        if config.dopamine_write_gate:
            self.da_write_gain = nn.Parameter(torch.tensor(0.))
        if config.hypothesis_mode=='adaptive':
            self.hypotheses=AdaptiveHypotheses(d,config.latent_candidates)
        if config.hippocampal_circuit:
            self.hippocampus=HippocampalCircuit(d)
            self.hippocampal_gain=nn.Parameter(torch.tensor(.02))
        if config.thalamic_gate:
            self.thalamus=ThalamicGate(d)
            self.thalamic_gain=nn.Parameter(torch.tensor(0.02))
        if config.cortical_streams:
            self.streams=DualStreams(d)
            self.stream_gain=nn.Parameter(torch.tensor(0.02))
        if config.dopamine_plasticity:
            self.dopamine_system=DopamineSystem(d)
        if config.cerebellum:
            self.cerebellum=Cerebellum(d)
        if config.neuromodulation:
            self.neuromod=Neuromodulators(d, use_drift=config.body_drift_coupling)
            if config.dopamine_plasticity is False:
                raise ValueError('Neuromodulation consumes the dopamine system output')
        if config.ema_anchor_consolidation:
            self.anchor_pull=nn.Parameter(torch.tensor(0.3))
            self.anchor_rate=nn.Parameter(torch.tensor(1.5))
        if config.amygdala:
            self.amygdala=Amygdala(d)
            self.valence_projection=nn.Linear(2,d,bias=False)
        if config.drives:
            self.drives=HypothalamicDrives(d)
            self.drive_gain=nn.Parameter(torch.tensor(0.02))
        if config.competitive_retrieval:
            self.retrieval_temperature=nn.Parameter(torch.tensor(0.))
        if config.episodic_policy:
            self.episodic_policy=EpisodicPolicy(d)
        if config.cognitive_workspace:
            self.workspace=CognitiveWorkspace(d,slots=config.workspace_slots,joint=config.goal_joint_control,joint_hidden=config.goal_joint_hidden,contextual=config.contextual_goal_control,calibration=config.cortical_control_calibration)

    def initial_state(self, batch: int, binding: str) -> ExperienceState:
        if type(batch) is not int or batch < 1 or not binding:
            raise ValueError('Positive batch and nonempty parameter binding required')
        p, c = next(self.parameters()), self.config
        z = lambda *shape: p.new_zeros(batch, *shape)
        result = ExperienceState(z(c.levels, c.width), z(c.episode_slots, c.width),
            z(c.episode_slots, c.width), z(c.episode_slots, c.width),
            torch.zeros(batch, c.episode_slots, device=p.device, dtype=torch.bool),
            z(c.width, c.width), z(c.semantic_slots, c.width),
            torch.zeros(batch, device=p.device, dtype=torch.long),
            torch.zeros(batch, device=p.device, dtype=torch.long), binding)
        if c.regional_circuit:
            result.cortex,result.relay,result.selection = z(3,3,c.width),z(4,c.width),z(5,c.width)
        if c.hippocampal_circuit:
            result.hippocampal_population=z(len(HippocampalCircuit.populations),c.width)
            result.hippocampal_fast,result.hippocampal_eligibility=z(c.width,c.width),z(c.width,c.width)
            result.hippocampal_updates=torch.zeros(batch,device=p.device,dtype=torch.long)
            result.hippocampal_events,result.hippocampal_provenance=z(c.episode_slots,c.width),z(c.episode_slots,c.width)
        if c.thalamic_gate:
            result.thalamic=z(len(ThalamicGate.populations),c.width)
        if c.cerebellum:
            result.cerebellar_prediction=z(c.width)
        if c.dopamine_plasticity:
            result.dopamine=z(1).squeeze(-1)
        if c.neuromodulation:
            result.neuromod_last=z(3)
        if c.drives:
            result.drive_state=z(3)
        if c.dual_condition_write:
            result.write_threshold=result.write_threshold if False else z(1)
        if c.ema_anchor_consolidation:
            result.semantic_anchor=z(c.semantic_slots, c.width)
        if c.archive_slots > 0:
            result.archive_keys=result.archive_values=result.archive_sources=\
                z(c.archive_slots, c.width)
            result.archive_occupied=torch.zeros(batch, c.archive_slots,
                                                device=p.device, dtype=torch.bool)
            result.archive_writes=torch.zeros(batch, device=p.device, dtype=torch.long)
        if c.echo_slots > 0:
            result.echo_keys=torch.full((batch, c.echo_slots), -1,
                                        device=p.device, dtype=torch.long)
            result.echo_values=z(c.echo_slots, c.width)
            result.echo_count=torch.zeros(batch, device=p.device, dtype=torch.long)
        if c.episodic_policy:
            result.episode_admissions=torch.zeros(batch,device=p.device,dtype=torch.long)
            result.episode_event_ids=torch.full((batch,c.episode_slots),-1,device=p.device,dtype=torch.long)
            result.admission_distribution=z(c.episode_slots+1)
            result.admission_distribution[:,0]=1.
            if c.archive_slots>0:
                result.archive_event_ids=torch.full((batch,c.archive_slots),-1,device=p.device,dtype=torch.long)
        if c.cognitive_workspace:
            result.workspace_vectors=z(c.workspace_slots,c.width)
            result.workspace_sources=z(c.workspace_slots,c.width)
            result.workspace_ids=torch.full((batch,c.workspace_slots),-1,device=p.device,dtype=torch.long)
            result.workspace_kinds=torch.zeros(batch,c.workspace_slots,device=p.device,dtype=torch.long)
            result.workspace_statuses=torch.zeros_like(result.workspace_kinds)
            result.goal_vectors=z(c.workspace_slots,c.width);result.goal_sources=z(c.workspace_slots,c.width)
            result.goal_ids=torch.full((batch,c.workspace_slots),-1,device=p.device,dtype=torch.long)
            result.goal_statuses=torch.zeros_like(result.goal_ids)
            for name in ('workspace_seen','workspace_admissions','cognitive_ticks','deliberations'):
                setattr(result,name,torch.zeros(batch,device=p.device,dtype=torch.long))
            result.elapsed_seconds=z(1).squeeze(-1)
            result.deadline_seconds=z(1).squeeze(-1)
            result.cognitive_context=z(6);result.cognitive_context[:,4]=1.
            result.cognitive_distribution=z(5);result.cognitive_distribution[:,0]=1.
            if c.goal_joint_control:
                result.goal_priority=z(c.workspace_slots);result.goal_icost=z(c.workspace_slots)
                result.goal_deadline_at=z(c.workspace_slots).double()
                result.goal_last_action=z(c.workspace_slots).double()
                result.goal_evidence=torch.zeros(batch,c.workspace_slots,c.goal_evidence_slots,
                                                 device=p.device,dtype=torch.bool)
                for name in ('goal_deadline_mask','goal_accept','goal_sources_typed','goal_ev_ok','goal_ev_open',
                             'goal_attempts','goal_revision','goal_can_act'):
                    setattr(result,name,torch.zeros(batch,c.workspace_slots,device=p.device,dtype=torch.long))
                result.cognitive_joint=z(c.workspace_slots+1,len(ACTIONS))
                result.cognitive_joint[:,-1,0]=1.  # before any tick: wait, no goal
                result.cognitive_goal=torch.full((batch,),c.workspace_slots,device=p.device,dtype=torch.long)
                if c.cortical_goal_control:result.cognitive_goal_identity=torch.full((batch,),-1,device=p.device,dtype=torch.long)
        from .mental_archive import initialize
        result=initialize(result,c.mental_archive_slots,c.width)
        return result

    def validate_state(self, state: ExperienceState, batch: int):
        c, ref = self.config, next(self.parameters())
        from .mental_archive import validate
        validate(state,c.mental_archive_slots,c.width)
        shapes = dict(working=(batch,c.levels,c.width), keys=(batch,c.episode_slots,c.width),
            values=(batch,c.episode_slots,c.width), sources=(batch,c.episode_slots,c.width),
            occupied=(batch,c.episode_slots), association=(batch,c.width,c.width),
            semantic=(batch,c.semantic_slots,c.width), writes=(batch,), consolidations=(batch,))
        if c.regional_circuit:
            shapes.update(cortex=(batch,3,3,c.width),relay=(batch,4,c.width),selection=(batch,5,c.width))
        elif any(getattr(state,name,None) is not None for name in ('cortex','relay','selection')):
            raise ValueError('Unexpected regional state in the ancestor control')
        hippocampal_names=('hippocampal_population','hippocampal_fast','hippocampal_eligibility',
                           'hippocampal_updates','hippocampal_events','hippocampal_provenance')
        if c.hippocampal_circuit:
            shapes.update(hippocampal_population=(batch,len(HippocampalCircuit.populations),c.width),
                hippocampal_fast=(batch,c.width,c.width),hippocampal_eligibility=(batch,c.width,c.width),
                hippocampal_updates=(batch,),hippocampal_events=(batch,c.episode_slots,c.width),
                hippocampal_provenance=(batch,c.episode_slots,c.width))
        elif any(getattr(state,name,None) is not None for name in hippocampal_names):
            raise ValueError('Unexpected hippocampal state in the ancestor control')
        r8_names=('thalamic','cerebellar_prediction','dopamine','neuromod_last','drive_state',
                  'write_threshold','semantic_anchor','archive_keys','archive_values',
                  'archive_sources','archive_occupied','archive_writes')
        if any(getattr(state,name,None) is not None for name in r8_names) and not c.r8_active:
            raise ValueError('Unexpected r8 state in the ancestor control')
        if c.thalamic_gate: shapes.update(thalamic=(batch,len(ThalamicGate.populations),c.width))
        if c.cerebellum: shapes.update(cerebellar_prediction=(batch,c.width))
        if c.dopamine_plasticity: shapes.update(dopamine=(batch,))
        if c.neuromodulation: shapes.update(neuromod_last=(batch,3))
        if c.drives: shapes.update(drive_state=(batch,3))
        if c.dual_condition_write: shapes.update(write_threshold=(batch,1))
        if c.ema_anchor_consolidation: shapes.update(semantic_anchor=(batch,c.semantic_slots,c.width))
        if c.archive_slots > 0:
            shapes.update(archive_keys=(batch,c.archive_slots,c.width),
                archive_values=(batch,c.archive_slots,c.width),
                archive_sources=(batch,c.archive_slots,c.width),
                archive_occupied=(batch,c.archive_slots),archive_writes=(batch,))
        if c.echo_slots > 0:
            shapes.update(echo_keys=(batch,c.echo_slots),
                echo_values=(batch,c.echo_slots,c.width),echo_count=(batch,))
        policy_names=('episode_admissions','episode_event_ids','archive_event_ids','admission_distribution')
        if c.episodic_policy:
            shapes.update(episode_admissions=(batch,),episode_event_ids=(batch,c.episode_slots),
                admission_distribution=(batch,c.episode_slots+1))
            if c.archive_slots>0:shapes.update(archive_event_ids=(batch,c.archive_slots))
        elif any(getattr(state,name) is not None for name in policy_names):
            raise ValueError('Unexpected episodic policy state')
        if c.cognitive_workspace:
            shapes.update(workspace_vectors=(batch,c.workspace_slots,c.width),workspace_sources=(batch,c.workspace_slots,c.width),
                workspace_ids=(batch,c.workspace_slots),workspace_kinds=(batch,c.workspace_slots),workspace_statuses=(batch,c.workspace_slots),
                goal_vectors=(batch,c.workspace_slots,c.width),goal_sources=(batch,c.workspace_slots,c.width),
                goal_ids=(batch,c.workspace_slots),goal_statuses=(batch,c.workspace_slots),
                workspace_seen=(batch,),workspace_admissions=(batch,),cognitive_ticks=(batch,),deliberations=(batch,),
                elapsed_seconds=(batch,),deadline_seconds=(batch,),cognitive_context=(batch,6),cognitive_distribution=(batch,5))
            if c.goal_joint_control:
                shapes.update(goal_priority=(batch,c.workspace_slots),goal_icost=(batch,c.workspace_slots),
                    goal_deadline_at=(batch,c.workspace_slots),goal_deadline_mask=(batch,c.workspace_slots),
                    goal_accept=(batch,c.workspace_slots),goal_sources_typed=(batch,c.workspace_slots),
                    goal_evidence=(batch,c.workspace_slots,c.goal_evidence_slots),
                    goal_ev_ok=(batch,c.workspace_slots),goal_ev_open=(batch,c.workspace_slots),
                    goal_attempts=(batch,c.workspace_slots),
                    goal_last_action=(batch,c.workspace_slots),goal_revision=(batch,c.workspace_slots),
                    goal_can_act=(batch,c.workspace_slots),
                    cognitive_joint=(batch,c.workspace_slots+1,len(ACTIONS)),cognitive_goal=(batch,))
                if c.cortical_goal_control:shapes['cognitive_goal_identity']=(batch,)
                elif state.cognitive_goal_identity is not None:raise ValueError('Unexpected cortical selected-goal identity')
            elif any(getattr(state,name,None) is not None for name in ('goal_priority','goal_evidence','cognitive_joint')):
                raise ValueError('Unexpected goal-joint state')
        elif any(getattr(state,name) is not None for name in WORKSPACE_FIELDS):raise ValueError('Unexpected cognitive workspace state')
        for name, shape in shapes.items():
            t = getattr(state, name)
            dtype = torch.bool if name in ('occupied','archive_occupied') else (
                torch.long if name in ('writes','consolidations','hippocampal_updates',
                                       'archive_writes','echo_keys','echo_count','episode_admissions',
                                       'episode_event_ids','archive_event_ids','workspace_ids','workspace_kinds','workspace_statuses',
                                       'workspace_seen','workspace_admissions','goal_ids','goal_statuses','cognitive_ticks','deliberations',
                                       'goal_deadline_mask','goal_accept','goal_sources_typed','goal_ev_ok','goal_ev_open',
                                       'goal_attempts','goal_revision','goal_can_act','cognitive_goal','cognitive_goal_identity') else
                torch.float64 if name in ('goal_deadline_at','goal_last_action') else
                torch.bool if name in ('goal_evidence',) else ref.dtype)
            if not isinstance(t,Tensor) or tuple(t.shape) != shape or t.device != ref.device or t.dtype != dtype:
                raise ValueError(f'Invalid state tensor: {name}')
            if dtype == ref.dtype and not torch.isfinite(t).all():
                raise ValueError(f'Nonfinite state: {name}')
        if (state.writes < 0).any() or (state.consolidations < 0).any():
            raise ValueError('Negative state counters')
        if c.hippocampal_circuit and ((state.hippocampal_updates<0).any() or (state.hippocampal_fast<0).any()):
            raise ValueError('Invalid hippocampal updates or negative excitatory fast connection')
        if c.episodic_policy:
            if ((state.episode_admissions<0)|(state.episode_admissions>state.writes)).any():
                raise ValueError('Invalid actual episode admission count')
            if not torch.equal(state.episode_event_ids>0,state.occupied):raise ValueError('Hot provenance identities differ')
            if (state.episode_event_ids>state.writes[:,None]).any():raise ValueError('Future event identity in hot memory')
            if c.archive_slots>0:
                if not torch.equal(state.archive_event_ids>0,state.archive_occupied):raise ValueError('Archive provenance identities differ')
                if (state.archive_event_ids>state.writes[:,None]).any():raise ValueError('Future event identity in archive')
            p=state.admission_distribution
            if ((p<0)|(p>1)).any() or not torch.allclose(p.sum(-1),torch.ones_like(p[:,0]),atol=1e-5):
                raise ValueError('Invalid skip/slot action distribution')
        if c.cognitive_workspace:
            if ((state.goal_ids>state.workspace_seen[:,None])|(state.goal_ids<-1)|(state.goal_ids==0)).any():raise ValueError('Invalid protected goal identity')
            if ((state.goal_statuses<0)|(state.goal_statuses>=len(STATUSES))).any():raise ValueError('Invalid protected goal status')
            if ((state.workspace_ids>state.workspace_seen[:,None])|(state.workspace_ids<-1)|(state.workspace_ids==0)).any():
                raise ValueError('Invalid mental identity: not a public event identity')
            if ((state.workspace_kinds<0)|(state.workspace_kinds>=len(KINDS))).any() or ((state.workspace_statuses<0)|(state.workspace_statuses>=len(STATUSES))).any():
                raise ValueError('Invalid mental origin/status code')
            if ((state.workspace_seen<0)|(state.workspace_admissions<0)|(state.workspace_admissions>state.workspace_seen)).any():
                raise ValueError('Invalid mental counters')
            if ((state.cognitive_ticks<0)|(state.deliberations<0)|(state.deliberations>state.cognitive_ticks)|(state.elapsed_seconds<0)).any():
                raise ValueError('Invalid opportunity/deliberation/elapsed clock')
            p=state.cognitive_distribution
            if ((p<0)|(p>1)).any() or not torch.allclose(p.sum(-1),torch.ones_like(p[:,0]),atol=1e-5):raise ValueError('Invalid cognitive action allocation')
            if c.goal_joint_control:
                if ((state.cognitive_goal<0)|(state.cognitive_goal>c.workspace_slots)).any():raise ValueError('Invalid selected goal row')
                if c.cortical_goal_control and bool(((state.cognitive_goal_identity<-1)|(state.cognitive_goal_identity==0)|(state.cognitive_goal_identity>state.workspace_seen)).any()):raise ValueError('Invalid selected mental-record identity')
                if ((state.goal_accept<0)|(state.goal_accept>=len(ACCEPTANCE))).any():raise ValueError('Invalid goal acceptance code')
                if ((state.goal_sources_typed<0)|(state.goal_sources_typed>=len(SOURCE_KINDS))).any():raise ValueError('Invalid goal source code')
                if ((state.goal_statuses!=0)&(state.goal_statuses>=len(STATUSES))).any():raise ValueError('Invalid goal status code')
                if ((state.goal_deadline_mask!=0)&(state.goal_deadline_mask!=1)).any():raise ValueError('Invalid goal deadline mask')
                if ((state.goal_priority<0)|(state.goal_priority>1)).any():raise ValueError('Invalid goal priority')
                if ((state.goal_icost<0)|(state.goal_ev_ok<0)|(state.goal_ev_open<0)|(state.goal_attempts<0)).any():raise ValueError('Invalid goal cost/gap counter')
                if ((state.goal_can_act<0)|(state.goal_can_act>1)).any():raise ValueError('Invalid goal autonomy')
                if ((state.goal_deadline_mask>0)&(state.goal_deadline_at<=0)).any():raise ValueError('Invalid goal deadline')
                q=state.cognitive_joint
                if ((q<0)|(q>1)).any() or not torch.allclose(q.sum((-1,-2)),torch.ones_like(q[:,0,0]),atol=1e-5):raise ValueError('Invalid joint goal/action allocation')

    def _sparse_key(self, x):
        key = torch.tanh(self.key(x))
        n = max(1, round(self.config.width * self.config.sparse_key_fraction))
        indices = key.abs().topk(n, dim=-1).indices
        selected = torch.zeros_like(key).scatter(-1, indices, key.gather(-1, indices))
        return F.normalize(selected, dim=-1)

    def read_evidence(self, query: Tensor, candidates: Tensor, state: ExperienceState,
                      posterior: Tensor, *, mode: Literal['candidate','entropy','uniform']='candidate',
                      goal: Tensor | None = None,echo_query_hash: Tensor | None = None,learning_trace=None):
        """Candidate identity and entropy-only feedback are separable interventions.

        `posterior` indexes the actual candidate vectors. Simultaneous permutation
        of vectors and weights is equivariance, NOT a wrong-content intervention.
        """
        b, k, d = candidates.shape
        q = query[:, None].expand(-1, k, -1)
        if mode == 'candidate':
            addresses = self.read_query(torch.cat((q, candidates), -1))
            mixing = posterior
            gain = torch.ones_like(posterior[:, :1])
        elif mode in ('entropy','uniform'):
            addresses = self.read_query(torch.cat((q, torch.zeros_like(candidates)), -1))
            mixing = torch.full_like(posterior, 1 / k)
            entropy = -(posterior * posterior.clamp_min(1e-12).log()).sum(-1, keepdim=True)
            gain = 1 + entropy / max(1.0, float(torch.log(torch.tensor(float(k))))) if mode == 'entropy' else 1
        else:
            raise ValueError(f'Unknown read intervention: {mode}')
        if self.config.archive_slots > 0 and state.archive_keys is not None:
            all_keys = torch.cat((state.keys, state.archive_keys), 1)
            all_srcs = torch.cat((state.sources, state.archive_sources), 1)
            all_occ = torch.cat((state.occupied, state.archive_occupied), 1)
            all_values = torch.cat((state.values, state.archive_values), 1)
        else:
            all_keys, all_srcs, all_occ = state.keys, state.sources, state.occupied
            all_values = state.values
        scores = torch.einsum('bkd,bed->bke', addresses, all_keys + all_srcs) / d**0.5
        if self.config.perception_memory:
            scores=scores+.25*torch.einsum('bkd,bed->bke',F.normalize(candidates,dim=-1),F.normalize(all_values,dim=-1))
        if self.config.competitive_retrieval:
            fraction = self.config.retrieval_top_fraction
            if self.config.adaptive_retrieval_k:
                # A1: activity-dependent k (theta-gamma capacity is bounded, not
                # fixed) — ambiguous posterior widens the hypothesis set.
                ent = -(posterior * posterior.clamp_min(1e-12).log()).sum(-1, keepdim=True)
                cap = float(torch.log(torch.tensor(float(posterior.shape[-1]))))
                fraction = (fraction * (1.0 + 0.75 * ent / max(cap, 1e-6))).clamp(max=1.0)
            if self.config.per_episode_gate:
                # v2: the gate depends on the CURRENT query too (KDA data-
                # dependent gating); spans the FULL hierarchy (hot + archive).
                q = query[:, None].expand(-1, all_values.shape[1], -1)
                slot_gate = self.episode_gate(
                    torch.cat((all_values, all_srcs, q, all_values*q), -1)).sigmoid().transpose(1, 2)
                scores = scores + torch.log(slot_gate.clamp_min(1e-6))
            temperature=F.softplus(self.retrieval_temperature)+0.5
            if learning_trace is not None:
                learning_trace['attention']=masked_distribution(scores*temperature,all_occ[:,None,:])
            attention = sparse_attention(scores, all_occ[:, None, :],temperature, fraction)
        else:
            attention = masked_distribution(scores, all_occ[:, None, :])
            if learning_trace is not None:learning_trace['attention']=attention
        if self.config.goal_gated_retrieval and goal is not None:
            # L4 (r8.5): retrieval-enable volume gate, goal-conditioned —
            # centered at 1 (sigmoid(0)=0.5 with zero-init weights).
            volume = (2 * self.goal_retrieval_gate(torch.cat((query, goal), -1)).sigmoid())
            attention = attention * volume[:, None, :]
        evidence = torch.einsum('bke,bed->bkd', attention, all_values)
        if self.config.echo_slots > 0 and state.echo_keys is not None \
                and echo_query_hash is not None:
            hit = (state.echo_keys == echo_query_hash[:, None]).to(evidence.dtype)
            echoed = (state.echo_values * hit[:, :, None]).sum(1)
            alpha = self.echo_gain.sigmoid() * hit.sum(-1, keepdim=True).clamp(max=1)
            evidence = evidence + alpha * echoed[:, None, :]
        combined = torch.einsum('bk,bkd->bd', mixing, evidence) * gain
        return evidence, combined, attention

    def step(self, x: Tensor, working: Tensor, state: ExperienceState, candidates: Tensor | None,
             goal: Tensor, *, read_mode='candidate', feedback_mode='normal',
             resource: Tensor | None = None, regional_state=None,circuit_mode='normal',hypothesis_state=None,
             hippocampal_state=None,hippocampal_mode='normal',cerebellar_prediction_prev=None,
             thalamic_mode='normal',cerebellar_mode='normal',amygdala_mode='off',da_prev=None,
             drift=None,echo_query_hash=None,body_bridge_context=None,body_choice_prior=None) -> CoreStep:
        b, d = x.shape
        if self.config.cognitive_workspace:
            x=x+self.workspace.read_gain.tanh()*self.workspace.read(x,state)
            goal=goal+self.workspace.goal_gain.tanh()*self.workspace.goal_context(state)
        hippocampal_population=hippocampal_state if hippocampal_state is not None else state.hippocampal_population
        hippocampal_mismatch=None
        coupled_input=x
        hippocampal_context=None
        if self.config.hippocampal_circuit:
            initial=self.hippocampus(x,torch.zeros_like(x),hippocampal_population,
                state.hippocampal_fast,state.hippocampal_eligibility,mode=hippocampal_mode)
            hippocampal_population=initial.population
            hippocampal_context=initial.context
            coupled_input=x+self.hippocampal_gain.tanh()*initial.context
        elif hippocampal_mode!='normal':
            raise ValueError('Hippocampal intervention requires the subregional architecture')
        thalamic=None
        if self.config.thalamic_gate:
            stats=torch.stack((coupled_input.abs().mean(-1),coupled_input.square().mean(-1),
                working[:,-1].abs().mean(-1),resource[:,0] if resource is not None else coupled_input[:,0]*0),-1)
            bg_out=arousal=surprise=None
            if self.config.cerebellar_thalamic_gain and state.cerebellar_prediction is not None:
                surprise=(state.cerebellar_prediction-coupled_input).abs().mean(-1)
            if self.config.bg_thalamic_loop:
                old_selection_bg=(state.selection if regional_state is None else regional_state[2])
                bg_out=old_selection_bg[:,4].mean(-1)
            if self.config.arousal_gating:
                arousal=state.drive_state[:,0]
            thalamic_result=self.thalamus(coupled_input,goal,stats,mode=thalamic_mode,
                selection_output=bg_out,arousal=arousal,surprise=surprise)
            thalamic=thalamic_result.populations
            coupled_input=coupled_input+self.thalamic_gain.tanh()*thalamic_result.routed
        cerebellar_prediction=cerebellar_error=None
        if self.config.cerebellum:
            cerebellar_prediction,cerebellar_error,cerebellar_context=self.cerebellum(
                coupled_input,cerebellar_prediction_prev,mode=cerebellar_mode)
            if cerebellar_context is not None:
                coupled_input=coupled_input+cerebellar_context
        valence=None
        if self.config.amygdala:
            valence=self.amygdala(coupled_input,mode=amygdala_mode)
            goal=goal+self.valence_projection(valence)
        drive_vector=initiate_logit=None
        if self.config.drives:
            drive_vector,initiate_logit=self.drives(state.drive_state)
            goal=goal+self.drive_gain.tanh()*drive_vector
        cognitive_control=None
        if self.config.cognitive_workspace:
            cognitive_control=self.workspace.control_logits(state,current=working[:,-1]+coupled_input)
            initiate_logit=torch.logsumexp(cognitive_control[:,2:],-1)-torch.logsumexp(cognitive_control[:,:2],-1)
        population=None
        if candidates is None:
            if self.config.hypothesis_mode!='adaptive':
                raise ValueError('Candidate-free transition requires an adaptive population')
            population=self.hypotheses(coupled_input,goal,working,state,hypothesis_state)
            candidates=population.vectors
        candidate_gate=None if population is None else population.gates
        if candidates.ndim != 3 or candidates.shape[0] != b or candidates.shape[2] != d or candidates.shape[1] < 1:
            raise ValueError('A nonempty variable candidate matrix B,K,D is required')
        if goal.shape != x.shape or working.shape != state.working.shape:
            raise ValueError('Working state or goal shape mismatch')
        if feedback_mode not in ('normal', 'detach', 'uniform', 'wrong_binding'):
            raise ValueError('Unknown candidate feedback intervention')
        resource = x.new_ones(b, 1) if resource is None else resource
        if resource.shape != (b, 1) or not torch.isfinite(resource).all():
            raise ValueError('One finite resource signal per stream required')
        top = self.context(torch.cat((coupled_input, working[:, -1]), -1))
        logits = torch.einsum('bd,bkd->bk', top, candidates) / d**0.5
        separated = self.config.judgment_mode == 'separated'
        field = self.native_field(top, candidates, torch.zeros_like(candidates), goal,
            feedback_mode=feedback_mode,candidate_gate=candidate_gate,
            language_prior=body_choice_prior,language_prior_gain=self.config.language_prior_gain,
            body_context=body_bridge_context) if separated else None
        used_allocation = None
        cortex,relay,selection = (regional_state if regional_state is not None else
                                 (state.cortex,state.relay,state.selection))
        edges,language_context = None,None
        stream_it=stream_pt=None
        if self.config.cortical_streams:
            old_selection=(state.selection if regional_state is None else regional_state[2])
            stream_it,stream_pt=self.streams(top,old_selection[:,4],goal)
        dopamine=None
        if self.config.dopamine_plasticity:
            old_selection=(state.selection if regional_state is None else regional_state[2])
            value_now=self.dopamine_system.value(top,old_selection[:,4])
            if cerebellar_error is not None:
                realized=-cerebellar_error.norm(dim=-1)
            elif hippocampal_mismatch is not None:
                realized=-hippocampal_mismatch.norm(dim=-1)
            else:
                realized=x.new_zeros(b)
            da_prev=x.new_zeros(b) if da_prev is None else da_prev
            mb=None
            if self.config.dopamine_mb_target:
                inferred=(hippocampal_context if hippocampal_context is not None
                          else (cerebellar_prediction if cerebellar_prediction is not None else top))
                mb=self.dopamine_system.mb_value(inferred)
            if self.config.valence_dopamine:
                bias=self.amygdala.reward_bias(valence)
                if bias is not None:
                    realized=realized+bias
            dopamine,rpe=self.dopamine_system(value_now,realized,da_prev,mb_value=mb)
        neuromod=None
        if self.config.neuromodulation:
            if cerebellar_error is not None:
                mismatch=cerebellar_error.abs().mean(-1)
            elif hippocampal_mismatch is not None:
                mismatch=hippocampal_mismatch.abs().mean(-1)
            else:
                mismatch=x.new_zeros(b)
            uncertainty=-(field.feedback[:,:,0]*field.feedback[:,:,0].clamp_min(1e-12).log()).sum(-1)
            neuromod=self.neuromod(top,mismatch,uncertainty,
                dopamine if dopamine is not None else x.new_zeros(b),drift=drift)
        for _ in range(self.config.inner_rounds):
            p = field.read_allocation if separated else logits.softmax(-1)
            if not separated:
                if feedback_mode == 'detach':
                    p = p.detach()
                elif feedback_mode == 'uniform':
                    p = torch.full_like(p, 1/p.shape[-1])
                elif feedback_mode == 'wrong_binding':
                    p = p.roll(1, -1)  # candidate vectors deliberately NOT permuted
            used_allocation = p
            abstain_mass = None
            if self.config.abstain_read and p.shape[-1] == candidates.shape[1] + 1:
                abstain_mass = p[:, -1:]
                p = p[:, :-1]
            learning_trace={}if torch.is_grad_enabled()else None
            per_candidate, episodic, attention = self.read_evidence(top, candidates, state, p,
                mode=read_mode, goal=goal if self.config.goal_gated_retrieval else None,
                echo_query_hash=echo_query_hash,learning_trace=learning_trace)
            # Candidate weights already sum to 1-abstain_mass. Multiplying
            # again would unintentionally square the no-read attenuation.
            if neuromod is not None:
                episodic = episodic * (1 + 0.5 * neuromod['ne'][:, None])
            if self.config.hippocampal_circuit:
                hypothesis_context=torch.einsum('bk,bkd->bd',p,candidates)
                retrieved=self.hippocampus(top+hypothesis_context,torch.zeros_like(x),hippocampal_population,
                    state.hippocampal_fast,state.hippocampal_eligibility,mode=hippocampal_mode)
                hippocampal_population,hippocampal_mismatch=retrieved.population,retrieved.mismatch
                episodic=episodic+self.hippocampal_gain.tanh()*retrieved.context
            semantic_scores = torch.einsum('bd,bsd->bs', top, state.semantic) / d**0.5
            semantic = torch.einsum('bs,bsd->bd', semantic_scores.softmax(-1), state.semantic)
            association = torch.einsum('bd,bde->be', F.normalize(top, dim=-1), state.association)
            predicted_context = self.context(torch.cat((top, goal), -1))
            entropy = -(p*p.clamp_min(1e-12).log()).sum(-1, keepdim=True)
            routes = self.route(torch.cat((top, goal, entropy, resource), -1)).softmax(-1)
            messages = torch.stack((x, episodic, association, semantic, predicted_context, goal), 1)
            message = torch.einsum('br,brd->bd', routes, messages)
            if self.config.regional_circuit:
                result=self.regional(coupled_input if self.config.thalamic_gate else x,goal,episodic,
                    semantic,association,candidates,field,
                    cortex,relay,selection,self.proposals,self.update_gates,self.norms,
                    self.level_rates,resource,mode=circuit_mode,
                    selection_context=(episodic if self.config.hippocampal_bg_loop else None),
                    context_message=message+(self.stream_gain.tanh()*stream_pt
                                             if stream_pt is not None else 0))
                cortex,relay,selection=result.cortex,result.relay,result.selection
                top,language_context,edges=result.native_context,result.language_context,result.edges
                if self.config.cortical_streams:
                    stream_it,stream_pt=self.streams(top,selection[:,4],goal)
                    top=top+self.stream_gain.tanh()*stream_it
                    language_context=language_context+self.stream_gain.tanh()*stream_it
                # Backward-compatible summary; NOT the only maintained state.
                working=cortex.mean(1)
            else:
                if circuit_mode!='normal':
                    raise ValueError('Circuit intervention requires regional architecture')
                old, levels = working, []
                for level in range(self.config.levels):
                    bottom = x if level == 0 else old[:, level-1]
                    higher = goal if level == self.config.levels-1 else old[:, level+1]
                    joined = torch.cat((old[:,level], bottom, higher, message), -1)
                    proposal = torch.tanh(self.proposals[level](joined))
                    rate = self.update_gates[level](joined).sigmoid() * self.level_rates[level]
                    levels.append(self.norms[level](old[:,level] + rate*(proposal-old[:,level])))
                working = torch.stack(levels, 1)
                top = working[:, -1]
            q = top[:,None].expand_as(candidates)
            g = goal[:,None].expand_as(candidates)
            outcomes = self.outcome(torch.cat((q, candidates, per_candidate), -1))
            value = torch.einsum('bkd,bd->bk', outcomes, goal) / d**0.5
            if separated:
                field = self.native_field(top, candidates, per_candidate, goal,
                    feedback_mode=feedback_mode,candidate_gate=candidate_gate,body_context=body_bridge_context,
                    language_prior=body_choice_prior,language_prior_gain=self.config.language_prior_gain)
                logits = field.choice_logits
            else:
                logits = self.judge(torch.cat((q, candidates, per_candidate, g), -1)).squeeze(-1) + value
        control = self.control(top)
        if self.config.drives:
            control = torch.cat((control, initiate_logit[:, None]), -1)
        return CoreStep(working, language_context if self.config.regional_circuit else top,
            logits, control, outcomes, attention, routes,
            field.predicate_logits if separated else None,
            field.action_values if separated else value, used_allocation,cortex,relay,selection,edges,population,
            hippocampal_population,hippocampal_mismatch,
            thalamic,cerebellar_prediction,cerebellar_error,dopamine,
            None if neuromod is None else torch.stack((neuromod['da'],neuromod['ach'],neuromod['ne']),-1),
            drive_vector,initiate_logit if self.config.drives else None,cognitive_control,
            None if learning_trace is None else learning_trace['attention'])

    def commit_observation(self, encoded: Tensor, source: Tensor, working: Tensor,
                           state: ExperienceState,*,regions=None,hippocampal_population=None,
                           hippocampal_mode='normal',thalamic=None,cerebellar_prediction=None,
                           dopamine=None,neuromod=None,mismatch=None,uncertainty=None,
                           echo_hash_in=None) -> ExperienceState:
        """Append an observed event; no last-write-wins semantic oracle is hidden here.

        r8: neuromodulated plasticity — ACh scales the write gate (learning
        mode), dopamine scales the association delta (dMSN-persistent
        analogue); both are multiplicative factors centered on 1 and ABSENT in
        the ancestor config, so r7 behavior is unchanged when r8 is off.
        """
        b, d = encoded.shape
        self.validate_state(state, b)
        if self.config.regional_circuit and regions is None:
            raise ValueError('Regional observation commit requires all updated circuit populations')
        if source.shape != encoded.shape or working.shape != state.working.shape:
            raise ValueError('Observation provenance/working shape mismatch')
        if (state.writes == torch.iinfo(torch.long).max).any():
            raise OverflowError('Observation counter exhausted')
        key, value, provenance = self._sparse_key(encoded), self.value(encoded), self.source(source)
        if self.config.perception_memory:value=encoded+.1*value
        slot = state.writes.remainder(self.config.episode_slots)
        decision=None
        if self.config.episodic_policy:
            limits={}
            if self.config.preserve_free_capacity:
                empty=~state.occupied
                any_hot=empty.any(-1)
                any_cold=(~state.archive_occupied).any(-1)if self.config.archive_slots else torch.zeros_like(any_hot)
                limits=dict(write_mask=torch.where(any_hot[:,None],empty,torch.ones_like(empty)),
                            allow_skip=~(any_hot|any_cold))
            decision=self.episodic_policy(key,value,provenance,working[:,-1],
                state.keys,state.values,state.sources,state.occupied,state.episode_event_ids,state.writes,**limits)
            slot=(decision.action-1).clamp_min(0)
            select=decision.write_weights[:,:,None]
        else:
            select=F.one_hot(slot,self.config.episode_slots).to(encoded.dtype)[:,:,None]
        semantic = state.semantic
        if self.config.evict_consolidate:
            # Evict-means-written: when the ring buffer is about to overwrite an
            # occupied slot, that slot's content is pulled into the semantic
            # slow state first (LiveMem eviction contract / context-editing
            # semantics: cleared from the viewport, kept retrievable).
            if self.config.episodic_policy:
                evicted=(state.occupied[:,:,None].to(encoded.dtype)*select).sum(1,keepdim=True)
                evicted_content=(state.values*select).sum(1,keepdim=True)
            else:
                evicted = state.occupied.gather(1, slot[:, None]).to(state.semantic.dtype)[:, :, None]
                evicted_content = state.values.gather(1, slot[:, None, None].expand(-1, 1, state.semantic.shape[-1]))
            semantic = semantic + 0.25 * evicted * (
                evicted_content - semantic.mean(1, keepdim=True))
        replace = lambda old, new: old*(1-select)+new[:,None]*select
        predicted = torch.einsum('bd,bde->be', key, state.association)
        raw_strength = self.write_gate(torch.cat((encoded, value-predicted), -1)).sigmoid()
        write_thr=None
        if self.config.dual_condition_write:
            # SR-TTT style (arXiv:2603.06642): attenuate writes below 0.8x the
            # recent surprise level; smooth clamp-style gate, EMA threshold.
            thr = (state.write_threshold if state.write_threshold is not None
                   else raw_strength.new_full((b, 1), 0.5))
            gate = torch.sigmoid((raw_strength - 0.8*thr) / (0.1*thr.abs() + 1e-3))
            strength = raw_strength * (0.2 + 0.8*gate)
            write_thr = 0.9*thr + 0.1*raw_strength.mean(-1, keepdim=True)
        else:
            strength = raw_strength
        if neuromod is not None:
            strength = strength*(0.5+neuromod[:,1][:,None]).clamp(0.,2.)
        da_scale = None if dopamine is None else (1+dopamine).clamp(0.,2.)[:,None,None]
        delta = torch.einsum('bd,be->bde', key, value-predicted)*strength[:,:,None]
        if da_scale is not None:
            delta = delta*da_scale
        association = self.config.association_decay*state.association + delta
        occupied = state.occupied | select.detach().squeeze(-1).bool()
        regional = (dict(zip(('cortex','relay','selection'),regions)) if regions is not None else
                    {name:getattr(state,name) for name in ('cortex','relay','selection')})
        hippocampal={}
        if self.config.hippocampal_circuit:
            if hippocampal_population is None:
                raise ValueError('Commit requires the actual updated hippocampal population')
            if (state.hippocampal_updates==torch.iinfo(torch.long).max).any():
                raise OverflowError('Local plasticity counter exhausted')
            encoded_result=self.hippocampus(encoded,source,hippocampal_population,
                state.hippocampal_fast,state.hippocampal_eligibility,phase='encode',mode=hippocampal_mode)
            encoded_fast=encoded_result.fast
            if self.config.dopamine_write_gate and dopamine is not None:
                # L2 (r8.4): novelty/reward-gated LTP analogue — the dopamine
                # level scales how much of the fast-weight delta settles.
                fast_delta=encoded_result.fast-state.hippocampal_fast
                encoded_fast=state.hippocampal_fast+fast_delta*(
                    1+self.da_write_gain.tanh()*dopamine[:,None,None])
            hippocampal=dict(hippocampal_population=encoded_result.population,
                hippocampal_fast=encoded_fast,hippocampal_eligibility=encoded_result.eligibility,
                hippocampal_updates=state.hippocampal_updates+1,
                hippocampal_events=replace(state.hippocampal_events,encoded),
                hippocampal_provenance=replace(state.hippocampal_provenance,source))
        r8_extra={}
        if self.config.cognitive_workspace:
            r8_extra.update({name:getattr(state,name) for name in WORKSPACE_FIELDS})
            if self.config.goal_joint_control:r8_extra.update({name:getattr(state,name)for name in GOAL_FIELDS})
        if self.config.episodic_policy:
            r8_extra.update(episode_admissions=state.episode_admissions+(decision.action>0).long(),
                episode_event_ids=torch.where(select.detach().squeeze(-1).bool(),
                    (state.writes+1)[:,None].expand_as(state.episode_event_ids),state.episode_event_ids),
                admission_distribution=decision.probabilities)
        if self.config.thalamic_gate:
            r8_extra['thalamic']=thalamic if thalamic is not None else state.thalamic
        if self.config.cerebellum:
            r8_extra['cerebellar_prediction']=(cerebellar_prediction if cerebellar_prediction is not None
                                               else state.cerebellar_prediction)
        if self.config.dopamine_plasticity:
            r8_extra['dopamine']=dopamine if dopamine is not None else state.dopamine
        if self.config.neuromodulation:
            r8_extra['neuromod_last']=neuromod if neuromod is not None else state.neuromod_last
        if self.config.archive_slots > 0:
            # Promotion: ONLY when the overwritten hot slot was occupied does
            # the evicted episode append to the cold archive ring.
            if self.config.episodic_policy:
                hot_occ=(state.occupied[:,:,None].to(encoded.dtype)*select).sum(1,keepdim=True)
                old_keys=(state.keys*select).sum(1);old_vals=(state.values*select).sum(1)
                old_srcs=(state.sources*select).sum(1)
                old_ids=state.episode_event_ids.gather(1,slot[:,None]).squeeze(1)
                cold_limits={}
                if self.config.preserve_free_capacity:
                    empty=~state.archive_occupied;any_empty=empty.any(-1)
                    cold_limits=dict(write_mask=torch.where(any_empty[:,None],empty,torch.ones_like(empty)),allow_skip=~any_empty)
                cold=self.episodic_policy(old_keys,old_vals,old_srcs,working[:,-1],
                    state.archive_keys,state.archive_values,state.archive_sources,
                    state.archive_occupied,state.archive_event_ids,state.writes,item_id=old_ids,stage=1,**cold_limits)
                a_sel=cold.write_weights[:,:,None]*hot_occ
                r8_extra['archive_event_ids']=torch.where(a_sel.detach().squeeze(-1).bool(),
                    old_ids[:,None].expand_as(state.archive_event_ids),state.archive_event_ids)
            else:
                hot_occ = state.occupied.gather(1, slot[:, None]).to(encoded.dtype)[:, :, None]
                a_slot = (state.archive_writes if state.archive_writes is not None
                          else torch.zeros_like(state.writes)).remainder(self.config.archive_slots)
                a_sel = F.one_hot(a_slot, self.config.archive_slots).to(encoded.dtype)[:, :, None]*hot_occ
                old_keys = state.keys.gather(1, slot[:, None, None].expand(-1, 1, d)).squeeze(1)
                old_vals = state.values.gather(1, slot[:, None, None].expand(-1, 1, d)).squeeze(1)
                old_srcs = state.sources.gather(1, slot[:, None, None].expand(-1, 1, d)).squeeze(1)
            a_rep = lambda t, v: t * (1 - a_sel) + v[:, None] * a_sel
            r8_extra.update(
                archive_keys=a_rep(state.archive_keys, old_keys),
                archive_values=a_rep(state.archive_values, old_vals),
                archive_sources=a_rep(state.archive_sources, old_srcs),
                archive_occupied=state.archive_occupied | a_sel.detach().squeeze(-1).bool(),
                archive_writes=state.archive_writes + a_sel.detach().squeeze(-1).long().sum(-1))
        if self.config.echo_slots > 0 and echo_hash_in is not None:
            fingerprints = echo_hash_in.reshape(-1)
            if fingerprints.shape != (b,):
                raise ValueError('One observation fingerprint per stream required')
            slot_h = fingerprints.remainder(self.config.echo_slots)
            rows_h = torch.arange(b, device=encoded.device)
            r8_extra['echo_keys']=state.echo_keys.clone()
            r8_extra['echo_keys'][rows_h, slot_h]=fingerprints
            r8_extra['echo_values']=state.echo_values.clone()
            r8_extra['echo_values'][rows_h, slot_h]=value.detach()
            r8_extra['echo_count']=state.echo_count+1
        if self.config.dual_condition_write:
            r8_extra['write_threshold']=(write_thr if write_thr is not None
                                         else state.write_threshold)
        if self.config.ema_anchor_consolidation:
            r8_extra['semantic_anchor']=state.semantic_anchor
        if self.config.drives:
            curiosity,mismatch_ema,idle=(state.drive_state[:,i] for i in range(3))
            novelty=encoded.new_zeros(b) if mismatch is None else mismatch.abs().mean(-1)
            entropy=encoded.new_zeros(b) if uncertainty is None else uncertainty
            r8_extra['drive_state']=torch.stack((
                0.9*curiosity+0.1*entropy,0.9*mismatch_ema+0.1*novelty,
                idle if self.config.cognitive_workspace else (idle+1.0/self.config.episode_slots).clamp(max=1.0)),-1)
        return ExperienceState(working, replace(state.keys,key), replace(state.values,value),
            replace(state.sources,provenance), occupied, association, semantic,
            state.writes+1, state.consolidations, state.binding,**regional,**hippocampal,**r8_extra)

    def consolidate(self, state: ExperienceState,*,hippocampal_mode='normal') -> ExperienceState:
        """Explicit slower update. Not an implemented biological replay/skill system."""
        self.validate_state(state, state.working.shape[0])
        if (state.consolidations == torch.iinfo(torch.long).max).any():
            raise OverflowError('Consolidation counter exhausted')
        if self.config.archive_slots > 0 and state.archive_keys is not None:
            all_keys = torch.cat((state.keys, state.archive_keys), 1)
            all_vals = torch.cat((state.values, state.archive_values), 1)
            all_occ = torch.cat((state.occupied, state.archive_occupied), 1)
        else:
            all_keys, all_vals, all_occ = state.keys, state.values, state.occupied
        scores = torch.einsum('sd,bed->bse', self.semantic_queries, all_keys) / self.config.width**0.5
        weights = masked_distribution(scores, all_occ[:,None])
        proposal = torch.einsum('bse,bed->bsd', weights, all_vals)
        rate = self.config.consolidation_rate * state.occupied.any(-1)[:,None,None]
        if self.config.neuromodulation and state.neuromod_last is not None:
            rate = rate*(0.5+state.neuromod_last[:,1])[:,None,None].clamp(0.,2.)
        semantic = state.semantic + rate*(proposal-state.semantic)
        if self.config.ema_anchor_consolidation:
            # LaCET-style (arXiv:2604.07350) streaming-EMA anchor: slow slots
            # are pulled back toward their long-run mean so replay cannot
            # drift them away — the anti-wander spring for permanent online.
            anchor = (state.semantic_anchor if state.semantic_anchor is not None
                      else semantic)
            semantic = semantic - self.anchor_pull.sigmoid()*(semantic - anchor)
            extra_anchor = self.anchor_rate.sigmoid()*anchor + (1-self.anchor_rate.sigmoid())*semantic
        else:
            extra_anchor = None
        extra={}
        if self.config.ema_anchor_consolidation:
            extra['semantic_anchor']=extra_anchor
        if self.config.hippocampal_circuit:
            population,fast,eligibility=state.hippocampal_population,state.hippocampal_fast,state.hippocampal_eligibility
            count=state.occupied.sum(-1)
            if (state.hippocampal_updates>torch.iinfo(torch.long).max-count).any():
                raise OverflowError('Local replay/plasticity counter exhausted')
            rows=torch.arange(state.working.shape[0],device=state.working.device)
            start=torch.where(state.writes>=self.config.episode_slots,
                              state.writes.remainder(self.config.episode_slots),torch.zeros_like(state.writes))
            if self.config.episodic_policy:
                order=state.episode_event_ids.masked_fill(~state.occupied,torch.iinfo(torch.long).max).argsort(dim=-1,stable=True)
            for offset in range(self.config.episode_slots):
                slot=(order[:,offset] if self.config.episodic_policy else
                      (start+offset).remainder(self.config.episode_slots))
                active=state.occupied[rows,slot,None]
                replay=self.hippocampus(state.hippocampal_events[rows,slot],
                    state.hippocampal_provenance[rows,slot],population,fast,eligibility,
                    phase='encode',mode=hippocampal_mode)
                population=torch.where(active[:,:,None],replay.population,population)
                fast=torch.where(active[:,:,None],replay.fast,fast)
                eligibility=torch.where(active[:,:,None],replay.eligibility,eligibility)
            extra.update(hippocampal_population=population,hippocampal_fast=fast,
                hippocampal_eligibility=eligibility,hippocampal_updates=state.hippocampal_updates+count)
        if self.config.drives and state.drive_state is not None:
            extra['drive_state']=torch.stack((state.drive_state[:,0],state.drive_state[:,1],
                torch.zeros_like(state.drive_state[:,2])),-1)
        return ExperienceState(**{**state.tensors(),**extra,'semantic': semantic,
            'consolidations': state.consolidations+1}, binding=state.binding)

    def specification(self):
        revision = ('0.23' if self.config.r8_active else
                    '0.21' if self.config.hippocampal_circuit else '0.20' if self.config.hypothesis_mode=='adaptive' else '0.19' if self.config.regional_circuit else
                    '0.17' if self.config.judgment_mode == 'separated' or self.config.coupling_mode == 'layer_interleaved' else '0.16')
        return {'schema': 'treyrin-structural-core-'+revision, 'config': asdict(self.config),
                'capability_accepted': False, 'brain_structure_requirement_satisfied': False}
