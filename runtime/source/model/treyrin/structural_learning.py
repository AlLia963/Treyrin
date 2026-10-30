"""Joint learning lifecycle for the structural candidate, including frozen r2.

One session is one chronological experience stream (B=1), not a batch of times.
Only public observations are committed. Targets stay on the loss side. Every
optimizer update invalidates the old neural state and rebuilds it from the same
public journal. This is an executable learning contract, not a capability claim.
"""
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch.nn import functional as F

from .learning_checkpoint import (
    _optimizer_groups, load_learning_checkpoint, save_learning_checkpoint,
)
from .retention_reference import RetentionReference
from .cognitive_workspace import MentalRecord,CognitiveTick,KINDS,STATUSES,ACTIONS


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def _hash(value):
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class PublicObservation:
    event_id: str
    text: str
    source: str
    known_at: str
    occurred_at: str = ''
    scope: str = ''

    def __post_init__(self):
        if any(not isinstance(v, str) for v in asdict(self).values()):
            raise ValueError('Observation fields must be public strings')
        if any(not getattr(self, key).strip() for key in ('event_id', 'text', 'source', 'known_at')):
            raise ValueError('Public event ID, text, provenance and knowledge time required')

    def provenance_text(self):
        return _json({key: value for key, value in asdict(self).items() if key != 'text'})


@dataclass(frozen=True)
class JointExample:
    prompt: str
    answer: str
    candidates: tuple[str, ...]
    native_target: tuple[float, ...]
    goal: str = ''
    native_kind: str = 'choice'
    hypothesis_targets: tuple[str,...] = ()
    hypothesis_rejections: tuple[str,...] = ()
    predicate_targets: tuple[float,...] = ()
    utility_targets: tuple[float,...] = ()
    control_target: tuple[float,...] = ()  # continue / wait / request-evidence
    initiate_target: float | None = None
    cognitive_control_target: tuple[float,...] = ()
    supporting_events: tuple[str,...] = ()  # loss-only public source identities
    disposable_events: tuple[str,...] = ()  # future-use training labels, never routing inputs
    choice_labels: bool = False  # PUBLIC prompt binds A/B/... to descriptions

    def __post_init__(self):
        if type(self.choice_labels)is not bool:raise ValueError('Explicit public label-format flag required')
        if not isinstance(self.prompt, str) or not self.prompt or not isinstance(self.answer, str) or not self.answer:
            raise ValueError('Nonempty prompt and teacher answer required')
        if not isinstance(self.goal, str) or any(not isinstance(c, str) or not c for c in self.candidates):
            raise ValueError('Public candidate descriptions and goal must be strings')
        if len(self.candidates) != len(self.native_target):
            raise ValueError('Target must index exactly these public candidates')
        if self.native_kind not in ('choice','predicate','language'):
            raise ValueError('Unsupported native target type')
        if self.native_kind=='language':
            if self.candidates or self.native_target:
                raise ValueError('Candidate-free language supervision has no supplied native labels')
        elif not self.candidates:
            raise ValueError('Explicit native targets require aligned public candidates')
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in self.native_target):
            raise ValueError('Native probability targets must be finite and in [0,1]')
        if self.native_kind == 'choice' and not math.isclose(sum(self.native_target), 1., abs_tol=1e-6):
            raise ValueError('A normalized categorical target is required for choice')
        if any(not isinstance(s,str) or not s.strip() for s in (*self.hypothesis_targets,*self.hypothesis_rejections)):
            raise ValueError('Hypothesis supervision must contain explicit nonempty descriptions')
        if self.hypothesis_rejections and not self.hypothesis_targets:
            raise ValueError('Rejections require an explicitly supplied admissible reference set')
        if set(self.hypothesis_targets)&set(self.hypothesis_rejections):
            raise ValueError('A description cannot be both admissible and rejected')
        if self.hypothesis_targets and self.native_kind!='language':
            raise ValueError('Unordered hypothesis supervision uses the candidate-free path')
        for name in ('predicate_targets','utility_targets'):
            target=getattr(self,name)
            if target and (len(target)!=len(self.candidates) or not self.candidates or
                           any(not math.isfinite(x) for x in target)):
                raise ValueError('Aligned finite '+name+' required')
        if self.predicate_targets and (self.native_kind=='predicate' or
                any(not 0<=x<=1 for x in self.predicate_targets)):
            raise ValueError('Independent [0,1] predicate targets cannot duplicate native predicate supervision')
        if self.control_target and (len(self.control_target)!=3 or
                any(not math.isfinite(x) or not 0<=x<=1 for x in self.control_target) or
                not math.isclose(sum(self.control_target),1.,abs_tol=1e-6)):
            raise ValueError('Normalized continue/wait/request-evidence control target required')
        if self.initiate_target is not None and (not math.isfinite(self.initiate_target) or
                                                not 0<=self.initiate_target<=1):
            raise ValueError('Initiation target must be a probability in [0,1]')
        if self.cognitive_control_target and (len(self.cognitive_control_target)!=len(ACTIONS) or
                any(not math.isfinite(x) or not 0<=x<=1 for x in self.cognitive_control_target) or
                not math.isclose(sum(self.cognitive_control_target),1.,abs_tol=1e-6)):
            raise ValueError('Normalized typed cognitive action target required')
        if any(not isinstance(x,str) or not x.strip() for x in self.supporting_events) or len(set(self.supporting_events))!=len(self.supporting_events):
            raise ValueError('Unique public evidence identities required for loss-only read grounding')
        if any(not isinstance(x,str) or not x.strip() for x in self.disposable_events) or len(set(self.disposable_events))!=len(self.disposable_events) or set(self.supporting_events)&set(self.disposable_events):
            raise ValueError('Disjoint unique loss-only disposal identities required')


@dataclass(frozen=True)
class InternalConsolidation:
    operation_id: str
    known_at: str

    def __post_init__(self):
        if any(not isinstance(v,str) or not v.strip() for v in asdict(self).values()):
            raise ValueError('Explicit internal operation identity and logical time required')


@dataclass(frozen=True)
class ComputationMode:
    read_mode: str = 'candidate'
    feedback_mode: str = 'normal'
    body_feedback: bool = True
    scope: str = 'all'
    circuit_mode: str = 'normal'
    hippocampal_mode: str = 'normal'
    thalamic_mode: str = 'normal'
    cerebellar_mode: str = 'normal'
    amygdala_mode: str = 'off'

    def __post_init__(self):
        if self.read_mode not in ('candidate','entropy','uniform') or self.feedback_mode not in ('normal','detach','uniform','wrong_binding'):
            raise ValueError('Unknown internal computation intervention')
        if type(self.body_feedback) is not bool or self.scope not in ('all','query'):
            raise ValueError('Explicit body-feedback flag and intervention scope required')
        if self.circuit_mode not in ('normal','no_cross','gain_only','no_selection'):
            raise ValueError('Unknown circuit intervention')
        if self.hippocampal_mode not in ('normal','no_recurrence','no_direct','no_feedback'):
            raise ValueError('Unknown hippocampal pathway intervention')
        if self.thalamic_mode not in ('normal','no_relay','no_gate','open_gate'):
            raise ValueError('Unknown thalamic intervention')
        if self.cerebellar_mode not in ('normal','lesion','no_context'):
            raise ValueError('Unknown cerebellar intervention')
        if self.amygdala_mode not in ('normal','off'):
            raise ValueError('Unknown amygdala intervention')

    def forward_options(self):
        return {key:value for key,value in asdict(self).items() if key != 'scope'
                and not (key in ('circuit_mode','hippocampal_mode','thalamic_mode','cerebellar_mode') and value=='normal')
                and not (key == 'amygdala_mode' and value=='off')}


class StructuralLearningSession:
    """Single CPU/CUDA stream; eval-mode computation WITH gradients.

    The encoder identity must be a verified SHA256 of tokenizer assets. It is
    separate from weight identity: matching weights do not prove token identity.
    Closed categorical choices and independent binary predicates have distinct
    proper losses. Neither implements a complete-world posterior or establishes
    calibration without task evaluation. Frozen r2 supports choice only.
    """
    def __init__(self, model, encode: Callable[[str], Sequence[int]], *, encoding_sha256: str | None = None,
                 mode: ComputationMode = ComputationMode(),tokenizer_directory=None,
                 retention_reference=None,retention_texts=(),auto_trim: dict | None = None):
        if tokenizer_directory is not None:
            root=Path(tokenizer_directory).resolve()
            files=sorted(p for p in root.rglob('*') if p.is_file())
            if not files:raise ValueError('Tokenizer assets are empty')
            h=hashlib.sha256()
            for path in files:
                h.update(path.relative_to(root).as_posix().encode());h.update(path.read_bytes())
            computed=h.hexdigest()
            if encoding_sha256 is not None and encoding_sha256!=computed:
                raise ValueError('Supplied tokenizer identity differs from actual assets')
            encoding_sha256=computed
        if not isinstance(encoding_sha256,str):raise ValueError('Tokenizer assets or explicit verified identity required')
        if len(encoding_sha256) != 64 or any(c not in '0123456789abcdef' for c in encoding_sha256):
            raise ValueError('Verified tokenizer SHA256 required')
        self.device = next(model.core.parameters()).device
        if model.training or self.device.type not in ('cpu', 'cuda') or any(
                p.device != self.device for p in model.parameters()):
            raise ValueError('Lifecycle requires eval mode and one CPU/CUDA device')
        self.model, self.encode = model, encode
        self.retention_reference=retention_reference
        self.retention_texts=tuple(retention_texts)
        if retention_reference is not None:
            if not isinstance(retention_reference,RetentionReference):
                raise ValueError('Pinned frozen reference required')
            retention_reference.check(model)
            if not self.retention_texts or any(not isinstance(t,str) or not t for t in self.retention_texts):
                raise ValueError('Explicit neutral retention probes required')
        elif self.retention_texts or any(p.requires_grad for p in model.body.parameters()):
            raise ValueError('Language-body learning requires a declared independent retention reference')
        # r8.9 auto-trim (context-editing aligned): when unconsolidated
        # observations exceed `trigger`, one consolidation runs automatically —
        # the viewport is cleared, the content stays retrievable in memory.
        if auto_trim is not None:
            if not isinstance(auto_trim, dict) or type(auto_trim.get('trigger')) is not int                     or auto_trim['trigger'] < 1:
                raise ValueError('auto_trim requires an integer trigger >= 1')
        self.auto_trim = None if auto_trim is None else dict(auto_trim)
        self._auto_trim_identity = _json(self.auto_trim)
        self._last_trim_obs = 0
        self.encoding_sha256 = encoding_sha256
        # r8.3 body-drift coupling: EMA of the retention KL measures how far
        # the trainable body has moved from its frozen reference; the session
        # feeds it to the core as a neuromodulator scalar each forward.
        self.drift_ema = None
        if model.config.body_drift_coupling and not any(
                p.requires_grad for p in model.body.parameters()):
            raise ValueError('Body-drift coupling requires a trainable body')
        if not isinstance(mode,ComputationMode):
            raise ValueError('A declared immutable computation mode required')
        self.mode = self._declared_mode = mode
        self.observations: tuple[PublicObservation, ...] = ()
        self.consolidations: tuple[InternalConsolidation, ...] = ()
        self.operations: tuple[tuple[str,int], ...] = ()
        self.mental_records:tuple[MentalRecord,...]=()
        self.cognitive_ticks:tuple[CognitiveTick,...]=()
        self.tick_actions:tuple[str,...]=()
        self.tick_goals:tuple[int|None,...]=()
        self.optimizer_steps = 0
        self._parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self._frozen_versions = tuple((id(p), p._version) for p in model.parameters() if not p.requires_grad)
        self.contract_identity = self._contract_identity()
        self._set_state(model.initial_state(1).detached())

    def _contract_identity(self):
        h = hashlib.sha256()
        h.update(self.encoding_sha256.encode())
        h.update(self._auto_trim_identity.encode())
        h.update(_json(dict(reference=self.retention_reference.identity if self.retention_reference else None,
            probes=self.retention_texts,scope='empty-experience last-token KL; cyclic probe per update')).encode())
        h.update(Path(inspect.getfile(RetentionReference)).read_bytes())
        h.update(_json(asdict(self.mode)).encode())
        h.update(self.model._execution_configuration().encode())
        for cls in (type(self), type(self.model), type(self.model.core)):
            h.update(Path(inspect.getfile(cls)).read_bytes())
        h.update(Path(inspect.getfile(save_learning_checkpoint)).read_bytes())
        for name, tensor in self.model.named_parameters():
            h.update(_json((name, list(tensor.shape), str(tensor.dtype), tensor.requires_grad)).encode())
            if not tensor.requires_grad:
                h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        for name,tensor in self.model.named_buffers():
            h.update(_json(('buffer',name,list(tensor.shape),str(tensor.dtype))).encode())
            h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    @property
    def journal_sha256(self):
        return _hash(_json(self._journal()).encode())

    def _journal(self):
        result={'observations':[asdict(o) for o in self.observations],
                'consolidations':[asdict(o) for o in self.consolidations],
                'operations':self.operations}
        if self.model.config.cognitive_workspace:
            result.update(mental_records=[asdict(o) for o in self.mental_records],
                cognitive_ticks=[asdict(o) for o in self.cognitive_ticks],tick_actions=self.tick_actions)
            if self.model.config.goal_joint_control:result['tick_goals']=list(self.tick_goals)
        return result

    def _parse_journal(self,journal):
        """One chronology/reference validator for both inference and training restore."""
        expected={'observations','consolidations','operations'}
        if self.model.config.cognitive_workspace:expected|={'mental_records','cognitive_ticks','tick_actions'}
        if self.model.config.goal_joint_control:expected|={'tick_goals'}
        if set(journal)!=expected:raise ValueError('Invalid operation journal fields')
        observations=tuple(PublicObservation(**o) for o in journal['observations'])
        consolidations=tuple(InternalConsolidation(**o) for o in journal['consolidations'])
        mental=tuple(MentalRecord(**o) for o in journal.get('mental_records',()))
        ticks=tuple(CognitiveTick(**o) for o in journal.get('cognitive_ticks',()))
        actions=tuple(journal.get('tick_actions',()))
        if len(actions)!=len(ticks) or any(a not in ACTIONS for a in actions):raise ValueError('Historical cognitive actions missing/invalid')
        goals=tuple(journal.get('tick_goals',()))
        if self.model.config.goal_joint_control and (len(goals)!=len(ticks) or any(
                g is not None and (type(g) is not int or not 0<=g<=self.model.config.workspace_slots) for g in goals)):
            raise ValueError('Historical cognitive goal rows missing/invalid')
        if len(set(observations))!=len(observations) or len({o.operation_id for o in consolidations})!=len(consolidations):raise ValueError('Repeated factual/consolidation operation')
        if len({r.record_id for r in mental})!=len(mental) or len({t.operation_id for t in ticks})!=len(ticks):raise ValueError('Repeated mental/cognitive identity')
        collections={'observation':observations,'consolidation':consolidations}
        if self.model.config.cognitive_workspace:collections.update(mental=mental,tick=ticks)
        counters={name:0 for name in collections};known_events=set();known_mental={};revised=set()
        operations=tuple(tuple(o) for o in journal['operations'])
        for kind,index in operations:
            if kind not in counters or type(index) is not int or index!=counters[kind] or index>=len(collections[kind]):raise ValueError('Invalid chronological operation index')
            record=collections[kind][index]
            if kind=='observation':known_events.add(record.event_id)
            if kind=='mental':
                if not set(record.evidence_ids)<=known_events:raise ValueError('Mental record cites unknown/future public evidence')
                if record.revises and record.revises not in known_mental:raise ValueError('Mental revision cites unknown/future identity')
                if record.revises:
                    if record.revises in revised:raise ValueError('Revise the latest record, not an already superseded branch')
                    prior=known_mental[record.revises]
                    if prior.kind!=record.kind and not {prior.kind,record.kind}<={'task','goal'}:raise ValueError('A revision cannot silently change the origin/function kind')
                    revised.add(record.revises)
                known_mental[record.record_id]=record
            counters[kind]+=1
        if any(counters[k]!=len(v) for k,v in collections.items()):raise ValueError('Journal operation coverage differs')
        return observations,consolidations,mental,ticks,actions,goals,operations

    def _set_state(self, state):
        self.state = state
        self._state_stamp = (None if state is None else
            (id(state),state.binding,tuple((name,id(t),t._version) for name,t in state.tensors().items())))

    def _check(self):
        if _json(self.auto_trim) != self._auto_trim_identity:
            raise ValueError('Auto-trim policy changed: construct a new learning contract')
        if self.retention_reference is not None:self.retention_reference.check(self.model)
        if self.state is None:
            raise ValueError('Incomplete update/restore: no valid committed state')
        stamp = (id(self.state),self.state.binding,tuple((name,id(t),t._version) for name,t in self.state.tensors().items()))
        if stamp != self._state_stamp:
            raise ValueError('Unjournaled state mutation: use observation/consolidation transitions')
        if self.mode != self._declared_mode:
            raise ValueError('Computation mode changed: construct a distinct learning contract')
        if self.model.training:
            raise ValueError('Deterministic replay requires eval mode; gradients remain enabled')
        now = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        if tuple((n, id(p)) for n, p in now.items()) != tuple((n, id(p)) for n, p in self._parameters.items()):
            raise ValueError('Trainable membership changed: construct a new learning contract')
        if self._frozen_versions != tuple((id(p), p._version) for p in self.model.parameters() if not p.requires_grad):
            raise ValueError('Frozen body changed: construct a new learning contract')
        self.model._check(self.state, 1)

    def _ids(self, text):
        ids = torch.tensor([list(self.encode(text))], dtype=torch.long, device=self.device)
        self.model._tokens(ids, torch.ones_like(ids))
        return ids

    def _ingest(self, event, state):
        content, source = self._ids(event.text), self._ids(event.provenance_text())
        options=({} if self.mode.scope=='query' or self.mode==ComputationMode()
                 else self.mode.forward_options())
        # One commit implementation for normal/intervened computation. The old
        # duplicate path dropped echo and r8 state whenever a mode was active.
        return self.model.ingest(content,torch.ones_like(content),source,torch.ones_like(source),
                                 state,origin='observation',**options)

    def _consolidate_state(self,state):
        options=({'hippocampal_mode':self.mode.hippocampal_mode}
                 if self.mode.scope=='all' and self.mode.hippocampal_mode!='normal' else {})
        return self.model.consolidate(state,**options)

    def _mental_state(self,record,state):
        ids,source=self._ids(record.text),self._ids(record.provenance_text())
        prior={r.record_id:i+1 for i,r in enumerate(self.mental_records)}
        revision=prior.get(record.revises) if record.revises else None
        action_goal=prior.get(record.goal_action_of)if record.goal_action_of else None
        if record.goal_action_of and action_goal is None:raise ValueError('Execution receipt requires a known goal identity')
        return self.model.remember_mental(ids,torch.ones_like(ids),source,torch.ones_like(source),state,
            kind=record.kind,status=record.status,revision_id=revision,goal_metadata=record.goal_metadata,
            evidence_count=len(record.evidence_ids),action_goal_id=action_goal,action_known_at=record.known_at)

    def remember(self,record:MentalRecord):
        self._check()
        if not self.model.config.cognitive_workspace or not isinstance(record,MentalRecord):raise ValueError('Typed non-fact record required')
        if record in self.mental_records:return {'status':'exact_mental_replay_ignored'}
        if any(r.record_id==record.record_id for r in self.mental_records):raise ValueError('A mental revision requires a new identity')
        if not set(record.evidence_ids)<={o.event_id for o in self.observations}:raise ValueError('Unknown/future public evidence identity')
        if record.revises and record.revises not in {r.record_id for r in self.mental_records}:raise ValueError('Unknown/future mental revision identity')
        if record.revises:
            if any(r.revises==record.revises for r in self.mental_records):raise ValueError('Revise the latest record, not an already superseded branch')
            prior=next(r for r in self.mental_records if r.record_id==record.revises)
            if prior.kind!=record.kind and not {prior.kind,record.kind}<={'task','goal'}:raise ValueError('A revision cannot silently change the origin/function kind')
        with torch.no_grad():state=self._mental_state(record,self.state).detached()
        self.mental_records+= (record,);self.operations+=(('mental',len(self.mental_records)-1),)
        self._set_state(state)
        return dict(status='non_fact_recorded',mental_records=len(self.mental_records),public_observations=len(self.observations))

    def tick(self,tick:CognitiveTick,*,cortical_read=None):
        self._check()
        if not self.model.config.cognitive_workspace or not isinstance(tick,CognitiveTick):raise ValueError('Explicit cognitive opportunity required')
        if tick in self.cognitive_ticks:return {'status':'exact_tick_replay_ignored'}
        if any(t.operation_id==tick.operation_id for t in self.cognitive_ticks):raise ValueError('Cognitive identity conflicts')
        with torch.no_grad():state=self.model.cognitive_tick(self.state,tick,cortical_read=cortical_read,drift_level=self._drift_level(),**self.mode.forward_options()).detached()
        if self.model.config.goal_joint_control:
            # the joint head chose the cell; the row is part of the decision
            goal_slot=int(state.cognitive_goal[0])
            table=state.cognitive_joint[0,goal_slot]
            action=ACTIONS[int(table.argmax())]
            probability=float(table.max())
            goal_id=(None if goal_slot>=self.model.config.workspace_slots
                     else int(state.goal_ids[0,goal_slot]))
            goal=goal_slot if goal_id is not None else None
        else:
            action=ACTIONS[int(state.cognitive_distribution[0].argmax())]
            probability=float(state.cognitive_distribution[0].max())
            goal_slot=goal_id=goal=None
        self.cognitive_ticks+=(tick,);self.tick_actions+=(action,);self.tick_goals+=(goal,);self.operations+=(('tick',len(self.cognitive_ticks)-1),)
        self._set_state(state)
        return dict(action=action,probabilities=state.cognitive_distribution[0].cpu().tolist(),
            goal_slot=goal_slot,goal_id=goal_id,probability=probability,
            opportunities=int(state.cognitive_ticks[0]),deliberations=int(state.deliberations[0]),
            elapsed_seconds=float(state.elapsed_seconds[0]),calibrated=False)

    def observe(self, event: PublicObservation, *, origin='observation'):
        self._check()
        if origin != 'observation' or not isinstance(event, PublicObservation):
            raise ValueError('Only public observations may enter the factual journal')
        if event in self.observations:
            return {'status': 'exact_replay_ignored', 'journal_sha256': self.journal_sha256}
        # Same ID with a different payload is retained as a conflict, never
        # silently overwritten or resolved using a hidden teacher answer.
        conflict = any(o.event_id == event.event_id for o in self.observations)
        with torch.no_grad():
            staged = self._ingest(event, self.state).detached()
        return self._accept_observation(event,staged,conflict=conflict)

    def _accept_observation(self,event,staged,*,conflict):
        """Finalize one already-computed causal observation, including trim.

        Shared by ordinary training intake and native single-pass perception.
        The trusted native runtime commits only its actual prefill output.
        """
        if staged.binding!=self.state.binding or not torch.equal(staged.writes,self.state.writes+1):
            raise ValueError('One observed model transition required')
        self.observations += (event,)
        self.operations += (('observation',len(self.observations)-1),)
        self._set_state(staged)
        trimmed = None
        if self.auto_trim is not None:
            since_trim = len(self.observations) - self._last_trim_obs
            if since_trim >= self.auto_trim['trigger']:
                trimmed = self.consolidate(InternalConsolidation(
                    operation_id=f"auto-trim-{len(self.observations)}-{len(self.consolidations)}",
                    known_at=event.known_at))
                self._last_trim_obs = len(self.observations)
        result = {'status': 'conflict_preserved' if conflict else 'observed',
                'events': len(self.observations), 'journal_sha256': self.journal_sha256}
        if trimmed is not None:
            result['auto_trimmed'] = {'consolidations': len(self.consolidations)}
        return result

    def consolidate(self, operation: InternalConsolidation):
        """Internal slow-state transition; never a new factual observation."""
        self._check()
        if not isinstance(operation,InternalConsolidation):
            raise ValueError('An explicit internal consolidation operation required')
        if operation in self.consolidations:
            return {'status':'exact_operation_replay_ignored','journal_sha256':self.journal_sha256}
        if any(o.operation_id==operation.operation_id for o in self.consolidations):
            raise ValueError('Internal operation ID reused with a different logical time')
        with torch.no_grad():
            state = self._consolidate_state(self.state).detached()
        self.consolidations += (operation,)
        self.operations += (('consolidation',len(self.consolidations)-1),)
        self._set_state(state)
        return {'status':'consolidated','observation_events':len(self.observations),
                'consolidation_operations':len(self.consolidations),'journal_sha256':self.journal_sha256}

    def rebuild(self, *, gradients=False,admission_trace=None,metadata_audit=False):
        """All observations, in time order. No implicit truncation or batching."""
        if self.model.config.native_single_pass and (gradients or not metadata_audit):
            raise ValueError('Native-conditioned experience requires its joint capsule; ordinary journal rebuild is not an encoder migration')
        def run():
            state = self.model.initial_state(1)
            for kind,index in self.operations:
                if kind=='observation':
                    event=self.observations[index];state=self._ingest(event,state)
                    if admission_trace is not None and self.model.config.episodic_policy:
                        admission_trace.append((event.event_id,index+1,state.admission_distribution))
                elif kind=='consolidation':state=self._consolidate_state(state)
                elif kind=='mental':state=self._mental_state(self.mental_records[index],state)
                elif kind=='tick':state=self.model.cognitive_tick(state,self.cognitive_ticks[index],
                        replay_action=ACTIONS.index(self.tick_actions[index]),
                        replay_goal=self.tick_goals[index]if self.model.config.goal_joint_control else None,
                    drift_level=self._drift_level(),**self.mode.forward_options())
                else:raise ValueError('Unknown chronological operation')
            return state
        if gradients:
            return run()
        with torch.no_grad():
            return run().detached()

    def _drift_level(self):
        return (self.model.core.norms[0].weight.new_tensor([self.drift_ema])
            if self.model.config.body_drift_coupling and self.drift_ema is not None else None)

    def _forward(self, prompt, candidates, goal, state, *, answer='', mode=None,body_choice_labels=False):
        prefix = self._ids(prompt)
        ids = torch.cat((prefix, self._ids(answer)), -1) if answer else prefix
        encoded = [self._ids(c)[0] for c in candidates]
        kwargs=dict(decision_positions=torch.tensor([prefix.shape[1]-1], device=self.device))
        if body_choice_labels:kwargs['body_choice_labels']=True
        if encoded:
            # Padding is masked and never used as evidence or teacher targets.
            padded = torch.nn.utils.rnn.pad_sequence(encoded,batch_first=True,padding_value=0)
            cmask=(torch.arange(padded.shape[1],device=self.device)[None]
                   <torch.tensor([len(c) for c in encoded],device=self.device)[:,None])
            kwargs.update(candidate_ids=padded[None],candidate_mask=cmask.long()[None])
        if goal:
            g = self._ids(goal)
            kwargs.update(goal_ids=g, goal_mask=torch.ones_like(g))
        selected = self.mode if mode is None else mode
        drift=self._drift_level()
        output = self.model(ids, torch.ones_like(ids), state, drift_level=drift,
                            **kwargs, **selected.forward_options())
        return output, ids, prefix.shape[1]

    @torch.no_grad()
    def query(self, prompt, candidates, *, goal='', mode=None,body_choice_labels=False):
        self._check()
        if mode is not None and not isinstance(mode,ComputationMode):
            raise ValueError('Explicit fixed-state query intervention required')
        return self._forward(prompt, candidates, goal, self.state,mode=mode,body_choice_labels=body_choice_labels)[0]

    def losses(self, example: JointExample):
        self._check()
        for reference in (*example.supporting_events,*example.disposable_events):
            if sum(o.event_id==reference for o in self.observations)!=1:
                raise ValueError('Read grounding needs one unambiguous observed public event: '+reference)
        trace=[]
        state = self.rebuild(gradients=True,admission_trace=trace)
        output, ids, prompt_length = self._forward(example.prompt, example.candidates,
            example.goal, state, answer=example.answer,body_choice_labels=example.choice_labels)
        # Training-only scratch. Neither the references nor teacher words enter
        # persistent experience or the forward that made the prompt decision.
        output.learning_memory_state=state
        output.learning_admission_trace=trace
        target = output.native_logits.new_tensor([example.native_target])
        if example.native_kind=='language':
            native=output.shared_hidden.sum()*0
        elif example.native_kind == 'choice':
            native = -(target * F.log_softmax(output.native_logits, -1)).sum(-1).mean()
        else:
            predicates = getattr(output, 'predicate_logits', None)
            if predicates is None:
                raise ValueError('Independent predicates require the separated native field')
            native = F.binary_cross_entropy_with_logits(predicates, target)
        # Position prompt_length-1 predicts the FIRST answer token. Teacher
        # tokens subsequently predict their successors. Native judgment stays
        # at the last prompt position, before any answer is observed.
        answer_logits = output.language_logits[:, prompt_length-1:-1]
        answer_tokens = ids[:, prompt_length:]
        language = F.cross_entropy(answer_logits.float().reshape(-1, answer_logits.shape[-1]),
                                   answer_tokens.reshape(-1))
        return native, language, output

    def hypothesis_loss(self,example,output):
        """Set coverage plus admissible/rejected contrast in frozen-body space.

        References are loss-side descriptions, never model candidates or facts.
        Similarity fitting is not an independent truth/spark evaluator. This
        initial target metric inherits limitations of the frozen language body.
        """
        if not example.hypothesis_targets:
            return output.shared_hidden.sum()*0
        if output.hypothesis_vectors is None or output.hypothesis_gates is None:
            raise ValueError('Hypothesis targets require an internal adaptive population')
        descriptions=(*example.hypothesis_targets,*example.hypothesis_rejections)
        references=[]
        with torch.no_grad():
            for text in descriptions:
                ids=self._ids(text)
                if self.retention_reference is not None:
                    hidden=self.retention_reference.hidden(ids,torch.ones_like(ids))
                else:
                    hidden=self.model.body.hidden(self.model.body.embed(ids),torch.ones_like(ids))
                references.append(hidden[:,-1].to(device=self.device,dtype=self.model.to_body.weight.dtype))
        reference=F.normalize(torch.stack(references,1).detach(),dim=-1)
        predicted=F.normalize(self.model.to_body(output.hypothesis_vectors),dim=-1)
        similarities=torch.einsum('bkd,brd->bkr',predicted,reference)
        # Selected slots alone transmit. The straight-through allocator is a
        # declared surrogate; it does not turn count/quality into a truth score.
        count=len(example.hypothesis_targets)
        positive=similarities[:,:,:count]
        weights=self.model.core.hypotheses.selection_distribution((positive/.1).transpose(1,2),
            output.hypothesis_gates[:,None].expand(-1,count,-1))
        coverage=(weights*(1-positive.transpose(1,2))).sum(-1).mean()
        if not example.hypothesis_rejections:
            return coverage
        admissibility=torch.logsumexp(similarities/.1,-1)-torch.logsumexp(positive/.1,-1)
        allocation=output.hypothesis_gates/output.hypothesis_gates.sum(-1,keepdim=True)
        return coverage+(admissibility*allocation).sum(-1).mean()

    def retention_loss(self,index=None):
        if self.retention_reference is None:return next(self.model.parameters()).new_zeros(())
        self.retention_reference.check(self.model)
        selected=self.optimizer_steps%len(self.retention_texts) if index is None else index
        if type(selected) is not int or not 0<=selected<len(self.retention_texts):
            raise ValueError('Retention probe index outside the declared set')
        ids=self._ids(self.retention_texts[selected]);mask=torch.ones_like(ids)
        reference=self.retention_reference.logits(ids,mask)[:,-1].float().log_softmax(-1)
        predicted=self.model(ids,mask,self.model.initial_state(1),**self.mode.forward_options()).language_logits[:,-1].float().log_softmax(-1)
        return (reference.exp()*(reference-predicted)).sum(-1).mean().clamp_min(0)

    def typed_losses(self,example,output):
        """Optional loss-side labels; absent labels make no quality claim.

        Utility is a real-valued outcome/cost label, not a truth probability.
        Initiation labels train the internal head, not an external scheduler.
        """
        zero=output.shared_hidden.sum()*0
        losses={k:zero for k in ('predicate','utility','control','initiate')}
        if example.supporting_events:
            if not self.model.config.episodic_policy or not hasattr(output,'learning_memory_state'):
                raise ValueError('Read grounding requires actual episodic training scratch')
            memory=output.learning_memory_state
            ordinals={i+1 for i,o in enumerate(self.observations) if o.event_id in example.supporting_events}
            slots=memory.episode_event_ids
            if memory.archive_event_ids is not None:slots=torch.cat((slots,memory.archive_event_ids),-1)
            target=torch.zeros_like(slots,dtype=torch.bool)
            for identity in ordinals:target|=slots==identity
            position=self._ids(example.prompt).shape[1]-1
            dense=getattr(output,'soft_candidate_reads',None)
            attention=(output.candidate_reads if dense is None else dense)[:,position].float()
            if attention.shape[-1]!=slots.shape[-1]:raise ValueError('Grounding/source slot alignment differs')
            mass=(attention*target[:,None]).sum(-1).clamp(max=1.)
            # Missing evidence is not reassigned to some unrelated remaining slot.
            losses['evidence_read']=-mass.clamp_min(1e-8).log().mean()
            if self.model.config.abstain_read:
                allocation=output.read_allocation[:,position].float()
                losses['evidence_engage']=-allocation[:,:-1].sum(-1).clamp_min(1e-8).log().mean()
            # The hard policy chooses one physical address or skip. A large
            # SUM of many weak write probabilities can still lose to skip.
            # Ground at least one executable write, not aggregate keep mass.
            admissions=[p[:,1:].amax(-1).clamp_min(1e-8).log().neg().mean()
                for eid,ordinal,p in output.learning_admission_trace if ordinal in ordinals]
            if len(admissions)!=len(ordinals):raise ValueError('Missing causal admission trace')
            losses['evidence_admission']=torch.stack(admissions).mean()
        if example.disposable_events:
            skipped=[-p[:,0].clamp_min(1e-8).log().mean()for eid,ordinal,p in output.learning_admission_trace if eid in example.disposable_events]
            if len(skipped)!=len(example.disposable_events):raise ValueError('Missing causal disposal trace')
            losses['evidence_skip']=torch.stack(skipped).mean()
        if example.predicate_targets:
            if output.predicate_logits is None:raise ValueError('Separated predicate head required')
            target=output.predicate_logits.new_tensor([example.predicate_targets])
            losses['predicate']=F.binary_cross_entropy_with_logits(output.predicate_logits,target)
        if example.utility_targets:
            if output.action_values is None:raise ValueError('Utility head required')
            target=output.action_values.new_tensor([example.utility_targets])
            losses['utility']=F.smooth_l1_loss(output.action_values,target)
        if example.control_target:
            target=output.control_logits.new_tensor([example.control_target])
            losses['control']=-(target*output.control_logits[:,:3].float().log_softmax(-1)).sum(-1).mean()
        if example.initiate_target is not None:
            if output.initiate_logits is None:raise ValueError('Internal initiation head required')
            position=self._ids(example.prompt).shape[1]-1
            logits=output.initiate_logits[:,position]
            losses['initiate']=F.binary_cross_entropy_with_logits(logits,
                logits.new_full(logits.shape,example.initiate_target))
        if example.cognitive_control_target:
            if output.cognitive_controls is None:raise ValueError('Persistent cognition control head required')
            position=self._ids(example.prompt).shape[1]-1
            logits=output.cognitive_controls[:,position].float()
            target=logits.new_tensor([example.cognitive_control_target])
            losses['cognitive_control']=-(target*logits.log_softmax(-1)).sum(-1).mean()
        return losses

    def rebuild_with_capsule(self,owner,certificate):
        """Validate and retain the numeric-identical frozen-basis native banks.

        Output-only learning changes a shared vocabulary/native readout, not
        the input encoder. The certified transaction carries the four banks;
        no journal text is replayed as a newly measured experience.
        """
        if owner.session is not self:raise ValueError('Native bank owner differs')
        certificate.verify(self.model);self._check()
        owner.codes.check(self.state);owner.mental_codes.check(self.state)
        if owner.capsule.encoder_binding!=self.state.binding:raise ValueError('Native capsule binding differs')
        return self.state,owner.capsule,owner.codes,owner.mental_codes

    def train_step(self, example, optimizer, *, native_weight=1., language_weight=1.,
                   hypothesis_weight=.1,retention_weight=1.,typed_weight=1.,max_grad_norm=1.,native_owner=None):
        self._check()
        if self.model.config.native_single_pass:
            if native_owner is not None:
                from individual_learning import learn
                return learn(native_owner,example.prompt,example.answer,steps=1,retention_weight=retention_weight)
            raise ValueError('Native lifetime updates require a frozen-basis transaction; legacy full-body rebuild is not valid')
        _optimizer_groups(optimizer, self._parameters)
        if any(not math.isfinite(v) or v <= 0 for v in (native_weight, language_weight, max_grad_norm)):
            raise ValueError('Finite positive loss weights and gradient limit required')
        if not math.isfinite(hypothesis_weight) or hypothesis_weight<0:
            raise ValueError('Nonnegative finite hypothesis-loss weight required')
        if not math.isfinite(retention_weight) or retention_weight<0:
            raise ValueError('Nonnegative finite retention-loss weight required')
        if not math.isfinite(typed_weight) or typed_weight<0:
            raise ValueError('Nonnegative finite typed-loss weight required')
        optimizer.zero_grad(set_to_none=True)
        native,language,output=self.losses(example)
        hypotheses=self.hypothesis_loss(example,output)
        retention=self.retention_loss()
        typed=self.typed_losses(example,output)
        if self.model.config.body_drift_coupling:
            level = float(retention.detach())
            self.drift_ema = level if self.drift_ema is None else 0.9*self.drift_ema + 0.1*level
        loss=(native_weight*native+language_weight*language+hypothesis_weight*hypotheses+
              retention_weight*retention+typed_weight*sum(typed.values()))
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite joint loss: no optimizer update made')
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(list(self._parameters.values()), max_grad_norm,
                                                   error_if_nonfinite=True)
        old_binding = self.state.binding
        self._set_state(None)  # Queries are forbidden during an incomplete update.
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.optimizer_steps += 1  # Count actual optimizer calls, including a later rebuild failure.
        if any(not torch.isfinite(p).all() for p in self._parameters.values()):
            raise ValueError('Optimizer produced nonfinite parameters; session invalidated')
        self.model.seal_parameters()
        self._set_state(self.rebuild())
        self._check()
        return {'optimizer_steps': self.optimizer_steps, 'native_loss': float(native.detach()),
                'language_loss': float(language.detach()), 'grad_norm_before_clip': float(grad_norm),
                'hypothesis_alignment_loss':float(hypotheses.detach()),
                'typed_losses':{k:float(v.detach()) for k,v in typed.items()},
                'typed_weight':typed_weight,
                'retention_kl':float(retention.detach()),'retention_weight':retention_weight,
                'retention_probe_index':((self.optimizer_steps-1)%len(self.retention_texts) if self.retention_texts else None),
                'old_binding': old_binding, 'binding': self.state.binding,
                'public_events_reencoded': len(self.observations),
                'consolidations_replayed':len(self.consolidations),
                'computation_mode':asdict(self.mode),
                'journal_sha256': self.journal_sha256}

    @classmethod
    def upgrade_observation_session(cls, previous, optimizer):
        """Upgrade the old observation-only protocol, preserving actual state.

        Rebuild and compare every tensor before adopting; do not retag a state.
        The SAME optimizer is retained and checked. This is a protocol upgrade
        on one body, not a cross-backbone learned transfer mechanism.
        """
        previous._check()
        if hasattr(previous,'operations') or previous.state.consolidations.any():
            raise ValueError('Only the prior observation-only protocol can be upgraded here')
        result = cls(previous.model,previous.encode,encoding_sha256=previous.encoding_sha256)
        result.observations = tuple(PublicObservation(**asdict(o)) for o in previous.observations)
        result.operations = tuple(('observation',i) for i in range(len(result.observations)))
        _optimizer_groups(optimizer,result._parameters)
        rebuilt = result.rebuild()
        if rebuilt.binding != previous.state.binding or any(not torch.equal(t,previous.state.tensors()[name])
                                                           for name,t in rebuilt.tensors().items()):
            raise ValueError('Protocol upgrade failed to reconstruct exact previous experience')
        result._set_state(rebuilt)
        result.optimizer_steps = previous.optimizer_steps
        result._check()
        return result

    def save(self, path, optimizer):
        """Exclusive checkpoint: trainable weights/optimizer plus public journal.

        Frozen weights stay in the separately identified body asset. This is
        not a complete deployable model export or a lifelong storage service.
        """
        self._check()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        save_learning_checkpoint(path/'learning', self.model, optimizer,
            binding=self.contract_identity, step=self.optimizer_steps)
        public = _json(self._journal())+'\n'
        (path/'journal.json').write_text(public, encoding='utf-8')
        controls = {'auto_trim':self.auto_trim,'last_trim_obs':self._last_trim_obs,
                    'drift_ema':self.drift_ema}
        (path/'control-state.json').write_text(_json(controls)+'\n',encoding='utf-8')
        manifest = {'schema': 'treyrin-structural-learning-v3',
            'contract_identity': self.contract_identity, 'parameter_binding': self.state.binding,
            'optimizer_steps': self.optimizer_steps, 'journal_sha256': self.journal_sha256,
            'files': {name: _hash((path/name).read_bytes()) for name in
                      ('journal.json', 'control-state.json', 'learning/manifest.json')}}
        (path/'manifest.json').write_text(_json(manifest)+'\n', encoding='utf-8')
        return manifest

    def restore(self, path, optimizer):
        """Restore into an empty, identically constructed candidate session."""
        self._check()
        if self.operations or self.optimizer_steps:
            raise ValueError('Restore requires a new empty session')
        path = Path(path)
        m = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
        if m.get('schema') != 'treyrin-structural-learning-v3' or m.get('contract_identity') != self.contract_identity:
            raise ValueError('Learning architecture/frozen body/tokenizer contract differs')
        if set(m.get('files', {})) != {'journal.json', 'control-state.json', 'learning/manifest.json'}:
            raise ValueError('Unexpected session files')
        for name, expected in m['files'].items():
            if _hash((path/name).read_bytes()) != expected:
                raise ValueError('Session hash differs: '+name)
        journal = json.loads((path/'journal.json').read_text(encoding='utf-8'))
        if _hash(_json(journal).encode()) != m['journal_sha256']:
            raise ValueError('Invalid operation journal')
        observations,consolidations,mental,ticks,actions,goals,operations=self._parse_journal(journal)
        controls=json.loads((path/'control-state.json').read_text(encoding='utf-8'))
        if set(controls)!={'auto_trim','last_trim_obs','drift_ema'} or controls['auto_trim']!=self.auto_trim:
            raise ValueError('Invalid saved lifecycle policy')
        last_trim=controls['last_trim_obs'];drift=controls['drift_ema']
        if type(last_trim) is not int or not 0<=last_trim<=len(observations):
            raise ValueError('Invalid saved trim position')
        if drift is not None and (type(drift) not in (int,float) or not math.isfinite(drift) or drift<0):
            raise ValueError('Invalid saved body drift')
        self._set_state(None)
        steps = load_learning_checkpoint(path/'learning', self.model, optimizer, binding=self.contract_identity)
        if steps != m['optimizer_steps'] or self.model.seal_parameters() != m['parameter_binding']:
            raise ValueError('Restored weights/step differ from the session manifest')
        self.observations, self.optimizer_steps = observations, steps
        self.consolidations,self.operations = consolidations,operations
        self.mental_records,self.cognitive_ticks,self.tick_actions=mental,ticks,actions
        self.tick_goals=goals if len(goals)==len(ticks) else (None,)*len(ticks)
        self._last_trim_obs,self.drift_ema=last_trim,drift
        self._set_state(self.rebuild())
        self._check()
        return m
