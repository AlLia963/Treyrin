"""Persistent NON-FACT workspace and model-internal opportunity control.

Kinds identify origin/function, not truth. Retraction records withdraw a model
proposal; they never assert the opposite world proposition. The runtime supplies
time/cost/resource opportunities; a neural policy chooses wait/think/speak/ask/act.
"""
from dataclasses import dataclass,asdict
import math
import json
import torch
from torch import nn

from .goal_control import GoalActionHead

KINDS=('hypothesis','belief','assistant_output','task','goal','action')
STATUSES=('active','superseded','retracted','completed')
ACTIONS=('wait','think','speak','request_evidence','act')
ACCEPTANCE=('pending','accepted','declined')
SOURCE_KINDS=('user_message','device_log','document','tool_result','model_output','schedule')
GOAL_FIELDS=('goal_priority','goal_icost','goal_deadline_at','goal_deadline_mask','goal_accept','goal_sources_typed',
    'goal_evidence','goal_ev_ok','goal_ev_open','goal_attempts','goal_last_action','goal_revision','goal_can_act','cognitive_joint','cognitive_goal','cognitive_goal_identity')
from .mental_archive import FIELDS as MENTAL_ARCHIVE_FIELDS
WORKSPACE_FIELDS=('workspace_vectors','workspace_sources','workspace_ids','workspace_kinds','workspace_statuses',
    'workspace_seen','workspace_admissions','goal_vectors','goal_sources','goal_ids','goal_statuses',
    'cognitive_ticks','deliberations','elapsed_seconds','deadline_seconds','cognitive_context','cognitive_distribution')+MENTAL_ARCHIVE_FIELDS

@dataclass(frozen=True)
class MentalRecord:
    record_id:str
    text:str
    kind:str
    known_at:str
    source:str='model'
    evidence_ids:tuple[str,...]=()
    revises:str=''
    status:str='active'
    goal_metadata:str=''
    goal_action_of:str=''
    action_name:str=''

    def __post_init__(self):
        if any(not isinstance(getattr(self,n),str) or not getattr(self,n).strip() for n in
               ('record_id','text','kind','known_at','source','status')):
            raise ValueError('Nonempty mental identity/content/origin/time required')
        if self.kind not in KINDS or self.status not in STATUSES:
            raise ValueError('Mental kind/status is not an external fact or truth label')
        if not isinstance(self.revises,str) or any(not isinstance(e,str) or not e for e in self.evidence_ids):
            raise ValueError('Explicit prior mental identity and public evidence identities required')
        object.__setattr__(self,'evidence_ids',tuple(self.evidence_ids))
        if not isinstance(self.goal_metadata,str)or not isinstance(self.action_name,str):raise ValueError('Immutable typed metadata/action required')
        if self.goal_metadata:
            from .goal_metadata import parse_goal_metadata
            if self.kind not in('goal','task'):raise ValueError('Goal metadata requires a goal/task record')
            data=parse_goal_metadata(self.goal_metadata,self.evidence_ids)
            if data['acceptance']=='declined'and self.status!='retracted':raise ValueError('Declined goal must be retracted')
        if bool(self.goal_action_of)!=bool(self.action_name)or not isinstance(self.goal_action_of,str):raise ValueError('Paired action/goal identity required')
        if self.goal_action_of and(self.kind!='action'or self.status!='completed'or self.action_name not in('speak','request_evidence','act')):
            raise ValueError('Only a completed execution receipt may record an interruption')

    def provenance_text(self):
        return json.dumps({k:v for k,v in asdict(self).items() if k!='text'},ensure_ascii=False,sort_keys=True)

@dataclass(frozen=True)
class CognitiveTick:
    operation_id:str
    known_at:str
    elapsed_seconds:float
    deadline_remaining:float|None=None
    interruption_cost:float=0.
    resource:float=1.
    clear_deadline:bool=False

    def __post_init__(self):
        if any(not isinstance(v,str) or not v.strip() for v in (self.operation_id,self.known_at)):
            raise ValueError('Explicit cognitive opportunity identity/time required')
        for name in ('elapsed_seconds','interruption_cost','resource'):
            value=getattr(self,name)
            if type(value) not in (float,int) or not math.isfinite(value) or value<0:
                raise ValueError('Finite nonnegative time/cost/resource required')
        if self.resource>1:raise ValueError('Normalized resource in [0,1] required')
        if type(self.clear_deadline) is not bool or (self.clear_deadline and self.deadline_remaining is not None):raise ValueError('Deadline clear and remaining time cannot conflict')
        if self.deadline_remaining is not None and (type(self.deadline_remaining) not in (float,int) or not math.isfinite(self.deadline_remaining)):
            raise ValueError('Finite remaining deadline or explicit unknown required')

    def features(self,reference,total_elapsed):
        deadline=self.deadline_remaining
        return reference.new_tensor([[math.log1p(self.elapsed_seconds)/10,
            float(deadline is not None),0. if deadline is None else math.copysign(math.log1p(abs(deadline))/10,deadline),
            math.log1p(self.interruption_cost),self.resource,math.log1p(total_elapsed)/10]])

class CognitiveWorkspace(nn.Module):
    def __init__(self,width,*,slots=8,joint=False,joint_hidden=16,contextual=False,calibration=False):
        super().__init__();self.width=width;self.slots=slots;self.joint=joint;self.contextual=contextual
        self.kind=nn.Embedding(len(KINDS),width);self.status=nn.Embedding(len(STATUSES),width)
        self.key=nn.Linear(width,width,bias=False);self.query=nn.Linear(width,width,bias=False)
        self.value=nn.Linear(width,width,bias=False)
        self.control=nn.Sequential(nn.Linear(2*width+3+6,width),nn.SiLU(),nn.Linear(width,len(ACTIONS)))
        self.read_gain=nn.Parameter(torch.tensor(.02));self.goal_gain=nn.Parameter(torch.tensor(.02))
        self.idle_tau_log=nn.Parameter(torch.tensor(math.log(3600.)))
        # A conservative learned prior, not an external timer's decision.
        nn.init.zeros_(self.control[-1].bias)
        with torch.no_grad():self.control[-1].bias[0]=1.
        # ONE head for (goal row, action).  Absent when joint=False, so the
        # ancestor parameter set is untouched.
        self.goal_joint=None
        if joint:
            self.goal_joint=GoalActionHead(slots=slots,hidden=joint_hidden,goal_dim=width)
        self.cortical_calibration=None
        if calibration:
            from .cortical_calibration import CorticalUtilityCalibration
            self.cortical_calibration=CorticalUtilityCalibration(width)

    def decorated(self,state):
        return state.workspace_vectors+.1*state.workspace_sources+self.kind(state.workspace_kinds)+self.status(state.workspace_statuses)

    def goal_context(self,state):
        active=(state.goal_ids>0)&(state.goal_statuses==STATUSES.index('active'))
        weights=active.to(state.goal_vectors.dtype)
        return (state.goal_vectors*weights[:,:,None]).sum(1)/weights.sum(-1,keepdim=True).clamp_min(1.)

    def read(self,query,state):
        content=self.decorated(state)
        scores=torch.einsum('bd,bwd->bw',self.query(query),self.key(content))/self.width**.5
        occupied=state.workspace_ids>0
        from .mental_archive import decorated
        cold=decorated(self,state)
        if cold is not None:
            duplicate=(state.mental_archive_ids[:,:,None]==state.workspace_ids[:,None,:]).any(-1)
            content=torch.cat((content,cold),1)
            occupied=torch.cat((occupied,(state.mental_archive_ids>0)&~duplicate),1)
            scores=torch.einsum('bd,bwd->bw',self.query(query),self.key(content))/self.width**.5
        weights=scores.masked_fill(~occupied,torch.finfo(scores.dtype).min).softmax(-1)*occupied
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
        return torch.einsum('bw,bwd->bd',weights,self.value(content))

    def control_logits(self,state,*,current=None):
        drives=state.drive_state if state.drive_state is not None else state.working.new_zeros(state.working.shape[0],3)
        current=state.working[:,-1] if current is None else current
        return self.control(torch.cat((current,self.goal_context(state),drives,state.cognitive_context),-1))

    def goal_action(self,state,features,context,allowed=None):
        """Joint (goal row, action) decision from the single learned head."""
        if self.goal_joint is None:raise ValueError('Goal-joint control is disabled')
        if self.contextual:
            from .joint_context import joint_context_policy
            return joint_context_policy(self,state,features,context,allowed)
        null=state.goal_vectors.new_zeros(state.goal_vectors.shape[0],1,self.width)
        vectors=torch.cat((state.goal_vectors,null),1)
        return self.goal_joint(features,context,vectors,allowed,return_diagnostics=True)
