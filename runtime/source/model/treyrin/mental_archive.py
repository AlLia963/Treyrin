"""Typed cold neural memory for non-facts, disjoint from observed events.

Physical admission uses the existing learned episodic actor. Empty-capacity
legality preserves messages; after saturation the actor chooses skip/replace.
No journal lookup or truth conversion is performed.
"""
from dataclasses import replace
import torch

FIELDS=('mental_archive_vectors','mental_archive_sources','mental_archive_ids',
        'mental_archive_kinds','mental_archive_statuses','mental_archive_admissions')

def initialize(state,slots,width):
    if not slots:return state
    ref=state.working;b=ref.shape[0];shape=(b,slots)
    return replace(state,mental_archive_vectors=ref.new_zeros(b,slots,width),
        mental_archive_sources=ref.new_zeros(b,slots,width),
        mental_archive_ids=torch.full(shape,-1,device=ref.device,dtype=torch.long),
        mental_archive_kinds=torch.zeros(shape,device=ref.device,dtype=torch.long),
        mental_archive_statuses=torch.zeros(shape,device=ref.device,dtype=torch.long),
        mental_archive_admissions=torch.zeros(b,device=ref.device,dtype=torch.long))

def validate(state,slots,width):
    if not slots:
        if any(getattr(state,n,None)is not None for n in FIELDS):raise ValueError('Unexpected non-fact archive')
        return
    b=state.working.shape[0]
    for n in FIELDS:
        t=getattr(state,n);shape=((b,slots,width)if n.endswith(('vectors','sources'))else (b,)if n.endswith('admissions')else(b,slots))
        dtype=state.working.dtype if n.endswith(('vectors','sources'))else torch.long
        if not isinstance(t,torch.Tensor)or tuple(t.shape)!=shape or t.dtype!=dtype or t.device!=state.working.device:
            raise ValueError('Invalid typed non-fact archive: '+n)
        if dtype==state.working.dtype and not bool(torch.isfinite(t).all()):raise ValueError('Nonfinite non-fact archive')
    ids=state.mental_archive_ids
    if bool(((ids>state.workspace_seen[:,None])|(ids==0)|(ids<-1)).any()):raise ValueError('Invalid archived mental identity')
    for n,limit in(('mental_archive_kinds',6),('mental_archive_statuses',4)):
        if bool(((getattr(state,n)<0)|(getattr(state,n)>=limit)).any()):raise ValueError('Invalid archived mental type/status')
    if bool(((state.mental_archive_admissions<0)|(state.mental_archive_admissions>state.workspace_seen)).any()):raise ValueError('Invalid mental archive admission counter')

def decorated(workspace,state):
    if state.mental_archive_ids is None:return None
    return (state.mental_archive_vectors+.1*state.mental_archive_sources+
            workspace.kind(state.mental_archive_kinds)+workspace.status(state.mental_archive_statuses))

def commit(model,before,after,value,source,kind,status,revision_id):
    if not model.config.mental_archive_slots:return after
    from .cognitive_workspace import KINDS,STATUSES
    ids=before.mental_archive_ids;statuses=before.mental_archive_statuses
    if revision_id is not None:
        statuses=torch.where(ids==revision_id,STATUSES.index('retracted'if status=='retracted'else'superseded'),statuses)
    staged=replace(after,mental_archive_statuses=statuses)
    content=decorated(model.core.workspace,staged);proposal=value+model.core.workspace.kind.weight[KINDS.index(kind)]+model.core.workspace.status.weight[STATUSES.index(status)]
    free=ids<0;hasfree=free.any(-1)
    decision=model.core.episodic_policy(torch.nn.functional.normalize(value,dim=-1),proposal,source,
        before.working[:,-1]+model.core.workspace.goal_context(after),
        torch.nn.functional.normalize(before.mental_archive_vectors,dim=-1),content,before.mental_archive_sources,
        ids>0,ids,before.workspace_seen,stage=3,
        write_mask=torch.where(hasfree[:,None],free,torch.ones_like(free)),allow_skip=~hasfree)
    w=decision.write_weights[:,:,None];hard=w.detach().squeeze(-1).bool()
    return replace(staged,mental_archive_vectors=before.mental_archive_vectors*(1-w)+value[:,None]*w,
        mental_archive_sources=before.mental_archive_sources*(1-w)+source[:,None]*w,
        mental_archive_ids=torch.where(hard,after.workspace_seen[:,None],ids),
        mental_archive_kinds=torch.where(hard,KINDS.index(kind),before.mental_archive_kinds),
        mental_archive_statuses=torch.where(hard,STATUSES.index(status),statuses),
        mental_archive_admissions=before.mental_archive_admissions+(decision.action>0).long())
