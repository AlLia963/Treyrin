"""Source-grounded inference branches; the model and brain state stay shared."""
import torch
from native_stream import Capsule

def revision_chain(records,record_id):
    """Ordered record ids from record_id to the current head; raises on a cycle."""
    by_target={}
    for record in records:
        if record.revises:by_target[record.revises]=record
    chain=[record_id];seen={record_id};current=record_id
    while current in by_target:
        successor=by_target[current]
        if successor.record_id in seen:
            raise ValueError('Revision chain cycles at '+successor.record_id)
        seen.add(successor.record_id);chain.append(successor.record_id);current=successor.record_id
    return chain

def current_head(records,record_id):
    """Current head of a multi-hop revision chain (cycle-safe)."""
    head=revision_chain(records,record_id)[-1]
    return next((record for record in records if record.record_id==head),None)

def source_episode(native,problem,request=''):
    from role_conversation import ConversationEpisode
    text=native._frame('user',problem)
    if request:text+=native._frame('user',request)
    text+=native._assistant_prefix
    ids=torch.tensor([native.candidate.tokenizer(text,add_special_tokens=False)['input_ids']],device=native.session.state.keys.device)
    capsule=Capsule.empty(native.candidate.model.body,window=native.window,binding=native.candidate.model._binding)
    options=native.session.mode.forward_options();options['drift_level']=native.session._drift_level()
    with torch.no_grad():return ConversationEpisode(native.candidate.model,native.session.state,ids,capsule,**options)

def route(native,current,normal,selected=None):
    c=native.candidate
    previous=[o for o in native.session.observations if o.source in ('human_user','user') and o.event_id!=native.pending_user['event_id']]
    if not previous or not callable(getattr(c,'judge',None)):return normal,None
    with torch.no_grad():
        decision=c.judge(current,('直接处理当前问题','核对并修订先前的模型回答'),viewport=0)
        p=decision['probabilities']
        if max(range(len(p)),key=lambda i:p[i])!=1:return normal,dict(used=False,route_probabilities=p)
        # Reconstruct causal reply links from the committed operation stream.
        # These are source identities, not a rule for the numerical answer.
        turns={};last=None
        for kind,index in native.session.operations:
            if kind=='observation':
                event=native.session.observations[index]
                if event.source in ('human_user','user'):last=event
            elif kind=='mental' and last is not None:
                record=native.session.mental_records[index]
                if record.kind=='assistant_output':turns[index+1]=(last,record)
        known={o.event_id:o for o in previous}
        # Read attention proposes records; selecting a correction source needs
        # a separate judgment over causal question/answer pairs. A salient but
        # unrelated old observation must not silently become the original task.
        batch=[(o,r) for o,r in turns.values() if o.event_id in known]
        if not batch:return normal,dict(used=False,route_probabilities=p)
        while len(batch)>1:
            winners=[]
            for start in range(0,len(batch),52):
                group=batch[start:start+52]
                if len(group)==1:winners.append(group[0]);continue
                choice=c.judge('应重新核对下面哪一轮回答的原始问题？当前请求：'+current,
                    tuple('用户原题：'+o.text+'；当时模型回答：'+r.text for o,r in group),viewport=0)['probabilities']
                winners.append(group[max(range(len(choice)),key=lambda i:choice[i])])
            batch=winners
        source,old=batch[0]
    chain=revision_chain(native.session.mental_records,old.record_id)
    head=next((r for r in native.session.mental_records if r.record_id==chain[-1]),old)
    old=head
    return source_episode(native,source.text,current),dict(used=True,source_event_id=source.event_id,
        route_probabilities=p,revision_of=old.record_id if old else '',
        revision_chain=chain,revision_hops=len(chain)-1,same_brain_state=True,
        decoder_history='original user evidence and current request; prior assistant claims excluded')

def recompute(candidate,problem,*,max_new_tokens=None):
    native=getattr(candidate,'active_native',None)
    if native is None:raise ValueError('Attach the existing permanent native session before correction')
    ep=source_episode(native,problem)
    with torch.no_grad():r=ep.generate_result(max_new_tokens,eos_token_id=native._eos_token_ids())
    return dict(answer=candidate.tokenizer.decode(r.token_ids,skip_special_tokens=True),finish_reason=r.finish_reason,
        eos_reached=r.eos_reached,generation_budget=r.budget,same_brain_state=True)

def revise(candidate,problem,old_claim,*,max_new_tokens=None):
    fresh=recompute(candidate,problem,max_new_tokens=max_new_tokens)
    p=candidate.judge(problem,(old_claim,fresh['answer']),viewport=0)['probabilities']
    selected=max(range(len(p)),key=lambda i:p[i])
    return dict(schema='treyrin-source-correction-v2',problem=problem,old_claim=old_claim,fresh=fresh,
        probabilities=p,changed=selected==1,revised_answer=(old_claim,fresh['answer'])[selected])
