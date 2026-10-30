"""Same-body natural intent -> zero or more typed, non-factual commitments.

Empty goals are normal. Validation checks provenance/schema/capacity only;
intent and urgency are never inferred by a keyword router.
"""
from datetime import datetime
import json
from treyrin.intake import parse_goal_proposal
from treyrin.goal_metadata import goal_metadata

def parse_intents(raw,*,known_evidence,active_goals,now):
    text=raw.strip()
    if text.startswith('```'):
        text=text.split('\n',1)[-1].strip()
        if text.endswith('```'):text=text[:-3].strip()
    data=json.loads(text)
    if not isinstance(data,dict)or set(data)!={'goals'}or not isinstance(data['goals'],list):
        raise ValueError('Explicit goals list required')
    proposals=[parse_goal_proposal(json.dumps(p,ensure_ascii=False),known_evidence=known_evidence,
        known_records=active_goals,now=now)for p in data['goals']]
    if len({p.record_id for p in proposals})!=len(proposals):raise ValueError('Repeated generated goal identity')
    if any(p.record_id in active_goals for p in proposals):raise ValueError('Revision requires a new immutable identity')
    parents=[p.revision_of for p in proposals if p.revision_of]
    if len(set(parents))!=len(parents):raise ValueError('Conflicting simultaneous revisions of one commitment')
    return proposals

class IntentLifecycle:
    def __init__(self,native):self.native=native;self.attempts=[]
    def intake(self,user_text,*,evidence_id,known_at,max_new_tokens=None,commit=True):
        n=self.native;observations={o.event_id:o for o in n.session.observations}
        if evidence_id not in observations:raise ValueError('Actual observed source identity required')
        s=n.session.state;active_ordinals=set(s.goal_ids[(s.goal_ids>0)&(s.goal_statuses==0)].tolist())
        active=[r.record_id for i,r in enumerate(n.session.mental_records,1)if i in active_ordinals]
        # The public message is the present request, not a recovered old journal
        # payload. Older semantics are available only through neural replay.
        prompt=('判断本条用户消息是否提出承诺/任务、修订或取消。闲聊没有任务时必须输出{"goals":[]}。'
            '可同时提出多项，禁止为了填字段虚构任务。仅输出JSON {"goals":[...]}。每项格式：'
            '{"record_id":"goal-唯一新身份","kind":"goal或task","text":"具体目标",'
            '"source_kind":"user_message","evidence_ids":["'+evidence_id+'"],'
            '"priority":0.5,"interruption_cost":0.1,"deadline_at":null,"confidence":0.8,'
            '"acceptance":"accepted或pending或declined","autonomy":"notify_only或can_act",'
            '"revision_of":null}。取消用declined且revision_of指向当前目标；授权不清用pending。'
            '不要扩大用户的行动授权；无截止时间填null。\n现在：'+known_at+
            '\n当前可修订目标身份：'+json.dumps(active,ensure_ascii=False)+'\n用户消息：'+user_text)
        generated=n.query(prompt,max_new_tokens=max_new_tokens)
        attempt=dict(known_at=known_at,raw=generated,prompt=prompt,status='generated');self.attempts.append(attempt)
        proposals=parse_intents(generated['answer'],known_evidence=observations,active_goals=active,now=datetime.fromisoformat(known_at))
        saved=(n.session,n.capsule,n.codes,n.mental_codes,n.candidate.session)
        before=int(s.writes[0]);receipts=[]
        try:
            for p in proposals:
                if not commit:continue
                metadata=goal_metadata(priority=p.priority,interruption_cost=p.interruption_cost,
                    deadline_at=None if p.deadline_at is None else datetime.fromisoformat(p.deadline_at).timestamp(),
                    acceptance=p.acceptance,source_kind=p.source_kind,evidence_ok=0,evidence_open=0,
                    can_act=p.autonomy=='can_act')
                receipts.append(n.remember(dict(record_id=p.record_id,text=p.text,kind=p.kind,known_at=known_at,
                    source='same-body-user-intent-proposal',evidence_ids=p.evidence_ids,revises=p.revision_of or'',
                    status='retracted'if p.acceptance=='declined'else'active',goal_metadata=metadata)))
            if int(n.session.state.writes[0])!=before:raise RuntimeError('Intent extraction became a factual observation')
        except Exception:
            n.session,n.capsule,n.codes,n.mental_codes,n.candidate.session=saved
            attempt['status']='rejected';raise
        attempt['status']='accepted'
        return dict(proposals=[p.as_dict()for p in proposals],receipts=receipts,raw_generation=generated,
                    no_goal=len(proposals)==0,actual_generator='same persistent body',capability_verified=False)
