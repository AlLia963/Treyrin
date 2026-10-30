"""Same-body, typed hypothesis generation and revision.

Validation checks shape/provenance, never whether a scientific claim is true.
Choice probabilities rank generated descriptions; they are not calibrated
truth probabilities. No factual observation is committed by this interface.
"""
from dataclasses import dataclass
import hashlib
import json
import math


@dataclass(frozen=True)
class Spark:
    description:str
    prediction:str
    test:str
    evidence_ids:tuple
    confidence:float

    def text(self):
        return json.dumps(dict(description=self.description,prediction=self.prediction,
            test=self.test,confidence=self.confidence,confidence_calibrated=False),ensure_ascii=False,sort_keys=True)


def parse_sparks(raw,known_ids,*,maximum=None):
    text=raw.strip()
    if text.startswith('```'):
        text=text.split('\n',1)[-1]
        if text.rstrip().endswith('```'):text=text.rstrip()[:-3]
        text=text.strip()
    try:
        value=json.loads(text)
    except json.JSONDecodeError:
        if not text:
            raise ValueError('Empty Spark answer')
        return [Spark(text,'未结构化预测','等待后续观察检验',(),0.5)]
    if not isinstance(value,dict)or set(value)!={'hypotheses'}:raise ValueError('Exactly one hypotheses object required')
    rows=value['hypotheses']
    if not isinstance(rows,list)or not rows:raise ValueError('At least one explicit hypothesis required')
    out=[];descriptions=set();known=set(known_ids)
    for row in rows:
        if not isinstance(row,dict)or set(row)!={'description','prediction','test','evidence_ids','confidence'}:
            raise ValueError('Explicit description/prediction/test/evidence/confidence fields required')
        for n in('description','prediction','test'):
            if not isinstance(row[n],str)or not row[n].strip():raise ValueError('Nonempty hypothesis field required: '+n)
        if row['description']in descriptions:raise ValueError('Duplicate hypothesis description')
        descriptions.add(row['description'])
        refs=row['evidence_ids'];confidence=row['confidence']
        if not isinstance(refs,list)or any(not isinstance(x,str)for x in refs)or len(set(refs))!=len(refs)or not set(refs)<=known:
            raise ValueError('Hypothesis can cite only known public evidence identities')
        if type(confidence)not in(float,int)or not math.isfinite(confidence)or not 0<=confidence<=1:raise ValueError('Finite model-reported confidence required')
        out.append(Spark(row['description'],row['prediction'],row['test'],tuple(refs),float(confidence)))
    return out


class SparkLifecycle:
    def __init__(self,native_session):self.native=native_session

    def propose(self,problem,*,known_at,max_new_tokens=None):
        n=self.native;all_known=[o.event_id for o in n.session.observations];known=all_known
        prompt=('针对问题提出若干可检验的候选解释，不能把候选当作事实。说明每个解释独特的可观察预测、'
            '最小验证方法和不确定性。只用已观察证据；证据不足可不引用。只输出JSON，格式为'
            '{"hypotheses":[{"description":"...","prediction":"...","test":"...",'
            '"evidence_ids":[],"confidence":0.3}]}。confidence是未经校准的自评。\n'
            +'可引用的近期公开身份：'+json.dumps(known,ensure_ascii=False)+'\n问题：'+problem)
        before=int(n.session.state.writes[0])
        generated=n.query(prompt,max_new_tokens=max_new_tokens)
        proposals=parse_sparks(generated['answer'],all_known)
        # These exact texts define the same-body candidate population. There
        # is no attachment of a convenient sentence to an arbitrary old vector.
        ranked=n.query('哪些候选解释最值得优先验证？考虑当前证据、区分预测和验证成本。\n'+problem,
                       tuple(p.text()for p in proposals),max_new_tokens=max_new_tokens)
        scores=ranked['native_probabilities']
        if len(scores)!=len(proposals):raise ValueError('Native population differs from generated descriptions')
        from treyrin.cognitive_workspace import MentalRecord
        records=[]
        for i,p in enumerate(proposals):
            payload=p.text();n.candidate.encode(payload) # refuse overflow before any mutation
            identity=hashlib.sha256((problem+'|'+known_at+'|'+str(int(n.session.state.workspace_seen[0]))+'|'+str(i)).encode()).hexdigest()[:20]
            records.append(MentalRecord(record_id='spark-'+identity,text=payload,kind='hypothesis',
                known_at=known_at,source='model-generated',evidence_ids=p.evidence_ids))
        # Individual records still pass the core's real non-fact admission
        # policy. A skipped proposal is reported, not secretly written as fact.
        receipts=[n.remember(r.__dict__)for r in records]
        if int(n.session.state.writes[0])!=before:raise RuntimeError('Hypothesis proposal polluted factual experience')
        return dict(raw_generation=generated,proposals=[dict(record_id=r.record_id,**p.__dict__)for r,p in zip(records,proposals)],
            native_ranking=scores,ranking_interpretation='priority for testing, not truth confidence',
            preferred_index=ranked['native_choice_index'],admissions=receipts,capability_verified=False)

    def revise(self,record_id,question,*,known_at,max_new_tokens=None):
        n=self.native;prior={r.record_id:r for r in n.session.mental_records}
        if record_id not in prior or prior[record_id].kind!='hypothesis':raise ValueError('Existing hypothesis identity required')
        known={o.event_id for o in n.session.observations}
        prompt=('复查内部假设 '+record_id+'。依据目前公开观察决定保留修订还是撤回；假设不等于事实。'
            '只输出JSON {"status":"active或retracted","text":"理由、修订后的预测及下一项验证",'
            '"evidence_ids":["仅已知公开身份"]}。\n'+question)
        answer=n.query(prompt,max_new_tokens=max_new_tokens)
        text=answer['answer'].strip()
        if text.startswith('```'):
            text=text.split('\n',1)[-1].strip()
            if text.endswith('```'):text=text[:-3].strip()
        data=json.loads(text)
        if not isinstance(data,dict)or set(data)!={'status','text','evidence_ids'}or data['status']not in('active','retracted'):
            raise ValueError('Explicit hypothesis revision schema required')
        refs=data['evidence_ids']
        if not isinstance(refs,list)or any(not isinstance(x,str)for x in refs)or not set(refs)<=known:
            raise ValueError('Revision cited unknown evidence')
        if not isinstance(data['text'],str)or not data['text'].strip():raise ValueError('Revision reason required')
        n.candidate.encode(data['text'])
        before=int(n.session.state.writes[0]);ordinal=int(n.session.state.workspace_seen[0])+1
        record=dict(record_id=record_id+'-r'+str(ordinal),text=data['text'],kind='hypothesis',known_at=known_at,
            source='model-revision',evidence_ids=tuple(refs),revises=record_id,status=data['status'])
        receipt=n.remember(record)
        if int(n.session.state.writes[0])!=before:raise RuntimeError('Hypothesis revision polluted factual experience')
        return dict(raw_generation=answer,record=record,receipt=receipt,capability_verified=False)
