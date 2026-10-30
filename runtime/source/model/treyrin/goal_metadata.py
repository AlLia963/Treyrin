"""Immutable goal metadata protocol; values are declarations, not world truth."""
import json,math

FIELDS={'priority','interruption_cost','deadline_at','acceptance','source_kind','evidence_ok','evidence_open','can_act'}

def parse_goal_metadata(value,evidence_ids=()):
    if not isinstance(value,str)or not value:raise ValueError('Immutable JSON goal metadata required')
    data=json.loads(value)
    if not isinstance(data,dict)or set(data)!=FIELDS:raise ValueError('Explicit goal metadata schema required')
    for name in('priority','interruption_cost'):
        x=data[name]
        if type(x)not in(float,int)or not math.isfinite(x)or x<0:raise ValueError('Finite declared goal cost/priority required')
    if data['priority']>1:raise ValueError('Declared priority must be in [0,1]')
    deadline=data['deadline_at']
    if deadline is not None and(type(deadline)not in(float,int)or not math.isfinite(deadline)or deadline<=0):raise ValueError('Positive epoch deadline or null required')
    if data['acceptance']not in('accepted','pending','declined')or data['source_kind']not in('user_message','device_log','document','tool_result','model_output','schedule'):
        raise ValueError('Known acceptance/source types required')
    for name in('evidence_ok','evidence_open'):
        if type(data[name])is not int or data[name]<0:raise ValueError('Nonnegative declared evidence counts required')
    if data['evidence_ok']>len(evidence_ids):raise ValueError('Satisfied count exceeds cited public sources')
    if type(data['can_act'])is not bool:raise ValueError('Explicit action permission required')
    if data['source_kind']=='model_output'and data['acceptance']=='accepted':raise ValueError('Model proposal cannot accept itself as user authority')
    return data

def goal_metadata(**kwargs):
    data=dict(priority=.5,interruption_cost=0.,deadline_at=None,acceptance='pending',
        source_kind='model_output',evidence_ok=0,evidence_open=0,can_act=False)
    data.update(kwargs);text=json.dumps(data,ensure_ascii=False,sort_keys=True,separators=(',',':'))
    # Evidence count validation with real references occurs in MentalRecord.
    return text
