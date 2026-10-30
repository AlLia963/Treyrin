"""Runtime self-information committed as a REAL structural input.

`describe` returns text computed by self_description.py from the live structure
and the recorded learning path. It is NOT a persona, NOT a stored answer and NOT
a hand-written capability list: every value is read from the loaded objects, so
the text changes whenever the structure or the learning path changes. The role
conversation may commit it as one ordinary observation, which is why an identity
question is answered from an input that actually exists.
"""
from self_description import self_description


def facts(candidate):
    return self_description(candidate)['facts']


def describe(candidate):
    app=getattr(candidate,'runtime_app',None)
    if app is not None:
        import json
        data=self_description(candidate)['facts']
        memory=app.native.live_memory
        return json.dumps(dict(identity={'name':'Treyrin','chinese_name':'特丽琳','kind':type(candidate.model).__name__},
            language=data['language'],brain=data['structure'],
            memory={'active_window':app.native.window,'page_units':memory.store.config.page_units,
                'persistent_experiences':memory.count(),'storage':'可增长的分页经历库与分层神经工作态'},
            learning={'optimizer_steps':candidate.session.optimizer_steps,'individual_update':'冻结输入基准的输出参数更新，影响语言与原生判断'},
            interfaces=['conversation','decide','joint','spark','materials','tasks','tools','learning','save','resume'],
            adopted_evidence=getattr(candidate,'release_evidence',{})),ensure_ascii=False)
    return self_description(candidate)['statement']


def capabilities(candidate):
    return self_description(candidate)['capabilities']


def observe(candidate, *, known_at=None):
    from self_description import structure_record
    return structure_record(candidate, known_at=known_at)
