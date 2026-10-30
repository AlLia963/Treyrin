"""Observation-dependent training fixtures; targets never enter model input.

Each pair has identical query, candidate order and provenance. Only observed
content differs. Evidence-free deterministic prediction cannot get both right.
These are synthetic local-trial records, not claims about released models.
"""
import hashlib
import itertools
import json
import random


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def build_course():
    fields={'train':{'num_hidden_layers':(40,64,93),'hidden_size':(4096,5120,7168),
                     'vocab_size':(65536,129280,163840)},
            'development':{'batch_size':(16,32,64),'context_chunks':(3,6,9)}}
    records=[]
    for split,parameters in fields.items():
        styles=('plain','correction','withdrawal') if split=='train' else ('heldout_plain','heldout_correction')
        for key,values in parameters.items():
            for style in styles:
                for pair_index,(a,b) in enumerate(itertools.combinations(values,2)):
                    initial=next(v for v in values if v not in (a,b))
                    candidates=[f'The active value is {v}.' for v in values]+['No active value is reported.']
                    rng=random.Random(f'treyrin-paired-v1:{split}:{key}:{style}:{pair_index}')
                    rng.shuffle(candidates)
                    question=f'According to the current public trial record, what is the active value of {key}?'
                    if style.startswith('heldout'):
                        question=f'Using the records you observed, identify the presently valid {key} setting.'
                    pair_id=f'{split}:{key}:{style}:{pair_index}'
                    for variant,value in enumerate((a,b)):
                        observations=[]
                        if style in ('correction','withdrawal','heldout_correction'):
                            observations.append(dict(event_id='record-1',text=f'Trial record: {key} is set to {initial}.',
                                source='local trial record',known_at='logical:1',occurred_at='',
                                scope='synthetic training record, not a model-release claim'))
                        if style=='withdrawal' and variant==1:
                            text=f'The preceding {key} setting is withdrawn; there is no replacement setting.'
                            supported='No active value is reported.'
                            answer=f' No active value of {key} is reported after the withdrawal.'
                        else:
                            if style=='heldout_correction':
                                text=f'Correction to the earlier entry: use {value} for {key}; the previous setting no longer applies.'
                            elif style in ('correction','withdrawal'):
                                text=f'The author replaces the preceding {key} setting with {value}.'
                            elif style=='heldout_plain':
                                text=f'The current recorded setting for {key} reads {value}.'
                            else:text=f'Trial record: the active {key} setting is {value}.'
                            supported=f'The active value is {value}.'
                            answer=f' The active value of {key} is {value}.'
                        observations.append(dict(event_id='current-record',text=text,source='author trial update',
                            known_at='logical:2',occurred_at='',scope='synthetic training record, not a model-release claim'))
                        target=[float(c==supported) for c in candidates]
                        assert sum(target)==1
                        record=dict(pair_id=pair_id,variant=variant,split=split,observations=observations,
                            example=dict(prompt=question,answer=answer,candidates=candidates,native_target=target,
                                native_kind='choice',goal='',hypothesis_targets=[],hypothesis_rejections=[]))
                        record['semantic_sha256']=_digest(dict(observations=observations,
                            question=question,candidates=sorted(candidates)))
                        records.append(record)
    pairs={}
    for record in records:pairs.setdefault(record['pair_id'],[]).append(record)
    for pair in pairs.values():
        left,right=pair
        assert left['example']['prompt']==right['example']['prompt']
        assert left['example']['candidates']==right['example']['candidates']
        assert left['example']['native_target']!=right['example']['native_target']
        assert len(left['observations'])==len(right['observations'])
        for a,b in zip(left['observations'],right['observations']):
            assert {k:v for k,v in a.items() if k!='text'}=={k:v for k,v in b.items() if k!='text'}
        assert left['observations']!=right['observations']
    train_hash={r['semantic_sha256'] for r in records if r['split']=='train'}
    dev_hash={r['semantic_sha256'] for r in records if r['split']=='development'}
    assert not train_hash&dev_hash
    return records,dict(schema='treyrin-paired-reading-v1',records=len(records),pairs=len(pairs),
        unique_semantic_train_records=len(train_hash),unique_semantic_development_records=len(dev_hash),
        train_records=sum(r['split']=='train' for r in records),
        development_records=sum(r['split']=='development' for r in records),
        evidence_free_deterministic_pair_both_correct_upper_bound=0.,
        scope='synthetic local-trial reading/revision; not a general research or intuition benchmark',
        records_sha256=_digest(records))
