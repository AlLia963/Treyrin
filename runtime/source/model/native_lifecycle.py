"""Model-owned lifecycle operators embedded in the llama process.

The callback executes the already loaded CUDA decoder. No second language
model, HTTP client, subprocess, prompt file, or external decision service is
used here. Original admission, revision, intent and joint-control functions
operate on the same resident tensors as the native layer operators.
"""
from __future__ import annotations
import ctypes,json,struct,hashlib,math,copy,uuid,os
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
from dataclasses import asdict
import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from safetensors.torch import load,save
from treyrin.structural_core import CoreConfig,StructuralCore,ExperienceState
from treyrin.structural_model import TreyrinStructuralModel
from treyrin.structural_learning import StructuralLearningSession,ComputationMode
from treyrin.cognitive_workspace import MentalRecord,CognitiveTick,ACTIONS
from treyrin.cortical_control import CorticalOpportunityRead,effective_opportunity
from spark_lifecycle import SparkLifecycle
from intent_lifecycle import IntentLifecycle
from public_labels import ALPHABET

def unpack(payload):
    n=struct.unpack_from('<Q',payload)[0]
    header=json.loads(payload[8:8+n])
    metadata=json.loads(header.get('__metadata__',{}).get('treyrin_exchange','{}'))
    return metadata,load(payload)

def pack(tensors,metadata):
    return save({n:t.detach().cpu().contiguous() for n,t in tensors.items()},
        metadata={'treyrin_exchange':json.dumps(metadata,ensure_ascii=False,allow_nan=False)})

def now():return datetime.now(timezone.utc).isoformat()

class Graph(nn.Module):
    _tokens=TreyrinStructuralModel._tokens
    remember_mental=TreyrinStructuralModel.remember_mental
    goal_action_inputs=TreyrinStructuralModel.goal_action_inputs
    cognitive_tick=TreyrinStructuralModel.cognitive_tick
    def __init__(self,engine):
        super().__init__();self.engine=engine
        from modeling_treyrin_runtime import MODEL_CONFIG
        configuration=MODEL_CONFIG
        self.config=CoreConfig(**configuration['core_config'])
        self.body=SimpleNamespace(vocab_size=configuration['text_config']['vocab_size'])
        self.core=StructuralCore(self.config).eval().requires_grad_(False)
        _,weights=self.engine.decoder('parameters')
        self.core.load_state_dict(weights,strict=True)
        self._binding=engine.binding
    def _check(self,state,batch):
        if state.binding!=self._binding:raise ValueError('Resident model binding differs')
        self.core.validate_state(state,batch)
    def encode_descriptions(self,ids,mask):
        b,k,t=ids.shape;rows=[]
        for row,valid in zip(ids.reshape(-1,t),mask.reshape(-1,t)):
            values=row[valid.bool()].tolist()
            _,result=self.engine.decoder('encode',token_ids=values)
            rows.append(result['encoded'].reshape(32))
        return torch.stack(rows).reshape(b,k,32)

class Session:
    _mental_state=StructuralLearningSession._mental_state
    remember=StructuralLearningSession.remember
    tick=StructuralLearningSession.tick
    def _check(self):self.model._check(self.state,1)
    def _ids(self,text):return torch.tensor([self.engine.tokens(text)],dtype=torch.long)
    def _set_state(self,state):self.state=state
    def _drift_level(self):return torch.zeros(1)

class Tokenizer:
    def __init__(self,engine):self.engine=engine
    def decode(self,ids,**kwargs):return self.engine.decoder('decode',token_ids=list(ids))[0]['text']

class Engine:
    def __init__(self,binding,callback,owner):
        torch.set_num_threads(2)
        self.binding=binding
        callback_type=ctypes.CFUNCTYPE(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_size_t))
        self.callback=callback_type(callback);self.owner=owner
        self.model=Graph(self)
        self.candidate=SimpleNamespace(model=self.model,encode=self.tokens,tokenizer=Tokenizer(self))
        self.model._binding=binding
        self.pending_user=None
        self.capsule=SimpleNamespace(binding=binding,owner=owner,kind='native_decoder_cache')
        self.codes=SimpleNamespace(binding=binding,owner=owner,kind='native_paged_episode_codes')
    def tokens(self,text):return self.decoder('tokenize',text=text)[0]['token_ids']
    def decoder(self,mode,**request):
        values={}
        if hasattr(self,'session'):
            values={'state_'+n:t for n,t in self.session.state.tensors().items()}
        encoded=pack(values,dict(mode=mode,**request))
        buffer=ctypes.create_string_buffer(encoded);size=ctypes.c_size_t()
        pointer=self.callback(self.owner,buffer,len(encoded),ctypes.byref(size))
        if not pointer:raise RuntimeError('Native decoder returned no result')
        metadata,tensors=unpack(ctypes.string_at(pointer,size.value))
        if metadata.get('error'):raise RuntimeError(metadata['error'])
        return metadata,tensors
    def _query_episode(self,question,choices=(),**options):
        metadata,result=self.decoder('query',question=question,candidates=list(choices),generate=False)
        output=SimpleNamespace(**result)
        episode=SimpleNamespace(model=self.model,state=self.session.state,output=output)
        return episode,dict(replayed_record_addresses=metadata.get('replayed_addresses',[]))
    def query(self,question,candidates=(),max_new_tokens=None):
        metadata,result=self.decoder('query',question=question,candidates=list(candidates),generate=not bool(candidates),
            max_new_tokens=max_new_tokens)
        logits=result['native_logits'].reshape(-1).float()
        probabilities=logits.softmax(-1).tolist()
        return dict(answer=metadata.get('answer',''),native_probabilities=probabilities,
            native_choice_index=int(logits.argmax()),generation=metadata)
    def remember(self,payload):
        record=MentalRecord(**payload)
        before=self.session.state
        branch=copy.copy(self.session)
        result=branch.remember(record)
        if not torch.equal(before.writes,branch.state.writes):raise RuntimeError('Non-fact record altered observed-event count')
        self.session=branch;self.candidate.session=branch
        if result['status']!='exact_mental_replay_ignored':
            ordinal=int(self.session.state.workspace_seen[0]);ids=self.tokens(record.text)
            self.mental_codes=dict(self.mental_codes);self.mental_codes[str(ordinal)]=ids
            self.new_mental.append(dict(ordinal=ordinal,**asdict(record),token_ids=ids))
        return result
    def _known_at(self,command):return command.get('known_at') or now()
    def tick(self,command):
        fields={k:v for k,v in command.items() if k in CognitiveTick.__dataclass_fields__}
        fields.setdefault('operation_id','opportunity-'+uuid.uuid4().hex)
        fields.setdefault('known_at',now());fields.setdefault('elapsed_seconds',0.)
        tick=CognitiveTick(**fields)
        if tick in self.session.cognitive_ticks:return self.session.tick(tick)
        if any(t.operation_id==tick.operation_id for t in self.session.cognitive_ticks):raise ValueError('Cognitive opportunity identity conflicts')
        read=None
        if tick.resource>0:
            s=self.session.state;effective=effective_opportunity(s,tick)
            active=((s.goal_ids[0]>0)&(s.goal_statuses[0]==0)).nonzero().flatten().tolist()
            goals=[];cells=[];choices=[]
            descriptions=('等待：此刻暂不值得推进或打扰','内部思考：分析、比较、校核已有记录','发言：提交已有充分依据的结果',
                '补充证据：取得缺少的观测或请求必要信息','执行：实施已经授权且前置条件满足的操作；选择本身不是成功回执')
            for slot in active:
                ordinal=int(s.goal_ids[0,slot]);ids=self.mental_codes.get(str(ordinal))
                if not ids:raise ValueError('Protected goal lacks its actual model-owned token code')
                goals.append(dict(physical_goal_slot=slot,mental_ordinal=ordinal,actual_model_code=self.candidate.tokenizer.decode(ids),
                    acceptance=int(s.goal_accept[0,slot]),can_act=bool(s.goal_can_act[0,slot]),priority=float(s.goal_priority[0,slot]),
                    interruption_cost=float(s.goal_icost[0,slot])))
            for slot in [*active,self.model.config.workspace_slots]:
                for action,text in enumerate(descriptions):
                    cells.append((slot,action));choices.append(('无目标' if slot==self.model.config.workspace_slots else '目标槽'+str(slot))+'；'+text)
            public=dict(now=tick.known_at,elapsed_seconds=tick.elapsed_seconds,interruption_cost=tick.interruption_cost,
                resource=tick.resource,deadline_remaining=effective.deadline_remaining,clear_deadline=tick.clear_deadline,goals=goals)
            question='这是一次真实认知资源分配机会。根据当前记忆和公开时间、成本、权限及目标码，选择一个目标和动作。不能虚构观测或完成回执。\n'+json.dumps(public,ensure_ascii=False)
            episode,selected=self._query_episode(question,choices)
            read=CorticalOpportunityRead.from_episode(self.model,s,episode,cells,selected.get('replayed_record_addresses',()),tick=tick)
        result=self.session.tick(tick,cortical_read=read)
        result.update(operation_id=tick.operation_id,execution_completed=False)
        self.lifecycle['last_tick_at']=tick.known_at
        self.lifecycle.setdefault('action_queue',[]).append(result)
        return result
    def decide(self,command):
        choices=command.get('candidates')
        if choices is None:
            question=command['question'];meta,latent=self.decoder('query',question=question,candidates=[],generate=False)
            probs=latent['native_logits'].reshape(-1).float().softmax(-1)
            return dict(kind='internal_hypothesis_population',probabilities=probs.tolist(),selected_index=int(probs.argmax()),
                hypothesis_vectors=latent['hypothesis_vectors'].tolist(),calibrated=False,generated_text_parsed=False)
        if not isinstance(choices,list) or len(choices)<2 or any(not isinstance(v,str) or not v.strip() for v in choices):
            raise ValueError('Two or more nonempty public candidates required')
        result=self.query(command['question'],tuple(choices))
        index=result['native_choice_index'];p=torch.tensor(result['native_probabilities'])
        return dict(kind='closed_candidate_choice',space=choices,probabilities=p.tolist(),selected_index=index,
            selected_label=ALPHABET[index] if len(choices)<=len(ALPHABET) else str(index+1),selected_text=choices[index],normalized_entropy=float(-(p*p.clamp_min(1e-12).log()).sum()/math.log(len(p))),
            calibrated=False,generated_text_parsed=False)
    def learn(self,command,values):
        question=command.get('question');preferred=command.get('preferred');steps=command.get('steps',1)
        if not isinstance(question,str) or not question.strip() or not isinstance(preferred,str) or not preferred.strip():
            raise ValueError('Explicit question and preferred answer required')
        if type(steps)is not int or steps<1:raise ValueError('A positive explicit learning step count is required')
        names=('body.output_adapter.A','body.output_adapter.B')
        local={n:nn.Parameter(values[n].clone()) for n in names}
        rate=float(command.get('learning_rate',1e-5))
        if not math.isfinite(rate) or rate<=0:raise ValueError('Finite positive feedback learning rate required')
        optimizer=torch.optim.AdamW(list(local.values()),lr=rate,weight_decay=.01)
        for n,p in local.items():
            if 'learn_step_'+n in values:
                optimizer.state[p]={key:values['learn_'+key+'_'+n].clone() for key in ('step','exp_avg','exp_avg_sq')}
        rank=local[names[0]].shape[0]
        def delta(h):return F.linear(F.linear(h.float(),local[names[0]]),local[names[1]])/rank
        targets=self.tokens(preferred)
        eos=self.decoder('eos')[0]['token_ids']
        if eos:targets.append(eos[0])
        with torch.no_grad():
            _,data=self.decoder('teacher',question=question,target_ids=targets)
            hidden=data['body_hidden'];current=data['language_logits'];base=current-delta(hidden)
            probes=[]
            for text in ('请简洁回答：一加一等于几？','阅读记录时，应当区分已知事实和待验证的假设。'):
                _,probe=self.decoder('query',question=text,candidates=[],generate=False)
                h=probe['body_hidden'];logits=probe['language_logits'];probes.append((h,logits-delta(h),logits.log_softmax(-1)))
        target=torch.tensor(targets,dtype=torch.long).reshape(1,-1)
        weight=float(command.get('retention_weight',1.));initial=None
        if not math.isfinite(weight) or weight<0:raise ValueError('Finite nonnegative retention weight required')
        before={n:v.detach().clone() for n,v in local.items()}
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            preferred_loss=F.cross_entropy((base+delta(hidden)).reshape(-1,base.shape[-1]),target.reshape(-1))
            preserve=preferred_loss.new_zeros(())
            for h,b,old in probes:
                new=(b+delta(h)).log_softmax(-1)
                preserve=preserve+(old.exp()*(old-new)).sum(-1).mean().clamp_min(0)
            loss=preferred_loss+weight*preserve/max(1,len(probes))
            if not torch.isfinite(loss):raise ValueError('Nonfinite feedback loss; individual unchanged')
            loss.backward();torch.nn.utils.clip_grad_norm_(list(local.values()),1.,error_if_nonfinite=True);optimizer.step()
            if initial is None:initial=float(preferred_loss.detach())
        for n,p in local.items():
            if not torch.isfinite(p).all() or any(not torch.isfinite(v).all() for v in optimizer.state[p].values()):
                raise ValueError('Nonfinite output patch or Adam state; current individual unchanged')
            if (optimizer.state[p]['exp_avg_sq']<0).any():raise ValueError('Negative Adam squared moment')
            values[n]=p.detach()
            for key,value in optimizer.state[p].items():values['learn_'+key+'_'+n]=value.detach()
        self.lifecycle['optimizer_steps']=self.lifecycle.get('optimizer_steps',0)+steps
        self.lifecycle['readout_generation']=self.lifecycle.get('readout_generation',0)+1
        return dict(changed=[n for n in names if not torch.equal(before[n],local[n])],steps=steps,
            optimizer_steps=self.lifecycle['optimizer_steps'],preferred_loss_before=initial,preferred_loss_last=float(preferred_loss.detach()),
            retention_kl=float(preserve.detach()),teacher_in_experience=False,frozen_encoder_unchanged=True,
            learning_scope='individual vocabulary readout and corresponding native choice rows',banks_numeric_identical=True)
    def exchange(self,payload):
        metadata,values=unpack(payload);command=metadata['command'];op=command['operation']
        self.lifecycle=copy.deepcopy(metadata.get('lifecycle',{}));self.mental_codes=self.lifecycle.setdefault('mental_codes',{})
        self.new_mental=[]
        state=ExperienceState(**{n[6:]:v for n,v in values.items() if n.startswith('state_')},binding=self.binding)
        s=Session();s.engine=self;s.model=self.model;s.state=state
        s.mental_records=tuple(MentalRecord(**r) for r in self.lifecycle.get('mental_records',[]))
        s.observations=tuple(SimpleNamespace(event_id=v['id']) for v in metadata.get('observations',[]))
        s.cognitive_ticks=tuple(CognitiveTick(**v) for v in self.lifecycle.get('cognitive_ticks',[]))
        s.tick_actions=tuple(self.lifecycle.get('tick_actions',[]));s.tick_goals=tuple(self.lifecycle.get('tick_goals',[]))
        s.operations=tuple(tuple(v) for v in self.lifecycle.get('operations',[]));s.mode=ComputationMode(amygdala_mode='normal')
        self.session=s;self.candidate.session=s;s._check()
        with torch.no_grad():
            if op=='status':result=self.status()
            elif op=='decide':result=self.decide(command)
            elif op=='spark':result=SparkLifecycle(self).propose(command['problem'],known_at=self._known_at(command),max_new_tokens=command.get('max_new_tokens'))
            elif op=='revise_spark':result=SparkLifecycle(self).revise(command['record_id'],command['question'],known_at=self._known_at(command),max_new_tokens=command.get('max_new_tokens'))
            elif op=='remember':result=self.remember(command['record'])
            elif op=='intents':result=IntentLifecycle(self).intake(command['text'],evidence_id=command['evidence_id'],known_at=self._known_at(command),max_new_tokens=command.get('max_new_tokens'))
            elif op=='tick':result=self.tick(command)
            elif op=='actions':result=dict(actions=self.lifecycle.get('action_queue',[]))
            elif op=='configure':
                changes={k:command[k] for k in ('enabled','interval_seconds','resource','interruption_cost') if k in command}
                if 'interval_seconds' in changes and (not math.isfinite(changes['interval_seconds']) or changes['interval_seconds']<=0):raise ValueError('Positive finite opportunity interval required')
                if 'resource' in changes and not 0<=changes['resource']<=1:raise ValueError('Normalized available resource required')
                self.lifecycle.setdefault('scheduler',{}).update(changes);result=self.status()
            elif op=='learn':result=None
            else:raise ValueError('Unknown model lifecycle operation '+str(op))
        if op=='learn':
            before={n:v.clone() for n,v in s.state.tensors().items()}
            with torch.enable_grad():result=self.learn(command,values)
            if any(not torch.equal(before[n],v) for n,v in s.state.tensors().items()):raise RuntimeError('Output learning altered resident bank values')
        s=self.session
        self.lifecycle.update(mental_codes=self.mental_codes,mental_records=[asdict(r) for r in s.mental_records],cognitive_ticks=[asdict(t) for t in s.cognitive_ticks],
            tick_actions=list(s.tick_actions),tick_goals=list(s.tick_goals),operations=list(s.operations))
        values.update({'state_'+n:t for n,t in s.state.tensors().items()})
        return pack(values,dict(result=result,lifecycle=self.lifecycle,new_mental=self.new_mental))
    def status(self):
        s=self.session.state
        return dict(schema='treyrin-native-lifecycle-v1',same_process=True,language_backend='llama.cpp CUDA',
            mental_records=len(self.session.mental_records),active_goals=int(((s.goal_ids>0)&(s.goal_statuses==0)).sum()),
            cognitive_ticks=int(s.cognitive_ticks[0]),deliberations=int(s.deliberations[0]),
            optimizer_steps=self.lifecycle.get('optimizer_steps',0),readout_generation=self.lifecycle.get('readout_generation',0),
            scheduler=self.lifecycle.get('scheduler',{'enabled':True,'interval_seconds':60.,'resource':1.,'interruption_cost':0.}),
            pending_actions=len(self.lifecycle.get('action_queue',[])))
