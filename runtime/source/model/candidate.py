"""One local Treyrin candidate: Qwen3.5 language/core/learning/resident state.

The frozen language asset is packaged locally; no external decision or retrieval
service is called. Task competence and lifelong-memory acceptance remain tests.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import os
import uuid
from datetime import datetime,timezone
import torch
from transformers import AutoConfig,AutoTokenizer,Qwen3_5ForCausalLM,GenerationConfig,BitsAndBytesConfig
from peft import LoraConfig,get_peft_model
from safetensors.torch import save_file,load_file
from treyrin.structural_core import CoreConfig,ExperienceState
from treyrin.structural_model import HFBodyPort,TreyrinStructuralModel
from native_port import NativeBodyPort as HFBodyPort
from treyrin.structural_learning import (StructuralLearningSession,PublicObservation,
                                         InternalConsolidation,JointExample,ComputationMode)
from treyrin.retention_reference import CachedRetentionReference
from treyrin.learning_checkpoint import load_learning_checkpoint
from treyrin.cognitive_workspace import MentalRecord,CognitiveTick,ACTIONS

ROOT=Path(__file__).resolve().parent
SOURCE=Path(r'D:\MAT\Tryline\models\qwen3.5-4b')
SCHEMA='treyrin-formal-v1'
PROBES=('请简洁回答：一加一等于几？','阅读记录时，应当区分已知事实和待验证的假设。')

CONFIG=dict(width=32,episode_slots=8,semantic_slots=4,latent_candidates=4,cognitive_clock='event',language_prior_gain=1.,perception_memory=True,response_cognitive_stride=4,
    judgment_mode='separated',coupling_mode='layer_interleaved',body_rounds=1,
    layer_sites=(7,15,23,31),regional_circuit=True,hypothesis_mode='adaptive',
    hippocampal_circuit=True,thalamic_gate=True,cortical_streams=True,
    dopamine_plasticity=True,cerebellum=True,neuromodulation=True,amygdala=True,
    drives=True,competitive_retrieval=True,deep_native_field=True,native_body_bridge=True,
    dopamine_mb_target=True,dual_condition_write=True,bg_thalamic_loop=True,
    valence_dopamine=True,arousal_gating=True,ema_anchor_consolidation=True,
    body_drift_coupling=True,hippocampal_bg_loop=True,dopamine_write_gate=True,
    cerebellar_thalamic_gain=True,adaptive_retrieval_k=True,archive_slots=8192,
    echo_slots=127,evict_consolidate=True,per_episode_gate=True,
    goal_gated_retrieval=True,abstain_read=True,episodic_policy=True,cognitive_workspace=True,workspace_slots=8,mental_archive_slots=16384,
    preserve_free_capacity=True,token_code_attention=True,
    native_single_pass=True,goal_joint_control=True,goal_joint_hidden=16,contextual_goal_control=True,code_attention_hops=2,shared_choice_readout=True,cortical_goal_control=True,cortical_control_calibration=True)

def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def code_hashes():
    return {p.relative_to(ROOT).as_posix():sha(p) for p in
            [Path(__file__),ROOT/'episode_codes.py',ROOT/'replay_session.py',ROOT/'native_stream.py',ROOT/'native_port.py',ROOT/'native_decode.py',ROOT/'native_session.py',ROOT/'message_stream.py',ROOT/'decode_runtime.py',ROOT/'resident_store.py',ROOT/'treyrin_cli.py',
             ROOT/'mental_codes.py',ROOT/'spark_lifecycle.py',ROOT/'intent_lifecycle.py',ROOT/'public_labels.py',ROOT/'native_cognition.py',ROOT/'joint_context.py',
             ROOT/'role_conversation.py',ROOT/'generation_result.py',ROOT/'release_adapter.py',
             ROOT/'self_description.py',ROOT/'runtime_self_info.py',
             ROOT/'preassembly_api.py',ROOT/'interaction_state.py',ROOT/'verify_revise.py',
             ROOT/'live_memory.py',ROOT/'individual_learning.py',ROOT/'fixed_output_transaction.py',ROOT/'treyrin_app.py',
             *sorted((ROOT/'treyrin').glob('*.py')),
             *sorted((ROOT/'tools').glob('*.py')),
             *sorted((ROOT/'tools').glob('*/*.py'))]}

def prepare_assets():
    target=ROOT/'assets'
    if (target/'asset-manifest.json').exists():
        return json.loads((target/'asset-manifest.json').read_text('utf-8'))
    (target/'body').mkdir(parents=True,exist_ok=False)
    (target/'tokenizer').mkdir()
    for name in ('config.json','model.safetensors.index.json',
                 'model.safetensors-00001-of-00002.safetensors',
                 'model.safetensors-00002-of-00002.safetensors'):
        source=SOURCE/name;destination=target/'body'/name
        if name.endswith('.safetensors'):os.link(source,destination)
        else:shutil.copyfile(source,destination)
    tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True)
    tok.save_pretrained(target/'tokenizer')
    config=AutoConfig.from_pretrained(SOURCE,local_files_only=True).text_config
    gen=GenerationConfig.from_model_config(config)
    gen.pad_token_id=tok.pad_token_id
    gen.save_pretrained(target/'body')
    files={p.relative_to(target).as_posix():sha(p) for p in sorted(target.rglob('*')) if p.is_file()}
    manifest=dict(schema='treyrin-local-body-assets-v1',files=files,
                  upstream='Qwen/Qwen3.5-4B',commit='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
                  weight_storage='local hardlinks; read-only execution; hashes checked on load')
    (target/'asset-manifest.json').write_text(canonical(manifest)+'\n',encoding='utf-8')
    return manifest

class Candidate:
    def __init__(self,*,seed=101,verify_assets=True):
        torch.set_num_threads(2);torch.manual_seed(seed);random.seed(seed)
        self.assets=prepare_assets();asset_root=ROOT/'assets'
        if verify_assets:
            for name,expected in self.assets['files'].items():
                if sha(asset_root/name)!=expected:raise ValueError('Local asset differs: '+name)
        self.tokenizer=AutoTokenizer.from_pretrained(asset_root/'tokenizer',local_files_only=True)
        self.encoding_sha=hashlib.sha256(canonical({n:h for n,h in self.assets['files'].items()
            if n.startswith('tokenizer/')}).encode()).hexdigest()
        config=AutoConfig.from_pretrained(asset_root/'body',local_files_only=True).text_config
        config._attn_implementation='eager'
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
        base,info=Qwen3_5ForCausalLM.from_pretrained(asset_root/'body',config=config,
            generation_config=GenerationConfig.from_pretrained(asset_root/'body',local_files_only=True),
            dtype=torch.bfloat16,quantization_config=quant,device_map={'':'cuda:0'},
            local_files_only=True,output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError('Incomplete local language asset: '+str(info))
        base.config._name_or_path='treyrin-local-qwen3.5-4b@'+self.assets['commit']
        base.eval().requires_grad_(False)
        asset_id=hashlib.sha256(canonical(self.assets).encode()).hexdigest()
        from public_labels import ALPHABET
        labels=[self.tokenizer(label,add_special_tokens=False)['input_ids']for label in ALPHABET]
        if any(len(x)!=1 for x in labels):raise ValueError('One-token public choice labels required by this body adapter')
        label_ids=[x[0]for x in labels]
        port=HFBodyPort(base,asset_id,choice_label_token_ids=label_ids).eval()
        probe_ids=[torch.tensor([self.encode(t)],device='cuda') for t in PROBES]
        self.reference=CachedRetentionReference(port,[(ids,torch.ones_like(ids)) for ids in probe_ids])
        adapted=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,
            target_modules=['out_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM')).eval()
        self.model=TreyrinStructuralModel(HFBodyPort(adapted,asset_id,checkpoint_layers=True,choice_label_token_ids=label_ids,output_adapter_rank=2),
                                         CoreConfig(**CONFIG),freeze_body=None).eval()
        self.model.body.runtime_identity.update(native_dialogue_protocol='role-correct-single-user-v1',
            native_dialogue_source_sha256=sha(ROOT/'role_conversation.py'),
            generation_transport_source_sha256=sha(ROOT/'generation_result.py'))
        self.session=self.new_session()
        self.optimizer=self.make_optimizer()
        self.conversation=[]
        self.lineage=[]
        self.decode_cache=False
        self.decode_cache_validation=None

    def make_optimizer(self,group_names=None):
        if group_names is not None:
            parameters={n:p for n,p in self.model.named_parameters() if p.requires_grad}
            flat=[n for group in group_names for n in group]
            if len(set(flat))!=len(flat) or set(flat)!=set(parameters):raise ValueError('Saved optimizer groups do not cover exact named parameters')
            return torch.optim.AdamW([{'params':[parameters[n]for n in group]}for group in group_names],weight_decay=.01)
        body=[p for n,p in self.model.named_parameters() if p.requires_grad and n.startswith('body.')]
        core=[p for n,p in self.model.named_parameters() if p.requires_grad and not n.startswith('body.')]
        return torch.optim.AdamW([{'params':body,'lr':1e-5},{'params':core,'lr':1e-4}],weight_decay=.01)

    def warm_start_from_release(self,directory,*,probe_weights=None):
        """Explicit weight migration; no old neural state is relabeled or adopted.

        Frozen assets and every old payload are checked. Only the named new
        compatibility branch may be missing. The optimizer is deliberately new.
        """
        directory=Path(directory)
        manifest=json.loads((directory/'model'/'candidate-manifest.json').read_text('utf-8'))
        if manifest['schema']!='treyrin-integrated-candidate-v1':raise ValueError('Expected frozen v1 release')
        if manifest['assets_sha256']!=sha(ROOT/'assets'/'asset-manifest.json'):
            raise ValueError('Warm-start frozen assets differ')
        for name,expected in manifest['code'].items():
            if sha(directory/name)!=expected:raise ValueError('Frozen release source differs')
        for name,expected in manifest['files'].items():
            path=(directory/'model'/name).resolve()
            if not path.is_relative_to((directory/'model').resolve()) or sha(path)!=expected:
                raise ValueError('Warm-start payload differs')
        old=load_file(directory/'model'/'session'/'learning'/'trainable.safetensors',device='cpu')
        current={n:p for n,p in self.model.named_parameters() if p.requires_grad}
        added=set(current)-set(old)
        if set(old)-set(current) or not added or any(not n.startswith('core.native_field.compatibility.') for n in added):
            raise ValueError('Unreviewed warm-start parameter difference')
        with torch.no_grad():
            for name,value in old.items():
                if current[name].shape!=value.shape or current[name].dtype!=value.dtype:
                    raise ValueError('Warm-start tensor shape/dtype differs: '+name)
                current[name].copy_(value)
        if probe_weights is not None:
            states=torch.load(probe_weights,map_location='cpu',weights_only=True)
            self.model.core.native_field.compatibility.load_state_dict(states['body_interaction']['extra'],strict=True)
        self.model.seal_parameters()
        self.session=self.new_session();self.optimizer=self.make_optimizer();self.conversation=[]
        self.lineage=[dict(source_release=str(directory),source_manifest_sha256=sha(directory/'model'/'candidate-manifest.json'),
            inherited_optimizer_steps=manifest['optimizer_steps'],optimizer_reset=True,neural_state_adopted=False,
            new_parameters=sorted(added),probe_warm_start=probe_weights is not None)]
        return self.lineage[-1]

    def encode(self,text):
        ids=self.tokenizer(text,add_special_tokens=False)['input_ids']
        if not ids:raise ValueError('Nonempty encoded text required')
        return ids

    def warm_start_v2(self):
        """Explicit weight migration; no v2 neural state is relabeled."""
        origin=json.loads((ROOT/'ORIGIN.json').read_text('utf-8'))
        directory=Path(origin['source_release']);source=directory/'model-ready'
        manifest=json.loads((source/'candidate-manifest.json').read_text('utf-8'))
        if manifest['schema']!='treyrin-integrated-candidate-v2':raise ValueError('Expected verified v2 source')
        if manifest['assets_sha256']!=sha(ROOT/'assets'/'asset-manifest.json'):raise ValueError('Migration assets differ')
        for name,expected in manifest['code'].items():
            if sha(directory/name)!=expected:raise ValueError('Frozen v2 source differs')
        for name,expected in manifest['files'].items():
            path=(source/name).resolve()
            if not path.is_relative_to(source.resolve()) or sha(path)!=expected:raise ValueError('Frozen v2 payload differs')
        current_config=asdict(self.model.config)
        if any(canonical(current_config[k])!=canonical(v) for k,v in manifest['core_config'].items()):raise ValueError('Unreviewed inherited configuration change')
        old=load_file(source/'session'/'learning'/'trainable.safetensors',device='cpu')
        current={n:p for n,p in self.model.named_parameters() if p.requires_grad}
        added=set(current)-set(old)
        if set(old)-set(current) or not added or any(not n.startswith(('core.episodic_policy.','core.workspace.')) for n in added):
            raise ValueError('Unexpected migration parameter set')
        with torch.no_grad():
            for name,value in old.items():
                if current[name].shape!=value.shape or current[name].dtype!=value.dtype:raise ValueError('Migration shape/dtype differs')
                current[name].copy_(value)
        self.model.seal_parameters();self.session=self.new_session();self.optimizer=self.make_optimizer()
        self.session.optimizer_steps=manifest['optimizer_steps'];self.conversation=[]
        self.lineage=manifest.get('training_lineage',[])+[dict(source_release=str(directory),
            source_manifest_sha256=sha(source/'candidate-manifest.json'),weights_exactly_preserved=len(old),
            new_parameters=sorted(added),optimizer_reset=True,old_neural_state_adopted=False)]
        return self.lineage[-1]

    def new_session(self):
        return StructuralLearningSession(self.model,self.encode,encoding_sha256=self.encoding_sha,
            retention_reference=self.reference,retention_texts=PROBES,
            auto_trim={'trigger':8},mode=ComputationMode(amygdala_mode='normal'))

    def begin_training_episode(self):
        # Explicit episodic curriculum reset; never called by resident chat.
        s=self.session;s._check()
        s.observations=();s.consolidations=();s.operations=();s._last_trim_obs=0
        s.mental_records=();s.cognitive_ticks=();s.tick_actions=();s.tick_goals=()
        s._set_state(self.model.initial_state(1).detached())

    def observe(self,record):
        return self.session.observe(PublicObservation(**record))

    def observe_text(self,text,*,event_id,source,known_at,on_chunk=None):
        from message_stream import text_chunks,source_identity
        # Unlimited input is explicitly segmented; this does not claim that
        # every fact survives the model's finite neural compression.
        encode=lambda s:self.tokenizer(s,add_special_tokens=False)['input_ids']
        count=tokens=0
        for part in text_chunks(text,encode):
            self.observe(dict(event_id=f'{event_id}:part:{count:08d}',text=part['text'],source=source,known_at=known_at,
                scope=canonical(dict(message_id=event_id,start=part['start'],end=part['end']))))
            count+=1;tokens+=part['tokens']
            if on_chunk:on_chunk(dict(chunks=count,source_tokens=tokens,source_characters=part['end']))
        return dict(message_id=event_id,source_sha256=source_identity(text),chunks=count,ingested_tokens=tokens,
            original_characters=len(text),usable_memory_accepted=False)

    def remember(self,record):
        return self.session.remember(MentalRecord(**record))

    def tick(self,record):
        return self.session.tick(CognitiveTick(**record))

    @torch.no_grad()
    def propose(self,question,*,count=2,max_new_tokens=None):
        device=next(self.model.core.parameters()).device
        ids=torch.tensor([self.encode(self.prompt(question,viewport=0))],device=device)
        drift=(next(self.model.core.parameters()).new_tensor([self.session.drift_ema])
            if self.model.config.body_drift_coupling and self.session.drift_ema is not None else None)
        proposals,output=self.model.propose_and_judge(ids,self.session.state,count=count,use_decoder_cache=self.decode_cache,
            max_hypothesis_tokens=max_new_tokens,drift_level=drift,**self.session.mode.forward_options())
        records=[]
        for index,tokens in enumerate(proposals[0]):
            tokens=tokens[:self.model.last_hypothesis_lengths[index]]
            text=self.tokenizer.decode(tokens,skip_special_tokens=True).strip()
            if not text:
                records.append(dict(retained=False,reason='empty decoded model proposal'));continue
            rid='hypothesis-'+uuid.uuid4().hex
            self.remember(dict(record_id=rid,text=text,kind='hypothesis',
                known_at=datetime.now(timezone.utc).isoformat(),source='model'))
            records.append(dict(record_id=rid,text=text,retained=True,origin='model_hypothesis_not_fact'))
        descriptions=[r['text']for r in records if r.get('retained')]
        selected=self.judge(question,descriptions,viewport=0)['probabilities']if len(descriptions)>=2 else output.native_logits[0].float().softmax(-1).cpu().tolist()
        return dict(proposals=records,choice_probabilities=selected,selection_uses_generated_public_descriptions=len(descriptions)>=2,
            calibrated=False,novelty_or_usefulness_validated=False)

    def prompt(self,question,candidates=(),*,viewport=1):
        if type(viewport) is not int or viewport<0:raise ValueError('Explicit nonnegative public viewport required')
        recent=self.session.observations[-viewport:] if viewport else ()
        text=('最近记录：\n'+'\n'.join(o.text for o in recent)+'\n' if recent else '')+question
        from public_labels import public_label
        if candidates:text+='\n可选项：\n'+'\n'.join(f'{public_label(i)}. {c}'for i,c in enumerate(candidates))+'\n请先输出选项字母，再简短说明理由。'
        return self.tokenizer.apply_chat_template([
            {'role':'system','content':'你是特丽琳。依据记录回答，区分证据与推测，简洁说明。'},
            {'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)

    def example(self,row):
        private=row['private'];public=row['public']
        keys=('native_kind','predicate_targets','utility_targets','control_target','initiate_target','cognitive_control_target','supporting_events','disposable_events')
        extra={k:private[k] for k in keys if k in private}
        for k in ('predicate_targets','utility_targets','control_target','cognitive_control_target','supporting_events','disposable_events'):
            if k in extra:extra[k]=tuple(extra[k])
        return JointExample(self.prompt(public['question'],public['candidates'],viewport=public.get('viewport',1)),private['answer'],
            tuple(public['candidates']),tuple(private['native_target']),choice_labels=bool(public['candidates']),**extra)

    def train(self,row):
        self.begin_training_episode()
        self.apply_public(row['public'])
        return self.session.train_step(self.example(row),self.optimizer,native_weight=3.,
            language_weight=.3,retention_weight=1.,typed_weight=.3)

    def apply_public(self,public):
        if 'stream' in public:
            if any(public.get(k) for k in ('observations','mental_records','cognitive_ticks')):raise ValueError('Use one chronological public stream, not duplicate grouped inputs')
            for operation in public['stream']:
                kind=operation['kind'];record=operation['record']
                if kind=='observation':self.observe(record)
                elif kind=='mental':self.remember(record)
                elif kind=='tick':self.tick(record)
                elif kind=='consolidation':self.session.consolidate(InternalConsolidation(**record))
                else:raise ValueError('Unknown public operation type')
        else:
            if public.get('mental_records') or public.get('cognitive_ticks'):raise ValueError('Mental/time operations require an ordered public stream')
            for observation in public['observations']:self.observe(observation)

    @torch.no_grad()
    def judge(self,question,candidates,*,goal='',viewport=1):
        output=self.session.query(self.prompt(question,candidates,viewport=viewport),tuple(candidates),goal=goal,body_choice_labels=bool(candidates))
        return dict(probabilities=output.native_logits.float().softmax(-1)[0].cpu().tolist(),
            predicate_probabilities=output.predicate_logits.float().sigmoid()[0].cpu().tolist(),
            utility=output.action_values[0].float().cpu().tolist(),
            control_logits=output.control_logits[0].float().cpu().tolist(),
            cognitive_action_probabilities=(None if output.cognitive_controls is None else
                output.cognitive_controls[0,-1].float().softmax(-1).cpu().tolist()),
            read_skip_mass=float(output.read_allocation[0,-1,-1]),calibrated=False)

    def inspect_structure(self):
        self.session._check();s=self.session;state=s.state
        statuses={r.record_id:r.status for r in s.mental_records}
        for record in s.mental_records:
            if record.revises:statuses[record.revises]='retracted' if record.status=='retracted' else 'superseded'
        modules={'language_body':self.model.body,'regional_cortex':self.model.core.regional,
            'hippocampus':self.model.core.hippocampus,'native_field':self.model.core.native_field,
            'episodic_policy':self.model.core.episodic_policy,'cognitive_workspace':self.model.core.workspace}
        return dict(schema=SCHEMA,configuration=asdict(self.model.config),
            components={k:type(v).__name__ for k,v in modules.items()},
            trainable_parameters=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            public_observations=len(s.observations),hot_admissions=int(state.episode_admissions[0]),
            hot_event_ids=state.episode_event_ids[0].cpu().tolist(),archive_event_ids=state.archive_event_ids[0].cpu().tolist(),
            mental_records=[dict(record_id=r.record_id,kind=r.kind,status=statuses[r.record_id],created_status=r.status,
                revises=r.revises,evidence_ids=r.evidence_ids) for r in s.mental_records],
            clocks=dict(observation_events=int(state.writes[0]),mental_records=int(state.workspace_seen[0]),
                cognitive_opportunities=int(state.cognitive_ticks[0]),actual_deliberations=int(state.deliberations[0]),
                consolidations=int(state.consolidations[0]),optimizer_calls=s.optimizer_steps,
                elapsed_seconds=float(state.elapsed_seconds[0])),
            cognitive_policy=self.model.drive_report(state),parameter_binding=state.binding,
            capability_accepted=False,all_user_structural_requirements_satisfied=False,
            outstanding=['1M usable memory and compression','persistent cross-source revision dependencies',
                'bounded state transport after slow learning','general ability retention and useful autonomous behavior'])

    def self_description(self):
        """Honest structural self-report; computed from the live objects, never a canned answer."""
        from self_description import self_description
        return self_description(self)

    @torch.no_grad()
    def reply(self,question,*,candidates=(),max_new_tokens=None,viewport=1):
        prompt=self.prompt(question,candidates,viewport=viewport)
        device=next(self.model.core.parameters()).device
        ids=torch.tensor([self.encode(prompt)],device=device)
        kwargs={}
        if candidates:
            encoded=[torch.tensor(self.encode(x),device=device) for x in candidates]
            padded=torch.nn.utils.rnn.pad_sequence(encoded,batch_first=True,padding_value=0)
            mask=torch.arange(padded.shape[1],device=device)[None]<torch.tensor([len(x) for x in encoded],device=device)[:,None]
            kwargs=dict(candidate_ids=padded[None],candidate_mask=mask.long()[None])
        drift=(next(self.model.core.parameters()).new_tensor([self.session.drift_ema])
               if self.model.config.body_drift_coupling and self.session.drift_ema is not None else None)
        answer=self.model.generate(ids,self.session.state,max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,drift_level=drift,
            use_decoder_cache=getattr(self,'decode_cache',False),
            body_choice_labels=bool(candidates),
            **kwargs,**self.session.mode.forward_options())
        text=self.tokenizer.decode(answer[0,ids.shape[1]:],skip_special_tokens=True)
        self.conversation.append(dict(question=question,answer=text,origin='model_output_not_fact'))
        return text

    def save(self,directory,*,verification=None):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        self.session.save(directory/'session',self.optimizer)
        save_file({k:v.detach().cpu().clone().contiguous() for k,v in self.session.state.tensors().items()},
                  directory/'resident.safetensors')
        (directory/'conversation.json').write_text(canonical(self.conversation)+'\n',encoding='utf-8')
        if verification is not None:
            (directory/'verification.json').write_text(canonical(verification)+'\n',encoding='utf-8')
        declaration=dict(schema=SCHEMA,core_config=asdict(self.model.config),code=code_hashes(),
            assets_sha256=sha(ROOT/'assets'/'asset-manifest.json'),encoding_sha256=self.encoding_sha,
            parameter_binding=self.session.state.binding,contract_identity=self.session.contract_identity,
            optimizer_steps=self.session.optimizer_steps,capability_accepted=False)
        declaration['training_lineage']=self.lineage
        declaration['runtime_settings']=dict(decode_cache=self.decode_cache,cache_validation=self.decode_cache_validation)
        declaration['files']={p.relative_to(directory).as_posix():sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}
        (directory/'candidate-manifest.json').write_text(canonical(declaration)+'\n',encoding='utf-8')
        return declaration

    def restore(self,directory):
        directory=Path(directory)
        manifest=json.loads((directory/'candidate-manifest.json').read_text('utf-8'))
        if (manifest['schema']!=SCHEMA or manifest['code']!=code_hashes() or
            manifest['assets_sha256']!=sha(ROOT/'assets'/'asset-manifest.json') or
            manifest['encoding_sha256']!=self.encoding_sha or
            manifest['contract_identity']!=self.session.contract_identity):
            raise ValueError('Candidate source/body/tokenizer/lifecycle identity differs')
        for name,expected in manifest['files'].items():
            path=(directory/name).resolve()
            if not path.is_relative_to(directory.resolve()) or sha(path)!=expected:
                raise ValueError('Candidate payload hash differs: '+name)
        s=self.session
        if s.operations or s.optimizer_steps:raise ValueError('Restore requires a fresh resident session')
        optimizer_manifest=json.loads((directory/'session'/'learning'/'manifest.json').read_text('utf-8'))
        self.optimizer=self.make_optimizer(group_names=optimizer_manifest['optimizer_groups'])
        steps=load_learning_checkpoint(directory/'session'/'learning',self.model,self.optimizer,binding=s.contract_identity)
        if steps!=manifest['optimizer_steps'] or self.model.seal_parameters()!=manifest['parameter_binding']:
            raise ValueError('Restored candidate parameter/step differs')
        journal=json.loads((directory/'session'/'journal.json').read_text('utf-8'))
        control=json.loads((directory/'session'/'control-state.json').read_text('utf-8'))
        if control['auto_trim']!=s.auto_trim:raise ValueError('Saved resident policy differs')
        s.observations,s.consolidations,s.mental_records,s.cognitive_ticks,s.tick_actions,s.tick_goals,s.operations=s._parse_journal(journal)
        tensors=load_file(directory/'resident.safetensors',device='cuda')
        state=ExperienceState(**tensors,binding=manifest['parameter_binding'])
        self.model._check(state,1)
        if int(state.writes[0])!=len(s.observations) or int(state.consolidations[0])!=len(s.consolidations):
            raise ValueError('Resident counters disagree with journal')
        if self.model.config.cognitive_workspace:
            if int(state.workspace_seen[0])!=len(s.mental_records) or int(state.cognitive_ticks[0])!=len(s.cognitive_ticks):raise ValueError('Cognitive journal counters disagree')
            if int(state.deliberations[0])!=sum(a=='think' for a in s.tick_actions):raise ValueError('Historical action trace disagrees')
        s.optimizer_steps=steps;s._last_trim_obs=control['last_trim_obs'];s.drift_ema=control['drift_ema']
        s._set_state(state);s._check()
        self.conversation=json.loads((directory/'conversation.json').read_text('utf-8'))
        self.lineage=manifest.get('training_lineage',[])
        runtime=manifest.get('runtime_settings',{})
        self.decode_cache=runtime.get('decode_cache',False)
        self.decode_cache_validation=runtime.get('cache_validation')
        if self.decode_cache and not (isinstance(self.decode_cache_validation,dict) and self.decode_cache_validation.get('passed') is True):
            raise ValueError('Cached runtime needs an explicit successful real-body validation')
        if self.decode_cache and self.decode_cache_validation.get('parameter_binding')!=self.session.state.binding:
            raise ValueError('Cached validation belongs to different trained parameters')
        return dict(restored=True,optimizer_steps=steps,observations=len(s.observations),
                    neural_state_restored_directly=True,history_replayed=False)
