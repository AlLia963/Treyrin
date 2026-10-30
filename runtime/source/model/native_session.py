"""Atomic core/public chronology + bounded native body experience.

The native prefix contains observed public/typed records, not private targets.
Queries fork neural state and never append hypothetical answers to experience.
Parameter changes are rejected until an explicit encoder migration is supplied.
"""
import copy
import hashlib
import json
from pathlib import Path
import torch
from native_stream import Capsule,digest
from native_decode import NativeEpisode

def system_prefix(tokenizer,text):
    # The official Qwen template rejects system-only conversations. Render a
    # structural placeholder, then remove the EXACT separately rendered user
    # fragment. The placeholder never reaches model tokens or experience.
    placeholder='TREYRIN_TEMPLATE_BOUNDARY_7d4d8e'
    user=tokenizer.apply_chat_template([dict(role='user',content=placeholder)],tokenize=False,add_generation_prompt=False,enable_thinking=False)
    combined=tokenizer.apply_chat_template([dict(role='system',content=text),dict(role='user',content=placeholder)],
        tokenize=False,add_generation_prompt=False,enable_thinking=False)
    if not user or not combined.endswith(user):raise ValueError('System-prefix template decomposition not supported')
    return combined[:-len(user)]

class NativeSession:
    def __init__(self,candidate,*,window=8192,initialize=True):
        self.candidate=candidate;self.session=candidate.session;self.window=window
        self.capsule=Capsule.empty(candidate.model.body,window=window,binding=candidate.model._binding)
        if initialize:
            if self.session.operations:raise ValueError('Native boot requires an empty source; do not retag existing memory')
            system=system_prefix(candidate.tokenizer,'你是特丽琳。依据记录回答，区分证据与推测，简洁说明。')
            episode=self._episode(system,self.session.state)
            self.capsule=episode.commit_observed_prefix()

    def _episode(self,text,state,choices=(),*,body_feedback=None):
        c=self.candidate;encoded=c.tokenizer(text,add_special_tokens=False)['input_ids']
        if not encoded:raise ValueError('Nonempty native message required')
        ids=torch.tensor([encoded],device=next(c.model.parameters()).device)
        options=self.session.mode.forward_options();options['drift_level']=self.session._drift_level()
        if body_feedback is not None:
            if type(body_feedback) is not bool:raise ValueError('Explicit declared cortical-feedback intervention required')
            options['body_feedback']=body_feedback
        if choices:
            encoded=[torch.tensor(c.encode(x),device=ids.device)for x in choices]
            padded=torch.nn.utils.rnn.pad_sequence(encoded,batch_first=True,padding_value=0)
            mask=torch.arange(padded.shape[1],device=ids.device)[None]<torch.tensor([len(x)for x in encoded],device=ids.device)[:,None]
            options.update(candidate_ids=padded[None],candidate_mask=mask.long()[None],body_choice_labels=True)
        with torch.no_grad():return NativeEpisode(c.model,state,ids,self.capsule,**options)

    def _record(self,kind,record):
        c=self.candidate;old=self.session
        branch=copy.copy(old)
        from treyrin.structural_learning import PublicObservation
        from treyrin.cognitive_workspace import MentalRecord
        # Idempotence applies to the entire neural conversation, not only the
        # symbolic journal. A replayed request cannot advance body time again.
        if kind=='observation'and PublicObservation(**record)in old.observations:
            return dict(status='exact_replay_ignored',journal_sha256=old.journal_sha256)
        if kind=='mental'and MentalRecord(**record)in old.mental_records:
            return dict(status='exact_mental_replay_ignored')
        native_commit=kind=='observation'and c.model.config.native_single_pass
        with torch.no_grad():
            if kind=='observation'and not native_commit:result=branch.observe(PublicObservation(**record))
            elif kind=='mental':result=branch.remember(MentalRecord(**record))
            elif not native_commit:raise ValueError('Only explicit observed/mental records can commit native experience')
        metadata={k:v for k,v in record.items()if k!='text'}
        content='公开记录：\n'+kind+' '+json.dumps(metadata,ensure_ascii=False,sort_keys=True)+' '+record['text']
        text=c.tokenizer.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=False,enable_thinking=False)
        episode=self._episode(text,old.state)
        capsule=episode.commit_observed_prefix()
        if native_commit:
            event=PublicObservation(**record);branch._check()
            # Only the final bounded chunk's populations are returned by the
            # decoder. Event/perception clocks use its actual final position.
            if c.model.config.cognitive_clock!='event'or not c.model.config.perception_memory:
                raise ValueError('Native single-pass requires final-position perception')
            part=episode.ids[:,-episode.output.body_hidden.shape[1]:]
            source=branch._ids(event.provenance_text());echo=branch._ids(event.text)
            with torch.no_grad():
                staged=c.model.commit_observation_output(part,torch.ones_like(part),source,torch.ones_like(source),
                    old.state,episode.output,echo_ids=echo,**branch.mode.forward_options()).detached()
            result=branch._accept_observation(event,staged,conflict=any(o.event_id==event.event_id for o in old.observations))
        # Commit both states only after core validation and native execution.
        self.session=branch;self.capsule=capsule;c.session=branch
        return result

    def observe(self,record):return self._record('observation',record)
    def remember(self,record):return self._record('mental',record)

    def query(self,question,choices=(),*,max_new_tokens=None):
        c=self.candidate;text=question
        from public_labels import public_label
        if choices:text+='\n选项：\n'+'\n'.join(f'{public_label(i)}. {x}'for i,x in enumerate(choices))+'\n请先输出选项字母，再简短说明理由。'
        prompt=c.tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        episode=self._episode(prompt,self.session.state,choices)
        with torch.no_grad():
            tokens=episode.generate(max_new_tokens,eos_token_id=c.tokenizer.eos_token_id)
            native=episode.output.native_logits[0].float().softmax(-1).cpu().tolist()
        return dict(answer=c.tokenizer.decode(tokens,skip_special_tokens=True),answer_token_ids=tokens,
            native_probabilities=native,native_choice_index=max(range(len(native)),key=lambda i:native[i]),
            observed_native_tokens=self.capsule.seen,attention_window=self.window,usable_million_memory_accepted=False)

    def save(self,path):
        root=Path(path);root.mkdir(exist_ok=False)
        self.candidate.session=self.session;self.candidate.save(root/'core');self.capsule.save(root/'native')
        m=dict(schema='treyrin-joint-native-session-v1',native_manifest_sha256=digest(root/'native'/'manifest.json'),
            core_manifest_sha256=digest(root/'core'/'candidate-manifest.json'),window=self.window,
            source_sha256=digest(__file__),native_tokens=self.capsule.seen)
        (root/'manifest.json').write_text(json.dumps(m,sort_keys=True,indent=2),encoding='utf-8')

    @classmethod
    def restore(cls,path,candidate):
        root=Path(path);m=json.loads((root/'manifest.json').read_text('utf-8'))
        if m['schema']!='treyrin-joint-native-session-v1'or m['source_sha256']!=digest(__file__):raise ValueError('Joint native schema/implementation differs')
        if m['native_manifest_sha256']!=digest(root/'native'/'manifest.json')or m['core_manifest_sha256']!=digest(root/'core'/'candidate-manifest.json'):
            raise ValueError('Joint native/core manifests changed')
        candidate.restore(root/'core');result=cls(candidate,window=m['window'],initialize=False)
        result.capsule=Capsule.restore(root/'native',candidate.model.body,binding=candidate.model._binding)
        if result.capsule.seen!=m['native_tokens']:raise ValueError('Native/core chronology capsule count differs')
        return result
