"""Role-correct causal conversation through the unchanged Treyrin brain graph.

Public source records and live user requests are distinct operations. Generated
assistant tokens remain assistant history; their typed memory is nonfactual.
No task classifier, result replacement, journal search or style prompt is used.
"""
from __future__ import annotations
import copy,hashlib,json,sys,time,uuid
from datetime import datetime
from pathlib import Path
import torch
from generation_result import GenerationResultMixin,GenerationInterrupted
from native_decode import NativeEpisode
from native_stream import Capsule
from replay_session import NativeReplaySession
from episode_codes import EpisodeCodes
from mental_codes import MentalCodes
from treyrin.structural_learning import PublicObservation
from treyrin.cognitive_workspace import MentalRecord
from message_stream import text_chunks

def timestamp():return datetime.now().astimezone().isoformat()

class ConversationEpisode(GenerationResultMixin,NativeEpisode):pass

def role_frame(tokenizer,role,text):
    options=dict(tokenize=False,add_generation_prompt=False,enable_thinking=False)
    if role=='user':return tokenizer.apply_chat_template([dict(role='user',content=text)],**options)
    # Qwen3.5 requires a user query even to format a historical assistant/tool
    # segment. Remove an exact separately rendered boundary; it never becomes
    # model input, a record, or an additional instruction.
    boundary='TREYRIN_ROLE_FRAME_6efa7241'
    while boundary in text:boundary+='_x'
    prefix=tokenizer.apply_chat_template([dict(role='user',content=boundary)],**options)
    complete=tokenizer.apply_chat_template([dict(role='user',content=boundary),dict(role=role,content=text)],**options)
    if not complete.startswith(prefix):raise ValueError('Official role segment has an incompatible prefix')
    return complete[len(prefix):]

def generation_prefix(tokenizer):
    placeholder='TREYRIN_ROLE_BOUNDARY_d060f752'
    message=[dict(role='user',content=placeholder)]
    plain=tokenizer.apply_chat_template(message,tokenize=False,add_generation_prompt=False,enable_thinking=False)
    opened=tokenizer.apply_chat_template(message,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    if not opened.startswith(plain)or len(opened)==len(plain):raise ValueError('Generation prefix cannot be isolated from official template')
    return opened[len(plain):]

def assistant_closure(tokenizer):
    placeholder='TREYRIN_ROLE_CLOSURE_c3296dc7'
    frame=role_frame(tokenizer,'assistant',placeholder)
    if frame.count(placeholder)!=1:raise ValueError('Assistant closure cannot be isolated from official template')
    return tokenizer(frame.split(placeholder)[1],add_special_tokens=False)['input_ids']

class RoleConversationSession(NativeReplaySession):
    def __init__(self,candidate,*,window=8192,capacity=2048,system_prompt=None,self_info=False,paging_root=None):
        if candidate.session.operations:raise ValueError('Role ABI starts from an empty conversation, never retags old experience')
        # An empty native stream, with the official role template only. No
        # evidence-auditor system instruction is inserted into all tasks.
        super().__init__(candidate,window=window,capacity=capacity,initialize=False)
        self.codes=EpisodeCodes.empty(self.session.state,capacity)
        self.mental_codes=MentalCodes.empty(self.session.state,capacity)
        self.pending_user=None;self.turn_count=0
        self._assistant_prefix=generation_prefix(candidate.tokenizer)
        self._assistant_closure=assistant_closure(candidate.tokenizer)
        candidate.active_native=self
        if paging_root is not None:self.enable_paging(paging_root)
        if system_prompt is not None:
            if not isinstance(system_prompt,str)or not system_prompt:raise ValueError('Explicit nonempty system string or None required')
            from native_session import system_prefix
            self.capsule=self._episode(system_prefix(candidate.tokenizer,system_prompt),self.session.state).commit_observed_prefix()
        if self_info:self._inject_self_info()

    def _inject_self_info(self):
        from runtime_self_info import describe
        text=describe(self.candidate)
        record=dict(event_id='runtime-self-info',text=text,source='runtime_config',known_at=timestamp())
        self._record_role('observation',record,speaker='system',frame=self._frame('tool',text))

    def save(self,path):
        """One role-conversation generation: four banks plus the role fields.

        A pending live user request is part of the conversation state, so its
        frame and the prefix capsule it was committed on are persisted too; a
        restart can still answer that exact request. The assistant prefix stays
        the actual template text (not a list of characters) and is re-derived
        against the tokenizer on restore.
        """
        root=Path(path)
        if root.exists():raise ValueError('Role conversation snapshot directory must be new')
        NativeReplaySession.save(self,root)
        pending=None
        if self.pending_user is not None:
            before=self.pending_user['before']
            if not isinstance(before,Capsule):raise ValueError('Pending request must keep its prefix capsule')
            if before.encoder_binding!=self.candidate.model._binding:raise ValueError('Pending prefix capsule belongs to different parameters')
            if before.seen>self.capsule.seen:raise ValueError('Pending prefix capsule is younger than the committed capsule')
            Capsule.save(before,root/'pending-before')
            pending={name:self.pending_user[name] for name in ('text','frame','event_id','known_at')}
            if any(not isinstance(value,str)or not value for value in pending.values()):
                raise ValueError('Pending request fields must be nonempty strings')
            pending['prefix_capsule']='pending-before'
            pending['generated_tokens']=list(self.pending_user.get('generated_tokens',[]))
        state=dict(schema='treyrin-role-conversation-v1',
                   assistant_prefix=self._assistant_prefix,
                   assistant_closure=[int(x)for x in self._assistant_closure],
                   pending_user=pending,turn_count=int(self.turn_count),
                   native_tokens=int(self.capsule.seen),window=int(self.window),capacity=int(self.capacity),
                   parameter_binding=self.candidate.model._binding,
                   read_continuation=getattr(self,'read_continuation',None),
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if not isinstance(state['assistant_prefix'],str)or not state['assistant_prefix']:
            raise ValueError('Role assistant prefix must be the actual template text')
        (root/'role-state.json').write_text(json.dumps(state,sort_keys=True,indent=2),encoding='utf-8')
        return state

    @classmethod
    def restore(cls,path,candidate):
        root=Path(path)
        state=json.loads((root/'role-state.json').read_text('utf-8'))
        # NativeSession.restore calls candidate.restore(core), which refuses a
        # non-fresh session; give the nested restore a fresh session shell.
        if getattr(candidate.session,'optimizer_steps',0):
            candidate.session=candidate.new_session()
        if (state.get('schema')!='treyrin-role-conversation-v1'
                or state.get('source_sha256')!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
            raise ValueError('Role conversation source/schema differs')
        prefix=state['assistant_prefix'];closure=[int(x)for x in state['assistant_closure']]
        # The official template must still produce exactly the saved role
        # frames; otherwise an old conversation would be silently retagged.
        if generation_prefix(candidate.tokenizer)!=prefix:
            raise ValueError('Official role prefix changed; refuse to retag the conversation')
        if assistant_closure(candidate.tokenizer)!=closure:
            raise ValueError('Official assistant closure changed; refuse to retag the conversation')
        result=super().restore(path,candidate)
        if state.get('parameter_binding')!=candidate.model._binding:
            raise ValueError('Role conversation belongs to different restored parameters')
        result._assistant_prefix=prefix
        result._assistant_closure=closure
        result.turn_count=int(state['turn_count'])
        result.read_continuation=state.get('read_continuation')
        if result.capsule.seen!=int(state['native_tokens']):
            raise ValueError('Native capsule count differs from the saved role transcript')
        pending=state.get('pending_user')
        if pending is None:result.pending_user=None
        else:
            capsule_dir=root/pending['prefix_capsule']
            if not (capsule_dir/'manifest.json').is_file()or not (capsule_dir/'native.safetensors').is_file():
                raise ValueError('Missing pending prefix capsule; that live request cannot be resumed')
            capsule=Capsule.restore(capsule_dir,candidate.model.body,binding=candidate.model._binding)
            if capsule.seen>result.capsule.seen:raise ValueError('Pending prefix capsule is younger than the committed capsule')
            result.pending_user=dict(text=pending['text'],frame=pending['frame'],before=capsule,
                                     event_id=pending['event_id'],known_at=pending['known_at'])
            if pending.get('generated_tokens'):result.pending_user['generated_tokens']=list(pending['generated_tokens'])
        candidate.session=result.session
        candidate.active_native=result
        return result

    def prefill_message(self,role,content):
        if self.pending_user is not None:raise ValueError('Controlled history cannot replace a pending live request')
        if not isinstance(content,str)or not content:raise ValueError('Nonempty controlled history content required')
        if role=='user':
            record=dict(event_id='history-user-'+uuid.uuid4().hex,text=content,source='human_user',known_at=timestamp())
            return self._record_role('observation',record,speaker='user',frame=self._frame('user',content))
        if role=='assistant':
            record=dict(record_id='history-assistant-'+uuid.uuid4().hex,text=content,kind='assistant_output',
                source='assistant_history',known_at=timestamp())
            return self._record_role('mental',record,speaker='assistant',frame=self._frame('assistant',content))
        raise ValueError('Controlled history roles are user or assistant; set system_prompt before begin_case')

    def _episode(self,text,state,choices=(),*,body_feedback=None):
        c=self.candidate;encoded=c.tokenizer(text,add_special_tokens=False)['input_ids']
        if not encoded:raise ValueError('Nonempty causal frame required')
        ids=torch.tensor([encoded],device=next(c.model.parameters()).device)
        options=self.session.mode.forward_options();options['drift_level']=self.session._drift_level()
        if body_feedback is not None:options['body_feedback']=body_feedback
        if choices:
            rows=[torch.tensor(c.encode(x),device=ids.device)for x in choices]
            padded=torch.nn.utils.rnn.pad_sequence(rows,batch_first=True,padding_value=0)
            mask=torch.arange(padded.shape[1],device=ids.device)[None]<torch.tensor([len(x)for x in rows],device=ids.device)[:,None]
            options.update(candidate_ids=padded[None],candidate_mask=mask.long()[None],body_choice_labels=True)
        with torch.no_grad():return ConversationEpisode(c.model,state,ids,self.capsule,**options)

    def _frame(self,role,text):
        return role_frame(self.candidate.tokenizer,role,text)

    def _eos_token_ids(self):
        body=getattr(self.candidate.model.body,'model',None)
        generation=getattr(body,'generation_config',None)
        values=[getattr(generation,'eos_token_id',None),
            getattr(getattr(body,'config',None),'eos_token_id',None),self.candidate.tokenizer.eos_token_id]
        result=set()
        for value in values:
            if value is None:continue
            for token in value if isinstance(value,(list,tuple,set))else(value,):
                if type(token)is not int or token<0:raise ValueError('Invalid generation EOS token')
                result.add(token)
        # Registered equality with the raw arm: the local model-derived
        # generation config contains endoftext; the chat template closes with
        # im_end. Either is a genuine stop, never a length-budget success.
        return sorted(result)if result else None

    def _memory_tokens(self,kind,event,speaker):
        # Protocol data lives in owned replay codes; it does not replace the
        # current user's request in the live decoder stream.
        payload=dict(record_kind=kind,speaker=speaker,provenance=json.loads(event.provenance_text()),text=event.text)
        frame=self._frame('tool',json.dumps(payload,ensure_ascii=False,sort_keys=True))
        ids=self.candidate.tokenizer(frame,add_special_tokens=False)['input_ids']
        if not ids:raise ValueError('Nonempty typed replay frame required')
        if len(ids)>self.capacity:
            if getattr(self,'live_memory',None) is None:raise ValueError('Whole typed replay frame exceeds capacity; enable paged memory')
            # Fixed GPU banks are a resident code page. The complete immutable
            # token stream is committed by _archive_event before publication.
            ids=ids[:self.capacity]
        return torch.tensor([ids],device=self.session.state.keys.device)

    def _record_role(self,kind,record,*,speaker,frame):
        c=self.candidate;old=self.session;before=self.capsule;codes=self.codes;mental=self.mental_codes
        event=PublicObservation(**record)if kind=='observation'else MentalRecord(**record)
        if event in (old.observations if kind=='observation'else old.mental_records):return dict(status='exact_replay_ignored')
        replay=self._memory_tokens(kind,event,speaker);branch=copy.copy(old)
        if kind=='mental':branch.remember(event)
        elif not c.model.config.native_single_pass:raise ValueError('Single-pass public perception is required')
        episode=self._episode(frame,old.state);capsule=episode.commit_observed_prefix()
        if kind=='observation':
            source=branch._ids(event.provenance_text());echo=branch._ids(event.text)
            part=episode.ids[:,-episode.output.body_hidden.shape[1]:]
            with torch.no_grad():
                staged=c.model.commit_observation_output(part,torch.ones_like(part),source,torch.ones_like(source),
                    old.state,episode.output,echo_ids=echo,**branch.mode.forward_options()).detached()
            result=branch._accept_observation(event,staged,conflict=any(o.event_id==event.event_id for o in old.observations))
            new_codes=codes.commit(old.state,branch.state,replay,native_start=before.seen);new_mental=mental
        else:
            result=dict(status='mental_recorded');new_codes=codes
            new_mental=mental.commit(old.state,branch.state,replay,native_start=before.seen)
        new_codes.check(branch.state);new_mental.check(branch.state)
        self._archive_event(kind,event,branch.state,before.seen)
        self.session=branch;c.session=branch;self.capsule=capsule;self.codes=new_codes;self.mental_codes=new_mental
        return result

    def _record(self,kind,record):
        if kind=='mental'and record.get('kind')=='assistant_output':
            return self._record_role(kind,record,speaker='assistant',frame=self._frame('assistant',record['text']))
        payload=json.dumps(dict(record_kind=kind,**record),ensure_ascii=False,sort_keys=True)
        return self._record_role(kind,record,speaker='tool',frame=self._frame('tool',payload))

    def user_turn(self,text,*,event_id=None,known_at=None):
        if self.pending_user is not None:raise ValueError('Respond to the pending user turn before accepting another')
        record=dict(event_id=event_id or'user-'+uuid.uuid4().hex,text=text,source='human_user',known_at=known_at or timestamp())
        before=self.capsule;frame=self._frame('user',text)
        result=self._record_role('observation',record,speaker='user',frame=frame)
        self.pending_user=dict(text=text,frame=frame,before=before,event_id=record['event_id'],known_at=record['known_at'])
        return result

    def _response_episode(self,replay_budget):
        if self.pending_user is None:raise ValueError('A live user request is required')
        first=self._episode(self._assistant_prefix,self.session.state)
        selected=self._select_replay(first,self.pending_user['text'],replay_budget,False)
        if not selected['token_ids']:return first,selected
        # Rebuild ONLY scratch before the last live user turn: old memory is
        # bounded data, then the actual current user remains the final request.
        observed=self.capsule
        try:
            self.capsule=self.pending_user['before']
            ids=torch.cat(selected['token_ids'])[None]
            options=self.session.mode.forward_options();options['drift_level']=self.session._drift_level()
            with torch.no_grad():
                replay=ConversationEpisode(self.candidate.model,self.session.state,ids,self.capsule,**options)
            self.capsule=replay.commit_observed_prefix()
            final=self._episode(self.pending_user['frame']+self._assistant_prefix,self.session.state)
        finally:self.capsule=observed
        return final,selected

    def _commit_assistant(self,answer,episode,generated,known_at,*,had_replay=False,revision_of=''):
        old=self.session;branch=copy.copy(old);mental=self.mental_codes;start=self.capsule.seen
        raw_encode=lambda text:self.candidate.tokenizer(text,add_special_tokens=False)['input_ids']
        chunks=list(text_chunks(answer,raw_encode,budget=448))if answer.strip()else[dict(text='[empty model output]')]
        root='assistant-'+uuid.uuid4().hex
        for index,chunk in enumerate(chunks):
            record=MentalRecord(record_id=root+'-'+str(index),text=chunk['text'],kind='assistant_output',
                source='model_output',known_at=known_at,evidence_ids=(),revises=revision_of if index==0 else '')
            previous=branch.state;replay=self._memory_tokens('mental',record,'assistant')
            branch.remember(record);mental=mental.commit(previous,branch.state,replay,native_start=start)
            self._archive_event('mental',record,branch.state,start)
        cache=generated.cache
        if had_replay:
            # Retrieved memory is hypothetical read scratch. Commit only the
            # actual delivered assistant tokens onto the real user prefix;
            # never turn reactivation into duplicate observed experience.
            live=self._episode(self._assistant_prefix,old.state);cache=live.cache
            all_ids=torch.tensor([generated.token_ids],device=episode.ids.device)
            with torch.no_grad():
                for start_at in range(0,all_ids.shape[1],448):
                    ids=all_ids[:,start_at:start_at+448]
                    self.candidate.model(ids,torch.ones_like(ids),old.state,_decode_cache=cache,**live.options)
            episode=live
        tail=list(self._assistant_closure)
        # EOS was already sampled and fed into cache. Supply only template
        # separators after it; a length-limited turn needs explicit transport
        # closure but remains marked incomplete in the returned diagnostics.
        if generated.eos_reached and tail and tail[0]==generated.token_ids[-1]:tail=tail[1:]
        if tail:
            ids=torch.tensor([tail],device=episode.ids.device)
            with torch.no_grad():self.candidate.model(ids,torch.ones_like(ids),episode.state,_decode_cache=cache,**episode.options)
        capsule=Capsule(cache.body_cache,cache.seen_tokens,cache.native_window,self.candidate.model._binding)
        self.codes.check(branch.state);mental.check(branch.state)
        if int(branch.state.writes)!=int(old.state.writes):raise ValueError('Assistant output changed factual write count')
        self.session=branch;self.candidate.session=branch;self.mental_codes=mental;self.capsule=capsule
        self.pending_user=None;self.turn_count+=1

    def respond(self,*,max_new_tokens=None,replay_budget=4,known_at=None,on_token=None,should_cancel=None):
        started=time.monotonic();episode,selected=self._response_episode(replay_budget)
        from verify_revise import route
        episode,correction=route(self,self.pending_user['text'],episode,selected)
        eos=self._eos_token_ids()
        prefix=list(self.pending_user.get('generated_tokens',[]))
        def progress(token,tokens):
            self.pending_user['generated_tokens']=tokens
            if on_token is not None:on_token(token,tokens)
        try:
            with torch.no_grad():generated=episode.generate_result(max_new_tokens,eos_token_id=eos,
                reserved_tokens=len(self._assistant_closure),on_token=progress,
                should_cancel=should_cancel,prefix_tokens=prefix)
        except GenerationInterrupted as exc:generated=exc.result
        answer=self.candidate.tokenizer.decode(generated.token_ids,skip_special_tokens=True)
        if generated.finish_reason=='cancelled':self.pending_user['generated_tokens']=generated.token_ids
        else:self._commit_assistant(answer,episode,generated,known_at or timestamp(),
            had_replay=bool(selected['token_ids']) or bool(correction and correction['used']),
            revision_of=correction.get('revision_of','') if correction and correction['used'] else '')
        return dict(answer=answer,answer_token_ids=generated.token_ids,finish_reason=generated.finish_reason,
            diagnostics=dict(eos_reached=generated.eos_reached,truncated=not generated.eos_reached,
                generation_budget=max_new_tokens,seconds=time.monotonic()-started,user_role='user',assistant_role='assistant',
                resumed_prefix_tokens=len(prefix),resumable=self.pending_user is not None,
                correction=correction,
                eos_token_ids=list(eos)if isinstance(eos,(list,tuple,set))else eos,
                user_observation_count=len(self.session.observations),assistant_output_count=sum(r.kind=='assistant_output'for r in self.session.mental_records),
                committed_native_tokens=self.capsule.seen,actual_sampled_assistant_tokens_committed=True,
                hypothetical_replay_committed=False,
                replayed_record_addresses=selected.get('replayed_record_addresses',[]),parameter_updates=0,
                selected_record_addresses=selected.get('selected_record_addresses',[]),
                read_outcome=selected.get('read_outcome'),
                read_abstained=bool(selected.get('read_abstained')),
                replay_budget_rejected=list(selected.get('replay_budget_rejected',[])),
                model_binding=self.candidate.model._binding))

    def turn(self,text,*,max_new_tokens=None,replay_budget=4):
        self.user_turn(text);return self.respond(max_new_tokens=max_new_tokens,replay_budget=replay_budget)

class ConversationAdapter:
    def __init__(self,candidate,*,window=8192,capacity=2048,system_prompt=None,self_info=False):
        self.candidate=candidate;self.window=window;self.capacity=capacity;self.self_info=self_info
        self.base_steps=candidate.session.optimizer_steps;self._system_prompt=system_prompt;self.begin_case()
    @property
    def system_prompt(self):return self._system_prompt
    @property
    def stop_token_ids(self):return self.native._eos_token_ids()
    @property
    def eos_token_ids(self):return self.native._eos_token_ids()
    @system_prompt.setter
    def system_prompt(self,value):
        if value is not None and(not isinstance(value,str)or not value):raise ValueError('Explicit nonempty system string or None required')
        if hasattr(self,'native')and self.native.session.operations:raise ValueError('Cannot alter system after experience has begun')
        self._system_prompt=value
        if hasattr(self,'native'):self.begin_case()
    def begin_case(self):
        c=self.candidate;c.session=c.new_session();c.session.optimizer_steps=self.base_steps;c.conversation=[]
        self.native=RoleConversationSession(c,window=self.window,capacity=self.capacity,system_prompt=self._system_prompt,self_info=getattr(self,'self_info',True));return self
    def turn(self,text,max_new_tokens=None):return self.native.turn(text,max_new_tokens=max_new_tokens)
    def prefill_message(self,role,content):return self.native.prefill_message(role,content)
    def prefill_history(self,messages):
        for message in messages:self.prefill_message(message['role'],message['content'])
        return self

def factory(root,checkpoint,*,window=8192,capacity=2048,system_prompt=None):
    root=Path(root).resolve();sys.path.insert(0,str(root))
    import candidate
    if Path(candidate.__file__).resolve()!=root/'candidate.py':raise ValueError('Another candidate source already imported in this process')
    c=candidate.Candidate();c.restore(Path(checkpoint))
    return ConversationAdapter(c,window=window,capacity=capacity,system_prompt=system_prompt)
