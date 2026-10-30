"""A second hippocampal read/replay round inside the same neural model session.

Physical observed codes are retained only where the existing core actually
writes/promotes them. The core's neural read allocation chooses reactivation.
The application neither searches a journal nor chooses a semantic answer.
"""
import json
from pathlib import Path
import torch
from safetensors.torch import save_file,load_file
from episode_codes import EpisodeCodes
from mental_codes import MentalCodes
from native_session import NativeSession
from native_stream import digest
from native_decode import NativeEpisode
from treyrin.structural_learning import PublicObservation

class NativeReplaySession(NativeSession):
    def __init__(self,candidate,*,window=8192,capacity=2048,initialize=True):
        super().__init__(candidate,window=window,initialize=initialize)
        self.capacity=capacity
        self.codes=EpisodeCodes.empty(self.session.state,capacity)if initialize else None
        self.mental_codes=MentalCodes.empty(self.session.state,capacity)if initialize else None
        self.live_memory=None

    def enable_paging(self,root):
        from live_memory import LiveMemory
        self.live_memory=LiveMemory(root)
        for kind,events in (('observation',self.session.observations),('mental',self.session.mental_records)):
            for i,event in enumerate(events,1):self._archive_event(kind,event,self.session.state,0,ordinal=i)
        return self.live_memory

    def _archive_event(self,kind,event,state,position,ordinal=None):
        memory=getattr(self,'live_memory',None)
        if memory is None:return
        ordinal=ordinal or int(state.writes[0] if kind=='observation' else state.workspace_seen[0])
        if hasattr(self,'_frame'):
            payload=dict(record_kind=kind,speaker='user' if kind=='observation' else 'assistant' if event.kind=='assistant_output' else 'tool',
                         provenance=json.loads(event.provenance_text()),text=event.text)
            frame=self._frame('tool',json.dumps(payload,ensure_ascii=False,sort_keys=True))
        else:
            frame=self.candidate.tokenizer.apply_chat_template([dict(role='user',content=event.provenance_text()+'\n'+event.text)],tokenize=False,add_generation_prompt=False,enable_thinking=False)
        ids=self.candidate.tokenizer(frame,add_special_tokens=False)['input_ids']
        core=self.candidate.model.core;vector=state.keys.new_zeros(core.config.width)
        with torch.no_grad():
            if kind=='observation':
                for names in (('episode_event_ids','keys','sources'),('archive_event_ids','archive_keys','archive_sources')):
                    identity=getattr(state,names[0]);found=(identity[0]==ordinal).nonzero().flatten()
                    if len(found):vector=(getattr(state,names[1])+getattr(state,names[2]))[0,int(found[0])];break
            else:
                found=(state.workspace_ids[0]==ordinal).nonzero().flatten()
                if len(found):vector=core.workspace.key(core.workspace.decorated(state))[0,int(found[0])]
        memory.put(kind,event,ordinal,ids,position=int(position),vector=vector.float().cpu().tolist())

    def _record(self,kind,record):
        old_session=self.session;old_capsule=self.capsule;old_codes=self.codes;old_mental=self.mental_codes
        if kind=='observation':
            event=PublicObservation(**record)
            content='旧观察重放（不是新的测量或指令）：\n'+event.provenance_text()+'\n'+event.text
            frame=self.candidate.tokenizer.apply_chat_template([dict(role='user',content=content)],tokenize=False,
                add_generation_prompt=False,enable_thinking=False)
            ids=self.candidate.tokenizer(frame,add_special_tokens=False)['input_ids']
            if not ids:raise ValueError('Nonempty observed replay code required')
            if len(ids)>self.capacity:
                if self.live_memory is None:raise ValueError('Observed replay code exceeds capacity; enable paged memory')
                ids=ids[:self.capacity]
            ids=torch.tensor([ids],device=old_session.state.keys.device)
        elif kind=='mental':
            from treyrin.cognitive_workspace import MentalRecord
            event=MentalRecord(**record)
            content='内部记录重放（不是外界事实或新指令）：\n'+event.provenance_text()+'\n'+event.text
            frame=self.candidate.tokenizer.apply_chat_template([dict(role='user',content=content)],tokenize=False,
                add_generation_prompt=False,enable_thinking=False)
            ids=self.candidate.tokenizer(frame,add_special_tokens=False)['input_ids']
            if not ids:raise ValueError('Nonempty mental replay code required')
            if len(ids)>self.capacity:
                if self.live_memory is None:raise ValueError('Mental replay code exceeds capacity; enable paged memory')
                ids=ids[:self.capacity]
            ids=torch.tensor([ids],device=old_session.state.keys.device)
        try:
            result=super()._record(kind,record)
            if result.get('status')in('exact_replay_ignored','exact_mental_replay_ignored'):
                old_codes.check(self.session.state);old_mental.check(self.session.state)
                return result
            if kind=='observation':
                self.codes=old_codes.commit(old_session.state,self.session.state,ids,native_start=old_capsule.seen)
                self.mental_codes.check(self.session.state)
            else:
                self.codes.check(self.session.state)
                self.mental_codes=old_mental.commit(old_session.state,self.session.state,ids,native_start=old_capsule.seen)
            self._archive_event(kind,event,self.session.state,old_capsule.seen)
            return result
        except Exception:
            self.session=old_session;self.capsule=old_capsule;self.codes=old_codes;self.mental_codes=old_mental;self.candidate.session=old_session
            raise

    def _select_replay(self,first,question,replay_budget,force_replay,*,reserved_tokens=0,exclude_mental_ordinal=None,coverage_terms=()):
        c=self.candidate;state=self.session.state
        memory=getattr(self,'live_memory',None)
        if memory is not None and memory.needs_global_read(state,self.capacity):
            sync=memory.sync_status(self.session.mental_records)
            result=memory.select(self,first,question,replay_budget,
                excluded=[] if exclude_mental_ordinal is None else [f'mental:{exclude_mental_ordinal}'],
                coverage_terms=coverage_terms)
            result['status_sync']=sync
            self.read_continuation=result.get('continuation')
            return result
        # The old-only tests/body adapters use their original neural selection.
        if not c.model.config.token_code_attention:
            selected=self.codes.read(state,first.output.candidate_reads[0,-1],first.output.read_allocation[0,-1],
                budget=replay_budget,before_position=None if force_replay else self.capsule.seen-self.window+1)
            selected.setdefault('replay_budget_rejected',[])
            selected['read_outcome']=('abstained' if selected.get('read_abstained')
                else ('read' if selected.get('token_ids') else 'no_selection'))
            return selected
        if type(replay_budget)is not int or replay_budget<1:raise ValueError('Replay budget must be >=1')
        a=first.output.candidate_reads[0,-1];allocation=first.output.read_allocation[0,-1]
        compact=getattr(self,'compact_replay',True)
        all_lengths=torch.cat((self.codes.hot_lengths,self.codes.cold_lengths),1)[0]
        factual_indices=(all_lengths>0).nonzero().flatten() if compact else torch.arange(len(all_lengths),device=all_lengths.device)
        if compact:
            hot=(self.codes.hot_lengths[0]>0).nonzero().flatten();cold=(self.codes.cold_lengths[0]>0).nonzero().flatten()
            facts=torch.cat((self.codes.hot[0,hot],self.codes.cold[0,cold]),0)
        else:facts=torch.cat((self.codes.hot,self.codes.cold),1)[0]
        fact_lengths=all_lengths[factual_indices]
        fact_ids=torch.cat((self.codes.hot_ids,self.codes.cold_ids),1)[0,factual_indices]
        fact_positions=torch.cat((self.codes.hot_native_start,self.codes.cold_native_start),1)[0,factual_indices]
        mental=self.mental_codes.bank(state,compact=compact);factual_count=len(fact_lengths)
        values=torch.cat((facts,mental['codes']),0);lengths=torch.cat((fact_lengths,mental['lengths']))
        positions=torch.cat((fact_positions,mental['positions']));identities=torch.cat((fact_ids,mental['ids']))
        self.last_read_code_elements=values.numel()
        workspace=c.model.core.workspace
        query=first.output.scratch_working[:,-1]
        goal_vectors=state.goal_vectors+.1*state.goal_sources+workspace.kind.weight[4]+workspace.status(state.goal_statuses)
        vectors=torch.cat((workspace.decorated(state),goal_vectors),1)
        from treyrin.mental_archive import decorated as archive_decorated
        cold=archive_decorated(workspace,state)
        if cold is not None:vectors=torch.cat((vectors,cold),1)
        if compact:vectors=vectors[:,mental['physical_indices']]
        scores=(workspace.key(vectors)*workspace.query(query)[:,None]).sum(-1)[0]/workspace.width**.5
        legal=mental['lengths']>0
        mental_prior=scores.masked_fill(~legal,-1e9).softmax(-1)*legal
        mental_prior=mental_prior/mental_prior.sum().clamp_min(1e-30)
        factual_prior=(a*allocation[:-1,None]).sum(0)[factual_indices]
        factual_prior=factual_prior/factual_prior.sum().clamp_min(1e-30)
        g=workspace.read_gain.sigmoid()
        if not bool(legal.any()):g=g*0
        elif not bool((fact_lengths>0).any()):g=g*0+1
        prior=torch.cat(((1-g)*factual_prior,g*mental_prior))[None]
        public_ids=c.tokenizer(question,add_special_tokens=False)['input_ids']
        with torch.no_grad():
            result=c.model.core.code_attention(torch.tensor(public_ids,device=values.device),values,lengths,prior,
                allocation.new_tensor([1-float(allocation[-1]),float(allocation[-1])]))
        selected=dict(token_ids=[],event_ordinals=[],selected_record_addresses=[],
            read_abstained=result['abstained'],replay_budget_rejected=[],
            read_outcome='abstained' if result['abstained'] else None)
        if result['abstained']:return selected
        p=result['probabilities'][:-1]
        if exclude_mental_ordinal is not None:
            duplicate=(torch.arange(len(p),device=p.device)>=factual_count)&(identities==exclude_mental_ordinal)
            p=p.masked_fill(duplicate,0)
        rank=p.topk(min(len(p),replay_budget)).indices
        rank=[int(j)for j in rank if int(lengths[j])>0 and float(p[j])>0]
        selected['selected_record_addresses']=[dict(kind='observation'if j<factual_count else'mental',ordinal=int(identities[j]))for j in rank]
        if not force_replay and not any(int(positions[j])<self.capsule.seen-self.window+1 for j in rank):
            selected['read_outcome']='inside_window'
            return selected
        # Whole episodes only, bounded reactivation, ordered by actual native
        # chronology across factual and non-factual counters.
        budget=self.window//2-reserved_tokens;pieces=[];chosen=[];rejected=[]
        for j in rank:
            tokens=values[j,:int(lengths[j])].long()
            label=('observation:'if j<factual_count else'mental:')+str(int(identities[j]))
            note='以上旧记录的实际内部来源地址为 '+label+'。地址仅标识来源，不保证命题为真；引用时请准确使用此地址。'
            if j>=factual_count:
                status=int(mental['statuses'][j-factual_count])
                from treyrin.cognitive_workspace import STATUSES
                note+='当前状态：'+STATUSES[status]+'。它是内部记录，不是已观测事实。'
            frame=c.tokenizer.apply_chat_template([dict(role='user',content=note)],
                tokenize=False,add_generation_prompt=False,enable_thinking=False)
            suffix=c.tokenizer(frame,add_special_tokens=False)['input_ids']
            tokens=torch.cat((tokens,tokens.new_tensor(suffix)))
            if len(tokens)>budget:
                rejected.append(dict(kind='observation'if j<factual_count else'mental',ordinal=int(identities[j]),required_tokens=len(tokens)))
                continue
            pieces.append((int(positions[j]),tokens));chosen.append(j);budget-=len(tokens)
        selected['token_ids']=[t for _,t in sorted(pieces,key=lambda x:x[0])]
        selected['event_ordinals']=[int(identities[j])for j in chosen if j<factual_count]
        selected['replayed_record_addresses']=[dict(kind='observation'if j<factual_count else'mental',ordinal=int(identities[j]))for j in chosen]
        selected['replay_budget_rejected']=rejected
        selected['read_outcome']=('read' if selected['token_ids'] else
                                  ('budget_rejected' if rejected else 'no_selection'))
        return selected

    def _selected_goal_flash(self,*,include_action=True):
        """Model-chosen pointer -> owned code -> cortical read, never journal text."""
        c=self.candidate;state=self.session.state
        if not c.model.config.cortical_goal_control or int(state.cognitive_ticks[0])==0:return None
        slot=int(state.cognitive_goal[0]);slots=c.model.config.workspace_slots
        if slot>=slots:return None
        identity=int(state.cognitive_goal_identity[0])
        if identity<=0 or int(state.goal_ids[0,slot])!=identity or int(state.goal_statuses[0,slot])!=0:return None
        self.mental_codes.check(state)
        length=int(self.mental_codes.goal_lengths[0,slot])
        if int(self.mental_codes.goal_ids[0,slot])!=identity or length<=0:raise ValueError('Chosen goal lost its model-owned code')
        from treyrin.cognitive_workspace import ACTIONS
        action=ACTIONS[int(state.cognitive_joint[0,slot].argmax())]
        note=f'模型的神经控制层刚才选择了这个目标（内部来源地址mental:{identity}）与动作{action}。这是内部意图，不是外界事实或执行完成回执；请围绕该目标理解当前工作，再依据用户问题回应。'
        if not include_action:
            note=f'上一次神经控制分配的目标位于内部来源地址mental:{identity}。这只是先前的目标焦点，不要求本轮继续选择它；请结合新证据与所有活动目标重新分配资源，不要无收益地重复已经完成的取证或操作。'
        frame=c.tokenizer.apply_chat_template([dict(role='user',content=note)],tokenize=False,add_generation_prompt=False,enable_thinking=False)
        tokens=self.mental_codes.goals[0,slot,:length].long()
        suffix=tokens.new_tensor(c.tokenizer(frame,add_special_tokens=False)['input_ids'])
        combined=torch.cat((tokens,suffix))
        if len(combined)>self.window//2:
            raise ValueError('Whole selected-goal code exceeds the bounded focus budget')
        return dict(tokens=combined,ordinal=identity,goal_slot=slot,action=action)

    def _query_episode(self,question,choices=(),*,replay_budget=4,force_replay=False,body_feedback=None,control_opportunity=False):
        c=self.candidate;text=question
        from public_labels import public_label
        if choices:text+='\n选项：\n'+'\n'.join(f'{public_label(i)}. {x}'for i,x in enumerate(choices))+'\n请先输出选项字母，再简短说明理由。'
        prompt=c.tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        first=self._episode(prompt,self.session.state,choices,body_feedback=body_feedback)
        focus=self._selected_goal_flash(include_action=not control_opportunity)
        if focus is None:selected=self._select_replay(first,question,replay_budget,force_replay,
            coverage_terms=tuple(choices))
        else:
            if type(replay_budget)is not int or not 1<=replay_budget<=8:raise ValueError('Replay budget must be 1..8')
            if replay_budget==1:
                selected=dict(token_ids=[],event_ordinals=[],selected_record_addresses=[],
                    read_abstained=False,replay_budget_rejected=[],read_outcome='goal_focus')
            else:selected=self._select_replay(first,question,replay_budget-1,force_replay,
                reserved_tokens=len(focus['tokens']),exclude_mental_ordinal=focus['ordinal'],
                coverage_terms=tuple(choices))
            selected['token_ids'].append(focus['tokens'])
            address=dict(kind='mental',ordinal=focus['ordinal'])
            selected.setdefault('replayed_record_addresses',[]).append(address)
            selected.setdefault('replay_budget_rejected',[])
            selected['read_abstained']=False
            selected['read_outcome']='goal_focus'
            selected['neural_goal_focus']={k:v for k,v in focus.items() if k!='tokens'}
        if choices and getattr(self,'live_memory',None) is not None and 'coverage' not in selected:
            selected['coverage']=self.live_memory.coverage(choices)
        if selected['token_ids']:
            # Replay into a fork of the pre-question native state, then ask the
            # SAME question. Hypothetical reads never alter committed history.
            observed=self.capsule
            replay_ids=torch.cat(selected['token_ids'])[None]
            options=self.session.mode.forward_options();options['drift_level']=self.session._drift_level()
            if body_feedback is not None:options['body_feedback']=body_feedback
            with torch.no_grad():
                replay=NativeEpisode(c.model,self.session.state,replay_ids,observed,**options)
                temporary=replay.commit_observed_prefix()
            try:
                self.capsule=temporary;final=self._episode(prompt,self.session.state,choices,body_feedback=body_feedback)
            finally:self.capsule=observed
        else:final=first
        return final,selected

    def _answer_once(self,question,choices,*,max_new_tokens=None,replay_budget=4,force_replay=False):
        c=self.candidate
        final,selected=self._query_episode(question,choices,replay_budget=replay_budget,force_replay=force_replay)
        selected_ids=selected.get('selected_record_addresses',list(selected['event_ordinals']))
        from public_labels import public_label
        with torch.no_grad():
            labels=[c.tokenizer(public_label(i),add_special_tokens=False)['input_ids']for i in range(len(choices))]
            if any(len(x)!=1 for x in labels):raise ValueError('Single-token public choice labels required')
            tokens=final.generate(max_new_tokens,eos_token_id=c.tokenizer.eos_token_id,
                allowed_first_tokens=[x[0]for x in labels]if choices else None)
            native=final.output.native_logits[0].float().softmax(-1).cpu().tolist()
        return dict(answer=c.tokenizer.decode(tokens,skip_special_tokens=True),answer_token_ids=tokens,
            finish_reason=final.last_generation.finish_reason,eos_reached=final.last_generation.eos_reached,
            generation_budget=final.last_generation.budget,
            native_probabilities=native,native_choice_index=max(range(len(native)),key=lambda i:native[i]),
            observed_native_tokens=self.capsule.seen,attention_window=self.window,
            replayed_event_ordinals=selected['event_ordinals'],read_abstained=selected['read_abstained'],
            read_outcome=selected.get('read_outcome'),
            selected_event_ordinals=selected_ids,
            replayed_record_addresses=selected.get('replayed_record_addresses',[]),
            replay_budget_rejected=selected.get('replay_budget_rejected',[]),
            coverage=selected.get('coverage'),status_sync=selected.get('status_sync'),
            neural_goal_focus=selected.get('neural_goal_focus'),
            million_context_accepted=False)

    def query(self,question,choices=(),*,max_new_tokens=None,replay_budget=4,force_replay=False,
              order_control=False,unknown_margin=None):
        """Answer; with order_control=True run both option orders and decide on the semantic mean."""
        first=self._answer_once(question,choices,max_new_tokens=max_new_tokens,
            replay_budget=replay_budget,force_replay=force_replay)
        first['order_control']=False;first['protocol']=None;first['unknown']=False
        if not order_control or len(choices)<2:return first
        from readout_protocol import counterbalanced
        swapped=tuple(reversed(tuple(choices)))
        second=self._answer_once(question,swapped,max_new_tokens=max_new_tokens,
            replay_budget=replay_budget,force_replay=force_replay)
        protocol=counterbalanced(first['native_probabilities'],second['native_probabilities'],
            unknown_margin=unknown_margin)
        result=dict(first)
        result.update(order_control=True,protocol=protocol,unknown=protocol['unknown'],
            native_probabilities=protocol['semantic_probabilities'],
            native_choice_index=protocol['choice'],
            order_cells=dict(forward=dict(choices=list(choices),
                                          probabilities=first['native_probabilities']),
                             swapped=dict(choices=list(swapped),
                                          probabilities=second['native_probabilities'])),
            answers=dict(forward=first['answer'],swapped=second['answer']))
        return result

    def tick(self,tick):
        if not self.candidate.model.config.cortical_goal_control:
            return self.session.tick(tick)
        from native_cognition import native_cognitive_tick
        return native_cognitive_tick(self,tick)

    def save(self,path):
        root=Path(path);root.mkdir(exist_ok=False);super().save(root/'core-native')
        self.codes.check(self.session.state)
        save_file({n:t.detach().cpu().contiguous().clone()for n,t in self.codes.tensors().items()},root/'episode-codes.safetensors')
        self.mental_codes.check(self.session.state)
        save_file({n:t.detach().cpu().contiguous().clone()for n,t in self.mental_codes.tensors().items()},root/'mental-codes.safetensors')
        m=dict(schema='treyrin-episodic-native-session-v2',capacity=self.capacity,
            core_native_manifest_sha256=digest(root/'core-native'/'manifest.json'),codes_sha256=digest(root/'episode-codes.safetensors'),
            source_sha256=digest(__file__),codes_source_sha256=digest(Path(__file__).with_name('episode_codes.py')),
            mental_codes_sha256=digest(root/'mental-codes.safetensors'),mental_source_sha256=digest(Path(__file__).with_name('mental_codes.py')))
        (root/'manifest.json').write_text(json.dumps(m,sort_keys=True,indent=2),encoding='utf-8')
        if getattr(self,'live_memory',None) is not None:
            m['paged_memory']=self.live_memory.snapshot()
            (root/'manifest.json').write_text(json.dumps(m,sort_keys=True,indent=2),encoding='utf-8')

    @classmethod
    def restore(cls,path,candidate):
        root=Path(path);m=json.loads((root/'manifest.json').read_text('utf-8'))
        if m['schema']!='treyrin-episodic-native-session-v2'or m['source_sha256']!=digest(__file__)or m['codes_source_sha256']!=digest(Path(__file__).with_name('episode_codes.py'))or m['mental_source_sha256']!=digest(Path(__file__).with_name('mental_codes.py')):
            raise ValueError('Neural replay source/schema differs')
        if m['core_native_manifest_sha256']!=digest(root/'core-native'/'manifest.json')or m['codes_sha256']!=digest(root/'episode-codes.safetensors'):
            raise ValueError('Neural replay payload differs')
        base=NativeSession.restore(root/'core-native',candidate);result=cls.__new__(cls);result.__dict__.update(base.__dict__)
        result.capacity=m['capacity'];values=load_file(root/'episode-codes.safetensors',device=str(candidate.session.state.keys.device))
        result.codes=EpisodeCodes(**values,binding=candidate.model._binding);result.codes.check(candidate.session.state)
        if m['mental_codes_sha256']!=digest(root/'mental-codes.safetensors'):raise ValueError('Mental-code payload differs')
        values=load_file(root/'mental-codes.safetensors',device=str(candidate.session.state.keys.device))
        result.mental_codes=MentalCodes(**values,binding=candidate.model._binding);result.mental_codes.check(candidate.session.state)
        result.live_memory=None
        if m.get('paged_memory'):
            from live_memory import LiveMemory
            result.live_memory=LiveMemory.restore(m['paged_memory'])
        return result