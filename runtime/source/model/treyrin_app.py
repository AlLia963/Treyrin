"""Treyrin's single local model, permanent session and ordinary work surface."""
from __future__ import annotations
import argparse,contextlib,hashlib,json,os,subprocess,sys,time,uuid
from dataclasses import asdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,default=str);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def json_object(text):
    text=text.strip()
    if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
    return json.loads(text)

class StoreLease:
    """An OS-owned writer lease; a killed process releases it automatically."""
    def __init__(self,path):
        self.file=Path(path).open('a+b')
        if os.fstat(self.file.fileno()).st_size==0:self.file.write(b'0');self.file.flush()
        self.file.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.file.close();raise RuntimeError('This permanent session already has an active writer: '+str(path))
    def close(self):
        if not self.file.closed:self.file.close()
    def __del__(self):self.close()

class MaterialCollection:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
    def status(self):
        return [json.loads(p.read_text(encoding='utf-8')) for p in sorted(self.root.glob('*/COVERAGE.json'))]
    def save_component(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        write(path/'STATUS.json',dict(schema='treyrin-material-collection-v1',directories=self.status()))
        return dict(schema='treyrin-material-collection-v1',state='present',directories=len(self.status()))
    def progress(self):return dict(directories=self.status())

class Treyrin:
    def __init__(self,store=None,*,candidate=None,workspace=None,window=8192,auto_intents=True):
        from treyrin.preassembly_session import PreassemblySession,TaskStoreComponent,PagingStoreComponent
        from treyrin.task_store import TaskStore
        from preassembly_api import PreassemblyAPI
        self.root=Path(store or ROOT/'resident-session').resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self.lease=StoreLease(self.root/'writer.lock')
        self.workspace=Path(workspace or self.root/'work').resolve();self.workspace.mkdir(parents=True,exist_ok=True)
        self.auto_intents=auto_intents
        if candidate is None:
            from candidate import Candidate
            from treyrin.preassembly_session import model_directory
            candidate=Candidate();candidate.restore(model_directory(ROOT))
        self.candidate=candidate
        self.tasks=TaskStore.open(self.root/'tasks')
        self.materials=MaterialCollection(self.root/'materials')
        providers={'tasks':TaskStoreComponent(self.tasks),'materials':self.materials}
        self.resident=PreassemblySession.open(self.root,candidate=candidate,providers=providers,
            window=window,self_info=False,create=True,source=ROOT)
        self.native=self.resident.conversation
        if getattr(self.native,'live_memory',None) is None:self.native.enable_paging(self.root/'memory')
        self.resident.providers['paging']=PagingStoreComponent(self.native.live_memory.store)
        self.api=PreassemblyAPI(candidate,session=self.native)
        self.tools=LocalTools(self)
        candidate.runtime_app=self
        evidence=ROOT/'EVIDENCE.json'
        candidate.release_evidence=json.loads(evidence.read_text(encoding='utf-8')) if evidence.exists() else {}
        if not self.native.session.operations and hasattr(candidate,'assets'):self.native._inject_self_info()

    def save(self,reason='manual'):return self.resident.commit(reason=reason).sequence
    def close(self):
        self.tasks.close()
        if getattr(self.native,'live_memory',None):self.native.live_memory.close()
        self.lease.close()

    def _intake(self,text,event_id,known_at):
        from intent_lifecycle import IntentLifecycle
        route=self.api.decide(text,candidates=('这是一项需要保持目标、材料或执行进度的委托','这是一次直接问答或普通交流'))
        if route['judgment']['selected_index']!=0:return dict(proposals=[],route=route['judgment'])
        result=IntentLifecycle(self.native).intake(text,evidence_id=event_id,known_at=known_at,commit=False)
        known={o.event_id for o in self.native.session.observations}
        for p in result['proposals']:
            if p.get('acceptance')=='declined':
                if p.get('revision_of'):self.tasks.cancel(p['revision_of'],reason=text)
            elif p.get('revision_of'):
                self.tasks.revise(p['revision_of'],changes={'objective':p['text']},author='user',note=text)
            else:self.tasks.submit(text,p,constraints=(text,),known_evidence=known,request_id=event_id+':'+p['record_id'])
        self._activate_tasks()
        return result

    def _activate_tasks(self):
        from treyrin.goal_metadata import goal_metadata
        state=self.native.session.state
        active=set(int(i) for i in state.goal_ids[(state.goal_ids>0)&(state.goal_statuses==0)].tolist())
        records=self.native.session.mental_records
        names={r.record_id for i,r in enumerate(records,1) if i in active}
        free=int(state.goal_ids.shape[1])-len(active)
        for row in self.tasks.report()['tasks']:
            if free<=0:break
            if row['status']!='active' or row['task_id'] in names:continue
            task=self.tasks.read(row['task_id'])
            previous=next((r for r in reversed(records) if r.record_id==task.task_id),None)
            if previous is not None:continue
            meta=goal_metadata(priority=task.priority,interruption_cost=task.interruption_cost,
                acceptance=task.acceptance,source_kind=task.source_kind,evidence_ok=len(task.evidence_ids),
                evidence_open=len(task.open_preconditions()),can_act=task.autonomy=='can_act' and not task.open_preconditions())
            self.native.remember(dict(record_id=task.task_id,kind='task',text=json.dumps(self.tasks.context(task.task_id),ensure_ascii=False),
                source='task-store',known_at=task.updated_at,evidence_ids=task.evidence_ids,goal_metadata=meta))
            free-=1

    def ask(self,text,*,on_token=None,should_cancel=None):
        started=time.monotonic();accepted=self.resident.begin_request(text)
        intake=None
        if self.auto_intents:
            try:intake=self._intake(text,accepted['event_id'],accepted['known_at'])
            except (ValueError,KeyError) as exc:intake=dict(status='proposal_rejected',error=str(exc))
        # The current input and its brain/capsule are durable before decoding.
        self.save('request-accepted')
        preparation=time.monotonic()-started
        result=self.resident.respond(on_token=on_token,should_cancel=should_cancel)
        result['timing']['decoder_first_token_seconds']=result['timing']['first_token_seconds']
        if result['timing']['first_token_seconds'] is not None:result['timing']['first_token_seconds']+=preparation
        result['intake']=intake;result['timing']['end_to_end_seconds']=time.monotonic()-started
        return result

    def resume(self,*,on_token=None):
        # Resident resume consumes the durable identity and saved token prefix.
        return self.resident.resume(on_token=on_token)

    def decide(self,question,*,candidates=None,situation='',goal='',joint=False):
        self.native.prefill_message('user',question)
        method=self.api.joint if joint else self.api.decide
        result=method(question,candidates=candidates,situation=situation,goal=goal,commit_judgment=True)
        if joint:
            answer=result.get('language',{}).get('answer') or result.get('language_answer')
            if answer:self.native.prefill_message('assistant',answer)
        self.save('joint' if joint else 'decision')
        return result

    def spark(self,problem):
        result=self.api.spark(problem);self.save('spark');return result

    def read_more(self):
        """Continue an explicit memory read, retaining its source and unread pages."""
        import torch
        cursor=getattr(self.native,'read_continuation',None)
        if not cursor:return dict(status='complete',remaining=None)
        n=self.native;first=n._episode(n._frame('user',cursor['question'])+n._assistant_prefix,n.session.state)
        selected=n.live_memory.select(n,first,cursor['question'],4,cursor=cursor)
        old=n.capsule
        try:
            if selected['token_ids']:
                from role_conversation import ConversationEpisode
                ids=torch.cat(selected['token_ids'])[None]
                with torch.no_grad():n.capsule=ConversationEpisode(self.candidate.model,n.session.state,ids,old,**first.options).commit_observed_prefix()
            result=self.api.express('依据刚读取的原文页，继续处理这个问题，保留来源与尚未解决的缺口：'+cursor['question'],replay_budget=1)
        finally:n.capsule=old
        n.read_continuation=selected['continuation']
        if result['answer'].strip():n.remember(dict(record_id='reading-'+uuid.uuid4().hex,kind='judgment',text=result['answer'],source='model_reading',known_at=self.tasks.now()))
        self.save('continued-reading')
        return dict(status='more' if n.read_continuation else 'complete',read=selected['replayed_record_addresses'],answer=result['answer'],remaining=n.read_continuation)

    def learn(self,question,preferred,*,steps=1):
        from individual_learning import learn
        result=learn(self.native,question,preferred,steps=steps)
        self.save('individual-learning');result['generation_committed']=self.resident.generation.sequence
        return result

    def ingest(self,directory,*,stop_after_segments=None):
        from treyrin.material_intake import IntakeSession
        source=Path(directory).resolve();identity=hashlib.sha256(str(source).encode()).hexdigest()[:20]
        work=self.root/'materials'/identity
        intake=IntakeSession(source,work,segment_chars=2048)
        try:
            intake.ingest(stop_after_segments=stop_after_segments)
            count=0;observed=0;known={o.event_id for o in self.native.session.observations}
            for entry in intake.manifest.entries:
                for part in intake.segment_store.segments(entry.entry_id,entry.attempt):
                    eid='material-'+hashlib.sha256((str(source)+entry.sha256+part['segment_id']).encode()).hexdigest()
                    if eid in known:observed+=1;continue
                    payload=part['text']
                    self.native.observe(dict(event_id=eid,text=payload,source='document:'+str(source/entry.rel_path),
                        known_at=intake.now(),scope=json.dumps(dict(file_sha256=entry.sha256,location={k:v for k,v in part.items() if k not in ('text',)}),ensure_ascii=False)))
                    count+=1;observed+=1;known.add(eid)
                    if count%8==0:self.save('materials-progress')
            coverage=intake.inspection_report();coverage['model_observed_segments']=observed;coverage['newly_observed_segments']=count
            coverage['scope']='parsed source segments committed to the model; subsequent analysis is recorded separately'
            write(work/'COVERAGE.json',coverage);self.save('materials-ingested')
            return coverage
        finally:intake.close()

    def work(self,task_id,*,steps=8,allow_code=False):
        task=self.tasks.read(task_id)
        self.tasks.start(task_id)
        trace=[]
        for _ in range(steps):
            self._activate_tasks()
            control=self.api.work_control(dict(elapsed_seconds=0.,resource=1.))
            allocation=control.get('allocation') or {}
            action=allocation.get('action')
            allocated_goal=allocation.get('goal_identity')
            if allocation.get('allocated') and allocated_goal not in (None, task_id):
                trace.append(dict(control=control, blocked='goal_mismatch', requested_goal=task_id))
                self.save('work-blocked');break
            context=json.dumps(self.tasks.context(task_id),ensure_ascii=False)
            prompt=('根据当前委托和已经返回的真实工具结果，提出下一步操作。仅输出JSON。'
                '可选：{"tool":"read_text","path":"相对路径"}；{"tool":"list","path":"."}；'
                '{"tool":"write_text","path":"相对路径","text":"完整内容"}；'
                '{"tool":"python","code":"代码"}；{"tool":"finish","summary":"交付结果"}；'
                '{"tool":"wait","reason":"当前缺口"}。工具写入路径相对工作目录。'
                'finish时同时给出artifacts路径列表和已经完成的subtasks标识；只能根据真实回执确认完成。\n'+context+
                '\n当前神经目标与动作分配：'+json.dumps(control['allocation'],ensure_ascii=False))
            generated=self.api.express(prompt)
            operation=json_object(generated['answer']);tool=operation.get('tool')
            if tool in ('finish','wait'):
                if tool=='finish':
                    artifacts=[]
                    for name in operation.get('artifacts',[]):
                        p=self.tools.path(name)
                        artifacts.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
                    for sid in operation.get('subtasks',[]):self.tasks.progress(task_id,subtask=sid,status='done',note=operation.get('summary',''))
                    self.tasks.complete(task_id,note=operation.get('summary',''))
                    self._finish_goal(task_id)
                    operation['verified_artifacts']=artifacts
                trace.append(dict(operation=operation,control=control));break
            if action in ('wait','think','request_evidence','speak') or (tool in ('write_text','python') and action!='act'):
                trace.append(dict(operation=operation,control=control,blocked='neural_action_not_act',action=action))
                self.save('work-control')
                if action=='wait':break
                continue
            action_id='action-'+uuid.uuid4().hex
            result=self.tools.execute(operation,action_id=action_id,allow_code=allow_code)
            self.tasks.tool_receipt(task_id,tool=tool,action=tool,status='ok' if result['ok'] else 'failed',request_id=action_id,detail=result)
            trace.append(dict(operation=operation,result=result,control=control))
            self.save('tool-result')
        self.save('work-batch')
        return dict(task=self.tasks.read(task_id).as_dict(),trace=trace,workspace=str(self.workspace))

    def _finish_goal(self,task_id):
        record=next((r for r in reversed(self.native.session.mental_records) if r.record_id==task_id),None)
        if record:
            self.native.remember(dict(record_id=task_id+'-completed-'+uuid.uuid4().hex,kind=record.kind,
                text='已交付：'+record.text,status='completed',source='tool-executor',known_at=self.tasks.now(),
                evidence_ids=record.evidence_ids,revises=record.record_id,goal_metadata=record.goal_metadata))

class LocalTools:
    def __init__(self,app):self.app=app
    def path(self,value):
        p=(self.app.workspace/str(value)).resolve()
        if not p.is_relative_to(self.app.workspace):raise ValueError('Tool path leaves the configured work directory')
        return p
    def execute(self,spec,*,action_id,allow_code=False):
        journal=self.app.root/'tool-receipts';journal.mkdir(exist_ok=True)
        path=journal/(hashlib.sha256(action_id.encode()).hexdigest()+'.json')
        digest=hashlib.sha256(json.dumps(spec,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if path.exists():
            prior=json.loads(path.read_text(encoding='utf-8'))
            if prior['spec_sha256']!=digest:raise ValueError('Action identity reused with different arguments')
            if prior['state']=='executed':return prior['result']
            raise ValueError('Interrupted side effect needs reconciliation before replay: '+action_id)
        write(path,dict(state='executing',spec_sha256=digest,spec=spec))
        start=time.monotonic()
        try:
            tool=spec['tool']
            if tool=='read_text':
                p=self.path(spec['path']);value=p.read_text(encoding='utf-8-sig')
            elif tool=='list':
                p=self.path(spec.get('path','.'));value=[dict(name=x.name,directory=x.is_dir(),bytes=x.stat().st_size if x.is_file() else None) for x in p.iterdir()]
            elif tool=='write_text':
                p=self.path(spec['path']);p.parent.mkdir(parents=True,exist_ok=True)
                temp=p.with_name(p.name+'.tmp-'+uuid.uuid4().hex);temp.write_text(spec['text'],encoding='utf-8');os.replace(temp,p)
                value=dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            elif tool=='python':
                if not allow_code:raise ValueError('Enable --allow-code to execute model-proposed Python')
                completed=subprocess.run([sys.executable,'-c',spec['code']],cwd=self.app.workspace,capture_output=True,text=True,encoding='utf-8',errors='replace')
                value=dict(exit_code=completed.returncode,stdout=completed.stdout,stderr=completed.stderr)
                if completed.returncode:raise RuntimeError(json.dumps(value,ensure_ascii=False))
            else:raise ValueError('Unknown tool: '+str(tool))
            result=dict(ok=True,value=value,seconds=time.monotonic()-start)
        except Exception as exc:result=dict(ok=False,error=type(exc).__name__+': '+str(exc),seconds=time.monotonic()-start)
        write(path,dict(state='executed',spec_sha256=digest,result=result))
        self.app.native.observe(dict(event_id=action_id,text=json.dumps(result,ensure_ascii=False),source='tool_result',known_at=self.app.tasks.now()))
        return result

def main(argv=None):
    parser=argparse.ArgumentParser(description='特丽琳 Treyrin：同一模型、永久会话、决策与工作')
    parser.add_argument('--store',type=Path,default=ROOT/'resident-session')
    parser.add_argument('--workspace',type=Path)
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('status');commands.add_parser('chat');commands.add_parser('resume');commands.add_parser('read-more')
    ask=commands.add_parser('ask');ask.add_argument('text')
    for name in ('decide','joint'):
        p=commands.add_parser(name);p.add_argument('question');p.add_argument('--candidates',nargs='+');p.add_argument('--situation',default='');p.add_argument('--goal',default='')
    p=commands.add_parser('spark');p.add_argument('problem')
    p=commands.add_parser('ingest');p.add_argument('directory',type=Path)
    p=commands.add_parser('learn');p.add_argument('question');p.add_argument('preferred');p.add_argument('--steps',type=int,default=1)
    p=commands.add_parser('work');p.add_argument('task_id');p.add_argument('--steps',type=int,default=8);p.add_argument('--allow-code',action='store_true')
    args=parser.parse_args(argv)
    if args.command=='status':
        from treyrin.preassembly_session import Layout,startup_diagnostics
        print(json.dumps(startup_diagnostics(Layout(args.store)),ensure_ascii=False,indent=2,default=str));return 0
    app=Treyrin(args.store,workspace=args.workspace)
    try:
        if args.command=='chat':
            print('Treyrin 已加载；同一永久会话。/quit 保存退出，/resume 继续中断请求。')
            while True:
                try:text=input('你> ')
                except (EOFError,KeyboardInterrupt):break
                if text.strip()=='/quit':break
                try:
                    print('Treyrin> ',end='',flush=True);shown=['']
                    def stream(token,tokens):
                        value=app.candidate.tokenizer.decode(tokens,skip_special_tokens=True).rstrip('\ufffd')
                        if value.startswith(shown[0]):print(value[len(shown[0]):],end='',flush=True);shown[0]=value
                    result=app.resume(on_token=stream) if text.strip()=='/resume' else app.ask(text,on_token=stream)
                    if result['answer'].startswith(shown[0]):print(result['answer'][len(shown[0]):])
                    else:print('\n'+result['answer'])
                except KeyboardInterrupt:print('当前请求已保留，可使用 /resume。')
            app.save('exit');return 0
        if args.command=='ask':result=app.ask(args.text)
        elif args.command=='resume':result=app.resume()
        elif args.command in ('decide','joint'):result=app.decide(args.question,candidates=args.candidates,situation=args.situation,goal=args.goal,joint=args.command=='joint')
        elif args.command=='spark':result=app.spark(args.problem)
        elif args.command=='read-more':result=app.read_more()
        elif args.command=='ingest':result=app.ingest(args.directory)
        elif args.command=='learn':result=app.learn(args.question,args.preferred,steps=args.steps)
        elif args.command=='work':result=app.work(args.task_id,steps=args.steps,allow_code=args.allow_code)
        print(json.dumps(result,ensure_ascii=False,indent=2,default=str));return 0
    finally:app.close()

if __name__=='__main__':raise SystemExit(main())
