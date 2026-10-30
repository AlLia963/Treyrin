"""Local observe/judge/chat/save/reload entry for the integrated candidate.

The default entry owns ONE persisted role conversation: the four neural banks
(core experience state, KV/GDN native capsule, episodic codes, mental codes) and
the output profile are committed as a single generation by ResidentStore. An
explicit --model keeps the read-only snapshot path used by verification.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime,timezone
import copy
import json
from pathlib import Path
import uuid
import time
import queue
import threading
import math
from candidate import Candidate,ROOT
from resident_store import ConversationRepository


def attach_conversation(candidate,repository,*,window=8192,capacity=2048):
    """Restore the persisted conversation generation for this candidate, or start a new one."""
    if repository is None:
        from role_conversation import RoleConversationSession
        return RoleConversationSession(candidate,window=window,capacity=capacity,self_info=True),None
    return repository.open(candidate,window=window,capacity=capacity,self_info=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    pointer=ROOT/'CURRENT-MODEL.json'
    default_model=(ROOT/json.loads(pointer.read_text('utf-8'))['directory'] if pointer.exists() else ROOT/'model-ready')
    p.add_argument('--model',type=Path,help='Explicit snapshot; disables the default conversation generation')
    p.add_argument('--store',type=Path,help='Conversation store; defaults to the single local resident')
    p.add_argument('--question');p.add_argument('--candidates',nargs='*',default=[])
    p.add_argument('--max-new-tokens',type=int,default=None)
    p.add_argument('--verify-reload',action='store_true')
    p.add_argument('--verification-output',type=Path,help='Write the verification receipt to this file')
    p.add_argument('--autonomous',action='store_true',help='Offer idle-time opportunities; the model chooses operations')
    p.add_argument('--opportunity-seconds',type=float,default=5.)
    args=p.parse_args()
    if not math.isfinite(args.opportunity_seconds) or args.opportunity_seconds<=0:p.error('opportunity-seconds must be positive and finite')
    # A standard exported package is loaded by its own model entry, not by the
    # research candidate directory format.
    if args.model is not None and (Path(args.model)/'treyrin-manifest.json').is_file():
        if not args.question:raise ValueError('Question required with --model')
        import torch
        from modeling_treyrin import TreyrinForCausalLM
        from transformers import AutoTokenizer
        root=Path(args.model).resolve()
        model=TreyrinForCausalLM.from_pretrained(root,device='cpu',dtype='bfloat16',
            verify_assets='full',require_binding=True)
        tokenizer=AutoTokenizer.from_pretrained(root/'tokenizer',local_files_only=True)
        text=tokenizer.apply_chat_template([{'role':'user','content':args.question}],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tokenizer(text,return_tensors='pt')['input_ids']
        out=model.generate(input_ids=ids,max_new_tokens=args.max_new_tokens,
            eos_token_id=[248044,248046])
        answer=tokenizer.decode(out[0,ids.shape[1]:],skip_special_tokens=True)
        result=dict(answer=answer,finish_reason='stop',model=str(root),binding=model.binding)
        print(json.dumps(result,ensure_ascii=False));return result
    repository=(None if (args.model is not None or args.verify_reload)
        else ConversationRepository(args.store or ROOT/'resident'))
    with (repository.locked() if repository else nullcontext()):
        candidate=Candidate();conversation=None;generation=None;receipt=None
        if repository is None:
            model=args.model or default_model
            receipt=candidate.restore(model)
        else:
            model=None
            # A generation stores banks/profile, never random base weights.
            receipt=candidate.restore(default_model)
            conversation,generation=attach_conversation(candidate,repository)
        run(candidate,args,repository,conversation,receipt,generation)


def run(candidate,args,repository,conversation,receipt,generation=None):
    # Keep the previous Python invocation contract usable while the product
    # entry itself uses the role-correct permanent application.
    if generation is None and isinstance(conversation,dict) and hasattr(receipt,'save'):
        store=receipt
        if not args.question:raise ValueError('Legacy invocation requires an explicit question')
        candidate.observe(dict(event_id='user-'+uuid.uuid4().hex,text=args.question,
            source='用户发言（未独立验证）',known_at=datetime.now(timezone.utc).isoformat()))
        store.save(candidate)
        result=dict(answer=candidate.reply(args.question,candidates=args.candidates,max_new_tokens=args.max_new_tokens,viewport=2))
        store.save(candidate);print(json.dumps(result,ensure_ascii=False));return result
    def persist():
        if repository is not None:repository.save(candidate,conversation)

    def propose(text,count=2):
        """Propose on a discarded branch; commit only the typed hypothesis records.

        The branch keeps the neural proposal generation out of the committed
        experience stream; the retained records are re-committed through the
        role conversation so the capsule and the other three banks stay one
        generation.
        """
        branch=copy.copy(conversation.session);original=conversation.session
        candidate.session=branch
        try:result=candidate.propose(text,count=count,max_new_tokens=args.max_new_tokens)
        finally:candidate.session=original
        records=[]
        for record in result['proposals']:
            if not record.get('retained'):records.append(record);continue
            committed=conversation.remember(dict(record_id=record['record_id'],text=record['text'],
                kind='hypothesis',known_at=datetime.now(timezone.utc).isoformat(),source='model'))
            records.append(dict(record,committed_status=committed.get('status')))
        return dict(result,proposals=records,hypothesis_branch_discarded=True)

    def chat(question):
        result=conversation.turn(question,max_new_tokens=args.max_new_tokens)
        output=dict(answer=result['answer'],finish_reason=result['finish_reason'],diagnostics=result['diagnostics'])
        if args.candidates:output['native']=candidate.judge(question,args.candidates,viewport=2)
        persist()
        return output

    def ingest(text,event_id,source):
        from message_stream import text_chunks,source_identity
        encode=lambda s:candidate.tokenizer(s,add_special_tokens=False)['input_ids']
        count=tokens=0;known_at=datetime.now(timezone.utc).isoformat()
        for part in text_chunks(text,encode):
            conversation.observe(dict(event_id=f'{event_id}:part:{count:08d}',text=part['text'],source=source,
                known_at=known_at,scope=json.dumps(dict(message_id=event_id,start=part['start'],end=part['end']),
                    sort_keys=True,separators=(',',':'))))
            count+=1;tokens+=part['tokens']
        persist()
        return dict(message_id=event_id,source_sha256=source_identity(text),chunks=count,ingested_tokens=tokens,
            original_characters=len(text),usable_memory_accepted=False)

    def opportunity(elapsed):
        result=conversation.tick(dict(operation_id='idle-'+uuid.uuid4().hex,
            known_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=elapsed))
        if result['action']=='think':
            result['hypotheses']=propose('围绕当前目标和未解决问题提出可检验的假设；明确它们是待验证推测。',
                count=2)
        if result['action'] in ('speak','request_evidence'):
            result['answer']=candidate.reply('根据当前目标、经历和待验证问题，表达此刻有必要主动提出的内容；区分事实与推测。',
                max_new_tokens=args.max_new_tokens,viewport=0)
            conversation.remember(dict(record_id='initiative-'+uuid.uuid4().hex,text=result['answer'] or '[empty model output]',
                kind='assistant_output',known_at=datetime.now(timezone.utc).isoformat(),source='model'))
        if result['action']!='wait':persist();print(json.dumps(dict(initiative=result),ensure_ascii=False),flush=True)
        return result

    if args.verify_reload:
        model=args.model or (ROOT/json.loads((ROOT/'CURRENT-MODEL.json').read_text('utf-8'))['directory']
            if (ROOT/'CURRENT-MODEL.json').exists() else ROOT/'model-ready')
        check=(model/'verification.json' if (model/'verification.json').exists() else ROOT/'resident-check.json')
        expected=json.loads(check.read_text('utf-8'))
        actual=candidate.judge(expected['question'],expected['candidates'],viewport=expected.get('viewport',1))
        assert actual==expected['judgment'],'Fresh-process resident output differs'
        answer=candidate.reply(expected['question'],candidates=expected['candidates'],max_new_tokens=expected.get('max_new_tokens'),viewport=expected.get('viewport',1))
        assert answer==expected['answer'],'Fresh-process language output differs'
        result=dict(passed=True,restore=receipt,judgment_equal=True,language_equal=True,
                    model_directory=str(model.resolve()),observations=len(candidate.session.observations),
                    external_model_calls=0,external_retrieval_calls=0)
        (args.verification_output or ROOT/'reload-verification-v3.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result));return
    if args.question:
        if conversation is not None:result=chat(args.question)
        else:
            result=dict(answer=candidate.reply(args.question,candidates=args.candidates,
                                                max_new_tokens=args.max_new_tokens))
            if args.candidates:result['native']=candidate.judge(args.question,args.candidates)
        print(json.dumps(result,ensure_ascii=False));return
    print('Treyrin loaded. Commands: chat <text>, observe <JSON>, judge <JSON>, remember <typed JSON>, tick <JSON>, save <directory>, status, quit')
    inbox=None
    if getattr(args,'autonomous',False):
        inbox=queue.Queue()
        def read_input():
            while True:
                try:line=input()
                except (EOFError,KeyboardInterrupt):inbox.put('quit');return
                inbox.put(line)
                if line.strip()=='quit':return
        threading.Thread(target=read_input,daemon=True).start()
    last_opportunity=time.monotonic();last_saved=last_opportunity
    while True:
        try:line=(inbox.get(timeout=args.opportunity_seconds) if inbox else input())
        except queue.Empty:
            now=time.monotonic();opportunity(now-last_opportunity);last_opportunity=now
            if last_opportunity-last_saved>=60.:persist();last_saved=last_opportunity
            continue
        except (EOFError,KeyboardInterrupt):persist();return
        if line.strip()=='quit':persist();return
        try:
            command,_,payload=line.partition(' ')
            if command=='observe':result=conversation.observe(json.loads(payload));persist()
            elif command=='judge':
                request=json.loads(payload);result=candidate.judge(request['question'],request['candidates'])
            elif command=='chat':result=chat(payload)
            elif command=='remember':result=conversation.remember(json.loads(payload));persist()
            elif command=='tick':result=conversation.tick(json.loads(payload));persist()
            elif command=='propose':result=propose(payload);persist()
            elif command=='ingest-file':
                request=json.loads(payload);path=Path(request['path']).resolve()
                text=path.read_text(encoding=request.get('encoding','utf-8'))
                result=ingest(text,request.get('message_id','file-'+uuid.uuid4().hex),request.get('source',str(path)))
            elif command=='status':result=candidate.inspect_structure()
            elif command=='identity':
                from self_description import structure_record
                result=dict(candidate.self_description())
                if conversation is not None:
                    result['committed_observation']=conversation.observe(structure_record(candidate));persist()
            elif command=='save':result=candidate.save(Path(payload))
            else:result={'error':'Unknown command'}
        except (ValueError,KeyError,TypeError,OSError) as exc:
            session=getattr(candidate,'session',None)
            if session is not None and hasattr(session,'_check'):
                try:session._check()
                except Exception as invalid:
                    print(json.dumps(dict(error=str(exc),state_error=str(invalid),restore_required=True),ensure_ascii=False),flush=True)
                    return
            result=dict(error_type=type(exc).__name__,error=str(exc),conversation_continues=True)
        print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
