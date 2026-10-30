"""Crash-safe continuation pointer for one local conversation.

Snapshots retain chronology. Identical immutable payloads share local storage;
no event is discarded, summarized or inferred by this persistence layer.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import uuid

GENERATION_SCHEMA='treyrin-resident-generation-v1'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class ResidentStore:
    def __init__(self,directory):self.root=Path(directory).resolve()

    def read(self):
        pointer=self.root/'CURRENT.json'
        if not pointer.exists():return None
        record=json.loads(pointer.read_text('utf-8'))
        target=(self.root/record['directory']).resolve()
        if not target.is_relative_to(self.root):raise ValueError('Resident pointer escapes its store')
        if digest(target/'candidate-manifest.json')!=record['manifest_sha256']:
            raise ValueError('Resident snapshot identity differs')
        return record,target

    def save(self,candidate,role_session=None):
        self.root.mkdir(parents=True,exist_ok=True)
        previous=self.read();sequence=previous[0]['sequence']+1 if previous else 1
        conversation=previous[0]['conversation_id'] if previous else str(uuid.uuid4())
        directory=self.root/'snapshots'/f'{sequence:012d}-{uuid.uuid4().hex[:8]}'
        candidate.save(directory)
        if role_session is not None:
            role_session.save(directory/'role-session')
            manifest=json.loads((directory/'candidate-manifest.json').read_text('utf-8'))
            manifest['files']={p.relative_to(directory).as_posix():digest(p)
                for p in sorted(directory.rglob('*')) if p.is_file()}
            manifest['role_session']=True
            (directory/'candidate-manifest.json').write_text(json.dumps(manifest,sort_keys=True)+chr(10),encoding='utf-8')
        manifest=json.loads((directory/'candidate-manifest.json').read_text('utf-8'))
        if previous:
            old_manifest=json.loads((previous[1]/'candidate-manifest.json').read_text('utf-8'))
            for name,expected in manifest['files'].items():
                old=previous[1]/name;new=(directory/name).resolve()
                if not new.is_relative_to(directory.resolve()):raise ValueError('Snapshot payload escapes')
                if old_manifest['files'].get(name)==expected and digest(old)==expected:
                    # Only our newly created immutable copy is replaced.
                    # The old committed snapshot and pointer remain intact.
                    new.unlink();os.link(old,new)
        record=dict(schema='treyrin-resident-store-v1',conversation_id=conversation,
            sequence=sequence,directory=directory.relative_to(self.root).as_posix(),
            manifest_sha256=digest(directory/'candidate-manifest.json'),
            role_session=bool(manifest.get('role_session')))
        temporary=self.root/('CURRENT-'+uuid.uuid4().hex+'.tmp')
        with temporary.open('w',encoding='utf-8') as f:
            json.dump(record,f,ensure_ascii=False,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(temporary,self.root/'CURRENT.json')
        return record

    @contextmanager
    def locked(self):
        import msvcrt
        self.root.mkdir(parents=True,exist_ok=True)
        with (self.root/'resident.lock').open('a+b') as handle:
            if handle.tell()==0:handle.write(b'0');handle.flush()
            handle.seek(0)
            try:msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            except OSError:raise RuntimeError('This conversation is already open in another process') from None
            try:yield self
            finally:
                handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)

    # -- conversation generations: profile and the four banks, one commit --

    def read_generation(self):
        pointer=self.root/'CURRENT-CONVERSATION.json'
        if not pointer.exists():return None
        record=json.loads(pointer.read_text('utf-8'))
        if record.get('schema')!=GENERATION_SCHEMA:raise ValueError('Unknown conversation generation schema')
        target=(self.root/record['directory']).resolve()
        if not target.is_relative_to(self.root):raise ValueError('Generation pointer escapes its store')
        if digest(target/'manifest.json')!=record['manifest_sha256']:
            raise ValueError('Conversation generation manifest differs')
        return record,target

    @staticmethod
    def _link_identical(directory,previous):
        """Share byte-identical payloads with the previous generation, never the pointer."""
        if previous is None:return
        manifest=json.loads((previous[1]/'manifest.json').read_text('utf-8'))
        for name,expected in manifest.get('files',{}).items():
            new=directory/name;old=previous[1]/name
            if not new.is_file()or not old.is_file():continue
            if digest(new)!=expected or digest(old)!=expected:continue
            new.unlink();os.link(old,new)

    def save_generation(self,candidate,session):
        """Commit the profile and the four banks as ONE conversation generation.

        The four banks are the core experience state, the KV/GDN native capsule,
        the episodic code bank and the mental code bank. They are written from a
        single session object only after their cross-checks pass, then published
        through one atomic pointer replace. A partial or mixed generation is
        never visible to a reader, and the old generation stays intact.
        """
        import torch
        from safetensors.torch import save_file
        from role_conversation import RoleConversationSession
        if getattr(session,'candidate',None)is not candidate:
            raise ValueError('A generation must be saved from a session bound to this candidate')
        state=session.session.state
        if state.binding!=candidate.model._binding:
            raise ValueError('Core state belongs to different trained parameters')
        if session.capsule.encoder_binding!=candidate.model._binding:
            raise ValueError('Native capsule belongs to different trained parameters')
        session.codes.check(state);session.mental_codes.check(state)
        adapter=getattr(candidate.model.body,'output_adapter',None)
        if adapter is None:raise ValueError('Profile persistence requires the output-only adapter')
        self.root.mkdir(parents=True,exist_ok=True)
        previous=self.read_generation()
        sequence=previous[0]['sequence']+1 if previous else 1
        conversation=previous[0]['conversation_id'] if previous else str(uuid.uuid4())
        directory=self.root/'generations'/f'{sequence:012d}-{uuid.uuid4().hex[:8]}'
        directory.mkdir(parents=True,exist_ok=False)
        session.save(directory/'conversation')
        save_file({'A':adapter.A.detach().cpu().clone().contiguous(),
                   'B':adapter.B.detach().cpu().clone().contiguous()},directory/'profile.safetensors')
        moments={}
        for name,parameter in (('A',adapter.A),('B',adapter.B)):
            saved=candidate.optimizer.state.get(parameter)
            moments[name]=({key:(value.detach().cpu().clone()if torch.is_tensor(value)else value)
                            for key,value in saved.items()}if saved else {})
        torch.save(moments,directory/'profile-moments.pt')
        self._link_identical(directory,previous)
        manifest=dict(schema=GENERATION_SCHEMA,sequence=sequence,conversation_id=conversation,
            parameter_binding=candidate.model._binding,
            conversation_kind=('role-conversation'if isinstance(session,RoleConversationSession)else'native-replay'),
            conversation_manifest_sha256=digest(directory/'conversation'/'manifest.json'),
            role_state_sha256=(digest(directory/'conversation'/'role-state.json')
                if (directory/'conversation'/'role-state.json').exists()else None),
            profile_sha256=digest(directory/'profile.safetensors'),
            profile_moments_sha256=digest(directory/'profile-moments.pt'),
            profile_tensors=['body.output_adapter.A','body.output_adapter.B'],
            optimizer_steps=int(session.session.optimizer_steps),
            four_banks=dict(native_tokens=int(session.capsule.seen),window=int(session.window),
                capacity=int(getattr(session,'capacity',0)),
                observations=len(session.session.observations),
                mental_records=len(session.session.mental_records),
                cognitive_ticks=int(state.cognitive_ticks[0]),writes=int(state.writes[0]),
                episode_code_tensors=len(session.codes.tensors()),
                mental_code_tensors=len(session.mental_codes.tensors())))
        manifest['files']={path.relative_to(directory).as_posix():digest(path)
                           for path in sorted(directory.rglob('*'))if path.is_file()}
        (directory/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2),encoding='utf-8')
        record=dict(schema=GENERATION_SCHEMA,sequence=sequence,conversation_id=conversation,
            directory=directory.relative_to(self.root).as_posix(),manifest_sha256=digest(directory/'manifest.json'))
        temporary=self.root/('CURRENT-CONVERSATION-'+uuid.uuid4().hex+'.tmp')
        with temporary.open('w',encoding='utf-8')as handle:
            json.dump(record,handle,ensure_ascii=False,sort_keys=True);handle.write('\n')
            handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,self.root/'CURRENT-CONVERSATION.json')
        return record

    def check_generation(self,record,directory,candidate,session):
        """Require one generation: profile weights, four banks and counters agree."""
        import torch
        from safetensors.torch import load_file
        directory=Path(directory)
        manifest=json.loads((directory/'manifest.json').read_text('utf-8'))
        if manifest.get('schema')!=GENERATION_SCHEMA:raise ValueError('Unknown conversation generation schema')
        if manifest['sequence']!=record['sequence']or manifest['conversation_id']!=record['conversation_id']:
            raise ValueError('Cross-generation conversation pointer refused')
        if manifest['parameter_binding']!=candidate.model._binding:
            raise ValueError('Conversation generation belongs to different trained parameters')
        for name,expected in manifest['files'].items():
            path=(directory/name).resolve()
            if not path.is_relative_to(directory.resolve())or digest(path)!=expected:
                raise ValueError('Conversation generation payload differs: '+name)
        values=load_file(directory/'profile.safetensors',device='cpu')
        adapter=getattr(candidate.model.body,'output_adapter',None)
        if adapter is None:raise ValueError('Generation carries a profile but this body has no output adapter')
        for name,tensor in (('A',adapter.A),('B',adapter.B)):
            if (name not in values or values[name].shape!=tensor.shape
                    or not torch.equal(values[name],tensor.detach().cpu())):
                raise ValueError('Profile weights differ from the restored parameters; cross-generation profile refused')
        state=session.session.state;banks=manifest['four_banks']
        if int(state.writes[0])!=banks['writes']or len(session.session.observations)!=banks['observations']:
            raise ValueError('Core experience bank disagrees with the committed generation')
        if int(state.workspace_seen[0])!=banks['mental_records']or len(session.session.mental_records)!=banks['mental_records']:
            raise ValueError('Mental record bank disagrees with the committed generation')
        if int(state.cognitive_ticks[0])!=banks['cognitive_ticks']:
            raise ValueError('Cognitive opportunity bank disagrees with the committed generation')
        if session.capsule.seen!=banks['native_tokens']:
            raise ValueError('Native KV/GDN capsule count disagrees with the committed generation')
        session.codes.check(state);session.mental_codes.check(state)
        return manifest


class ConversationRepository:
    """Open/save one persisted role conversation generation for the default CLI."""

    def __init__(self,directory):self.store=ResidentStore(directory)

    def open(self,candidate,*,window=8192,capacity=2048,system_prompt=None,self_info=False):
        from role_conversation import RoleConversationSession
        saved=self.store.read_generation()
        if saved is None:
            return RoleConversationSession(candidate,window=window,capacity=capacity,
                system_prompt=system_prompt,self_info=self_info),None
        record,directory=saved
        session=RoleConversationSession.restore(directory/'conversation',candidate)
        self.store.check_generation(record,directory,candidate,session)
        return session,record

    def save(self,candidate,session):return self.store.save_generation(candidate,session)

    @contextmanager
    def locked(self):
        with self.store.locked():yield self
