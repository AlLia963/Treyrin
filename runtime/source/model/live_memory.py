"""Durable pages behind the model's finite neural working set.

The core retains its trained read/selection computation. This adapter owns exact
payloads and an incremental token-address index; it never supplies an answer.
Evicted records and long records are ranked using the same learned query keys
plus token-address overlap, then read as bounded pages into inference scratch.

Read states: a read result carries `read_abstained` (neural abstain only), a typed
`read_outcome` (abstained / no_records / no_selection / inside_window / budget_rejected /
read / goal_focus) and the `replay_budget_rejected` list, so a budget rejection or an
empty store is never reported as abstention.

Status ownership: the sqlite metadata row is the single owner of a record's
status. Every write that changes status (creation with a declared retracted or
completed status, revision supersession, explicit transition) mirrors that
status to both paged experiences, and reopening reconciles the paged index from
the metadata owner. A status that cannot be reconciled raises instead of
silently diverging.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict
import hashlib,json,math,sqlite3
from pathlib import Path
import torch
from treyrin.paging_store import PagedExperienceStore,STATUSES,TRANSITIONS,StoreConfig

class RevisionTargetMissing(ValueError):
    """A mental record cites a revision target that this memory does not hold."""
    def __init__(self,target):
        self.target=str(target)
        super().__init__('Durable revision target missing: '+self.target)

def features(ids,space=8192):
    out=Counter()
    for t in ids:out[(int(t)*1103515245+12345)&(space-1)]+=.25
    for a,b in zip(ids,ids[1:]):out[(int(a)*214013+int(b)*2531011+137)&(space-1)]+=1.
    return {k:math.log1p(v) for k,v in out.items()}

class LiveMemory:
    def __init__(self,root,*,limit=None,page_units=None):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        config=StoreConfig(page_units=page_units) if page_units is not None else None
        self.store=PagedExperienceStore(self.root/'pages',config=config)
        self.db=sqlite3.connect(self.root/'ranking.sqlite')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS records(seq INTEGER PRIMARY KEY, address TEXT UNIQUE,
          payload TEXT, tokens TEXT, metadata TEXT, vector TEXT, norm REAL, position INTEGER,
          bank_kind TEXT, record_kind TEXT, source_kind TEXT, epistemic_kind TEXT, text TEXT);
        CREATE TABLE IF NOT EXISTS features(address TEXT, feature INTEGER, value REAL,
          PRIMARY KEY(address,feature));
        CREATE INDEX IF NOT EXISTS feature_lookup ON features(feature);
        CREATE TABLE IF NOT EXISTS frequency(feature INTEGER PRIMARY KEY, count INTEGER);
        ''')
        columns={r[1] for r in self.db.execute('PRAGMA table_info(records)')}
        for name in ('bank_kind','record_kind','source_kind','epistemic_kind','text'):
            if name not in columns:self.db.execute(f'ALTER TABLE records ADD COLUMN {name} TEXT')
        self.db.execute("UPDATE records SET bank_kind=substr(address,1,instr(address,':')-1) WHERE bank_kind IS NULL")
        self.db.execute("UPDATE records SET record_kind=json_extract(metadata,'$.kind') WHERE record_kind IS NULL")
        self.db.execute("UPDATE records SET source_kind=json_extract(metadata,'$.source') WHERE source_kind IS NULL")
        self.db.execute("UPDATE records SET epistemic_kind=CASE WHEN bank_kind='observation' THEN 'source_record' ELSE 'internal_record_not_fact' END WHERE epistemic_kind IS NULL")
        self.db.commit();self.limit=limit
        missing_text=self.db.execute('SELECT count(*) FROM records WHERE text IS NULL').fetchone()[0]
        if missing_text:
            with self.db:
                for address,payload in self.db.execute('SELECT address,payload FROM records WHERE text IS NULL').fetchall():
                    self.db.execute('UPDATE records SET text=? WHERE address=?',(self.store.read_text(payload),address))
        self._counts=Counter();self._largest=0
        for r in self.rows():
            self._counts[r[1].split(':')[0]]+=1
            self._largest=max(self._largest,self.store.record(r[3]).total_units)
        self._reconciled=self._reconcile_status_on_open()

    def count(self):
        return self.db.execute('SELECT count(*) FROM records'+(' WHERE seq<=?' if self.limit is not None else ''),
                               () if self.limit is None else (self.limit,)).fetchone()[0]

    def put(self,kind,event,ordinal,tokens,*,position,vector):
        address=f'{kind}:{ordinal}';meta=asdict(event);text=meta.pop('text')
        bank_kind='observation' if kind=='observation' else 'mental'
        record_kind='observation' if kind=='observation' else getattr(event,'kind',kind)
        source_kind='user_input' if kind=='observation' and event.source in ('human_user','user') else ('external_document' if kind=='observation' else ('model_output' if record_kind=='assistant_output' else 'internal_record'))
        epistemic_kind='source_record' if kind=='observation' else ('model_output_not_fact' if record_kind=='assistant_output' else 'internal_record_not_fact')
        status=getattr(event,'status','active')
        if status not in STATUSES:raise ValueError('Durable record status is not a declared status: '+str(status))
        meta.update(bank_kind=bank_kind,record_kind=record_kind,source_kind=source_kind,epistemic_kind=epistemic_kind,ordinal=ordinal,address=address,status=status)
        existing=self.db.execute('SELECT metadata,payload,tokens FROM records WHERE address=?',(address,)).fetchone()
        encoded=json.dumps(meta,ensure_ascii=False,sort_keys=True)
        if existing:
            if existing[0]!=encoded or self.store.read_text(existing[1])!=text or list(self.store.read_tokens(existing[2]))!=list(tokens):
                raise ValueError('Durable experience address reused with different content: '+address)
            seq=self.db.execute('SELECT seq FROM records WHERE address=?',(address,)).fetchone()[0]
            if self.limit is not None and seq>self.limit:
                self.limit=seq;self._counts[kind]+=1;self._largest=max(self._largest,len(tokens))
            return address
        target=getattr(event,'revises',None)
        prior=None
        if target:
            found=self.db.execute("SELECT payload,tokens,metadata FROM records WHERE json_extract(metadata,'$.record_id')=?",(target,)).fetchall()
            if not found:raise RevisionTargetMissing(target)
            if len(found)!=1:raise ValueError('Durable revision target is not unique: '+target)
            if json.loads(found[0][2]).get('status','active')!='active':raise ValueError('Durable revision target is not active: '+target)
            prior=found[0]
            for experience_id in (prior[0],prior[1]):
                if self.store.status(experience_id)!='active':
                    raise RuntimeError('Durable revision target status differs in the paged store: '+target)
        kw=dict(source_kind=source_kind,source=event.source or kind,known_at=event.known_at,
                parent_kind='event',parent_id=getattr(event,'event_id',getattr(event,'record_id',address)),
                meta=meta,status=status)
        body=self.store.admit_text(text,**kw).experience.experience_id
        code=self.store.admit_tokens(tokens,**kw).experience.experience_id
        f=features(tokens);norm=math.sqrt(sum(x*x for x in f.values())) or 1.
        with self.db:
            self.db.execute('INSERT INTO records(address,payload,tokens,metadata,vector,norm,position,bank_kind,record_kind,source_kind,epistemic_kind,text) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                (address,body,code,encoded,json.dumps(vector),norm,int(position),bank_kind,record_kind,source_kind,epistemic_kind,text))
            self.db.executemany('INSERT INTO features VALUES(?,?,?)',[(address,k,v) for k,v in f.items()])
            self.db.executemany('INSERT INTO frequency VALUES(?,1) ON CONFLICT(feature) DO UPDATE SET count=count+1',[(k,) for k in f])
            if target:
                updated=self.db.execute("UPDATE records SET metadata=json_set(metadata,'$.status','superseded','$.successor',?) WHERE json_extract(metadata,'$.record_id')=? AND json_extract(metadata,'$.status')='active'",(address,target))
                if updated.rowcount!=1:raise RuntimeError('Durable revision target changed during write: '+target)
        if prior is not None:
            self._mirror_status(prior[0],'superseded',by=body,reason='revision')
            self._mirror_status(prior[1],'superseded',by=body,reason='revision')
        self._counts[kind]+=1;self._largest=max(self._largest,len(tokens))
        if self.limit is not None:self.limit=self.db.execute('SELECT max(seq) FROM records').fetchone()[0]
        return address

    def rows(self):
        where=' WHERE seq<=?' if self.limit is not None else ''
        return self.db.execute('SELECT seq,address,payload,tokens,metadata,vector,norm,position FROM records'+where+' ORDER BY seq',
                               () if self.limit is None else (self.limit,)).fetchall()

    def record(self,address):
        row=self.db.execute('SELECT payload,tokens,metadata FROM records WHERE address=?'+(' AND seq<=?' if self.limit is not None else ''),
                            (address,) if self.limit is None else (address,self.limit)).fetchone()
        if row is None:raise KeyError(address)
        return dict(text=self.store.read_text(row[0]),token_ids=self.store.read_tokens(row[1]),**json.loads(row[2]))

    def status(self,address):
        row=self.db.execute('SELECT metadata FROM records WHERE address=?',(address,)).fetchone()
        if row is None:raise KeyError(address)
        return json.loads(row[0]).get('status','active')

    def transition(self,address,status,*,reason=''):
        """Change one record's status in the owner row and mirror both paged experiences."""
        if status not in STATUSES:raise ValueError('Durable record status is not a declared status: '+str(status))
        row=self.db.execute('SELECT payload,tokens,metadata FROM records WHERE address=?',(address,)).fetchone()
        if row is None:raise KeyError(address)
        meta=json.loads(row[2]);current=meta.get('status','active')
        if status==current:return current
        if status not in TRANSITIONS[current]:
            raise ValueError('Durable status transition not allowed: %s -> %s'%(current,status))
        meta['status']=status
        encoded=json.dumps(meta,ensure_ascii=False,sort_keys=True)
        with self.db:
            updated=self.db.execute('UPDATE records SET metadata=? WHERE address=?',(encoded,address))
            if updated.rowcount!=1:raise RuntimeError('Durable status update affected %d rows'%updated.rowcount)
        for experience_id in (row[0],row[1]):
            self._mirror_status(experience_id,status,by='live_memory',reason=reason or 'transition')
        return status

    def status_audit(self):
        """Return every record whose owner status differs from either paged experience."""
        where=' WHERE seq<=?' if self.limit is not None else ''
        mismatches=[]
        for address,payload,tokens,metadata in self.db.execute(
                'SELECT address,payload,tokens,metadata FROM records'+where+' ORDER BY seq',
                () if self.limit is None else (self.limit,)):
            status=json.loads(metadata).get('status','active')
            payload_status=self.store.status(payload);token_status=self.store.status(tokens)
            if payload_status!=status or token_status!=status:
                mismatches.append(dict(address=address,metadata=status,payload=payload_status,tokens=token_status))
        return mismatches

    def coverage(self,terms):
        """Typed per-term coverage over the stored source texts (owner copy).

        Returns record counts and addresses per term plus the terms with no record,
        so an absence question is answered by the store instead of by a four-record
        read window.
        """
        terms=[str(term) for term in terms]
        coverage={}
        for term in terms:
            if term in coverage:continue
            if self.limit is None:
                rows=self.db.execute('SELECT address FROM records WHERE instr(text,?)>0',(term,)).fetchall()
            else:
                rows=self.db.execute('SELECT address FROM records WHERE seq<=? AND instr(text,?)>0',
                                     (self.limit,term)).fetchall()
            coverage[term]=dict(records=len(rows),addresses=[row[0] for row in rows])
        return dict(terms=terms,coverage=coverage,
                    unknown_terms=[term for term in terms if coverage[term]['records']==0])

    def sync_status(self,records):
        """Apply core mental-record statuses to the owner rows (core -> paged copy)."""
        applied=[];skipped=[];blocked=[]
        for record in records:
            if isinstance(record,(tuple,list)):
                record_id=record[0] if len(record)>0 else None
                status=record[1] if len(record)>1 else None
            else:
                record_id=getattr(record,'record_id',None);status=getattr(record,'status',None)
            if not record_id or status not in STATUSES:
                blocked.append(dict(record_id=record_id,status=status,reason='invalid_record_status'));continue
            row=self.db.execute("SELECT address,metadata FROM records WHERE json_extract(metadata,'$.record_id')=?",
                                (record_id,)).fetchone()
            if row is None:
                skipped.append(dict(record_id=record_id,reason='not_in_paged_memory'));continue
            current=json.loads(row[1]).get('status','active')
            if current==status:
                skipped.append(dict(record_id=record_id,reason='already_current',status=status));continue
            if status not in TRANSITIONS[current]:
                blocked.append(dict(record_id=record_id,reason='illegal_transition',
                                    from_status=current,to_status=status));continue
            self.transition(row[0],status,reason='core-sync')
            applied.append(dict(record_id=record_id,address=row[0],from_status=current,to_status=status))
        return dict(applied=applied,skipped=skipped,blocked=blocked)

    def revision_chain(self,record_id):
        """Ordered record ids from record_id to the current head; raises on a cycle.

        Successor pointers are stored as durable addresses; this walks address ->
        record_id so the chain is expressed in logical record identities.
        """
        chain=[record_id];seen={record_id};current=record_id
        while True:
            row=self.db.execute(
                "SELECT json_extract(metadata,'$.successor') FROM records WHERE json_extract(metadata,'$.record_id')=?",
                (current,)).fetchone()
            if row is None:raise KeyError(current)
            successor=row[0]
            if not successor:return chain
            nxt=self.db.execute("SELECT json_extract(metadata,'$.record_id') FROM records WHERE address=?",
                                (successor,)).fetchone()
            if nxt is None:raise KeyError(str(successor))
            successor_id=nxt[0]
            if successor_id in seen:raise ValueError('Durable revision chain cycles at '+str(successor_id))
            seen.add(successor_id);chain.append(successor_id);current=successor_id

    def revision_head(self,record_id):
        """Current head of a multi-hop revision chain (cycle-safe)."""
        return self.revision_chain(record_id)[-1]

    def _mirror_status(self,experience_id,status,*,by='',reason=''):
        if self.store.status(experience_id)==status:return status
        try:
            return self.store.set_status(experience_id,status,by=by,reason=reason)
        except Exception as exc:
            raise RuntimeError('Paged store status mirror failed for %s (%s): %s'%(experience_id,status,exc)) from exc

    def _reconcile_status_on_open(self):
        """Make the metadata owner authoritative over the paged index when reopening."""
        repaired=[];blocked=[]
        for address,payload,tokens,metadata in self.db.execute(
                'SELECT address,payload,tokens,metadata FROM records ORDER BY seq'):
            status=json.loads(metadata).get('status','active')
            for role,experience_id in (('payload',payload),('tokens',tokens)):
                current=self.store.status(experience_id)
                if current==status:continue
                if status in TRANSITIONS.get(current,()):
                    self._mirror_status(experience_id,status,by='live_memory.reconcile',reason='open-reconcile')
                    repaired.append(dict(address=address,role=role,store_status=current,owner_status=status))
                else:
                    blocked.append(dict(address=address,role=role,store_status=current,owner_status=status))
        if blocked:
            raise RuntimeError('Durable memory status divergence cannot be reconciled: '+
                               json.dumps(blocked[:3],ensure_ascii=False))
        return repaired

    def needs_global_read(self,state,capacity):
        facts=set(int(x) for x in torch.cat((state.episode_event_ids,state.archive_event_ids),1)[0].tolist() if x>0)
        banks=[t for t in (state.workspace_ids,state.goal_ids,getattr(state,'mental_archive_ids',None)) if t is not None]
        mental=set(int(x) for x in torch.cat(banks,1)[0].tolist() if x>0)
        return self._counts['observation']>len(facts) or self._counts['mental']>len(mental) or self._largest>capacity

    def select(self,native,first,question,batch_size,*,excluded=(),cursor=None,coverage_terms=()):
        if type(batch_size)is not int or batch_size<1:raise ValueError('Positive per-batch read size required')
        rows=[r for r in self.rows() if r[1] not in set(excluded)]
        result=dict(token_ids=[],event_ordinals=[],selected_record_addresses=[],replayed_record_addresses=[],
                    read_abstained=False,read_outcome=None,replay_budget_rejected=[],paged=True,
                    continuation=None,abstain_mass=None)
        if coverage_terms:
            result['coverage']=self.coverage(coverage_terms)
        allocation=getattr(getattr(first,'output',None),'read_allocation',None)
        if allocation is not None and getattr(allocation,'numel',lambda:0)():
            mass=float(allocation[0,-1].float()[-1]);result['abstain_mass']=mass
            if mass>=0.5:
                result['read_abstained']=True;result['read_outcome']='abstained'
                return result
        if not rows:
            result['read_outcome']='no_records'
            return result
        c=native.candidate;core=c.model.core;q=first.output.scratch_working[:,-1]
        with torch.no_grad():
            fq=core.read_query(torch.cat((q,torch.zeros_like(q)),-1))[0].float().cpu()
            mq=core.workspace.query(q)[0].float().cpu()
        raw_query=c.tokenizer(question,add_special_tokens=False)['input_ids'];qf=features(raw_query)
        n=self.count()
        if self.limit is None:df={k:v for k,v in self.db.execute('SELECT feature,count FROM frequency')}
        else:df={k:v for k,v in self.db.execute('SELECT f.feature,count(*) FROM features f JOIN records r ON r.address=f.address WHERE r.seq<=? GROUP BY f.feature',(self.limit,))}
        idf={k:math.log1p(n)-math.log1p(df.get(k,0)) for k in qf}
        qnorm=math.sqrt(sum((v*idf[k])**2 for k,v in qf.items())) or 1.
        content=Counter();addresses={r[1] for r in rows}
        for k,v in qf.items():
            if idf[k]<=0:continue
            for address,tf in self.db.execute('SELECT address,value FROM features WHERE feature=?',(k,)):
                if address in addresses:content[address]+=tf*v*idf[k]**2
        scores=[]
        for r in rows:
            v=torch.tensor(json.loads(r[5]),dtype=torch.float32)
            query=fq if r[1].startswith('observation:') else mq
            semantic=float((v*query).sum())/max(1.,math.sqrt(v.numel())) if v.numel()==query.numel() else 0.
            scores.append(semantic+float(core.code_attention.gain)*content[r[1]]/(qnorm*r[6]))
        ranked=sorted(zip(rows,scores),key=lambda x:x[1],reverse=True)
        if cursor:
            rank_ids=cursor['addresses'];lookup={r[1]:r for r in rows}
            ranked=[(lookup[a],0.) for a in rank_ids if a in lookup]
        selected=ranked[:batch_size];remaining=[r[1] for r,_ in ranked[batch_size:]]
        pending=[(a,p) for a,p in (cursor or {}).get('pages',{}).items() if a in remaining]
        budget=max(1,native.window//2);pieces=[]
        for r,score in selected:
            meta=json.loads(r[4]);address=r[1];tokens=self.store.read_tokens(r[3])
            page=max(1,min(self.store.config.page_units,(native.window//2)//batch_size-192))
            chunks=[tokens[i:i+page] for i in range(0,len(tokens),page)]
            def overlap(part):
                f=features(part);return sum(f.get(k,0)*v*idf[k] for k,v in qf.items())
            requested=(cursor or {}).get('pages',{}).get(address)
            chosen=requested[0] if requested else max(range(len(chunks)),key=lambda i:overlap(chunks[i])) if chunks else 0
            if chosen>=len(chunks):continue
            unread=requested[1:] if requested else [i for i in range(len(chunks)) if i!=chosen]
            if unread:pending.append((address,unread))
            text=c.tokenizer.decode(chunks[chosen],skip_special_tokens=True) if chunks else ''
            payload=json.dumps(dict(source_address=address,page=chosen,pages=len(chunks),status=meta.get('status','active'),
                source=meta.get('source'),record_kind=meta.get('record_kind',meta.get('kind')),epistemic='source_record' if address.startswith('observation:') else 'internal_record_not_external_fact',text=text),ensure_ascii=False)
            frame=native._frame('tool',payload) if hasattr(native,'_frame') else c.tokenizer.apply_chat_template([dict(role='user',content=payload)],tokenize=False,add_generation_prompt=False,enable_thinking=False)
            ids=c.tokenizer(frame,add_special_tokens=False)['input_ids']
            if len(ids)>budget:
                pending=[(a,p) for a,p in pending if a!=address]+[(address,[chosen]+unread)]
                result['replay_budget_rejected'].append(dict(address=address,required_tokens=len(ids)));continue
            pieces.append((r[7],torch.tensor(ids,device=q.device,dtype=torch.long)));budget-=len(ids)
            k,num=address.split(':');a=dict(kind=k,ordinal=int(num),page=chosen,pages=len(chunks))
            result['selected_record_addresses'].append(a);result['replayed_record_addresses'].append(a)
            if k=='observation':result['event_ordinals'].append(int(num))
        result['token_ids']=[x for _,x in sorted(pieces,key=lambda x:x[0])]
        result['read_abstained']=False
        result['read_outcome']=('read' if result['token_ids'] else
                                ('budget_rejected' if result['replay_budget_rejected'] else 'no_selection'))
        next_addresses=[a for a,_ in pending]+remaining
        result['continuation']=dict(addresses=next_addresses,pages=dict(pending),question=question) if next_addresses else None
        return result

    def snapshot(self):
        rows=self.rows();seq=rows[-1][0] if rows else 0
        checksum=hashlib.sha256(json.dumps(rows,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        return dict(schema='treyrin-live-memory-v1',root=str(self.root),sequence=seq,records=len(rows),sha256=checksum)

    @classmethod
    def restore(cls,description):
        obj=cls(description['root'],limit=description['sequence'])
        if obj.snapshot()!=description:raise ValueError('Paged memory committed index differs from snapshot')
        return obj

    def close(self):self.store.close();self.db.close()