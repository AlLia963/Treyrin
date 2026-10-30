"""Protected, explicitly non-factual language codes for native mental state.

The core's physical workspace/goal decisions are authoritative. This bank
cannot invent a goal, turn a hypothesis into observation, or recover a payload
from a text journal. Old nonempty states require an explicit migration.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class MentalCodes:
    workspace:torch.Tensor
    workspace_lengths:torch.Tensor
    workspace_ids:torch.Tensor
    workspace_positions:torch.Tensor
    goals:torch.Tensor
    goal_lengths:torch.Tensor
    goal_ids:torch.Tensor
    goal_positions:torch.Tensor
    binding:str
    archive:torch.Tensor|None=None
    archive_lengths:torch.Tensor|None=None
    archive_ids:torch.Tensor|None=None
    archive_positions:torch.Tensor|None=None

    @classmethod
    def empty(cls,state,capacity=2048):
        if type(capacity)is not int or capacity<1:raise ValueError('Explicit mental-code capacity required')
        if bool((state.workspace_ids>0).any())or bool((state.goal_ids>0).any())or bool((state.workspace_seen!=0).any()):
            raise ValueError('Mental-code boot requires empty non-fact history')
        def bank(ids):
            b,s=ids.shape
            return (torch.zeros(b,s,capacity,dtype=torch.int32,device=ids.device),torch.zeros_like(ids),
                    ids.clone(),torch.full_like(ids,-1))
        cold=getattr(state,'mental_archive_ids',None)
        if cold is not None and bool((cold>0).any()):raise ValueError('Archive boot requires empty non-fact history')
        extra={}if cold is None else dict(zip(('archive','archive_lengths','archive_ids','archive_positions'),bank(cold)))
        return cls(*bank(state.workspace_ids),*bank(state.goal_ids),state.binding,**extra)

    def check(self,state):
        if self.binding!=state.binding:raise ValueError('Mental-code representation binding differs')
        banks=[
            (self.workspace,self.workspace_lengths,self.workspace_ids,self.workspace_positions,state.workspace_ids),
            (self.goals,self.goal_lengths,self.goal_ids,self.goal_positions,state.goal_ids)]
        cold=getattr(state,'mental_archive_ids',None)
        if cold is not None:banks.append((self.archive,self.archive_lengths,self.archive_ids,self.archive_positions,cold))
        elif any(getattr(self,n)is not None for n in('archive','archive_lengths','archive_ids','archive_positions')):raise ValueError('Unexpected mental-code archive')
        for codes,lengths,ids,positions,expected in banks:
            if not all(isinstance(t,torch.Tensor)for t in(codes,lengths,ids,positions)):raise ValueError('Missing mental-code archive')
            if codes.dtype!=torch.int32 or codes.ndim!=3 or codes.shape[:2]!=ids.shape:raise ValueError('Mental-code schema differs')
            if any(t.dtype!=torch.long or t.shape!=ids.shape for t in(lengths,ids,positions)):raise ValueError('Mental-code identity/length schema differs')
            if not torch.equal(ids,expected):raise ValueError('Mental-code physical identities differ')
            if not torch.equal(lengths>0,ids>0)or not torch.equal(positions>=0,ids>0):raise ValueError('Mental-code occupancy differs')
            if bool(((lengths<0)|(lengths>codes.shape[-1])).any())or bool((codes<0).any()):raise ValueError('Invalid mental-code payload')

    def commit(self,before,after,tokens,*,native_start):
        self.check(before)
        if after.binding!=self.binding or not torch.equal(after.workspace_seen,before.workspace_seen+1):
            raise ValueError('One actual mental transition required')
        if not torch.equal(after.writes,before.writes):raise ValueError('Mental event cannot increment factual writes')
        if tokens.ndim!=2 or tokens.dtype!=torch.long or tokens.shape[0]!=self.workspace.shape[0]or not 0<tokens.shape[1]<=self.workspace.shape[-1]:
            raise ValueError('Mental codes must fit explicitly; never truncate')
        if type(native_start)is not int or native_start<0 or bool((tokens<0).any())or bool((tokens>torch.iinfo(torch.int32).max).any()):raise ValueError('Actual native position and valid tokens required')
        def advance(codes,lengths,ids,positions,newids):
            change=ids!=newids
            if bool((change&(newids!=after.workspace_seen[:,None])).any()):raise ValueError('Mental write does not match actual new identity')
            payload=torch.zeros(codes.shape[0],codes.shape[-1],dtype=torch.int32,device=codes.device)
            payload[:,:tokens.shape[-1]]=tokens.to(torch.int32)
            return (torch.where(change[:,:,None],payload[:,None],codes),
                    torch.where(change,torch.full_like(lengths,tokens.shape[-1]),lengths),newids.clone(),
                    torch.where(change,torch.full_like(positions,native_start),positions))
        extra={}if self.archive is None else dict(zip(('archive','archive_lengths','archive_ids','archive_positions'),
            advance(self.archive,self.archive_lengths,self.archive_ids,self.archive_positions,after.mental_archive_ids)))
        result=MentalCodes(*advance(self.workspace,self.workspace_lengths,self.workspace_ids,self.workspace_positions,after.workspace_ids),
            *advance(self.goals,self.goal_lengths,self.goal_ids,self.goal_positions,after.goal_ids),self.binding,**extra)
        result.check(after);return result

    def bank(self,state,*,compact=False):
        self.check(state)
        if self.workspace.shape[0]!=1:raise ValueError('One native conversation required')
        if compact:
            occupied=torch.cat((self.workspace_ids,self.goal_ids),1)
            duplicate=(self.workspace_ids[:,:,None]==self.goal_ids[:,None,:]).any(-1)&(self.workspace_ids>0)
            banks=[(self.workspace,self.workspace_lengths.masked_fill(duplicate,0),self.workspace_ids,self.workspace_positions,state.workspace_statuses,state.workspace_kinds,False),
                (self.goals,self.goal_lengths,self.goal_ids,self.goal_positions,state.goal_statuses,torch.full_like(self.goal_ids,4),True)]
            if self.archive is not None:
                dup=(self.archive_ids[:,:,None]==occupied[:,None,:]).any(-1)
                banks.append((self.archive,self.archive_lengths.masked_fill(dup,0),self.archive_ids,self.archive_positions,state.mental_archive_statuses,state.mental_archive_kinds,False))
            pieces={k:[] for k in ('codes','lengths','ids','positions','statuses','kinds','protected','physical_indices')};offset=0
            for codes,lengths,ids,positions,statuses,kinds,protected in banks:
                index=(lengths[0]>0).nonzero().flatten()
                for name,value in (('codes',codes),('lengths',lengths),('ids',ids),('positions',positions),('statuses',statuses),('kinds',kinds)):
                    pieces[name].append(value[0,index])
                pieces['protected'].append(torch.full_like(index,protected,dtype=torch.bool))
                pieces['physical_indices'].append(index+offset);offset+=ids.shape[1]
            return {k:torch.cat(v,0) for k,v in pieces.items()}
        # A protected goal can outlive its workspace slot. If it occupies both,
        # retain one address, with the protected physical goal authoritative.
        duplicate=(self.workspace_ids[:,:,None]==self.goal_ids[:,None,:]).any(-1)&(self.workspace_ids>0)
        lengths=torch.cat((self.workspace_lengths.masked_fill(duplicate,0),self.goal_lengths),1)[0]
        result=dict(codes=torch.cat((self.workspace,self.goals),1)[0],lengths=lengths,
            ids=torch.cat((self.workspace_ids,self.goal_ids),1)[0],
            positions=torch.cat((self.workspace_positions,self.goal_positions),1)[0],
            statuses=torch.cat((state.workspace_statuses,state.goal_statuses),1)[0],
            kinds=torch.cat((state.workspace_kinds,torch.full_like(state.goal_ids,4)),1)[0],
            protected=torch.cat((torch.zeros_like(state.workspace_ids,dtype=torch.bool),torch.ones_like(state.goal_ids,dtype=torch.bool)),1)[0])
        if self.archive is not None:
            # Every identity has one active neural address. Live goal rows are
            # authoritative, followed by workspace; the archive keeps old rows.
            dup=(self.archive_ids[:,:,None]==torch.cat((self.workspace_ids,self.goal_ids),1)[:,None,:]).any(-1)
            extras=dict(codes=self.archive[0],lengths=self.archive_lengths.masked_fill(dup,0)[0],
                ids=self.archive_ids[0],positions=self.archive_positions[0],statuses=state.mental_archive_statuses[0],
                kinds=state.mental_archive_kinds[0],protected=torch.zeros_like(self.archive_ids[0],dtype=torch.bool))
            result={k:torch.cat((v,extras[k]),0)for k,v in result.items()}
        return result

    def tensors(self):return {n:getattr(self,n)for n in self.__dataclass_fields__ if n!='binding'and getattr(self,n)is not None}
