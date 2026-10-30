"""Model-state episodic codes aligned to the core's actual physical decisions.

No document search or semantic rule chooses a source. The recurrent core's
candidate/read allocation selects retained observed codes. Full public source
metadata travels with each episode; replay is not a new observed event.
"""
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class EpisodeCodes:
    hot:torch.Tensor
    hot_lengths:torch.Tensor
    hot_ids:torch.Tensor
    cold:torch.Tensor
    cold_lengths:torch.Tensor
    cold_ids:torch.Tensor
    binding:str
    hot_native_start:torch.Tensor|None=None
    cold_native_start:torch.Tensor|None=None

    @classmethod
    def empty(cls,state,capacity):
        if type(capacity)is not int or capacity<1:raise ValueError('Explicit positive per-event code capacity required')
        if bool(state.occupied.any())or bool(state.archive_occupied.any()):raise ValueError('Code-bank boot requires genuinely empty core memory')
        device=state.keys.device;b,e=state.episode_event_ids.shape;a=state.archive_event_ids.shape[1]
        return cls(torch.zeros(b,e,capacity,dtype=torch.int32,device=device),torch.zeros(b,e,dtype=torch.long,device=device),
            state.episode_event_ids.clone(),torch.zeros(b,a,capacity,dtype=torch.int32,device=device),
            torch.zeros(b,a,dtype=torch.long,device=device),state.archive_event_ids.clone(),state.binding,
            torch.full_like(state.episode_event_ids,-1),torch.full_like(state.archive_event_ids,-1))

    def check(self,state):
        if self.binding!=state.binding:raise ValueError('Episode code encoder/core binding differs')
        if not torch.equal(self.hot_ids,state.episode_event_ids)or not torch.equal(self.cold_ids,state.archive_event_ids):
            raise ValueError('Physical episode identities are not aligned')
        for codes,lengths,ids in [(self.hot,self.hot_lengths,self.hot_ids),(self.cold,self.cold_lengths,self.cold_ids)]:
            if codes.dtype!=torch.int32 or lengths.dtype!=torch.long or ids.dtype!=torch.long or codes.ndim!=3 or codes.shape[:2]!=lengths.shape or lengths.shape!=ids.shape:
                raise ValueError('Episodic token-code schema differs')
            if bool(((lengths<0)|(lengths>codes.shape[-1])).any())or not torch.equal(lengths>0,ids>0):
                raise ValueError('Episodic code occupancy/length differs')
            if bool((codes<0).any()):raise ValueError('Negative episodic token code')
        for positions,ids in [(self.hot_native_start,self.hot_ids),(self.cold_native_start,self.cold_ids)]:
            if positions is None or positions.dtype!=torch.long or positions.shape!=ids.shape or not torch.equal(positions>=0,ids>0):
                raise ValueError('Episodic native stream positions differ')

    def commit(self,before,after,tokens,*,native_start=None):
        self.check(before)
        if after.binding!=self.binding or not torch.equal(after.writes,before.writes+1):raise ValueError('One causal core observation transition required')
        if tokens.ndim!=2 or tokens.shape[0]!=self.hot.shape[0]or tokens.shape[1]<1 or tokens.shape[1]>self.hot.shape[-1]:
            raise ValueError('Observed code stream must fit explicitly; never truncate silently')
        if tokens.dtype!=torch.long or bool((tokens<0).any())or bool((tokens>torch.iinfo(torch.int32).max).any()):raise ValueError('Observed token codes must be valid long IDs')
        changed_hot=after.episode_event_ids!=self.hot_ids
        native_start=int(before.writes[0])if native_start is None else native_start
        if type(native_start)is not int or native_start<0:raise ValueError('Actual causal native position required')
        if bool((changed_hot&(after.episode_event_ids!=after.writes[:,None])).any()):raise ValueError('Hot write is not the current observed event')
        packed=torch.zeros(self.hot.shape[0],self.hot.shape[-1],dtype=torch.int32,device=self.hot.device)
        packed[:,:tokens.shape[1]]=tokens.to(torch.int32)
        hot=torch.where(changed_hot[:,:,None],packed[:,None],self.hot)
        lengths=torch.where(changed_hot,torch.full_like(self.hot_lengths,tokens.shape[1]),self.hot_lengths)
        changed_cold=after.archive_event_ids!=self.cold_ids
        # Cold promotion can only copy an ACTUALLY evicted hot episode, not
        # regenerate a record from its label or from some external journal.
        matches=after.archive_event_ids[:,:,None]==self.hot_ids[:,None,:]
        if bool((changed_cold&((matches.sum(-1)!=1)|(after.archive_event_ids<1))).any()):raise ValueError('Cold promotion does not identify one prior hot event')
        old_slot=matches.long().argmax(-1)
        old_codes=self.hot.gather(1,old_slot[:,:,None].expand(-1,-1,self.hot.shape[-1]))
        old_lengths=self.hot_lengths.gather(1,old_slot)
        cold=torch.where(changed_cold[:,:,None],old_codes,self.cold)
        cold_lengths=torch.where(changed_cold,old_lengths,self.cold_lengths)
        hot_positions=torch.where(changed_hot,torch.full_like(self.hot_native_start,native_start),self.hot_native_start)
        cold_positions=torch.where(changed_cold,self.hot_native_start.gather(1,old_slot),self.cold_native_start)
        result=EpisodeCodes(hot,lengths,after.episode_event_ids.clone(),cold,cold_lengths,after.archive_event_ids.clone(),self.binding,hot_positions,cold_positions)
        result.check(after);return result

    def read(self,state,attention,allocation,*,budget=3,before_position=None):
        self.check(state)
        if self.hot.shape[0]!=1:raise ValueError('One conversation replay at a time')
        if type(budget)is not int or not 1<=budget<=8:raise ValueError('Bounded neural replay budget 1..8 required')
        values=torch.cat((self.hot,self.cold),1)[0];lengths=torch.cat((self.hot_lengths,self.cold_lengths),1)[0]
        identities=torch.cat((self.hot_ids,self.cold_ids),1)[0]
        positions=torch.cat((self.hot_native_start,self.cold_native_start),1)[0]
        eligible=lengths>0
        if before_position is not None:
            if type(before_position)is not int:raise ValueError('Explicit native cache position boundary required')
            eligible=eligible&(positions<before_position)
        if attention.ndim!=2 or attention.shape[1]!=len(values)or allocation.shape!=(attention.shape[0]+1,):raise ValueError('Actual candidate/read surfaces must align')
        if not torch.isfinite(attention).all()or not torch.isfinite(allocation).all()or bool((attention<0).any())or bool((allocation<0).any()):raise ValueError('Invalid neural read allocation')
        winner=int(allocation.argmax())
        if winner==len(allocation)-1 or not bool(eligible.any()):return dict(token_ids=[],event_ordinals=[],read_abstained=winner==len(allocation)-1)
        weights=attention[winner].masked_fill(~eligible,float('-inf'))
        count=min(budget,int(eligible.sum()))
        chosen=weights.topk(count).indices;chosen=chosen[weights[chosen]>0]
        chosen=chosen[identities[chosen].argsort()]
        return dict(token_ids=[values[j,:int(lengths[j])].long()for j in chosen],
            event_ordinals=identities[chosen].cpu().tolist(),read_abstained=False,
            selected_candidate=winner,read_weights=attention[winner,chosen].detach().cpu().tolist())

    def tensors(self):return {n:getattr(self,n)for n in ('hot','hot_lengths','hot_ids','cold','cold_lengths','cold_ids','hot_native_start','cold_native_start')}
