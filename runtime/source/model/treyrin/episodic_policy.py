"""Causal, trainable episodic skip/replace decisions inside the model.

Only current item, public provenance, working context and existing memory are
inputs. Future usefulness is a training target, never a routing input. The
forward action is discrete; autograd uses an explicitly biased straight-through
softmax surrogate. Zero-initialized residuals start from an oldest-item policy.
"""
from dataclasses import dataclass
import torch
from torch import nn,Tensor

@dataclass
class EpisodicDecision:
    write_weights:Tensor       # B,E; exact zero or one forward, ST gradient
    probabilities:Tensor       # B,E+1; skip then physical slots, not truth
    action:Tensor              # B long; zero=skip, j+1=write physical slot j

class EpisodicPolicy(nn.Module):
    def __init__(self,width):
        super().__init__();self.width=width
        self.event=nn.Sequential(nn.Linear(4*width+2,width),nn.SiLU(),nn.Linear(width,2))
        self.slot=nn.Sequential(nn.Linear(7*width+3,width),nn.SiLU(),nn.Linear(width,1))
        for module in (self.event[-1],self.slot[-1]):
            nn.init.zeros_(module.weight);nn.init.zeros_(module.bias)

    def forward(self,key,value,source,context,keys,values,sources,occupied,event_ids,tick,
                *,item_id=None,stage=0,write_mask=None,allow_skip=None):
        b,e,d=keys.shape
        if d!=self.width or values.shape!=keys.shape or sources.shape!=keys.shape:
            raise ValueError('Aligned episodic key/value/source required')
        if any(t.shape!=(b,d) for t in (key,value,source,context)) or occupied.shape!=(b,e) or event_ids.shape!=(b,e):
            raise ValueError('Causal item/context/occupancy/identity shapes differ')
        now=tick+1
        item_id=now if item_id is None else item_id
        age=(now[:,None]-event_ids).clamp_min(0).to(value.dtype)/now.to(value.dtype)[:,None]
        new_age=(now-item_id).clamp_min(0).to(value.dtype)/now.to(value.dtype)
        phase=value.new_full((b,),float(stage))
        event=self.event(torch.cat((key,value,source,context,new_age[:,None],phase[:,None]),-1))
        expand=lambda x:x[:,None].expand(-1,e,-1)
        slot=self.slot(torch.cat((keys,values,sources,expand(key),expand(value),expand(context),
            values*expand(value),age[:,:,None],expand(new_age[:,None]),expand(phase[:,None])),-1)).squeeze(-1)
        # Identity/occupancy determine allocation order only, not importance.
        oldest=event_ids.argmin(-1)
        base=value.new_full((b,e),-2.).scatter(1,oldest[:,None],0.)
        logits=torch.cat((event[:,:1]-2.,base+event[:,1:2]+slot),-1)
        # Resource constraints do not assign importance or look at future
        # targets. The unmasked learned actor still orders all legal slots.
        if write_mask is not None or allow_skip is not None:
            legal=torch.ones_like(logits,dtype=torch.bool)
            if write_mask is not None:
                if write_mask.shape!=(b,e)or write_mask.dtype!=torch.bool:raise ValueError('Boolean physical write mask required')
                legal[:,1:]=write_mask
            if allow_skip is not None:
                if allow_skip.shape!=(b,)or allow_skip.dtype!=torch.bool:raise ValueError('Boolean skip mask required')
                legal[:,0]=allow_skip
            if not bool(legal.any(-1).all()):raise ValueError('At least one legal storage action required')
            logits=logits.masked_fill(~legal,float('-inf'))
        probabilities=logits.softmax(-1)
        action=logits.argmax(-1)
        hard=torch.nn.functional.one_hot(action,e+1).to(value.dtype)
        routing=(hard+(probabilities-probabilities.detach()) if torch.is_grad_enabled() else hard)
        return EpisodicDecision(routing[:,1:],probabilities,action)
