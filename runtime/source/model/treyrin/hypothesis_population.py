"""Input-conditioned, revisable internal alternatives with variable occupancy.

Capacity is a resource bound, not a fixed ontology. The vectors are NOT yet
validated semantic hypotheses. Selection is a straight-through engineering
surrogate; masks suppress actual normalized transmission, not just vectors.
The population is query scratch and never enters factual observation slots.
"""
from dataclasses import dataclass

import torch
from torch import Tensor,nn
from torch.nn import functional as F


def gated_distribution(logits:Tensor,gate:Tensor):
    if logits.shape!=gate.shape or not (gate.detach().sum(-1)>0).all():
        raise ValueError('Aligned gates with at least one active alternative required')
    if not torch.isfinite(gate).all() or ((gate.detach()<0)|(gate.detach()>1)).any():
        raise ValueError('Selection gates must be finite and bounded')
    # Inactive slots cannot set the stabilizing maximum. This avoids underflow
    # if an inactive slot has an arbitrarily large logit.
    active=gate.detach()>0
    shift=logits.masked_fill(~active,torch.finfo(logits.dtype).min).max(-1,keepdim=True).values
    centered=torch.where(active,logits-shift,torch.zeros_like(logits))
    weights=centered.exp()*gate
    return weights/weights.sum(-1,keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)


@dataclass
class HypothesisPopulation:
    vectors:Tensor              # B,C,D; meanings depend on context/experience
    gates:Tensor                # B,C; forward hard, backward relaxed
    selection_logits:Tensor     # B,C; allocation score, NOT truth probability
    episode_attention:Tensor    # B,C,E; provenance tracing, NOT evidence quality


class AdaptiveHypotheses(nn.Module):
    selection_distribution=staticmethod(gated_distribution)

    def __init__(self,width,capacity):
        super().__init__()
        if capacity<2:
            raise ValueError('At least two possible alternatives required')
        self.width,self.capacity=width,capacity
        self.seed=nn.Linear(3*width,width)
        self.address=nn.Linear(2*width,width,bias=False)
        self.propose=nn.Sequential(nn.Linear(5*width,width),nn.SiLU(),nn.Linear(width,width))
        self.transition=nn.GRUCell(2*width,width)
        self.keep=nn.Linear(3*width,1)
        self.norm=nn.LayerNorm(width)
        # No per-slot learned prototype: a shared conditional generator rolls
        # forward to produce each possible alternative.

    def forward(self,x,goal,working,state,previous=None):
        b,d=x.shape
        if d!=self.width or goal.shape!=x.shape:
            raise ValueError('Aligned input and goal required')
        if previous is not None and previous.vectors.shape!=(b,self.capacity,d):
            raise ValueError('Scratch population has different capacity')
        context=torch.cat((x,working[:,-1],goal),-1)
        seed=self.seed(context).tanh()
        memory=state.values
        vectors,logits,attention=[],[],[]
        coverage=torch.zeros_like(x)
        for index in range(self.capacity):
            old=(torch.zeros_like(x) if previous is None else
                 previous.vectors[:,index]*previous.gates[:,index,None])
            query=self.address(torch.cat((seed,goal),-1))
            scores=torch.einsum('bd,bed->be',query,state.keys+state.sources)/d**.5
            weights=scores.masked_fill(~state.occupied,torch.finfo(scores.dtype).min).softmax(-1)
            weights=weights*state.occupied
            weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
            evidence=torch.einsum('be,bed->bd',weights,memory)
            fast=torch.einsum('bd,bde->be',F.normalize(seed,dim=-1),state.association)
            proposal=self.norm(self.propose(torch.cat((seed,evidence,fast,old,coverage),-1)))
            selection=self.keep(torch.cat((proposal,x,coverage),-1)).squeeze(-1)
            vectors.append(proposal);logits.append(selection);attention.append(weights)
            coverage=coverage+(proposal-coverage)/(index+1)
            seed=self.transition(torch.cat((proposal,evidence),-1),seed)
        selection_logits=torch.stack(logits,1)
        soft=selection_logits.sigmoid()
        hard=(soft>=.5).to(soft.dtype)
        # A nonempty computational population; forced survivor chosen by the
        # allocator, not assumed to be correct. No teacher count is consulted.
        fallback=F.one_hot(selection_logits.argmax(-1),self.capacity).to(soft.dtype)
        hard=torch.where(hard.sum(-1,keepdim=True)>0,hard,fallback)
        gates=hard+(soft-soft.detach())
        return HypothesisPopulation(torch.stack(vectors,1),gates,selection_logits,
                                    torch.stack(attention,1))
