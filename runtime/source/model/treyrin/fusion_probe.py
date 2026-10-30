"""Untrained operator probe for reciprocal native decision/language coupling.

This is neither a language model nor a decision-model reproduction. It checks
whether a proposed continuous feedback route actually exists before training.
Successful gradient flow is not evidence of intuition, useful thinking, or
biological correspondence. No generated text is used as a decision handoff.
"""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Workspace:
    slots: torch.Tensor


class ReciprocalLatentProbe(nn.Module):
    def __init__(self, width=32, slots=4, choices=3, vocabulary=17):
        super().__init__()
        if width % 4 or min(slots,choices,vocabulary)<=0:
            raise ValueError('Positive dimensions and four attention heads required')
        self.width,self.slot_count,self.choices=width,slots,choices
        self.initial=nn.Parameter(torch.randn(slots,width)*.02)
        self.query=nn.Linear(width,width,bias=False)
        self.read=nn.MultiheadAttention(width,4,batch_first=True,dropout=0.)
        self.write=nn.MultiheadAttention(width,4,batch_first=True,dropout=0.)
        self.transition=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,2*width),
                                      nn.GELU(),nn.Linear(2*width,width))
        self.judgment=nn.Linear(width,choices)
        self.choice_dictionary=nn.Parameter(torch.randn(choices,width)*.1)
        self.token_embedding=nn.Embedding(vocabulary,width)
        self.language=nn.Linear(width,vocabulary,bias=False)
        self.message_role=nn.Embedding(3,width)
        self.write_gate=nn.Parameter(torch.full((width,),-.5))

    def initial_state(self,batch):
        if batch<=0: raise ValueError('Empty episode batch')
        return Workspace(self.initial[None].expand(batch,-1,-1))

    def _write(self,state,message,role):
        if message.shape!=(state.slots.shape[0],self.width):
            raise ValueError('Message must have one vector per episode')
        tagged=message+self.message_role.weight[role]
        read,_=self.write(state.slots,tagged[:,None,:],tagged[:,None,:],need_weights=False)
        delta=self.transition(state.slots+read)
        return Workspace(state.slots+self.write_gate.sigmoid()*delta)

    def observe(self,state,features):
        return self._write(state,features,0)

    def decision(self,state,query):
        read,_=self.read(self.query(query)[:,None,:],state.slots,state.slots,need_weights=False)
        return self.judgment(read[:,0]+query)

    def decision_feedback(self,state,query,*,enabled=True,probability_override=None):
        logits=self.decision(state,query)
        probability=logits.softmax(-1) if probability_override is None else probability_override
        if probability.shape!=(state.slots.shape[0],self.choices):
            raise ValueError('Decision distribution shape mismatch')
        if not torch.isfinite(probability).all() or (probability<0).any():
            raise ValueError('Invalid decision distribution')
        if not torch.allclose(probability.sum(-1),torch.ones_like(probability[:,0]),atol=1e-6):
            raise ValueError('Decision probabilities must sum to one')
        # Preserve uncertainty rather than exchanging only argmax or a text token.
        message=probability@self.choice_dictionary
        return (self._write(state,message,1) if enabled else state),logits

    def language_step(self,state,previous_token,*,write_back=True):
        # One already-observed token is consumed. Future target tokens cannot be
        # fed through a bidirectional teacher-forcing context by this interface.
        token=self.token_embedding(previous_token)
        updated=self._write(state,token,2) if write_back else state
        return updated,self.language(updated.slots.mean(1))
