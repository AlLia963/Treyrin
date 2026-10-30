"""Readout-only individual-learning capacity; never changes decoder/input basis.

Two FP32 master tensors. This module's only consumers must be vocabulary
projection and its explicitly indexed label rows. A separate verified
transaction is still required to adopt changed parameters with old state.
"""
import torch
from torch import nn
from torch.nn import functional as F


class OutputAdapter(nn.Module):
    def __init__(self,hidden_size,vocab_size,rank=2,*,device=None):
        super().__init__()
        if any(type(v) is not int or v<=0 for v in (hidden_size,vocab_size,rank)):
            raise ValueError('Positive declared output dimensions required')
        self.hidden_size=hidden_size;self.vocab_size=vocab_size;self.rank=rank
        self.A=nn.Parameter(torch.empty(rank,hidden_size,device=device,dtype=torch.float32))
        self.B=nn.Parameter(torch.zeros(vocab_size,rank,device=device,dtype=torch.float32))
        nn.init.normal_(self.A,std=hidden_size**-.5)

    def latent(self,hidden):
        if hidden.shape[-1]!=self.hidden_size:raise ValueError('Actual final cortical hidden dimension differs')
        return F.linear(hidden.to(self.A.dtype),self.A)

    def delta(self,hidden):return F.linear(self.latent(hidden),self.B)/self.rank

    def label_delta(self,hidden,token_ids):
        if token_ids.ndim!=1 or token_ids.dtype!=torch.long or bool(((token_ids<0)|(token_ids>=self.vocab_size)).any()):
            raise ValueError('Actual vocabulary label rows required')
        return F.linear(self.latent(hidden),self.B.index_select(0,token_ids))/self.rank
