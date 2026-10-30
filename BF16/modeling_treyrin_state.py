"""Model-owned factual experience and non-factual Spark state."""
import dataclasses
import importlib
import torch
from torch import nn

class TreyrinMemory(nn.Module):
    def __init__(self, state):
        super().__init__()
        self._template = state
        for name, value in state.tensors().items():
            self.register_buffer(name, value, persistent=True)
    def state(self, binding):
        return dataclasses.replace(self._template, binding=binding,
            **{name:getattr(self,name) for name in self._buffers})
    def adopt(self, state):
        incoming=state.tensors()
        if set(incoming)!=set(self._buffers):
            raise ValueError('A state-schema migration is required')
        for name,value in incoming.items():
            self._buffers[name]=value
        self._template=state

class TreyrinSparkState(nn.Module):
    def __init__(self, width, count, episodes, device):
        super().__init__()
        self.register_buffer('vectors',torch.zeros(1,count,width,device=device))
        self.register_buffer('gates',torch.zeros(1,count,device=device))
        self.register_buffer('selection_logits',torch.zeros(1,count,device=device))
        self.register_buffer('episode_attention',torch.zeros(1,count,episodes,device=device))
        self.register_buffer('updates',torch.zeros(1,dtype=torch.long,device=device))
    def population(self):
        HypothesisPopulation=importlib.import_module('treyrin.hypothesis_population').HypothesisPopulation
        return HypothesisPopulation(*(getattr(self,k).detach().clone() for k in
            ('vectors','gates','selection_logits','episode_attention')))
    @torch.no_grad()
    def adopt(self,population):
        if population is None:return
        for name in ('vectors','gates','selection_logits','episode_attention'):
            value=getattr(population,name)
            target=getattr(self,name)
            if value.shape!=target.shape:raise ValueError('Spark state shape differs: '+name)
            target.copy_(value.detach().to(target))
        self.updates.add_(1)
