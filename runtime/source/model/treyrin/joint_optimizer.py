"""One joint update: inherited core AdamW and memory-efficient body SGD.

The two parameter sets are disjoint. No reference-teacher parameter is included.
This is a CPU training choice, not a biological plasticity claim.
"""
from collections import ChainMap

import torch


class CoreBodyOptimizer:
    def __init__(self,core_parameters,body_parameters,*,core_lr=1e-4,body_lr=3e-5):
        core_parameters,body_parameters=list(core_parameters),list(body_parameters)
        if not core_parameters or not body_parameters or {id(p) for p in core_parameters}&{id(p) for p in body_parameters}:
            raise ValueError('Nonempty disjoint core/body parameter sets required')
        self.core=torch.optim.AdamW(core_parameters,lr=core_lr)
        self.body=torch.optim.SGD(body_parameters,lr=body_lr,momentum=0.,weight_decay=0.,foreach=False)

    @property
    def param_groups(self):return [*self.core.param_groups,*self.body.param_groups]

    @property
    def state(self):return ChainMap(self.core.state,self.body.state)

    def zero_grad(self,*,set_to_none=True):
        self.core.zero_grad(set_to_none=set_to_none);self.body.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.core.step();self.body.step()

    def state_dict(self):
        return dict(schema='treyrin-core-adam-body-sgd-v1',core=self.core.state_dict(),body=self.body.state_dict())

    def load_state_dict(self,state):
        if set(state)!= {'schema','core','body'} or state['schema']!='treyrin-core-adam-body-sgd-v1':
            raise ValueError('Joint optimizer state differs')
        self.core.load_state_dict(state['core']);self.body.load_state_dict(state['body'])
        if any(g['momentum']!=0 or g['weight_decay']!=0 for g in self.body.param_groups):
            raise ValueError('Body SGD policy differs')
