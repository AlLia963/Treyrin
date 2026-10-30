"""Shared neural utility correction that directly observes cortical semantics.

Initially exactly zero. This is a trainable component, not an if-PASS-act rule.
Actual useful behavior still needs feedback training and independent tests.
"""
import torch
from torch import nn


class CorticalUtilityCalibration(nn.Module):
    def __init__(self,width,hidden=32,bound=2.):
        super().__init__();self.width=width;self.bound=bound
        self.net=nn.Sequential(nn.Linear(52+2*width,hidden),nn.SiLU(),nn.Linear(hidden,5))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def inputs(self,cortical,state,features,context,allowed):
        b,rows,actions=cortical.shape
        if actions!=5 or features.shape!=(b,rows,29) or context.shape!=(b,9) or allowed.shape!=cortical.shape:
            raise ValueError('Actual aligned cortical/goal/clock/legality surfaces required')
        mask=allowed.to(cortical.dtype)
        count=mask.sum((1,2),keepdim=True).clamp_min(1.)
        center=(cortical*mask).sum((1,2),keepdim=True)/count
        scale=(((cortical-center).square()*mask).sum((1,2),keepdim=True)/count).sqrt().clamp_min(.25)
        neural=((cortical-center)/scale).clamp(-8,8)*mask
        goals=torch.cat((state.goal_vectors,state.goal_vectors.new_zeros(b,1,self.width)),1)
        current=state.working[:,-1,None].expand(-1,rows,-1)
        drives=state.drive_state if state.drive_state is not None else current.new_zeros(b,3)
        clock=state.cognitive_context[:,None].expand(-1,rows,-1)
        return torch.cat((neural,features,context[:,None].expand(-1,rows,-1),goals,current,
                          drives[:,None].expand(-1,rows,-1),clock),-1).float()

    def forward(self,cortical,state,features,context,allowed):
        x=self.inputs(cortical,state,features,context,allowed)
        return self.bound*self.net(x.to(self.net[0].weight.dtype)).float().tanh()
