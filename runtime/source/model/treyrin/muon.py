"""Checkpointable Muon/AdamW hybrid using a Newton-Schulz Muon update.

This implements an optimizer comparison arm, not a learning-benefit claim.
"""
from __future__ import annotations
import torch
from torch import Tensor


@torch.no_grad()
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5) -> Tensor:
    a,b,c=3.4445,-4.7750,2.0315
    X=G.bfloat16();transposed=G.size(-2)>G.size(-1)
    if transposed:X=X.mT
    X=X/(X.norm(dim=(-2,-1),keepdim=True)+1e-7)
    for _ in range(steps):
        A=X@X.mT
        X=a*X+(b*A+c*A@A)@X
    return X.mT if transposed else X


class MuonWithAuxAdam(torch.optim.Optimizer):
    def __init__(self,param_groups:list[dict],lr:float=3e-4,momentum:float=.95,
                 nesterov:bool=True,ns_steps:int=5,weight_decay:float=0.):
        if lr<=0 or not 0<=momentum<1 or ns_steps<1 or weight_decay<0:
            raise ValueError('Invalid optimizer hyperparameters')
        groups=[]
        for original in param_groups:
            group=dict(original)
            group['params']=[p for p in group['params'] if p.requires_grad]
            group['use_muon']=bool(group.pop('muon',group.get('use_muon',False)))
            if group['use_muon'] and any(p.ndim!=2 for p in group['params']):
                raise ValueError('Muon accepts 2D matrices only')
            if group['params']:groups.append(group)
        super().__init__(groups,dict(lr=lr,momentum=momentum,nesterov=nesterov,
            ns_steps=ns_steps,weight_decay=weight_decay,betas=(.9,.999),eps=1e-8))

    @property
    def groups(self):
        return self.param_groups

    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():loss=closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:continue
                if p.grad.is_sparse:raise ValueError('Sparse gradients unsupported')
                g=p.grad;state=self.state[p]
                if group['weight_decay']:p.mul_(1-group['lr']*group['weight_decay'])
                if group['use_muon']:
                    buf=state.setdefault('momentum_buffer',torch.zeros_like(p))
                    buf.mul_(group['momentum']).add_(g)
                    direction=g+group['momentum']*buf if group['nesterov'] else buf
                    update=zeropower_via_newtonschulz5(direction,group['ns_steps'])
                    scale=max(1.,p.size(-2)/p.size(-1))**.5
                    p.add_(update.to(p.dtype),alpha=-group['lr']*scale)
                else:
                    state['step']=int(state.get('step',0))+1
                    m=state.setdefault('exp_avg',torch.zeros_like(p))
                    v=state.setdefault('exp_avg_sq',torch.zeros_like(p))
                    b1,b2=group['betas'];n=state['step']
                    m.mul_(b1).add_(g,alpha=1-b1)
                    v.mul_(b2).addcmul_(g,g,value=1-b2)
                    denom=(v.sqrt()/(1-b2**n)**.5).add_(group['eps'])
                    p.addcdiv_(m,denom,value=-group['lr']/(1-b1**n))
        return loss


def hybrid_optimizer(model,*,lr:float,muon_lr:float|None=None,momentum:float=.95):
    muon,adam=[],[]
    auxiliary=('body.','embed','head','native_field.choice','native_field.predicate',
               'native_field.utility','native_field.read','control','initiate')
    for name,p in model.named_parameters():
        if p.requires_grad:
            (muon if p.ndim==2 and not any(tag in name for tag in auxiliary) else adam).append(p)
    return MuonWithAuxAdam([{'params':muon,'muon':True,'lr':muon_lr or lr},
                           {'params':adam,'muon':False,'lr':lr}],lr=lr,momentum=momentum)
