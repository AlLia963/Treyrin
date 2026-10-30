"""Subregional memory-circuit hypothesis inside the language/decision model.

Distinct populations, CA1 compartments, signed local paths and event-only fast
plasticity are actual computations. This is not a full hippocampus simulation.
Rates, coding densities and topology are engineering choices, not measurements.
Read-time hypotheses do not modify the committed fast weights or eligibility.
"""
from dataclasses import dataclass

import torch
from torch import Tensor,nn
from torch.nn import functional as F

from .regional_circuit import PositiveLinear


@dataclass
class HippocampalResult:
    population:Tensor
    fast:Tensor
    eligibility:Tensor
    context:Tensor
    mismatch:Tensor


class HippocampalCircuit(nn.Module):
    populations=('ec2','ec3','ec5','superficial_inhibition','dg_granule','dg_inhibition',
        'ca3_pyramidal','ca3_inhibition','ca2_context','ca1_basal','ca1_apical',
        'ca1_soma','subiculum','distal_inhibition')
    modes=('normal','no_recurrence','no_direct','no_feedback')

    def __init__(self,width):
        super().__init__()
        self.width=width
        self.ec2_input=nn.Linear(2*width,width)
        self.ec3_input=nn.Linear(2*width,width)
        self.ec2_feedback,self.ec3_feedback=PositiveLinear(width),PositiveLinear(width)
        self.superficial_inhibition=PositiveLinear(width)
        self.dg_input=nn.Linear(width,width)
        self.dg_inhibition=PositiveLinear(width)
        self.dg_to_ca3,self.ec2_to_ca3=PositiveLinear(width),nn.Linear(width,width)
        self.ca3_inhibition=PositiveLinear(width)
        self.ca2_input=nn.Linear(3*width,width)
        self.ca3_to_basal,self.ca2_to_basal=PositiveLinear(width),PositiveLinear(width)
        self.ec3_to_apical=PositiveLinear(width)
        self.distal_inhibition=PositiveLinear(width)
        self.ca1_to_subiculum,self.ec3_to_subiculum=PositiveLinear(width),PositiveLinear(width)
        self.subiculum_to_ec5=PositiveLinear(width)
        self.ec3_on,self.ec3_off=nn.Linear(width,width),nn.Linear(width,width)
        self.prediction=nn.Linear(width,width)
        self.plasticity=nn.Linear(3*width,1)
        self.readout=nn.Linear(4*width,width)
        self.bias=nn.Parameter(torch.zeros(5,width))
        self.rates=nn.Parameter(torch.zeros(len(self.populations)))
        self.eligibility_decay=nn.Parameter(torch.tensor(1.))
        self.fast_decay=nn.Parameter(torch.tensor(5.))
        self.temporal_mix=nn.Parameter(torch.tensor(0.))
        self.write_rate=nn.Parameter(torch.tensor(-2.))
        topology_rng=torch.Generator(device='cpu').manual_seed(246100+width)
        mask=torch.rand(width,width,generator=topology_rng)<.25
        mask.fill_diagonal_(False)
        self.register_buffer('recurrent_topology',mask)

    @staticmethod
    def sparse_rate(value,fraction):
        count=max(1,round(value.shape[-1]*fraction))
        indices=value.topk(count,-1).indices
        return torch.zeros_like(value).scatter(-1,indices,value.gather(-1,indices))

    def forward(self,x,source,population,fast,eligibility,*,phase='read',mode='normal'):
        b,d=x.shape
        if d!=self.width or source.shape!=x.shape or population.shape!=(b,len(self.populations),d):
            raise ValueError('Explicit subregional population/source shape required')
        if fast.shape!=(b,d,d) or eligibility.shape!=fast.shape:
            raise ValueError('Distinct fast association and eligibility matrices required')
        if phase not in ('read','encode') or mode not in self.modes:
            raise ValueError('Declared read/encode phase and pathway intervention required')
        old={name:population[:,i] for i,name in enumerate(self.populations)}
        rates=self.rates.sigmoid()
        def mix(name,proposal):
            index=self.populations.index(name)
            return old[name]+rates[index]*(proposal-old[name])
        feedback_enabled=float(mode!='no_feedback')
        superficial=mix('superficial_inhibition',self.superficial_inhibition(old['subiculum']).sigmoid())
        shared=torch.cat((x,source),-1)
        ec2=mix('ec2',(self.ec2_input(shared)+feedback_enabled*(
            self.ec2_feedback(old['ec5'])-superficial)).sigmoid())
        ec3_drive=self.ec3_input(shared)+feedback_enabled*(self.ec3_feedback(old['ec5'])-superficial)
        on,off=.01+.24*self.ec3_on(ec3_drive).sigmoid(),.01+.24*self.ec3_off(ec3_drive).sigmoid()
        # Bounded Markov population transition. A discrete state transition is
        # not the incorrectly signed continuous derivative sometimes rendered
        # by HTML equations; no biological time unit is claimed here.
        ec3=old['ec3']+(1-old['ec3'])*on-old['ec3']*off
        dg_inhibition=mix('dg_inhibition',self.dg_inhibition(old['dg_granule']).sigmoid())
        dg=self.sparse_rate((self.dg_input(ec2)-dg_inhibition).sigmoid(),.25)
        dg=mix('dg_granule',dg)
        dg=self.sparse_rate(dg,.25)
        ca3_inhibition=mix('ca3_inhibition',self.ca3_inhibition(old['ca3_pyramidal']).sigmoid())
        recurrent=torch.einsum('bd,bde->be',old['ca3_pyramidal'],fast)
        if mode=='no_recurrence':recurrent=torch.zeros_like(recurrent)
        ca3=mix('ca3_pyramidal',(self.ec2_to_ca3(ec2)+self.dg_to_ca3(dg)+recurrent-
            ca3_inhibition+self.bias[0]).sigmoid())
        ca3=self.sparse_rate(ca3,.5)
        ca2=mix('ca2_context',self.ca2_input(torch.cat((source,ec2,old['ca2_context']),-1)).sigmoid())
        basal=mix('ca1_basal',(self.ca3_to_basal(ca3)+self.ca2_to_basal(ca2)+self.bias[1]).sigmoid())
        distal=mix('distal_inhibition',self.distal_inhibition(old['ec5']).sigmoid())
        direct=self.ec3_to_apical(ec3)*(1-distal)
        if mode=='no_direct':direct=torch.zeros_like(direct)
        apical=mix('ca1_apical',direct.sigmoid())
        if mode=='no_direct':apical=torch.zeros_like(apical)
        soma=mix('ca1_soma',F.relu(basal*(.2+apical)-self.bias[2].sigmoid()).clamp(max=1))
        subiculum=mix('subiculum',(self.ca1_to_subiculum(soma)+self.ec3_to_subiculum(ec3)+self.bias[3]).sigmoid())
        ec5=mix('ec5',(self.subiculum_to_ec5(subiculum)+self.bias[4]).sigmoid())
        mismatch=apical-self.prediction(ca3).sigmoid()
        values=dict(ec2=ec2,ec3=ec3,ec5=ec5,superficial_inhibition=superficial,
            dg_granule=dg,dg_inhibition=dg_inhibition,ca3_pyramidal=ca3,
            ca3_inhibition=ca3_inhibition,ca2_context=ca2,ca1_basal=basal,
            ca1_apical=apical,ca1_soma=soma,subiculum=subiculum,distal_inhibition=distal)
        updated=torch.stack([values[name] for name in self.populations],1)
        context=self.readout(torch.cat((soma,subiculum,ca2,mismatch),-1))
        if phase=='encode':
            # Local nonnegative auto/temporal associations constrained by a
            # fixed sparse recurrent topology; no hidden answer enters here.
            temporal=torch.einsum('bd,be->bde',old['ca3_pyramidal'],ca3)
            decay=self.eligibility_decay.sigmoid()
            eligibility=decay*eligibility+(1-decay)*temporal
            auto=torch.einsum('bd,be->bde',ca3,ca3)
            target=auto+self.temporal_mix.sigmoid()*eligibility
            gate=self.plasticity(torch.cat((x,source,mismatch.abs()),-1)).sigmoid()
            fast=(self.fast_decay.sigmoid()*fast+
                self.write_rate.sigmoid()*gate[:,:,None]*target)*self.recurrent_topology
            fast=fast/fast.sum(-1,keepdim=True).clamp_min(1.)
        return HippocampalResult(updated,fast,eligibility,context,mismatch)
