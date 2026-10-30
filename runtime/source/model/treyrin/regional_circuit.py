"""Regional recurrent computation: a trainable circuit hypothesis.

The population/compartment states and signed pathways are real computations.
Their names are NOT claims of anatomical fidelity. In particular, striatal
populations may coactivate; direct/indirect is not a universal GO/NO-GO switch.
Content relay and connectivity modulation are distinct operations. Rate values,
dimensions and region assignments are engineering hypotheses, not measurements.
"""
from dataclasses import dataclass

import torch
from torch import Tensor,nn
from torch.nn import functional as F


class PositiveLinear(nn.Module):
    """Sign-constrained connection; the caller supplies excitation/inhibition."""
    def __init__(self,width):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(width,width))

    def forward(self,x):
        return F.linear(x,F.softplus(self.raw)/x.shape[-1])


class SelectionCircuit(nn.Module):
    populations = ('direct','indirect','pallidal_external','subthalamic','output_inhibition')

    def __init__(self,width,*,context_bias=False):
        super().__init__()
        self.direct = nn.Linear(2*width,width)
        self.indirect = nn.Linear(2*width,width)
        self.hyperdirect = nn.Linear(width,width)
        self.d2_to_external = PositiveLinear(width)
        self.external_to_sub = PositiveLinear(width)
        self.direct_to_output = PositiveLinear(width)
        self.sub_to_output = PositiveLinear(width)
        self.bias = nn.Parameter(torch.zeros(3,width))
        self.rates = nn.Parameter(torch.zeros(5))
        # L1 (r8.4): ventral-striatum analogue — hippocampal context biases the
        # direct/indirect proposals before rate mixing. Off => identical ancestor.
        self.context_bias = context_bias
        if context_bias:
            self.context_direct = nn.Linear(width,width)
            self.context_indirect = nn.Linear(width,width)

    def forward(self,executive,alternatives,old,*,override=None,context=None):
        if old.shape != (executive.shape[0],5,executive.shape[1]):
            raise ValueError('Five explicit selection populations required')
        rates = self.rates.sigmoid()
        mix = lambda i,proposal: old[:,i]+rates[i]*(proposal-old[:,i])
        joined = torch.cat((executive,alternatives),-1)
        direct = mix(0,self.direct(joined).sigmoid())
        indirect = mix(1,self.indirect(joined).sigmoid())
        if self.context_bias and context is not None:
            direct = mix(0,(self.direct(joined)+self.context_direct(context)).sigmoid())
            indirect = mix(1,(self.indirect(joined)+self.context_indirect(context)).sigmoid())
        if override:
            for key,value in override.items():
                if key not in ('direct','indirect') or value.shape!=executive.shape or not torch.isfinite(value).all() or ((value<0)|(value>1)).any():
                    raise ValueError('Invalid fixed-population intervention')
            direct = override.get('direct',direct)
            indirect = override.get('indirect',indirect)
        external = mix(2,(self.bias[0]-self.d2_to_external(indirect)).sigmoid())
        sub = mix(3,(self.bias[1]+self.hyperdirect(executive)-self.external_to_sub(external)).sigmoid())
        output = mix(4,(self.bias[2]-self.direct_to_output(direct)+self.sub_to_output(sub)).sigmoid())
        return torch.stack((direct,indirect,external,sub,output),1),1-output


@dataclass
class RegionalResult:
    cortex: Tensor            # B,3,3,D: region x compartment
    relay: Tensor             # B,4,D: content, modulation, inhibitory x2
    selection: Tensor         # B,5,D: distinct signed pathway populations
    native_context: Tensor    # B,D
    language_context: Tensor  # B,D; distinct readout, same internal circuit
    edges: Tensor             # B,target,source; real content transmission


class RegionalCircuit(nn.Module):
    regions = ('language','association','executive')
    compartments = ('bottom_up','recurrent','top_down')
    relay_populations = ('content','connectivity_modulation','content_inhibition','connection_inhibition')

    def __init__(self,width,*,context_bias=False):
        super().__init__()
        self.width = width
        self.role_input = nn.ModuleList(nn.Linear(3*width,width) for _ in self.regions)
        self.role_feedback = nn.ModuleList(nn.Linear(3*width,width) for _ in self.regions)
        self.queries,self.keys,self.values = (nn.Linear(width,width,bias=False) for _ in range(3))
        self.content = nn.Linear(3*width,width)
        # Non-content magnitude/dispersion/uncertainty/resource statistics.
        # This restriction alone does NOT prove a biological categorically-free code.
        self.modulation = nn.Linear(8,width)
        self.reticular = nn.Linear(3*width,2*width)
        self.connectivity = nn.Parameter(torch.randn(3,3,width)*.02)
        self.local_rates = nn.Parameter(torch.zeros(3,3))
        self.relay_rates = nn.Parameter(torch.zeros(4))
        self.selection_loop = SelectionCircuit(width, context_bias=context_bias)
        self.native_readout = nn.Linear(2*width,width)
        self.language_readout = nn.Linear(2*width,width)
        self.register_buffer('allowed_edges',torch.tensor([
            [1,1,0],[1,1,1],[0,1,1]],dtype=torch.bool))

    def forward(self,x,goal,episodic,semantic,association,candidates,field,cortex,relay,selection,
                proposals,update_gates,norms,level_rates,resource,*,mode='normal',edge_override=None,
                context_message=None,selection_context=None):
        if mode not in ('normal','no_cross','gain_only','no_selection'):
            raise ValueError('Unknown regional-circuit intervention')
        b,d=x.shape
        context_message=torch.zeros_like(x) if context_message is None else context_message
        if cortex.shape!=(b,3,3,d) or relay.shape!=(b,4,d) or selection.shape!=(b,5,d):
            raise ValueError('Explicit regional/relay/selection state required')
        recurrent = cortex[:,:,1]
        if field.feedback is None:
            raise ValueError('Regional circuit needs explicitly intervened native feedback')
        alternative = torch.einsum('bk,bkd->bd',field.feedback[:,:,0]*field.feedback[:,:,2].sigmoid(),candidates)
        selected,selection_gate = self.selection_loop(recurrent[:,2],alternative,selection,
            context=selection_context)
        if mode=='no_selection':
            selection_gate = torch.ones_like(selection_gate)
        probabilities = field.feedback[:,:,0]
        entropy = -(probabilities*probabilities.clamp_min(1e-12).log()).sum(-1,keepdim=True)
        statistics = torch.cat((recurrent.abs().mean(-1),recurrent.square().mean(-1),entropy,resource),-1)
        proposed_relay = torch.stack((
            self.content(torch.cat((recurrent[:,1],episodic,semantic),-1)).tanh(),
            self.modulation(statistics).sigmoid(),
            *self.reticular(torch.cat((recurrent[:,2],recurrent[:,1],goal),-1)).sigmoid().chunk(2,-1)),1)
        relay = relay+self.relay_rates.sigmoid()[None,:,None]*(proposed_relay-relay)
        content,modulator,content_inhibition,connection_inhibition = relay.unbind(1)
        scores = torch.einsum('brd,bsd->brs',self.queries(recurrent),self.keys(recurrent))/d**.5
        if mode=='gain_only':
            scores = torch.zeros_like(scores)
        scores = scores+torch.einsum('bd,rsd->brs',modulator*(1-connection_inhibition),self.connectivity)/d**.5
        allowed = self.allowed_edges
        if mode=='no_cross':
            allowed = allowed & torch.eye(3,dtype=torch.bool,device=x.device)
        if edge_override is not None:
            if edge_override.shape!=(3,3) or edge_override.dtype!=torch.bool or edge_override.device!=x.device:
                raise ValueError('Explicit boolean target/source edge mask required')
            allowed = allowed & edge_override
        if not allowed.any(-1).all():
            raise ValueError('Each region needs at least one incoming content path')
        edges = scores.masked_fill(~allowed,torch.finfo(scores.dtype).min).softmax(-1)
        transmitted = torch.einsum('brs,bsd->brd',edges,self.values(recurrent))
        relayed_content = content*(1-content_inhibition)
        role_inputs = ((x,relayed_content,transmitted[:,0]),
                       (episodic,semantic,association+transmitted[:,1]),
                       (goal,alternative,transmitted[:,2]))
        role_higher = ((recurrent[:,1],goal,transmitted[:,0]),
                       (recurrent[:,2],goal,transmitted[:,1]),
                       (recurrent[:,2],goal,transmitted[:,2]+context_message))
        regions = []
        for region in range(3):
            bottom_input = self.role_input[region](torch.cat(role_inputs[region],-1))
            higher_input = self.role_feedback[region](torch.cat(role_higher[region],-1))
            local=[]
            for compartment in range(3):
                old=cortex[:,region,compartment]
                bottom=bottom_input if compartment==0 else cortex[:,region,compartment-1]
                higher=higher_input if compartment==2 else cortex[:,region,compartment+1]
                joined=torch.cat((old,bottom,higher,transmitted[:,region]),-1)
                # Reuse the ancestor's learned local operator; region inputs,
                # feedback, activity and connection states are now distinct.
                proposal=proposals[compartment](joined).tanh()
                gate=update_gates[compartment](joined).sigmoid()*level_rates[compartment]
                gate=gate*self.local_rates[region,compartment].sigmoid()*selection_gate
                local.append(norms[compartment](old+gate*(proposal-old)))
            regions.append(torch.stack(local,1))
        cortex=torch.stack(regions,1)
        native=self.native_readout(torch.cat((cortex[:,1,1],cortex[:,2,1]),-1))
        language=self.language_readout(torch.cat((cortex[:,0,1],cortex[:,1,1]),-1))
        return RegionalResult(cortex,relay,selected,native,language,edges)
