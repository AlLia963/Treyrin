"""Typed native quantities inside recurrent computation, not an external policy.

Truth, categorical choice, utility and reading allocation have different sample
spaces and training targets. None of these untrained outputs is calibrated.
Keeping the quantities separate prevents a high utility from being added to a
truth logit by construction; it does not prove absence of motivated reasoning
through the subsequently shared recurrent state.
"""
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from .hypothesis_population import gated_distribution


@dataclass
class NativeField:
    choice_logits: Tensor         # B,K; CLOSED categorical set, not world beliefs
    predicate_logits: Tensor      # B,K; independent binary propositions
    action_values: Tensor         # B,K; real-valued utility, not a probability
    read_allocation: Tensor       # B,K; compute allocation, not evidence support
    feedback: Tensor | None = None  # B,K,3; intervened views, never report as beliefs


class NativeDecisionField(nn.Module):
    def __init__(self, width, *, deep: bool = False, abstain: bool = False, body_bridge: bool = False):
        super().__init__()
        if type(deep) is not bool or type(abstain) is not bool or type(body_bridge) is not bool:
            raise ValueError('Explicit deep-field and abstain policies required')
        self.deep = deep
        self.abstain = abstain
        if deep:
            # r8 readout-head fix: the shallow single-linear features collapsed
            # observation differences before the choice head (h5v2 diagnosis).
            self.features = nn.Sequential(nn.Linear(3 * width, width), nn.SiLU(),
                                          nn.Linear(width, width), nn.LayerNorm(width))
            self.choice = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
        else:
            self.features = nn.Sequential(nn.Linear(3*width, width), nn.SiLU())
            self.choice = nn.Linear(width, 1)
        self.predicate = nn.Linear(width, 1)
        self.utility = nn.Sequential(nn.Linear(2*width, width), nn.SiLU(), nn.Linear(width, 1))
        self.read = nn.Sequential(nn.Linear(width+3, width), nn.SiLU(), nn.Linear(width, 1))
        if abstain:
            # A2 (r8.5): an explicit no-read allocation slot — the field can
            # route reading volume away from all candidates.
            self.abstain_head = nn.Linear(width, 1)
        self.compatibility=NativeCompatibility(width) if body_bridge else None

    def forward(self, context, candidates, evidence, goal, *, feedback_mode='normal',candidate_gate=None,
                body_context=None,language_prior=None,language_prior_gain=0.):
        if candidates.ndim != 3 or evidence.shape != candidates.shape:
            raise ValueError('Aligned public candidates and per-candidate evidence required')
        b, k, d = candidates.shape
        if context.shape != (b, d) or goal.shape != (b, d) or k < 1:
            raise ValueError('Invalid common context/goal/candidate shape')
        q = context[:, None].expand_as(candidates)
        g = goal[:, None].expand_as(candidates)
        features = self.features(torch.cat((q, candidates, evidence), -1))
        raw_choice = self.choice(features).squeeze(-1)
        if language_prior is not None:
            if language_prior.shape!=raw_choice.shape or not bool(torch.isfinite(language_prior).all()):
                raise ValueError('Aligned finite public choice-label logits required')
            # Categorical evidence from the SAME body at the final cortical
            # visit, before its new feedback. Never add this to world predicates
            # or utilities. No decoded string or external classifier is used.
            raw_choice=raw_choice+language_prior_gain*language_prior.float().log_softmax(-1)
        if self.compatibility is not None and body_context is not None:
            if body_context.shape!=context.shape:raise ValueError('Aligned internal language context required')
            raw_choice=raw_choice+self.compatibility(context,candidates,evidence,body_context)
        predicate = self.predicate(features).squeeze(-1)
        values = self.utility(torch.cat((features, g), -1)).squeeze(-1)
        if candidate_gate is None:
            choice,probabilities=raw_choice,raw_choice.softmax(-1)
            live=torch.ones_like(raw_choice)
        else:
            if candidate_gate.shape!=(b,k):
                raise ValueError('Candidate occupancy shape differs')
            probabilities=gated_distribution(raw_choice,candidate_gate)
            live=candidate_gate
            choice=raw_choice.masked_fill(candidate_gate.detach()==0,torch.finfo(raw_choice.dtype).min)
        feedback = torch.stack((probabilities, predicate.sigmoid()*live, values*live), -1)
        if feedback_mode == 'detach':
            feedback = feedback.detach()
        elif feedback_mode == 'uniform':
            uniform=(torch.full_like(choice,1/k) if candidate_gate is None else
                     gated_distribution(torch.zeros_like(raw_choice),candidate_gate))
            feedback = torch.stack((uniform,.5*live,torch.zeros_like(values)), -1)
        elif feedback_mode == 'wrong_binding':
            if candidate_gate is None:
                feedback = feedback.roll(1,1)
            else:
                indices=torch.arange(k,device=feedback.device)[None].expand(b,-1).clone()
                for row in range(b):
                    active=(candidate_gate[row].detach()>0).nonzero().flatten()
                    indices[row,active]=active.roll(1)
                feedback=feedback.gather(1,indices[:,:,None].expand(-1,-1,3))
        elif feedback_mode != 'normal':
            raise ValueError('Unknown native-field feedback intervention')
        allocation_logits=self.read(torch.cat((features,feedback),-1)).squeeze(-1)
        if self.abstain:
            abstain_logits=self.abstain_head(features.mean(1)).squeeze(-1)[:,None]
            full=torch.cat((allocation_logits,abstain_logits),-1)
            full=(full.softmax(-1) if candidate_gate is None else
                  gated_distribution(full,torch.cat((candidate_gate,torch.ones_like(abstain_logits)),-1)))
            allocation=full
        else:
            allocation=(allocation_logits.softmax(-1) if candidate_gate is None else
                        gated_distribution(allocation_logits,candidate_gate))
        return NativeField(choice, predicate, values, allocation, feedback)


class NativeCompatibility(nn.Module):
    """Candidate interactions with regional, episodic and same-graph body content.

    A zero final layer preserves the old readout until this branch is trained.
    """
    def __init__(self,width):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(6*width,width),nn.SiLU(),nn.Linear(width,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,context,candidates,evidence,body_context):
        d=context.shape[-1]
        q=torch.nn.functional.layer_norm(context,(d,))[:,None].expand_as(candidates)
        c=torch.nn.functional.layer_norm(candidates,(d,))
        e=torch.nn.functional.layer_norm(evidence,(d,))
        b=torch.nn.functional.layer_norm(body_context,(d,))[:,None].expand_as(candidates)
        return self.net(torch.cat((q,c,q*c,e*c,b,b*c),-1)).squeeze(-1)
