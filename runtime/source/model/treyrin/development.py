"""Developmental value readout: factored actions and reachable-state rehearsal.

This is still a frozen-language-backbone expansion, not a new pretrained LLM.
No exact value solver, utility formula, or labels are called by forward().
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from .decision import Problem


class DevelopmentalReadout(nn.Module):
    """Local commitment values, contextual observation values, exact zero defer.

    A terminal commitment does not need unrelated candidate actions. Observation
    value does depend on alternatives and resources. Both use one state encoding.
    This deliberately modest inductive bias is tested against a root-only fit.
    """

    def __init__(self, representation_dim: int = 2560):
        super().__init__()
        self.language = nn.Sequential(nn.LayerNorm(representation_dim),
                                      nn.Linear(representation_dim, 16), nn.Tanh())
        self.commit = nn.Sequential(nn.Linear(27, 96), nn.SiLU(),
                                    nn.Linear(96, 64), nn.SiLU(), nn.Linear(64, 1))
        self.observe = nn.Sequential(nn.Linear(53, 96), nn.SiLU(),
                                     nn.Linear(96, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, representation: torch.Tensor, numeric: torch.Tensor):
        language = self.language(representation)
        local = torch.cat((numeric[..., :1], numeric[..., 2:6], numeric[..., 29:35]), dim=-1)
        commitment = self.commit(torch.cat((language, local), dim=-1)).squeeze(-1)
        observation = self.observe(torch.cat((language, numeric), dim=-1)).squeeze(-1)
        # The environment defines defer as zero; learning a different value is an
        # avoidable inconsistency. This is not a reference-solver fallback.
        return torch.where(numeric[..., -1].bool(), torch.zeros_like(commitment),
                           torch.where(numeric[..., -2].bool(), observation, commitment))


def reachable_family(problem: Problem) -> tuple[Problem, ...]:
    """Keep the initial state and both physically possible observation branches.

    Do not manufacture a negative-budget or impossible training state.
    Families, rather than individual rows, must remain in the same data split.
    """
    if not problem.feasible("observe"):
        return (problem,)
    children = []
    for signal in (0, 1):
        probability, child = problem.posterior(signal)
        if probability <= 0:
            raise ValueError("A curriculum family requires positive-likelihood branches")
        children.append(replace(child, id=f"{problem.id}/signal-{signal}"))
    return problem, *children
