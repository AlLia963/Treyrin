"""Explicit two-state decision research environment, not an open-world world model.

The exact evaluator is used for labels and evaluation ONLY. The learned policy does
not call it. Costs, preferences and observation likelihoods are supplied inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
import random


@dataclass(frozen=True)
class Option:
    id: str
    rewards: tuple[float, float]
    cost: float
    time: float
    risks: tuple[float, float]


@dataclass(frozen=True)
class Problem:
    id: str
    domain: str
    belief: float
    options: tuple[Option, ...]
    budget: float
    weights: tuple[float, float, float, float]
    risk_cap: float
    probe_cost: float
    probe_time: float
    probe_accuracy: float
    allow_probe: bool = True

    def __post_init__(self):
        numbers = [self.belief, self.budget, self.risk_cap, self.probe_cost, self.probe_time,
                   self.probe_accuracy, *self.weights]
        for option in self.options:
            numbers += [*option.rewards, option.cost, option.time, *option.risks]
            if min(option.cost, option.time) < 0 or any(not 0 <= r <= 1 for r in option.risks):
                raise ValueError("Invalid action cost, time or risk")
        if not all(math.isfinite(x) for x in numbers):
            raise ValueError("All numerical inputs must be finite")
        if not 0 <= self.belief <= 1 or not 0 <= self.risk_cap <= 1:
            raise ValueError("Invalid probability")
        if not 0.5 <= self.probe_accuracy <= 1 or min(self.budget, self.probe_cost, self.probe_time) < 0:
            raise ValueError("Invalid observation model or resource budget")
        ids = [option.id for option in self.options]
        if len(ids) != len(set(ids)) or {"defer", "observe"}.intersection(ids):
            raise ValueError("Action IDs must be unique and cannot use reserved IDs")
        if len(self.options) != 3:
            raise ValueError("The v1 neural readout supports exactly three task options")

    @property
    def actions(self) -> tuple[str, ...]:
        return tuple(option.id for option in self.options) + ("defer", "observe")

    def feasible(self, action: str) -> bool:
        if action == "defer":
            return True
        if action == "observe":
            return self.allow_probe and self.probe_cost <= self.budget
        option = next((o for o in self.options if o.id == action), None)
        if option is None:
            return False
        risk = self.belief * option.risks[0] + (1 - self.belief) * option.risks[1]
        return option.cost <= self.budget and risk <= self.risk_cap + 1e-12

    def posterior(self, signal: int) -> tuple[float, "Problem"]:
        if signal not in (0, 1):
            raise ValueError("Signal must be 0 or 1")
        q, p = self.probe_accuracy, self.belief
        likelihood_a = q if signal == 0 else 1 - q
        likelihood_b = 1 - q if signal == 0 else q
        probability = p * likelihood_a + (1 - p) * likelihood_b
        posterior = p * likelihood_a / probability if probability else p
        return probability, replace(self, belief=posterior, budget=self.budget - self.probe_cost,
                                    allow_probe=False)

    def canonical_options(self):
        return sorted(self.options, key=lambda o: (o.rewards, o.cost, o.time, o.risks))

    def encode_action(self, action: str) -> tuple[str, list[float]]:
        """Only observable inputs; no solver values, labels or hidden sampled state."""
        options = self.canonical_options()
        def raw(option):
            return [option.rewards[0] / 40, option.rewards[1] / 40,
                    option.cost / 10, option.time / 10, *option.risks]
        common = [self.belief, self.budget / 10, *self.weights, self.risk_cap,
                  self.probe_cost / 10, self.probe_time / 10, self.probe_accuracy,
                  float(self.allow_probe)]
        common += [v for option in options for v in raw(option)]
        if action in ("defer", "observe"):
            candidate = [0.0] * 6
            kind = action
        else:
            option = next(o for o in options if o.id == action)
            candidate = raw(option)
            kind = "commit"
        numeric = common + candidate + [float(action == "observe"), float(action == "defer")]
        # Canonical numerical content deliberately excludes arbitrary action names.
        content = {"task": self.domain, "belief_state_A": self.belief,
                   "budget": self.budget, "preference_benefit_cost_time_risk": self.weights,
                   "risk_limit": self.risk_cap,
                   "options_reward_A_reward_B_cost_time_risk_A_risk_B": [raw(o) for o in options],
                   "observation": [self.probe_cost, self.probe_time, self.probe_accuracy, self.allow_probe],
                   "candidate": kind, "candidate_features": candidate}
        return json.dumps(content, separators=(",", ":"), ensure_ascii=False), numeric

    def encode_context(self) -> str:
        content = json.loads(self.encode_action("defer")[0])
        content.pop("candidate")
        content.pop("candidate_features")
        def compact(value):
            if isinstance(value, float):
                return round(value, 4)
            if isinstance(value, list):
                return [compact(item) for item in value]
            if isinstance(value, dict):
                return {key: compact(item) for key, item in value.items()}
            return value
        return json.dumps(compact(content), separators=(",", ":"), ensure_ascii=False)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "Problem":
        value = dict(value)
        value["options"] = tuple(Option(id=o["id"], rewards=tuple(o["rewards"]), cost=o["cost"],
                                        time=o["time"], risks=tuple(o["risks"])) for o in value["options"])
        value["weights"] = tuple(value["weights"])
        return cls(**value)


def option_value(problem: Problem, option: Option) -> float:
    p, (benefit, cost, delay, risk) = problem.belief, problem.weights
    return (benefit * (p * option.rewards[0] + (1 - p) * option.rewards[1])
            - cost * option.cost - delay * option.time
            - risk * (p * option.risks[0] + (1 - p) * option.risks[1]))


def oracle_values(problem: Problem) -> dict[str, float]:
    """Exact two-stage Bayes decision reference; never called by model.decide()."""
    values = {o.id: option_value(problem, o) for o in problem.options if problem.feasible(o.id)}
    values["defer"] = 0.0
    if problem.feasible("observe"):
        continuation = 0.0
        for signal in (0, 1):
            probability, updated = problem.posterior(signal)
            continuation += probability * max(oracle_values(updated).values())
        values["observe"] = (continuation - problem.weights[1] * problem.probe_cost
                             - problem.weights[2] * problem.probe_time)
    return values


def generate_problems(count: int, seed: int, shift: bool = False) -> list[Problem]:
    rng = random.Random(seed)
    problems = []
    for index in range(count):
        scale = 2.5 if shift else 1.0
        domain = "research_resource_allocation" if index % 2 == 0 else "environment_route_choice"
        options = []
        for position in range(3):
            if position == 0:
                reward = rng.uniform(1, 12) * scale
                rewards = (reward, reward)
                risks = (rng.uniform(0, 0.08),) * 2
            else:
                good, bad = rng.uniform(8, 30) * scale, rng.uniform(-18, 4) * scale
                risk_a, risk_b = rng.uniform(0, 0.15), rng.uniform(0.1, 0.7)
                rewards = (good, bad) if position == 1 else (bad, good)
                risks = (risk_a, risk_b) if position == 1 else (risk_b, risk_a)
            options.append(Option(f"option_{position}", rewards, rng.uniform(0, 8) * scale,
                                  rng.uniform(0.1, 6), risks))
        problems.append(Problem(
            id=f"{seed}:{index}", domain=domain, belief=rng.uniform(0.05, 0.95),
            options=tuple(options), budget=rng.uniform(2, 12) * scale,
            weights=(rng.uniform(0.5, 1.5), rng.uniform(0.2, 1.5), rng.uniform(0, 1), rng.uniform(0, 8)),
            risk_cap=rng.choice([0.15, 0.3, 0.6, 1.0]), probe_cost=rng.uniform(0, 3) * scale,
            probe_time=rng.uniform(0, 2), probe_accuracy=rng.uniform(0.55, 0.99),
        ))
    return problems
