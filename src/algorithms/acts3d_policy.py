"""ACTS-3D research-only tri-action decision policy.

This module implements the *routing decision*, not the CNN/SNN models,
cross-fitted utility estimator, event encoder or conditional GPU kernels.

The router never calls the analog or spiking models, and it never accesses
ground-truth masks. A caller must pass calibrated predictions made strictly
from inputs available before refinement.

Inputs:
  predicted gains: expected loss improvements over baseline
  event_gain: spiking real events vs identical SNN with zero events
  spike_vs_analog_gain: relative event-bearing SNN vs matched analog
  harm probabilities: estimated probability of harmful correction
  cost: measured or calibrated incremental inference costs in the SAME unit
        as the budget (e.g., GPU joules or seconds; NEVER mix)
Outputs:
  one of 'skip', 'analog', 'spike' for each region, within budget.

No claim of biological time, causal patient effect, calibrated probability,
hardware energy saving or clinical safety follows from this reference.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

Action = Literal["skip", "analog", "spike"]


@dataclass(frozen=True)
class RegionEstimate:
    """All fields must be computed without test-time ground truth."""
    region_id: str
    gland: Literal["left", "right"]
    gain_analog: float
    gain_spike: float
    event_gain: float
    spike_vs_analog_gain: float
    harm_prob_analog: float
    harm_prob_spike: float
    cost_analog: float
    cost_spike: float


@dataclass(frozen=True)
class RoutingConfig:
    budget: float
    max_harm_probability: float = 0.05
    min_event_gain: float = 0.0
    min_spike_vs_analog_gain: float = 0.0
    harm_penalty: float = 0.0
    cost_penalty: float = 0.0
    solver: Literal["greedy", "dp"] = "greedy"
    cost_quantum: float = 0.01
    max_dp_budget_units: int = 10000


@dataclass(frozen=True)
class Decision:
    region_id: str
    gland: str
    action: Action
    predicted_gain: float
    predicted_harm_probability: float
    predicted_cost: float
    objective_gain: float


@dataclass(frozen=True)
class RoutingResult:
    decisions: tuple[Decision, ...]
    budget: float
    expected_total_cost: float
    expected_total_gain: float
    expected_total_objective: float
    solver: str

    @property
    def selected_count(self) -> int:
        return sum(d.action != "skip" for d in self.decisions)


def _finite_nonnegative(number: float, name: str) -> None:
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative, got {number}")


def _validate(regions: Sequence[RegionEstimate], config: RoutingConfig) -> None:
    _finite_nonnegative(config.budget, "budget")
    _finite_nonnegative(config.min_event_gain, "min_event_gain")
    _finite_nonnegative(config.min_spike_vs_analog_gain, "min_spike_vs_analog_gain")
    _finite_nonnegative(config.harm_penalty, "harm_penalty")
    _finite_nonnegative(config.cost_penalty, "cost_penalty")
    if not 0 <= config.max_harm_probability <= 1:
        raise ValueError("max_harm_probability must be a probability")
    if config.solver not in ("greedy", "dp"):
        raise ValueError("unknown solver")
    if not math.isfinite(config.cost_quantum) or config.cost_quantum <= 0:
        raise ValueError("cost_quantum must be positive finite")
    if config.max_dp_budget_units < 1:
        raise ValueError("max_dp_budget_units must be positive")
    ids = set()
    for region in regions:
        if not region.region_id or region.region_id in ids:
            raise ValueError("region_id must be unique and nonempty")
        ids.add(region.region_id)
        if region.gland not in ("left", "right"):
            raise ValueError("invalid gland")
        for field in ("gain_analog", "gain_spike", "event_gain",
                      "spike_vs_analog_gain"):
            if not math.isfinite(getattr(region, field)):
                raise ValueError(f"{region.region_id}.{field} must be finite")
        for field in ("harm_prob_analog", "harm_prob_spike"):
            probability = getattr(region, field)
            if not 0 <= probability <= 1 or not math.isfinite(probability):
                raise ValueError(f"{region.region_id}.{field} invalid probability")
        for field in ("cost_analog", "cost_spike"):
            cost = getattr(region, field)
            if cost <= 0 or not math.isfinite(cost):
                raise ValueError(f"{region.region_id}.{field} must be positive finite")


def _options(region: RegionEstimate, config: RoutingConfig) -> list[Decision]:
    result: list[Decision] = [
        Decision(region.region_id, region.gland, "skip", 0., 0., 0., 0.)
    ]
    candidates = (
        ("analog", region.gain_analog, region.harm_prob_analog,
         region.cost_analog, True),
        ("spike", region.gain_spike, region.harm_prob_spike,
         region.cost_spike,
         region.event_gain > config.min_event_gain and
         region.spike_vs_analog_gain > config.min_spike_vs_analog_gain),
    )
    for action, gain, harm, cost, eligible in candidates:
        utility = gain - config.harm_penalty * harm - config.cost_penalty * cost
        if (eligible and gain > 0 and utility > 0 and
                harm <= config.max_harm_probability and
                cost <= config.budget + 1e-12):
            result.append(Decision(region.region_id, region.gland, action,
                                   gain, harm, cost, utility))
    return result


def _route_greedy(options: list[list[Decision]], budget: float) -> list[Decision]:
    """Practical approximate multiple-choice knapsack; not globally optimal."""
    all_candidates = [
        (d, index) for index, choices in enumerate(options)
        for d in choices if d.action != "skip"
    ]
    # Deterministic tie-breaking, independent of hash/random seeds.
    all_candidates.sort(
        key=lambda item: (-item[0].objective_gain / item[0].predicted_cost,
                          -item[0].objective_gain, item[0].region_id,
                          item[0].action)
    )
    decisions = [choices[0] for choices in options]
    selected = set()
    remaining = budget
    for item, index in all_candidates:
        if index not in selected and item.predicted_cost <= remaining + 1e-12:
            decisions[index] = item
            selected.add(index)
            remaining -= item.predicted_cost
    return decisions


def _route_dp(options: list[list[Decision]], config: RoutingConfig) -> list[Decision]:
    """Exact multiple-choice knapsack on conservatively quantized costs.

    Every action cost is ceiled to a whole quantum. The output is therefore
    never over budget, but need not be optimal for the original real costs.
    Computational complexity O(n_regions * n_budget_units * 3).
    """
    units = math.floor(config.budget / config.cost_quantum + 1e-10)
    if units > config.max_dp_budget_units:
        raise ValueError(
            f"DP budget of {units} units exceeds limit "
            f"{config.max_dp_budget_units}; raise quantum or use greedy"
        )
    # state: used_units -> (total_objective, tuple_of_action_decisions)
    states: dict[int, tuple[float, tuple[Decision, ...]]] = {0: (0., ())}
    for group in options:
        updated: dict[int, tuple[float, tuple[Decision, ...]]] = {}
        for used, (score, history) in states.items():
            for decision in group:
                charge = math.ceil(decision.predicted_cost / config.cost_quantum - 1e-12)
                next_units = used + charge
                if next_units > units:
                    continue
                next_score = score + decision.objective_gain
                prev = updated.get(next_units)
                next_history = history + (decision,)
                if prev is None or next_score > prev[0] + 1e-12:
                    updated[next_units] = (next_score, next_history)
                # Exact score ties retain the first deterministic path.
        states = updated
    if not states:
        raise RuntimeError("infeasible: skip should always keep state 0")
    # Prefer more objective; on a tie prefer lower cost / fewer selected.
    _, (_, best) = max(
        states.items(),
        key=lambda pair: (pair[1][0], -pair[0],
                          -sum(d.action != "skip" for d in pair[1][1]))
    )
    return list(best)


def route_actions(
    regions: Sequence[RegionEstimate], config: RoutingConfig
) -> RoutingResult:
    """Choose one safe, event-attributable action per candidate ROI.

    Risk control is based on an *estimated* probability calibrated on held-out
    development patients. It does not guarantee error-free segmentations.
    The modeled costs and gains are predictions, not measured experimental
    outcomes. Measure actual energy/quality independently.
    """
    _validate(regions, config)
    options = [_options(region, config) for region in regions]
    if config.solver == "dp":
        chosen = _route_dp(options, config)
    else:
        chosen = _route_greedy(options, config.budget)
    cost = sum(d.predicted_cost for d in chosen)
    if cost > config.budget + 1e-9:
        raise AssertionError("policy allocated more than its budget")
    return RoutingResult(
        tuple(chosen), config.budget, cost,
        sum(d.predicted_gain for d in chosen),
        sum(d.objective_gain for d in chosen),
        config.solver,
    )
