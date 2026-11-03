"""ACTS-3D conditional-execution contract (framework-independent reference).

A routed action is NOT equivalent to computing all experts and multiplying
outputs by zero. This orchestrator invokes only the experts actually selected
by the ACTS-3D decision policy.

Real GPU integration must replace the callback implementations with batched
ROI extraction, analog/SNN execution and baseline-preserving logits fusion.
Inputs and outputs here are deliberately framework-agnostic to keep the
algorithm invariants CPU-testable.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from src.algorithms.acts3d_policy import Decision


@dataclass(frozen=True)
class DispatchReport:
    output: Any
    analog_region_ids: tuple[str, ...]
    spike_region_ids: tuple[str, ...]


def execute_selected(
    baseline_output: Any,
    decisions: Sequence[Decision],
    *,
    analog_batch: Callable[[tuple[str, ...]], Mapping[str, Any]],
    spike_batch: Callable[[tuple[str, ...]], Mapping[str, Any]],
    merge_patch: Callable[[Any, str, Any], None],
) -> DispatchReport:
    """Dispatch only the chosen expert, preserving the baseline elsewhere.

    A caller must guarantee: no overlapping ROI labels unless fusion order
    is explicitly resolved, correct spatial/original-CT coordinates, and
    absolutely no use of ground truth inside these callbacks.

    A batch callback must return exactly one prediction for every selected ROI.
    """
    if len({d.region_id for d in decisions}) != len(decisions):
        raise ValueError("duplicate region IDs")
    analog = tuple(d.region_id for d in decisions if d.action == "analog")
    spike = tuple(d.region_id for d in decisions if d.action == "spike")
    if any(d.action not in ("skip", "analog", "spike") for d in decisions):
        raise ValueError("invalid selected action")

    output = copy.deepcopy(baseline_output)
    for names, callback in ((analog, analog_batch), (spike, spike_batch)):
        if not names:
            continue   # MUST not touch an unselected expert.
        updates = callback(names)
        if set(updates) != set(names):
            raise ValueError("expert returned wrong ROI IDs")
        for name in names:
            merge_patch(output, name, updates[name])
    return DispatchReport(output=output, analog_region_ids=analog,
                          spike_region_ids=spike)
