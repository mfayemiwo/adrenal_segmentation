"""CPU-only tests of ACTS-3D algorithmic routing invariants, no GPU required."""
import unittest

from src.algorithms.acts3d_policy import (
    RegionEstimate, RoutingConfig, route_actions, region_from_outcomes,
)


def region(name="r1", *, gland="left", analog=0.1, spike=0.2, event=0.1,
           versus=0.1, ha=0.01, hs=0.01, ca=2.0, cs=3.0):
    return RegionEstimate(name, gland, analog, spike, event, versus,
                          ha, hs, ca, cs)


class ACTS3DPolicyTests(unittest.TestCase):
    def test_intervention_potentials_derive_consistent_event_and_expert_gains(self):
        r = region_from_outcomes(
            "counterfactual", "left", loss_baseline=1.0,
            loss_analog=0.8, loss_spike=0.6, loss_spike_zero_events=0.95,
            harm_prob_analog=0.01, harm_prob_spike=0.01,
            cost_analog=1.0, cost_spike=2.0)
        self.assertAlmostEqual(r.gain_analog, 0.2)
        self.assertAlmostEqual(r.gain_spike, 0.4)
        self.assertAlmostEqual(r.event_gain, 0.35)
        self.assertAlmostEqual(r.spike_vs_analog_gain, 0.2)
        out = route_actions([r], RoutingConfig(budget=2., solver="dp", cost_quantum=1.))
        self.assertEqual(out.decisions[0].action, "spike")

    def test_real_events_choose_spike_when_excess_advantage_present(self):
        out = route_actions([region()], RoutingConfig(budget=4., solver="dp", cost_quantum=1.))
        self.assertEqual(out.decisions[0].action, "spike")
        self.assertEqual(out.expected_total_cost, 3.)

    def test_zero_event_effect_forbids_spiking_even_if_spike_gain_high(self):
        out = route_actions([region(event=0.0)], RoutingConfig(budget=4.))
        self.assertEqual(out.decisions[0].action, "analog")

    def test_analog_control_beats_spiking_if_specific_advantage_nonpositive(self):
        out = route_actions([region(analog=0.3, spike=0.1, versus=-0.2)],
                            RoutingConfig(budget=5., solver="dp", cost_quantum=1.))
        self.assertEqual(out.decisions[0].action, "analog")

    def test_harm_calibration_rejects_unsafe_refinement(self):
        out = route_actions([region(ha=0.20, hs=0.25)],
                            RoutingConfig(budget=100., max_harm_probability=0.05))
        self.assertEqual(out.decisions[0].action, "skip")
        self.assertEqual(out.expected_total_cost, 0.)

    def test_empty_and_zero_budget_are_safe(self):
        self.assertEqual(route_actions([], RoutingConfig(budget=0.)).selected_count, 0)
        out = route_actions([region()], RoutingConfig(budget=0.))
        self.assertEqual(out.decisions[0].action, "skip")

    def test_multiple_choice_budget_cannot_select_both_experts_per_region(self):
        rows = [
            region("a", spike=0.8, ca=3, cs=4),
            region("b", gland="right", spike=0.6, ca=3, cs=4)
        ]
        for solver in ("greedy", "dp"):
            out = route_actions(rows, RoutingConfig(
                budget=4., solver=solver, cost_quantum=1.))
            self.assertEqual(len(out.decisions), len(rows))
            self.assertEqual(len({d.region_id for d in out.decisions}), 2)
            self.assertLessEqual(out.expected_total_cost, 4.000001)
            self.assertLessEqual(out.selected_count, 1)

    def test_dynamic_programming_beats_greedy_counterexample(self):
        rows = [
            region("one", analog=13, spike=0, ca=7, cs=999),
            region("two", analog=9, spike=0, ca=5, cs=999),
            region("three", analog=9, spike=0, ca=5, cs=999),
        ]
        greedy = route_actions(rows, RoutingConfig(budget=10., solver="greedy"))
        optimal = route_actions(rows, RoutingConfig(
            budget=10., solver="dp", cost_quantum=1.))
        self.assertEqual(greedy.expected_total_gain, 13.)
        self.assertEqual(optimal.expected_total_gain, 18.)
        self.assertEqual(optimal.expected_total_cost, 10.)

    def test_calibrated_harm_penalty_can_make_action_not_worthwhile(self):
        out = route_actions([region(analog=0.05, spike=0.07, ha=0.04, hs=0.04)],
                            RoutingConfig(budget=10., harm_penalty=3.0))
        self.assertEqual(out.decisions[0].action, "skip")

    def test_cost_budget_must_use_same_units(self):
        with self.assertRaises(ValueError):
            route_actions([region(ca=-1)], RoutingConfig(budget=10.))
        with self.assertRaises(ValueError):
            route_actions([region()], RoutingConfig(budget=-1.))

    def test_dp_cost_quantization_is_conservative(self):
        out = route_actions([
            region("x", analog=3., spike=0., ca=1.01),
            region("y", analog=3., spike=0., ca=1.01),
        ], RoutingConfig(budget=2.03, solver="dp", cost_quantum=0.1))
        self.assertEqual(out.selected_count, 1)
        self.assertLessEqual(out.expected_total_cost, 2.03)

    def test_spikes_not_selected_from_spike_count_alone(self):
        # A simulated high firing rate is NOT an input to the router:
        # only calibrated, positive S-versus-analog and event-intervention
        # improvement can make the spiking branch eligible.
        out = route_actions([
            region(event=-0.1, versus=0.9, spike=1.0)],
            RoutingConfig(budget=10.))
        self.assertEqual(out.decisions[0].action, "analog")


if __name__ == "__main__":
    unittest.main()
