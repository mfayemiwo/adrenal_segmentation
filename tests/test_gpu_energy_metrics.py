"""CPU-only tests for AMD-SMI energy measurement maths."""
import unittest
from scripts.measure_gpu_energy import get_energy_joules, get_power_watts, integrate_power, normalize_bdf, select_gpu


class FakeSMI:
    def amdsmi_get_energy_count(self, handle):
        return {"energy_accumulator": 1_500_000, "counter_resolution": 2}

    def amdsmi_get_power_info(self, handle):
        return {"current_socket_power": 300.0, "average_socket_power": 250.0}


class EnergyMetricTests(unittest.TestCase):
    def test_energy_counter_microjoules_converted_to_joules(self):
        self.assertEqual(get_energy_joules(FakeSMI(), object()), 3.0)

    def test_prefer_current_socket_power(self):
        self.assertEqual(get_power_watts(FakeSMI(), object()), 300.0)

    def test_trapezoid_integral_joules(self):
        samples = [
            {"t_seconds": 0.0, "power_w": 100.0},
            {"t_seconds": 2.0, "power_w": 200.0},
        ]
        self.assertAlmostEqual(integrate_power(samples), 300.0)

    def test_normalize_amdsmi_and_hip_pci_bus_ids(self):
        self.assertEqual(normalize_bdf("4a:00.0"), "0000:4a:00.0")
        self.assertEqual(normalize_bdf("0000:4A:00.0"), "0000:4a:00.0")
        self.assertIsNone(normalize_bdf("not_a_bdf"))

    def test_select_single_visible_device(self):
        self.assertEqual(select_gpu(None, ["handle"], None),
                         (0, "only_one_amdsmi_device_visible"))

    def test_no_readings_is_unknown_not_zero_energy(self):
        self.assertIsNone(integrate_power([{"t_seconds": 0.0, "power_w": None},
                                           {"t_seconds": 1.0, "power_w": None}]))


if __name__ == "__main__":
    unittest.main()
