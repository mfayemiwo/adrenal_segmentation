"""CPU-only AMD-SMI import-location regression tests; no real GPU needed."""
from __future__ import annotations
import importlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.measure_gpu_energy import load_amdsmi


class AMDImportDiscoveryTests(unittest.TestCase):
    def test_loads_binding_from_rocm_share_without_pip_install(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "rocm-6.3.3"
            package = root / "share" / "amd_smi" / "amdsmi"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("TEST_FIXTURE = 633\n")
            old_sys_path = sys.path[:]
            old_module = sys.modules.pop("amdsmi", None)
            try:
                with patch.dict(os.environ, {"ROCM_PATH": str(root)}):
                    result = load_amdsmi()
                    self.assertEqual(result.TEST_FIXTURE, 633)
                    self.assertEqual(
                        Path(result.__file__).resolve(),
                        (package / "__init__.py").resolve(),
                    )
            finally:
                sys.path[:] = old_sys_path
                sys.modules.pop("amdsmi", None)
                if old_module is not None:
                    sys.modules["amdsmi"] = old_module

    def test_refuses_mismatched_user_gpu_index(self):
        from scripts.measure_gpu_energy import select_gpu
        with self.assertRaises(ValueError):
            select_gpu(None, [object()], gpu_index=2)

    def test_rocm_vram_usage_megabytes_not_bytes(self):
        from scripts.measure_gpu_energy import get_vram_bytes
        class Dummy:
            @staticmethod
            def amdsmi_get_gpu_vram_usage(_):
                return {"vram_used": 6620, "vram_total": 196300}
        self.assertEqual(get_vram_bytes(Dummy(), None), 6_620_000_000)
        self.assertAlmostEqual(get_vram_bytes(Dummy(), None) / (1024 ** 3),
                               6.165355443954468, places=4)

    def test_direct_energy_uses_documented_microjoule_resolution(self):
        from scripts.measure_gpu_energy import get_energy_joules
        class Dummy:
            @staticmethod
            def amdsmi_get_energy_count(_):
                return {"energy_accumulator": 6000000,
                        "counter_resolution": 2.0}
        self.assertAlmostEqual(get_energy_joules(Dummy(), None), 12.)

if __name__ == "__main__":
    unittest.main()
