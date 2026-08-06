"""Save / load design round-trip."""
from __future__ import annotations
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import default_params
from design_io import save_design, load_design, design_document


class TestDesignIO(unittest.TestCase):
    def test_roundtrip_elements_and_source(self):
        p = default_params()
        p["source"]["mode"] = "single"
        p["fov_width"] = 42.0
        p["elements"][0].update(
            enabled=True, R1=33.3, R2=-44.4, surface_mode="cylinder_x", material="N_BK7"
        )
        p["elements"][1].update(
            enabled=True, R1=25.0, R2=-25.0, surface_mode="cylinder_y"
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "group.json"
            save_design(path, p, name="my_group", notes="test stack")
            p2, meta = load_design(path)
        self.assertEqual(meta["name"], "my_group")
        self.assertEqual(meta["notes"], "test stack")
        self.assertEqual(p2["source"]["mode"], "single")
        self.assertAlmostEqual(p2["fov_width"], 42.0)
        self.assertAlmostEqual(p2["elements"][0]["R1"], 33.3)
        self.assertEqual(p2["elements"][0]["surface_mode"], "cylinder_x")
        self.assertEqual(p2["elements"][1]["surface_mode"], "cylinder_y")
        self.assertTrue(p2["elements"][0]["enabled"])
        self.assertTrue(p2["elements"][1]["enabled"])
        self.assertEqual(len(p2["elements"]), 5)

    def test_bare_params_legacy(self):
        p = default_params()
        doc = {"source": p["source"], "elements": p["elements"], "lens_z_start": 5.0,
               "target_z": 90, "fov_width": 10, "fov_height": 8, "fov_cx": 0, "fov_cy": 0,
               "map_half_w": 20, "map_half_h": 16, "map_res": 64, "total_rays": 1000,
               "display_rays": 100, "custom_n": 1.5, "apply_fresnel": True}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bare.json"
            path.write_text(__import__("json").dumps(doc), encoding="utf-8")
            p2, meta = load_design(path)
        self.assertEqual(meta["format"], "bare_params")
        self.assertAlmostEqual(p2["lens_z_start"], 5.0)

    def test_document_wrapper(self):
        p = default_params()
        doc = design_document(p, name="x", notes="y")
        self.assertEqual(doc["format"], "optiflux_design")
        self.assertIn("params", doc)
        self.assertEqual(doc["name"], "x")

    def test_blockers_roundtrip(self):
        p = default_params()
        p["blockers"] = [
            {
                "enabled": True,
                "label": "Stop",
                "z": 22.5,
                "shape": "circle",
                "orient": "vertical",
                "outer_w": 18.0,
                "outer_h": 18.0,
                "inner_w": 6.0,
                "inner_h": 6.0,
                "length": 1.0,
                "x0": 0.0,
                "y0": 0.0,
                "thickness": 1.0,
            },
            {
                "enabled": True,
                "label": "Barrel",
                "z": 40.0,
                "shape": "circle",
                "orient": "tube",
                "outer_w": 14.0,
                "outer_h": 14.0,
                "inner_w": 0.0,
                "length": 55.0,
                "x0": 0.0,
                "y0": 0.0,
                "thickness": 1.0,
            },
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "with_stop.json"
            save_design(path, p, name="stop_design")
            p2, _ = load_design(path)
        self.assertEqual(len(p2["blockers"]), 2)
        self.assertAlmostEqual(p2["blockers"][0]["z"], 22.5)
        self.assertEqual(p2["blockers"][0]["shape"], "circle")
        self.assertEqual(p2["blockers"][0]["orient"], "vertical")
        self.assertAlmostEqual(p2["blockers"][0]["inner_w"], 6.0)
        self.assertEqual(p2["blockers"][1]["orient"], "tube")
        self.assertAlmostEqual(p2["blockers"][1]["length"], 55.0)


if __name__ == "__main__":
    unittest.main()
