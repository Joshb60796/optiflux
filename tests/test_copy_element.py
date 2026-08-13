"""Copy one lens-stack slot onto another (UI “Copy from” dropdown)."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import (
    assemble_surfaces,
    copy_element,
    default_params,
    pad_elements,
)


class TestCopyElement(unittest.TestCase):
    def test_copies_radii_material_and_enables_dest(self):
        p = default_params()
        p["elements"][0].update(
            enabled=True,
            R1=22.0,
            R2=-33.0,
            thickness=7.5,
            aperture=14.0,
            air_after=4.0,
            material="N_BK7",
            surface_mode="biconic",
            R1y=18.0,
            R2y=-28.0,
            k1=-0.5,
            A4_1=1e-5,
        )
        p["elements"][1]["enabled"] = False
        out = copy_element(p["elements"], src=0, dst=1)
        self.assertTrue(out[1]["enabled"])
        self.assertEqual(out[1]["R1"], 22.0)
        self.assertEqual(out[1]["R2"], -33.0)
        self.assertEqual(out[1]["thickness"], 7.5)
        self.assertEqual(out[1]["aperture"], 14.0)
        self.assertEqual(out[1]["material"], "N_BK7")
        self.assertEqual(out[1]["surface_mode"], "biconic")
        self.assertEqual(out[1]["R1y"], 18.0)
        self.assertEqual(out[1]["k1"], -0.5)
        # Source unchanged
        self.assertEqual(out[0]["R1"], 22.0)
        self.assertTrue(out[0]["enabled"])

    def test_same_index_is_noop(self):
        p = default_params()
        before = [dict(e) for e in p["elements"]]
        out = copy_element(p["elements"], src=0, dst=0)
        self.assertEqual(out[0]["R1"], before[0]["R1"])

    def test_copied_element_sits_after_source_air_gap(self):
        p = default_params()
        p["elements"][0].update(enabled=True, thickness=6.0, air_after=5.0)
        p["elements"][1]["enabled"] = False
        p["lens_z_start"] = 3.0
        elems = copy_element(p["elements"], src=0, dst=1)
        surfs = assemble_surfaces(elems, 3.0)
        # E1 front at 3, rear at 9; air 5 → E2 front at 14
        e1_front = next(s.z_vertex for s in surfs if s.label == "E1S1")
        e2_front = next(s.z_vertex for s in surfs if s.label == "E2S1")
        self.assertAlmostEqual(e1_front, 3.0, places=5)
        self.assertAlmostEqual(e2_front, 3.0 + 6.0 + 5.0, places=5)

    def test_out_of_range_raises(self):
        p = default_params()
        with self.assertRaises(IndexError):
            copy_element(p["elements"], src=0, dst=99)

    def test_does_not_alias_source_dict(self):
        p = default_params()
        out = copy_element(p["elements"], src=0, dst=2)
        out[2]["R1"] = 1.23
        self.assertNotEqual(out[0]["R1"], 1.23)


if __name__ == "__main__":
    unittest.main()
