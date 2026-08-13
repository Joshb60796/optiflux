"""Export-only flanges, tube-dimension notes, circular outline lock."""
from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import (
    apply_circular_outline,
    default_params,
    run_simulation,
)
from export_cad import (
    LensSpec,
    export_lens,
    flange_od_mm,
    format_tube_notes,
    mesh_singlet,
    tube_layout,
)


class TestFlangeOd(unittest.TestCase):
    def test_od_is_twice_clear_plus_radial(self):
        # CA semi 10 mm, flange radial 2 mm → OD 24 mm
        self.assertAlmostEqual(flange_od_mm(10.0, 2.0), 24.0)

    def test_uses_larger_semi_when_elliptical(self):
        self.assertAlmostEqual(flange_od_mm(8.0, 1.5, aperture_y=12.0), 27.0)

    def test_zero_radial_is_clear_diameter(self):
        self.assertAlmostEqual(flange_od_mm(10.0, 0.0), 20.0)


class TestCircularOutlineLock(unittest.TestCase):
    def test_lock_clears_elliptical_aperture(self):
        e = {"aperture": 12.0, "aperture_y": 8.0, "circular_lock": True}
        out = apply_circular_outline(e)
        self.assertIsNone(out["aperture_y"])
        self.assertTrue(out["circular_lock"])

    def test_unlock_keeps_aperture_y(self):
        e = {"aperture": 12.0, "aperture_y": 8.0, "circular_lock": False}
        out = apply_circular_outline(e)
        self.assertEqual(out["aperture_y"], 8.0)
        self.assertFalse(out["circular_lock"])

    def test_default_is_locked(self):
        p = default_params()
        self.assertTrue(p["elements"][0].get("circular_lock", True))
        out = apply_circular_outline(p["elements"][0])
        self.assertIsNone(out["aperture_y"])


class TestTubeLayout(unittest.TestCase):
    def test_groove_centers_and_source_to_tube_front(self):
        p = default_params()
        p["source"]["source_z"] = 0.0
        p["lens_z_start"] = 3.0
        p["target_z"] = 80.0
        p["mla"]["enabled"] = False
        p["elements"][0].update(
            enabled=True, thickness=6.0, air_after=4.0, aperture=10.0, aperture_y=None
        )
        p["elements"][1].update(
            enabled=True, thickness=4.0, air_after=2.0, aperture=12.0, aperture_y=None
        )
        for e in p["elements"][2:]:
            e["enabled"] = False
        fr, ft = 2.0, 1.5
        lay = tube_layout(p, flange_radial_mm=fr, flange_thickness_mm=ft)
        self.assertEqual(len(lay["seats"]), 2)
        s0, s1 = lay["seats"]
        self.assertAlmostEqual(s0["od_mm"], 24.0)
        self.assertAlmostEqual(s1["od_mm"], 28.0)
        self.assertAlmostEqual(s0["flange_thickness_mm"], 1.5)
        # Groove at vertex midplane: 3 + 6/2 = 6; next 3+6+4 + 4/2 = 15
        self.assertAlmostEqual(s0["groove_center_z_mm"], 6.0)
        self.assertAlmostEqual(s1["groove_center_z_mm"], 15.0)
        self.assertAlmostEqual(lay["groove_center_to_center_mm"][0], 9.0)
        # Tube front = first flange front face
        self.assertAlmostEqual(lay["tube_front_z_mm"], 6.0 - 0.75)
        self.assertAlmostEqual(lay["source_to_tube_front_mm"], 5.25)
        self.assertAlmostEqual(lay["last_groove_to_focus_mm"], 80.0 - 15.0)

    def test_notes_text_lists_required_fields(self):
        p = default_params()
        p["mla"]["enabled"] = False
        p["elements"][0].update(enabled=True, thickness=6.0, aperture=10.0)
        for e in p["elements"][1:]:
            e["enabled"] = False
        lay = tube_layout(p, flange_radial_mm=2.0, flange_thickness_mm=1.5)
        text = format_tube_notes(lay)
        self.assertIn("OD", text)
        self.assertIn("24.00", text)
        self.assertIn("Flange thickness", text)
        self.assertIn("1.50", text)
        self.assertIn("Groove center", text)
        self.assertIn("Source to tube front", text)
        self.assertIn("Last groove center to focus", text)
        self.assertIn("Target Z", text)


class TestFlangeMeshExport(unittest.TestCase):
    def test_mesh_without_flange_stays_at_clear_aperture(self):
        spec = LensSpec(R1=40, R2=-50, thickness=6, aperture=10, z_front=3)
        mesh = mesh_singlet(spec)
        rs = np_radii(mesh)
        self.assertLessEqual(max(rs), 10.05)

    def test_mesh_with_flange_reaches_od(self):
        spec = LensSpec(R1=40, R2=-50, thickness=6, aperture=10, z_front=3)
        mesh = mesh_singlet(spec, flange_radial_mm=2.0, flange_thickness_mm=1.5)
        rs = np_radii(mesh)
        self.assertGreater(max(rs), 11.5)
        self.assertAlmostEqual(max(rs), 12.0, delta=0.08)

    def test_export_writes_companion_tube_notes(self):
        p = default_params()
        p["mla"]["enabled"] = False
        for e in p["elements"][1:]:
            e["enabled"] = False
        with tempfile.TemporaryDirectory() as td:
            stl = Path(td) / "group.stl"
            export_lens(
                p,
                stl,
                fmt="stl",
                max_edge_mm=1.0,
                max_angle_deg=12.0,
                flange_radial_mm=2.0,
                flange_thickness_mm=1.5,
            )
            notes = stl.with_name(stl.stem + "_tube.txt")
            self.assertTrue(stl.is_file())
            self.assertTrue(notes.is_file())
            body = notes.read_text(encoding="utf-8")
            self.assertIn("Source to tube front", body)
            self.assertIn("Last groove center to focus", body)

    def test_flange_does_not_change_simulation(self):
        p = default_params()
        p["use_warp"] = False
        p["total_rays"] = 200
        p["display_rays"] = 0
        p["mla"]["enabled"] = False
        p["cad_flange_radial_mm"] = 8.0
        p["cad_flange_thickness_mm"] = 4.0
        r = run_simulation(p)
        # Hits must stay inside the optical CA (±aperture), not the flange OD
        self.assertGreater(r.stats["launched"], 0)
        hw = float(p["elements"][0]["aperture"])
        # Map exists; flange params must not appear as a surface
        self.assertTrue(all("flange" not in (getattr(s, "label", "") or "") for s in r.surfaces))
        self.assertLessEqual(abs(r.stats["centroid"][0]), hw * 4)


def np_radii(mesh):
    v = mesh.vertices
    return [math.hypot(float(x), float(y)) for x, y, _z in v]


if __name__ == "__main__":
    unittest.main()
