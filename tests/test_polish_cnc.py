"""4-axis spherical-tool polish paths (A = optical axis, Z = radial)."""
from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import default_params
from export_cad import (
    LensSpec,
    ball_tool_centers,
    export_lens_cnc_profile,
    flange_cut_z,
    meridian_air_normals,
    optical_meridian,
    polish_machine_coords,
    polish_toolpath,
)


def _pcx_spec():
    # Convex toward source: R1 > 0, rear plano
    return LensSpec(R1=40.0, R2=0.0, thickness=6.0, aperture=10.0, z_front=3.0)


class TestOpticalMeridian(unittest.TestCase):
    def test_starts_on_axis_and_ends_at_ca(self):
        spec = _pcx_spec()
        mer = optical_meridian(spec, "front")
        self.assertGreater(len(mer), 8)
        self.assertAlmostEqual(mer[0, 0], 0.0, places=6)
        self.assertAlmostEqual(mer[-1, 0], 10.0, places=5)
        # Front vertex at z_front
        self.assertAlmostEqual(mer[0, 1], 3.0, places=5)

    def test_rear_vertex_at_front_plus_thickness(self):
        spec = _pcx_spec()
        mer = optical_meridian(spec, "rear")
        self.assertAlmostEqual(mer[0, 1], 9.0, places=5)


class TestMeridianNormalsAndBall(unittest.TestCase):
    def test_front_normals_point_to_air_minus_z(self):
        spec = _pcx_spec()
        mer = optical_meridian(spec, "front", n=24)
        nrm = meridian_air_normals(mer[:, 0], mer[:, 1], "front")
        # On axis the front air normal is -Z
        self.assertLess(nrm[0, 1], -0.7)
        # Unit length
        for row in nrm:
            self.assertAlmostEqual(math.hypot(row[0], row[1]), 1.0, places=5)

    def test_rear_normals_point_to_air_plus_z(self):
        spec = _pcx_spec()
        mer = optical_meridian(spec, "rear", n=16)
        nrm = meridian_air_normals(mer[:, 0], mer[:, 1], "rear")
        self.assertGreater(nrm[0, 1], 0.7)

    def test_ball_center_is_tool_radius_from_surface(self):
        spec = _pcx_spec()
        mer = optical_meridian(spec, "front", n=20)
        nrm = meridian_air_normals(mer[:, 0], mer[:, 1], "front")
        rt = 5.0
        tc = ball_tool_centers(mer[:, 0], mer[:, 1], nrm[:, 0], nrm[:, 1], rt)
        for i in range(len(mer)):
            d = math.hypot(tc[i, 0] - mer[i, 0], tc[i, 1] - mer[i, 1])
            self.assertAlmostEqual(d, rt, places=5)


class TestMachineFrame(unittest.TestCase):
    def test_cut_face_origin_puts_front_vertex_at_half_thickness_plus_tool(self):
        spec = _pcx_spec()
        mer = optical_meridian(spec, "front", n=12)
        nrm = meridian_air_normals(mer[:, 0], mer[:, 1], "front")
        rt = 4.0
        tc = ball_tool_centers(mer[:, 0], mer[:, 1], nrm[:, 0], nrm[:, 1], rt)
        xyz = polish_machine_coords(spec, "front", tc[:, 0], tc[:, 1], x_origin="cut")
        # Vertex: X = t/2 + Rt, Y = 0, Z = 0
        self.assertAlmostEqual(xyz[0, 1], 0.0, places=6)
        self.assertAlmostEqual(xyz[0, 2], 0.0, places=5)
        self.assertAlmostEqual(xyz[0, 0], 3.0 + rt, places=4)

    def test_radial_z_follows_tool_radius_at_rim(self):
        spec = _pcx_spec()
        mer = optical_meridian(spec, "front", n=16)
        nrm = meridian_air_normals(mer[:, 0], mer[:, 1], "front")
        tc = ball_tool_centers(mer[:, 0], mer[:, 1], nrm[:, 0], nrm[:, 1], 5.0)
        xyz = polish_machine_coords(spec, "front", tc[:, 0], tc[:, 1], x_origin="cut")
        self.assertGreater(xyz[-1, 2], 8.0)


class TestPolishToolpath(unittest.TestCase):
    def test_helical_a_increases(self):
        spec = _pcx_spec()
        path = polish_toolpath(spec, "front", tool_radius_mm=5.0, stepover_mm=0.5, strategy="helical")
        a = path["a_deg"]
        self.assertGreater(len(a), 8)
        self.assertGreater(float(a[-1]), float(a[0]) + 90.0)
        self.assertTrue(all(a[i] <= a[i + 1] + 1e-9 for i in range(len(a) - 1)))

    def test_rings_spin_at_stations(self):
        spec = _pcx_spec()
        path = polish_toolpath(
            spec, "front", tool_radius_mm=5.0, stepover_mm=1.0, strategy="rings", revs_per_ring=2.0
        )
        self.assertGreater(path["n_spins"], 4)
        self.assertGreater(float(path["a_deg"][-1]), 360.0)


class TestExportPolishNc(unittest.TestCase):
    def test_writes_a_axis_and_rewrites_gcode_suffix(self):
        p = default_params()
        p["mla"]["enabled"] = False
        for e in p["elements"][1:]:
            e["enabled"] = False
        p["cad_polish_tool_dia_mm"] = 10.0
        p["cad_polish_stepover_mm"] = 0.8
        p["cad_polish_surface"] = "front"
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "half.gcode"
            out = export_lens_cnc_profile(p, dest)
            self.assertEqual(out.suffix.lower(), ".nc")
            body = out.read_text(encoding="ascii")
            self.assertIn("G21", body)
            self.assertIn("G90", body)
            self.assertRegex(body, r"\bA[0-9.-]")
            self.assertIn("M3", body)
            self.assertIn("M30", body)
            self.assertIn("radial", body.lower())

    def test_both_surfaces_write_two_files(self):
        p = default_params()
        p["mla"]["enabled"] = False
        for e in p["elements"][1:]:
            e["enabled"] = False
        p["cad_polish_surface"] = "both"
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "job.nc"
            out = export_lens_cnc_profile(p, dest)
            self.assertTrue(out.is_file())
            front = dest.with_name("job_front.nc")
            rear = dest.with_name("job_rear.nc")
            self.assertTrue(front.is_file())
            self.assertTrue(rear.is_file())


if __name__ == "__main__":
    unittest.main()
