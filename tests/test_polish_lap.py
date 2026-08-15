"""Cylindrical abrasive polish-lap STL (negative optical face + 1/4-20 tap)."""
from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import default_params
from export_cad import (
    LAP_HOLE_DEPTH_MM,
    LAP_HOLE_KEEP_MM,
    UNC_1_4_20_TAP_DRILL_MM,
    LensSpec,
    export_polish_lap,
    lap_jobs,
    lap_output_paths,
    mesh_polish_lap,
    mesh_signed_volume,
    parse_lap_faces,
    stack_singlet_specs,
)


def _pcx():
    return LensSpec(R1=40.0, R2=0.0, thickness=6.0, aperture=10.0, z_front=3.0)


def _biconvex():
    return LensSpec(R1=30.0, R2=-30.0, thickness=6.0, aperture=10.0, z_front=0.0)


def _pcv():
    # Concave toward the source
    return LensSpec(R1=-40.0, R2=0.0, thickness=5.0, aperture=10.0, z_front=0.0)


def _radii_xy(pts):
    return np.hypot(pts[:, 0], pts[:, 1])


def _working_face(mesh):
    """Optical-face vertices (exclude the well lip and the tap hole)."""
    z = mesh.vertices[:, 2]
    z_max = float(z.max())
    # Well lip sits well_extra (3 mm) above the highest optical point
    keep = (z > LAP_HOLE_DEPTH_MM + 0.5) & (z < z_max - 1.2)
    return mesh.vertices[keep]


class TestParseLapFaces(unittest.TestCase):
    def test_front_rear_both(self):
        self.assertEqual(parse_lap_faces("front"), ["front"])
        self.assertEqual(parse_lap_faces("rear"), ["rear"])
        self.assertEqual(parse_lap_faces("both"), ["front", "rear"])

    def test_default_is_front(self):
        self.assertEqual(parse_lap_faces(""), ["front"])
        self.assertEqual(parse_lap_faces(None), ["front"])


class TestLapJobsAndPaths(unittest.TestCase):
    def test_jobs_one_element_front(self):
        p = default_params()
        p["mla"]["enabled"] = False
        for e in p["elements"][1:]:
            e["enabled"] = False
        jobs = lap_jobs(p, surface="front")
        self.assertEqual(jobs, [(0, "front")])

    def test_jobs_stack_both(self):
        p = default_params()
        p["mla"]["enabled"] = False
        p["elements"][0]["enabled"] = True
        p["elements"][1]["enabled"] = True
        for e in p["elements"][2:]:
            e["enabled"] = False
        jobs = lap_jobs(p, surface="both")
        self.assertEqual(
            jobs,
            [(0, "front"), (0, "rear"), (1, "front"), (1, "rear")],
        )

    def test_paths_single_keeps_name(self):
        dest = Path("tool.stl")
        paths = lap_output_paths(dest, [(0, "front")], n_elements=1)
        self.assertEqual([p.name for p in paths], ["tool.stl"])

    def test_paths_multi_suffix(self):
        dest = Path("tool.stl")
        jobs = [(0, "front"), (0, "rear"), (1, "front")]
        paths = lap_output_paths(dest, jobs, n_elements=2)
        self.assertEqual(
            [p.name for p in paths],
            ["tool_E1_front.stl", "tool_E1_rear.stl", "tool_E2_front.stl"],
        )


class TestStackSingletSpecs(unittest.TestCase):
    def test_ignores_mla_and_disabled(self):
        p = default_params()
        p["mla"]["enabled"] = True
        p["elements"][0].update(enabled=True, R1=22.0, thickness=4.0)
        p["elements"][1].update(enabled=True, R1=18.0, thickness=3.0)
        for e in p["elements"][2:]:
            e["enabled"] = False
        specs = stack_singlet_specs(p)
        self.assertEqual(len(specs), 2)
        self.assertAlmostEqual(specs[0].R1, 22.0)
        self.assertAlmostEqual(specs[1].thickness, 3.0)


class TestMeshPolishLap(unittest.TestCase):
    def test_pcx_front_is_cavity(self):
        mesh = mesh_polish_lap(_pcx(), "front", n_radial=16, n_theta=32)
        # Working face: center deeper than rim → female of a convex lens
        face = _working_face(mesh)
        rs = _radii_xy(face)
        axis = face[rs < 0.4]
        rim = face[(rs > 9.4) & (rs < 10.6)]
        self.assertGreater(len(axis), 0)
        self.assertGreater(len(rim), 0)
        self.assertLess(float(axis[:, 2].mean()), float(rim[:, 2].mean()) - 0.3)

    def test_pcv_front_is_boss(self):
        mesh = mesh_polish_lap(_pcv(), "front", n_radial=16, n_theta=32)
        face = _working_face(mesh)
        rs = _radii_xy(face)
        axis = face[rs < 0.4]
        rim = face[(rs > 9.4) & (rs < 10.6)]
        self.assertGreater(float(axis[:, 2].mean()), float(rim[:, 2].mean()) + 0.3)

    def test_biconvex_rear_is_cavity(self):
        mesh = mesh_polish_lap(_biconvex(), "rear", n_radial=16, n_theta=32)
        face = _working_face(mesh)
        rs = _radii_xy(face)
        axis = face[rs < 0.4]
        rim = face[(rs > 9.4) & (rs < 10.6)]
        self.assertLess(float(axis[:, 2].mean()), float(rim[:, 2].mean()) - 0.3)

    def test_envelope_is_cylinder_not_box(self):
        mesh = mesh_polish_lap(_pcx(), "front", n_radial=12, n_theta=48, wall_mm=6.0)
        rs = _radii_xy(mesh.vertices)
        r_max = float(rs.max())
        # Outer wall vertices sit on one radius (cylinder), not a square
        outer = mesh.vertices[rs > r_max - 0.05]
        self.assertGreater(len(outer), 20)
        self.assertLess(float(np.std(_radii_xy(outer))), 0.08)
        # A rectangular block of the same half-width would put corners at r*√2
        self.assertLess(r_max, 10.0 + 6.2 + 0.5)  # CA 10 + wall 6 + slip

    def test_mount_face_at_z0_with_tap_hole(self):
        mesh = mesh_polish_lap(_pcx(), "front", n_radial=12, n_theta=36)
        z = mesh.vertices[:, 2]
        self.assertAlmostEqual(float(z.min()), 0.0, places=5)
        rs = _radii_xy(mesh.vertices)
        r_hole = 0.5 * UNC_1_4_20_TAP_DRILL_MM
        mount = mesh.vertices[z < 0.05]
        hole_ring = mount[np.abs(rs[z < 0.05] - r_hole) < 0.08]
        self.assertGreater(len(hole_ring), 8)
        # Blind: optical surface stays above the hole
        opt_z = float(z.max())
        self.assertGreater(opt_z, LAP_HOLE_DEPTH_MM + LAP_HOLE_KEEP_MM - 0.2)

    def test_hole_does_not_break_optical_face(self):
        mesh = mesh_polish_lap(_pcx(), "front", n_radial=14, n_theta=32)
        rs = _radii_xy(mesh.vertices)
        # Points on the optical dish (inside CA, high Z) stay above the hole bottom
        dish = mesh.vertices[(rs < 10.2) & (mesh.vertices[:, 2] > LAP_HOLE_DEPTH_MM)]
        self.assertGreater(len(dish), 10)
        self.assertGreater(float(dish[:, 2].min()), LAP_HOLE_DEPTH_MM + 1.0)

    def test_positive_volume(self):
        mesh = mesh_polish_lap(_pcx(), "front", n_radial=12, n_theta=28)
        vol = mesh_signed_volume(mesh)
        self.assertGreater(vol, 100.0)

    def test_plano_is_flat_working_face(self):
        spec = LensSpec(R1=0.0, R2=0.0, thickness=4.0, aperture=8.0, z_front=0.0)
        mesh = mesh_polish_lap(spec, "front", n_radial=10, n_theta=24)
        face = _working_face(mesh)
        rs = _radii_xy(face)
        disk = face[rs < 8.1]
        self.assertGreater(len(disk), 5)
        self.assertLess(float(disk[:, 2].max() - disk[:, 2].min()), 0.05)


class TestExportPolishLap(unittest.TestCase):
    def test_writes_binary_stl_and_notes(self):
        p = default_params()
        p["mla"]["enabled"] = False
        for e in p["elements"][1:]:
            e["enabled"] = False
        p["cad_lap_surface"] = "front"
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "lap.stl"
            out = export_polish_lap(p, dest)
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 80 + 50 * 50)
            notes = dest.with_name("lap_tool.txt")
            self.assertTrue(notes.is_file())
            body = notes.read_text(encoding="utf-8")
            self.assertIn("1/4-20", body)
            self.assertIn("5.105", body)

    def test_both_faces_two_files(self):
        p = default_params()
        p["mla"]["enabled"] = False
        for e in p["elements"][1:]:
            e["enabled"] = False
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "job.stl"
            out = export_polish_lap(p, dest, surface="both")
            self.assertTrue(out.is_file())
            front = dest.with_name("job_front.stl")
            rear = dest.with_name("job_rear.stl")
            self.assertTrue(front.is_file())
            self.assertTrue(rear.is_file())


if __name__ == "__main__":
    unittest.main()
