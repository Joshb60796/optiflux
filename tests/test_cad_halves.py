"""Print-half STL split at the flange midplane, and CNC profile .nc export."""
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
    CNC_PROFILE_EXT,
    LensSpec,
    export_lens,
    export_lens_cnc_profile,
    flange_cut_z,
    half_stl_paths,
    mesh_singlet,
    split_mesh_at_z,
)


def _radii(mesh):
    return [math.hypot(float(x), float(y)) for x, y, _z in mesh.vertices]


class TestFlangeCutZ(unittest.TestCase):
    def test_midplane_of_element(self):
        spec = LensSpec(R1=40, R2=-50, thickness=6.0, aperture=10, z_front=3.0)
        self.assertAlmostEqual(flange_cut_z(spec), 6.0)


class TestSplitMeshAtZ(unittest.TestCase):
    def test_halves_meet_at_flange_midplane(self):
        spec = LensSpec(R1=40, R2=-50, thickness=6.0, aperture=10, z_front=3.0)
        mesh = mesh_singlet(
            spec, n_radial=16, n_theta=32, flange_radial_mm=2.0, flange_thickness_mm=1.5
        )
        z_cut = flange_cut_z(spec)
        front, rear = split_mesh_at_z(mesh, z_cut)
        self.assertGreater(len(front.faces), 10)
        self.assertGreater(len(rear.faces), 10)
        fz = front.vertices[:, 2]
        rz = rear.vertices[:, 2]
        self.assertLessEqual(float(fz.max()), z_cut + 1e-4)
        self.assertGreaterEqual(float(rz.min()), z_cut - 1e-4)
        # Build face is the cut — both halves have vertices on the plane
        self.assertLess(abs(float(fz.max()) - z_cut), 0.05)
        self.assertLess(abs(float(rz.min()) - z_cut), 0.05)
        # Flange OD still present on each half (flat ring to print from)
        self.assertGreater(max(_radii(front)), 11.5)
        self.assertGreater(max(_radii(rear)), 11.5)

    def test_no_flange_still_caps(self):
        spec = LensSpec(R1=40, R2=-50, thickness=6.0, aperture=10, z_front=3.0)
        mesh = mesh_singlet(spec, n_radial=12, n_theta=24)
        front, rear = split_mesh_at_z(mesh, flange_cut_z(spec))
        self.assertGreater(len(front.faces), 8)
        self.assertGreater(len(rear.faces), 8)


class TestHalfExportFiles(unittest.TestCase):
    def test_naming_single_element(self):
        paths = half_stl_paths(Path("lens.stl"), 1)
        self.assertEqual([p.name for p in paths], ["lens_front.stl", "lens_rear.stl"])

    def test_naming_stack(self):
        paths = half_stl_paths(Path("group.stl"), 2)
        names = [p.name for p in paths]
        self.assertEqual(
            names,
            [
                "group_E1_front.stl",
                "group_E1_rear.stl",
                "group_E2_front.stl",
                "group_E2_rear.stl",
            ],
        )

    def test_export_writes_two_stls_and_not_the_whole_solid(self):
        p = default_params()
        p["mla"]["enabled"] = False
        for e in p["elements"][1:]:
            e["enabled"] = False
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "print.stl"
            out = export_lens(
                p,
                dest,
                fmt="stl",
                max_edge_mm=1.0,
                max_angle_deg=12.0,
                flange_radial_mm=2.0,
                flange_thickness_mm=1.5,
                split_halves=True,
            )
            front = dest.with_name("print_front.stl")
            rear = dest.with_name("print_rear.stl")
            self.assertTrue(front.is_file())
            self.assertTrue(rear.is_file())
            self.assertFalse(dest.is_file(), "whole-lens file must not be written for halves")
            self.assertTrue(out.is_file())
            self.assertIn(out.name, ("print_front.stl", "print_rear.stl"))


class TestCncProfileNc(unittest.TestCase):
    def test_default_extension_is_nc(self):
        self.assertEqual(CNC_PROFILE_EXT, ".nc")

    def test_export_writes_nc_not_gcode(self):
        p = default_params()
        p["mla"]["enabled"] = False
        for e in p["elements"][1:]:
            e["enabled"] = False
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "lens.gcode"
            out = export_lens_cnc_profile(p, dest)
            self.assertEqual(out.suffix.lower(), ".nc")
            self.assertTrue(out.is_file())
            body = out.read_text(encoding="ascii")
            self.assertIn("G21", body)
            self.assertIn("G90", body)
            self.assertIn("G1", body)
            self.assertIn("A", body)
            self.assertTrue(body.strip().endswith("M30") or "M30" in body)


if __name__ == "__main__":
    unittest.main()
