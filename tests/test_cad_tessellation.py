"""
STL tessellation tolerances: max vertex (edge) length and max facet angle.
"""
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
    export_lens,
    mesh_singlet,
    tessellation_for_specs,
    tessellation_from_tolerance,
)


class TestTessellationFromTolerance(unittest.TestCase):
    def test_tighter_edge_increases_segments(self):
        coarse = tessellation_from_tolerance(10.0, [25.0], max_edge_mm=1.0, max_angle_deg=15.0)
        fine = tessellation_from_tolerance(10.0, [25.0], max_edge_mm=0.2, max_angle_deg=15.0)
        self.assertGreater(fine[0], coarse[0])
        self.assertGreater(fine[1], coarse[1])

    def test_tighter_angle_increases_counts_on_curve(self):
        loose = tessellation_from_tolerance(12.0, [15.0], max_edge_mm=5.0, max_angle_deg=10.0)
        tight = tessellation_from_tolerance(12.0, [15.0], max_edge_mm=5.0, max_angle_deg=1.0)
        self.assertGreater(tight[0], loose[0])
        self.assertGreaterEqual(tight[1], loose[1])

    def test_plano_uses_edge_length_only(self):
        n_r, n_t = tessellation_from_tolerance(
            10.0, [0.0, None], max_edge_mm=0.5, max_angle_deg=0.5
        )
        # 10 / 0.5 = 20 rings; 2π·10 / 0.5 ≈ 126 segments
        self.assertGreaterEqual(n_r, 20)
        self.assertGreaterEqual(n_t, 120)
        # Angle must not explode a plano into the hoop 2π/δ cap (~720 at 0.5°)
        self.assertLess(n_t, 400)

    def test_clamps_to_max(self):
        n_r, n_t = tessellation_from_tolerance(
            80.0,
            [20.0],
            max_edge_mm=0.01,
            max_angle_deg=0.1,
            n_radial_max=64,
            n_theta_max=128,
        )
        self.assertEqual(n_r, 64)
        self.assertEqual(n_t, 128)

    def test_invalid_inputs_still_return_mins(self):
        n_r, n_t = tessellation_from_tolerance(
            0.0, [], max_edge_mm=0.0, max_angle_deg=0.0
        )
        self.assertGreaterEqual(n_r, 4)
        self.assertGreaterEqual(n_t, 12)


class TestTessellationForSpecs(unittest.TestCase):
    def test_uses_largest_aperture(self):
        small = [LensSpec(R1=30, R2=-30, thickness=3, aperture=5, z_front=0)]
        large = [
            LensSpec(R1=30, R2=-30, thickness=3, aperture=5, z_front=0),
            LensSpec(R1=40, R2=-40, thickness=3, aperture=20, z_front=4),
        ]
        ns = tessellation_for_specs(small, max_edge_mm=0.4, max_angle_deg=8.0)
        nl = tessellation_for_specs(large, max_edge_mm=0.4, max_angle_deg=8.0)
        self.assertGreater(nl[1], ns[1])


class TestMeshRespectsEdgeLength(unittest.TestCase):
    def test_rim_chord_at_most_max_edge(self):
        spec = LensSpec(R1=25, R2=-30, thickness=5, aperture=10, z_front=0)
        max_edge = 0.35
        n_r, n_t = tessellation_from_tolerance(
            spec.aperture, [spec.R1, spec.R2], max_edge_mm=max_edge, max_angle_deg=12.0
        )
        mesh = mesh_singlet(spec, n_radial=n_r, n_theta=n_t)
        # Outer-ring consecutive vertices (front, skip center)
        # Polar layout: index 1 + (n_r-1)*n_t .. 
        start = 1 + (n_r - 1) * n_t
        chords = []
        for i in range(n_t):
            a = mesh.vertices[start + i]
            b = mesh.vertices[start + (i + 1) % n_t]
            chords.append(float(math.dist(a, b)))
        self.assertGreater(len(chords), 10)
        self.assertLessEqual(max(chords), max_edge * 1.08)

    def test_export_lens_accepts_tolerances(self):
        p = default_params()
        p["mla"]["enabled"] = False
        for e in p["elements"][1:]:
            e["enabled"] = False
        with tempfile.TemporaryDirectory() as td:
            coarse = Path(td) / "coarse.stl"
            fine = Path(td) / "fine.stl"
            export_lens(
                p, coarse, fmt="stl", max_edge_mm=1.2, max_angle_deg=12.0
            )
            export_lens(
                p, fine, fmt="stl", max_edge_mm=0.25, max_angle_deg=2.0
            )
            self.assertGreater(fine.stat().st_size, coarse.stat().st_size)


if __name__ == "__main__":
    unittest.main()
