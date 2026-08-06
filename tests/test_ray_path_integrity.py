"""
Ray-path integrity: media, TIR policy, no spurious reflections, monotonic +Z.

Run via: python validate_physics.py  (discovers this module too)
or:       python tests/test_ray_path_integrity.py
"""
from __future__ import annotations

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import (
    OpticalSurface,
    build_source_array,
    build_surfaces,
    default_params,
    run_simulation,
    snell_refract,
    trace_ray,
)
from rect_fov import design_crossed_cylinders_for_rect_fov


def _paths_for(params, n_sample=400, **trace_kw):
    dies = build_source_array(params["source"])
    surfs = build_surfaces(
        params["elements"], params["lens_z_start"], params.get("mla"), dies
    )
    active = [d for d in dies if d.enabled]
    tot = sum(d.flux for d in active)
    paths = []
    for _ in range(n_sample):
        r = random.random() * tot
        die = active[0]
        for dd in active:
            r -= dd.flux
            if r <= 0:
                die = dd
                break
        o, d, pwr, wl = die.spawn_ray(1.0)
        kw = dict(
            custom_n=1.5,
            apply_fresnel=True,
            absorb_on_tir=True,
            store_path=True,
            max_reflections=0,
            kill_backward=True,
        )
        kw.update(trace_kw)
        ok, pt, pout, path = trace_ray(
            o, d, pwr, wl, surfs, params["target_z"], **kw
        )
        paths.append((ok, pt, pout, path, surfs))
    return paths


class TestTwoSidedMedia(unittest.TestCase):
    def test_front_surface_air_to_glass(self):
        surfs = build_surfaces(
            [{
                "enabled": True, "R1": 0, "R2": 0, "thickness": 5,
                "air_after": 1, "aperture": 20, "material": "N_BK7",
                "k1": 0, "k2": 0, "A4_1": 0, "A4_2": 0,
                "surface_mode": "rotational",
            }],
            z_start=2.0,
        )
        self.assertEqual(surfs[0].material_before, "AIR")
        self.assertNotEqual(surfs[0].material_after, "AIR")
        self.assertEqual(surfs[1].material_after, "AIR")
        self.assertNotEqual(surfs[1].material_before, "AIR")

    def test_window_power_fresnel_two_surfaces(self):
        surfs = build_surfaces(
            [{
                "enabled": True, "R1": 0, "R2": 0, "thickness": 4,
                "air_after": 1, "aperture": 20, "material": "N_BK7",
                "surface_mode": "rotational",
                "k1": 0, "k2": 0, "A4_1": 0, "A4_2": 0,
            }],
            z_start=1.0,
        )
        ok, pt, p, path = trace_ray(
            (0, 0, 0), (0, 0, 1), 1.0, 550, surfs, 40,
            apply_fresnel=True, absorb_on_tir=True, store_path=True,
        )
        self.assertTrue(ok)
        self.assertLess(p, 1.0)
        self.assertGreater(p, 0.85)
        # only refract events, no reflect
        self.assertNotIn("reflect", path.events)
        self.assertEqual(path.n_reflections, 0)


class TestNoSpuriousReflection(unittest.TestCase):
    def test_default_no_backward_segments(self):
        p = default_params()
        p["source"]["half_angle_deg"] = 80
        for ok, pt, pout, path, _ in _paths_for(p, 500):
            if path is None or len(path.history) < 2:
                continue
            for j in range(1, len(path.history)):
                dz = path.history[j][2] - path.history[j - 1][2]
                self.assertGreaterEqual(dz, -1e-3, msg=path.history)

    def test_strong_lens_absorb_tir_no_reflect_events(self):
        p = default_params()
        p["source"]["mode"] = "single"
        p["elements"][0].update(
            R1=4, R2=-4, thickness=10, aperture=8, material="N_SF11"
        )
        p["source"]["half_angle_deg"] = 89
        p["lens_z_start"] = 1.0
        n_reflect_events = 0
        n_tir_term = 0
        for ok, pt, pout, path, _ in _paths_for(p, 800, absorb_on_tir=True, max_reflections=0):
            if path is None:
                continue
            n_reflect_events += path.events.count("reflect")
            if path.terminated == "tir_absorb":
                n_tir_term += 1
            for j in range(1, len(path.history)):
                self.assertGreaterEqual(
                    path.history[j][2] - path.history[j - 1][2], -1e-3
                )
        self.assertEqual(n_reflect_events, 0)
        # Some high-angle rays should TIR-absorb on exit face
        self.assertGreater(n_tir_term, 0)

    def test_allow_tir_reflect_can_go_backward(self):
        """With TIR bounce enabled, some paths may reverse — documents the hazard."""
        p = default_params()
        p["source"]["mode"] = "single"
        p["elements"][0].update(R1=4, R2=-4, thickness=10, aperture=8, material="N_SF11")
        p["source"]["half_angle_deg"] = 89
        p["lens_z_start"] = 1.0
        saw_back = False
        saw_reflect = False
        for ok, pt, pout, path, _ in _paths_for(
            p, 1000, absorb_on_tir=False, max_reflections=5, kill_backward=False
        ):
            if path is None:
                continue
            if "reflect" in path.events:
                saw_reflect = True
            for j in range(1, len(path.history)):
                if path.history[j][2] < path.history[j - 1][2] - 1e-3:
                    saw_back = True
        self.assertTrue(saw_reflect or saw_back)

    def test_crossed_cylinders_no_reflect_default(self):
        d = design_crossed_cylinders_for_rect_fov(
            fov_width=48, fov_height=32, target_z=100, aperture=12
        )
        p = default_params()
        p["elements"] = d["elements"]
        p["lens_z_start"] = d["lens_z_start"]
        p["target_z"] = 100
        p["source"]["mode"] = "single"
        p["source"]["half_angle_deg"] = 70
        for ok, pt, pout, path, _ in _paths_for(p, 400):
            if path is None:
                continue
            self.assertEqual(path.n_reflections, 0)
            self.assertNotIn("reflect", path.events)

    def test_history_points_near_surfaces_or_ends(self):
        p = default_params()
        for ok, pt, pout, path, surfs in _paths_for(p, 200):
            if path is None or len(path.history) < 3:
                continue
            # intermediate points should lie on some surface
            for pt_h in path.history[1:-1]:
                near = False
                for s in surfs:
                    zs = s.surface_z(pt_h[0], pt_h[1])
                    if zs is not None and abs(pt_h[2] - zs) < 0.15:
                        near = True
                        break
                self.assertTrue(near, msg=f"orphan point {pt_h}")


class TestInteractionAccounting(unittest.TestCase):
    def test_power_never_increases(self):
        p = default_params()
        p["total_rays"] = 500
        # sample individual rays
        for ok, pt, pout, path, _ in _paths_for(p, 300):
            if path is None:
                continue
            self.assertLessEqual(path.power, 1.0 + 1e-9)
            self.assertGreaterEqual(path.power, 0.0)

    def test_sim_stats_tir_field(self):
        p = default_params()
        p["elements"][0].update(R1=8, R2=-10, thickness=6, aperture=12)
        p["source"]["half_angle_deg"] = 85
        p["total_rays"] = 2000
        p["absorb_on_tir"] = True
        r = run_simulation(p)
        self.assertIn("n_tir_absorb", r.stats)
        self.assertEqual(r.stats["n_reflections"], 0)
        self.assertGreaterEqual(r.stats["n_tir_absorb"], 0)

    def test_max_interactions_finite(self):
        p = default_params()
        for ok, pt, pout, path, _ in _paths_for(p, 100):
            if path is None:
                continue
            self.assertLessEqual(len(path.history), 40)

    def test_snell_normal_incidence_glass_exit(self):
        # glass → air normal: no TIR
        T, tir = snell_refract((0, 0, 1), (0, 0, 1), 1.5, 1.0)
        # Wait: N should face incident. Incident from glass going +Z toward exit.
        # Exit surface normal pointing toward glass (incident) is -Z for surface with +Z geometric normal
        T, tir = snell_refract((0, 0, 1), (0, 0, -1), 1.5, 1.0)
        self.assertFalse(tir)
        self.assertAlmostEqual(T[2], 1.0, places=6)


class TestSurfaceHitGeometry(unittest.TestCase):
    def test_intersect_only_forward_t(self):
        s = OpticalSurface(z_vertex=10, radius=0, aperture=50, material_after="N_BK7")
        # Ray going away
        hit = s.intersect((0, 0, 20), (0, 0, 1), t_min=1e-5)
        self.assertIsNone(hit)
        hit2 = s.intersect((0, 0, 0), (0, 0, 1), t_min=1e-5)
        self.assertIsNotNone(hit2)
        self.assertGreater(hit2[0], 0)

    def test_aperture_rejects_outside(self):
        s = OpticalSurface(z_vertex=5, radius=0, aperture=2.0)
        hit = s.intersect((0, 0, 0), (0, 0, 1))
        self.assertIsNotNone(hit)
        # angled ray that hits plane outside aperture
        # at z=5, x = 0 + t*dx, need t=5/dz
        hit2 = s.intersect((0, 0, 0), (0.9, 0, 0.1))  # hits z=5 at x=45
        # may fail intersection or aperture
        if hit2 is not None:
            self.fail("should not hit inside aperture")


def run_all():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(run_all())
