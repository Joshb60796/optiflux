"""
Ball lenses and collimation-to-FOV optimization.

A ball lens is a full sphere: R1 = +R, R2 = −R, thickness = 2R,
clear semi-aperture = R. Sequential tracing uses the shared sphere
(near hit = front, far hit from inside = rear).
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import (
    OpticalSurface,
    build_surfaces,
    collimation_metrics,
    default_params,
    lensmaker_f,
    run_simulation,
    trace_ray,
)
from export_cad import (
    LensSpec,
    build_lens_specs_from_params,
    mesh_signed_volume,
    mesh_singlet,
)
from lens_shapes import (
    apply_shape,
    ball_efl,
    ball_front_focal_length,
    ball_radius_from_current,
    constrain_ball_element,
    is_ball_element,
    is_ball_shape,
    shape_id_from_label,
    shape_label_from_id,
)
from optimizer import (
    OptimizeConfig,
    apply_vector,
    build_variable_list,
    config_from_panel,
    evaluate_fov_flux,
    normalize_objective,
)


def _ball_params(R: float = 5.0, *, n_mat: str = "N_BK7", half_angle: float = 20.0) -> dict:
    p = default_params()
    p["use_warp"] = False
    p["mla"]["enabled"] = False
    p["source"]["mode"] = "single"
    p["source"]["die_width"] = 0.05
    p["source"]["die_height"] = 0.05
    p["source"]["half_angle_deg"] = half_angle
    p["source"]["source_z"] = 0.0
    p["elements"][0] = apply_shape(
        "ball",
        R_mag=R,
        material=n_mat,
        air_after=2.0,
    )
    p["elements"][0]["enabled"] = True
    for e in p["elements"][1:]:
        e["enabled"] = False
    n = 1.5168  # N-BK7-ish; lensmaker uses catalog at wavelength
    ffl = ball_front_focal_length(R, n)
    p["lens_z_start"] = max(0.4, ffl)
    p["target_z"] = 80.0
    p["total_rays"] = 400
    p["display_rays"] = 40
    p["map_res"] = 32
    return p


class TestBallShapeCatalog(unittest.TestCase):
    def test_ball_is_in_dropdown(self):
        self.assertEqual(shape_id_from_label(shape_label_from_id("ball")), "ball")
        self.assertTrue(is_ball_shape("ball"))
        self.assertFalse(is_ball_shape("biconvex"))

    def test_apply_shape_locks_sphere_geometry(self):
        el = apply_shape("ball", R_mag=6.0, thickness=4.0, aperture=10.0, material="N_BK7")
        self.assertEqual(el["shape_id"], "ball")
        self.assertAlmostEqual(el["R1"], 6.0)
        self.assertAlmostEqual(el["R2"], -6.0)
        self.assertAlmostEqual(el["thickness"], 12.0)
        self.assertAlmostEqual(el["aperture"], 6.0)
        self.assertEqual(el["k1"], 0.0)
        self.assertEqual(el["k2"], 0.0)
        self.assertEqual(el["A4_1"], 0.0)
        self.assertEqual(el["A4_2"], 0.0)
        self.assertTrue(is_ball_element(el))

    def test_switching_to_ball_uses_clear_aperture_not_old_radii(self):
        """Default biconvex is R=40/−50, t=6, ap=10 — ball must be R=10, not 50."""
        r = ball_radius_from_current(
            aperture=10.0, r_mag=50.0, R1=40.0, R2=-50.0, thickness=6.0
        )
        self.assertAlmostEqual(r, 10.0)
        el = apply_shape("ball", R_mag=r, aperture=10.0, thickness=6.0)
        self.assertAlmostEqual(el["R1"], 10.0)
        self.assertAlmostEqual(el["R2"], -10.0)
        self.assertAlmostEqual(el["thickness"], 20.0)
        self.assertAlmostEqual(el["aperture"], 10.0)
        self.assertTrue(el.get("aperture_y") in (None, 10.0))

    def test_scaling_existing_ball_honours_R_mag(self):
        r = ball_radius_from_current(
            aperture=10.0, r_mag=15.0, R1=10.0, R2=-10.0, thickness=20.0
        )
        self.assertAlmostEqual(r, 15.0)

    def test_constrain_ball_snaps_free_params(self):
        el = {
            "shape_id": "ball",
            "R1": 4.0,
            "R2": -3.0,
            "thickness": 3.0,
            "aperture": 20.0,
            "k1": -0.5,
            "A4_1": 1e-4,
            "surface_mode": "biconic",
            "R1y": 8.0,
        }
        out = constrain_ball_element(el)
        self.assertAlmostEqual(out["R1"], 4.0)
        self.assertAlmostEqual(out["R2"], -4.0)
        self.assertAlmostEqual(out["thickness"], 8.0)
        self.assertAlmostEqual(out["aperture"], 4.0)
        self.assertEqual(out["surface_mode"], "rotational")
        self.assertIsNone(out.get("R1y"))
        self.assertEqual(out["k1"], 0.0)

    def test_ball_efl_matches_lensmaker(self):
        R, n, t = 5.0, 1.5, 10.0
        self.assertAlmostEqual(ball_efl(R, n), n * R / (2.0 * (n - 1.0)), places=9)
        self.assertAlmostEqual(ball_efl(R, n), lensmaker_f(R, -R, n, t), places=9)
        self.assertAlmostEqual(ball_front_focal_length(R, n), ball_efl(R, n) - R, places=9)


class TestBallSurfacesAndTrace(unittest.TestCase):
    def test_build_surfaces_keeps_full_hemisphere_aperture(self):
        el = apply_shape("ball", R_mag=5.0, material="N_BK7")
        surfs = build_surfaces([el], z_start=3.0)
        self.assertEqual(len(surfs), 2)
        self.assertEqual(surfs[0].geom, "sphere")
        self.assertEqual(surfs[1].geom, "sphere")
        self.assertGreater(surfs[0].aperture, 4.5)
        self.assertAlmostEqual(surfs[0].aperture, surfs[1].aperture, places=6)
        self.assertAlmostEqual(surfs[0].z_vertex + 2.0 * 5.0, surfs[1].z_vertex, places=6)

    def test_on_axis_ray_hits_front_then_rear(self):
        el = apply_shape("ball", R_mag=5.0, material="N_BK7")
        surfs = build_surfaces([el], z_start=3.0)
        ok, pt, pwr, path = trace_ray(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0,
            550.0,
            surfs,
            40.0,
            store_path=True,
        )
        self.assertTrue(ok)
        self.assertIsNotNone(path)
        self.assertGreaterEqual(path.n_refractions, 2)
        self.assertEqual(path.terminated, "target")
        zs = [h[2] for h in path.history]
        self.assertGreaterEqual(len(zs), 3)
        # Vertex, then far vertex, then target
        self.assertAlmostEqual(zs[1], 3.0, delta=0.05)
        self.assertAlmostEqual(zs[2], 13.0, delta=0.05)

    def test_ray_that_misses_the_ball_does_not_enter_from_the_back(self):
        el = apply_shape("ball", R_mag=5.0, material="N_BK7")
        surfs = build_surfaces([el], z_start=3.0)
        # Parallel to +Z, 8 mm off axis — outside a 5 mm radius sphere
        ok, pt, pwr, path = trace_ray(
            (8.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0,
            550.0,
            surfs,
            40.0,
            store_path=True,
        )
        self.assertTrue(ok)
        self.assertEqual(path.n_refractions, 0)
        self.assertEqual(path.terminated, "target")

    def test_paraxial_focal_ray_exits_nearly_collimated(self):
        from materials_catalog import refractive_index

        R = 5.0
        el = apply_shape("ball", R_mag=R, material="N_BK7")
        n = refractive_index("N_BK7", 550.0)
        ffl = ball_front_focal_length(R, n)
        surfs = build_surfaces([el], z_start=ffl)
        # Ray from the front focal point at 5° to +Z — a collimator maps this
        # to a nearly axial output (paraxial; some spherical aberration remains).
        th = math.radians(5.0)
        d = (0.0, math.sin(th), math.cos(th))
        ok, _pt, _pwr, path = trace_ray(
            (0.0, 0.0, 0.0),
            d,
            1.0,
            550.0,
            surfs,
            80.0,
            store_path=True,
            apply_fresnel=False,
        )
        self.assertTrue(ok)
        self.assertGreaterEqual(path.n_refractions, 2)
        fd = path.final_dir
        self.assertIsNotNone(fd)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, fd[2]))))
        self.assertLess(ang, 2.5)

    def test_sphere_intersect_matches_analytic_vertex(self):
        s = OpticalSurface(
            z_vertex=4.0,
            radius=8.0,
            aperture=8.0,
            geom="sphere",
            sphere_select="near",
        )
        hit = s.intersect((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        self.assertIsNotNone(hit)
        t, p, n = hit
        self.assertAlmostEqual(p[2], 4.0, places=6)
        self.assertAlmostEqual(abs(n[2]), 1.0, places=5)


class TestBallCad(unittest.TestCase):
    def test_mesh_vertices_lie_on_sphere(self):
        R = 6.0
        spec = LensSpec(
            R1=R, R2=-R, thickness=2.0 * R, aperture=R, z_front=2.0, shape_id="ball"
        )
        mesh = mesh_singlet(spec, n_radial=16, n_theta=32)
        zc = 2.0 + R
        radii = []
        for v in mesh.vertices:
            radii.append(math.sqrt(v[0] ** 2 + v[1] ** 2 + (v[2] - zc) ** 2))
        self.assertGreater(len(radii), 20)
        self.assertAlmostEqual(min(radii), R, delta=0.08)
        self.assertAlmostEqual(max(radii), R, delta=0.08)

    def test_mesh_volume_is_sphere(self):
        R = 4.0
        spec = LensSpec(
            R1=R, R2=-R, thickness=2.0 * R, aperture=R, z_front=0.0, shape_id="ball"
        )
        mesh = mesh_singlet(spec, n_radial=24, n_theta=48)
        vol = abs(mesh_signed_volume(mesh))
        expect = 4.0 / 3.0 * math.pi * R ** 3
        self.assertGreater(vol, 0.0)
        self.assertAlmostEqual(vol, expect, delta=0.08 * expect)

    def test_export_stack_includes_ball_spec(self):
        p = _ball_params(5.0)
        specs, mode = build_lens_specs_from_params(p)
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].shape_id, "ball")
        self.assertAlmostEqual(specs[0].thickness, 10.0)

    def test_mesh_singlet_does_not_stretch_ball_with_rim_fudge(self):
        R = 5.0
        spec = LensSpec(
            R1=R, R2=-R, thickness=2.0 * R, aperture=R, z_front=1.0, shape_id="ball"
        )
        mesh = mesh_singlet(spec, n_radial=12, n_theta=24)
        z = mesh.vertices[:, 2]
        self.assertAlmostEqual(float(z.min()), 1.0, delta=0.05)
        self.assertAlmostEqual(float(z.max()), 1.0 + 2.0 * R, delta=0.05)


class TestCollimationMetrics(unittest.TestCase):
    def test_parallel_bundle_has_near_zero_rms(self):
        samples = [(0.0, 0.0, 1.0, 1.0, 0.0, 0.0) for _ in range(20)]
        m = collimation_metrics(samples, fov_width=40.0, fov_height=32.0)
        self.assertLess(m["rms_deg"], 1e-6)
        self.assertLess(m["fov_rms_deg"], 1e-6)

    def test_diverging_bundle_has_larger_rms(self):
        samples = []
        for i in range(40):
            th = math.radians(8.0 if i % 2 else -8.0)
            samples.append((0.0, math.sin(th), math.cos(th), 1.0, 0.0, 0.0))
        m = collimation_metrics(samples, fov_width=40.0, fov_height=32.0)
        self.assertGreater(m["rms_deg"], 6.0)
        self.assertLess(m["rms_deg"], 10.0)

    def test_simulation_reports_collimation_stats(self):
        p = _ball_params(5.0)
        p["total_rays"] = 250
        r = run_simulation(p)
        col = r.stats.get("collimation") or {}
        self.assertIn("rms_deg", col)
        self.assertGreaterEqual(col.get("n_samples", 0), 1)

    def test_ball_collimator_is_tighter_than_bare_source(self):
        bare = _ball_params(5.0)
        bare["elements"][0]["enabled"] = False
        bare["source"]["half_angle_deg"] = 15.0
        bare["total_rays"] = 600
        with_ball = _ball_params(5.0)
        with_ball["source"]["half_angle_deg"] = 15.0
        with_ball["total_rays"] = 600
        rb = run_simulation(bare)
        rw = run_simulation(with_ball)
        rms_bare = float((rb.stats.get("collimation") or {}).get("rms_deg", 90.0))
        rms_ball = float((rw.stats.get("collimation") or {}).get("rms_deg", 90.0))
        self.assertGreater(rms_bare, 8.0)
        self.assertLess(rms_ball, rms_bare * 0.65)


class TestCollimationOptimizer(unittest.TestCase):
    def test_normalize_collimate_aliases(self):
        self.assertEqual(normalize_objective("collimate"), "collimate")
        self.assertEqual(normalize_objective("Collimated beam"), "collimate")
        self.assertEqual(normalize_objective("collimation"), "collimate")

    def test_config_from_panel_collimate_does_not_add_lenses(self):
        cfg = config_from_panel(
            objective="collimate",
            allow_extra_lenses=True,
            extra_lenses=4,
        )
        self.assertEqual(cfg.objective, "collimate")
        self.assertFalse(cfg.two_phase)
        self.assertEqual(cfg.extra_anamorphic_lenses, 0)

    def test_collimate_score_prefers_tighter_bundle(self):
        p = default_params()
        cfg = OptimizeConfig(objective="collimate", force_cpu=True, two_phase=False)

        def _sim(rms, power_in=0.35, plane_power=0.4, coverage=0.6):
            fov = {
                "power_in": power_in,
                "fraction": power_in,
                "uniformity": 0.7,
                "coverage": coverage,
                "profile_fill": coverage,
                "size_error": 0.1,
                "aspect_error": 0.05,
                "footprint_aspect": 1.1,
                "orientation_flipped": 0.0,
                "edge_sharpness": 0.4,
            }
            stats = {
                "source_power": 1.0,
                "map_power": plane_power * 0.9,
                "plane_power": plane_power,
                "collection": power_in,
                "fov": fov,
                "collimation": {"rms_deg": rms, "fov_rms_deg": rms, "n_samples": 100},
            }
            return SimpleNamespace(stats=stats, map=None, paths=[], dies=[], surfaces=[])

        with patch("optimizer.run_simulation", return_value=_sim(12.0)):
            sc_wide, *_ = evaluate_fov_flux(p, cfg)
        with patch("optimizer.run_simulation", return_value=_sim(1.5)):
            sc_tight, *_ = evaluate_fov_flux(p, cfg)
        self.assertGreater(sc_tight, sc_wide)

    def test_collimate_does_not_reward_dark_perfect_pencil(self):
        p = default_params()
        cfg = OptimizeConfig(objective="collimate", force_cpu=True)

        def _sim(rms, power_in):
            fov = {
                "power_in": power_in,
                "fraction": power_in,
                "uniformity": 0.9,
                "coverage": 0.9 if power_in > 0.2 else 0.05,
                "profile_fill": 0.9 if power_in > 0.2 else 0.05,
                "size_error": 0.05 if power_in > 0.2 else 0.8,
                "aspect_error": 0.0,
                "footprint_aspect": 1.0,
                "orientation_flipped": 0.0,
                "edge_sharpness": 0.5,
            }
            stats = {
                "source_power": 1.0,
                "map_power": power_in,
                "plane_power": power_in,
                "collection": power_in,
                "fov": fov,
                "collimation": {"rms_deg": rms, "fov_rms_deg": rms, "n_samples": 20},
            }
            return SimpleNamespace(stats=stats, map=None, paths=[], dies=[], surfaces=[])

        with patch("optimizer.run_simulation", return_value=_sim(0.2, 0.01)):
            sc_dark, *_ = evaluate_fov_flux(p, cfg)
        with patch("optimizer.run_simulation", return_value=_sim(2.5, 0.4)):
            sc_lit, *_ = evaluate_fov_flux(p, cfg)
        self.assertGreater(sc_lit, sc_dark)

    def test_ball_optimizer_vars_keep_sphere_lock(self):
        p = _ball_params(5.0)
        cfg = OptimizeConfig(
            objective="collimate",
            optimize_radii=True,
            optimize_thickness=True,
            optimize_aperture=True,
            optimize_air_gaps=True,
            optimize_lens_z=True,
        )
        vars_ = build_variable_list(p, cfg)
        names = [v.name for v in vars_]
        self.assertTrue(any(n.endswith(".R1") or n.endswith(".ball_R") for n in names))
        self.assertFalse(any(n.endswith(".R2") for n in names))
        self.assertFalse(any(n.endswith(".thickness") for n in names))
        self.assertFalse(any(n.endswith(".aperture") for n in names))

    def test_apply_vector_preserves_ball_shape(self):
        p = _ball_params(5.0)
        cfg = OptimizeConfig(objective="collimate", optimize_thickness=True, optimize_aperture=True)
        vars_ = build_variable_list(p, cfg)
        x = []
        for v in vars_:
            if v.name.endswith(".R1") or v.name.endswith(".ball_R"):
                x.append(7.0)
            else:
                x.append(0.5 * (v.lo + v.hi))
        out = apply_vector(p, vars_, x)
        e = out["elements"][0]
        self.assertEqual(e["shape_id"], "ball")
        self.assertTrue(is_ball_element(e))
        self.assertAlmostEqual(abs(e["R1"]), abs(e["R2"]))
        self.assertAlmostEqual(e["thickness"], 2.0 * abs(e["R1"]), places=5)

    def test_geometry_penalty_does_not_punish_ball(self):
        from optimizer import _geometry_penalty

        p = _ball_params(5.0)
        pen = _geometry_penalty(p, 5.0)
        self.assertLess(pen, 0.05)


if __name__ == "__main__":
    unittest.main()
