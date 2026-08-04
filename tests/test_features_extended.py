"""
Extended feature tests: FOV metrics, optimizer, progressive display paths,
anamorphic swap, element stack, ray classification, collection bounds.

Run via: python validate_physics.py
or:       python -m unittest tests.test_features_extended -v
"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import (
    MAX_ELEMENTS,
    blank_element,
    build_source_array,
    build_surfaces,
    default_params,
    element_id_from_label,
    pad_elements,
    run_simulation,
    trace_ray,
)
from mla_geometry import (
    channel_aim_to_fov,
    thin_lens_focal_length_mm,
)
from optimizer import (
    OptimizeConfig,
    evaluate_fov_flux,
    inject_anamorphic_lenses,
    optimize_fov_flux,
)
from progressive import run_simulation_progressive
from rect_fov import (
    design_biconic_singlet_for_rect_fov,
    design_crossed_cylinders_for_rect_fov,
    swap_anamorphic_xy_element,
    swap_anamorphic_xy_params,
)


def _base_sim(**kw):
    p = default_params()
    p["source"]["mode"] = "single"
    p["mla"] = dict(p.get("mla") or {})
    p["mla"]["enabled"] = False
    p["use_warp"] = False
    p["total_rays"] = 1200
    p["display_rays"] = 80
    p["map_res"] = 48
    for k, v in kw.items():
        if k == "elements":
            p["elements"] = v
        elif k in p.get("source", {}) and k != "mode":
            p["source"][k] = v
        else:
            p[k] = v
    return p


# ═══════════════════════════════════════════════════════════════════════════
# Element stack / defaults
# ═══════════════════════════════════════════════════════════════════════════


class TestElementStack(unittest.TestCase):
    def test_default_has_five_slots(self):
        p = default_params()
        self.assertEqual(len(p["elements"]), MAX_ELEMENTS)
        self.assertEqual(MAX_ELEMENTS, 5)
        self.assertTrue(p["elements"][0]["enabled"])
        self.assertFalse(any(e["enabled"] for e in p["elements"][1:]))

    def test_pad_elements_grows_and_truncates(self):
        short = [blank_element(enabled=True)]
        padded = pad_elements(short, 5)
        self.assertEqual(len(padded), 5)
        self.assertTrue(padded[0]["enabled"])
        self.assertFalse(padded[4]["enabled"])
        long = pad_elements([blank_element() for _ in range(8)], 5)
        self.assertEqual(len(long), 5)

    def test_blank_element_disabled(self):
        e = blank_element()
        self.assertFalse(e["enabled"])
        self.assertEqual(e["surface_mode"], "rotational")


# ═══════════════════════════════════════════════════════════════════════════
# FOV metrics: profile fill, coverage, orientation, collection ≤ 1
# ═══════════════════════════════════════════════════════════════════════════


class TestFovMetrics(unittest.TestCase):
    def test_collection_never_exceeds_one(self):
        p = _base_sim(total_rays=2500, display_rays=50)
        r = run_simulation(p)
        self.assertGreaterEqual(r.stats["collection"], 0.0)
        self.assertLessEqual(r.stats["collection"], 1.0 + 1e-9)

    def test_fov_metrics_keys_present(self):
        p = _base_sim(total_rays=1500)
        r = run_simulation(p)
        fov = r.stats["fov"]
        for key in (
            "fraction",
            "uniformity",
            "coverage",
            "size_error",
            "aspect_error",
            "footprint_aspect",
            "target_aspect",
            "sig_x",
            "sig_y",
            "orientation_flipped",
            "profile_fill",
            "profile_fill_x",
            "profile_fill_y",
        ):
            self.assertIn(key, fov)
            self.assertTrue(math.isfinite(float(fov[key])), msg=key)

    def test_profile_fill_in_unit_interval(self):
        p = _base_sim(total_rays=2000)
        r = run_simulation(p)
        fov = r.stats["fov"]
        for k in ("profile_fill", "profile_fill_x", "profile_fill_y", "coverage"):
            self.assertGreaterEqual(fov[k], 0.0)
            self.assertLessEqual(fov[k], 1.0 + 1e-9)

    def test_target_aspect_matches_params(self):
        p = _base_sim(fov_width=50.0, fov_height=25.0, total_rays=800)
        r = run_simulation(p)
        self.assertAlmostEqual(r.stats["fov"]["target_aspect"], 2.0, places=5)

    def test_empty_map_metrics_safe(self):
        from engine import IrradianceMap

        m = IrradianceMap(20.0, 20.0, 16, 16)
        fov = m.fov_metrics(10.0, 8.0)
        self.assertEqual(fov["fraction"], 0.0)
        self.assertEqual(fov["profile_fill"], 0.0)
        # Empty map: size_error is a large under-fill penalty, not NaN
        self.assertGreaterEqual(fov["size_error"], 1.0)
        self.assertTrue(math.isfinite(fov["size_error"]))
        self.assertTrue(math.isfinite(fov["aspect_error"]))


# ═══════════════════════════════════════════════════════════════════════════
# Ray path element tagging
# ═══════════════════════════════════════════════════════════════════════════


class TestRayElementHits(unittest.TestCase):
    def test_element_id_from_label(self):
        self.assertEqual(element_id_from_label("E1S1"), "E1")
        self.assertEqual(element_id_from_label("E2S2"), "E2")
        self.assertEqual(element_id_from_label("MLA3S1"), "MLA3")
        self.assertEqual(element_id_from_label(""), "")

    def test_single_lens_paths_tag_E1(self):
        p = _base_sim(total_rays=400, display_rays=60)
        r = run_simulation(p)
        tagged = [path for path in r.paths if path.elements_hit]
        self.assertGreater(len(tagged), 5)
        for path in tagged:
            self.assertIn("E1", path.elements_hit)

    def test_two_element_partial_possible(self):
        p = _base_sim(total_rays=800, display_rays=120)
        p["elements"][1]["enabled"] = True
        p["elements"][1]["R1"] = 40.0
        p["elements"][1]["R2"] = -40.0
        p["elements"][1]["aperture"] = 6.0  # smaller → some rays miss E2
        p["elements"][1]["thickness"] = 3.0
        p["elements"][0]["air_after"] = 2.0
        r = run_simulation(p)
        n_partial = 0
        n_full = 0
        for path in r.paths:
            hits = [h for h in (path.elements_hit or []) if not str(h).startswith("MLA")]
            if len(hits) == 1:
                n_partial += 1
            elif len(hits) >= 2:
                n_full += 1
        # With a smaller second aperture we expect some partials and some full
        self.assertGreater(n_full + n_partial, 10)
        self.assertGreater(n_full, 0)


# ═══════════════════════════════════════════════════════════════════════════
# Partial-stack coloring vs side-view silhouette (meridional truth)
# ═══════════════════════════════════════════════════════════════════════════


class TestPartialColorMeridionalConsistency(unittest.TestCase):
    """
    Regression for the screenshot bug: amber 'partial' rays looked like they
    went through every lens in the side view because large-|X| rays were
    projected onto Y–Z and crossed the lens silhouette without hitting the
    circular clear aperture in 3D.
    """

    def _three_element_result(self):
        p = default_params()
        p["source"]["mode"] = "single"
        p["mla"] = dict(p.get("mla") or {})
        p["mla"]["enabled"] = False
        p["use_warp"] = False
        p["total_rays"] = 3500
        p["display_rays"] = 500
        p["lens_z_start"] = 3.0
        p["elements"][0].update(
            enabled=True,
            R1=25.0,
            R2=-40.0,
            thickness=8.0,
            aperture=10.0,
            air_after=2.0,
            material="ACRYLIC_PMMA",
        )
        p["elements"][1].update(
            enabled=True,
            R1=0.0,
            R2=0.0,
            thickness=6.0,
            aperture=10.0,
            air_after=2.0,
            material="ACRYLIC_PMMA",
        )
        p["elements"][2].update(
            enabled=True,
            R1=30.0,
            R2=0.0,
            thickness=8.0,
            aperture=10.0,
            air_after=1.0,
            material="ACRYLIC_PMMA",
        )
        for e in p["elements"][3:]:
            e["enabled"] = False
        return run_simulation(p)

    def test_on_axis_through_all_is_full(self):
        from engine import build_surfaces, trace_ray

        p = default_params()
        p["elements"][0].update(enabled=True, R1=25, R2=-40, thickness=8, aperture=10, air_after=2)
        p["elements"][1].update(enabled=True, R1=0, R2=0, thickness=6, aperture=10, air_after=2)
        p["elements"][2].update(enabled=True, R1=30, R2=0, thickness=8, aperture=10)
        for e in p["elements"][3:]:
            e["enabled"] = False
        surfs = build_surfaces(p["elements"], 3.0, None, None)
        ok, _, _, path = trace_ray(
            (0, 0, 0), (0, 0, 1), 1.0, 550.0, surfs, 80.0, store_path=True
        )
        self.assertTrue(ok)
        self.assertEqual(set(path.elements_hit), {"E1", "E2", "E3"})

    def test_false_silhouette_partials_are_off_axis_in_x(self):
        """
        Deterministic geometry of the screenshot bug: a ray can sit inside the
        clear-aperture radius at the *vertex plane* yet miss the curved surface
        (intersection lands outside CA). That path is correctly *not* tagged E3,
        but its Y–Z projection still crosses the drawn lens silhouette.
        """
        from engine import OpticalSurface, build_surfaces, trace_ray

        # Curved E3-like surface: R=30, ap=10 at z=21
        e3 = OpticalSurface(z_vertex=21.0, radius=30.0, aperture=10.0)
        o = (6.087, 6.510, 19.0)
        d = (0.3061, 0.3603, 0.8812)
        nrm = math.sqrt(sum(a * a for a in d))
        d = (d[0] / nrm, d[1] / nrm, d[2] / nrm)
        # At the vertex plane the ray is still inside the CA circle
        t_plane = (21.0 - o[2]) / d[2]
        xp = o[0] + d[0] * t_plane
        yp = o[1] + d[1] * t_plane
        self.assertLess(math.hypot(xp, yp), 10.0 + 0.05)
        # But the true surface intersection is outside the CA → miss
        self.assertIsNone(e3.intersect(o, d))

        # Full stack: on-axis hits all three; a steep off-axis ray misses E3
        p = default_params()
        p["elements"][0].update(
            enabled=True, R1=25, R2=-40, thickness=8, aperture=10, air_after=2,
            material="ACRYLIC_PMMA",
        )
        p["elements"][1].update(
            enabled=True, R1=0, R2=0, thickness=6, aperture=10, air_after=2,
            material="ACRYLIC_PMMA",
        )
        p["elements"][2].update(
            enabled=True, R1=30, R2=0, thickness=8, aperture=10,
            material="ACRYLIC_PMMA",
        )
        for e in p["elements"][3:]:
            e["enabled"] = False
        surfs = build_surfaces(p["elements"], 3.0, None, None)
        ok, _, _, path = trace_ray(
            (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0, 550.0, surfs, 80.0, store_path=True
        )
        self.assertTrue(ok)
        self.assertEqual(set(path.elements_hit), {"E1", "E2", "E3"})
        # Meridional rays with modest height still hit E3
        for y0 in (0.0, 1.0, 2.0, 3.0):
            ok, _, _, path = trace_ray(
                (0.0, y0, 0.0), (0.0, 0.02, 1.0), 1.0, 550.0, surfs, 80.0, store_path=True
            )
            self.assertTrue(ok, msg=f"meridional y0={y0}")
            self.assertIn("E3", path.elements_hit, msg=f"meridional y0={y0}")


    def test_meridional_slice_hides_false_silhouette_partials(self):
        """Side-view filter: paths with large |X| are excluded from the Y–Z plot."""
        from engine import path_in_meridional_slice

        r = self._three_element_result()
        e3 = next(s for s in r.surfaces if s.label == "E3S1")
        z3, ap3 = float(e3.z_vertex), float(e3.aperture)
        slice_half = max(0.8, 0.10 * ap3)
        kept_false = 0
        for path in r.paths:
            hit = list(path.elements_hit or [])
            if "E3" in hit or "E1" not in hit:
                continue
            if not path_in_meridional_slice(path, slice_half):
                continue
            # Path would be drawn in side view — must not silhouette through E3
            xy = None
            for i in range(len(path.history) - 1):
                z0, z1 = path.history[i][2], path.history[i + 1][2]
                if (z0 - z3) * (z1 - z3) <= 0 and abs(z1 - z0) > 1e-12:
                    t = (z3 - z0) / (z1 - z0)
                    xy = (
                        path.history[i][0] + t * (path.history[i + 1][0] - path.history[i][0]),
                        path.history[i][1] + t * (path.history[i + 1][1] - path.history[i][1]),
                    )
                    break
            if xy is None:
                continue
            # Inner 80% of the clear aperture: edge rays can expand past CA on a
            # curved surface between the vertex plane and the true intersect.
            if abs(xy[1]) <= ap3 * 0.80 and "E3" not in hit:
                kept_false += 1
        self.assertEqual(
            kept_false,
            0,
            "meridional slice still draws partial rays that silhouette through E3",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Anamorphic design + swap
# ═══════════════════════════════════════════════════════════════════════════


class TestAnamorphicDesign(unittest.TestCase):
    def test_crossed_landscape_stronger_y(self):
        """Landscape FOV → weaker X (longer f_x), stronger Y (shorter f_y)."""
        d = design_crossed_cylinders_for_rect_fov(
            fov_width=40, fov_height=20, target_z=80, aperture=12
        )
        meta = d["meta"]
        self.assertGreater(meta["f_x"], meta["f_y"])
        self.assertEqual(d["elements"][0]["surface_mode"], "cylinder_x")
        self.assertEqual(d["elements"][1]["surface_mode"], "cylinder_y")

    def test_biconic_landscape_Rx_gt_Ry(self):
        d = design_biconic_singlet_for_rect_fov(
            fov_width=40, fov_height=20, target_z=80, aperture=12
        )
        e = d["elements"][0]
        self.assertEqual(e["surface_mode"], "biconic")
        self.assertGreater(abs(float(e["R1"])), abs(float(e["R1y"])))

    def test_swap_cylinder_x_to_y(self):
        el = {
            "enabled": True,
            "surface_mode": "cylinder_x",
            "R1": 20.0,
            "R2": 0.0,
            "R1y": 0.0,
            "R2y": 0.0,
        }
        swap_anamorphic_xy_element(el)
        self.assertEqual(el["surface_mode"], "cylinder_y")
        self.assertAlmostEqual(float(el["R1y"]), 20.0)
        self.assertAlmostEqual(float(el["R1"]), 0.0)

    def test_swap_params_roundtrip(self):
        d = design_crossed_cylinders_for_rect_fov(
            fov_width=40, fov_height=32, target_z=80, aperture=12
        )
        p = default_params()
        p["elements"] = d["elements"]
        p2 = swap_anamorphic_xy_params(p)
        p3 = swap_anamorphic_xy_params(p2)
        self.assertEqual(p3["elements"][0]["surface_mode"], p["elements"][0]["surface_mode"])
        self.assertEqual(p3["elements"][1]["surface_mode"], p["elements"][1]["surface_mode"])

    def test_landscape_footprint_aspect_gt_one(self):
        d = design_crossed_cylinders_for_rect_fov(
            fov_width=50, fov_height=25, target_z=100, aperture=12
        )
        p = _base_sim(
            elements=d["elements"],
            lens_z_start=d.get("lens_z_start", 5.0),
            fov_width=50,
            fov_height=25,
            target_z=100,
            total_rays=3500,
            map_half_w=55,
            map_half_h=40,
        )
        r = run_simulation(p)
        self.assertGreater(r.stats["fov"]["footprint_aspect"], 1.0)
        self.assertEqual(r.stats["fov"]["orientation_flipped"], 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# MLA aim geometry
# ═══════════════════════════════════════════════════════════════════════════


class TestMlaAim(unittest.TestCase):
    def test_channel_aim_outer_die_positive_tilt_inward(self):
        # Die at +x should get negative tilt_y (aim toward −X / FOV centre)
        x0, y0, tx, ty = channel_aim_to_fov(
            2.0,
            0.0,
            0.0,
            lens_z=3.0,
            target_z=80.0,
            fov_cx=0.0,
            fov_cy=0.0,
            focal_length=5.0,
            aperture=0.6,
            pitch=1.6,
            aim_strength=1.0,
        )
        self.assertLess(ty, 0.0)  # aims toward −X
        # Optical centre further out than die for positive lens
        self.assertGreater(x0, 2.0 - 1e-9)

    def test_aim_strength_zero_no_offset(self):
        x0, y0, tx, ty = channel_aim_to_fov(
            3.0,
            1.0,
            0.0,
            lens_z=3.0,
            target_z=50.0,
            focal_length=4.0,
            aperture=0.5,
            pitch=1.6,
            aim_strength=0.0,
        )
        self.assertAlmostEqual(x0, 3.0, places=6)
        self.assertAlmostEqual(y0, 1.0, places=6)
        self.assertAlmostEqual(tx, 0.0, places=6)
        self.assertAlmostEqual(ty, 0.0, places=6)

    def test_thin_lens_biconvex_positive_f(self):
        f = thin_lens_focal_length_mm(30.0, -30.0, 1.5, 4.0)
        self.assertGreater(f, 0.0)
        self.assertLess(f, 100.0)


# ═══════════════════════════════════════════════════════════════════════════
# Optimizer inject + objective
# ═══════════════════════════════════════════════════════════════════════════


class TestOptimizer(unittest.TestCase):
    def test_inject_resets_to_collector_plus_anamorphics(self):
        p = default_params()
        p["elements"][1]["enabled"] = True
        p["elements"][2]["enabled"] = True
        p["elements"][3]["enabled"] = True
        out = inject_anamorphic_lenses(p, n_extra=2, mode="crossed")
        en = [e["enabled"] for e in out["elements"]]
        self.assertEqual(sum(1 for x in en if x), 3)
        modes = [e.get("surface_mode") for e in out["elements"] if e.get("enabled")]
        self.assertIn("cylinder_x", modes)
        self.assertIn("cylinder_y", modes)

    def test_inject_biconic_single_extra(self):
        p = default_params()
        out = inject_anamorphic_lenses(p, n_extra=1, mode="biconic")
        en = [i for i, e in enumerate(out["elements"]) if e.get("enabled")]
        self.assertEqual(len(en), 2)
        self.assertEqual(out["elements"][en[1]]["surface_mode"], "biconic")

    def test_evaluate_fov_flux_finite(self):
        p = _base_sim(total_rays=600)
        cfg = OptimizeConfig(rays_per_eval=400, map_res=32, force_cpu=True)
        score, ff, uni, col, ae, fa = evaluate_fov_flux(p, cfg)
        self.assertTrue(math.isfinite(score))
        self.assertGreaterEqual(ff, 0.0)
        self.assertLessEqual(ff, 1.0 + 1e-6)
        self.assertLessEqual(col, 1.0 + 1e-6)

    def test_optimize_small_budget_improves_or_stable(self):
        p = _base_sim(total_rays=400)
        cfg = OptimizeConfig(
            rays_per_eval=350,
            max_evals=12,
            population_size=3,
            polish=False,
            seed=42,
            two_phase=False,
            force_cpu=True,
            fill_weight=0.5,
            aspect_weight=0.0,
            uniformity_weight=0.1,
        )
        sc0, ff0, *_ = evaluate_fov_flux(p, cfg)
        result = optimize_fov_flux(p, cfg)
        self.assertGreaterEqual(result.n_evals, 1)
        self.assertTrue(math.isfinite(result.score))
        self.assertLessEqual(result.collection, 1.0 + 1e-6)
        # Best score should be at least as good as the seed (MC noise: allow small dip)
        self.assertGreaterEqual(result.score, sc0 * 0.85)

    def test_two_phase_keeps_at_most_three_enabled(self):
        p = default_params()
        p["source"]["mode"] = "single"
        p["mla"]["enabled"] = False
        p["use_warp"] = False
        cfg = OptimizeConfig(
            rays_per_eval=300,
            max_evals=14,
            population_size=3,
            polish=False,
            seed=3,
            two_phase=True,
            extra_anamorphic_lenses=2,
            anamorphic_mode="crossed",
            force_cpu=True,
            aspect_weight=1.0,
            fill_weight=1.0,
        )
        result = optimize_fov_flux(p, cfg)
        n_en = sum(1 for e in result.params["elements"] if e.get("enabled"))
        self.assertLessEqual(n_en, 3)
        self.assertIn(result.phase, ("1", "1+2", "2"))


# ═══════════════════════════════════════════════════════════════════════════
# Progressive display path count
# ═══════════════════════════════════════════════════════════════════════════


class TestProgressiveDisplayPaths(unittest.TestCase):
    def test_display_per_batch_respected(self):
        p = _base_sim(total_rays=500)
        results = []

        def on_batch(res, bi, n):
            results.append(res)

        run_simulation_progressive(
            p,
            batch_cb=on_batch,
            n_batches=2,
            rays_per_batch=400,
            display_per_batch=75,
            progress_cb=None,
            should_cancel=None,
        )
        self.assertGreaterEqual(len(results), 1)
        # Last batch should carry approximately display_per_batch paths
        self.assertEqual(len(results[-1].paths), 75)

    def test_more_display_means_more_paths(self):
        p = _base_sim()
        r_small = []
        r_large = []

        run_simulation_progressive(
            p,
            batch_cb=lambda res, bi, n: r_small.append(res),
            n_batches=1,
            rays_per_batch=500,
            display_per_batch=40,
        )
        run_simulation_progressive(
            p,
            batch_cb=lambda res, bi, n: r_large.append(res),
            n_batches=1,
            rays_per_batch=500,
            display_per_batch=120,
        )
        self.assertEqual(len(r_small[-1].paths), 40)
        self.assertEqual(len(r_large[-1].paths), 120)


# ═══════════════════════════════════════════════════════════════════════════
# Multi-element build_surfaces / labels
# ═══════════════════════════════════════════════════════════════════════════


class TestSurfaceLabels(unittest.TestCase):
    def test_stack_labels_E1_E2(self):
        p = default_params()
        p["elements"][1]["enabled"] = True
        surfs = build_surfaces(p["elements"], 3.0, None, None)
        labels = [s.label for s in surfs]
        self.assertTrue(any(l.startswith("E1") for l in labels))
        self.assertTrue(any(l.startswith("E2") for l in labels))

    def test_on_axis_ray_hits_both_surfaces(self):
        p = default_params()
        p["elements"][1]["enabled"] = False
        for e in p["elements"][2:]:
            e["enabled"] = False
        surfs = build_surfaces(p["elements"], 3.0, None, None)
        ok, pt, pout, path = trace_ray(
            (0, 0, 0),
            (0, 0, 1),
            1.0,
            550.0,
            surfs,
            80.0,
            apply_fresnel=True,
            store_path=True,
        )
        self.assertTrue(ok)
        self.assertIsNotNone(path)
        self.assertGreaterEqual(path.n_refractions, 2)
        self.assertIn("E1", path.elements_hit)


if __name__ == "__main__":
    unittest.main(verbosity=2)
