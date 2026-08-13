"""
Optimizer panel goals, extra-lens policy, LED size limits, and the
target-plane ray-budget bug (GUI used a fixed 5×5000 regardless of slider).
"""
from __future__ import annotations

import math
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import (
    MAX_ELEMENTS,
    SOURCE_DIE_MAX_MM,
    IrradianceMap,
    default_params,
    pad_elements,
)
from optimizer import (
    OptimizeConfig,
    OptimizeResult,
    config_from_panel,
    evaluate_fov_flux,
    inject_anamorphic_lenses,
    optimize_fov_flux,
    select_two_phase_winner,
)
from progressive import map_ray_budget, run_simulation_progressive


def _fake_sim(
    *,
    source_power: float = 1.0,
    power_in: float = 0.3,
    plane_power: float = 0.5,
    collection: float = 0.5,
    uniformity: float = 0.7,
    coverage: float = 0.6,
    profile_fill: float = 0.6,
    size_error: float = 0.1,
    aspect_error: float = 0.05,
    footprint_aspect: float = 1.25,
    orientation_flipped: float = 0.0,
    edge_sharpness: float = 0.4,
):
    fov = {
        "power_in": power_in,
        "fraction": power_in / max(source_power, 1e-30),
        "uniformity": uniformity,
        "coverage": coverage,
        "profile_fill": profile_fill,
        "size_error": size_error,
        "aspect_error": aspect_error,
        "footprint_aspect": footprint_aspect,
        "orientation_flipped": orientation_flipped,
        "edge_sharpness": edge_sharpness,
        "sig_x": 10.0,
        "sig_y": 8.0,
    }
    stats = {
        "source_power": source_power,
        "map_power": plane_power * 0.9,
        "plane_power": plane_power,
        "collection": collection,
        "fov": fov,
    }
    return SimpleNamespace(stats=stats, map=None, paths=[], dies=[], surfaces=[])


class TestStackCapacity(unittest.TestCase):
    def test_max_elements_is_eight(self):
        self.assertEqual(MAX_ELEMENTS, 8)

    def test_default_params_has_eight_slots(self):
        p = default_params()
        self.assertEqual(len(p["elements"]), 8)
        self.assertTrue(p["elements"][0]["enabled"])
        self.assertFalse(any(e["enabled"] for e in p["elements"][1:]))

    def test_pad_to_eight(self):
        out = pad_elements([], 8)
        self.assertEqual(len(out), 8)


class TestLedSizeLimits(unittest.TestCase):
    def test_die_slider_range_fits_35mm_source(self):
        self.assertGreaterEqual(SOURCE_DIE_MAX_MM, 35.0)

    def test_simulation_accepts_35mm_die(self):
        from engine import run_simulation

        p = default_params()
        p["source"]["die_width"] = 35.0
        p["source"]["die_height"] = 35.0
        p["source"]["mode"] = "single"
        p["use_warp"] = False
        p["total_rays"] = 200
        p["display_rays"] = 10
        r = run_simulation(p)
        self.assertGreaterEqual(r.stats["launched"], 1)


class TestMapRayBudget(unittest.TestCase):
    def test_splits_requested_total(self):
        batches, rpb = map_ray_budget(10000, n_batches=5)
        self.assertEqual(batches * rpb, 10000)

    def test_more_requested_rays_means_more_traced(self):
        _, lo = map_ray_budget(2000, n_batches=5)
        _, hi = map_ray_budget(50000, n_batches=5)
        self.assertGreater(hi, lo)

    def test_progressive_launches_near_requested_total(self):
        p = default_params()
        p["source"]["mode"] = "single"
        p["mla"]["enabled"] = False
        p["use_warp"] = False
        p["total_rays"] = 800
        n_batches, rpb = map_ray_budget(800, n_batches=4)
        last = []

        def on_batch(res, bi, n):
            last.append(res)

        run_simulation_progressive(
            p,
            batch_cb=on_batch,
            n_batches=n_batches,
            rays_per_batch=rpb,
            display_per_batch=0,
        )
        self.assertGreaterEqual(len(last), 1)
        self.assertAlmostEqual(last[-1].stats["launched"], n_batches * rpb, delta=1)


class TestEdgeSharpnessMetric(unittest.TestCase):
    def _fill_disk(self, hard: bool) -> IrradianceMap:
        m = IrradianceMap(20.0, 20.0, 64, 64)
        dx = 40.0 / 64
        dy = 40.0 / 64
        for iy in range(64):
            for ix in range(64):
                x = -20.0 + (ix + 0.5) * dx
                y = 20.0 - (iy + 0.5) * dy
                r = math.hypot(x, y)
                if hard:
                    p = 1.0 if r <= 8.0 else 0.0
                else:
                    p = math.exp(-0.5 * (r / 8.0) ** 2)
                if p > 1e-12:
                    m.deposit(x, y, p)
        return m

    def test_hard_disk_sharper_than_gaussian(self):
        sharp = self._fill_disk(True).edge_sharpness()
        blur = self._fill_disk(False).edge_sharpness()
        self.assertGreater(sharp, 0.0)
        self.assertGreater(blur, 0.0)
        self.assertGreater(sharp, blur)

    def test_fov_metrics_include_edge_sharpness(self):
        m = self._fill_disk(True)
        fov = m.fov_metrics(40.0, 40.0)
        self.assertIn("edge_sharpness", fov)
        self.assertGreater(fov["edge_sharpness"], 0.0)
        self.assertTrue(math.isfinite(fov["edge_sharpness"]))

    def test_empty_map_sharpness_zero(self):
        m = IrradianceMap(10.0, 10.0, 16, 16)
        self.assertEqual(m.edge_sharpness(), 0.0)
        self.assertEqual(m.fov_metrics(8.0, 8.0)["edge_sharpness"], 0.0)


class TestConfigFromPanel(unittest.TestCase):
    def test_rect_fill_with_allow_extra(self):
        cfg = config_from_panel(
            objective="rect_fill",
            allow_extra_lenses=True,
            extra_lenses=3,
            anamorphic_mode="crossed",
        )
        self.assertEqual(cfg.objective, "rect_fill")
        self.assertTrue(cfg.two_phase)
        self.assertTrue(cfg.allow_extra_lenses)
        self.assertEqual(cfg.extra_anamorphic_lenses, 3)
        self.assertGreater(cfg.aspect_weight, 0.0)

    def test_rect_fill_without_allow_does_not_add(self):
        cfg = config_from_panel(
            objective="rect_fill",
            allow_extra_lenses=False,
            extra_lenses=4,
        )
        self.assertFalse(cfg.two_phase)
        self.assertFalse(cfg.allow_extra_lenses)
        self.assertEqual(cfg.extra_anamorphic_lenses, 0)

    def test_focus_never_two_phase(self):
        cfg = config_from_panel(
            objective="focus",
            allow_extra_lenses=True,
            extra_lenses=4,
        )
        self.assertEqual(cfg.objective, "focus")
        self.assertFalse(cfg.two_phase)
        self.assertEqual(cfg.extra_anamorphic_lenses, 0)

    def test_evenness_never_two_phase(self):
        cfg = config_from_panel(
            objective="evenness",
            allow_extra_lenses=True,
            extra_lenses=2,
        )
        self.assertEqual(cfg.objective, "evenness")
        self.assertFalse(cfg.two_phase)
        self.assertEqual(cfg.extra_anamorphic_lenses, 0)

    def test_extra_capped_at_seven_with_eight_slots(self):
        cfg = config_from_panel(
            objective="rect_fill",
            allow_extra_lenses=True,
            extra_lenses=99,
        )
        self.assertEqual(cfg.extra_anamorphic_lenses, MAX_ELEMENTS - 1)
        self.assertLessEqual(cfg.extra_anamorphic_lenses, 8)

    def test_extra_eight_requested_is_accepted_up_to_cap(self):
        cfg = config_from_panel(
            objective="rect_fill",
            allow_extra_lenses=True,
            extra_lenses=8,
        )
        self.assertGreaterEqual(cfg.extra_anamorphic_lenses, 7)
        self.assertLessEqual(cfg.extra_anamorphic_lenses, 8)


class TestFocusAndEvennessScoring(unittest.TestCase):
    def test_focus_prefers_sharper_border(self):
        p = default_params()
        cfg = OptimizeConfig(objective="focus", force_cpu=True, two_phase=False)
        soft = _fake_sim(edge_sharpness=0.15, power_in=0.4, plane_power=0.45)
        crisp = _fake_sim(edge_sharpness=0.8, power_in=0.4, plane_power=0.45)
        with patch("optimizer.run_simulation", return_value=soft):
            sc_soft, *_ = evaluate_fov_flux(p, cfg)
        with patch("optimizer.run_simulation", return_value=crisp):
            sc_crisp, *_ = evaluate_fov_flux(p, cfg)
        self.assertGreater(sc_crisp, sc_soft)

    def test_evenness_prefers_uniform_fill(self):
        p = default_params()
        cfg = OptimizeConfig(objective="evenness", force_cpu=True, two_phase=False)
        peaked = _fake_sim(uniformity=0.2, coverage=0.4, profile_fill=0.4)
        even = _fake_sim(uniformity=0.92, coverage=0.9, profile_fill=0.9)
        with patch("optimizer.run_simulation", return_value=peaked):
            sc_peak, *_ = evaluate_fov_flux(p, cfg)
        with patch("optimizer.run_simulation", return_value=even):
            sc_even, *_ = evaluate_fov_flux(p, cfg)
        self.assertGreater(sc_even, sc_peak)

    def test_rect_fill_still_rewards_coverage(self):
        p = default_params()
        cfg = OptimizeConfig(objective="rect_fill", force_cpu=True, coverage_mix=0.95)
        hot = _fake_sim(coverage=0.15, profile_fill=0.2, uniformity=0.2, power_in=0.4)
        filled = _fake_sim(coverage=0.9, profile_fill=0.9, uniformity=0.85, power_in=0.4)
        with patch("optimizer.run_simulation", return_value=hot):
            sc_hot, *_ = evaluate_fov_flux(p, cfg)
        with patch("optimizer.run_simulation", return_value=filled):
            sc_fill, *_ = evaluate_fov_flux(p, cfg)
        self.assertGreater(sc_fill, sc_hot)


class TestInjectAndAllowExtra(unittest.TestCase):
    def test_inject_six_extras_uses_available_slots(self):
        p = default_params()
        out = inject_anamorphic_lenses(p, n_extra=6, mode="crossed")
        n_en = sum(1 for e in out["elements"] if e.get("enabled"))
        # collector + extras, capped by stack size
        self.assertGreaterEqual(n_en, 4)
        self.assertLessEqual(n_en, MAX_ELEMENTS)
        self.assertLessEqual(len(out["elements"]), MAX_ELEMENTS)

    def test_optimize_without_allow_keeps_element_count(self):
        p = default_params()
        p["source"]["mode"] = "single"
        p["mla"]["enabled"] = False
        p["use_warp"] = False
        before = sum(1 for e in p["elements"] if e.get("enabled"))
        cfg = config_from_panel(
            objective="rect_fill",
            allow_extra_lenses=False,
            extra_lenses=4,
            rays_per_eval=250,
            max_evals=8,
            polish=False,
        )
        cfg.population_size = 3
        cfg.seed = 1
        cfg.force_cpu = True
        result = optimize_fov_flux(p, cfg)
        after = sum(1 for e in result.params["elements"] if e.get("enabled"))
        self.assertEqual(after, before)

    def test_rect_fill_keeps_p2_when_flux_dips_but_aspect_improves(self):
        """Extra surfaces cost Fresnel; a small FOV-flux drop must not discard them."""
        p1 = default_params()
        p2 = inject_anamorphic_lenses(p1, n_extra=5, mode="crossed")
        r1 = OptimizeResult(
            params=p1,
            score=0.50,
            fov_flux=0.40,
            uniformity=0.72,
            collection=0.40,
            aspect_error=0.25,
            footprint_aspect=1.02,
        )
        r2 = OptimizeResult(
            params=p2,
            score=0.44,
            fov_flux=0.35,
            uniformity=0.68,
            collection=0.35,
            aspect_error=0.07,
            footprint_aspect=1.24,
        )
        winner, phase = select_two_phase_winner(
            r1, r2, OptimizeConfig(objective="rect_fill")
        )
        self.assertEqual(phase, "1+2")
        n_en = sum(1 for e in winner.params["elements"] if e.get("enabled"))
        self.assertGreaterEqual(n_en, 4)

    def test_rect_fill_still_rejects_collapsed_p2(self):
        p1 = default_params()
        p2 = inject_anamorphic_lenses(p1, n_extra=2, mode="crossed")
        r1 = OptimizeResult(
            params=p1, score=0.5, fov_flux=0.40, uniformity=0.7, collection=0.4
        )
        r2 = OptimizeResult(
            params=p2, score=0.01, fov_flux=0.01, uniformity=0.1, collection=0.01
        )
        winner, phase = select_two_phase_winner(
            r1, r2, OptimizeConfig(objective="rect_fill")
        )
        self.assertEqual(phase, "1")
        self.assertIs(winner, r1)

    def test_rect_fill_with_five_extras_keeps_them(self):
        """End-to-end: allowing 5 extras must not finish on the single collector."""
        p = default_params()
        p["source"]["mode"] = "single"
        p["mla"]["enabled"] = False
        p["use_warp"] = False
        p["fov_width"] = 50.0
        p["fov_height"] = 28.0
        cfg = config_from_panel(
            objective="rect_fill",
            allow_extra_lenses=True,
            extra_lenses=5,
            anamorphic_mode="crossed",
            rays_per_eval=220,
            max_evals=10,
            polish=False,
        )
        cfg.population_size = 3
        cfg.force_cpu = True
        cfg.seed = 4
        result = optimize_fov_flux(p, cfg)
        n_en = sum(1 for e in result.params["elements"] if e.get("enabled"))
        self.assertEqual(result.phase, "1+2")
        self.assertGreaterEqual(n_en, 4)
        modes = [
            e.get("surface_mode")
            for e in result.params["elements"]
            if e.get("enabled")
        ]
        self.assertTrue(
            any(m in ("cylinder_x", "biconic") for m in modes),
            f"expected anamorphic elements, got {modes}",
        )

    def test_focus_optimize_does_not_inject(self):
        p = default_params()
        p["source"]["mode"] = "single"
        p["mla"]["enabled"] = False
        p["use_warp"] = False
        before = sum(1 for e in p["elements"] if e.get("enabled"))
        cfg = OptimizeConfig(
            objective="focus",
            two_phase=True,
            allow_extra_lenses=True,
            extra_anamorphic_lenses=3,
            rays_per_eval=250,
            max_evals=8,
            population_size=3,
            polish=False,
            seed=2,
            force_cpu=True,
        )
        result = optimize_fov_flux(p, cfg)
        after = sum(1 for e in result.params["elements"] if e.get("enabled"))
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
