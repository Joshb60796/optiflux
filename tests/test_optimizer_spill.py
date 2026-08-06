"""
High-value tests for FOV optimizer scoring: spill, waste, containment,
and current-stack vs rectangular OptimizeConfig contracts.

Scoring-path tests mock run_simulation so results are deterministic.
"""
from __future__ import annotations

import math
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import default_params
from optimizer import (
    OptimizeConfig,
    build_variable_list,
    evaluate_fov_flux,
    inject_anamorphic_lenses,
    optimize_fov_flux,
)


def _fake_sim(
    *,
    source_power: float = 1.0,
    power_in: float = 0.3,
    plane_power: float = 0.8,
    collection: float = 0.8,
    uniformity: float = 0.7,
    coverage: float = 0.6,
    profile_fill: float = 0.6,
    size_error: float = 0.1,
    aspect_error: float = 0.05,
    footprint_aspect: float = 1.25,
    orientation_flipped: float = 0.0,
):
    """Minimal SimResult stand-in for evaluate_fov_flux."""
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
        "sig_x": 10.0,
        "sig_y": 8.0,
    }
    stats = {
        "source_power": source_power,
        "map_power": plane_power * 0.9,
        "plane_power": plane_power,
        "collection": collection,
        "fov": fov,
        "efl": 40.0,
        "n_dies": 1,
        "n_surfaces": 2,
    }
    return SimpleNamespace(stats=stats, map=None, paths=[], dies=[], surfaces=[])


class TestSpillContainmentScoring(unittest.TestCase):
    def test_spill_weight_lowers_score_when_spill_present(self):
        """Same FOV flux & plane power with spill: higher spill_weight → lower score."""
        p = default_params()
        # FOV 0.3 of source; plane 0.8 → spill = 0.5
        fake = _fake_sim(power_in=0.3, plane_power=0.8, collection=0.8)
        with patch("optimizer.run_simulation", return_value=fake):
            sc0, ff0, *_ = evaluate_fov_flux(
                p,
                OptimizeConfig(
                    spill_weight=0.0,
                    waste_weight=0.0,
                    aspect_weight=0.0,
                    fill_weight=0.0,
                    coverage_mix=0.5,
                    force_cpu=True,
                ),
            )
            sc1, ff1, *_ = evaluate_fov_flux(
                p,
                OptimizeConfig(
                    spill_weight=5.0,
                    waste_weight=0.0,
                    aspect_weight=0.0,
                    fill_weight=0.0,
                    coverage_mix=0.5,
                    force_cpu=True,
                ),
            )
        self.assertAlmostEqual(ff0, ff1, places=9)
        self.assertGreater(sc0, 0.0)
        self.assertLess(sc1, sc0)

    def test_zero_spill_insensitive_to_spill_weight(self):
        """When all plane power is in FOV, spill_weight must not change score."""
        p = default_params()
        fake = _fake_sim(power_in=0.5, plane_power=0.5, collection=0.5)
        with patch("optimizer.run_simulation", return_value=fake):
            sc0, *_ = evaluate_fov_flux(
                p,
                OptimizeConfig(
                    spill_weight=0.0, waste_weight=0.0, fill_weight=0.0, aspect_weight=0.0
                ),
            )
            sc1, *_ = evaluate_fov_flux(
                p,
                OptimizeConfig(
                    spill_weight=10.0, waste_weight=0.0, fill_weight=0.0, aspect_weight=0.0
                ),
            )
        self.assertAlmostEqual(sc0, sc1, places=9)

    def test_waste_weight_lowers_score(self):
        p = default_params()
        fake = _fake_sim(power_in=0.2, plane_power=0.2, collection=0.2)
        with patch("optimizer.run_simulation", return_value=fake):
            sc0, *_ = evaluate_fov_flux(
                p,
                OptimizeConfig(
                    spill_weight=0.0, waste_weight=0.0, fill_weight=0.0, aspect_weight=0.0
                ),
            )
            sc1, *_ = evaluate_fov_flux(
                p,
                OptimizeConfig(
                    spill_weight=0.0, waste_weight=4.0, fill_weight=0.0, aspect_weight=0.0
                ),
            )
        self.assertLess(sc1, sc0)

    def test_higher_coverage_raises_score(self):
        """Hot-spot (low coverage) loses to better FOV fill at same flux."""
        p = default_params()
        hot = _fake_sim(
            power_in=0.4, plane_power=0.45, collection=0.45,
            coverage=0.15, profile_fill=0.2, uniformity=0.2,
        )
        filled = _fake_sim(
            power_in=0.4, plane_power=0.45, collection=0.45,
            coverage=0.9, profile_fill=0.9, uniformity=0.85,
        )
        cfg = OptimizeConfig(
            spill_weight=1.0,
            waste_weight=0.5,
            fill_weight=1.0,
            aspect_weight=0.0,
            coverage_mix=0.95,
            uniformity_weight=1.0,
        )
        with patch("optimizer.run_simulation", return_value=hot):
            sc_hot, *_ = evaluate_fov_flux(p, cfg)
        with patch("optimizer.run_simulation", return_value=filled):
            sc_fill, *_ = evaluate_fov_flux(p, cfg)
        self.assertGreater(sc_fill, sc_hot)

    def test_better_containment_raises_score(self):
        """Same FOV flux: less plane spill → higher score (contain_factor)."""
        p = default_params()
        leaky = _fake_sim(power_in=0.25, plane_power=0.9, collection=0.9)
        tight = _fake_sim(power_in=0.25, plane_power=0.28, collection=0.28)
        cfg = OptimizeConfig(
            spill_weight=0.0,  # isolation: contain_factor only
            waste_weight=0.0,
            fill_weight=0.0,
            aspect_weight=0.0,
            coverage_mix=0.0,  # ignore coverage mix
            uniformity_weight=0.0,
        )
        with patch("optimizer.run_simulation", return_value=leaky):
            sc_leaky, *_ = evaluate_fov_flux(p, cfg)
        with patch("optimizer.run_simulation", return_value=tight):
            sc_tight, *_ = evaluate_fov_flux(p, cfg)
        self.assertGreater(sc_tight, sc_leaky)

    def test_orientation_flip_penalizes(self):
        p = default_params()
        ok = _fake_sim(orientation_flipped=0.0, aspect_error=0.1)
        bad = _fake_sim(orientation_flipped=1.0, aspect_error=0.1)
        cfg = OptimizeConfig(
            spill_weight=0.0, waste_weight=0.0, fill_weight=0.0, aspect_weight=0.0
        )
        with patch("optimizer.run_simulation", return_value=ok):
            sc_ok, *_ = evaluate_fov_flux(p, cfg)
        with patch("optimizer.run_simulation", return_value=bad):
            sc_bad, *_ = evaluate_fov_flux(p, cfg)
        self.assertAlmostEqual(sc_bad, sc_ok * 0.35, places=9)

    def test_live_evaluate_finite(self):
        """Integration: real MC still returns finite score with spill weights."""
        p = default_params()
        p["source"]["mode"] = "single"
        p["total_rays"] = 500
        p["use_warp"] = False
        cfg = OptimizeConfig(
            rays_per_eval=400,
            map_res=32,
            force_cpu=True,
            spill_weight=2.0,
            waste_weight=1.0,
            coverage_mix=0.9,
        )
        score, ff, uni, col, ae, fa = evaluate_fov_flux(p, cfg)
        self.assertTrue(math.isfinite(score))
        self.assertGreaterEqual(ff, 0.0)
        self.assertLessEqual(ff, 1.0 + 1e-6)
        self.assertLessEqual(col, 1.0 + 1e-6)


class TestOptimizeConfigContracts(unittest.TestCase):
    """Header 'current stack' vs left-panel rectangular optimizer settings."""

    def _current_stack_cfg(self) -> OptimizeConfig:
        return OptimizeConfig(
            two_phase=False,
            extra_anamorphic_lenses=0,
            aspect_weight=0.0,
            spill_weight=1.2,
            waste_weight=0.6,
            coverage_mix=0.85,
            force_cpu=True,
        )

    def _rect_cfg(self) -> OptimizeConfig:
        return OptimizeConfig(
            two_phase=True,
            extra_anamorphic_lenses=2,
            aspect_weight=1.5,
            spill_weight=2.0,
            waste_weight=1.0,
            coverage_mix=0.95,
            anamorphic_mode="crossed",
            force_cpu=True,
        )

    def test_current_stack_config_no_phase2(self):
        cfg = self._current_stack_cfg()
        self.assertFalse(cfg.two_phase)
        self.assertEqual(cfg.extra_anamorphic_lenses, 0)
        self.assertEqual(cfg.aspect_weight, 0.0)

    def test_rect_config_enables_phase2(self):
        cfg = self._rect_cfg()
        self.assertTrue(cfg.two_phase)
        self.assertGreaterEqual(cfg.extra_anamorphic_lenses, 1)
        self.assertGreater(cfg.aspect_weight, 0.0)
        self.assertGreaterEqual(cfg.spill_weight, self._current_stack_cfg().spill_weight)

    def test_current_stack_variables_only_enabled_elements(self):
        p = default_params()
        p["elements"][0]["enabled"] = True
        p["elements"][1]["enabled"] = True
        p["elements"][2]["enabled"] = False
        cfg = self._current_stack_cfg()
        cfg.optimize_radii = True
        cfg.optimize_thickness = True
        vars_ = build_variable_list(p, cfg)
        names = [v.name for v in vars_]
        # Element 0 and 1 free; element 2 disabled
        self.assertTrue(any(n.startswith("elements.0.") for n in names))
        self.assertTrue(any(n.startswith("elements.1.") for n in names))
        self.assertFalse(any(n.startswith("elements.2.") for n in names))
        self.assertTrue(any(n == "lens_z_start" for n in names))

    def test_optimize_element_indices_restricts(self):
        p = default_params()
        for i in range(3):
            p["elements"][i]["enabled"] = True
        cfg = OptimizeConfig(
            two_phase=False,
            optimize_element_indices=[1],
            optimize_radii=True,
            optimize_thickness=False,
            optimize_air_gaps=False,
            optimize_aperture=False,
            optimize_lens_z=False,
        )
        vars_ = build_variable_list(p, cfg)
        names = [v.name for v in vars_]
        self.assertTrue(all(n.startswith("elements.1.") for n in names))
        self.assertFalse(any("elements.0." in n for n in names))

    def test_two_phase_small_budget_runs(self):
        p = default_params()
        p["source"]["mode"] = "single"
        p["mla"]["enabled"] = False
        p["use_warp"] = False
        cfg = OptimizeConfig(
            rays_per_eval=300,
            max_evals=12,
            population_size=3,
            polish=False,
            seed=7,
            two_phase=True,
            extra_anamorphic_lenses=2,
            anamorphic_mode="crossed",
            force_cpu=True,
            aspect_weight=1.5,
            spill_weight=2.0,
            waste_weight=1.0,
            fill_weight=1.5,
            coverage_mix=0.9,
        )
        result = optimize_fov_flux(p, cfg)
        self.assertGreaterEqual(result.n_evals, 1)
        self.assertTrue(math.isfinite(result.score))
        self.assertIn(result.phase, ("1", "1+2", "2", ""))
        n_en = sum(1 for e in result.params["elements"] if e.get("enabled"))
        self.assertLessEqual(n_en, 3)

    def test_inject_does_not_run_when_extra_zero(self):
        """two_phase with extra=0 falls through to single phase (no inject)."""
        p = default_params()
        p["use_warp"] = False
        before_modes = [e.get("surface_mode") for e in p["elements"] if e.get("enabled")]
        cfg = OptimizeConfig(
            rays_per_eval=250,
            max_evals=8,
            population_size=3,
            polish=False,
            seed=1,
            two_phase=True,
            extra_anamorphic_lenses=0,
            force_cpu=True,
        )
        result = optimize_fov_flux(p, cfg)
        after_en = [e for e in result.params["elements"] if e.get("enabled")]
        # Single-phase: still one collector (no forced crossed pair)
        self.assertEqual(len(after_en), len([m for m in before_modes]))


class TestInjectAnamorphicHelpers(unittest.TestCase):
    def test_inject_crossed_two(self):
        p = default_params()
        out = inject_anamorphic_lenses(p, n_extra=2, mode="crossed")
        modes = [e.get("surface_mode") for e in out["elements"] if e.get("enabled")]
        self.assertIn("cylinder_x", modes)
        self.assertIn("cylinder_y", modes)


if __name__ == "__main__":
    unittest.main()
