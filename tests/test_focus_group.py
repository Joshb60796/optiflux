"""Large flood layout limits, default resin, and source-image focus search."""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import (
    DEFAULT_ELEMENT,
    LENS_SEMI_MAX_MM,
    LENS_Z_MAX_MM,
    default_params,
    run_simulation,
)
from focus import (
    enabled_stack_length_mm,
    focus_group_on_target,
    group_z_bounds,
    image_blur_at_z,
    map_half_covering_fov,
    paraxial_object_distances,
    source_field_points,
)


class TestDefaultFormlabsClear(unittest.TestCase):
    def test_default_element_is_formlabs_clear(self):
        self.assertEqual(DEFAULT_ELEMENT["material"], "FORMLABS_CLEAR")

    def test_default_params_element1_is_formlabs_clear(self):
        p = default_params()
        self.assertEqual(p["elements"][0]["material"], "FORMLABS_CLEAR")
        self.assertEqual(p["elements"][2]["material"], "FORMLABS_CLEAR")


class TestLargeFloodLayout(unittest.TestCase):
    def test_30mm_source_200mm_fov_150mm_throw_traces(self):
        p = default_params()
        p["use_warp"] = False
        p["total_rays"] = 250
        p["display_rays"] = 0
        p["mla"]["enabled"] = False
        p["source"]["die_width"] = 30.0
        p["source"]["die_height"] = 30.0
        p["source"]["source_z"] = 0.0
        p["target_z"] = 150.0
        p["fov_width"] = 200.0
        p["fov_height"] = 200.0
        p["map_half_w"] = 120.0
        p["map_half_h"] = 120.0
        p["lens_z_start"] = 55.0
        p["elements"][0]["aperture"] = 28.0
        r = run_simulation(p)
        self.assertGreater(r.stats["launched"], 0)
        self.assertTrue(math.isfinite(float(r.stats.get("collection", 0.0))))

    def test_map_grows_to_cover_200mm_fov(self):
        hw, hh = map_half_covering_fov(200.0, 200.0, 0.0, 0.0, 50.0, 40.0)
        self.assertGreaterEqual(hw, 100.0)
        self.assertGreaterEqual(hh, 100.0)

    def test_ui_ranges_allow_mid_throw_group(self):
        self.assertGreaterEqual(LENS_Z_MAX_MM, 150.0)
        self.assertGreaterEqual(LENS_SEMI_MAX_MM, 28.0)


class TestFocusGroup(unittest.TestCase):
    def test_stack_length_excludes_trailing_air(self):
        p = default_params()
        p["mla"]["enabled"] = False
        p["elements"][0].update(enabled=True, thickness=6.0, air_after=4.0)
        p["elements"][1].update(enabled=True, thickness=4.0, air_after=99.0)
        for e in p["elements"][2:]:
            e["enabled"] = False
        # 6 + 4 (air) + 4 thick; trailing 99 is not part of the glass pack
        self.assertAlmostEqual(enabled_stack_length_mm(p), 14.0)

    def test_bounds_keep_group_between_source_and_target(self):
        p = default_params()
        p["source"]["source_z"] = 0.0
        p["target_z"] = 150.0
        p["elements"][0].update(enabled=True, thickness=6.0, air_after=2.0)
        for e in p["elements"][1:]:
            e["enabled"] = False
        lo, hi = group_z_bounds(p)
        self.assertGreater(lo, 0.0)
        self.assertLess(hi, 150.0)
        self.assertGreater(hi, lo)

    def test_field_points_include_30mm_corners(self):
        p = default_params()
        p["source"]["die_width"] = 30.0
        p["source"]["die_height"] = 30.0
        pts = source_field_points(p)
        xs = [a[0] for a in pts]
        ys = [a[1] for a in pts]
        self.assertGreaterEqual(max(xs), 13.0)
        self.assertLessEqual(min(xs), -13.0)
        self.assertGreaterEqual(max(ys), 13.0)

    def test_paraxial_distances_real_when_throw_allows(self):
        p = default_params()
        p["mla"]["enabled"] = False
        p["source"]["source_z"] = 0.0
        p["target_z"] = 120.0
        # Strong biconvex so 4f < throw
        p["elements"][0].update(
            enabled=True, R1=16.0, R2=-16.0, thickness=5.0, aperture=12.0
        )
        for e in p["elements"][1:]:
            e["enabled"] = False
        us = paraxial_object_distances(p)
        self.assertGreaterEqual(len(us), 1)
        for u in us:
            self.assertGreater(u, 0.5)
            self.assertLess(u, 119.0)

    def test_focus_search_reduces_blur_vs_bad_z(self):
        p = default_params()
        p["use_warp"] = False
        p["mla"]["enabled"] = False
        p["source"]["source_z"] = 0.0
        p["source"]["die_width"] = 2.0
        p["source"]["die_height"] = 2.0
        p["source"]["half_angle_deg"] = 25.0
        p["target_z"] = 120.0
        p["elements"][0].update(
            enabled=True,
            R1=16.0,
            R2=-16.0,
            thickness=5.0,
            aperture=14.0,
            material="FORMLABS_CLEAR",
        )
        for e in p["elements"][1:]:
            e["enabled"] = False
        p["lens_z_start"] = 50.0
        bad = image_blur_at_z(p, 50.0, rays_per_point=24)
        out = focus_group_on_target(
            p, n_scan=13, n_refine=7, rays_per_point=24
        )
        self.assertTrue(out["ok"])
        self.assertLess(out["blur_mm"], bad["blur_mm"] * 1.05 + 0.5)
        lo, hi = group_z_bounds(p)
        self.assertGreaterEqual(out["lens_z_start"], lo - 1e-6)
        self.assertLessEqual(out["lens_z_start"], hi + 1e-6)


if __name__ == "__main__":
    unittest.main()
