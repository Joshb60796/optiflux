"""Side-view zoom must survive batch redraws and not squash the Y axis."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (
    group_slider_limits,
    resolve_applied_side_limits,
    side_full_extent,
    should_draw_display_rays,
)
from engine import default_params
from focus import group_z_bounds


class TestSideFullExtent(unittest.TestCase):
    def test_stores_independent_transverse_axes(self):
        ext = side_full_extent(0.0, 100.0, t_y=12.0, t_x=40.0)
        z0, z1, ymin, ymax, xmin, xmax = ext
        self.assertAlmostEqual(z0, 0.0)
        self.assertAlmostEqual(z1, 100.0)
        self.assertAlmostEqual(ymin, -12.0)
        self.assertAlmostEqual(ymax, 12.0)
        self.assertAlmostEqual(xmin, -40.0)
        self.assertAlmostEqual(xmax, 40.0)


class TestResolveAppliedSideLimits(unittest.TestCase):
    def test_missing_ylim_does_not_reset_zoomed_z(self):
        """Regression: X–Z ylim unset used to call set_xlim(full) on the shared Z axis."""
        xlim, ylim = resolve_applied_side_limits(
            stored_xlim=(20.0, 40.0),
            stored_ylim=None,
            zmin=0.0,
            zmax=120.0,
            tmin=-15.0,
            tmax=15.0,
        )
        self.assertAlmostEqual(xlim[0], 20.0)
        self.assertAlmostEqual(xlim[1], 40.0)
        self.assertAlmostEqual(ylim[0], -15.0)
        self.assertAlmostEqual(ylim[1], 15.0)

    def test_keeps_independent_y_zoom_when_z_is_full(self):
        xlim, ylim = resolve_applied_side_limits(
            stored_xlim=None,
            stored_ylim=(-4.0, 4.0),
            zmin=0.0,
            zmax=200.0,
            tmin=-20.0,
            tmax=20.0,
        )
        self.assertAlmostEqual(xlim[0], 0.0)
        self.assertAlmostEqual(xlim[1], 200.0)
        self.assertAlmostEqual(ylim[0], -4.0)
        self.assertAlmostEqual(ylim[1], 4.0)

    def test_xz_uses_its_own_transverse_extent(self):
        ext = side_full_extent(0.0, 80.0, t_y=10.0, t_x=35.0)
        _z0, _z1, ymin, ymax, xmin, xmax = ext
        _xlim, ylim = resolve_applied_side_limits(
            stored_xlim=None,
            stored_ylim=None,
            zmin=ext[0],
            zmax=ext[1],
            tmin=xmin,
            tmax=xmax,
        )
        self.assertAlmostEqual(ylim[0], -35.0)
        self.assertAlmostEqual(ylim[1], 35.0)
        self.assertNotAlmostEqual(ymin, xmin)


class TestGroupSliderLimits(unittest.TestCase):
    def test_matches_group_z_bounds(self):
        p = default_params()
        p["source"]["source_z"] = 0.0
        p["target_z"] = 150.0
        p["elements"][0].update(enabled=True, thickness=6.0, air_after=2.0)
        for e in p["elements"][1:]:
            e["enabled"] = False
        lo, hi = group_slider_limits(p)
        blo, bhi = group_z_bounds(p)
        self.assertAlmostEqual(lo, blo)
        self.assertAlmostEqual(hi, bhi)
        self.assertGreater(hi, lo)
        self.assertGreater(lo, 0.0)
        self.assertLess(hi, 150.0)


class TestShouldDrawDisplayRays(unittest.TestCase):
    def test_preview_with_auto_off_skips_rays(self):
        self.assertFalse(
            should_draw_display_rays(auto_run=False, dragging=False, preview=True)
        )

    def test_preview_with_auto_on_draws_rays(self):
        self.assertTrue(
            should_draw_display_rays(auto_run=True, dragging=False, preview=True)
        )

    def test_drag_never_draws_rays(self):
        self.assertFalse(
            should_draw_display_rays(auto_run=True, dragging=True, preview=True)
        )
        self.assertFalse(
            should_draw_display_rays(auto_run=True, dragging=True, preview=False)
        )

    def test_full_redraw_after_requested_trace_draws_rays(self):
        """Manual Trace rays still shows paths even if Auto-run is off."""
        self.assertTrue(
            should_draw_display_rays(auto_run=False, dragging=False, preview=False)
        )


if __name__ == "__main__":
    unittest.main()
