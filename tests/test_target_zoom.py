"""Target-plane zoom: virtual wall and recorded-map growth."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (
    AUTO_EXPAND_MAP_ON_ZOOM,
    MAP_HALF_MAX_MM,
    TGT_WALL_HALF_MM,
    map_half_to_cover_view,
    target_zoom_extent,
)
from engine import RayPath, target_plane_hits


class TestTargetZoomExtent(unittest.TestCase):
    def test_wall_is_much_larger_than_default_map(self):
        ext = target_zoom_extent(50.0, 40.0)
        self.assertLessEqual(ext[0], -TGT_WALL_HALF_MM)
        self.assertGreaterEqual(ext[1], TGT_WALL_HALF_MM)
        self.assertLessEqual(ext[2], -TGT_WALL_HALF_MM)
        self.assertGreaterEqual(ext[3], TGT_WALL_HALF_MM)
        self.assertGreater(TGT_WALL_HALF_MM, 40.0 * 10)

    def test_already_huge_map_wins(self):
        ext = target_zoom_extent(8000.0, 8000.0)
        self.assertAlmostEqual(ext[1], 8000.0)


class TestMapHalfToCoverView(unittest.TestCase):
    def test_grows_when_view_exceeds_map(self):
        w, h, grew = map_half_to_cover_view((-200, 200), (-180, 180), 50.0, 40.0)
        self.assertTrue(grew)
        self.assertGreater(w, 50.0)
        self.assertGreater(h, 40.0)
        self.assertGreaterEqual(w, 200.0)

    def test_does_not_shrink(self):
        w, h, grew = map_half_to_cover_view((-10, 10), (-8, 8), 50.0, 40.0)
        self.assertFalse(grew)
        self.assertAlmostEqual(w, 50.0)
        self.assertAlmostEqual(h, 40.0)

    def test_caps_at_max(self):
        w, h, grew = map_half_to_cover_view(
            (-1e6, 1e6), (-1e6, 1e6), 50.0, 40.0
        )
        self.assertTrue(grew)
        self.assertEqual(w, MAP_HALF_MAX_MM)
        self.assertEqual(h, MAP_HALF_MAX_MM)


class TestZoomDoesNotRewriteMap(unittest.TestCase):
    def test_auto_expand_stays_off(self):
        """Zooming out must not grow the recorded heatmap (that hides rays)."""
        self.assertFalse(AUTO_EXPAND_MAP_ON_ZOOM)


class TestTargetPlaneHits(unittest.TestCase):
    def test_includes_hits_inside_and_outside_the_map(self):
        inside = RayPath(history=[(0.0, 0.0, 0.0), (2.0, 1.0, 80.0)], terminated="target")
        outside = RayPath(history=[(0.0, 0.0, 0.0), (120.0, -90.0, 80.0)], terminated="target")
        miss = RayPath(history=[(0.0, 0.0, 0.0), (1.0, 1.0, 20.0)], terminated="absorb")
        hits = target_plane_hits([inside, outside, miss], target_z=80.0)
        self.assertIn((2.0, 1.0), hits)
        self.assertIn((120.0, -90.0), hits)
        self.assertEqual(len(hits), 2)

    def test_uses_last_point_when_z_matches(self):
        path = RayPath(history=[(0.0, 0.0, 0.0), (4.0, -3.0, 79.6)], terminated="miss")
        hits = target_plane_hits([path], target_z=80.0)
        self.assertEqual(hits, [(4.0, -3.0)])


if __name__ == "__main__":
    unittest.main()
