"""Target FOV resize / move math used by side-view and target-plane handles."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (
    FOV_SIZE_MAX_MM,
    FOV_SIZE_MIN_MM,
    clamp_fov_size,
    move_fov_center,
    pick_fov_rect_handle,
    resize_fov_from_end,
    resize_fov_rect,
)


class TestClampFovSize(unittest.TestCase):
    def test_bounds(self):
        self.assertEqual(clamp_fov_size(1.0), FOV_SIZE_MIN_MM)
        self.assertEqual(clamp_fov_size(999.0), FOV_SIZE_MAX_MM)
        self.assertAlmostEqual(clamp_fov_size(40.0), 40.0)


class TestResizeFovFromEnd(unittest.TestCase):
    def test_keeps_center_and_sets_span(self):
        cy, h = resize_fov_from_end(0.0, 32.0, 25.0)
        self.assertAlmostEqual(cy, 0.0)
        self.assertAlmostEqual(h, 50.0)

    def test_works_from_the_low_end(self):
        cy, h = resize_fov_from_end(4.0, 20.0, -16.0)
        self.assertAlmostEqual(cy, 4.0)
        self.assertAlmostEqual(h, 40.0)


class TestResizeFovRect(unittest.TestCase):
    def test_right_edge_keeps_left(self):
        cx, cy, w, h = resize_fov_rect(0.0, 0.0, 40.0, 32.0, edge="right", x=30.0, y=0.0)
        self.assertAlmostEqual(w, 50.0)
        # left stays at -20 → centre = -20 + 25
        self.assertAlmostEqual(cx, 5.0)
        self.assertAlmostEqual(h, 32.0)
        self.assertAlmostEqual(cy, 0.0)

    def test_top_edge_keeps_bottom(self):
        cx, cy, w, h = resize_fov_rect(0.0, 0.0, 40.0, 32.0, edge="top", x=0.0, y=26.0)
        self.assertAlmostEqual(h, 42.0)
        # bottom stays at -16 → centre = -16 + 21
        self.assertAlmostEqual(cy, 5.0)
        self.assertAlmostEqual(w, 40.0)

    def test_corner_sets_both(self):
        cx, cy, w, h = resize_fov_rect(
            0.0, 0.0, 40.0, 32.0, corner="ne", x=30.0, y=24.0
        )
        self.assertAlmostEqual(w, 50.0)
        self.assertAlmostEqual(h, 40.0)
        self.assertAlmostEqual(cx, 5.0)
        self.assertAlmostEqual(cy, 4.0)

    def test_lock_aspect_on_edge_scales_other(self):
        cx, cy, w, h = resize_fov_rect(
            0.0, 0.0, 40.0, 20.0, edge="right", x=40.0, y=0.0, lock_aspect=True
        )
        # left stays at -20; new width 60 → height scales 20 * (60/40) = 30
        self.assertAlmostEqual(w, 60.0)
        self.assertAlmostEqual(h, 30.0)
        self.assertAlmostEqual(cx, 10.0)


class TestMoveFovCenter(unittest.TestCase):
    def test_translates(self):
        cx, cy = move_fov_center(0.0, 0.0, 5.0, -3.0)
        self.assertAlmostEqual(cx, 5.0)
        self.assertAlmostEqual(cy, -3.0)


class TestPickFovRectHandle(unittest.TestCase):
    def test_picks_right_edge(self):
        hit = pick_fov_rect_handle(20.0, 0.0, 0.0, 0.0, 40.0, 32.0, hit_r=2.0)
        self.assertEqual(hit[0], "edge")
        self.assertEqual(hit[1], "right")

    def test_picks_ne_corner_over_edge(self):
        hit = pick_fov_rect_handle(20.0, 16.0, 0.0, 0.0, 40.0, 32.0, hit_r=3.0)
        self.assertEqual(hit[0], "corner")
        self.assertEqual(hit[1], "ne")

    def test_picks_move_inside(self):
        hit = pick_fov_rect_handle(1.0, 1.0, 0.0, 0.0, 40.0, 32.0, hit_r=2.0)
        self.assertEqual(hit[0], "move")

    def test_miss_outside(self):
        hit = pick_fov_rect_handle(80.0, 80.0, 0.0, 0.0, 40.0, 32.0, hit_r=2.0)
        self.assertIsNone(hit)


if __name__ == "__main__":
    unittest.main()
