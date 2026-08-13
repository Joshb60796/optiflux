"""
3D isometric view: optical→plot transform and scene builder smoke tests.
"""
from __future__ import annotations

import math
import os
import sys
import unittest
from typing import Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Headless backend before pyplot/axes3d use
import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure

from engine import default_blocker, default_params, run_simulation
from view_3d import (
    _pt,
    _w2p,
    build_scene,
    downsample_grid,
    irradiance_rgba,
)


def _optical_camera(elev: float, azim: float) -> Tuple[float, float, float]:
    """Matplotlib camera (plot frame) inverted through optical→plot."""
    er, ar = math.radians(elev), math.radians(azim)
    plot_x = math.cos(er) * math.cos(ar)
    plot_y = math.cos(er) * math.sin(ar)
    plot_z = math.sin(er)
    # Inverse of _pt: plot = (opt Z, opt X, opt Y)
    return plot_y, plot_z, plot_x


class TestWorldToPlotTransform(unittest.TestCase):
    """Target Plane pose: Y vertical (plot Z), +X toward viewer, +Z receding."""

    def test_optical_y_is_plot_vertical(self):
        # Matplotlib's screen-vertical axis is plot Z — same as Target Plane Y.
        self.assertEqual(_pt(0.0, 3.5, 0.0), (0.0, 0.0, 3.5))

    def test_optical_z_to_plot_x(self):
        self.assertEqual(_pt(0.0, 0.0, 10.0), (10.0, 0.0, 0.0))

    def test_optical_x_to_plot_y(self):
        self.assertEqual(_pt(5.0, 0.0, 0.0), (0.0, 5.0, 0.0))

    def test_combined_point(self):
        # (x,y,z)=(1,2,3) → (Z, X, Y) = (3, 1, 2)
        self.assertEqual(_pt(1.0, 2.0, 3.0), (3.0, 1.0, 2.0))

    def test_w2p_arrays(self):
        x = np.array([1.0, 0.0])
        y = np.array([0.0, 2.0])
        z = np.array([4.0, 0.0])
        px, py, pz = _w2p(x, y, z)
        np.testing.assert_allclose(px, [4.0, 0.0])
        np.testing.assert_allclose(py, [1.0, 0.0])
        np.testing.assert_allclose(pz, [0.0, 2.0])

    def test_w2p_scalar(self):
        px, py, pz = _w2p(2.0, -1.0, 7.0)
        self.assertEqual((px, py, pz), (7.0, 2.0, -1.0))

    def test_light_axis_monotonic_in_plot_x(self):
        """Points along optical +Z map to increasing plot X."""
        zs = [0.0, 10.0, 40.0, 80.0]
        pxs = [_pt(0.0, 0.0, z)[0] for z in zs]
        self.assertEqual(pxs, zs)
        self.assertTrue(all(pxs[i] < pxs[i + 1] for i in range(len(pxs) - 1)))


class TestBuildSceneSmoke(unittest.TestCase):
    def _axes(self):
        fig = Figure(figsize=(4, 3), dpi=72)
        ax = fig.add_subplot(111, projection="3d")
        return fig, ax

    def test_default_params_no_exception(self):
        fig, ax = self._axes()
        p = default_params()
        build_scene(ax, p, result=None, max_rays=0)
        self.assertTrue(math.isfinite(ax.get_xlim()[0]))
        self.assertTrue(math.isfinite(ax.get_xlim()[1]))
        title = ax.get_title()
        self.assertIn("right-drag", title.lower())
        self.assertIn("zoom", title.lower())
        # Optical labels: plot X=Z, plot Y=X, plot Z=Y (Y vertical)
        self.assertIn("Z", ax.get_xlabel())
        self.assertIn("X", ax.get_ylabel())
        self.assertIn("Y", ax.get_zlabel())
        fig.clf()

    def test_default_camera_from_plus_x_minus_z(self):
        """Open looking from +X / −Z so +X comes toward you and +Z recedes."""
        fig, ax = self._axes()
        build_scene(ax, default_params(), result=None, max_rays=0)
        elev = float(ax.elev)
        azim = float(ax.azim)
        ox, oy, oz = _optical_camera(elev, azim)
        self.assertGreater(ox, 0.0)
        self.assertLess(oz, 0.0)
        self.assertGreater(oy, 0.0)
        self.assertGreater(elev, 8.0)
        self.assertLess(elev, 35.0)
        fig.clf()

    def test_with_blockers_and_rays(self):
        fig, ax = self._axes()
        p = default_params()
        p["blockers"] = [
            default_blocker(
                z=20, shape="circle", outer_w=15, outer_h=15, inner_w=5, label="Stop"
            ),
            default_blocker(
                z=50, shape="rect", outer_w=40, outer_h=30, inner_w=0, label="Wall"
            ),
        ]
        p["total_rays"] = 300
        p["display_rays"] = 40
        p["use_warp"] = False
        r = run_simulation(p)
        build_scene(ax, p, r, max_rays=20)
        # Target Z should fall inside plot X range (optical Z → plot X)
        tz = float(p["target_z"])
        xmin, xmax = ax.get_xlim()
        self.assertLessEqual(xmin, tz + 1.0)
        self.assertGreaterEqual(xmax, 0.0 - 1.0)
        elev = getattr(ax, "elev", None)
        azim = getattr(ax, "azim", None)
        self.assertIsNotNone(elev)
        self.assertIsNotNone(azim)
        self.assertTrue(math.isfinite(float(azim)))
        fig.clf()

    def test_irradiance_rgba_matches_grid(self):
        g = np.zeros((10, 12))
        g[4:7, 5:8] = 1.0
        rgba = irradiance_rgba(g)
        self.assertEqual(rgba.shape, (10, 12, 4))
        # Hot patch is brighter than the empty border
        self.assertGreater(rgba[5, 6, :3].sum(), rgba[0, 0, :3].sum())
        self.assertGreater(rgba[5, 6, 3], rgba[0, 0, 3])

    def test_irradiance_rgba_empty_safe(self):
        rgba = irradiance_rgba(np.zeros((4, 4)))
        self.assertEqual(rgba.shape[-1], 4)
        self.assertTrue(np.all(np.isfinite(rgba)))

    def test_downsample_caps_size(self):
        g = np.ones((200, 180))
        d = downsample_grid(g, max_n=48)
        self.assertLessEqual(max(d.shape), 48)

    def test_scene_paints_target_map_from_result(self):
        fig, ax = self._axes()
        p = default_params()
        p["use_warp"] = False
        p["total_rays"] = 400
        p["display_rays"] = 30
        r = run_simulation(p)
        n0 = len(ax.collections)
        build_scene(ax, p, r, max_rays=15)
        # Heatmap + glass + FOV add collections beyond an empty axes
        self.assertGreater(len(ax.collections), n0)
        fig.clf()

    def test_empty_blockers_and_no_optics(self):
        fig, ax = self._axes()
        p = default_params()
        p["elements"][0]["enabled"] = False
        p["blockers"] = []
        build_scene(ax, p, None)
        xr = ax.get_xlim()
        self.assertGreater(xr[1] - xr[0], 1.0)
        fig.clf()

    def test_mla_scene(self):
        fig, ax = self._axes()
        p = default_params()
        p["source"]["mode"] = "cob"
        p["source"]["rows"] = 2
        p["source"]["cols"] = 2
        p["mla"]["enabled"] = True
        p["mla"]["aim_to_fov"] = False
        build_scene(ax, p, None, max_rays=0)
        self.assertTrue(math.isfinite(ax.get_ylim()[0]))
        fig.clf()


if __name__ == "__main__":
    unittest.main()
