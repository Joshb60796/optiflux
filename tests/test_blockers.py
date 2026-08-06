"""
Absorbing ray blockers / aperture stops.

Physics: zero-thickness plano surfaces that kill rays on hit (outer minus hole).
Display thickness does not affect ray results.
"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import (
    assemble_surfaces,
    build_blockers,
    default_blocker,
    default_params,
    run_simulation,
    trace_ray,
)


def _solid_circle(z=10.0, r=20.0, **kw):
    """Face-on solid disk (vertical stop)."""
    kw.setdefault("orient", "vertical")
    return default_blocker(z=z, shape="circle", outer_w=r, outer_h=r, inner_w=0, **kw)


def _stop_circle(z=10.0, outer=20.0, inner=5.0, **kw):
    """Face-on aperture stop (vertical)."""
    kw.setdefault("orient", "vertical")
    return default_blocker(
        z=z, shape="circle", outer_w=outer, outer_h=outer, inner_w=inner, inner_h=inner, **kw
    )


def _tube(z=20.0, r=10.0, length=40.0, **kw):
    kw.setdefault("orient", "tube")
    return default_blocker(
        z=z, shape="circle", outer_w=r, outer_h=r, length=length, **kw
    )


def _body(z=20.0, half_w=12.0, half_h=10.0, length=40.0, **kw):
    kw.setdefault("orient", "horizontal")
    return default_blocker(
        z=z, shape="rect", outer_w=half_w, outer_h=half_h, length=length, **kw
    )


class TestBuildBlockers(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(build_blockers([]), [])
        self.assertEqual(build_blockers(None), [])

    def test_solid_fields(self):
        surfs = build_blockers([_solid_circle(z=12.0, r=15.0, label="Wall")])
        self.assertEqual(len(surfs), 1)
        s = surfs[0]
        self.assertEqual(s.interaction, "absorb")
        self.assertEqual(s.aperture_shape, "circle")
        self.assertEqual(s.geom, "plane_z")
        self.assertAlmostEqual(s.z_vertex, 12.0)
        self.assertAlmostEqual(s.aperture, 15.0)
        self.assertIsNone(s.inner_aperture)
        self.assertAlmostEqual(s.radius, 0.0)

    def test_default_rect_is_horizontal_body(self):
        b = default_blocker(shape="rect")
        self.assertEqual(b["orient"], "horizontal")
        surfs = build_blockers([b])
        # Four walls: top, bot, left, right
        self.assertEqual(len(surfs), 4)
        geoms = {s.geom for s in surfs}
        self.assertEqual(geoms, {"plane_y", "plane_x"})

    def test_default_circle_is_tube(self):
        b = default_blocker(shape="circle")
        self.assertEqual(b["orient"], "tube")
        surfs = build_blockers([b])
        self.assertEqual(len(surfs), 1)
        self.assertEqual(surfs[0].geom, "cylinder_z")

    def test_disabled_skipped(self):
        b = _solid_circle()
        b["enabled"] = False
        self.assertEqual(build_blockers([b]), [])

    def test_thickness_ignored_by_physics(self):
        b1 = _solid_circle(z=10)
        b1["thickness"] = 1.0
        b2 = _solid_circle(z=10)
        b2["thickness"] = 10.0
        s1, s2 = build_blockers([b1])[0], build_blockers([b2])[0]
        ok1, _, p1, path1 = trace_ray(
            (0, 0, 0), (0, 0, 1), 1.0, 550, [s1], 50, store_path=True
        )
        ok2, _, p2, path2 = trace_ray(
            (0, 0, 0), (0, 0, 1), 1.0, 550, [s2], 50, store_path=True
        )
        self.assertFalse(ok1)
        self.assertFalse(ok2)
        self.assertEqual(path1.terminated, "absorb")
        self.assertEqual(path2.terminated, "absorb")


class TestAbsorbPhysics(unittest.TestCase):
    def test_solid_panel_kills_on_axis(self):
        surfs = build_blockers([_solid_circle(z=10, r=20)])
        ok, pt, pwr, path = trace_ray(
            (0, 0, 0), (0, 0, 1), 1.0, 550, surfs, 80, store_path=True
        )
        self.assertFalse(ok)
        self.assertIsNone(pt)
        self.assertEqual(pwr, 0.0)
        self.assertEqual(path.terminated, "absorb")
        self.assertIn("absorb", path.events)
        # Last history point near panel
        self.assertAlmostEqual(path.history[-1][2], 10.0, places=4)

    def test_solid_misses_outside_outer(self):
        surfs = build_blockers([_solid_circle(z=10, r=5)])
        # Ray aimed to cross z=10 at x=15 (outside)
        d = (15 / math.hypot(15, 10), 0, 10 / math.hypot(15, 10))
        ok, pt, pwr, path = trace_ray(
            (0, 0, 0), d, 1.0, 550, surfs, 80, store_path=True
        )
        self.assertTrue(ok)
        self.assertEqual(path.terminated, "target")

    def test_hole_passes_on_axis(self):
        surfs = build_blockers([_stop_circle(z=10, outer=20, inner=5)])
        ok, pt, pwr, path = trace_ray(
            (0, 0, 0), (0, 0, 1), 1.0, 550, surfs, 80,
            apply_fresnel=False, store_path=True,
        )
        self.assertTrue(ok)
        self.assertEqual(path.terminated, "target")
        self.assertAlmostEqual(pt[0], 0.0, places=5)
        self.assertAlmostEqual(pt[2], 80.0, places=5)

    def test_annulus_kills_off_axis(self):
        surfs = build_blockers([_stop_circle(z=10, outer=20, inner=5)])
        # Cross z=10 at y=8 (in annulus)
        d = (0, 8 / math.hypot(8, 10), 10 / math.hypot(8, 10))
        ok, pt, pwr, path = trace_ray(
            (0, 0, 0), d, 1.0, 550, surfs, 80, store_path=True
        )
        self.assertFalse(ok)
        self.assertEqual(path.terminated, "absorb")

    def test_rect_hole_pass_and_frame_kill(self):
        b = default_blocker(
            z=15, shape="rect", outer_w=20, outer_h=10, inner_w=4, inner_h=3
        )
        surfs = build_blockers([b])
        # On axis → through hole
        ok, _, _, path = trace_ray(
            (0, 0, 0), (0, 0, 1), 1.0, 550, surfs, 50, store_path=True
        )
        self.assertTrue(ok)
        self.assertEqual(path.terminated, "target")
        # Frame: x=10, y=0 at z=15
        d = (10 / math.hypot(10, 15), 0, 15 / math.hypot(10, 15))
        ok2, _, _, path2 = trace_ray(
            (0, 0, 0), d, 1.0, 550, surfs, 50, store_path=True
        )
        self.assertFalse(ok2)
        self.assertEqual(path2.terminated, "absorb")

    def test_both_sides_absorb(self):
        """Ray traveling −Z into panel also absorbs (if kill_backward off)."""
        surfs = build_blockers([_solid_circle(z=10, r=20)])
        ok, _, pwr, path = trace_ray(
            (0, 0, 20), (0, 0, -1), 1.0, 550, surfs, target_z=-5,
            kill_backward=False, store_path=True,
        )
        self.assertFalse(ok)
        self.assertEqual(path.terminated, "absorb")
        self.assertEqual(pwr, 0.0)

    def test_after_window_still_absorbs(self):
        elements = [{
            "enabled": True, "R1": 0, "R2": 0, "thickness": 3,
            "air_after": 1, "aperture": 30, "material": "N_BK7",
            "k1": 0, "k2": 0, "A4_1": 0, "A4_2": 0, "surface_mode": "rotational",
        }]
        blockers = [_solid_circle(z=20, r=25)]
        surfs = assemble_surfaces(elements, z_start=2.0, blockers=blockers)
        ok, _, _, path = trace_ray(
            (0, 0, 0), (0, 0, 1), 1.0, 550, surfs, 80,
            apply_fresnel=False, store_path=True,
        )
        self.assertFalse(ok)
        self.assertEqual(path.terminated, "absorb")
        self.assertGreaterEqual(path.n_refractions, 2)

    def test_sim_stats_n_absorb(self):
        p = default_params()
        p["elements"][0]["enabled"] = False
        p["blockers"] = [_solid_circle(z=5, r=50)]
        p["total_rays"] = 400
        p["source"]["mode"] = "single"
        p["source"]["half_angle_deg"] = 20
        r = run_simulation(p)
        self.assertIn("n_absorb", r.stats)
        self.assertGreater(r.stats["n_absorb"], 50)
        self.assertLess(r.stats["collection"], 0.15)

    def test_hole_sim_collects(self):
        p = default_params()
        p["elements"][0]["enabled"] = False
        p["blockers"] = [_stop_circle(z=5, outer=40, inner=8)]
        p["total_rays"] = 800
        p["source"]["mode"] = "single"
        p["source"]["half_angle_deg"] = 15
        r = run_simulation(p)
        self.assertGreater(r.stats["hit"], 20)
        self.assertGreater(r.stats["collection"], 0.05)


class TestAssembleSurfaces(unittest.TestCase):
    def test_optics_plus_blocker(self):
        p = default_params()
        p["blockers"] = [_solid_circle(z=50, r=10)]
        surfs = assemble_surfaces(
            p["elements"], p["lens_z_start"], blockers=p["blockers"]
        )
        labels = [s.label for s in surfs]
        self.assertTrue(any(lab.startswith("E1") for lab in labels))
        self.assertTrue(any(lab.startswith("BLK") for lab in labels))
        self.assertTrue(any(s.interaction == "absorb" for s in surfs))


class TestBlockerEdges(unittest.TestCase):
    def test_decentered_panel_misses_axis_kills_offset(self):
        """Panel centered at x=15: axis ray misses, ray through x=15 absorbs."""
        b = default_blocker(
            z=10, shape="circle", outer_w=5, outer_h=5, inner_w=0, label="Off"
        )
        b["x0"] = 15.0
        b["y0"] = 0.0
        surfs = build_blockers([b])
        ok_ax, _, _, path_ax = trace_ray(
            (0, 0, 0), (0, 0, 1), 1.0, 550, surfs, 40, store_path=True
        )
        self.assertTrue(ok_ax)
        self.assertEqual(path_ax.terminated, "target")
        # Aim through decenter: hit z=10 at x=15
        L = math.hypot(15.0, 10.0)
        d = (15.0 / L, 0.0, 10.0 / L)
        ok_off, _, _, path_off = trace_ray(
            (0, 0, 0), d, 1.0, 550, surfs, 40, store_path=True
        )
        self.assertFalse(ok_off)
        self.assertEqual(path_off.terminated, "absorb")

    def test_hole_edge_inside_vs_outside(self):
        """Just inside hole → pass; just outside hole (in frame) → absorb."""
        outer, inner = 20.0, 5.0
        surfs = build_blockers([_stop_circle(z=10, outer=outer, inner=inner)])
        # r=4.5 < 5 → through hole
        y_in = 4.5
        L = math.hypot(y_in, 10.0)
        d_in = (0.0, y_in / L, 10.0 / L)
        ok_in, _, _, path_in = trace_ray(
            (0, 0, 0), d_in, 1.0, 550, surfs, 50, store_path=True
        )
        self.assertTrue(ok_in)
        self.assertEqual(path_in.terminated, "target")
        # r=5.5 > 5 and < 20 → annulus
        y_out = 5.5
        L2 = math.hypot(y_out, 10.0)
        d_out = (0.0, y_out / L2, 10.0 / L2)
        ok_out, _, _, path_out = trace_ray(
            (0, 0, 0), d_out, 1.0, 550, surfs, 50, store_path=True
        )
        self.assertFalse(ok_out)
        self.assertEqual(path_out.terminated, "absorb")

    def test_rect_hole_corner_vs_frame(self):
        b = default_blocker(
            z=12, shape="rect", outer_w=20, outer_h=10, inner_w=4, inner_h=3
        )
        surfs = build_blockers([b])
        # Through rectangular hole near corner of hole (inside)
        ok, _, _, path = trace_ray(
            (0, 0, 0), (0, 0, 1), 1.0, 550, surfs, 40, store_path=True
        )
        self.assertTrue(ok)
        # Frame at |x|=10 (outer half-w 20, hole half-w 4)
        L = math.hypot(10.0, 12.0)
        d = (10.0 / L, 0.0, 12.0 / L)
        ok2, _, _, path2 = trace_ray(
            (0, 0, 0), d, 1.0, 550, surfs, 40, store_path=True
        )
        self.assertFalse(ok2)
        self.assertEqual(path2.terminated, "absorb")

    def test_progressive_reports_n_absorb(self):
        from progressive import run_simulation_progressive

        p = default_params()
        p["elements"][0]["enabled"] = False
        p["blockers"] = [_solid_circle(z=5, r=40)]
        p["source"]["mode"] = "single"
        p["source"]["half_angle_deg"] = 25
        p["use_warp"] = False
        r = run_simulation_progressive(
            p,
            n_batches=2,
            rays_per_batch=200,
            display_per_batch=30,
        )
        self.assertIsNotNone(r)
        self.assertIn("n_absorb", r.stats)
        self.assertGreater(r.stats["n_absorb"], 10)
        self.assertLess(r.stats["collection"], 0.25)

    def test_display_thickness_ignored_in_hit_z(self):
        """Face stop: hit is always at plane z, not ± thickness/2."""
        b = _solid_circle(z=12.0, r=20)
        b["thickness"] = 8.0
        surfs = build_blockers([b])
        ok, _, _, path = trace_ray(
            (0, 0, 0), (0, 0, 1), 1.0, 550, surfs, 40, store_path=True
        )
        self.assertFalse(ok)
        self.assertEqual(path.terminated, "absorb")
        self.assertAlmostEqual(path.history[-1][2], 12.0, places=4)


class TestHorizontalAndTube(unittest.TestCase):
    def test_tube_passes_on_axis(self):
        """Bore of tube: axial ray reaches target."""
        surfs = build_blockers([_tube(z=30, r=10, length=40)])
        ok, pt, _, path = trace_ray(
            (0, 0, 0), (0, 0, 1), 1.0, 550, surfs, 80, store_path=True
        )
        self.assertTrue(ok)
        self.assertEqual(path.terminated, "target")
        self.assertAlmostEqual(pt[2], 80.0, places=4)

    def test_tube_absorbs_radial_hit(self):
        """Ray that crosses the barrel wall is absorbed."""
        # Start inside bore at z=0, go outward +Y to hit r=10 at some z
        surfs = build_blockers([_tube(z=20, r=10, length=50)])
        # From (0,0,5) direction toward (0, 15, 25) — crosses cylinder
        o = (0.0, 0.0, 5.0)
        target_pt = (0.0, 15.0, 25.0)
        d = (
            target_pt[0] - o[0],
            target_pt[1] - o[1],
            target_pt[2] - o[2],
        )
        L = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
        d = (d[0] / L, d[1] / L, d[2] / L)
        ok, _, _, path = trace_ray(o, d, 1.0, 550, surfs, 100, store_path=True)
        self.assertFalse(ok)
        self.assertEqual(path.terminated, "absorb")

    def test_horizontal_body_passes_axis(self):
        surfs = build_blockers([_body(z=25, half_w=12, half_h=10, length=40)])
        ok, _, _, path = trace_ray(
            (0, 0, 0), (0, 0, 1), 1.0, 550, surfs, 80, store_path=True
        )
        self.assertTrue(ok)
        self.assertEqual(path.terminated, "target")

    def test_horizontal_body_absorbs_upward_ray(self):
        """Ray headed toward top wall y=+10 inside Z span is absorbed."""
        surfs = build_blockers([_body(z=20, half_w=15, half_h=10, length=40)])
        # From origin toward top wall midspan
        o = (0.0, 0.0, 5.0)
        # Hit plane y=10 at z=20: direction (0, 10, 15)
        d = (0.0, 10.0 / math.hypot(10, 15), 15.0 / math.hypot(10, 15))
        ok, _, _, path = trace_ray(o, d, 1.0, 550, surfs, 100, store_path=True)
        self.assertFalse(ok)
        self.assertEqual(path.terminated, "absorb")
        # Hit near y=10
        self.assertAlmostEqual(path.history[-1][1], 10.0, places=3)

    def test_horizontal_body_side_wall(self):
        surfs = build_blockers([_body(z=20, half_w=12, half_h=10, length=40)])
        o = (0.0, 0.0, 10.0)
        # Toward +X wall at x=12, z=20
        d = (12.0 / math.hypot(12, 10), 0.0, 10.0 / math.hypot(12, 10))
        ok, _, _, path = trace_ray(o, d, 1.0, 550, surfs, 80, store_path=True)
        self.assertFalse(ok)
        self.assertEqual(path.terminated, "absorb")


if __name__ == "__main__":
    unittest.main()
