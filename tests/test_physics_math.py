"""
Comprehensive physics & mathematics validation for OptiFlux.

Run:
    python -m pytest tests/test_physics_math.py -v
or:
    python tests/test_physics_math.py
"""
from __future__ import annotations

import math
import os
import sys
import unittest

# project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from materials_catalog import (
    LAMBDA_C,
    LAMBDA_D,
    LAMBDA_F,
    LAMBDA_HE_NE,
    MATERIALS,
    VISIBLE_NM_DEFAULT,
    VISIBLE_NM_MAX,
    VISIBLE_NM_MIN,
    abbe_number,
    clamp_visible_nm,
    material_id_from_name,
    refractive_index,
)
from engine import (
    OpticalSurface,
    build_source_array,
    build_surfaces,
    default_params,
    fresnel_T,
    lensmaker_f,
    run_simulation,
    sample_lambertian_cone,
    snell_refract,
    trace_ray,
    v_dot,
    v_norm,
)
from export_cad import sag_z, mesh_singlet, mesh_mla, LensSpec, build_lens_specs_from_params
from lens_shapes import apply_shape, SHAPE_LABELS
from mla_geometry import build_mla_lens_specs, scale_element_to_lenslet


# ═══════════════════════════════════════════════════════════════════════════
# Materials & wavelength
# ═══════════════════════════════════════════════════════════════════════════

class TestVisibleWavelength(unittest.TestCase):
    def test_visible_window(self):
        self.assertEqual(VISIBLE_NM_MIN, 380.0)
        self.assertEqual(VISIBLE_NM_MAX, 780.0)
        self.assertTrue(VISIBLE_NM_MIN <= VISIBLE_NM_DEFAULT <= VISIBLE_NM_MAX)

    def test_clamp(self):
        self.assertEqual(clamp_visible_nm(200), 380.0)
        self.assertEqual(clamp_visible_nm(900), 780.0)
        self.assertEqual(clamp_visible_nm(550), 550.0)

    def test_default_params_visible(self):
        p = default_params()
        wl = p["source"]["wavelength_nm"]
        self.assertGreaterEqual(wl, VISIBLE_NM_MIN)
        self.assertLessEqual(wl, VISIBLE_NM_MAX)


class TestMaterialsCatalog(unittest.TestCase):
    def test_bk7_nd(self):
        # Schott N-BK7 n_d = 1.5168
        n = refractive_index("N_BK7", LAMBDA_D)
        self.assertAlmostEqual(n, 1.5168, places=3)

    def test_bk7_alias(self):
        n1 = refractive_index("BK7", LAMBDA_D)
        n2 = refractive_index("N_BK7", LAMBDA_D)
        self.assertAlmostEqual(n1, n2, places=10)

    def test_fused_silica_nd(self):
        n = refractive_index("FUSED_SILICA", LAMBDA_D)
        self.assertAlmostEqual(n, 1.4585, places=3)

    def test_sf11_higher_than_bk7(self):
        self.assertGreater(
            refractive_index("N_SF11", LAMBDA_D),
            refractive_index("N_BK7", LAMBDA_D),
        )

    def test_acrylic_pmma(self):
        n = refractive_index("ACRYLIC_PMMA", LAMBDA_D)
        self.assertAlmostEqual(n, 1.491, places=2)
        n2 = refractive_index("PMMA", LAMBDA_D)
        self.assertAlmostEqual(n, n2, places=6)

    def test_polycarbonate_nd(self):
        n = refractive_index("POLYCARBONATE", LAMBDA_D)
        self.assertAlmostEqual(n, 1.585, places=2)

    def test_formlabs_clear_range(self):
        n = refractive_index("FORMLABS_CLEAR", LAMBDA_D)
        self.assertGreater(n, 1.50)
        self.assertLess(n, 1.58)

    def test_catalog_n_d_ref_matches_model(self):
        """All materials that declare n_d_ref must match the dispersion model."""
        for mid, m in MATERIALS.items():
            if "n_d_ref" not in m:
                continue
            n = refractive_index(mid, LAMBDA_D)
            tol = 0.002 if m.get("category") in ("glass", "crystal") else 0.005
            self.assertAlmostEqual(
                n, float(m["n_d_ref"]), delta=tol, msg=mid
            )

    def test_dispersion_normal(self):
        # Normal dispersion: n_F > n_d > n_C for glasses
        for mid in ("N_BK7", "N_SF11", "FUSED_SILICA", "ACRYLIC_PMMA"):
            nF = refractive_index(mid, LAMBDA_F)
            nd = refractive_index(mid, LAMBDA_D)
            nC = refractive_index(mid, LAMBDA_C)
            self.assertGreater(nF, nd, msg=mid)
            self.assertGreater(nd, nC, msg=mid)

    def test_abbe_bk7(self):
        # N-BK7 V_d ≈ 64.17
        V = abbe_number("N_BK7")
        self.assertIsNotNone(V)
        self.assertAlmostEqual(V, 64.17, delta=1.5)

    def test_abbe_sf11_lower(self):
        # Flints have lower Abbe than crowns
        self.assertLess(abbe_number("N_SF11"), abbe_number("N_BK7"))

    def test_all_materials_finite_visible(self):
        for mid in MATERIALS:
            for wl in (400, 450, 550, 650, 700, 780):
                n = refractive_index(mid, wl, custom_n=1.6)
                self.assertTrue(math.isfinite(n), f"{mid} @ {wl}")
                self.assertGreaterEqual(n, 1.0, f"{mid} @ {wl}")

    def test_display_name_roundtrip(self):
        mid = material_id_from_name("Acrylic / PMMA")
        self.assertEqual(mid, "ACRYLIC_PMMA")
        mid2 = material_id_from_name("Formlabs Clear Resin")
        self.assertEqual(mid2, "FORMLABS_CLEAR")


# ═══════════════════════════════════════════════════════════════════════════
# Vector / Snell / Fresnel
# ═══════════════════════════════════════════════════════════════════════════

class TestSnellFresnel(unittest.TestCase):
    def test_normal_incidence_no_bend(self):
        T, tir = snell_refract((0, 0, 1), (0, 0, 1), 1.0, 1.5)
        self.assertFalse(tir)
        self.assertAlmostEqual(T[0], 0.0, places=10)
        self.assertAlmostEqual(T[1], 0.0, places=10)
        self.assertAlmostEqual(T[2], 1.0, places=10)

    def test_snell_law_angle(self):
        n1, n2 = 1.0, 1.5
        th1 = math.radians(30)
        I = (math.sin(th1), 0.0, math.cos(th1))
        T, tir = snell_refract(I, (0, 0, 1), n1, n2)
        self.assertFalse(tir)
        th2 = math.asin(min(1.0, math.hypot(T[0], T[1])))
        expect = math.asin(n1 / n2 * math.sin(th1))
        self.assertAlmostEqual(th2, expect, places=8)

    def test_reciprocity(self):
        # Air→glass then glass→air recovers original angle (no TIR)
        n1, n2 = 1.0, 1.5
        th1 = math.radians(25)
        I = (math.sin(th1), 0.0, math.cos(th1))
        T, tir = snell_refract(I, (0, 0, 1), n1, n2)
        self.assertFalse(tir)
        # reverse: incident from glass along -T toward interface with N pointing to glass?
        # Ray in glass going toward -Z onto interface: direction = -T_flip
        # After first refraction T goes +Z into glass. Reverse ray is -T.
        I2 = (-T[0], -T[1], -T[2])
        # Normal still (0,0,1) pointing toward incident medium for reverse: glass is incident
        # so normal should face glass (+Z is into air side...). Our snell flips N if needed.
        T2, tir2 = snell_refract(I2, (0, 0, 1), n2, n1)
        self.assertFalse(tir2)
        # Should exit near -I
        self.assertAlmostEqual(T2[0], -I[0], places=6)
        self.assertAlmostEqual(T2[2], -I[2], places=6)

    def test_tir_critical_angle(self):
        n1, n2 = 1.5, 1.0
        th_c = math.asin(n2 / n1)
        # just below critical → transmit
        th = th_c - math.radians(1)
        I = (math.sin(th), 0.0, math.cos(th))
        _, tir = snell_refract(I, (0, 0, 1), n1, n2)
        self.assertFalse(tir)
        # above critical → TIR
        th = th_c + math.radians(1)
        I = (math.sin(th), 0.0, math.cos(th))
        R, tir = snell_refract(I, (0, 0, 1), n1, n2)
        self.assertTrue(tir)
        # reflection law: R_z should reverse relative to Z component
        self.assertAlmostEqual(R[0], I[0], places=6)
        self.assertAlmostEqual(R[2], -I[2], places=6)

    def test_fresnel_normal_incidence(self):
        n1, n2 = 1.0, 1.5
        T, tir = fresnel_T(n1, n2, 1.0)
        self.assertFalse(tir)
        # R = ((n1-n2)/(n1+n2))² , T = 1-R for power model
        R = ((n1 - n2) / (n1 + n2)) ** 2
        self.assertAlmostEqual(T, 1.0 - R, places=10)

    def test_fresnel_tir(self):
        T, tir = fresnel_T(1.5, 1.0, math.cos(math.radians(50)))
        self.assertTrue(tir)
        self.assertEqual(T, 0.0)

    def test_fresnel_unitarity(self):
        for th_deg in (0, 10, 20, 30, 40):
            cosi = math.cos(math.radians(th_deg))
            T, tir = fresnel_T(1.0, 1.5, cosi)
            self.assertFalse(tir)
            self.assertGreaterEqual(T, 0.0)
            self.assertLessEqual(T, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Surface sag & intersection
# ═══════════════════════════════════════════════════════════════════════════

class TestSurfaceSag(unittest.TestCase):
    def test_plano_sag_zero(self):
        s = OpticalSurface(z_vertex=5.0, radius=0.0, aperture=10)
        self.assertAlmostEqual(s.sag(0), 0.0)
        self.assertAlmostEqual(s.sag(5), 0.0)
        self.assertAlmostEqual(s.surface_z(3, 4), 5.0)

    def test_sphere_sag_formula(self):
        R = 50.0
        s = OpticalSurface(z_vertex=0.0, radius=R, aperture=20)
        r = 10.0
        # exact sphere: z = R - sqrt(R² - r²) for R>0 vertex at 0... 
        # our sag: c r² / (1+sqrt(1-c²r²)) with c=1/R equals R - sqrt(R²-r²)
        exact = R - math.sqrt(R * R - r * r)
        self.assertAlmostEqual(s.sag(r), exact, places=9)

    def test_conic_parabola(self):
        # k=-1 parabola: z = r²/(2R)
        R = 40.0
        s = OpticalSurface(z_vertex=0.0, radius=R, k=-1.0, aperture=15)
        r = 8.0
        self.assertAlmostEqual(s.sag(r), (r * r) / (2 * R), places=9)

    def test_export_sag_matches_engine(self):
        R, k, a4 = 30.0, -0.5, 1e-5
        s = OpticalSurface(z_vertex=0, radius=R, k=k, a4=a4, aperture=12)
        for r in (0, 1, 3, 5, 8):
            self.assertAlmostEqual(s.sag(r), sag_z(r, R, k, a4), places=12)

    def test_decenter_surface(self):
        s = OpticalSurface(z_vertex=2.0, radius=20.0, aperture=5, x0=1.5, y0=-2.0)
        # at axis of lenslet
        z_ax = s.surface_z(1.5, -2.0)
        self.assertAlmostEqual(z_ax, 2.0 + s.sag(0), places=9)
        # off aperture
        hit = s.intersect((1.5, -2.0, 0.0), (0, 0, 1))
        self.assertIsNotNone(hit)

    def test_intersect_plane(self):
        s = OpticalSurface(z_vertex=10.0, radius=0.0, aperture=50)
        hit = s.intersect((0, 0, 0), (0, 0, 1))
        self.assertIsNotNone(hit)
        t, p, n = hit
        self.assertAlmostEqual(p[2], 10.0, places=6)
        self.assertAlmostEqual(n[2], 1.0, places=6)

    def test_intersect_sphere_vertex(self):
        s = OpticalSurface(z_vertex=5.0, radius=25.0, aperture=15)
        hit = s.intersect((0, 0, 0), (0, 0, 1))
        self.assertIsNotNone(hit)
        _, p, n = hit
        self.assertAlmostEqual(p[0], 0.0, places=5)
        self.assertAlmostEqual(p[1], 0.0, places=5)
        self.assertAlmostEqual(p[2], 5.0, places=5)
        self.assertAlmostEqual(n[2], 1.0, places=4)


# ═══════════════════════════════════════════════════════════════════════════
# Lensmaker & shapes
# ═══════════════════════════════════════════════════════════════════════════

class TestLensmakerAndShapes(unittest.TestCase):
    def test_equiconvex_focal_length(self):
        R = 50.0
        n = 1.5
        d = 5.0
        f = lensmaker_f(R, -R, n, d)
        # thin lens approx: f = R/(2(n-1)) = 50
        # thick correction slightly longer
        self.assertAlmostEqual(f, R / (2 * (n - 1)), delta=3.0)

    def test_plano_convex_power(self):
        f = lensmaker_f(0.0, -40.0, 1.5, 4.0)
        # 1/f ≈ (n-1)/|R2|
        self.assertAlmostEqual(1 / f, 0.5 / 40.0, delta=0.002)

    def test_all_shapes_produce_finite_R(self):
        for _label, sid in SHAPE_LABELS:
            if sid == "custom":
                continue
            el = apply_shape(sid, R_mag=25, thickness=4, aperture=10)
            self.assertTrue(math.isfinite(el["R1"]))
            self.assertTrue(math.isfinite(el["R2"]))

    def test_shape_signs_biconvex(self):
        el = apply_shape("biconvex", R_mag=20)
        self.assertGreater(el["R1"], 0)
        self.assertLess(el["R2"], 0)

    def test_shape_pcx_flat_first(self):
        el = apply_shape("plano_convex_PCX", R_mag=20)
        self.assertEqual(el["R1"], 0.0)
        self.assertLess(el["R2"], 0)


# ═══════════════════════════════════════════════════════════════════════════
# Emission sampling
# ═══════════════════════════════════════════════════════════════════════════

class TestLambertianSampling(unittest.TestCase):
    def test_unit_direction(self):
        for _ in range(200):
            d = sample_lambertian_cone(90)
            self.assertAlmostEqual(math.hypot(*d), 1.0, places=8)
            self.assertGreaterEqual(d[2], 0.0)

    def test_cone_limit(self):
        half = 30.0
        cos_min = math.cos(math.radians(half))
        for _ in range(500):
            d = sample_lambertian_cone(half)
            self.assertGreaterEqual(d[2], cos_min - 1e-9)

    def test_mean_forward(self):
        # Lambertian: <cos θ> = 2/3 for full hemisphere
        N = 8000
        s = sum(sample_lambertian_cone(90)[2] for _ in range(N)) / N
        self.assertAlmostEqual(s, 2.0 / 3.0, delta=0.03)


# ═══════════════════════════════════════════════════════════════════════════
# End-to-end ray trace
# ═══════════════════════════════════════════════════════════════════════════

class TestRayTracePhysics(unittest.TestCase):
    def test_window_no_deviation(self):
        # Plano-plano plate should not deviate a normal ray
        surfaces = build_surfaces(
            [{
                "enabled": True, "R1": 0, "R2": 0, "thickness": 5,
                "air_after": 1, "aperture": 20, "material": "N_BK7",
                "k1": 0, "k2": 0, "A4_1": 0, "A4_2": 0,
            }],
            z_start=2.0,
        )
        ok, pt, pwr, _ = trace_ray(
            (0, 0, 0), (0, 0, 1), 1.0, 550.0, surfaces, target_z=50.0, apply_fresnel=False
        )
        self.assertTrue(ok)
        self.assertAlmostEqual(pt[0], 0.0, places=5)
        self.assertAlmostEqual(pt[1], 0.0, places=5)

    def test_fresnel_reduces_power(self):
        surfaces = build_surfaces(
            [{
                "enabled": True, "R1": 0, "R2": 0, "thickness": 3,
                "air_after": 1, "aperture": 20, "material": "N_BK7",
                "k1": 0, "k2": 0, "A4_1": 0, "A4_2": 0,
            }],
            z_start=1.0,
        )
        _, _, p_no, _ = trace_ray(
            (0, 0, 0), (0, 0, 1), 1.0, 550, surfaces, 40, apply_fresnel=False
        )
        _, _, p_yes, _ = trace_ray(
            (0, 0, 0), (0, 0, 1), 1.0, 550, surfaces, 40, apply_fresnel=True
        )
        self.assertAlmostEqual(p_no, 1.0, places=6)
        self.assertLess(p_yes, p_no)
        # two air-glass interfaces ≈ 0.96^2
        self.assertGreater(p_yes, 0.85)

    def test_positive_lens_bends_toward_axis(self):
        # Off-axis parallel ray through equiconvex should acquire dy/dz < 0 (toward axis)
        el = apply_shape("equiconvex", R_mag=30, thickness=4, aperture=12, material="N_BK7")
        surfaces = build_surfaces([el], z_start=5.0)
        y0 = 3.0
        ok, pt, _, path = trace_ray(
            (0, y0, 0), (0, 0, 1), 1.0, 550, surfaces, target_z=80.0,
            apply_fresnel=False, store_path=True,
        )
        self.assertTrue(ok)
        self.assertIsNotNone(path)
        h = path.history
        # Expect source → S1 → S2 → target
        self.assertGreaterEqual(len(h), 3)
        p0, p1 = h[-2], h[-1]  # exit segment after last surface toward target
        self.assertGreater(p1[2], p0[2])
        dy_dz = (p1[1] - p0[1]) / (p1[2] - p0[2])
        self.assertLess(dy_dz, 0.0, "focusing lens must bend +y ray toward axis")

    def test_mla_surfaces_per_die(self):
        p = default_params()
        p["source"]["mode"] = "cob"
        p["source"]["rows"] = 2
        p["source"]["cols"] = 2
        p["mla"]["enabled"] = True
        p["mla"]["aim_to_fov"] = False  # centres coincide with dies when aim is off
        dies = build_source_array(p["source"])
        surfs = build_surfaces(p["elements"], p["lens_z_start"], p["mla"], dies)
        # 4 dies × 2 surfaces = 8 (later elements disabled)
        self.assertEqual(len(surfs), 8)
        for s in surfs:
            self.assertTrue(
                any(math.isclose(s.x0, d.cx) and math.isclose(s.y0, d.cy) for d in dies)
            )

    def test_mla_aim_offsets_toward_fov_center(self):
        """With aim on, outer dies get lens centres shifted (still near the die)."""
        p = default_params()
        p["source"]["mode"] = "cob"
        p["source"]["rows"] = 2
        p["source"]["cols"] = 2
        p["source"]["pitch_x"] = 2.0
        p["source"]["pitch_y"] = 2.0
        p["mla"]["enabled"] = True
        p["mla"]["aim_to_fov"] = True
        p["mla"]["aim_strength"] = 1.0
        p["mla"]["_target_z"] = 40.0
        p["mla"]["_fov_cx"] = 0.0
        p["mla"]["_fov_cy"] = 0.0
        dies = build_source_array(p["source"])
        surfs = build_surfaces(p["elements"], p["lens_z_start"], p["mla"], dies)
        self.assertEqual(len(surfs), 8)
        # Pair surfaces with nearest die; offset should stay within ~half pitch
        for s in surfs:
            if not s.label.endswith("S1"):
                continue
            d = min(dies, key=lambda dd: math.hypot(s.x0 - dd.cx, s.y0 - dd.cy))
            self.assertLess(math.hypot(s.x0 - d.cx, s.y0 - d.cy), 1.05)
            # Outer dies should not sit exactly on axis-symmetric zero offset only —
            # at least some channels move (any non-zero for off-axis dies)
            if abs(d.cx) > 0.1 or abs(d.cy) > 0.1:
                # Offset direction is outward for positive lens aim (see channel_aim)
                self.assertGreater(math.hypot(s.x0, s.y0) + 1e-9, math.hypot(d.cx, d.cy) - 1e-6)

    def test_monte_carlo_collects_flux(self):
        p = default_params()
        p["total_rays"] = 1500
        p["source"]["wavelength_nm"] = 550
        p["elements"][0]["material"] = "ACRYLIC_PMMA"
        r = run_simulation(p)
        self.assertGreater(r.stats["hit"], 100)
        self.assertGreater(r.stats["collection"], 0.05)
        self.assertGreaterEqual(r.stats["collection"], 0.0)
        self.assertLessEqual(r.stats["collection"], 1.0 + 1e-6)

    def test_extended_source_not_point(self):
        # Die has finite size; spawn positions should vary
        dies = build_source_array({
            "mode": "single", "die_width": 2.0, "die_height": 1.0,
            "source_z": 0, "flux_per_die": 1, "wavelength_nm": 550,
            "half_angle_deg": 40,
        })
        xs, ys = [], []
        for _ in range(300):
            o, d, _, _ = dies[0].spawn_ray(1.0)
            xs.append(o[0])
            ys.append(o[1])
        self.assertGreater(max(xs) - min(xs), 0.5)
        self.assertGreater(max(ys) - min(ys), 0.3)

    def test_material_changes_focus(self):
        # Higher-n material → stronger lens → different EFL
        p = default_params()
        p["total_rays"] = 800
        p["elements"][0] = apply_shape("biconvex", R_mag=25, thickness=4, aperture=12, material="N_BK7")
        p["elements"][1]["enabled"] = False
        p["elements"][2]["enabled"] = False
        r1 = run_simulation(p)
        p["elements"][0]["material"] = "N_SF11"
        r2 = run_simulation(p)
        # Both should collect some light; EFLs differ
        self.assertNotAlmostEqual(r1.stats["efl"], r2.stats["efl"], places=2)


class TestAnamorphicRectFOV(unittest.TestCase):
    def test_biconic_different_meridians(self):
        s = OpticalSurface(
            z_vertex=0, radius=25.0, radius_y=40.0, mode="biconic", aperture=10
        )
        zx = s.sag_xy(5.0, 0.0)
        zy = s.sag_xy(0.0, 5.0)
        self.assertIsNotNone(zx)
        self.assertIsNotNone(zy)
        self.assertGreater(abs(zx - zy), 1e-4)

    def test_cylinder_x_no_y_power(self):
        s = OpticalSurface(z_vertex=0, radius=30.0, mode="cylinder_x", aperture=10)
        self.assertAlmostEqual(s.sag_xy(0.0, 6.0) or 0.0, 0.0, places=10)
        self.assertGreater(abs(s.sag_xy(6.0, 0.0) or 0.0), 0.1)

    def test_crossed_cylinder_design_valid_R(self):
        from rect_fov import design_crossed_cylinders_for_rect_fov

        d = design_crossed_cylinders_for_rect_fov(
            fov_width=48, fov_height=32, target_z=100, aperture=12
        )
        e1, e2 = d["elements"][0], d["elements"][1]
        self.assertEqual(e1["surface_mode"], "cylinder_x")
        self.assertEqual(e2["surface_mode"], "cylinder_y")
        self.assertGreater(abs(e1["R1"]), 12.0)
        self.assertGreater(abs(e2["R1y"]), 12.0)

    def test_rect_fov_sim_aspect_direction(self):
        """Wider FOV design should produce footprint aspect > 1 (σx > σy)."""
        import random

        from rect_fov import design_crossed_cylinders_for_rect_fov

        random.seed(11)
        d = design_crossed_cylinders_for_rect_fov(
            fov_width=60, fov_height=30, target_z=120, aperture=12
        )
        p = default_params()
        p["elements"] = d["elements"]
        p["lens_z_start"] = d["lens_z_start"]
        p["fov_width"] = 60
        p["fov_height"] = 30
        p["target_z"] = 120
        p["source"]["mode"] = "single"
        p["total_rays"] = 3000
        p["map_half_w"] = 60
        p["map_half_h"] = 40
        r = run_simulation(p)
        self.assertGreater(r.stats["fov"]["footprint_aspect"], 1.0)
        self.assertGreater(r.stats["collection"], 0.05)


class TestMeshExportConsistency(unittest.TestCase):
    def test_mesh_closed_manifold_counts(self):
        spec = LensSpec(R1=20, R2=-30, thickness=4, aperture=8, z_front=0)
        mesh = mesh_singlet(spec, n_radial=16, n_theta=32)
        self.assertEqual(mesh.vertices.shape[1], 3)
        self.assertGreater(len(mesh.faces), 100)
        # each face 3 indices in range
        self.assertTrue((mesh.faces >= 0).all())
        self.assertTrue((mesh.faces < len(mesh.vertices)).all())

    def test_vertex_units_mm_scale(self):
        spec = LensSpec(R1=25, R2=-25, thickness=5, aperture=10, z_front=0)
        mesh = mesh_singlet(spec, n_radial=12, n_theta=24)
        # aperture radius 10 mm → max radial ~10
        rad = np.sqrt(mesh.vertices[:, 0] ** 2 + mesh.vertices[:, 1] ** 2)
        self.assertLess(rad.max(), 10.5)
        self.assertGreater(rad.max(), 9.0)

    def test_mla_scales_element1_form_to_pitch(self):
        """Default bi-convex Element 1 → micro R/t when MLA scale_to_pitch is on."""
        p = default_params()
        p["source"]["mode"] = "cob"
        p["source"]["rows"] = 4
        p["source"]["cols"] = 4
        p["mla"]["enabled"] = True
        p["mla"]["scale_to_pitch"] = True
        dies = build_source_array(p["source"])
        specs, meta = build_mla_lens_specs(p, dies)
        self.assertEqual(len(specs), len(dies))
        self.assertLess(meta["scale"], 0.2)
        # Not macro cylinders: R scaled down from 40 mm design
        self.assertLess(abs(meta["R1"]), 8.0)
        self.assertGreater(abs(meta["R1"]), 0.5)
        edge = sag_z(meta["aperture"], meta["R1"])
        self.assertIsNotNone(edge)
        self.assertGreater(abs(edge), 0.03)  # visible dome, not flat top

    def test_mla_export_monolithic_with_domes(self):
        """CAD export is one solid plate with curved lenslets, not flat cylinders."""
        p = default_params()
        p["source"]["mode"] = "cob"
        p["source"]["rows"] = 4
        p["source"]["cols"] = 4
        p["mla"]["enabled"] = True
        p["mla"]["scale_to_pitch"] = True
        dies = build_source_array(p["source"])
        specs, mode = build_lens_specs_from_params(p, dies)
        self.assertEqual(mode, "mla")
        mesh = mesh_mla(specs, include_plate=True, n_radial=20)
        n = len(mesh.vertices) // 2
        front_z = mesh.vertices[:n, 2]
        span = float(front_z.max() - front_z.min())
        self.assertGreater(span, 0.04, "front face must show lenslet sag variation")
        self.assertGreater(len(mesh.faces), 500)
        # footprint covers full array, not isolated small disks only
        xs = mesh.vertices[:, 0]
        self.assertGreater(xs.max() - xs.min(), 3.0)


def run_all():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(run_all())
