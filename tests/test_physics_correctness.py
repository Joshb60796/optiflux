"""
High-rigor physics correctness tests for OptiFlux.

Validates Snell's law, Fresnel coefficients, surface geometry, thick-lens
focus, plano-plate lateral shift, dispersion / Abbe numbers, and energy
conservation against closed-form optics results.

Run:
    python -m pytest tests/test_physics_correctness.py -v
or:
    python validate_physics.py
"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from materials_catalog import (
    LAMBDA_C,
    LAMBDA_D,
    LAMBDA_F,
    LAMBDA_HE_NE,
    MATERIALS,
    abbe_number,
    refractive_index,
)
from engine import (
    OpticalSurface,
    build_surfaces,
    fresnel_T,
    lensmaker_f,
    run_simulation,
    sample_lambertian_cone,
    snell_refract,
    trace_ray,
    v_dot,
    v_norm,
    default_params,
)
from mla_geometry import thin_lens_focal_length_mm


# ═══════════════════════════════════════════════════════════════════════════
# Materials: n_d_ref, Sellmeier, Abbe
# ═══════════════════════════════════════════════════════════════════════════

class TestMaterialNdCatalog(unittest.TestCase):
    """Every material with n_d_ref must reproduce it at the d-line."""

    def test_all_n_d_ref_within_tolerance(self):
        for mid, m in MATERIALS.items():
            if "n_d_ref" not in m:
                continue
            n = refractive_index(mid, LAMBDA_D)
            ref = float(m["n_d_ref"])
            # Glasses: tight (Sellmeier). Plastics/resins: 0.005 allowed.
            tol = 0.002 if m.get("category") in ("glass", "crystal") else 0.005
            self.assertAlmostEqual(
                n, ref, delta=tol,
                msg=f"{mid}: n_d={n:.6f} vs ref={ref:.4f}",
            )

    def test_bk7_multi_wavelength_catalog(self):
        # Schott N-BK7 published-ish values (Sellmeier → literature match)
        # n_F ≈ 1.5224, n_d ≈ 1.5168, n_C ≈ 1.5143, n_HeNe ≈ 1.5151
        self.assertAlmostEqual(refractive_index("N_BK7", LAMBDA_F), 1.5224, places=3)
        self.assertAlmostEqual(refractive_index("N_BK7", LAMBDA_D), 1.5168, places=3)
        self.assertAlmostEqual(refractive_index("N_BK7", LAMBDA_C), 1.5143, places=3)
        self.assertAlmostEqual(refractive_index("N_BK7", LAMBDA_HE_NE), 1.5151, places=3)

    def test_abbe_catalog_glasses(self):
        # Published V_d (approx.)
        self.assertAlmostEqual(abbe_number("N_BK7"), 64.17, delta=1.0)
        self.assertAlmostEqual(abbe_number("N_SF11"), 25.68, delta=1.5)
        self.assertAlmostEqual(abbe_number("FUSED_SILICA"), 67.8, delta=2.0)
        # Crowns have higher Abbe than flints
        self.assertGreater(abbe_number("N_BK7"), abbe_number("N_SF5"))
        self.assertGreater(abbe_number("N_SK16"), abbe_number("N_SF11"))

    def test_plastic_abbe_reasonable(self):
        # Optical plastics: PMMA V≈50–60, PC V≈30
        self.assertGreater(abbe_number("ACRYLIC_PMMA"), 50.0)
        self.assertLess(abbe_number("POLYCARBONATE"), 40.0)
        self.assertGreater(abbe_number("POLYCARBONATE"), 20.0)

    def test_air_near_unity(self):
        n = refractive_index("AIR", 550)
        self.assertAlmostEqual(n, 1.000293, places=5)

    def test_higher_index_stronger_power(self):
        """Same radii: SF11 must have shorter |EFL| than BK7."""
        R1, R2, t = 40.0, -40.0, 4.0
        n_bk = refractive_index("N_BK7", LAMBDA_D)
        n_sf = refractive_index("N_SF11", LAMBDA_D)
        f_bk = lensmaker_f(R1, R2, n_bk, t)
        f_sf = lensmaker_f(R1, R2, n_sf, t)
        self.assertLess(abs(f_sf), abs(f_bk))


# ═══════════════════════════════════════════════════════════════════════════
# Snell's law & Fresnel equations
# ═══════════════════════════════════════════════════════════════════════════

class TestSnellLawExact(unittest.TestCase):
    def _refract_angle(self, th1_deg: float, n1: float, n2: float):
        th1 = math.radians(th1_deg)
        I = (math.sin(th1), 0.0, math.cos(th1))
        T, tir = snell_refract(I, (0.0, 0.0, 1.0), n1, n2)
        if tir:
            return None, True, T
        # angle from +Z (surface normal after snell may flip; T is unit dir)
        th2 = math.atan2(math.hypot(T[0], T[1]), abs(T[2]))
        return th2, False, T

    def test_snell_law_table(self):
        for th1_deg in (0, 5, 15, 30, 45, 60):
            for n1, n2 in ((1.0, 1.5), (1.0, 1.7), (1.5, 1.0)):
                th_c = math.degrees(math.asin(min(1.0, n2 / n1))) if n1 > n2 else 90.0
                if th1_deg >= th_c - 0.01:
                    continue
                th2, tir, _ = self._refract_angle(th1_deg, n1, n2)
                self.assertFalse(tir, msg=f"TIR unexpected @ {th1_deg} n={n1}->{n2}")
                expect = math.asin(n1 / n2 * math.sin(math.radians(th1_deg)))
                self.assertAlmostEqual(th2, expect, places=8)

    def test_critical_angle_boundary(self):
        n1, n2 = 1.5, 1.0
        th_c = math.degrees(math.asin(n2 / n1))
        # 0.1° below → transmit
        _, tir, _ = self._refract_angle(th_c - 0.1, n1, n2)
        self.assertFalse(tir)
        # 0.1° above → TIR
        _, tir, R = self._refract_angle(th_c + 0.1, n1, n2)
        self.assertTrue(tir)
        th = math.radians(th_c + 0.1)
        I = (math.sin(th), 0.0, math.cos(th))
        # Specular reflection: R_x = I_x, R_z = −I_z for N = +Z
        self.assertAlmostEqual(R[0], I[0], places=6)
        self.assertAlmostEqual(R[2], -I[2], places=6)

    def test_direction_unit_length(self):
        for th in (0, 20, 40, 55):
            th1 = math.radians(th)
            I = (math.sin(th1), 0.0, math.cos(th1))
            T, tir = snell_refract(I, (0, 0, 1), 1.0, 1.6)
            self.assertFalse(tir)
            self.assertAlmostEqual(math.hypot(*T), 1.0, places=10)
            T2, tir2 = snell_refract(I, (0, 0, 1), 1.6, 1.0)
            if not tir2:
                self.assertAlmostEqual(math.hypot(*T2), 1.0, places=10)

    def test_normal_flip_invariant(self):
        """Snell must give same transmitted ray if geometric normal is reversed."""
        th1 = math.radians(25)
        I = (math.sin(th1), 0.0, math.cos(th1))
        T_pos, t1 = snell_refract(I, (0, 0, 1), 1.0, 1.5)
        T_neg, t2 = snell_refract(I, (0, 0, -1), 1.0, 1.5)
        self.assertFalse(t1 or t2)
        self.assertAlmostEqual(T_pos[0], T_neg[0], places=9)
        self.assertAlmostEqual(T_pos[2], T_neg[2], places=9)

    def test_round_trip_air_glass_air(self):
        """Air→glass→air recovers the original direction (no TIR)."""
        th1 = math.radians(28)
        I = (math.sin(th1), 0.0, math.cos(th1))
        Tg, tir = snell_refract(I, (0, 0, 1), 1.0, 1.52)
        self.assertFalse(tir)
        Ta, tir2 = snell_refract(Tg, (0, 0, 1), 1.52, 1.0)
        self.assertFalse(tir2)
        self.assertAlmostEqual(Ta[0], I[0], places=7)
        self.assertAlmostEqual(Ta[2], I[2], places=7)


class TestFresnelExact(unittest.TestCase):
    def test_normal_incidence_formula(self):
        for n2 in (1.3, 1.5, 1.7, 2.0):
            T, tir = fresnel_T(1.0, n2, 1.0)
            self.assertFalse(tir)
            R = ((1.0 - n2) / (1.0 + n2)) ** 2
            self.assertAlmostEqual(T, 1.0 - R, places=12)

    def test_reciprocal_normal_interface(self):
        """Power reflectance at normal incidence is the same air↔glass."""
        T12, _ = fresnel_T(1.0, 1.5, 1.0)
        T21, _ = fresnel_T(1.5, 1.0, 1.0)
        self.assertAlmostEqual(T12, T21, places=12)

    def test_fresnel_vs_angle_monotone(self):
        """Unpolarized transmittance air→glass decreases as incidence rises (0–60°)."""
        prev = 1.0
        for th in (0, 15, 30, 45, 60):
            T, tir = fresnel_T(1.0, 1.5, math.cos(math.radians(th)))
            self.assertFalse(tir)
            self.assertLessEqual(T, prev + 1e-12)
            prev = T

    def test_tir_returns_zero(self):
        th_c = math.asin(1.0 / 1.5)
        T, tir = fresnel_T(1.5, 1.0, math.cos(th_c + 0.05))
        self.assertTrue(tir)
        self.assertEqual(T, 0.0)

    def test_two_surface_window_power(self):
        """BK7 window: T_total ≈ T_air→glass × T_glass→air at normal incidence."""
        wl = 550.0
        n_air = refractive_index("AIR", wl)
        n = refractive_index("N_BK7", wl)
        T_in, _ = fresnel_T(n_air, n, 1.0)
        T_out, _ = fresnel_T(n, n_air, 1.0)
        expect = T_in * T_out
        surfs = build_surfaces(
            [{
                "enabled": True, "R1": 0, "R2": 0, "thickness": 5,
                "air_after": 1, "aperture": 30, "material": "N_BK7",
                "k1": 0, "k2": 0, "A4_1": 0, "A4_2": 0,
                "surface_mode": "rotational",
            }],
            z_start=2.0,
        )
        ok, _, p, _ = trace_ray(
            (0, 0, 0), (0, 0, 1), 1.0, wl, surfs, 40.0, apply_fresnel=True
        )
        self.assertTrue(ok)
        self.assertAlmostEqual(p, expect, places=6)


# ═══════════════════════════════════════════════════════════════════════════
# Surface geometry: sag, normals, intersection
# ═══════════════════════════════════════════════════════════════════════════

class TestSphereGeometryAnalytic(unittest.TestCase):
    def test_sphere_sag_identity(self):
        for R in (20.0, 50.0, 100.0, -30.0):
            s = OpticalSurface(z_vertex=0.0, radius=R, aperture=min(abs(R) * 0.4, 12))
            for r in (0.0, 1.0, 3.0, 5.0):
                if r >= abs(R):
                    continue
                # Standard asphere k=0: sag = c r² / (1 + sqrt(1 − c² r²))
                # ≡ sign(R) * (|R| − sqrt(R² − r²))
                c = 1.0 / R
                disc = 1.0 - c * c * r * r
                if disc < 0:
                    continue
                expect = (c * r * r) / (1.0 + math.sqrt(disc))
                self.assertAlmostEqual(s.sag(r), expect, places=10)

    def test_sphere_normal_matches_analytic(self):
        """For R>0 vertex at 0, centre at (0,0,R); N = normalize(P−C) flipped to +Nz."""
        R = 50.0
        s = OpticalSurface(z_vertex=0.0, radius=R, aperture=20)
        for x, y in ((0, 0), (3, 0), (0, 5), (4, 3), (-2, 1)):
            z = s.surface_z(x, y)
            self.assertIsNotNone(z)
            # Centre of sphere
            cx, cy, cz = 0.0, 0.0, R
            # Outward from sphere body (toward free space for first surface)
            vx, vy, vz = x - cx, y - cy, z - cz
            L = math.hypot(vx, vy, vz)
            Nx, Ny, Nz = vx / L, vy / L, vz / L
            if Nz < 0:
                Nx, Ny, Nz = -Nx, -Ny, -Nz
            n = s.normal_at(x, y)
            self.assertIsNotNone(n)
            self.assertAlmostEqual(n[0], Nx, places=5)
            self.assertAlmostEqual(n[1], Ny, places=5)
            self.assertAlmostEqual(n[2], Nz, places=5)

    def test_sphere_intersection_closed_form(self):
        R = 50.0
        zv = 10.0
        s = OpticalSurface(z_vertex=zv, radius=R, aperture=25)
        y0 = 5.0
        hit = s.intersect((0, y0, 0), (0, 0, 1))
        self.assertIsNotNone(hit)
        _, p, _ = hit
        # Analytic: centre at z = zv+R; ray x=0,y=y0; (y0)²+(z−c)²=R²
        c = zv + R
        z_exact = c - math.sqrt(R * R - y0 * y0)
        self.assertAlmostEqual(p[1], y0, places=7)
        self.assertAlmostEqual(p[2], z_exact, places=6)

    def test_parabola_conic(self):
        R = 40.0
        s = OpticalSurface(z_vertex=0, radius=R, k=-1.0, aperture=15)
        for r in (0, 2, 6, 10):
            self.assertAlmostEqual(s.sag(r), r * r / (2 * R), places=9)

    def test_a4_asphere_term(self):
        a4 = 2e-5
        s = OpticalSurface(z_vertex=0, radius=0.0, a4=a4, aperture=12)
        r = 4.0
        self.assertAlmostEqual(s.sag(r), a4 * r ** 4, places=12)


# ═══════════════════════════════════════════════════════════════════════════
# Thick lens, plate shift, focusing
# ═══════════════════════════════════════════════════════════════════════════

class TestThickLensAndPlate(unittest.TestCase):
    def test_lensmaker_matches_mla_helper(self):
        for R1, R2, n, t in (
            (40, -40, 1.5, 3),
            (0, -25, 1.49, 4),
            (30, 0, 1.6, 2),
            (-20, 20, 1.5, 5),  # biconcave
        ):
            f1 = lensmaker_f(R1, R2, n, t)
            f2 = thin_lens_focal_length_mm(R1, R2, n, t)
            if math.isfinite(f1) and abs(f1) < 1e5:
                self.assertAlmostEqual(f1, f2, places=6)

    def test_thin_equiconvex_limit(self):
        R, n = 50.0, 1.5
        f_thin = R / (2 * (n - 1))  # 50 mm
        f = lensmaker_f(R, -R, n, thickness=0.01)
        self.assertAlmostEqual(f, f_thin, delta=0.05)

    def test_paraxial_focus_location(self):
        """Parallel ray at small height crosses axis near thick-lens focus."""
        n = 1.5
        R = 40.0
        t = 2.0
        f = lensmaker_f(R, -R, n, t)
        el = {
            "enabled": True, "R1": R, "R2": -R, "thickness": t,
            "air_after": 1, "aperture": 15, "material": "CUSTOM",
            "k1": 0, "k2": 0, "A4_1": 0, "A4_2": 0, "surface_mode": "rotational",
        }
        z0 = 5.0
        surfs = build_surfaces([el], z_start=z0)
        h = 1.5  # paraxial height
        ok, _, _, path = trace_ray(
            (0, h, 0), (0, 0, 1), 1.0, 587.6, surfs, target_z=200.0,
            custom_n=n, apply_fresnel=False, store_path=True,
        )
        self.assertTrue(ok)
        p_exit, p_tgt = path.history[-2], path.history[-1]
        dy = p_tgt[1] - p_exit[1]
        dz = p_tgt[2] - p_exit[2]
        self.assertNotAlmostEqual(dy, 0.0, places=6)
        t_cross = -p_exit[1] / dy
        z_cross = p_exit[2] + t_cross * dz
        z_mid = 0.5 * (surfs[0].z_vertex + surfs[1].z_vertex)
        # Thick-lens EFL is measured from principal planes ≈ near mid for thin;
        # allow a few mm for principal-plane shift + spherical aberration at h=1.5
        self.assertAlmostEqual(z_cross, z_mid + f, delta=2.5)

    def test_diverging_lens_bends_away(self):
        el = {
            "enabled": True, "R1": -40, "R2": 40, "thickness": 2,
            "air_after": 1, "aperture": 15, "material": "CUSTOM",
            "k1": 0, "k2": 0, "A4_1": 0, "A4_2": 0, "surface_mode": "rotational",
        }
        surfs = build_surfaces([el], z_start=5.0)
        ok, _, _, path = trace_ray(
            (0, 2, 0), (0, 0, 1), 1.0, 550, surfs, 80,
            custom_n=1.5, apply_fresnel=False, store_path=True,
        )
        self.assertTrue(ok)
        p0, p1 = path.history[-2], path.history[-1]
        dy_dz = (p1[1] - p0[1]) / (p1[2] - p0[2])
        self.assertGreater(dy_dz, 0.0, "negative lens must bend +y ray away from axis")

    def test_plano_plate_lateral_shift(self):
        """
        Constant-z lateral offset of a plate:
            Δx = t (tan θ1 − tan θ2),  n1 sin θ1 = n2 sin θ2
        Exit ray must stay parallel to the incident ray.
        Uses catalog air (n≈1.000293) so Snell matches the tracer media.
        """
        th1 = math.radians(30)
        wl = 550.0
        n1 = refractive_index("AIR", wl)
        n2 = 1.5
        th2 = math.asin(n1 / n2 * math.sin(th1))
        t_plate = 10.0
        dx_expect = t_plate * (math.tan(th1) - math.tan(th2))
        el = {
            "enabled": True, "R1": 0, "R2": 0, "thickness": t_plate,
            "air_after": 1, "aperture": 50, "material": "CUSTOM",
            "k1": 0, "k2": 0, "A4_1": 0, "A4_2": 0, "surface_mode": "rotational",
        }
        z_start = 5.0
        surfs = build_surfaces([el], z_start=z_start)
        I = (math.sin(th1), 0.0, math.cos(th1))
        ok, _, _, path = trace_ray(
            (0, 0, 0), I, 1.0, wl, surfs, 50,
            custom_n=n2, apply_fresnel=False, store_path=True,
        )
        self.assertTrue(ok)
        exit_p = path.history[-2]
        d_out = v_norm((
            path.history[-1][0] - exit_p[0],
            path.history[-1][1] - exit_p[1],
            path.history[-1][2] - exit_p[2],
        ))
        # Parallel exit
        self.assertAlmostEqual(d_out[0], I[0], places=6)
        self.assertAlmostEqual(d_out[2], I[2], places=6)
        z_exit = surfs[1].z_vertex
        x_no_plate = z_exit * math.tan(th1)
        dx = x_no_plate - exit_p[0]
        # Sub-mm agreement (Newton intersection + 1e-4 post-hit epsilon)
        self.assertAlmostEqual(dx, dx_expect, delta=0.005)

    def test_oblique_plate_no_angular_deviation(self):
        """Plate cannot change ray angle — only shift laterally."""
        th1 = math.radians(40)
        I = (math.sin(th1), 0.0, math.cos(th1))
        el = {
            "enabled": True, "R1": 0, "R2": 0, "thickness": 8,
            "air_after": 1, "aperture": 40, "material": "N_BK7",
            "k1": 0, "k2": 0, "A4_1": 0, "A4_2": 0, "surface_mode": "rotational",
        }
        surfs = build_surfaces([el], z_start=3.0)
        ok, _, _, path = trace_ray(
            (0, 0, 0), I, 1.0, 550, surfs, 60, apply_fresnel=False, store_path=True,
        )
        self.assertTrue(ok)
        exit_p = path.history[-2]
        d_out = v_norm((
            path.history[-1][0] - exit_p[0],
            path.history[-1][1] - exit_p[1],
            path.history[-1][2] - exit_p[2],
        ))
        self.assertAlmostEqual(d_out[0], I[0], places=5)
        self.assertAlmostEqual(d_out[2], I[2], places=5)


# ═══════════════════════════════════════════════════════════════════════════
# Cylinders / anamorphic power
# ═══════════════════════════════════════════════════════════════════════════

class TestCylinderMeridians(unittest.TestCase):
    def test_cylinder_x_focuses_only_x(self):
        """Cylinder_x: parallel ray offset in X bends toward axis; Y-offset does not."""
        el = {
            "enabled": True, "R1": 30, "R2": -30, "thickness": 3,
            "air_after": 1, "aperture": 12, "material": "CUSTOM",
            "k1": 0, "k2": 0, "A4_1": 0, "A4_2": 0,
            "surface_mode": "cylinder_x", "mode_s1": "cylinder_x", "mode_s2": "cylinder_x",
        }
        surfs = build_surfaces([el], z_start=5.0)
        # X offset → should acquire dx/dz < 0
        ok, _, _, path = trace_ray(
            (2, 0, 0), (0, 0, 1), 1.0, 550, surfs, 80,
            custom_n=1.5, apply_fresnel=False, store_path=True,
        )
        self.assertTrue(ok)
        p0, p1 = path.history[-2], path.history[-1]
        dx_dz = (p1[0] - p0[0]) / (p1[2] - p0[2])
        self.assertLess(dx_dz, 0.0)
        # Y offset → no power in Y
        ok2, _, _, path2 = trace_ray(
            (0, 2, 0), (0, 0, 1), 1.0, 550, surfs, 80,
            custom_n=1.5, apply_fresnel=False, store_path=True,
        )
        self.assertTrue(ok2)
        q0, q1 = path2.history[-2], path2.history[-1]
        dy_dz = (q1[1] - q0[1]) / (q1[2] - q0[2])
        self.assertAlmostEqual(dy_dz, 0.0, places=5)
        self.assertAlmostEqual(q1[1], 2.0, places=4)

    def test_cylinder_y_focuses_only_y(self):
        el = {
            "enabled": True, "R1": 30, "R2": -30, "thickness": 3,
            "air_after": 1, "aperture": 12, "material": "CUSTOM",
            "k1": 0, "k2": 0, "A4_1": 0, "A4_2": 0,
            "surface_mode": "cylinder_y", "mode_s1": "cylinder_y", "mode_s2": "cylinder_y",
        }
        surfs = build_surfaces([el], z_start=5.0)
        ok, _, _, path = trace_ray(
            (0, 2, 0), (0, 0, 1), 1.0, 550, surfs, 80,
            custom_n=1.5, apply_fresnel=False, store_path=True,
        )
        self.assertTrue(ok)
        p0, p1 = path.history[-2], path.history[-1]
        dy_dz = (p1[1] - p0[1]) / (p1[2] - p0[2])
        self.assertLess(dy_dz, 0.0)
        ok2, _, _, path2 = trace_ray(
            (2, 0, 0), (0, 0, 1), 1.0, 550, surfs, 80,
            custom_n=1.5, apply_fresnel=False, store_path=True,
        )
        self.assertTrue(ok2)
        q0, q1 = path2.history[-2], path2.history[-1]
        dx_dz = (q1[0] - q0[0]) / (q1[2] - q0[2])
        self.assertAlmostEqual(dx_dz, 0.0, places=5)


# ═══════════════════════════════════════════════════════════════════════════
# Dispersion → chromatic focus
# ═══════════════════════════════════════════════════════════════════════════

class TestChromaticEffects(unittest.TestCase):
    def test_blue_focuses_harder_than_red(self):
        """Normal dispersion: n_F > n_C → shorter EFL at blue for positive lens."""
        R1, R2, t = 30.0, -30.0, 4.0
        f_blue = lensmaker_f(R1, R2, refractive_index("N_BK7", LAMBDA_F), t)
        f_red = lensmaker_f(R1, R2, refractive_index("N_BK7", LAMBDA_C), t)
        self.assertLess(f_blue, f_red)
        self.assertGreater(f_blue, 0)
        self.assertGreater(f_red, 0)

    def test_traced_blue_bends_more(self):
        el = {
            "enabled": True, "R1": 25, "R2": -25, "thickness": 4,
            "air_after": 1, "aperture": 12, "material": "N_BK7",
            "k1": 0, "k2": 0, "A4_1": 0, "A4_2": 0, "surface_mode": "rotational",
        }
        surfs = build_surfaces([el], z_start=5.0)
        h = 3.0

        def exit_slope(wl):
            ok, _, _, path = trace_ray(
                (0, h, 0), (0, 0, 1), 1.0, wl, surfs, 100,
                apply_fresnel=False, store_path=True,
            )
            self.assertTrue(ok)
            p0, p1 = path.history[-2], path.history[-1]
            return (p1[1] - p0[1]) / (p1[2] - p0[2])

        # More negative slope = stronger focus toward axis
        self.assertLess(exit_slope(LAMBDA_F), exit_slope(LAMBDA_C))


# ═══════════════════════════════════════════════════════════════════════════
# Emission sampling & energy bookkeeping
# ═══════════════════════════════════════════════════════════════════════════

class TestEmissionAndEnergy(unittest.TestCase):
    def test_lambertian_pdf_mean_cos(self):
        """Full hemisphere Lambertian: ⟨cos θ⟩ = 2/3."""
        N = 12000
        mean = sum(sample_lambertian_cone(90.0)[2] for _ in range(N)) / N
        self.assertAlmostEqual(mean, 2.0 / 3.0, delta=0.02)

    def test_cone_hard_cutoff(self):
        half = 45.0
        cos_min = math.cos(math.radians(half))
        for _ in range(2000):
            d = sample_lambertian_cone(half)
            self.assertGreaterEqual(d[2], cos_min - 1e-9)
            self.assertAlmostEqual(math.hypot(*d), 1.0, places=9)

    def test_power_never_exceeds_launch(self):
        p = default_params()
        p["total_rays"] = 400
        p["apply_fresnel"] = True
        p["source"]["mode"] = "single"
        r = run_simulation(p)
        # Map + missed ≤ source (numerical slack for float)
        self.assertLessEqual(
            r.stats["map_power"] + r.stats.get("missed_power", 0.0),
            r.stats["source_power"] * 1.001,
        )

    def test_no_fresnel_preserves_unit_power_window(self):
        surfs = build_surfaces(
            [{
                "enabled": True, "R1": 0, "R2": 0, "thickness": 6,
                "air_after": 1, "aperture": 20, "material": "N_BK7",
                "k1": 0, "k2": 0, "A4_1": 0, "A4_2": 0, "surface_mode": "rotational",
            }],
            z_start=1.0,
        )
        ok, _, pwr, _ = trace_ray(
            (0, 0, 0), (0, 0, 1), 1.0, 550, surfs, 40, apply_fresnel=False
        )
        self.assertTrue(ok)
        self.assertAlmostEqual(pwr, 1.0, places=10)

    def test_collection_in_unit_interval(self):
        p = default_params()
        p["total_rays"] = 1500
        p["source"]["mode"] = "single"
        r = run_simulation(p)
        c = r.stats["collection"]
        self.assertGreaterEqual(c, 0.0)
        self.assertLessEqual(c, 1.0 + 1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# Vector helpers used by the tracer
# ═══════════════════════════════════════════════════════════════════════════

class TestVectorHelpers(unittest.TestCase):
    def test_norm_unit(self):
        v = v_norm((3, 0, 4))
        self.assertAlmostEqual(v[0], 0.6, places=10)
        self.assertAlmostEqual(v[2], 0.8, places=10)

    def test_dot_orthogonality(self):
        self.assertAlmostEqual(v_dot((1, 0, 0), (0, 1, 0)), 0.0, places=12)
        self.assertAlmostEqual(v_dot((1, 2, 3), (1, 2, 3)), 14.0, places=12)

    def test_zero_vector_norm(self):
        self.assertEqual(v_norm((0, 0, 0)), (0.0, 0.0, 0.0))


def run_all():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(run_all())
