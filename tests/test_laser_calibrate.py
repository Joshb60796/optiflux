"""Laser-pointer resin-index calibration (collimated pencil through the stack)."""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import default_params, lensmaker_f
from calibrate import (
    apply_calibrated_n,
    calibrate_n_from_laser,
    laser_calibration_guide,
    laser_spot,
)


def _singlet_params(*, n: float = 1.54, R: float = 40.0, t: float = 6.0, z0: float = 5.0):
    p = default_params()
    p["use_warp"] = False
    p["mla"]["enabled"] = False
    p["custom_n"] = n
    p["lens_z_start"] = z0
    p["source"]["source_z"] = 0.0
    p["source"]["wavelength_nm"] = 650.0
    p["elements"][0].update(
        enabled=True,
        R1=R,
        R2=-R,
        thickness=t,
        air_after=2.0,
        aperture=15.0,
        aperture_y=None,
        circular_lock=True,
        material="CUSTOM",
        surface_mode="rotational",
        k1=0.0,
        k2=0.0,
        A4_1=0.0,
        A4_2=0.0,
    )
    for e in p["elements"][1:]:
        e["enabled"] = False
    return p


class TestLaserSpot(unittest.TestCase):
    def test_on_axis_laser_stays_on_axis(self):
        p = _singlet_params()
        hit = laser_spot(p, laser_x=0.0, laser_y=0.0, laser_z=0.0, screen_z=80.0, n=1.54)
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit[0], 0.0, places=5)
        self.assertAlmostEqual(hit[1], 0.0, places=5)

    def test_higher_n_bends_more_toward_axis(self):
        p = _singlet_params()
        lo = laser_spot(p, laser_x=0.0, laser_y=4.0, laser_z=0.0, screen_z=40.0, n=1.48)
        hi = laser_spot(p, laser_x=0.0, laser_y=4.0, laser_z=0.0, screen_z=40.0, n=1.60)
        self.assertIsNotNone(lo)
        self.assertIsNotNone(hi)
        # Before focus, both still +Y; higher n is closer to the axis
        self.assertGreater(lo[1], 0.0)
        self.assertGreater(hi[1], 0.0)
        self.assertLess(hi[1], lo[1])


class TestCalibrateNFromLaser(unittest.TestCase):
    def test_recovers_n_from_forward_spot(self):
        truth = 1.547
        p = _singlet_params(n=truth)
        hx, hy = 0.0, 5.0
        screen = 55.0
        hit = laser_spot(p, laser_x=hx, laser_y=hy, laser_z=0.0, screen_z=screen, n=truth)
        self.assertIsNotNone(hit)
        fit = calibrate_n_from_laser(
            p,
            laser_x=hx,
            laser_y=hy,
            laser_z=0.0,
            screen_z=screen,
            spot_x=hit[0],
            spot_y=hit[1],
            wavelength_nm=650.0,
        )
        self.assertTrue(fit["ok"])
        self.assertAlmostEqual(fit["n"], truth, delta=0.004)
        self.assertLess(fit["residual_mm"], 0.05)

    def test_coaxial_offset_is_subtracted(self):
        truth = 1.54
        p = _singlet_params(n=truth)
        hit = laser_spot(p, 0.0, 4.0, 0.0, 50.0, n=truth)
        # Bench origin shifted by +0.3 mm in Y
        fit = calibrate_n_from_laser(
            p,
            laser_x=0.0,
            laser_y=4.0,
            laser_z=0.0,
            screen_z=50.0,
            spot_x=hit[0],
            spot_y=hit[1] + 0.3,
            coaxial_spot_x=0.0,
            coaxial_spot_y=0.3,
            wavelength_nm=650.0,
        )
        self.assertTrue(fit["ok"])
        self.assertAlmostEqual(fit["n"], truth, delta=0.005)

    def test_thin_lens_limit_matches_lensmaker_direction(self):
        """Recovered n should imply an EFL near the thick-lens formula."""
        n = 1.54
        p = _singlet_params(n=n, R=50.0, t=2.0)
        hit = laser_spot(p, 0.0, 3.0, 0.0, 45.0, n=n)
        fit = calibrate_n_from_laser(
            p,
            laser_x=0.0,
            laser_y=3.0,
            laser_z=0.0,
            screen_z=45.0,
            spot_x=hit[0],
            spot_y=hit[1],
            wavelength_nm=650.0,
        )
        f_lm = lensmaker_f(50.0, -50.0, fit["n"], 2.0)
        f_true = lensmaker_f(50.0, -50.0, n, 2.0)
        self.assertAlmostEqual(f_lm, f_true, delta=1.0)


class TestLaserCalibrationGuide(unittest.TestCase):
    def test_guide_is_a_full_bench_procedure(self):
        text = laser_calibration_guide().lower()
        self.assertGreater(len(text), 800)
        for needle in (
            "what you need",
            "step 1",
            "step 2",
            "step 3",
            "coaxial",
            "offset",
            "wavelength",
            "apply",
            "do not tilt",
        ):
            self.assertIn(needle, text)

    def test_help_file_includes_the_guide(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        help_txt = os.path.join(root, "HELP.txt")
        with open(help_txt, encoding="utf-8") as fh:
            body = fh.read()
        self.assertIn("LASER n CALIBRATION", body)
        self.assertIn("Step 1", body)
        self.assertIn("Do not tilt", body)


class TestApplyCalibratedN(unittest.TestCase):
    def test_sets_custom_n_and_switches_resin(self):
        p = default_params()
        p["elements"][0]["material"] = "FORMLABS_CLEAR"
        p["elements"][0]["enabled"] = True
        p["elements"][1]["material"] = "N_BK7"
        p["elements"][1]["enabled"] = True
        out = apply_calibrated_n(p, 1.552)
        self.assertAlmostEqual(out["custom_n"], 1.552)
        self.assertEqual(out["elements"][0]["material"], "CUSTOM")
        self.assertEqual(out["elements"][1]["material"], "N_BK7")


if __name__ == "__main__":
    unittest.main()
