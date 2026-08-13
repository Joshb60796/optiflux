"""
Calibrate printed-resin refractive index from a laser-pointer pencil.

A collimated (or nearly collimated) pointer along +Z at a known height
through the current stack is traced with the same Snell engine as the
simulator. Sweep n until the predicted screen spot matches the measured
dot. A coaxial (on-axis) reading is subtracted so a shifted bench origin
does not bias n.

Typical red pointer ≈ 650 nm; green ≈ 532 nm; HeNe = 632.8 nm.
"""
from __future__ import annotations

LASER_CALIBRATION_GUIDE = """\
LASER n CALIBRATION — printed Formlabs / custom resin
=====================================================

This measures the refractive index of a lens you already printed, using a
laser pointer and a card. OptiFlux traces the same +Z pencil the simulator
uses (Snell at every surface) and finds the n that puts the spot where you
saw it. Catalog Formlabs n is only approximate; this is how you replace it
with your resin, cure cycle, and polish.

The coaxial (centered) shot does not determine n. It only sets the origin
on the card. n comes from how far the dot moves when you shift the laser
sideways without tilting it.


What you need
-------------
  • The printed lens (or group) whose radii and thicknesses match the
    design currently loaded in OptiFlux. Measure center thickness with
    a caliper if you can, and type the real value into the element.
  • A laser pointer. Red cheap pointers are about 650 nm; green are
    about 532 nm; a HeNe lab laser is 632.8 nm. Enter the wavelength
    you actually use.
  • A flat card or paper screen you can mark, a ruler or calipers, and
    a way to hold the pointer parallel to the optical axis (+Z).
  • Eye safety: never look into the beam or a specular reflection off
    the lens.


Coordinate reminder
-------------------
  Light travels +Z (source toward the target). X is horizontal, Y is
  vertical, same as the Target Plane view. All entries are millimetres.
  Laser Z is where the pointer sits (usually the source plane, 0 mm).
  Screen / card Z is the card’s position on the same axis (often you
  can use Target Z if the card is at the design throw).


Setup
-----
  1. Load the design you printed (radii, thickness, air gaps, first
     vertex Z). Wrong geometry will be absorbed into a wrong n.
  2. Hold the lens as in the design: first vertex facing the laser,
     axis along +Z. A V-block or the printed tube helps.
  3. Place the card square to the axis at a known Z. Prefer a throw
     that is not right at the focus — the dot should still move when
     you change height. Farther is usually easier to measure.
  4. Keep the pointer parallel to +Z for every shot. Do not tilt it
     to “aim” at the card. Only translate it in X and/or Y.


Step 1 — coaxial (alignment)
----------------------------
  Aim the beam through the center of the lens, still +Z. Mark the dot
  on the card.

  If you define the card origin as that mark, enter Coaxial dot X = 0
  and Coaxial dot Y = 0.

  If you already drew axes on the card, measure the coaxial dot from
  those axes and enter those X and Y. The fit subtracts this reading
  so a shifted origin does not bias n.


Step 2 — shift the laser
------------------------
  Move the pointer sideways by a known amount (for example +5 mm in Y,
  or a bit of both X and Y). Do not rotate or tilt.

  Enter that translation as Laser offset X and Laser offset Y.

  Stay inside the clear aperture, but not on the rim (rim rays have
  more spherical aberration and will fake a different n).


Step 3 — mark the shifted dot
-----------------------------
  Mark the new dot on the same card. Measure it in the same axes as
  the coaxial mark. Enter Shifted dot X and Shifted dot Y.

  Also confirm:
    Laser Z     = pointer position (mm)
    Screen Z    = card position (mm)
    Wavelength  = your laser (nm)


Step 4 — Fit n
--------------
  Press “Fit n from laser dots”. OptiFlux traces a +Z ray at your
  offset through the current stack and searches n until the predicted
  (shifted − coaxial) spot matches what you typed.

  A good fit reports n and a small residual (tenths of a millimetre
  if the geometry is right). Element 1’s lensmaker EFL at that n is
  shown as a sanity check.


Step 5 — Apply
--------------
  Press “Apply fitted n to Custom / Formlabs”. That writes Custom n
  and switches enabled Formlabs (and Custom) elements to Custom so
  the irradiance map uses your measured index. Catalog glass (N-BK7
  and similar) is left alone.


Field list
----------
  Laser wavelength (nm)  Colour of the pointer. n is reported at this λ.
  Laser Z (mm)           Pointer position along +Z. Usually 0.
  Screen / card Z (mm)   Card position along +Z. Same origin as the design.
  Laser offset X, Y      How far you translated the pointer from the axis.
  Coaxial dot X, Y       Where the centered beam landed on the card.
  Shifted dot X, Y       Where the translated beam landed on the card.


If the fit looks wrong
----------------------
  • Residual of several millimetres: check that Screen Z is the card,
    not a guessed focus, and that offsets and dots use the same axes
    and the same millimetre origin.
  • n far from 1.50–1.58 for Clear resin: thickness or radii in the
    file probably do not match the print. Measure thickness.
  • On-axis only: a centered beam does not constrain n. You must shift.
  • Tilted pointer: the model assumes a ray parallel to +Z. Tilting
    is a different measurement and will not fit this n.
  • Card at the focus: the dot barely moves with n. Move the card.

This fit is one number at one wavelength. It does not include scatter,
layer lines, polish haze, or the finite width of the beam. Use it to
get the map’s n close to the part you printed.
"""


def laser_calibration_guide() -> str:
    """User-facing bench procedure for the laser n fit."""
    return LASER_CALIBRATION_GUIDE


import copy
import math
from typing import Any, Dict, Optional, Tuple

from engine import assemble_surfaces, build_source_array, lensmaker_f, trace_ray
from materials_catalog import material_id_from_name

# Photopolymers whose catalog n is approximate — replace with Custom after a fit.
_RESIN_IDS = {
    "FORMLABS_CLEAR",
    "FORMLABS_TOUGH",
    "FORMLABS_HIGH_TEMP",
    "FORMLABS_RIGID",
    "CUSTOM",
}


def _force_custom_n(params: Dict[str, Any], n: float, wavelength_nm: float) -> Dict[str, Any]:
    p = copy.deepcopy(params)
    p["custom_n"] = float(n)
    src = dict(p.get("source") or {})
    src["wavelength_nm"] = float(wavelength_nm)
    p["source"] = src
    for e in p.get("elements") or []:
        if not e.get("enabled", True):
            continue
        mid = material_id_from_name(str(e.get("material", "CUSTOM")))
        if mid in _RESIN_IDS:
            e["material"] = "CUSTOM"
    return p


def laser_spot(
    params: Dict[str, Any],
    laser_x: float,
    laser_y: float,
    laser_z: float,
    screen_z: float,
    n: float,
    wavelength_nm: float = 650.0,
) -> Optional[Tuple[float, float]]:
    """
    Trace one +Z pencil at (laser_x, laser_y, laser_z) through the stack.

    Returns (x, y) on the plane z = screen_z, or None if the ray misses.
    """
    p = _force_custom_n(params, n, wavelength_nm)
    dies = build_source_array(p.get("source") or {})
    mla = dict(p.get("mla") or {})
    mla["_target_z"] = float(screen_z)
    mla["_fov_cx"] = float(p.get("fov_cx", 0.0))
    mla["_fov_cy"] = float(p.get("fov_cy", 0.0))
    surfaces = assemble_surfaces(
        p.get("elements") or [],
        float(p.get("lens_z_start", 3.0)),
        mla=mla if mla.get("enabled") else None,
        dies=dies if mla.get("enabled") else None,
        blockers=p.get("blockers"),
    )
    ok, pt, _pwr, _path = trace_ray(
        (float(laser_x), float(laser_y), float(laser_z)),
        (0.0, 0.0, 1.0),
        1.0,
        float(wavelength_nm),
        surfaces,
        float(screen_z),
        custom_n=float(n),
        apply_fresnel=False,
        absorb_on_tir=True,
        store_path=False,
        kill_backward=True,
    )
    if not ok or pt is None:
        return None
    return float(pt[0]), float(pt[1])


def _rel_spot(
    params: Dict[str, Any],
    *,
    laser_x: float,
    laser_y: float,
    laser_z: float,
    screen_z: float,
    n: float,
    wavelength_nm: float,
) -> Optional[Tuple[float, float]]:
    off = laser_spot(params, laser_x, laser_y, laser_z, screen_z, n, wavelength_nm)
    on = laser_spot(params, 0.0, 0.0, laser_z, screen_z, n, wavelength_nm)
    if off is None or on is None:
        return None
    return off[0] - on[0], off[1] - on[1]


def calibrate_n_from_laser(
    params: Dict[str, Any],
    *,
    laser_x: float,
    laser_y: float,
    laser_z: float,
    screen_z: float,
    spot_x: float,
    spot_y: float,
    coaxial_spot_x: float = 0.0,
    coaxial_spot_y: float = 0.0,
    wavelength_nm: float = 650.0,
    n_lo: float = 1.30,
    n_hi: float = 1.75,
    samples: int = 36,
) -> Dict[str, Any]:
    """
    Fit a single refractive index so the traced pencil matches the measured
    (shifted − coaxial) spot on the screen.
    """
    meas = (
        float(spot_x) - float(coaxial_spot_x),
        float(spot_y) - float(coaxial_spot_y),
    )
    if math.hypot(laser_x, laser_y) < 1e-6:
        return {
            "ok": False,
            "n": float("nan"),
            "residual_mm": float("nan"),
            "predicted_xy": (0.0, 0.0),
            "efl_mm": float("nan"),
            "message": "Shift the laser in X and/or Y — an on-axis pencil does not constrain n.",
        }

    def err_at(nv: float) -> Tuple[float, Optional[Tuple[float, float]]]:
        pred = _rel_spot(
            params,
            laser_x=laser_x,
            laser_y=laser_y,
            laser_z=laser_z,
            screen_z=screen_z,
            n=nv,
            wavelength_nm=wavelength_nm,
        )
        if pred is None:
            return 1e9, None
        return math.hypot(pred[0] - meas[0], pred[1] - meas[1]), pred

    best_n = 0.5 * (n_lo + n_hi)
    best_e = 1e9
    best_pred: Optional[Tuple[float, float]] = None
    lo, hi = float(n_lo), float(n_hi)
    # Dense sample then shrink around the minimum (err is unimodal for a singlet).
    for _pass in range(3):
        step = (hi - lo) / max(int(samples), 8)
        n = lo
        while n <= hi + 0.5 * step:
            e, pred = err_at(n)
            if e < best_e:
                best_e = e
                best_n = n
                best_pred = pred
            n += step
        span = max(0.02, 0.35 * (hi - lo))
        lo = max(float(n_lo), best_n - span)
        hi = min(float(n_hi), best_n + span)

    efl = float("nan")
    e0 = next((e for e in (params.get("elements") or []) if e.get("enabled")), None)
    if e0 is not None and best_pred is not None:
        efl = lensmaker_f(
            float(e0.get("R1", 0.0) or 0.0),
            float(e0.get("R2", 0.0) or 0.0),
            best_n,
            float(e0.get("thickness", 0.0) or 0.0),
        )

    ok = best_e < 5.0 and best_pred is not None
    msg = (
        f"n = {best_n:.4f} at {wavelength_nm:.0f} nm · residual {best_e:.3f} mm"
        if ok
        else "Could not match the measured spot. Check alignment, screen Z, and that the ray clears the aperture."
    )
    return {
        "ok": ok,
        "n": best_n,
        "residual_mm": best_e,
        "predicted_xy": best_pred if best_pred is not None else (float("nan"), float("nan")),
        "efl_mm": efl,
        "wavelength_nm": float(wavelength_nm),
        "message": msg,
    }


def apply_calibrated_n(params: Dict[str, Any], n: float) -> Dict[str, Any]:
    """Write fitted n onto Custom n and switch photopolymer slots to CUSTOM."""
    out = copy.deepcopy(params)
    out["custom_n"] = float(n)
    out["calibrated_n"] = float(n)
    for e in out.get("elements") or []:
        if not e.get("enabled", True):
            continue
        mid = material_id_from_name(str(e.get("material", "")))
        if mid in _RESIN_IDS:
            e["material"] = "CUSTOM"
    return out
