"""
Rectangular FOV / camera-field illumination design helpers.

Goal: map light from a single LED or COB array onto a rectangular region of
specified aspect ratio (like a thermography / machine-vision camera FOV).

Uses anamorphic (cylindrical / biconic) optics so X and Y powers differ:
  - Cylinder X shapes the horizontal half-width of the field
  - Cylinder Y shapes the vertical half-height

Paraxial estimate (thin cylindrical lens, LED near focal plane, non-imaging):
  Place source ≈ at f so output is nearly collimated in that meridian, then
  residual / geometric scale sets the footprint. We choose different f_x, f_y
  so the far-field aspect ≈ FOV_w / FOV_h.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from materials_catalog import refractive_index, VISIBLE_NM_DEFAULT, material_id_from_name


def fov_aspect(width_mm: float, height_mm: float) -> float:
    h = max(float(height_mm), 1e-9)
    return float(width_mm) / h


def set_fov_from_aspect(aspect: float, height_mm: float) -> Tuple[float, float]:
    """Return (width, height) for fixed height and aspect = W/H."""
    h = max(float(height_mm), 1e-6)
    a = max(float(aspect), 1e-6)
    return a * h, h


def set_fov_from_diagonal(aspect: float, diagonal_mm: float) -> Tuple[float, float]:
    """Sensor-style: aspect W/H and diagonal → width, height."""
    a = max(float(aspect), 1e-6)
    d = max(float(diagonal_mm), 1e-6)
    # d² = W² + H², W = a H → H = d / sqrt(a²+1)
    h = d / math.sqrt(a * a + 1.0)
    return a * h, h


def thin_cylinder_R(f_mm: float, n: float, plano_first: bool = True) -> Tuple[float, float]:
    """
    Plano-convex cylindrical radii for focal length f in the powered meridian.
    Thin lens: f = R / (n-1) for PCX cylinder → |R| = f (n-1).
    Returns (R1, R2) with the flat surface first if plano_first.
    """
    n = max(float(n), 1.01)
    f = float(f_mm)
    if abs(f) < 1e-6:
        return 0.0, 0.0
    R = f * (n - 1.0)
    # Sign: light +Z; convex toward source for focusing cylinder first surface R>0
    # PCX flat first (common): R1=0, R2 = -|R| for positive f (like spherical PCX)
    if f > 0:
        if plano_first:
            return 0.0, -abs(R)
        return abs(R), 0.0
    # negative (diverging)
    if plano_first:
        return 0.0, abs(R)
    return -abs(R), 0.0


def design_crossed_cylinders_for_rect_fov(
    *,
    fov_width: float,
    fov_height: float,
    target_z: float,
    lens_z_start: float = 3.0,
    source_z: float = 0.0,
    half_angle_deg: float = 60.0,
    material: str = "ACRYLIC_PMMA",
    wavelength_nm: float = VISIBLE_NM_DEFAULT,
    aperture: float = 12.0,
    thickness: float = 4.0,
    air_gap: float = 2.0,
    custom_n: float = 1.5,
) -> Dict[str, Any]:
    """
    Build a 2-element crossed-cylinder condenser aimed at a rectangular FOV.

    Element 1: cylinder power in X (shapes FOV width)
    Element 2: cylinder power in Y (shapes FOV height)

    Focal lengths are set so the source→lens distance is near f for each
    meridian, with a ratio f_x/f_y ≈ FOV_w/FOV_h so residual beam spread
    scales with the desired aspect ratio at the target plane.
    """
    mid = material_id_from_name(material)
    n = refractive_index(mid, wavelength_nm, custom_n)
    aspect = max(fov_aspect(fov_width, fov_height), 0.1)

    # Clear apertures (elliptical) — must stay smaller than |R| of curved surfaces
    ap_x = float(aperture)
    ap_y = float(aperture)
    if aspect >= 1.0:
        ap_y = max(aperture / aspect, aperture * 0.5)
    else:
        ap_x = max(aperture * aspect, aperture * 0.5)
    ap_x = max(ap_x, 3.0)
    ap_y = max(ap_y, 3.0)
    ap_max = max(ap_x, ap_y)

    # Minimum |R| so the surface is defined across the clear aperture (sphere domain)
    R_min = ap_max * 1.35
    f_min = R_min / max(n - 1.0, 0.05)

    # Place LED near the mean focal plane for soft collimation in both meridians
    f_mean = max(f_min, 20.0)
    # Wider FOV meridian → longer f → slightly more residual divergence from on-axis angles
    f_x = f_mean * math.sqrt(aspect)
    f_y = f_mean / math.sqrt(aspect)
    f_x = max(f_x, f_min)
    f_y = max(f_y, f_min)

    # Suggest object distance ≈ 0.9 × min(f) so both meridians are near focus
    L_suggest = 0.9 * min(f_x, f_y)
    lens_z_start = max(float(lens_z_start), source_z + L_suggest)
    L = lens_z_start - source_z
    Z = max(target_z - lens_z_start, 5.0)

    R1x_a, R1x_b = thin_cylinder_R(f_x, n, plano_first=False)  # CX first for collection
    R2y_a, R2y_b = thin_cylinder_R(f_y, n, plano_first=False)
    # Enforce |R| ≥ R_min
    if abs(R1x_a) > 1e-9 and abs(R1x_a) < R_min:
        R1x_a = math.copysign(R_min, R1x_a)
    if abs(R2y_a) > 1e-9 and abs(R2y_a) < R_min:
        R2y_a = math.copysign(R_min, R2y_a)

    e1 = {
        "enabled": True,
        "surface_mode": "cylinder_x",
        "mode_s1": "cylinder_x",
        "mode_s2": "cylinder_x",
        "R1": R1x_a,
        "R2": R1x_b,
        "R1y": 0.0,
        "R2y": 0.0,
        "thickness": thickness,
        "air_after": air_gap,
        "aperture": ap_x,
        "aperture_y": ap_y,
        "material": mid,
        "k1": 0.0,
        "k2": 0.0,
        "k1y": 0.0,
        "k2y": 0.0,
        "A4_1": 0.0,
        "A4_2": 0.0,
        "A4_1y": 0.0,
        "A4_2y": 0.0,
    }
    e2 = {
        "enabled": True,
        "surface_mode": "cylinder_y",
        "mode_s1": "cylinder_y",
        "mode_s2": "cylinder_y",
        "R1": 0.0,
        "R2": 0.0,
        "R1y": R2y_a,
        "R2y": R2y_b,
        # For cylinder_y mode, radius field is unused for power; store Ry in R1y/R2y
        # Also put powered radii into R1/R2 as Ry for display consistency when mode is cylinder_y
        "thickness": thickness,
        "air_after": 1.0,
        "aperture": ap_x,
        "aperture_y": ap_y,
        "material": mid,
        "k1": 0.0,
        "k2": 0.0,
        "k1y": 0.0,
        "k2y": 0.0,
        "A4_1": 0.0,
        "A4_2": 0.0,
        "A4_1y": 0.0,
        "A4_2y": 0.0,
    }
    # cylinder_y uses radius_y from R1y — also set R1/R2 for engine fallback
    e2["R1"] = 0.0
    e2["R2"] = 0.0
    # Engine cylinder_y: radius_y from R1y/R2y — need to map side correctly
    # _surface_from_element side1 uses R1y, side2 uses R2y. Good.
    # But curvature_y for cylinder_y uses radius_y; mode cylinder_y uses Ry from radius_y field.
    # For side1, Rx=R1=0, Ry=R1y=R2y_a. Good.

    # Fix cylinder_y powered values into R1y/R2y
    e2["R1y"] = R2y_a
    e2["R2y"] = R2y_b

    e3 = {
        "enabled": False,
        "surface_mode": "rotational",
        "R1": 40.0,
        "R2": -25.0,
        "R1y": None,
        "R2y": None,
        "thickness": 3.0,
        "air_after": 1.0,
        "aperture": 11.0,
        "aperture_y": None,
        "material": mid,
        "k1": 0.0,
        "k2": 0.0,
        "A4_1": 0.0,
        "A4_2": 0.0,
    }

    meta = {
        "f_x": f_x,
        "f_y": f_y,
        "n": n,
        "aspect": aspect,
        "object_distance": L,
        "image_distance": Z,
        "description": (
            f"Crossed cylinders: f_x={f_x:.1f} mm, f_y={f_y:.1f} mm → FOV "
            f"{fov_width:.0f}×{fov_height:.0f} mm (aspect {aspect:.2f}), "
            f"lens Z={lens_z_start:.1f} mm, n={n:.3f}"
        ),
    }
    return {"elements": [e1, e2, e3], "meta": meta, "lens_z_start": lens_z_start}


def design_biconic_singlet_for_rect_fov(
    *,
    fov_width: float,
    fov_height: float,
    target_z: float,
    lens_z_start: float = 3.0,
    source_z: float = 0.0,
    material: str = "ACRYLIC_PMMA",
    wavelength_nm: float = VISIBLE_NM_DEFAULT,
    aperture: float = 14.0,
    thickness: float = 5.0,
    custom_n: float = 1.5,
) -> Dict[str, Any]:
    """
    Single biconic PCX-style element with different Rx, Ry for rectangular footprint.
    Simpler package than crossed pair; less degrees of freedom.
    """
    mid = material_id_from_name(material)
    n = refractive_index(mid, wavelength_nm, custom_n)
    aspect = max(fov_aspect(fov_width, fov_height), 0.1)

    ap_x = float(aperture)
    ap_y = float(aperture)
    if aspect >= 1.0:
        ap_y = max(aperture / aspect, aperture * 0.5)
    else:
        ap_x = max(aperture * aspect, aperture * 0.5)
    ap_x, ap_y = max(ap_x, 3.0), max(ap_y, 3.0)
    R_min = max(ap_x, ap_y) * 1.35
    f_min = R_min / max(n - 1.0, 0.05)
    f_mean = max(f_min, 20.0)
    f_x = max(f_mean * math.sqrt(aspect), f_min)
    f_y = max(f_mean / math.sqrt(aspect), f_min)
    L_suggest = 0.9 * min(f_x, f_y)
    lens_z_start = max(float(lens_z_start), source_z + L_suggest)

    # Convex first biconic: R = f (n-1)
    Rx = max(abs(f_x * (n - 1.0)), R_min)
    Ry = max(abs(f_y * (n - 1.0)), R_min)

    e1 = {
        "enabled": True,
        "surface_mode": "biconic",
        "mode_s1": "biconic",
        "mode_s2": "biconic",
        "R1": Rx,
        "R1y": Ry,
        "R2": 0.0,
        "R2y": 0.0,
        "thickness": thickness,
        "air_after": 1.5,
        "aperture": ap_x,
        "aperture_y": ap_y,
        "material": mid,
        "k1": 0.0,
        "k1y": 0.0,
        "k2": 0.0,
        "k2y": 0.0,
        "A4_1": 0.0,
        "A4_1y": 0.0,
        "A4_2": 0.0,
        "A4_2y": 0.0,
    }
    e2 = {
        "enabled": False,
        "surface_mode": "rotational",
        "R1": 30.0,
        "R2": -30.0,
        "thickness": 3.0,
        "air_after": 2.0,
        "aperture": 12.0,
        "material": mid,
        "k1": 0.0,
        "k2": 0.0,
        "A4_1": 0.0,
        "A4_2": 0.0,
    }
    e3 = {**e2, "enabled": False}
    meta = {
        "f_x": f_x,
        "f_y": f_y,
        "Rx": Rx,
        "Ry": Ry,
        "n": n,
        "aspect": aspect,
        "description": (
            f"Biconic singlet: Rx={Rx:.1f} mm, Ry={Ry:.1f} mm for "
            f"FOV {fov_width:.0f}×{fov_height:.0f} mm (aspect {aspect:.2f})"
        ),
    }
    return {"elements": [e1, e2, e3], "meta": meta, "lens_z_start": lens_z_start}


def footprint_aspect_from_map(samples_x, samples_y, powers, frac: float = 0.5) -> float:
    """
    Estimate illuminated aspect ratio from weighted samples: width/height of
    axis-aligned box containing `frac` of total power (greedy by |x|,|y|).
    """
    if not powers or sum(powers) <= 0:
        return 1.0
    import numpy as np

    x = np.asarray(samples_x, dtype=float)
    y = np.asarray(samples_y, dtype=float)
    p = np.asarray(powers, dtype=float)
    total = p.sum()
    # Sort by max-norm to grow a square-like region... better: separate percentiles
    order_x = np.argsort(np.abs(x))
    order_y = np.argsort(np.abs(y))
    # Find half-widths containing frac of power
    def half_width(arr, order):
        acc = 0.0
        for i in order:
            acc += p[i]
            if acc >= frac * total:
                return abs(arr[i])
        return abs(arr[order[-1]])

    hx = half_width(x, order_x)
    hy = half_width(y, order_y)
    return hx / max(hy, 1e-9)
