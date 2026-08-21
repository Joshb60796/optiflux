"""
Standard singlet lens shape catalog.

Sign convention (light travels +Z, source at smaller Z):
  R > 0  → center of curvature is on the +Z side of the vertex
  R = 0  → plano (treated as infinite radius)
  R1 > 0 first surface convex toward the source
  R2 < 0 second surface convex toward the image  → biconvex

All combinations of front × rear: plano / convex / concave.
"""
from __future__ import annotations

import math
from typing import Dict, Any, List, Tuple

# Surface kind codes
PLANO = "plano"
CONVEX = "convex"   # bulges toward −Z on front, toward +Z on rear (focusing for glass)
CONCAVE = "concave"

SHAPE_KINDS: List[Tuple[str, str, str]] = [
    # (id, front, rear)
    ("plano_plano", PLANO, PLANO),
    ("plano_convex_PCX", PLANO, CONVEX),       # flat toward source, convex rear
    ("convex_plano_PCX", CONVEX, PLANO),       # convex toward source, flat rear
    ("plano_concave_PCV", PLANO, CONCAVE),
    ("concave_plano_PCV", CONCAVE, PLANO),
    ("biconvex", CONVEX, CONVEX),
    ("biconcave", CONCAVE, CONCAVE),
    ("equiconvex", CONVEX, CONVEX),            # |R1|=|R2|
    ("equiconcave", CONCAVE, CONCAVE),
    ("pos_meniscus", CONVEX, CONCAVE),         # thicker at center than edges (net +)
    ("neg_meniscus", CONCAVE, CONVEX),         # thinner at center (net −)
    ("convex_concave", CONVEX, CONCAVE),
    ("concave_convex", CONCAVE, CONVEX),
]

# Human-readable labels for dropdown (order matters)
SHAPE_LABELS: List[Tuple[str, str]] = [
    ("(custom / manual R)", "custom"),
    ("Ball lens (full sphere)", "ball"),
    ("Plano–Plano (window)", "plano_plano"),
    ("Plano–Convex (PCX, flat first)", "plano_convex_PCX"),
    ("Plano–Convex (PCX, convex first)", "convex_plano_PCX"),
    ("Plano–Concave (PCV, flat first)", "plano_concave_PCV"),
    ("Plano–Concave (PCV, concave first)", "concave_plano_PCV"),
    ("Bi-Convex", "biconvex"),
    ("Equi-Convex (|R1|=|R2|)", "equiconvex"),
    ("Bi-Concave", "biconcave"),
    ("Equi-Concave (|R1|=|R2|)", "equiconcave"),
    ("Positive Meniscus (CX–CC)", "pos_meniscus"),
    ("Negative Meniscus (CC–CX)", "neg_meniscus"),
    ("Convex–Concave", "convex_concave"),
    ("Concave–Convex", "concave_convex"),
]

SHAPE_DESCRIPTIONS = {
    "custom": "Use R₁ / R₂ sliders directly.",
    "ball": (
        "Complete sphere: thickness = 2R, clear semi-aperture = R. "
        "Collects a wide Lambertian cone (set emission half-angle to 90° for "
        "a full hemisphere) and is a common collimator for LED / fiber sources."
    ),
    "plano_plano": "Parallel plate / window. No optical power.",
    "plano_convex_PCX": "Flat toward source, convex rear. Common condenser orientation.",
    "convex_plano_PCX": "Convex toward source, flat rear. Common collimator orientation.",
    "plano_concave_PCV": "Flat toward source, concave rear. Negative lens.",
    "concave_plano_PCV": "Concave toward source, flat rear. Negative lens.",
    "biconvex": "Both surfaces convex (focusing). |R1| and |R2| may differ.",
    "equiconvex": "Symmetric biconvex, |R1| = |R2|.",
    "biconcave": "Both surfaces concave (diverging).",
    "equiconcave": "Symmetric biconcave, |R1| = |R2|.",
    "pos_meniscus": "Convex first, concave second; net positive power (center thicker).",
    "neg_meniscus": "Concave first, convex second; net negative power (center thinner).",
    "convex_concave": "Same as positive meniscus form.",
    "concave_convex": "Same as negative meniscus form.",
}


def _R_for(kind: str, surface: str, magnitude: float) -> float:
    """
    Map surface kind to signed radius for front (surface='1') or rear (surface='2').
    magnitude > 0.
    """
    mag = abs(float(magnitude))
    if mag < 1e-9:
        mag = 25.0
    if kind == PLANO:
        return 0.0
    if surface == "1":
        # Front: convex → R>0, concave → R<0
        return mag if kind == CONVEX else -mag
    # Rear: convex (toward +image, bulging +Z at center relative to rim for focusing biconvex)
    # → R2 < 0; concave rear → R2 > 0
    return -mag if kind == CONVEX else mag


BALL_R_MIN_MM = 0.5
BALL_R_MAX_MM = 80.0


def is_ball_shape(shape_id: str | None) -> bool:
    return str(shape_id or "").strip().lower() == "ball"


def ball_radius(value: float) -> float:
    r = abs(float(value))
    if r < 1e-9:
        r = 5.0
    return max(BALL_R_MIN_MM, min(BALL_R_MAX_MM, r))


def ball_radius_from_current(
    *,
    aperture: float,
    r_mag: float,
    R1: float,
    R2: float,
    thickness: float,
) -> float:
    """
    Sphere radius when applying the ball library type.

    Switching from another form uses the current clear semi-aperture (the OD)
    so a 10 mm singlet becomes a 10 mm-radius ball — not a giant inherited
    from leftover |R1|/|R2| (default biconvex is 40/−50).

    If the element is already a ball, honour the |R| magnitude slider so the
    user can scale the sphere.
    """
    r1 = abs(float(R1) or 0.0)
    r2 = float(R2) or 0.0
    t = abs(float(thickness) or 0.0)
    ap = abs(float(aperture) or 0.0)
    rm = abs(float(r_mag) or 0.0)
    already = (
        r1 >= BALL_R_MIN_MM
        and r2 < 0.0
        and abs(r1 + r2) <= 1e-3 * max(r1, 1.0)
        and abs(t - 2.0 * r1) <= max(0.2, 0.05 * r1)
    )
    if already:
        return ball_radius(rm if rm >= BALL_R_MIN_MM else r1)
    if ap >= BALL_R_MIN_MM:
        return ball_radius(ap)
    return ball_radius(rm if rm >= BALL_R_MIN_MM else r1)


def ball_efl(radius: float, n: float) -> float:
    """Effective focal length of a ball lens: n R / (2 (n − 1))."""
    r = abs(float(radius))
    nn = float(n)
    if r < 1e-12 or abs(nn - 1.0) < 1e-12:
        return float("inf")
    return nn * r / (2.0 * (nn - 1.0))


def ball_front_focal_length(radius: float, n: float) -> float:
    """Distance from the front vertex to the front focal point (mm)."""
    f = ball_efl(radius, n)
    if not math.isfinite(f):
        return float("inf")
    return f - abs(float(radius))


def is_ball_element(element: Dict[str, Any] | None) -> bool:
    """True if the element is a ball lens (catalog id or locked sphere geometry)."""
    if not element:
        return False
    if is_ball_shape(element.get("shape_id")):
        return True
    try:
        r1 = float(element.get("R1") or 0.0)
        r2 = float(element.get("R2") or 0.0)
        thick = float(element.get("thickness") or 0.0)
    except (TypeError, ValueError):
        return False
    if r1 <= 0.2 or r2 >= -0.2:
        return False
    if abs(r1 + r2) > 1e-3 * max(r1, 1.0):
        return False
    if abs(thick - 2.0 * r1) > max(0.15, 0.02 * r1):
        return False
    if abs(float(element.get("k1") or 0.0)) > 1e-6:
        return False
    if abs(float(element.get("k2") or 0.0)) > 1e-6:
        return False
    if abs(float(element.get("A4_1") or 0.0)) > 1e-12:
        return False
    if abs(float(element.get("A4_2") or 0.0)) > 1e-12:
        return False
    mode = str(element.get("surface_mode") or "rotational").lower()
    if mode not in ("rotational", ""):
        return False
    return True


def constrain_ball_element(
    element: Dict[str, Any],
    radius: float | None = None,
) -> Dict[str, Any]:
    """
    Lock an element to a sphere: R1=+R, R2=−R, thickness=2R, aperture=R.
    Mutates and returns ``element``.
    """
    if radius is None:
        try:
            radius = abs(float(element.get("R1") or 0.0))
        except (TypeError, ValueError):
            radius = 5.0
    r = ball_radius(radius)
    element["enabled"] = bool(element.get("enabled", True))
    element["R1"] = r
    element["R2"] = -r
    element["thickness"] = 2.0 * r
    element["aperture"] = r
    element["aperture_y"] = None
    element["R1y"] = None
    element["R2y"] = None
    element["k1"] = 0.0
    element["k2"] = 0.0
    element["A4_1"] = 0.0
    element["A4_2"] = 0.0
    element["k1y"] = 0.0
    element["k2y"] = 0.0
    element["A4_1y"] = 0.0
    element["A4_2y"] = 0.0
    element["surface_mode"] = "rotational"
    element["mode_s1"] = "rotational"
    element["mode_s2"] = "rotational"
    element["shape_id"] = "ball"
    element["circular_lock"] = True
    if "air_after" not in element:
        element["air_after"] = 1.5
    if "material" not in element:
        element["material"] = "FORMLABS_CLEAR"
    return element


def apply_shape(
    shape_id: str,
    *,
    R_mag: float = 25.0,
    R2_mag: float | None = None,
    thickness: float = 4.0,
    aperture: float = 12.0,
    material: str = "PMMA",
    k1: float = 0.0,
    k2: float = 0.0,
    A4_1: float = 0.0,
    A4_2: float = 0.0,
    air_after: float = 1.5,
) -> Dict[str, Any]:
    """
    Return element dict for a named shape.
    R_mag: |R| for curved surfaces (mm). R2_mag optional second magnitude.
    """
    if is_ball_shape(shape_id):
        r = ball_radius(R_mag)
        return constrain_ball_element(
            {
                "enabled": True,
                "material": material,
                "air_after": float(air_after),
            },
            radius=r,
        )

    if shape_id == "custom" or not shape_id:
        return {
            "enabled": True,
            "R1": R_mag,
            "R2": -(R2_mag if R2_mag is not None else R_mag),
            "thickness": thickness,
            "air_after": air_after,
            "aperture": aperture,
            "material": material,
            "k1": k1,
            "k2": k2,
            "A4_1": A4_1,
            "A4_2": A4_2,
            "shape_id": "custom",
        }

    # Find front/rear kinds
    front, rear = PLANO, PLANO
    for sid, f, r in SHAPE_KINDS:
        if sid == shape_id:
            front, rear = f, r
            break

    r2m = R2_mag if R2_mag is not None else R_mag
    # Equi shapes force equal |R|
    if shape_id in ("equiconvex", "equiconcave"):
        r2m = R_mag

    R1 = _R_for(front, "1", R_mag)
    R2 = _R_for(rear, "2", r2m)

    # Meniscus power tuning: |R_concave| larger than |R_convex| for mild positive meniscus
    if shape_id in ("pos_meniscus", "convex_concave"):
        R1 = abs(R_mag)           # convex front
        R2 = abs(r2m) * 1.4       # weaker concave rear → net positive
    elif shape_id in ("neg_meniscus", "concave_convex"):
        R1 = -abs(R_mag)          # concave front
        R2 = -abs(r2m) * 1.4      # weaker convex rear → net negative

    return {
        "enabled": True,
        "R1": float(R1),
        "R2": float(R2),
        "thickness": float(thickness),
        "air_after": float(air_after),
        "aperture": float(aperture),
        "material": material,
        "k1": float(k1),
        "k2": float(k2),
        "A4_1": float(A4_1),
        "A4_2": float(A4_2),
        "shape_id": shape_id,
    }


def shape_dropdown_values() -> List[str]:
    return [label for label, _ in SHAPE_LABELS]


def shape_id_from_label(label: str) -> str:
    for lab, sid in SHAPE_LABELS:
        if lab == label:
            return sid
    return "custom"


def shape_label_from_id(shape_id: str) -> str:
    for lab, sid in SHAPE_LABELS:
        if sid == shape_id:
            return lab
    return SHAPE_LABELS[0][0]
