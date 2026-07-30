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
