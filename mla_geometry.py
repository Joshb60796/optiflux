"""
Shared MLA (micro-lens array) geometry from Element 1 + COB layout.

Ensures ray-tracer, side view, and CAD export use the same lenslet design:
  - Centers match COB dies
  - Clear aperture from pitch × fill (or manual)
  - Optional proportional scaling of R / thickness / A4 so Element 1's form
    remains a proper lens when reduced to die pitch (not a flat cylinder)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from export_cad import LensSpec, sag_xy


def die_pitch_mm(dies: list) -> float:
    if len(dies) < 2:
        return 1.6
    dists = []
    for i, a in enumerate(dies):
        best = 1e9
        for j, b in enumerate(dies):
            if i == j:
                continue
            best = min(best, math.hypot(a.cx - b.cx, a.cy - b.cy))
        if best < 1e8:
            dists.append(best)
    if not dists:
        return 1.6
    dists.sort()
    return dists[len(dists) // 2]


def lenslet_semi_aperture(mla: Dict[str, Any], dies: list, src: Optional[Dict] = None) -> float:
    ap = float(mla.get("lenslet_aperture", 0.0) or 0.0)
    if ap > 0:
        return ap
    fill = float(mla.get("fill_factor", 0.88))
    if dies and len(dies) >= 2:
        pitch = die_pitch_mm(dies)
    elif src:
        pitch = min(float(src.get("pitch_x", 1.6)), float(src.get("pitch_y", 1.6)))
    else:
        pitch = 1.6
    return max(0.1, 0.5 * pitch * fill)


def scale_element_to_lenslet(
    e0: Dict[str, Any],
    ap_lenslet: float,
    *,
    scale_geometry: bool = True,
) -> Dict[str, float]:
    """
    Map Element 1 design onto a lenslet.

    If scale_geometry: scale lengths so Element 1's design aperture maps to
    ap_lenslet, preserving surface shape (sag/aperture ratio). A4 scales as
    length^(−3). Keep Element 1's full design aperture in the UI — do not
    pre-shrink it to die size or R stays huge and lenslets look like flat cylinders.
    """
    design_ap = max(float(e0.get("aperture", 10.0)), 1e-6)
    # If design aperture is already micro-sized (≤ lenslet), treat R/t as final
    # when not scaling; when scaling with already-micro design, scale ≈ 1.
    if scale_geometry and design_ap > ap_lenslet * 1.05:
        scale = ap_lenslet / design_ap
    elif scale_geometry and design_ap < ap_lenslet * 0.5:
        # Design smaller than pitch — grow form slightly to fill lenslet
        scale = ap_lenslet / design_ap
    else:
        scale = 1.0
    scale = max(scale, 1e-4)

    def sc_R(R: float) -> float:
        if abs(R) < 1e-12:
            return 0.0
        return float(R) * scale

    R1 = sc_R(float(e0.get("R1", 20.0)))
    R2 = sc_R(float(e0.get("R2", -20.0)))
    r1y = e0.get("R1y", None)
    r2y = e0.get("R2y", None)
    R1y = sc_R(float(r1y)) if r1y is not None else None
    R2y = sc_R(float(r2y)) if r2y is not None else None

    thick = float(e0.get("thickness", 3.0)) * scale
    # Moldable plate: not paper-thin, not a deep cylinder
    thick = max(0.25, thick)
    thick = min(thick, max(ap_lenslet * 3.5, 0.4))

    # Rescue case: design aperture was already shrunk to die size while R stayed
    # macro → nearly flat tops (cylinder look). Only retarget when curvature is
    # essentially zero vs plate thickness (not intentional weak micro-lenses).
    if scale_geometry:
        def _edge_sag(R: float, ap: float) -> float:
            if abs(R) < 1e-12:
                return 0.0
            c = 1.0 / R
            r2 = ap * ap
            disc = 1.0 - c * c * r2
            if disc < 0:
                return abs(ap)  # clamped domain
            return abs((c * r2) / (1.0 + math.sqrt(max(0.0, disc))))

        sag1 = _edge_sag(R1, ap_lenslet)
        sag2 = _edge_sag(R2, ap_lenslet)
        # Flat if edge sag << plate thickness (classic "cylinder on slab")
        flat_thresh = max(0.008, 0.03 * thick)
        if max(sag1, sag2) < flat_thresh and (abs(R1) > 1e-9 or abs(R2) > 1e-9):
            target = max(0.1 * ap_lenslet, 0.12 * thick)
            def retarget(R: float) -> float:
                if abs(R) < 1e-12:
                    return 0.0
                sign = 1.0 if R > 0 else -1.0
                return sign * max(ap_lenslet * 1.2, (ap_lenslet ** 2) / (2.0 * target))

            if sag1 < flat_thresh and abs(R1) > 1e-9:
                R1 = retarget(R1)
            if sag2 < flat_thresh and abs(R2) > 1e-9:
                R2 = retarget(R2)
            if R1y is not None and abs(R1y) > 1e-9 and _edge_sag(R1y, ap_lenslet) < flat_thresh:
                R1y = retarget(R1y)
            if R2y is not None and abs(R2y) > 1e-9 and _edge_sag(R2y, ap_lenslet) < flat_thresh:
                R2y = retarget(R2y)

    a4_1 = float(e0.get("A4_1", 0.0))
    a4_2 = float(e0.get("A4_2", 0.0))
    if scale_geometry and abs(scale - 1.0) > 1e-12:
        # z ~ A4 r^4 → A4' = A4 / s^3 when r' = s r, z' = s z
        a4_1 = a4_1 / (scale ** 3)
        a4_2 = a4_2 / (scale ** 3)

    return {
        "R1": R1,
        "R2": R2,
        "R1y": R1y,
        "R2y": R2y,
        "thickness": thick,
        "aperture": ap_lenslet,
        "k1": float(e0.get("k1", 0.0)),
        "k2": float(e0.get("k2", 0.0)),
        "k1y": float(e0.get("k1y", e0.get("k1", 0.0))),
        "k2y": float(e0.get("k2y", e0.get("k2", 0.0))),
        "A4_1": a4_1,
        "A4_2": a4_2,
        "A4_1y": float(e0.get("A4_1y", a4_1)),
        "A4_2y": float(e0.get("A4_2y", a4_2)),
        "mode": str(e0.get("surface_mode", "rotational")),
        "scale": scale,
        "design_aperture": design_ap,
    }


def build_mla_lens_specs(
    params: Dict[str, Any],
    dies: Optional[list] = None,
) -> Tuple[List[LensSpec], Dict[str, Any]]:
    """
    LensSpec list for every die + metadata (scale, ap, thickness).
    """
    elements = params.get("elements", [])
    e0 = next((e for e in elements if e.get("enabled", True)), None)
    if e0 is None:
        return [], {}

    mla = params.get("mla", {}) or {}
    src = params.get("source", {}) or {}
    z0 = float(params.get("lens_z_start", 3.0))
    scale_geo = bool(mla.get("scale_to_pitch", True))

    if dies is None:
        from engine import build_source_array
        dies = build_source_array(src)

    if not dies:
        return [], {}

    ap = lenslet_semi_aperture(mla, dies, src)
    g = scale_element_to_lenslet(e0, ap, scale_geometry=scale_geo)

    # Ensure positive edge thickness at aperture for this scaled geometry
    from engine import OpticalSurface, max_aperture_positive_edge

    s1 = OpticalSurface(
        z_vertex=z0,
        radius=g["R1"],
        k=g["k1"],
        aperture=g["aperture"],
        a4=g["A4_1"],
        mode=g["mode"],
        radius_y=g["R1y"],
    )
    s2 = OpticalSurface(
        z_vertex=z0 + g["thickness"],
        radius=g["R2"],
        k=g["k2"],
        aperture=g["aperture"],
        a4=g["A4_2"],
        mode=g["mode"],
        radius_y=g["R2y"],
    )
    ap_ok = max_aperture_positive_edge(s1, s2, g["aperture"], min_edge=0.08)
    g["aperture"] = max(ap_ok, 0.08)

    specs: List[LensSpec] = []
    for d in dies:
        specs.append(
            LensSpec(
                R1=g["R1"],
                R2=g["R2"],
                thickness=g["thickness"],
                aperture=g["aperture"],
                k1=g["k1"],
                k2=g["k2"],
                A4_1=g["A4_1"],
                A4_2=g["A4_2"],
                x0=float(d.cx),
                y0=float(d.cy),
                z_front=z0,
                R1y=g["R1y"],
                R2y=g["R2y"],
                mode=g["mode"],
                aperture_y=g["aperture"],
            )
        )
    meta = {
        "aperture": g["aperture"],
        "thickness": g["thickness"],
        "scale": g["scale"],
        "R1": g["R1"],
        "R2": g["R2"],
        "z_front": z0,
        "n_lenslets": len(specs),
    }
    return specs, meta


def front_z_at(
    x: float,
    y: float,
    specs: List[LensSpec],
    land_sag: float,
) -> float:
    """Front surface Z: lenslet sag inside clear aperture, flat land between."""
    if not specs:
        return 0.0
    s0 = specs[0]
    mode = s0.mode or "rotational"
    R1y = s0.R1y if s0.R1y is not None else s0.R1
    best = None
    best_r = 1e99
    for s in specs:
        r = math.hypot(x - s.x0, y - s.y0)
        if r < best_r:
            best_r = r
            best = s
    assert best is not None
    ap = best.aperture
    if best_r <= ap + 1e-9:
        lx, ly = x - best.x0, y - best.y0
        sag = sag_xy(lx, ly, best.R1, R1y, best.k1, 0.0, best.A4_1, 0.0, mode)
        if sag is None:
            return best.z_front + land_sag
        return best.z_front + sag
    return best.z_front + land_sag


def rear_z_at(
    x: float,
    y: float,
    specs: List[LensSpec],
    land_sag: float,
) -> float:
    if not specs:
        return 1.0
    s0 = specs[0]
    mode = s0.mode or "rotational"
    R2y = s0.R2y if s0.R2y is not None else s0.R2
    z2 = s0.z_front + s0.thickness
    best = None
    best_r = 1e99
    for s in specs:
        r = math.hypot(x - s.x0, y - s.y0)
        if r < best_r:
            best_r = r
            best = s
    assert best is not None
    ap = best.aperture
    if best_r <= ap + 1e-9:
        lx, ly = x - best.x0, y - best.y0
        sag = sag_xy(lx, ly, best.R2, R2y, best.k2, 0.0, best.A4_2, 0.0, mode)
        if sag is None:
            return z2 + land_sag
        return z2 + sag
    return z2 + land_sag


def land_sags(specs: List[LensSpec]) -> Tuple[float, float]:
    """Rim sag values used for inter-lenslet flat lands."""
    if not specs:
        return 0.0, 0.0
    s = specs[0]
    mode = s.mode or "rotational"
    R1y = s.R1y if s.R1y is not None else s.R1
    R2y = s.R2y if s.R2y is not None else s.R2
    ap = s.aperture
    sf = sag_xy(ap, 0.0, s.R1, R1y, s.k1, 0.0, s.A4_1, 0.0, mode) or 0.0
    sr = sag_xy(ap, 0.0, s.R2, R2y, s.k2, 0.0, s.A4_2, 0.0, mode) or 0.0
    return sf, sr
