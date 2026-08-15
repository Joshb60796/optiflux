"""
Rigid-body focus of a lens group so the source is imaged onto the target.

This is the flashlight / magic-lantern conjugate: each point on the die
maps to a small blur circle on the FOV plane, so the emitter *shape*
becomes visible. That is not the same as the smallest flood spot
(global RMS) or the optimizer's 'sharpest illumination border'.
"""
from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine import (
    apply_tilt,
    assemble_surfaces,
    build_source_array,
    lensmaker_f,
    sample_lambertian_cone,
    trace_ray,
)
from materials_catalog import material_id_from_name, refractive_index, VISIBLE_NM_DEFAULT


def map_half_covering_fov(
    fov_w: float,
    fov_h: float,
    fov_cx: float = 0.0,
    fov_cy: float = 0.0,
    cur_hw: float = 0.0,
    cur_hh: float = 0.0,
    *,
    pad: float = 1.08,
) -> Tuple[float, float]:
    """Recorded-map half-sizes that fully contain the FOV rectangle (never shrink)."""
    need_w = (abs(float(fov_cx)) + 0.5 * float(fov_w)) * float(pad)
    need_h = (abs(float(fov_cy)) + 0.5 * float(fov_h)) * float(pad)
    return max(float(cur_hw), need_w), max(float(cur_hh), need_h)


def enabled_stack_length_mm(params: Dict[str, Any]) -> float:
    """First front vertex → last rear vertex (internal air included, trailing air not)."""
    enabled = [e for e in (params.get("elements") or []) if e.get("enabled", True)]
    if not enabled:
        return 0.0
    length = 0.0
    for i, e in enumerate(enabled):
        length += float(e.get("thickness", 0.0) or 0.0)
        if i + 1 < len(enabled):
            length += float(e.get("air_after", 0.0) or 0.0)
    return length


def group_z_bounds(params: Dict[str, Any]) -> Tuple[float, float]:
    src_z = float((params.get("source") or {}).get("source_z", 0.0))
    tgt_z = float(params.get("target_z", 80.0))
    pack = enabled_stack_length_mm(params)
    lo = src_z + 0.5
    hi = tgt_z - pack - 0.8
    if hi < lo + 1.0:
        hi = lo + 1.0
    return lo, hi


def source_field_points(params: Dict[str, Any]) -> List[Tuple[float, float, float]]:
    """Centre plus the four corners of the emitting field (union of dies)."""
    dies = build_source_array(params.get("source") or {})
    if not dies:
        src = params.get("source") or {}
        z = float(src.get("source_z", 0.0))
        return [(0.0, 0.0, z)]
    z = float(dies[0].cz)
    xs, ys = [], []
    for d in dies:
        xs.extend([d.cx - 0.5 * d.width, d.cx + 0.5 * d.width])
        ys.extend([d.cy - 0.5 * d.height, d.cy + 0.5 * d.height])
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    # Pull corners slightly inside so they still emit from the die, not the rim air
    inset = 0.04
    x0i = x0 + inset * (x1 - x0)
    x1i = x1 - inset * (x1 - x0)
    y0i = y0 + inset * (y1 - y0)
    y1i = y1 - inset * (y1 - y0)
    return [
        (cx, cy, z),
        (x0i, y0i, z),
        (x1i, y0i, z),
        (x0i, y1i, z),
        (x1i, y1i, z),
    ]


def _first_enabled(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for e in params.get("elements") or []:
        if e.get("enabled", True):
            return e
    return None


def paraxial_object_distances(params: Dict[str, Any]) -> List[float]:
    """
    Thin-lens conjugates for Element 1: object distances u that image
    the source onto the target (two roots when throw > 4f).
    """
    e0 = _first_enabled(params)
    if e0 is None:
        return []
    src = params.get("source") or {}
    wl = float(src.get("wavelength_nm", VISIBLE_NM_DEFAULT))
    custom_n = float(params.get("custom_n", 1.5))
    mid = material_id_from_name(str(e0.get("material", "FORMLABS_CLEAR")))
    n = refractive_index(mid, wl, custom_n)
    try:
        f = lensmaker_f(
            float(e0.get("R1", 0.0) or 0.0),
            float(e0.get("R2", 0.0) or 0.0),
            n,
            float(e0.get("thickness", 0.0) or 0.0),
        )
    except Exception:
        return []
    if not math.isfinite(f) or f <= 0.4:
        return []
    src_z = float(src.get("source_z", 0.0))
    tgt_z = float(params.get("target_z", 80.0))
    # Thin-lens throw from first vertex to target, minus pack to last vertex
    pack = enabled_stack_length_mm(params)
    L = (tgt_z - src_z) - 0.5 * pack
    if L <= 4.0 * f + 0.5:
        return []
    disc = L * L - 4.0 * f * L
    if disc < 0.0:
        return []
    root = math.sqrt(disc)
    us = [(L - root) / 2.0, (L + root) / 2.0]
    lo, hi = group_z_bounds(params)
    out = []
    for u in us:
        z = src_z + u
        if lo - 0.5 <= z <= hi + 0.5:
            out.append(float(z))
    return out


def image_blur_at_z(
    params: Dict[str, Any],
    lens_z: float,
    *,
    rays_per_point: int = 40,
    rng: Optional[random.Random] = None,
) -> Dict[str, float]:
    """
    Mean RMS blur of point images on the target (mm).

    Each sample point on the source launches a cone; RMS is about that
    point's own centroid so the geometric image size is not in the score.
    """
    rng = rng or random.Random(1)
    p = dict(params)
    p["lens_z_start"] = float(lens_z)
    src = p.get("source") or {}
    dies = build_source_array(src)
    mla = p.get("mla") or {}
    mla_use = mla if mla.get("enabled") else None
    surfs = assemble_surfaces(
        p.get("elements") or [],
        float(lens_z),
        mla_use,
        dies if mla_use else None,
        blockers=p.get("blockers"),
    )
    target_z = float(p.get("target_z", 80.0))
    custom_n = float(p.get("custom_n", 1.5))
    apply_fr = bool(p.get("apply_fresnel", True))
    absorb_tir = bool(p.get("absorb_on_tir", True))
    kill_back = bool(p.get("kill_backward", True))
    wl = float(src.get("wavelength_nm", VISIBLE_NM_DEFAULT))
    half = float(src.get("half_angle_deg", 60.0))
    tx = float(src.get("tilt_x", 0.0))
    ty = float(src.get("tilt_y", 0.0))
    n_rays = max(8, int(rays_per_point))
    blurs: List[float] = []
    launched = 0
    hit_n = 0
    for ox, oy, oz in source_field_points(p):
        xs: List[float] = []
        ys: List[float] = []
        for _ in range(n_rays):
            launched += 1
            # Local RNG via stdlib random is used by sample_*; seed once per eval
            d = sample_lambertian_cone(half)
            d = apply_tilt(d, tx, ty)
            hit, pt, _pwr, _path = trace_ray(
                (float(ox), float(oy), float(oz)),
                d,
                1.0,
                wl,
                surfs,
                target_z,
                custom_n=custom_n,
                apply_fresnel=apply_fr,
                absorb_on_tir=absorb_tir,
                store_path=False,
                kill_backward=kill_back,
            )
            if hit and pt is not None:
                xs.append(float(pt[0]))
                ys.append(float(pt[1]))
                hit_n += 1
        if len(xs) < 3:
            continue
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        acc = 0.0
        for x, y in zip(xs, ys):
            acc += (x - cx) ** 2 + (y - cy) ** 2
        blurs.append(math.sqrt(acc / len(xs)))
    hit_frac = hit_n / max(launched, 1)
    if not blurs or hit_frac < 0.02:
        return {
            "blur_mm": float("inf"),
            "hit_frac": hit_frac,
            "n_hits": float(hit_n),
        }
    return {
        "blur_mm": float(sum(blurs) / len(blurs)),
        "hit_frac": hit_frac,
        "n_hits": float(hit_n),
    }


def _score(ev: Dict[str, float]) -> float:
    blur = float(ev.get("blur_mm", float("inf")))
    hf = float(ev.get("hit_frac", 0.0))
    if not math.isfinite(blur) or hf < 0.02:
        return float("inf")
    # Prefer positions that still collect light
    return blur / max(hf, 0.08)


def focus_group_on_target(
    params: Dict[str, Any],
    *,
    n_scan: int = 21,
    n_refine: int = 11,
    rays_per_point: int = 40,
    should_cancel: Optional[Callable[[], bool]] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """
    Sweep lens_z_start (group moves as a rigid body) and return the Z that
    minimizes point-image blur on the target plane.
    """
    enabled = [e for e in (params.get("elements") or []) if e.get("enabled", True)]
    if not enabled:
        return {
            "ok": False,
            "message": "Enable at least one lens element.",
            "lens_z_start": float(params.get("lens_z_start", 3.0)),
            "blur_mm": float("inf"),
            "hit_frac": 0.0,
            "n_eval": 0,
        }
    lo, hi = group_z_bounds(params)
    n_scan = max(7, int(n_scan))
    n_refine = max(5, int(n_refine))
    zs = [lo + (hi - lo) * i / (n_scan - 1) for i in range(n_scan)]
    for z in paraxial_object_distances(params):
        if lo <= z <= hi and all(abs(z - a) > 0.4 for a in zs):
            zs.append(z)
    zs.sort()

    best_z = float(params.get("lens_z_start", lo))
    best_ev = {"blur_mm": float("inf"), "hit_frac": 0.0}
    best_s = float("inf")
    n_eval = 0

    def consider(z: float) -> None:
        nonlocal best_z, best_ev, best_s, n_eval
        ev = image_blur_at_z(params, z, rays_per_point=rays_per_point)
        n_eval += 1
        s = _score(ev)
        if s < best_s:
            best_s = s
            best_z = float(z)
            best_ev = ev

    for i, z in enumerate(zs):
        if should_cancel and should_cancel():
            return {
                "ok": False,
                "message": "Focus search cancelled.",
                "lens_z_start": best_z,
                "blur_mm": float(best_ev.get("blur_mm", float("inf"))),
                "hit_frac": float(best_ev.get("hit_frac", 0.0)),
                "n_eval": n_eval,
            }
        consider(z)
        if progress_cb:
            progress_cb((i + 1) / (len(zs) + n_refine), f"Focus scan {i + 1}/{len(zs)}…")

    # Refine in a window around the best coarse sample
    span = max(2.0, 0.18 * (hi - lo))
    rlo = max(lo, best_z - span)
    rhi = min(hi, best_z + span)
    for j in range(n_refine):
        if should_cancel and should_cancel():
            break
        z = rlo + (rhi - rlo) * j / max(n_refine - 1, 1)
        consider(z)
        if progress_cb:
            progress_cb(
                (len(zs) + j + 1) / (len(zs) + n_refine),
                f"Focus refine {j + 1}/{n_refine}…",
            )

    para = paraxial_object_distances(params)
    note = (
        f"Imaged the source onto the target (blur {best_ev['blur_mm']:.2f} mm RMS)."
        if para and any(abs(best_z - u) < 0.25 * (hi - lo) for u in para)
        else (
            "Best available focus along the bench "
            f"(blur {best_ev['blur_mm']:.2f} mm RMS). "
            "A true source image needs throw ≳ 4× EFL."
        )
    )
    if not math.isfinite(float(best_ev.get("blur_mm", float("inf")))):
        return {
            "ok": False,
            "message": "No rays reached the target at any group position. "
            "Open the aperture or move the group off the source.",
            "lens_z_start": float(params.get("lens_z_start", lo)),
            "blur_mm": float("inf"),
            "hit_frac": 0.0,
            "n_eval": n_eval,
        }
    return {
        "ok": True,
        "message": note,
        "lens_z_start": round(best_z, 3),
        "blur_mm": float(best_ev["blur_mm"]),
        "hit_frac": float(best_ev["hit_frac"]),
        "n_eval": n_eval,
        "bounds": (lo, hi),
        "paraxial_z": para,
    }
