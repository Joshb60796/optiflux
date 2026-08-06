"""
OptiFlux parameter optimizer.

Maximizes Monte-Carlo power delivered into the rectangular target FOV by tuning
lens radii, thicknesses, air gaps, apertures, and first-vertex Z.

Optional **two-phase rectangular** mode:
  Phase 1 — even illumination in the FOV (flux + uniformity; circular OK)
  Phase 2 — inject 1–4 elements (crossed cylinders, optional relay/biconic) and
            match footprint aspect σx/σy to the FOV width/height ratio

Uses SciPy differential evolution + optional Nelder–Mead polish.

Usage (library):
    from optimizer import optimize_fov_flux, OptimizeConfig
    result = optimize_fov_flux(params, OptimizeConfig(two_phase=True, extra_anamorphic_lenses=2))

Usage (CLI):
    python optimizer.py --rays 2500 --max-evals 80 --two-phase --extra-lenses 2
"""
from __future__ import annotations

import argparse
import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from engine import default_params, run_simulation


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class OptimizeConfig:
    """Controls search cost, objective, and which parameters move."""

    # Monte Carlo budget per evaluation (keep moderate — noise is expected)
    rays_per_eval: int = 2500
    map_res: int = 64
    # Global search
    max_evals: int = 80
    population_size: int = 12  # differential-evolution popsize multiplier
    seed: Optional[int] = 42
    # Local polish after DE
    polish: bool = True
    polish_maxiter: int = 40
    # Objective prioritizes a *bright, even rectangular* FOV fill:
    #   score ∝ fov_flux × coverage × profile_fill × uniformity
    #           / (1 + w_a*aspect_error) / (1 + w_s*size_error)
    # A small hot-spot can no longer beat a filled rectangle on flux alone.
    uniformity_weight: float = 0.8  # multiplies into uniformity term
    aspect_weight: float = 0.0  # 0 → ignore footprint aspect (circular OK)
    fill_weight: float = 2.0  # weight on size_error (under-fill of FOV)
    coverage_mix: float = 0.9  # how hard to require FOV bin / profile fill (0–1)
    # Penalize light that reaches the target *plane* outside the FOV rectangle
    # (spill) and light that never lands in the FOV at all (waste).
    spill_weight: float = 1.5  # plane power outside FOV / source
    waste_weight: float = 0.8  # (1 − fov_flux) soft penalty
    # Two-phase rectangular FOV design:
    #   Phase 1 — rotational (or current) optics → even light in FOV
    #   Phase 2 — add N anamorphic lenses → reshape footprint to FOV W/H
    two_phase: bool = False
    extra_anamorphic_lenses: int = 2  # 0–4 additional elements in phase 2
    anamorphic_mode: str = "crossed"  # "crossed" | "biconic"
    phase1_eval_fraction: float = 0.45  # share of max_evals for phase 1
    # Which free variables to include
    optimize_radii: bool = True
    optimize_thickness: bool = True
    optimize_air_gaps: bool = True
    optimize_aperture: bool = True
    optimize_lens_z: bool = True
    optimize_asphere: bool = False  # k1/k2/A4 — usually leave off at first
    # If set, only these element indices are free (None → every enabled element)
    optimize_element_indices: Optional[List[int]] = None
    # Soft penalty scale when geometry is unphysical
    penalty_scale: float = 5.0
    # Progress: called as progress_cb(fraction_0_to_1, message, best_score)
    progress_cb: Optional[Callable[[float, str, float], None]] = None
    # Return True to stop early
    should_cancel: Optional[Callable[[], bool]] = None
    # Force CPU (disable Warp) for deterministic-ish short evals
    force_cpu: bool = True


@dataclass
class OptimizeResult:
    params: Dict[str, Any]
    score: float
    fov_flux: float  # power in FOV / source power
    uniformity: float
    collection: float
    aspect_error: float = 0.0
    footprint_aspect: float = 1.0
    n_evals: int = 0
    history: List[Tuple[float, float]] = field(default_factory=list)  # (score, time)
    message: str = ""
    elapsed_s: float = 0.0
    phase: str = ""  # "", "1", "2", "1+2"


# ── Parameter packing ─────────────────────────────────────────────────────────


@dataclass
class _Var:
    name: str  # dotted path, e.g. "elements.0.R1" or "lens_z_start"
    lo: float
    hi: float
    scale: float = 1.0  # internal unit scale (usually 1)


def _get_path(params: Dict[str, Any], path: str) -> float:
    parts = path.split(".")
    cur: Any = params
    for p in parts:
        if p.isdigit():
            cur = cur[int(p)]
        else:
            cur = cur[p]
    if cur is None:
        return 0.0
    return float(cur)


def _set_path(params: Dict[str, Any], path: str, value: float) -> None:
    parts = path.split(".")
    cur: Any = params
    for p in parts[:-1]:
        if p.isdigit():
            cur = cur[int(p)]
        else:
            cur = cur[p]
    key = parts[-1]
    if key.isdigit():
        cur[int(key)] = value
    else:
        # Keep None semantics for optional anamorphic radii only when value is ~0
        # and mode is rotational — otherwise always write float.
        cur[key] = float(value)


def build_variable_list(
    params: Dict[str, Any],
    cfg: OptimizeConfig,
) -> List[_Var]:
    """
    Build the free-variable list from current params.

    Bounds are relative to the starting design so the search stays in a
    physically plausible neighbourhood of the user's setup.
    """
    vars_: List[_Var] = []
    source_z = float(params.get("source", {}).get("source_z", 0.0))
    target_z = float(params.get("target_z", 80.0))

    if cfg.optimize_lens_z:
        # Group object-distance strongly affects FOV fill size (conjugate scale).
        # Allow a wide throw from near the source to most of the way to the target.
        z0 = float(params.get("lens_z_start", 3.0))
        stack_t = 0.0
        for e in params.get("elements", []):
            if e.get("enabled", True):
                stack_t += float(e.get("thickness", 0.0)) + float(e.get("air_after", 0.0))
        lo = max(source_z + 0.4, 0.5)
        hi = max(lo + 2.0, min(target_z - max(stack_t, 2.0) - 1.0, target_z * 0.92))
        lo = min(lo, z0)
        hi = max(hi, z0 + 0.5)
        if hi <= lo:
            hi = lo + 5.0
        vars_.append(_Var("lens_z_start", lo, hi))

    only = cfg.optimize_element_indices
    for i, e in enumerate(params.get("elements", [])):
        if not e.get("enabled", True):
            continue
        if only is not None and i not in only:
            continue
        mode = str(e.get("surface_mode", "rotational")).lower()
        ap = max(float(e.get("aperture", 10.0)), 1.0)
        r_ref = max(
            abs(float(e.get("R1", 25.0) or 0.0)),
            abs(float(e.get("R2", 25.0) or 0.0)),
            abs(float(e.get("R1y") or 0.0)),
            abs(float(e.get("R2y") or 0.0)),
            ap * 1.5,
            5.0,
        )

        if cfg.optimize_radii:
            # Allow sign-preserving radius moves over a wide range
            for key in ("R1", "R2"):
                r = float(e.get(key, 0.0) or 0.0)
                if abs(r) < 1e-9:
                    # plano: allow weak curvature either side
                    vars_.append(_Var(f"elements.{i}.{key}", -r_ref * 2.0, r_ref * 2.0))
                else:
                    sign = 1.0 if r > 0 else -1.0
                    lo = sign * max(ap * 1.05, abs(r) * 0.25)
                    hi = sign * max(abs(r) * 3.0, ap * 2.0, 8.0)
                    if lo > hi:
                        lo, hi = hi, lo
                    vars_.append(_Var(f"elements.{i}.{key}", lo, hi))

            if mode in ("biconic", "cylinder_x", "cylinder_y"):
                for key in ("R1y", "R2y"):
                    ry = e.get(key, None)
                    r = float(ry) if ry is not None else float(e.get(key.replace("y", ""), 0.0) or 0.0)
                    if abs(r) < 1e-9:
                        vars_.append(_Var(f"elements.{i}.{key}", -r_ref * 2.0, r_ref * 2.0))
                    else:
                        sign = 1.0 if r > 0 else -1.0
                        lo = sign * max(ap * 1.05, abs(r) * 0.25)
                        hi = sign * max(abs(r) * 3.0, ap * 2.0, 8.0)
                        if lo > hi:
                            lo, hi = hi, lo
                        vars_.append(_Var(f"elements.{i}.{key}", lo, hi))

        if cfg.optimize_thickness:
            t = float(e.get("thickness", 4.0))
            vars_.append(
                _Var(
                    f"elements.{i}.thickness",
                    max(0.4, t * 0.35),
                    min(25.0, max(t * 2.5, t + 6.0)),
                )
            )

        if cfg.optimize_air_gaps:
            air = float(e.get("air_after", 1.0))
            vars_.append(
                _Var(
                    f"elements.{i}.air_after",
                    0.05,
                    min(40.0, max(air * 3.0, air + 8.0)),
                )
            )

        if cfg.optimize_aperture:
            vars_.append(
                _Var(
                    f"elements.{i}.aperture",
                    max(1.5, ap * 0.5),
                    min(40.0, max(ap * 1.8, ap + 5.0)),
                )
            )
            if e.get("aperture_y") is not None:
                apy = float(e["aperture_y"])
                vars_.append(
                    _Var(
                        f"elements.{i}.aperture_y",
                        max(1.5, apy * 0.5),
                        min(40.0, max(apy * 1.8, apy + 5.0)),
                    )
                )

        if cfg.optimize_asphere:
            for key, span in (
                ("k1", 3.0),
                ("k2", 3.0),
                ("A4_1", 5e-4),
                ("A4_2", 5e-4),
            ):
                v0 = float(e.get(key, 0.0) or 0.0)
                vars_.append(_Var(f"elements.{i}.{key}", v0 - span, v0 + span))

    return vars_


def apply_vector(base: Dict[str, Any], vars_: Sequence[_Var], x: Sequence[float]) -> Dict[str, Any]:
    p = copy.deepcopy(base)
    for var, val in zip(vars_, x):
        _set_path(p, var.name, float(val))
    # Keep shape_id as custom when radii move
    for e in p.get("elements", []):
        if e.get("enabled", True):
            e["shape_id"] = "custom"
    return p


# ── Geometry soft constraints ─────────────────────────────────────────────────


def _geometry_penalty(params: Dict[str, Any], scale: float) -> float:
    """
    Soft penalties for unphysical stacks so the optimizer prefers valid optics.
    Returns ≥ 0; added to the cost (minimization form).
    """
    pen = 0.0
    source_z = float(params.get("source", {}).get("source_z", 0.0))
    target_z = float(params.get("target_z", 80.0))
    z = float(params.get("lens_z_start", 3.0))

    if z <= source_z + 0.2:
        pen += (source_z + 0.2 - z) * 2.0

    for e in params.get("elements", []):
        if not e.get("enabled", True):
            continue
        thick = float(e.get("thickness", 1.0))
        air = float(e.get("air_after", 0.0))
        ap = float(e.get("aperture", 5.0))
        R1 = float(e.get("R1", 0.0) or 0.0)
        R2 = float(e.get("R2", 0.0) or 0.0)

        if thick < 0.3:
            pen += (0.3 - thick) * 5.0

        # Domain of spherical surface: |R| must exceed clear aperture
        for R in (R1, R2, e.get("R1y"), e.get("R2y")):
            if R is None:
                continue
            Rf = float(R)
            if abs(Rf) > 1e-9 and abs(Rf) < ap * 1.02:
                pen += (ap * 1.02 - abs(Rf)) * 0.5

        # Crude edge-thickness proxy for bi-convex / meniscus
        # sag ≈ r²/(2R); edge ≈ thick − sag1 + sag2 (sign-dependent)
        def _sag(R: float, r: float) -> float:
            if abs(R) < 1e-12:
                return 0.0
            c = 1.0 / R
            disc = 1.0 - c * c * r * r
            if disc < 0:
                return abs(r)  # outside domain → penalized above
            return (c * r * r) / (1.0 + math.sqrt(max(0.0, disc)))

        edge = thick - _sag(R1, ap) + _sag(R2, ap)
        # For typical sign convention (R1>0, R2<0 bi-convex) both sags push edge down
        edge_alt = thick - abs(_sag(R1, ap)) - abs(_sag(R2, ap))
        edge_est = min(edge, edge_alt)
        if edge_est < 0.15:
            pen += (0.15 - edge_est) * 3.0

        z += thick + air

    if z >= target_z - 1.0:
        pen += (z - (target_z - 1.0)) * 2.0

    return max(0.0, pen) * scale


# ── Anamorphic seeding (rectangular footprint) ────────────────────────────────


def _copy_element_template(src: Dict[str, Any], **overrides) -> Dict[str, Any]:
    e = copy.deepcopy(src)
    e.update(overrides)
    e["shape_id"] = "custom"
    return e


def inject_anamorphic_lenses(
    params: Dict[str, Any],
    n_extra: int,
    mode: str = "crossed",
) -> Dict[str, Any]:
    """
    Keep the existing (phase-1) enabled optics and append up to ``n_extra``
    anamorphic elements seeded from the rectangular-FOV design helpers.

    - n_extra ≥ 2 and mode "crossed" → cylinder_x then cylinder_y
    - otherwise → one biconic singlet (Rx ≠ Ry)

    Uses free (disabled) element slots when available; does not remove the
    phase-1 collector.
    """
    from rect_fov import (
        design_biconic_singlet_for_rect_fov,
        design_crossed_cylinders_for_rect_fov,
    )
    from materials_catalog import material_name_from_id

    p = copy.deepcopy(params)
    # Up to 4 extra slots (collector + 4 = 5 = MAX_ELEMENTS)
    n_extra = max(0, min(int(n_extra), 4))
    if n_extra < 1:
        return p

    from engine import MAX_ELEMENTS, pad_elements

    elements = pad_elements(list(p.get("elements") or []), MAX_ELEMENTS)

    # Keep only the primary phase-1 collector enabled as the base optic.
    # Extra enabled elements from a previous run would otherwise stack with the
    # new anamorphics (4+ surfaces → Fresnel loss, lower collection).
    enabled_idx = [i for i, e in enumerate(elements) if e.get("enabled", True)]
    if not enabled_idx:
        elements[0]["enabled"] = True
        enabled_idx = [0]
    collector_i = enabled_idx[0]
    for i, e in enumerate(elements):
        if i != collector_i:
            e["enabled"] = False

    free_idx = [i for i in range(len(elements)) if i != collector_i]
    e0 = elements[collector_i]
    mat = str(e0.get("material", "ACRYLIC_PMMA"))
    src = p.get("source") or {}
    fov_w = float(p.get("fov_width", 40.0))
    fov_h = float(p.get("fov_height", 32.0))
    target_z = float(p.get("target_z", 80.0))
    lens_z = float(p.get("lens_z_start", 3.0))
    z_cursor = lens_z + float(e0.get("thickness", 0)) + float(e0.get("air_after", 0))
    ap0 = float(e0.get("aperture", 12.0))
    t0 = max(3.0, float(e0.get("thickness", 4.0)) * 0.7)

    kwargs = dict(
        fov_width=fov_w,
        fov_height=fov_h,
        target_z=target_z,
        lens_z_start=max(lens_z, z_cursor - 1.0),
        source_z=float(src.get("source_z", 0.0)),
        material=mat,
        wavelength_nm=float(src.get("wavelength_nm", 550.0)),
        aperture=ap0,
        thickness=t0,
        custom_n=float(p.get("custom_n", 1.5)),
    )

    seeds: List[Dict[str, Any]] = []
    prefer_crossed = mode.lower().startswith("cross")

    if n_extra == 1 and not prefer_crossed:
        design = design_biconic_singlet_for_rect_fov(
            **{
                k: kwargs[k]
                for k in (
                    "fov_width",
                    "fov_height",
                    "target_z",
                    "lens_z_start",
                    "source_z",
                    "material",
                    "wavelength_nm",
                    "aperture",
                    "thickness",
                    "custom_n",
                )
            }
        )
        seeds = [design["elements"][0]]
    else:
        # Crossed cylinders form the rectangular core; further slots add a
        # mild spherical relay / field element to improve FOV evenness.
        design = design_crossed_cylinders_for_rect_fov(**kwargs)
        seeds = [design["elements"][0], design["elements"][1]]
        if n_extra >= 3:
            # Weak positive relay — helps homogenize without dominating power
            seeds.append(
                {
                    "enabled": True,
                    "shape_id": "custom",
                    "surface_mode": "rotational",
                    "R1": max(40.0, ap0 * 4.0),
                    "R2": -max(40.0, ap0 * 4.0),
                    "R1y": None,
                    "R2y": None,
                    "thickness": max(2.5, t0 * 0.6),
                    "air_after": 1.5,
                    "aperture": ap0,
                    "aperture_y": None,
                    "material": mat,
                    "k1": 0.0,
                    "k2": 0.0,
                    "A4_1": 0.0,
                    "A4_2": 0.0,
                }
            )
        if n_extra >= 4:
            design_b = design_biconic_singlet_for_rect_fov(
                **{
                    k: kwargs[k]
                    for k in (
                        "fov_width",
                        "fov_height",
                        "target_z",
                        "lens_z_start",
                        "source_z",
                        "material",
                        "wavelength_nm",
                        "aperture",
                        "thickness",
                        "custom_n",
                    )
                }
            )
            seeds.append(design_b["elements"][0])
        seeds = seeds[:n_extra]

    # Only use free (disabled) slots after the collector — never grow past
    # collector + len(seeds) enabled elements.
    slots = [i for i in free_idx if i > collector_i][: len(seeds)]
    if len(slots) < len(seeds):
        # fall back to any remaining free indices
        for i in free_idx:
            if i not in slots:
                slots.append(i)
            if len(slots) >= len(seeds):
                break
    slots = slots[: len(seeds)]

    for slot, seed in zip(slots, seeds):
        seed = dict(seed)
        seed["enabled"] = True
        seed["shape_id"] = "custom"
        seed["material"] = mat
        elements[slot] = seed

    # Modest air gap after collector into first anamorphic
    elements[collector_i]["air_after"] = max(
        0.5, float(elements[collector_i].get("air_after", 1.0))
    )

    p["elements"] = elements
    mla = dict(p.get("mla") or {})
    mla["enabled"] = False
    p["mla"] = mla
    p["_anamorphic_slots"] = list(slots)
    return p


# ── Objective ─────────────────────────────────────────────────────────────────


def evaluate_fov_flux(
    params: Dict[str, Any],
    cfg: OptimizeConfig,
) -> Tuple[float, float, float, float, float, float]:
    """
    Run a coarse Monte-Carlo and return
    (score, fov_flux, uniformity, collection, aspect_error, footprint_aspect).

    Score strongly rewards a *filled, even rectangle* of light in the FOV and
    penalizes light that misses the FOV:

      - fov_flux: power inside FOV / source power
      - coverage / profile_fill: FOV area and line-cut fill
      - uniformity: Emin/Emax over lit FOV bins
      - spill: plane power outside the FOV rectangle / source
      - waste: 1 − fov_flux (light not in FOV at all)
      - size_error / aspect_error: footprint shape vs FOV
    """
    p = copy.deepcopy(params)
    p["total_rays"] = int(cfg.rays_per_eval)
    p["display_rays"] = 0
    p["map_res"] = int(cfg.map_res)
    if cfg.force_cpu:
        p["use_warp"] = False

    try:
        result = run_simulation(p)
    except Exception:
        return -1.0, 0.0, 0.0, 0.0, 1.0, 1.0

    st = result.stats
    src = max(float(st.get("source_power", 0.0)), 1e-30)
    fov = st.get("fov") or {}
    power_in = float(fov.get("power_in", 0.0))
    if power_in <= 0 and st.get("map_power", 0) > 0:
        power_in = float(fov.get("fraction", 0.0)) * float(st["map_power"])
    fov_flux = power_in / src
    collection = float(st.get("collection", 0.0))
    plane_power = float(st.get("plane_power", st.get("map_power", 0.0)) or 0.0)
    # Light on the target plane but outside the FOV rectangle
    spill = max(0.0, (plane_power - power_in) / src)
    waste = max(0.0, 1.0 - fov_flux)
    uniformity = float(fov.get("uniformity", 0.0))
    aspect_error = float(fov.get("aspect_error", 0.0))
    footprint_aspect = float(fov.get("footprint_aspect", 1.0))
    coverage = float(fov.get("coverage", 0.0))
    size_error = float(fov.get("size_error", 1.0))
    orientation_flipped = float(fov.get("orientation_flipped", 0.0))
    # Line-cut fill (X & Y profiles through FOV centre) — matches the profile plot
    profile_fill = float(fov.get("profile_fill", coverage))

    # Require *area* fill + *line-cut* fill + *evenness*. A thin bright streak
    # through FOV centre scores high on flux but low on coverage/uniformity.
    mix = min(1.0, max(0.0, float(cfg.coverage_mix)))
    area_fill = 0.5 * coverage + 0.5 * profile_fill
    fill_factor = (1.0 - mix) + mix * area_fill
    # Uniformity is Emin/Emax over lit FOV bins — multiplicative so 10% uniform
    # cannot be rescued by a high flux term alone.
    w_u = max(0.0, float(cfg.uniformity_weight))
    uni_factor = (0.15 + 0.85 * uniformity) ** max(0.5, min(2.0, 0.5 + w_u))
    # Containment: of all plane power, how much is inside FOV (1 = no spill on plane)
    containment = power_in / max(plane_power, 1e-30) if plane_power > 1e-30 else 0.0
    contain_factor = 0.35 + 0.65 * min(1.0, max(0.0, containment))
    score = fov_flux * fill_factor * uni_factor * contain_factor
    w_a = float(cfg.aspect_weight)
    if w_a > 0:
        score = score / (1.0 + w_a * aspect_error)
    w_s = float(cfg.fill_weight)
    if w_s > 0:
        score = score / (1.0 + w_s * size_error)
    w_spill = max(0.0, float(getattr(cfg, "spill_weight", 0.0) or 0.0))
    if w_spill > 0:
        score = score / (1.0 + w_spill * spill)
    w_waste = max(0.0, float(getattr(cfg, "waste_weight", 0.0) or 0.0))
    if w_waste > 0:
        score = score / (1.0 + w_waste * waste * 0.5)
    # Landscape FOV must not keep a portrait beam (and vice versa).
    if orientation_flipped > 0.5:
        score *= 0.35
    # Mild complexity cost. Softer when the user explicitly asked for many
    # extra anamorphic elements (phase-2 stacks of 3–4).
    n_en = sum(1 for e in p.get("elements", []) if e.get("enabled", True))
    if n_en > 1:
        soft = 0.02 if int(getattr(cfg, "extra_anamorphic_lenses", 0) or 0) >= 3 else 0.04
        score = score / (1.0 + soft * (n_en - 1))
    score -= _geometry_penalty(p, cfg.penalty_scale)
    return score, fov_flux, uniformity, collection, aspect_error, footprint_aspect


# ── Main optimizer ────────────────────────────────────────────────────────────


def _optimize_once(
    base: Dict[str, Any],
    cfg: OptimizeConfig,
    *,
    label: str = "",
    frac_lo: float = 0.0,
    frac_hi: float = 1.0,
    eval_budget: Optional[int] = None,
) -> OptimizeResult:
    """Single-phase DE + optional polish on a fixed parameter vector set."""
    try:
        from scipy.optimize import differential_evolution, minimize
    except ImportError as exc:
        raise ImportError(
            "optimizer requires scipy. Install with: pip install scipy"
        ) from exc

    budget = int(eval_budget if eval_budget is not None else cfg.max_evals)
    budget = max(budget, 4)
    vars_ = build_variable_list(base, cfg)
    if not vars_:
        sc, ff, uni, col, ae, fa = evaluate_fov_flux(base, cfg)
        return OptimizeResult(
            params=base,
            score=sc,
            fov_flux=ff,
            uniformity=uni,
            collection=col,
            aspect_error=ae,
            footprint_aspect=fa,
            n_evals=1,
            message="No free variables — enable lens elements or expand OptimizeConfig flags.",
            phase=label,
        )

    bounds = [(v.lo, v.hi) for v in vars_]
    x0 = []
    for v in vars_:
        try:
            cur = _get_path(base, v.name)
        except Exception:
            cur = 0.5 * (v.lo + v.hi)
        x0.append(min(v.hi, max(v.lo, cur)))

    n_evals = [0]
    best_score = [-1e30]
    best_x = [list(x0)]
    history: List[Tuple[float, float]] = []
    t0 = time.perf_counter()
    cancelled = [False]
    budget_hit = [False]
    tag = f"{label} " if label else ""

    def _report(frac: float, msg: str) -> None:
        if cfg.progress_cb:
            g = frac_lo + (frac_hi - frac_lo) * max(0.0, min(1.0, frac))
            cfg.progress_cb(g, msg, best_score[0])

    def objective(x: Sequence[float]) -> float:
        if cfg.should_cancel and cfg.should_cancel():
            cancelled[0] = True
            return 1e6
        if n_evals[0] >= budget:
            budget_hit[0] = True
            return 1e6
        n_evals[0] += 1
        trial = apply_vector(base, vars_, x)
        score, ff, uni, col, ae, fa = evaluate_fov_flux(trial, cfg)
        if score > best_score[0]:
            best_score[0] = score
            best_x[0] = list(x)
            history.append((score, time.perf_counter() - t0))
        if n_evals[0] % max(1, budget // 10) == 0 or n_evals[0] <= 3:
            _report(
                min(0.95, n_evals[0] / max(budget, 1)),
                f"{tag}eval {n_evals[0]}/{budget}  score={best_score[0]:.4f}  "
                f"flux={ff * 100:.1f}%  aspect_err={ae * 100:.1f}%",
            )
        return -score

    def _de_callback(_xk, _convergence: float = 0.0) -> bool:
        if cancelled[0] or budget_hit[0] or n_evals[0] >= budget:
            return True
        if cfg.should_cancel and cfg.should_cancel():
            cancelled[0] = True
            return True
        return False

    # Multi-start on lens_z_start: FOV fill size is dominated by object distance.
    # Evaluate several group positions (near source → farther) before DE so the
    # search is not stuck at a local shape with the wrong conjugate scale.
    seed_vecs: List[List[float]] = [list(x0)]
    z_idx = next((i for i, v in enumerate(vars_) if v.name == "lens_z_start"), None)
    if z_idx is not None:
        z_lo, z_hi = bounds[z_idx]
        for frac in (0.05, 0.2, 0.4, 0.6, 0.85):
            xv = list(x0)
            xv[z_idx] = z_lo + frac * (z_hi - z_lo)
            seed_vecs.append(xv)

    _report(0.0, f"{tag}Multi-start ({len(seed_vecs)} seeds) · then DE…")
    for xv in seed_vecs:
        objective(xv)

    n_dim = len(vars_)
    pop = max(2, int(cfg.population_size))
    # DE population length must be ≥ 5; scipy uses popsize * n_dim individuals
    pop_n = max(5, pop * n_dim)
    # Build init population: best seeds + latin-ish fill
    import random as _rnd

    rng = _rnd.Random(cfg.seed if cfg.seed is not None else 0)
    init_pop: List[List[float]] = [list(best_x[0])]
    for xv in seed_vecs:
        if xv not in init_pop:
            init_pop.append(list(xv))
    while len(init_pop) < pop_n:
        row = []
        for (lo, hi) in bounds:
            row.append(lo + rng.random() * (hi - lo))
        # Bias ~40% of random rows toward a random seed's lens_z
        if z_idx is not None and rng.random() < 0.4 and seed_vecs:
            row[z_idx] = seed_vecs[rng.randrange(len(seed_vecs))][z_idx]
        init_pop.append(row)

    maxiter = max(2, int(math.ceil(max(1, budget - n_evals[0]) / max(pop_n, 1))))

    _report(
        min(0.2, n_evals[0] / max(budget, 1)),
        f"{tag}DE · {n_dim} params · {cfg.rays_per_eval} rays/eval · best={best_score[0]:.4f}",
    )

    differential_evolution(
        objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=maxiter,
        popsize=pop,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=cfg.seed,
        polish=False,
        workers=1,
        updating="immediate",
        atol=1e-4,
        tol=1e-3,
        init=init_pop[:pop_n],
        callback=_de_callback,
    )

    x_best = list(best_x[0])
    if cfg.polish and not cancelled[0] and n_evals[0] < budget:
        _report(0.9, f"{tag}Local polish (Nelder–Mead)…")

        def local_obj(x: Sequence[float]) -> float:
            xc = [min(hi, max(lo, float(xi))) for (lo, hi), xi in zip(bounds, x)]
            return objective(xc)

        try:
            minimize(
                local_obj,
                x0=x_best,
                method="Nelder-Mead",
                options={
                    "maxiter": min(cfg.polish_maxiter, max(5, budget - n_evals[0])),
                    "xatol": 0.05,
                    "fatol": 1e-4,
                },
            )
            x_best = list(best_x[0])
        except Exception:
            pass

    final_params = apply_vector(base, vars_, x_best)
    score, ff, uni, col, ae, fa = evaluate_fov_flux(final_params, cfg)
    if score > best_score[0]:
        best_score[0] = score

    # If the footprint is landscape/portrait-flipped vs FOV, try swapping
    # anamorphic X↔Y once — often recovers the correct orientation.
    try:
        from rect_fov import swap_anamorphic_xy_params

        p_check = copy.deepcopy(final_params)
        p_check["total_rays"] = int(cfg.rays_per_eval)
        p_check["display_rays"] = 0
        p_check["map_res"] = int(cfg.map_res)
        if cfg.force_cpu:
            p_check["use_warp"] = False
        st = run_simulation(p_check).stats
        flipped = float((st.get("fov") or {}).get("orientation_flipped", 0.0))
        if flipped > 0.5:
            swapped = swap_anamorphic_xy_params(final_params)
            sc2, ff2, uni2, col2, ae2, fa2 = evaluate_fov_flux(swapped, cfg)
            n_evals[0] += 1
            if sc2 > score:
                final_params, score = swapped, sc2
                ff, uni, col, ae, fa = ff2, uni2, col2, ae2, fa2
                best_score[0] = max(best_score[0], score)
                _report(
                    0.98,
                    f"{tag}Auto-swapped X↔Y (was orientation-flipped) · score={score:.4f}",
                )
    except Exception:
        pass

    elapsed = time.perf_counter() - t0
    msg = (
        f"{tag}Done in {elapsed:.1f}s · {n_evals[0]} evals · "
        f"FOV flux={ff * 100:.1f}% · uniformity={uni * 100:.1f}% · "
        f"aspect_err={ae * 100:.1f}% · collection={col * 100:.1f}%"
    )
    if cancelled[0]:
        msg = "Cancelled — " + msg
    _report(1.0, msg)

    return OptimizeResult(
        params=final_params,
        score=best_score[0],
        fov_flux=ff,
        uniformity=uni,
        collection=col,
        aspect_error=ae,
        footprint_aspect=fa,
        n_evals=n_evals[0],
        history=history,
        message=msg,
        elapsed_s=elapsed,
        phase=label,
    )


def optimize_fov_flux(
    params: Optional[Dict[str, Any]] = None,
    cfg: Optional[OptimizeConfig] = None,
) -> OptimizeResult:
    """
    Maximize flux into the rectangular FOV (optionally uniformity + aspect).

    When ``cfg.two_phase`` is True and ``extra_anamorphic_lenses`` ≥ 1:

    * **Phase 1** — optimize current optics for FOV flux + uniformity
      (aspect weight forced to 0; rotational collection is fine).
    * **Phase 2** — inject up to N anamorphic elements (crossed cylinders or
      biconic), then optimize radii / gaps so the footprint aspect matches
      the rectangular FOV while keeping flux and uniformity.
    """
    base = copy.deepcopy(params if params is not None else default_params())
    cfg = cfg or OptimizeConfig()

    if not cfg.two_phase or int(cfg.extra_anamorphic_lenses) < 1:
        return _optimize_once(base, cfg, label="")

    # ── Phase 1: even light in FOV (ignore aspect) ──────────────────────────
    cfg1 = copy.deepcopy(cfg)
    cfg1.aspect_weight = 0.0
    cfg1.two_phase = False
    n1 = max(8, int(round(cfg.max_evals * float(cfg.phase1_eval_fraction))))
    n2 = max(8, int(cfg.max_evals) - n1)

    r1 = _optimize_once(
        base,
        cfg1,
        label="P1",
        frac_lo=0.0,
        frac_hi=0.48,
        eval_budget=n1,
    )
    if cfg.should_cancel and cfg.should_cancel():
        r1.message = "Cancelled during phase 1 — " + r1.message
        r1.phase = "1"
        return r1

    # ── Phase 2: add anamorphic lenses, match FOV rectangle ─────────────────
    seeded = inject_anamorphic_lenses(
        r1.params,
        n_extra=int(cfg.extra_anamorphic_lenses),
        mode=str(cfg.anamorphic_mode or "crossed"),
    )
    cfg2 = copy.deepcopy(cfg)
    cfg2.two_phase = False
    # Strongly reward rectangular footprint; keep some flux/uniformity pressure
    cfg2.aspect_weight = max(float(cfg.aspect_weight), 1.5)
    cfg2.uniformity_weight = max(float(cfg.uniformity_weight), 0.15)
    cfg2.optimize_asphere = bool(cfg.optimize_asphere)
    # Only free the newly injected anamorphic slots (+ their air gaps / apertures)
    slots = seeded.pop("_anamorphic_slots", None)
    if slots:
        # Free anamorphic elements; keep phase-1 collector radii fixed but still
        # move the whole group along Z (lens_z_start) so the FOV fill can be sized.
        cfg2.optimize_element_indices = list(slots)
        cfg2.optimize_lens_z = True

    r2 = _optimize_once(
        seeded,
        cfg2,
        label="P2",
        frac_lo=0.5,
        frac_hi=1.0,
        eval_budget=n2,
    )

    total_evals = r1.n_evals + r2.n_evals
    elapsed = r1.elapsed_s + r2.elapsed_s
    history = list(r1.history) + list(r2.history)

    # Keep phase-1 if anamorphics did not improve FOV power. Extra surfaces always
    # cost Fresnel; only accept P2 when it actually delivers more FOV flux
    # (small tolerance for MC noise).
    keep_p2 = r2.fov_flux >= r1.fov_flux * 0.98 and (
        r2.score >= r1.score * 0.95 or r2.fov_flux > r1.fov_flux
    )
    best = r2 if keep_p2 else r1
    phase = "1+2" if keep_p2 else "1"
    note = "" if keep_p2 else " · kept P1 (extra lenses did not improve FOV flux)"
    msg = (
        f"Two-phase done in {elapsed:.1f}s · {total_evals} evals · "
        f"FOV flux={best.fov_flux * 100:.1f}% · uniformity={best.uniformity * 100:.1f}% · "
        f"aspect_err={best.aspect_error * 100:.1f}% "
        f"(σx/σy={best.footprint_aspect:.3f}){note}"
    )
    return OptimizeResult(
        params=best.params,
        score=best.score,
        fov_flux=best.fov_flux,
        uniformity=best.uniformity,
        collection=best.collection,
        aspect_error=best.aspect_error,
        footprint_aspect=best.footprint_aspect,
        n_evals=total_evals,
        history=history,
        message=msg,
        elapsed_s=elapsed,
        phase=phase,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


def _cli() -> int:
    ap = argparse.ArgumentParser(
        description="Optimize OptiFlux lens parameters for maximum FOV flux."
    )
    ap.add_argument("--rays", type=int, default=2500, help="Rays per evaluation")
    ap.add_argument("--max-evals", type=int, default=80, help="Approx. max evaluations")
    ap.add_argument("--uniformity-weight", type=float, default=0.25)
    ap.add_argument("--aspect-weight", type=float, default=0.0)
    ap.add_argument(
        "--two-phase",
        action="store_true",
        help="Phase1 flux+uniformity, then add anamorphic lenses for rectangular FOV",
    )
    ap.add_argument(
        "--extra-lenses",
        type=int,
        default=2,
        help="Extra elements to add in phase 2 (0–4; 2=crossed pair, 3–4=+relay)",
    )
    ap.add_argument(
        "--anamorphic",
        choices=("crossed", "biconic"),
        default="crossed",
        help="Phase-2 anamorphic form",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-polish", action="store_true")
    ap.add_argument("--asphere", action="store_true", help="Also optimize k and A4")
    args = ap.parse_args()

    cfg = OptimizeConfig(
        rays_per_eval=args.rays,
        max_evals=args.max_evals,
        uniformity_weight=args.uniformity_weight,
        aspect_weight=args.aspect_weight,
        two_phase=bool(args.two_phase),
        extra_anamorphic_lenses=max(0, min(4, args.extra_lenses)),
        anamorphic_mode=args.anamorphic,
        seed=args.seed,
        polish=not args.no_polish,
        optimize_asphere=args.asphere,
    )

    def prog(frac: float, msg: str, best: float) -> None:
        print(f"[{frac * 100:5.1f}%] {msg}")

    cfg.progress_cb = prog
    mode = "two-phase rectangular" if cfg.two_phase else "single-phase"
    print(f"Starting FOV optimization ({mode}) from default_params()…")
    result = optimize_fov_flux(default_params(), cfg)
    print()
    print(result.message)
    print(f"Best score: {result.score:.5f}")
    print(f"FOV flux:   {result.fov_flux * 100:.2f} % of source power")
    print(f"Uniformity: {result.uniformity * 100:.2f} %")
    print(f"Aspect err: {result.aspect_error * 100:.2f} %  (σx/σy={result.footprint_aspect:.3f})")
    print(f"Collection: {result.collection * 100:.2f} %")
    print("lens_z_start:", result.params.get("lens_z_start"))
    for i, e in enumerate(result.params.get("elements", [])):
        if not e.get("enabled"):
            continue
        mode_s = e.get("surface_mode", "rotational")
        print(
            f"  E{i + 1} [{mode_s}]: R1={e['R1']:.3g} R2={e['R2']:.3g} "
            f"t={e['thickness']:.3g} air={e['air_after']:.3g} ap={e['aperture']:.3g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
