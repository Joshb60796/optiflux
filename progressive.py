"""
Progressive Monte Carlo batches for OptiFlux.

Goal: show a usable result quickly after a control change, then refine
while the user leaves the parameters alone.

Default cadence:
  - 5 batches
  - 5000 target-plane rays per batch (map accumulates)
  - 500 side-view paths per batch (paths replaced each batch so the
    side view stays readable)

Cancel between batches when the GUI generation counter advances.
"""
from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine import (
    IrradianceMap,
    RayPath,
    SimResult,
    VISIBLE_NM_DEFAULT,
    assemble_surfaces,
    blockers_need_cpu,
    build_source_array,
    lensmaker_f,
    refractive_index,
    trace_ray,
)


def _empty_stats() -> Dict[str, Any]:
    return {
        "launched": 0,
        "hit": 0,
        "collection": 0.0,
        "rms": 0.0,
        "ee50": 0.0,
        "ee86": 0.0,
        "peak_e": 0.0,
        "centroid": (0.0, 0.0),
        "fov": {
            "fraction": 0.0,
            "uniformity": 0.0,
            "cv": 0.0,
            "footprint_aspect": 1.0,
            "target_aspect": 1.0,
            "aspect_error": 0.0,
            "sig_x": 0.0,
            "sig_y": 0.0,
        },
        "source_power": 0.0,
        "map_power": 0.0,
        "efl": float("nan"),
        "n_dies": 0,
        "n_surfaces": 0,
        "n_tir_absorb": 0,
        "n_absorb": 0,
        "n_reflections": 0,
        "n_backward": 0,
        "n_miss": 0,
        "backend": "cpu",
        "batch": 0,
        "n_batches": 0,
    }


def _finalize_stats(
    imap: IrradianceMap,
    *,
    launched: int,
    hit: int,
    total_f: float,
    surfaces: list,
    active: list,
    params: Dict[str, Any],
    n_tir: int = 0,
    n_absorb: int = 0,
    n_reflect: int = 0,
    n_backward: int = 0,
    n_miss: int = 0,
    backend: str = "cpu",
    batch_i: int = 0,
    n_batches: int = 1,
) -> Dict[str, Any]:
    """
    Build stats for the progressive run.

    Each map batch deposits rays with power_per = total_f / rays_per_batch, i.e.
    one full-source realization. After ``batch_i`` such batches the map is the
    *sum* of independent realizations, so absolute power metrics (collection,
    peak irradiance, map_power, FOV power_in) are divided by ``batch_i`` to
    report the batch-averaged estimate. Ratio metrics (FOV fraction, uniformity,
    centroid, RMS) are invariant under that scale factor.
    """
    custom_n = float(params.get("custom_n", 1.5))
    fov_w = float(params.get("fov_width", 40.0))
    fov_h = float(params.get("fov_height", 32.0))
    fov_cx = float(params.get("fov_cx", 0.0))
    fov_cy = float(params.get("fov_cy", 0.0))
    cx, cy, _ = imap.centroid()
    fov = imap.fov_metrics(fov_w, fov_h, fov_cx, fov_cy)
    e0 = next((e for e in params.get("elements", []) if e.get("enabled")), None)
    efl = float("nan")
    if e0:
        n_use = refractive_index(
            e0.get("material", "N_BK7"),
            float(params.get("source", {}).get("wavelength_nm", VISIBLE_NM_DEFAULT)),
            custom_n,
        )
        efl = lensmaker_f(
            float(e0["R1"]), float(e0["R2"]), n_use, float(e0["thickness"])
        )

    # Number of independent full-source map batches completed (≥1 once any
    # batch has run). Avoids collection ≈ batch_count × true_efficiency.
    n_avg = max(int(batch_i), 1)
    map_power_avg = imap.total_power / n_avg
    missed_avg = float(getattr(imap, "missed_power", 0.0) or 0.0) / n_avg
    # Plane collection includes hits outside the map window (unfocused beams).
    plane_power_avg = map_power_avg + missed_avg
    peak_avg = imap.max_irradiance() / n_avg
    if "power_in" in fov:
        fov = dict(fov)
        fov["power_in"] = float(fov.get("power_in", 0.0)) / n_avg
        # min_e / max_e / mean_e scale with deposited power; uniformity & cv
        # are ratios and stay correct. Scale absolute irradiance-like fields.
        for k in ("min_e", "max_e", "mean_e"):
            if k in fov:
                fov[k] = float(fov[k]) / n_avg

    return {
        "launched": launched,
        "hit": hit,
        "collection": plane_power_avg / total_f if total_f > 0 else 0.0,
        "rms": imap.rms_radius(),
        "ee50": imap.encircled_radius(0.5),
        "ee86": imap.encircled_radius(0.86),
        "peak_e": peak_avg,
        "centroid": (cx, cy),
        "fov": fov,
        "source_power": total_f,
        "map_power": map_power_avg,
        "plane_power": plane_power_avg,
        "missed_power": missed_avg,
        "efl": efl,
        "n_dies": len(active),
        "n_surfaces": len(surfaces),
        "n_tir_absorb": n_tir,
        "n_absorb": n_absorb,
        "n_reflections": n_reflect,
        "n_backward": n_backward,
        "n_miss": n_miss,
        "backend": backend,
        "batch": batch_i,
        "n_batches": n_batches,
    }


def _trace_cpu_batch(
    *,
    active,
    surfaces,
    target_z: float,
    n_rays: int,
    n_display: int,
    total_f: float,
    custom_n: float,
    apply_fresnel: bool,
    absorb_tir: bool,
    max_refl: int,
    kill_backward: bool,
    imap: IrradianceMap,
    accumulate_map: bool = True,
) -> Tuple[List[RayPath], int, int, int, int, int, int, int]:
    """Trace a CPU batch. Returns (paths, launched, hit, tir, absorb, refl, back, miss)."""
    paths: List[RayPath] = []
    if n_rays < 1 or not active:
        return paths, 0, 0, 0, 0, 0, 0, 0
    power_per = total_f / n_rays
    launched = hit = 0
    n_tir = n_absorb = n_reflect = n_backward = n_miss = 0
    for _ in range(n_rays):
        r = random.random() * total_f
        die = active[0]
        for dd in active:
            r -= dd.flux
            if r <= 0:
                die = dd
                break
        o, d, pwr, wl = die.spawn_ray(power_per)
        # When this batch exists only to build side-view paths (n_display ≈ n_rays),
        # store every ray until the budget is filled. Otherwise sample randomly.
        display_only = (not accumulate_map) and n_display >= max(1, int(0.5 * n_rays))
        if n_display >= n_rays:
            store = len(paths) < n_display
        else:
            store = len(paths) < n_display and (
                random.random() < (n_display / max(n_rays, 1)) * 1.4
                or len(paths) < min(40, n_display)
            )
        # Display-only batch: mild bias toward both meridional planes so the
        # Y–Z and X–Z side views are both populated. Map stats stay unbiased.
        if display_only and store:
            ox, oy, oz = o
            # Keep emission near the die centre (both meridians)
            o = (0.25 * ox, 0.25 * oy, oz)
            dx, dy, dz = d
            # Soften both transverse components equally (not only X)
            dx *= 0.35
            dy *= 0.35
            nrm = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
            d = (dx / nrm, dy / nrm, dz / nrm)
        ok, pt, pwr_out, path = trace_ray(
            o,
            d,
            pwr,
            wl,
            surfaces,
            target_z,
            custom_n,
            apply_fresnel,
            absorb_tir,
            store,
            max_reflections=max_refl,
            kill_backward=kill_backward,
        )
        launched += 1
        if path is not None:
            term = path.terminated
            if term == "tir_absorb":
                n_tir += 1
            elif term == "absorb":
                n_absorb += 1
            elif term == "backward":
                n_backward += 1
            elif term == "miss":
                n_miss += 1
            n_reflect += path.n_reflections
        if accumulate_map and ok and pt is not None:
            hit += 1
            imap.deposit(pt[0], pt[1], pwr_out)
        if store and path is not None and len(path.history) >= 2:
            paths.append(path)
    return paths, launched, hit, n_tir, n_absorb, n_reflect, n_backward, n_miss


def _inject_warp_grid(imap: IrradianceMap, warp_grid, warp_stats) -> Tuple[int, int, str]:
    """Accumulate a Warp irradiance grid into imap. Returns (launched, hit, backend)."""
    import numpy as _np

    g = _np.asarray(warp_grid, dtype=float)
    if g.ndim == 2 and g.shape == (imap.ny, imap.nx):
        g = _np.flipud(g)
    flat = g.ravel()
    if len(flat) != len(imap.bins):
        return 0, 0, "warp"
    for i, v in enumerate(flat):
        if v:
            imap.bins[i] += float(v)
    imap.total_power += float(flat.sum())
    imap.hit_count += int(warp_stats.get("hit", 0))
    launched = int(warp_stats.get("launched", 0))
    hit = int(warp_stats.get("hit", 0))
    backend = str(warp_stats.get("backend", "warp"))
    return launched, hit, backend


def run_simulation_progressive(
    params: Dict[str, Any],
    batch_cb: Optional[Callable[[SimResult, int, int], None]] = None,
    n_batches: int = 5,
    rays_per_batch: int = 5000,
    display_per_batch: int = 500,
    progress_cb: Optional[Callable[[float], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Optional[SimResult]:
    """
    Progressive Monte Carlo.

    Parameters
    ----------
    batch_cb :
        Called after each batch as batch_cb(result, batch_index_1based, n_batches).
        The GUI uses this to redraw intermediate results.
    n_batches :
        How many refinement passes (default 5).
    rays_per_batch :
        Target-plane rays per batch (default 5000). Map accumulates.
    display_per_batch :
        Side-view path count per batch (default 500). Paths are replaced
        each batch so the side view stays clear.
    should_cancel :
        Optional poll; return True to stop remaining batches (new user input).
    """
    dies = build_source_array(params["source"])
    target_z = float(params.get("target_z", 80.0))
    fov_cx = float(params.get("fov_cx", 0.0))
    fov_cy = float(params.get("fov_cy", 0.0))
    mla = dict(params.get("mla") or {})
    mla["_target_z"] = target_z
    mla["_fov_cx"] = fov_cx
    mla["_fov_cy"] = fov_cy
    if bool(mla.get("enabled", False)) and bool(mla.get("aim_to_fov", True)):
        from mla_geometry import (
            apply_mla_die_aim,
            lenslet_semi_aperture,
            scale_element_to_lenslet,
            thin_lens_focal_length_mm,
        )
        from materials_catalog import refractive_index, material_id_from_name, VISIBLE_NM_DEFAULT

        e0 = next((e for e in params.get("elements", []) if e.get("enabled", True)), None)
        if e0 is not None:
            ap = lenslet_semi_aperture(mla, dies, params.get("source"))
            g = scale_element_to_lenslet(e0, ap, scale_geometry=bool(mla.get("scale_to_pitch", True)))
            mat = material_id_from_name(str(e0.get("material", "ACRYLIC_PMMA")))
            n_g = refractive_index(mat, VISIBLE_NM_DEFAULT, float(params.get("custom_n", 1.5)))
            f_mm = thin_lens_focal_length_mm(g["R1"], g["R2"], n_g, g["thickness"])
            apply_mla_die_aim(
                dies, {**params, "mla": mla}, focal_length=f_mm, aperture=g["aperture"]
            )
    surfaces = assemble_surfaces(
        params["elements"],
        float(params.get("lens_z_start", 3.0)),
        mla=mla,
        dies=dies,
        blockers=params.get("blockers"),
    )
    half_w = float(params.get("map_half_w", 50.0))
    half_h = float(params.get("map_half_h", 40.0))
    res = int(params.get("map_res", 96))
    ny = max(16, int(res * half_h / max(half_w, 1e-6)))
    imap = IrradianceMap(half_w, half_h, res, ny)

    custom_n = float(params.get("custom_n", 1.5))
    apply_fresnel = bool(params.get("apply_fresnel", True))
    absorb_tir = bool(params.get("absorb_on_tir", True))
    max_refl = int(params.get("max_reflections", 0))
    kill_backward = bool(params.get("kill_backward", True))
    use_warp = bool(params.get("use_warp", True))
    if use_warp and blockers_need_cpu(surfaces):
        use_warp = False

    active = [d for d in dies if d.enabled and d.flux > 0]
    if not active or n_batches < 1 or rays_per_batch < 1:
        stats = _empty_stats()
        result = SimResult(imap, [], stats, dies, surfaces)
        if batch_cb:
            batch_cb(result, 0, n_batches)
        return result

    total_f = sum(d.flux for d in active)
    launched = hit = 0
    n_tir = n_absorb = n_reflect = n_backward = n_miss = 0
    backend = "cpu"
    paths: List[RayPath] = []
    final: Optional[SimResult] = None

    for bi in range(n_batches):
        if should_cancel is not None and should_cancel():
            break

        batch_launched = 0
        batch_hit = 0
        warp_ok = False

        # Bulk target-plane rays (Warp if available)
        if use_warp and rays_per_batch >= 1000:
            try:
                from warp_backend import try_accelerate

                batch_params = dict(params)
                batch_params["total_rays"] = rays_per_batch

                def _prog(f, _bi=bi):
                    if progress_cb:
                        progress_cb((_bi + f) / n_batches)

                wg, ws = try_accelerate(
                    batch_params, dies, surfaces, progress_cb=_prog
                )
                if wg is not None:
                    bl, bh, backend = _inject_warp_grid(imap, wg, ws)
                    batch_launched = bl
                    batch_hit = bh
                    warp_ok = True
            except Exception as exc:
                print(f"[OptiFlux] Warp batch skipped: {exc}")

        if not warp_ok:
            _p, bl, bh, t, ab, rf, bk, ms = _trace_cpu_batch(
                active=active,
                surfaces=surfaces,
                target_z=target_z,
                n_rays=rays_per_batch,
                n_display=0,
                total_f=total_f,
                custom_n=custom_n,
                apply_fresnel=apply_fresnel,
                absorb_tir=absorb_tir,
                max_refl=max_refl,
                kill_backward=kill_backward,
                imap=imap,
                accumulate_map=True,
            )
            batch_launched = bl
            batch_hit = bh
            n_tir += t
            n_absorb += ab
            n_reflect += rf
            n_backward += bk
            n_miss += ms
            backend = "cpu"

        launched += batch_launched
        hit += batch_hit

        # Side-view paths: always CPU; replace each batch
        if display_per_batch > 0:
            new_paths, _, _, t2, ab2, rf2, bk2, ms2 = _trace_cpu_batch(
                active=active,
                surfaces=surfaces,
                target_z=target_z,
                n_rays=display_per_batch,
                n_display=display_per_batch,
                total_f=total_f,
                custom_n=custom_n,
                apply_fresnel=apply_fresnel,
                absorb_tir=absorb_tir,
                max_refl=max_refl,
                kill_backward=kill_backward,
                imap=imap,
                accumulate_map=False,
            )
            paths = new_paths
            n_tir += t2
            n_absorb += ab2
            n_reflect += rf2
            n_backward += bk2
            n_miss += ms2

        stats = _finalize_stats(
            imap,
            launched=launched,
            hit=hit,
            total_f=total_f,
            surfaces=surfaces,
            active=active,
            params=params,
            n_tir=n_tir,
            n_absorb=n_absorb,
            n_reflect=n_reflect,
            n_backward=n_backward,
            n_miss=n_miss,
            backend=backend,
            batch_i=bi + 1,
            n_batches=n_batches,
        )
        final = SimResult(imap, list(paths), stats, dies, surfaces)
        if batch_cb:
            batch_cb(final, bi + 1, n_batches)
        if progress_cb:
            progress_cb((bi + 1) / n_batches)

    return final
