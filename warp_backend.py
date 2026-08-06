"""
NVIDIA Warp accelerated Monte Carlo ray tracer backend for OptiFlux.

Requires: pip install warp-lang

Falls back gracefully if Warp is unavailable or no CUDA device is present.
The bulk irradiance map is computed on the GPU; a small number of display
paths remain on the pure-Python tracer for side-view visualization.

Units remain millimetres. Optical axis = +Z.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional Warp
_WARP_AVAILABLE = False
_wp = None
try:
    import warp as wp
    wp.init()
    _WARP_AVAILABLE = True
    _wp = wp
except Exception:
    pass


def warp_available() -> bool:
    """True if Warp imported and at least one CUDA device is usable."""
    if not _WARP_AVAILABLE:
        return False
    try:
        return _wp.get_cuda_device_count() > 0
    except Exception:
        return False


def warp_device_info() -> str:
    if not _WARP_AVAILABLE:
        return "Warp not installed (pip install warp-lang)"
    try:
        n = _wp.get_cuda_device_count()
        if n == 0:
            return "Warp present but no CUDA device"
        return f"Warp CUDA devices: {n}"
    except Exception as e:
        return f"Warp error: {e}"


# ── Warp structs & device functions (only defined when Warp is present) ──────

if _WARP_AVAILABLE:
    wp = _wp

    @wp.struct
    class WSurf:
        z_vertex: float
        radius: float          # Rx
        radius_y: float        # Ry (same as radius for rotational)
        k: float
        k_y: float
        a4: float
        a4_y: float
        aperture: float        # ap_x / outer semi-size
        aperture_y: float      # ap_y (same → circular when shape=circle)
        x0: float
        y0: float
        n_before: float
        n_after: float
        mode: int              # 0=rotational, 1=biconic, 2=cyl_x, 3=cyl_y
        active: int
        interaction: int       # 0=refract, 1=absorb
        aperture_shape: int    # 0=circle/ellipse, 1=rect
        inner_aperture: float  # hole semi-x; 0 = solid
        inner_aperture_y: float

    @wp.struct
    class WRay:
        o: wp.vec3
        d: wp.vec3
        power: float
        alive: int
        hit_x: float
        hit_y: float
        hit_power: float

    @wp.func
    def _curv(R: float) -> float:
        if wp.abs(R) < 1.0e-12:
            return 0.0
        return 1.0 / R

    @wp.func
    def sag_xy(
        x: float,
        y: float,
        radius: float,
        radius_y: float,
        k: float,
        k_y: float,
        a4: float,
        a4_y: float,
        mode: int,
    ) -> float:
        """Return sag relative to vertex, or a large sentinel if outside domain."""
        cx = float(_curv(radius))
        cy = float(_curv(radius_y))
        kk = float(k)
        kky = float(k_y)
        aa4 = float(a4)
        aa4y = float(a4_y)
        if mode == 2:  # cylinder_x
            cy = float(0.0)
            kky = float(0.0)
            aa4y = float(0.0)
        elif mode == 3:  # cylinder_y
            cx = float(0.0)
            kk = float(0.0)
            aa4 = float(0.0)

        x2 = float(x * x)
        y2 = float(y * y)
        arg = float(1.0 - (1.0 + kk) * cx * cx * x2 - (1.0 + kky) * cy * cy * y2)
        if arg < 0.0:
            return 1.0e6  # invalid
        denom = float(1.0 + wp.sqrt(arg))
        if wp.abs(denom) < 1.0e-18:
            return 1.0e6
        z = float((cx * x2 + cy * y2) / denom + aa4 * x2 * x2 + aa4y * y2 * y2)
        return z

    @wp.func
    def in_region(lx: float, ly: float, apx: float, apy: float, shape: int) -> int:
        """Outer/hole region test. shape: 0=circle/ellipse, 1=rect."""
        ax = float(wp.max(apx, 1.0e-12))
        if shape == 1:
            ay = float(wp.max(apy, 1.0e-12))
            return 1 if (wp.abs(lx) <= ax + 1.0e-9 and wp.abs(ly) <= ay + 1.0e-9) else 0
        if apy < 1.0e-12 or wp.abs(apy - ax) < 1.0e-12:
            return 1 if (lx * lx + ly * ly) <= (ax * ax + 1.0e-9) else 0
        ay = float(wp.max(apy, 1.0e-12))
        return 1 if ((lx / ax) * (lx / ax) + (ly / ay) * (ly / ay)) <= 1.0 + 1.0e-9 else 0

    @wp.func
    def in_hit_region(
        lx: float,
        ly: float,
        apx: float,
        apy: float,
        shape: int,
        interaction: int,
        inn_x: float,
        inn_y: float,
    ) -> int:
        if in_region(lx, ly, apx, apy, shape) == 0:
            return 0
        if interaction == 1 and inn_x > 1.0e-12:
            if in_region(lx, ly, inn_x, inn_y, shape) == 1:
                return 0
        return 1

    @wp.func
    def normal_at(
        x: float,
        y: float,
        radius: float,
        radius_y: float,
        k: float,
        k_y: float,
        a4: float,
        a4_y: float,
        mode: int,
        x0: float,
        y0: float,
    ) -> wp.vec3:
        """Surface normal (pointing roughly +Z). Finite-difference based."""
        lx = float(x - x0)
        ly = float(y - y0)
        eps = float(1.0e-4)
        z0 = float(sag_xy(lx, ly, radius, radius_y, k, k_y, a4, a4_y, mode))
        zx = float(sag_xy(lx + eps, ly, radius, radius_y, k, k_y, a4, a4_y, mode))
        zy = float(sag_xy(lx, ly + eps, radius, radius_y, k, k_y, a4, a4_y, mode))
        if z0 > 1.0e5 or zx > 1.0e5 or zy > 1.0e5:
            return wp.vec3(0.0, 0.0, 1.0)
        dx = float((zx - z0) / eps)
        dy = float((zy - z0) / eps)
        n = wp.vec3(-dx, -dy, 1.0)
        return wp.normalize(n)

    @wp.func
    def intersect_surface(
        o: wp.vec3,
        d: wp.vec3,
        s: WSurf,
        t_min: float,
        t_max: float,
    ) -> wp.vec4:
        """
        Newton / analytic intersection.
        Returns (t, px, py, pz) or t < 0 if miss.
        """
        if wp.abs(d[2]) < 1.0e-14:
            return wp.vec4(-1.0, 0.0, 0.0, 0.0)
        # dynamic so it can be mutated in the Newton loop
        t = float((s.z_vertex - o[2]) / d[2])
        if t < t_min or t > t_max:
            return wp.vec4(-1.0, 0.0, 0.0, 0.0)

        for _ in range(8):
            p = o + d * t
            lx = float(p[0] - s.x0)
            ly = float(p[1] - s.y0)
            sag = float(sag_xy(lx, ly, s.radius, s.radius_y, s.k, s.k_y, s.a4, s.a4_y, s.mode))
            if sag > 1.0e5:
                return wp.vec4(-1.0, 0.0, 0.0, 0.0)
            f = float(p[2] - (s.z_vertex + sag))
            nrm = normal_at(
                p[0], p[1], s.radius, s.radius_y, s.k, s.k_y, s.a4, s.a4_y, s.mode, s.x0, s.y0
            )
            dfdt = float(d[2] - (nrm[0] * d[0] + nrm[1] * d[1]))
            if wp.abs(nrm[2]) > 1.0e-8:
                dzdx = float(-nrm[0] / nrm[2])
                dzdy = float(-nrm[1] / nrm[2])
                dfdt = float(d[2] - (dzdx * d[0] + dzdy * d[1]))
            if wp.abs(dfdt) < 1.0e-14:
                break
            t = float(t - f / dfdt)
            if t < t_min * 0.5 or t > t_max:
                return wp.vec4(-1.0, 0.0, 0.0, 0.0)

        p = o + d * t
        lx = float(p[0] - s.x0)
        ly = float(p[1] - s.y0)
        if in_hit_region(
            lx, ly, s.aperture, s.aperture_y, s.aperture_shape,
            s.interaction, s.inner_aperture, s.inner_aperture_y,
        ) == 0:
            return wp.vec4(-1.0, 0.0, 0.0, 0.0)
        return wp.vec4(t, p[0], p[1], p[2])

    @wp.func
    def snell_refract_wp(
        I: wp.vec3,
        N: wp.vec3,
        n1: float,
        n2: float,
    ) -> wp.vec4:
        """
        Returns (Tx, Ty, Tz, tir_flag) where tir_flag = 1.0 if TIR.
        """
        cosi = float(-wp.dot(I, N))
        Nx = float(N[0])
        Ny = float(N[1])
        Nz = float(N[2])
        if cosi < 0.0:
            Nx = float(-Nx)
            Ny = float(-Ny)
            Nz = float(-Nz)
            cosi = float(-wp.dot(I, wp.vec3(Nx, Ny, Nz)))
        cosi = float(wp.clamp(cosi, 0.0, 1.0))
        eta = float(n1 / n2)
        k = float(1.0 - eta * eta * (1.0 - cosi * cosi))
        if k < 0.0:
            R = wp.normalize(I + wp.vec3(Nx, Ny, Nz) * (2.0 * cosi))
            return wp.vec4(R[0], R[1], R[2], 1.0)
        cost = float(wp.sqrt(k))
        T = wp.normalize(I * eta + wp.vec3(Nx, Ny, Nz) * (eta * cosi - cost))
        return wp.vec4(T[0], T[1], T[2], 0.0)


    @wp.func
    def fresnel_T_wp(n1: float, n2: float, cos_i: float) -> float:
        cos_i = float(wp.clamp(wp.abs(cos_i), 0.0, 1.0))
        eta = float(n1 / n2)
        sin_t2 = float(eta * eta * (1.0 - cos_i * cos_i))
        if sin_t2 > 1.0:
            return 0.0
        cos_t = float(wp.sqrt(1.0 - sin_t2))
        # Warp has no pow(float, int) — square by multiply, not ** 2
        ts = float((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t))
        tp = float((n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t))
        rs = float(ts * ts)
        rp = float(tp * tp)
        R = float(0.5 * (rs + rp))
        return float(wp.max(0.0, 1.0 - R))

    @wp.kernel
    def trace_kernel(
        rays: wp.array(dtype=WRay),
        surfaces: wp.array(dtype=WSurf),
        n_surf: int,
        target_z: float,
        apply_fresnel: int,
        absorb_on_tir: int,
        kill_backward: int,
        half_w: float,
        half_h: float,
        nx: int,
        ny: int,
        grid: wp.array2d(dtype=float),
        max_interactions: int,
    ):
        tid = wp.tid()
        ray = rays[tid]
        if ray.alive == 0:
            return

        # Explicit float()/int() so Warp treats these as dynamic (mutable in loops)
        o = wp.vec3(ray.o[0], ray.o[1], ray.o[2])
        d = wp.vec3(ray.d[0], ray.d[1], ray.d[2])
        power = float(ray.power)
        last_i = int(-1)

        for guard in range(max_interactions):
            if kill_backward != 0 and d[2] < -1.0e-9:
                rays[tid].alive = 0
                return

            # closest hit — all mutated vars declared dynamic
            best_t = float(1.0e30)
            best_i = int(-1)
            best_p = wp.vec3(0.0, 0.0, 0.0)
            best_n = wp.vec3(0.0, 0.0, 1.0)

            for i in range(n_surf):
                if i == last_i:
                    continue
                s = surfaces[i]
                if s.active == 0:
                    continue
                hit = intersect_surface(o, d, s, 1.0e-5, 1.0e5)
                t = float(hit[0])
                if t > 0.0 and t < best_t:
                    best_t = t
                    best_i = int(i)
                    best_p = wp.vec3(hit[1], hit[2], hit[3])
                    best_n = normal_at(
                        best_p[0], best_p[1],
                        s.radius, s.radius_y, s.k, s.k_y, s.a4, s.a4_y, s.mode,
                        s.x0, s.y0,
                    )

            # target plane?
            t_tgt = float(-1.0)
            if wp.abs(d[2]) > 1.0e-14:
                tt = float((target_z - o[2]) / d[2])
                if tt > 1.0e-5:
                    t_tgt = tt

            if t_tgt > 0.0 and (best_i < 0 or t_tgt < best_t):
                p = o + d * t_tgt
                rays[tid].hit_x = p[0]
                rays[tid].hit_y = p[1]
                rays[tid].hit_power = power
                rays[tid].alive = 0
                if wp.abs(p[0]) <= half_w and wp.abs(p[1]) <= half_h:
                    u = (p[0] + half_w) / (2.0 * half_w)
                    v = (p[1] + half_h) / (2.0 * half_h)
                    ix = int(wp.min(nx - 1, wp.max(0, int(u * float(nx)))))
                    iy = int(wp.min(ny - 1, wp.max(0, int(v * float(ny)))))
                    wp.atomic_add(grid, iy, ix, power)
                return

            if best_i < 0:
                rays[tid].alive = 0
                return

            # interact with surface
            s = surfaces[best_i]
            # Opaque absorb panel / aperture stop — kill before Snell
            if s.interaction == 1:
                rays[tid].alive = 0
                return

            n1 = float(s.n_before)
            n2 = float(s.n_after)
            if wp.dot(d, best_n) > 0.0:
                best_n = -best_n

            refr = snell_refract_wp(d, best_n, n1, n2)
            tir = float(refr[3])
            if tir > 0.5:
                if absorb_on_tir != 0:
                    rays[tid].alive = 0
                    return
                d = wp.vec3(refr[0], refr[1], refr[2])
            else:
                d = wp.vec3(refr[0], refr[1], refr[2])
                if apply_fresnel != 0:
                    inc = wp.normalize(best_p - o)
                    cos_i = float(wp.clamp(-wp.dot(inc, best_n), 0.0, 1.0))
                    T = float(fresnel_T_wp(n1, n2, cos_i))
                    power = power * T
                    if power < 1.0e-8:
                        rays[tid].alive = 0
                        return

            o = best_p + d * 1.0e-4  # epsilon push
            last_i = int(best_i)

        rays[tid].alive = 0


def _mode_to_int(mode: str) -> int:
    m = (mode or "rotational").lower()
    if m == "biconic":
        return 1
    if m == "cylinder_x":
        return 2
    if m == "cylinder_y":
        return 3
    return 0


def _build_wsurf_list(surfaces, wavelength_nm: float, custom_n: float):
    """Convert Python OpticalSurface list → list of WSurf (host side).

    Warp 1.15+ struct constructors do not accept keyword arguments, so we
    create a default instance and assign fields explicitly.
    """
    from materials_catalog import refractive_index

    out = []
    for s in surfaces:
        n_b = refractive_index(s.material_before, wavelength_nm, custom_n)
        n_a = refractive_index(s.material_after, wavelength_nm, custom_n)
        ry = s.radius_y if s.radius_y is not None else s.radius
        shape = str(getattr(s, "aperture_shape", "circle") or "circle").lower()
        shape_i = 1 if shape == "rect" else 0
        if s.aperture_y is not None and float(s.aperture_y) > 0:
            apy = float(s.aperture_y)
        elif shape_i == 1:
            apy = float(s.aperture)
        else:
            apy = float(s.aperture)
        inn = getattr(s, "inner_aperture", None)
        inn_x = float(inn) if inn is not None and float(inn) > 0 else 0.0
        inn_y_raw = getattr(s, "inner_aperture_y", None)
        if inn_y_raw is not None and float(inn_y_raw) > 0:
            inn_y = float(inn_y_raw)
        else:
            inn_y = inn_x

        ws = WSurf()
        ws.z_vertex = float(s.z_vertex)
        ws.radius = float(s.radius)
        ws.radius_y = float(ry)
        ws.k = float(s.k)
        ws.k_y = float(getattr(s, "k_y", s.k))
        ws.a4 = float(s.a4)
        ws.a4_y = float(getattr(s, "a4_y", s.a4))
        ws.aperture = float(s.aperture)
        ws.aperture_y = float(apy)
        ws.x0 = float(s.x0)
        ws.y0 = float(s.y0)
        ws.n_before = float(n_b)
        ws.n_after = float(n_a)
        ws.mode = _mode_to_int(s.mode)
        ws.active = 1 if s.active else 0
        ws.interaction = 1 if getattr(s, "interaction", "refract") == "absorb" else 0
        ws.aperture_shape = shape_i
        ws.inner_aperture = inn_x
        ws.inner_aperture_y = inn_y
        out.append(ws)
    return out


def run_warp_monte_carlo(
    dies,
    surfaces,
    target_z: float,
    map_half_w: float,
    map_half_h: float,
    map_res: int,
    total_rays: int,
    wavelength_nm: float,
    custom_n: float = 1.5,
    apply_fresnel: bool = True,
    absorb_on_tir: bool = True,
    kill_backward: bool = True,
    progress_cb=None,
    map_ny: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Launch parallel ray tracing on GPU (or Warp CPU if no CUDA).
    Returns (irradiance_grid [ny, nx], stats_dict).
    """
    if not _WARP_AVAILABLE:
        raise RuntimeError("Warp is not available")

    wp = _wp
    device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    nx = int(map_res)
    ny = int(map_ny) if map_ny is not None else max(16, int(map_res * map_half_h / max(map_half_w, 1e-6)))

    wsurfs = _build_wsurf_list(surfaces, wavelength_nm, custom_n)
    n_surf = len(wsurfs)
    if n_surf == 0:
        grid = np.zeros((ny, nx), dtype=np.float64)
        return grid, {"hit": 0, "launched": total_rays, "collection": 0.0}

    surf_arr = wp.array(wsurfs, dtype=WSurf, device=device)

    active = [d for d in dies if getattr(d, "enabled", True)]
    if not active:
        grid = np.zeros((ny, nx), dtype=np.float64)
        return grid, {"hit": 0, "launched": 0, "collection": 0.0}

    total_f = sum(d.flux for d in active)
    power_per = total_f / max(total_rays, 1)

    host_rays = []
    for i in range(total_rays):
        r = np.random.random() * total_f
        die = active[0]
        for dd in active:
            r -= dd.flux
            if r <= 0:
                die = dd
                break
        o, d, pwr, _ = die.spawn_ray(power_per)
        wr = WRay()
        wr.o = wp.vec3(float(o[0]), float(o[1]), float(o[2]))
        wr.d = wp.vec3(float(d[0]), float(d[1]), float(d[2]))
        wr.power = float(pwr)
        wr.alive = 1
        wr.hit_x = 0.0
        wr.hit_y = 0.0
        wr.hit_power = 0.0
        host_rays.append(wr)

    ray_arr = wp.array(host_rays, dtype=WRay, device=device)
    grid_wp = wp.zeros((ny, nx), dtype=float, device=device)

    wp.launch(
        trace_kernel,
        dim=total_rays,
        inputs=[
            ray_arr,
            surf_arr,
            n_surf,
            float(target_z),
            1 if apply_fresnel else 0,
            1 if absorb_on_tir else 0,
            1 if kill_backward else 0,
            float(map_half_w),
            float(map_half_h),
            nx,
            ny,
            grid_wp,
            32,
        ],
        device=device,
    )
    if progress_cb:
        progress_cb(1.0)

    grid = grid_wp.numpy().astype(np.float64)

    rays_host = ray_arr.numpy()
    try:
        hit = int(np.count_nonzero(rays_host["hit_power"] > 0))
    except (TypeError, ValueError, KeyError):
        hit = sum(1 for r in rays_host if getattr(r, "hit_power", 0) > 0)
    total_power = float(grid.sum())
    stats = {
        "launched": total_rays,
        "hit": hit,
        "collection": total_power / total_f if total_f > 0 else 0.0,
        "backend": f"warp/{device}",
    }
    return grid, stats


def try_accelerate(params: Dict[str, Any], dies, surfaces, progress_cb=None):
    """
    Attempt Warp acceleration for the irradiance map.
    Returns (grid_or_None, stats_or_None). Caller falls back to CPU on None.
    """
    if not warp_available() and not _WARP_AVAILABLE:
        return None, None
    try:
        from materials_catalog import VISIBLE_NM_DEFAULT
        wl = float(params["source"].get("wavelength_nm", VISIBLE_NM_DEFAULT))
        half_w = float(params["map_half_w"])
        half_h = float(params["map_half_h"])
        res = int(params["map_res"])
        ny = max(16, int(res * half_h / max(half_w, 1e-6)))
        grid, stats = run_warp_monte_carlo(
            dies=dies,
            surfaces=surfaces,
            target_z=float(params["target_z"]),
            map_half_w=half_w,
            map_half_h=half_h,
            map_res=res,
            total_rays=int(params["total_rays"]),
            wavelength_nm=wl,
            custom_n=float(params.get("custom_n", 1.5)),
            apply_fresnel=bool(params.get("apply_fresnel", True)),
            absorb_on_tir=bool(params.get("absorb_on_tir", True)),
            kill_backward=bool(params.get("kill_backward", True)),
            progress_cb=progress_cb,
            map_ny=ny,
        )
        return grid, stats
    except Exception as e:
        print(f"[OptiFlux] Warp backend failed, using CPU: {e}")
        return None, None
