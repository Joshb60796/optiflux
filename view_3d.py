"""
Interactive isometric / free 3D view of the OptiFlux optical layout.

Shows source dies, lens surfaces, absorbing blockers, target / FOV plane,
and a sample of ray polylines. Uses matplotlib Axes3D (no extra deps).

Display orientation matches the main-window Target Plane: optical Y is
vertical (matplotlib plot Z). Default camera looks from +X / −Z so +X
comes toward the viewer (down-right) and +Z recedes (top-right).
"""
from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from engine import (
    OpticalSurface,
    assemble_surfaces,
    build_source_array,
    default_params,
)


BG = "#0b0f14"
BG2 = "#111820"
FG = "#94a3b8"
FG_BRIGHT = "#e2e8f0"
SOURCE = "#fbbf24"
LENS = "#5eead4"
TARGET = "#f472b6"
FOV = "#a78bfa"
BLOCKER = "#64748b"
RAY = "#7dd3fc"
GLASS = (0.42, 0.96, 0.88, 0.22)
GLASS_EDGE = (0.78, 1.0, 0.96, 0.55)
GLASS_RIM = "#e6fffb"

IRRAD_CMAP = LinearSegmentedColormap.from_list(
    "optiflux3d",
    [
        (0.0, "#050510"),
        (0.18, "#1a1460"),
        (0.40, "#1a6ad4"),
        (0.62, "#20e0d0"),
        (0.82, "#f4e06a"),
        (1.0, "#ffffff"),
    ],
)


# ── Optical (X,Y,Z) → plot (X',Y',Z') ───────────────────────────────────────
# Matplotlib's screen-vertical axis is plot Z. Map optical Y there so the
# 3D window matches the main Target Plane (Y up, X right in that 2D view).
#   X' =  Z   (optical axis along plot X)
#   Y' =  X
#   Z' =  Y   (vertical)
# view_init(elev=18, azim=135): camera in optical +X / −Z, slightly above.

DEFAULT_ELEV = 18.0
DEFAULT_AZIM = 135.0


def _w2p(
    x: Union[float, np.ndarray, Sequence[float]],
    y: Union[float, np.ndarray, Sequence[float]],
    z: Union[float, np.ndarray, Sequence[float]],
) -> Tuple[Any, Any, Any]:
    """Map optical (x,y,z) → plot coordinates (Z, X, Y) so optical Y is vertical."""
    return z, x, y


def _pt(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Single point world → plot."""
    return (float(z), float(x), float(y))


def _lens_wire(
    s: OpticalSurface,
    n_theta: int = 28,
    n_ring: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a surface as rings; returns plot-frame (x', y', z') arrays."""
    apx = max(float(s.aperture), 0.2)
    apy = float(s.aperture_y) if s.aperture_y is not None and s.aperture_y > 0 else apx
    shape = (s.aperture_shape or "circle").lower()
    xs, ys, zs = [], [], []
    thetas = np.linspace(0, 2 * math.pi, n_theta)
    for ri in range(1, n_ring + 1):
        f = ri / n_ring
        for th in thetas:
            if shape == "rect":
                ct, st = math.cos(th), math.sin(th)
                scale = max(abs(ct), abs(st), 1e-9)
                lx = f * apx * ct / scale
                ly = f * apy * st / scale
            else:
                lx = f * apx * math.cos(th)
                ly = f * apy * math.sin(th)
            zw = s.surface_z(s.x0 + lx, s.y0 + ly)
            if zw is None:
                continue
            px, py, pz = _pt(s.x0 + lx, s.y0 + ly, zw)
            xs.append(px)
            ys.append(py)
            zs.append(pz)
    if not xs:
        px, py, pz = _pt(s.x0, s.y0, s.z_vertex)
        return np.array([px]), np.array([py]), np.array([pz])
    return np.asarray(xs), np.asarray(ys), np.asarray(zs)


def downsample_grid(grid: np.ndarray, max_n: int = 80) -> np.ndarray:
    """Thin a 2D map so 3D plot_surface stays interactive."""
    g = np.asarray(grid, dtype=float)
    if g.ndim != 2 or g.size == 0:
        return g
    ny, nx = g.shape
    if max(ny, nx) <= max_n:
        return g
    step_y = max(1, int(math.ceil(ny / max_n)))
    step_x = max(1, int(math.ceil(nx / max_n)))
    return g[::step_y, ::step_x]


def irradiance_rgba(
    grid: np.ndarray,
    *,
    log_scale: bool = False,
) -> np.ndarray:
    """
    Map an irradiance grid to RGBA using the same night-phosphor ramp
    as the main target-plane view. Empty bins stay dark and more transparent
    so the screen glows only where light lands.
    """
    g = np.asarray(grid, dtype=float)
    if g.ndim != 2 or g.size == 0:
        out = np.zeros((1, 1, 4), dtype=float)
        out[..., 3] = 0.2
        return out
    peak = float(np.max(g))
    if peak <= 1e-30:
        out = np.zeros(g.shape + (4,), dtype=float)
        out[..., :3] = (0.03, 0.03, 0.07)
        out[..., 3] = 0.22
        return out
    if log_scale:
        n = np.log1p(g * 50.0 / peak)
        n = n / (float(np.max(n)) + 1e-30)
    else:
        n = g / peak
    n = np.clip(n, 0.0, 1.0) ** 0.82
    rgba = np.asarray(IRRAD_CMAP(n), dtype=float)
    rgba[..., 3] = 0.20 + 0.80 * n
    return rgba


def _target_plot_mesh(
    tz: float,
    hw: float,
    hh: float,
    ny: int,
    nx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """plot-frame grids for a target plane at optical Z = tz."""
    xo = np.linspace(-hw, hw, nx)
    # Match imshow origin=upper: first row is +Y
    yo = np.linspace(hh, -hh, ny)
    XO, YO = np.meshgrid(xo, yo)
    ZO = np.full_like(XO, tz)
    return _w2p(XO, YO, ZO)


def _glass_faces(s: OpticalSurface, n_theta: int = 36, n_ring: int = 7) -> List[List[Tuple[float, float, float]]]:
    """Triangulated refractive surface (plot coords) for a glass look."""
    apx = max(float(s.aperture), 0.2)
    apy = float(s.aperture_y) if s.aperture_y is not None and s.aperture_y > 0 else apx
    shape = (s.aperture_shape or "circle").lower()
    thetas = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)

    def _xy(f: float, th: float) -> Tuple[float, float]:
        ct, st = math.cos(th), math.sin(th)
        if shape == "rect":
            sc = max(abs(ct), abs(st), 1e-9)
            return s.x0 + f * apx * ct / sc, s.y0 + f * apy * st / sc
        return s.x0 + f * apx * ct, s.y0 + f * apy * st

    rings: List[List[Tuple[float, float, float]]] = []
    zc = s.surface_z(s.x0, s.y0)
    if zc is None:
        zc = s.z_vertex
    center = _pt(s.x0, s.y0, zc)
    for ri in range(1, n_ring + 1):
        f = ri / n_ring
        ring = []
        for th in thetas:
            xw, yw = _xy(f, th)
            zw = s.surface_z(xw, yw)
            if zw is None:
                zw = s.z_vertex
            ring.append(_pt(xw, yw, zw))
        rings.append(ring)
    faces: List[List[Tuple[float, float, float]]] = []
    for i in range(n_theta):
        j = (i + 1) % n_theta
        faces.append([center, rings[0][i], rings[0][j]])
    for r in range(len(rings) - 1):
        a, b = rings[r], rings[r + 1]
        for i in range(n_theta):
            j = (i + 1) % n_theta
            faces.append([a[i], b[i], b[j], a[j]])
    return faces


def _rect_faces_world(
    z0: float,
    z1: float,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> List[List[Tuple[float, float, float]]]:
    """Six faces of an axis-aligned box in *optical* coords, transformed to plot."""
    corners_w = [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]
    return [[_pt(*c) for c in face] for face in corners_w]


def build_scene(
    ax,
    params: Dict[str, Any],
    result=None,
    max_rays: int = 80,
    *,
    log_scale: bool = False,
    preserve_view: bool = False,
) -> None:
    """Clear and draw the 3D layout onto ``ax`` (Axes3D)."""
    elev = getattr(ax, "elev", None) if preserve_view else None
    azim = getattr(ax, "azim", None) if preserve_view else None

    ax.cla()
    ax.set_facecolor("#04060c")
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        try:
            axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.05))
            axis.line.set_color((1.0, 1.0, 1.0, 0.18))
        except Exception:
            pass
    ax.tick_params(colors="#64748b", labelsize=6)
    # Axis labels describe *optical* coordinates after the Target-Plane remap
    ax.set_xlabel("Z (mm)  ·  light →", color=FG)
    ax.set_ylabel("X (mm)", color=FG)
    ax.set_zlabel("Y (mm)", color=FG)

    p = params if params else default_params()
    dies = build_source_array(p.get("source") or {})
    mla = dict(p.get("mla") or {})
    mla["_target_z"] = float(p.get("target_z", 80))
    mla["_fov_cx"] = float(p.get("fov_cx", 0))
    mla["_fov_cy"] = float(p.get("fov_cy", 0))
    surfaces = assemble_surfaces(
        p.get("elements") or [],
        float(p.get("lens_z_start", 3.0)),
        mla=mla if mla.get("enabled") else None,
        dies=dies if mla.get("enabled") else None,
        blockers=p.get("blockers"),
    )
    if result is not None and getattr(result, "dies", None):
        dies = result.dies

    tz = float(p.get("target_z", 80))
    hw = float(p.get("map_half_w", 50))
    hh = float(p.get("map_half_h", 40))
    fw = float(p.get("fov_width", 40))
    fh = float(p.get("fov_height", 32))
    fcx = float(p.get("fov_cx", 0))
    fcy = float(p.get("fov_cy", 0))

    # Optical axis — a hairline of light through the bench
    ax.plot(*_w2p([0.0, 0.0], [0.0, 0.0], [-1.0, tz + 2.0]), color="#334155", lw=0.6, alpha=0.7, ls="--")

    # Source dies as glowing chips facing +Z
    for die in dies:
        if not getattr(die, "enabled", True):
            continue
        dhw, dhh = die.width / 2, die.height / 2
        z0, z1 = die.cz - 0.18, die.cz + 0.05
        faces = _rect_faces_world(z0, z1, die.cx - dhw, die.cx + dhw, die.cy - dhh, die.cy + dhh)
        body = Poly3DCollection(
            faces, alpha=0.92, facecolor="#b45309", edgecolor="#fde68a", linewidths=0.35
        )
        ax.add_collection3d(body)
        emit = [
            _pt(die.cx - dhw, die.cy - dhh, z1),
            _pt(die.cx + dhw, die.cy - dhh, z1),
            _pt(die.cx + dhw, die.cy + dhh, z1),
            _pt(die.cx - dhw, die.cy + dhh, z1),
        ]
        ax.add_collection3d(
            Poly3DCollection([emit], alpha=0.95, facecolor="#fef3c7", edgecolor="#fffbeb", linewidths=0.2)
        )

    # Lenses (glass shells) and blockers
    for s in surfaces:
        if getattr(s, "interaction", "refract") == "absorb":
            g = (getattr(s, "geom", "plane_z") or "plane_z").lower()
            half = max(float(getattr(s, "extent_z", 5.0) or 5.0), 0.25)
            z0, z1 = s.z_vertex - half, s.z_vertex + half
            if g == "cylinder_z":
                r_out = max(float(s.aperture), 0.2)
                r_in = float(s.inner_aperture or 0.0)
                th = np.linspace(0, 2 * math.pi, 40)
                for r, alpha in ((r_out, 0.55), (r_in, 0.7) if r_in > 1e-6 else (None, None)):
                    if r is None:
                        continue
                    xw = s.x0 + r * np.cos(th)
                    yw = s.y0 + r * np.sin(th)
                    for zw in (z0, z1):
                        px, py, pz = _w2p(xw, yw, np.full_like(th, zw))
                        ax.plot(px, py, pz, color=BLOCKER, lw=1.1, alpha=alpha)
                    for i in range(0, len(th), 5):
                        px, py, pz = _w2p(
                            [xw[i], xw[i]], [yw[i], yw[i]], [z0, z1]
                        )
                        ax.plot(px, py, pz, color=BLOCKER, lw=0.55, alpha=0.4)
            elif g == "plane_y":
                y = s.y0 + float(getattr(s, "plane_offset", 0.0) or 0.0)
                ox = max(float(s.aperture), 0.2)
                xs = [s.x0 - ox, s.x0 + ox, s.x0 + ox, s.x0 - ox]
                ys = [y, y, y, y]
                zs = [z0, z0, z1, z1]
                verts = [_pt(xs[j], ys[j], zs[j]) for j in range(4)]
                ax.add_collection3d(
                    Poly3DCollection(
                        [verts], alpha=0.28, facecolor="#1e293b", edgecolor="#94a3b8", linewidths=0.45
                    )
                )
            elif g == "plane_x":
                x = s.x0 + float(getattr(s, "plane_offset", 0.0) or 0.0)
                oy = max(
                    float(s.aperture_y if s.aperture_y is not None else s.aperture), 0.2
                )
                verts = [
                    _pt(x, s.y0 - oy, z0),
                    _pt(x, s.y0 + oy, z0),
                    _pt(x, s.y0 + oy, z1),
                    _pt(x, s.y0 - oy, z1),
                ]
                ax.add_collection3d(
                    Poly3DCollection(
                        [verts], alpha=0.28, facecolor="#1e293b", edgecolor="#94a3b8", linewidths=0.45
                    )
                )
            else:
                thick = max(float(getattr(s, "display_thickness", 1.0) or 1.0), 0.5)
                z0s, z1s = s.z_vertex - thick / 2, s.z_vertex + thick / 2
                shape = (s.aperture_shape or "circle").lower()
                if shape == "rect":
                    ox = max(float(s.aperture), 0.2)
                    oy = max(float(s.aperture_y if s.aperture_y else s.aperture), 0.2)
                    faces = _rect_faces_world(
                        z0s, z1s, s.x0 - ox, s.x0 + ox, s.y0 - oy, s.y0 + oy
                    )
                    ax.add_collection3d(
                        Poly3DCollection(
                            faces, alpha=0.32, facecolor="#1e293b", edgecolor="#94a3b8", linewidths=0.45
                        )
                    )
                else:
                    r_out = max(float(s.aperture), 0.2)
                    th = np.linspace(0, 2 * math.pi, 36)
                    xw = s.x0 + r_out * np.cos(th)
                    yw = s.y0 + r_out * np.sin(th)
                    for zw in (z0s, z1s):
                        px, py, pz = _w2p(xw, yw, np.full_like(th, zw))
                        ax.plot(px, py, pz, color=BLOCKER, lw=1.2, alpha=0.5)
            continue

        # Glass body — translucent filled surface + luminous rim
        try:
            faces = _glass_faces(s)
            if faces:
                ax.add_collection3d(
                    Poly3DCollection(
                        faces,
                        facecolor=GLASS,
                        edgecolor=GLASS_EDGE,
                        linewidths=0.12,
                        antialiased=True,
                    )
                )
        except Exception:
            pass
        xs, ys, zs = _lens_wire(s, n_theta=48, n_ring=1)
        ax.plot(xs, ys, zs, color=GLASS_RIM, lw=1.35, alpha=0.9)

    # Target screen: same irradiance map as the main window, as a glowing wall
    imap = getattr(result, "map", None) if result is not None else None
    grid = None
    if imap is not None and hasattr(imap, "as_grid"):
        try:
            grid = np.asarray(imap.as_grid(), dtype=float)
            hw = float(getattr(imap, "half_w", hw))
            hh = float(getattr(imap, "half_h", hh))
        except Exception:
            grid = None

    bezel = [
        _pt(-hw * 1.04, -hh * 1.04, tz + 0.15),
        _pt(hw * 1.04, -hh * 1.04, tz + 0.15),
        _pt(hw * 1.04, hh * 1.04, tz + 0.15),
        _pt(-hw * 1.04, hh * 1.04, tz + 0.15),
    ]
    ax.add_collection3d(
        Poly3DCollection(
            [bezel], alpha=0.55, facecolor="#0a0a14", edgecolor="#64748b", linewidths=0.7
        )
    )

    if grid is not None and grid.size > 0 and float(np.max(grid)) > 0:
        g = downsample_grid(grid, max_n=72)
        rgba = irradiance_rgba(g, log_scale=log_scale)
        ny, nx = g.shape
        PX, PY, PZ = _target_plot_mesh(tz, hw, hh, ny, nx)
        fc = rgba[:-1, :-1] if rgba.shape[0] == ny and rgba.shape[1] == nx else rgba
        ax.plot_surface(
            PX,
            PY,
            PZ,
            facecolors=fc,
            shade=False,
            linewidth=0,
            antialiased=False,
            rstride=1,
            cstride=1,
        )
    else:
        screen = [
            _pt(-hw, -hh, tz),
            _pt(hw, -hh, tz),
            _pt(hw, hh, tz),
            _pt(-hw, hh, tz),
        ]
        ax.add_collection3d(
            Poly3DCollection(
                [screen], alpha=0.22, facecolor="#1a1040", edgecolor=TARGET, linewidths=0.6
            )
        )

    # FOV — luminous violet frame sitting just in front of the phosphor
    fov_c = [
        _pt(fcx - fw / 2, fcy - fh / 2, tz - 0.08),
        _pt(fcx + fw / 2, fcy - fh / 2, tz - 0.08),
        _pt(fcx + fw / 2, fcy + fh / 2, tz - 0.08),
        _pt(fcx - fw / 2, fcy + fh / 2, tz - 0.08),
        _pt(fcx - fw / 2, fcy - fh / 2, tz - 0.08),
    ]
    ax.plot(
        [c[0] for c in fov_c],
        [c[1] for c in fov_c],
        [c[2] for c in fov_c],
        color="#e9d5ff",
        lw=2.2,
        alpha=1.0,
    )
    ax.plot(
        [c[0] for c in fov_c],
        [c[1] for c in fov_c],
        [c[2] for c in fov_c],
        color=FOV,
        lw=0.8,
        alpha=0.85,
    )

    # Sample rays — glow pass + core, sparks where they kiss the screen
    paths = []
    if result is not None and getattr(result, "paths", None):
        paths = list(result.paths[: max(0, int(max_rays))])
    hit_x, hit_y, hit_z = [], [], []
    for path in paths:
        hist = getattr(path, "history", None) or []
        if len(hist) < 2:
            continue
        term = getattr(path, "terminated", "")
        if term == "absorb":
            col = (0.94, 0.27, 0.27, 0.45)
        elif term == "tir_absorb":
            col = (0.98, 0.57, 0.24, 0.42)
        elif term == "target":
            col = (0.55, 0.92, 1.0, 0.38)
        else:
            col = (0.49, 0.83, 0.99, 0.22)
        xs = [pt[0] for pt in hist]
        ys = [pt[1] for pt in hist]
        zs = [pt[2] for pt in hist]
        px, py, pz = _w2p(np.array(xs), np.array(ys), np.array(zs))
        ax.plot(px, py, pz, color=col, lw=0.55, alpha=min(0.85, col[3] + 0.18))
        last = hist[-1]
        if abs(float(last[2]) - tz) < 2.0 or term == "target":
            a, b, c = _pt(float(last[0]), float(last[1]), tz)
            hit_x.append(a)
            hit_y.append(b)
            hit_z.append(c)
    if hit_x:
        ax.scatter(hit_x, hit_y, hit_z, s=8, c="#fff7ed", alpha=0.55, linewidths=0, depthshade=False)

    # Bounds in optical space, then transform corners for equal aspect in plot
    xs_all = [-hw, hw, fcx - fw / 2, fcx + fw / 2]
    ys_all = [-hh, hh, fcy - fh / 2, fcy + fh / 2]
    zs_all = [0.0, tz]
    for die in dies:
        xs_all.extend([die.cx - die.width / 2, die.cx + die.width / 2])
        ys_all.extend([die.cy - die.height / 2, die.cy + die.height / 2])
        zs_all.append(die.cz)
    for s in surfaces:
        xs_all.extend([s.x0 - s.aperture, s.x0 + s.aperture])
        ys_all.extend([s.y0 - (s.aperture_y or s.aperture), s.y0 + (s.aperture_y or s.aperture)])
        zs_all.append(s.z_vertex)

    # Transform all extent corners to plot frame
    px_all, py_all, pz_all = [], [], []
    for xw in (min(xs_all), max(xs_all)):
        for yw in (min(ys_all), max(ys_all)):
            for zw in (min(zs_all), max(zs_all)):
                a, b, c = _pt(xw, yw, zw)
                px_all.append(a)
                py_all.append(b)
                pz_all.append(c)
    xmin, xmax = min(px_all), max(px_all)
    ymin, ymax = min(py_all), max(py_all)
    zmin, zmax = min(pz_all), max(pz_all)
    pad = 0.08
    xr = max(xmax - xmin, 1.0)
    yr = max(ymax - ymin, 1.0)
    zr = max(zmax - zmin, 1.0)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)
    r = 0.5 * max(xr, yr, zr) * (1 + pad)
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    # Three-quarter view from +X / −Z: Y vertical, +X toward viewer, +Z recedes.
    if elev is None or azim is None:
        ax.view_init(elev=DEFAULT_ELEV, azim=DEFAULT_AZIM)
    else:
        ax.view_init(elev=float(elev), azim=float(azim))
    ax.set_title(
        "OptiFlux 3D  ·  left-drag to rotate · right-drag to zoom",
        color=FG_BRIGHT,
        fontsize=10,
    )


def open_isometric_view(
    parent: tk.Misc,
    params: Dict[str, Any],
    result=None,
    *,
    get_params=None,
    get_result=None,
) -> tk.Toplevel:
    """
    Open a Toplevel with an interactive 3D scene.

    get_params / get_result: optional callables for the Update button.
    """
    win = tk.Toplevel(parent)
    win.title("OptiFlux — 3D optical bench")
    win.geometry("960x740")
    win.configure(bg=BG)
    win.minsize(640, 480)

    bar = tk.Frame(win, bg=BG2, height=36)
    bar.pack(side="top", fill="x")
    ttk.Label(
        bar,
        text="3D bench  ·  phosphor target matches the main-window map  ·  left-drag rotate · right-drag zoom",
        background=BG2,
        foreground=FG_BRIGHT,
    ).pack(side="left", padx=10, pady=6)

    fig = Figure(figsize=(8.4, 6.4), facecolor="#04060c", dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, top=0.96, bottom=0.02)
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    toolbar_frame = tk.Frame(win, bg=BG2)
    toolbar_frame.pack(side="bottom", fill="x")
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.update()

    log_var = tk.BooleanVar(value=False)
    first = {"done": False}

    def _refresh():
        p = get_params() if callable(get_params) else params
        r = get_result() if callable(get_result) else result
        build_scene(
            ax,
            p,
            r,
            max_rays=90,
            log_scale=bool(log_var.get()),
            preserve_view=first["done"],
        )
        first["done"] = True
        canvas.draw_idle()

    def _reset_cam():
        ax.view_init(elev=DEFAULT_ELEV, azim=DEFAULT_AZIM)
        canvas.draw_idle()

    ttk.Checkbutton(bar, text="Log map", variable=log_var, command=_refresh).pack(
        side="right", padx=6
    )
    ttk.Button(bar, text="Reset camera", command=_reset_cam).pack(side="right", padx=4, pady=4)
    ttk.Button(bar, text="Update", command=_refresh).pack(side="right", padx=4, pady=4)
    ttk.Button(bar, text="Close", command=win.destroy).pack(side="right", padx=4, pady=4)

    _refresh()
    # Keep a strong ref so GC does not kill the window helpers
    win._optiflux_3d = {"fig": fig, "ax": ax, "canvas": canvas, "refresh": _refresh}
    return win
