"""
Interactive isometric / free 3D view of the OptiFlux optical layout.

Shows source dies, lens surfaces, absorbing blockers, target / FOV plane,
and a sample of ray polylines. Uses matplotlib Axes3D (no extra deps).

Display orientation: optical coordinates are rotated +90° about +Y so that
light (+Z) runs along the plot X axis (to the right / into the scene).
"""
from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
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


# ── Optical (X,Y,Z) → plot (X',Y',Z') ───────────────────────────────────────
# Rotation +90° about optical +Y (right-hand rule):
#   X' =  Z   (optical axis → plot right)
#   Y' =  Y
#   Z' = −X
# With view_init(elev≈20, azim=−90), rays point right and slightly into the screen.

def _w2p(
    x: Union[float, np.ndarray, Sequence[float]],
    y: Union[float, np.ndarray, Sequence[float]],
    z: Union[float, np.ndarray, Sequence[float]],
) -> Tuple[Any, Any, Any]:
    """Map optical (x,y,z) → plot coordinates after +90° rotation about Y."""
    return z, y, np.negative(x) if not np.isscalar(x) else -x


def _pt(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Single point world → plot."""
    return (float(z), float(y), float(-x))


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


def build_scene(ax, params: Dict[str, Any], result=None, max_rays: int = 40) -> None:
    """Clear and draw the 3D layout onto ``ax`` (Axes3D)."""
    ax.cla()
    ax.set_facecolor(BG)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.tick_params(colors=FG_BRIGHT, labelsize=7)
    # Axis labels describe *optical* coordinates (after the Y-rotation remapping)
    ax.set_xlabel("Z (mm)  ·  light →", color=FG)
    ax.set_ylabel("Y (mm)", color=FG)
    ax.set_zlabel("X (mm)", color=FG)

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

    # Source dies as flat boxes
    for die in dies:
        if not getattr(die, "enabled", True):
            continue
        hw, hh = die.width / 2, die.height / 2
        z0, z1 = die.cz - 0.15, die.cz + 0.15
        faces = _rect_faces_world(z0, z1, die.cx - hw, die.cx + hw, die.cy - hh, die.cy + hh)
        coll = Poly3DCollection(faces, alpha=0.85, facecolor=SOURCE, edgecolor="#fde68a", linewidths=0.4)
        ax.add_collection3d(coll)

    # Lenses (refractive) as wire rings; blockers by geom (stop / baffle / tube)
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
                # Horizontal wall strip along Z
                y = s.y0 + float(getattr(s, "plane_offset", 0.0) or 0.0)
                ox = max(float(s.aperture), 0.2)
                xs = [s.x0 - ox, s.x0 + ox, s.x0 + ox, s.x0 - ox]
                ys = [y, y, y, y]
                zs = [z0, z0, z1, z1]
                verts = [_pt(xs[j], ys[j], zs[j]) for j in range(4)]
                coll = Poly3DCollection(
                    [verts], alpha=0.35, facecolor=BLOCKER, edgecolor="#94a3b8", linewidths=0.5
                )
                ax.add_collection3d(coll)
            elif g == "plane_x":
                x = s.x0 + float(getattr(s, "plane_offset", 0.0) or 0.0)
                oy = max(
                    float(s.aperture_y if s.aperture_y is not None else s.aperture), 0.2
                )
                xs = [x, x, x, x]
                ys = [s.y0 - oy, s.y0 + oy, s.y0 + oy, s.y0 - oy]
                zs = [z0, z0, z1, z1]
                verts = [_pt(xs[j], ys[j], zs[j]) for j in range(4)]
                coll = Poly3DCollection(
                    [verts], alpha=0.35, facecolor=BLOCKER, edgecolor="#94a3b8", linewidths=0.5
                )
                ax.add_collection3d(coll)
            else:
                # Face-on stop (plane_z)
                thick = max(float(getattr(s, "display_thickness", 1.0) or 1.0), 0.5)
                z0s, z1s = s.z_vertex - thick / 2, s.z_vertex + thick / 2
                shape = (s.aperture_shape or "circle").lower()
                if shape == "rect":
                    ox = max(float(s.aperture), 0.2)
                    oy = max(float(s.aperture_y if s.aperture_y else s.aperture), 0.2)
                    faces = _rect_faces_world(
                        z0s, z1s, s.x0 - ox, s.x0 + ox, s.y0 - oy, s.y0 + oy
                    )
                    coll = Poly3DCollection(
                        faces, alpha=0.35, facecolor=BLOCKER, edgecolor="#94a3b8", linewidths=0.5
                    )
                    ax.add_collection3d(coll)
                else:
                    r_out = max(float(s.aperture), 0.2)
                    th = np.linspace(0, 2 * math.pi, 36)
                    xw = s.x0 + r_out * np.cos(th)
                    yw = s.y0 + r_out * np.sin(th)
                    for zw in (z0s, z1s):
                        px, py, pz = _w2p(xw, yw, np.full_like(th, zw))
                        ax.plot(px, py, pz, color=BLOCKER, lw=1.2, alpha=0.5)
            continue

        # Refractive surface
        xs, ys, zs = _lens_wire(s)
        ax.plot(xs, ys, zs, color=LENS, lw=0.7, alpha=0.75)

    # Target plane + FOV rectangle
    tz = float(p.get("target_z", 80))
    hw = float(p.get("map_half_w", 50))
    hh = float(p.get("map_half_h", 40))
    fw = float(p.get("fov_width", 40))
    fh = float(p.get("fov_height", 32))
    fcx = float(p.get("fov_cx", 0))
    fcy = float(p.get("fov_cy", 0))
    corners = [
        _pt(-hw, -hh, tz), _pt(hw, -hh, tz), _pt(hw, hh, tz), _pt(-hw, hh, tz),
    ]
    coll = Poly3DCollection(
        [corners], alpha=0.12, facecolor=TARGET, edgecolor=TARGET, linewidths=0.8
    )
    ax.add_collection3d(coll)
    fov_c = [
        _pt(fcx - fw / 2, fcy - fh / 2, tz),
        _pt(fcx + fw / 2, fcy - fh / 2, tz),
        _pt(fcx + fw / 2, fcy + fh / 2, tz),
        _pt(fcx - fw / 2, fcy + fh / 2, tz),
        _pt(fcx - fw / 2, fcy - fh / 2, tz),
    ]
    ax.plot(
        [c[0] for c in fov_c],
        [c[1] for c in fov_c],
        [c[2] for c in fov_c],
        color=FOV, lw=1.8, alpha=0.95,
    )

    # Sample rays
    paths = []
    if result is not None and getattr(result, "paths", None):
        paths = result.paths[:max_rays]
    for path in paths:
        hist = getattr(path, "history", None) or []
        if len(hist) < 2:
            continue
        term = getattr(path, "terminated", "")
        if term == "absorb":
            col, al = "#ef4444", 0.55
        elif term == "tir_absorb":
            col, al = "#f97316", 0.5
        else:
            col, al = RAY, 0.35
        xs = [pt[0] for pt in hist]
        ys = [pt[1] for pt in hist]
        zs = [pt[2] for pt in hist]
        px, py, pz = _w2p(np.array(xs), np.array(ys), np.array(zs))
        ax.plot(px, py, pz, color=col, alpha=al, lw=0.7)

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
    # Camera: look so plot +X (optical +Z / light) runs right and into the scene
    ax.view_init(elev=18, azim=-90)
    ax.set_title(
        "OptiFlux 3D layout  ·  left-drag to rotate · right-drag to zoom",
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
    win.title("OptiFlux — 3D isometric view")
    win.geometry("900x700")
    win.configure(bg=BG)
    win.minsize(640, 480)

    bar = tk.Frame(win, bg=BG2, height=36)
    bar.pack(side="top", fill="x")
    ttk.Label(
        bar,
        text="3D view · source · lenses · blockers · target  ·  left-drag rotate · right-drag zoom",
        background=BG2,
        foreground=FG_BRIGHT,
    ).pack(side="left", padx=10, pady=6)

    fig = Figure(figsize=(8, 6), facecolor=BG, dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    toolbar_frame = tk.Frame(win, bg=BG2)
    toolbar_frame.pack(side="bottom", fill="x")
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.update()

    def _refresh():
        p = get_params() if callable(get_params) else params
        r = get_result() if callable(get_result) else result
        build_scene(ax, p, r)
        canvas.draw_idle()

    ttk.Button(bar, text="Update", command=_refresh).pack(side="right", padx=8, pady=4)
    ttk.Button(bar, text="Close", command=win.destroy).pack(side="right", padx=4, pady=4)

    _refresh()
    # Keep a strong ref so GC does not kill the window helpers
    win._optiflux_3d = {"fig": fig, "ax": ax, "canvas": canvas, "refresh": _refresh}
    return win
