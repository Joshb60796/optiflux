#!/usr/bin/env python3
"""
OptiFlux — Desktop GUI for LED/COB lens design (physics ray tracer).

Run:
    python app.py

Requires: Python 3.9+, tkinter, matplotlib, numpy
"""
from __future__ import annotations

import copy
import math
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox, filedialog
from typing import Any, Dict, Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap

from engine import (
    AIR_AFTER_MAX_MM,
    LENS_SEMI_MAX_MM,
    LENS_Z_MAX_MM,
    LENS_Z_MIN_MM,
    MATERIAL_NAMES,
    MAX_ELEMENTS,
    SOURCE_DIE_MAX_MM,
    SOURCE_DIE_MIN_MM,
    apply_circular_outline,
    blank_element,
    copy_element,
    default_blocker,
    default_params,
    pad_elements,
    run_simulation,
    SimResult,
    target_plane_hits,
)
from progressive import map_ray_budget, run_simulation_progressive
from calibrate import apply_calibrated_n, calibrate_n_from_laser, laser_calibration_guide
from view_3d import open_isometric_view
from materials_catalog import (
    VISIBLE_NM_DEFAULT,
    VISIBLE_NM_MAX,
    VISIBLE_NM_MIN,
    material_id_from_name,
    material_name_from_id,
)
from lens_shapes import (
    SHAPE_DESCRIPTIONS,
    apply_shape,
    shape_dropdown_values,
    shape_id_from_label,
    shape_label_from_id,
)
from export_cad import (
    CNC_PROFILE_EXT,
    build_lens_specs_from_params,
    export_lens,
    export_lens_cnc_profile,
    export_polish_lap,
    tessellation_for_specs,
    tessellation_from_tolerance,
)
from rect_fov import (
    design_biconic_singlet_for_rect_fov,
    design_crossed_cylinders_for_rect_fov,
    fov_aspect,
    swap_anamorphic_xy_params,
    set_fov_from_aspect,
)
from optimizer import OptimizeConfig, config_from_panel, optimize_fov_flux
from focus import focus_group_on_target, group_z_bounds, map_half_covering_fov


# ── Theme ────────────────────────────────────────────────────────────────────

BG = "#0b0f14"
BG2 = "#111820"
BG3 = "#17202b"
FG = "#94a3b8"
FG_BRIGHT = "#e2e8f0"
ACCENT = "#38bdf8"
SOURCE = "#fbbf24"
LENS = "#5eead4"
FOV = "#a78bfa"
TARGET = "#f472b6"
BLOCKER = "#64748b"
BORDER = "#1e2a3a"

# How far the user can zoom out relative to the default “full scene” frame.
# Previously capped at ~1.05×, which made scroll-out feel stuck at the fit view.
ZOOM_OUT_MAX = 12.0  # 12× the auto-fit width/height (side view)
# Target plane: a large virtual wall so zoom-out is not stuck at the map
# rectangle (default ±40–50 mm). Recorded map grows to cover the view.
TGT_WALL_HALF_MM = 2500.0  # ±2.5 m
MAP_HALF_MAX_MM = 5000.0  # slider / auto-grow cap
# Zoom is a camera. Growing the recorded heatmap to the virtual wall
# coarsens bins (individual rays vanish on zoom-in) and used to inflate
# collection to “everything on the wall.”
AUTO_EXPAND_MAP_ON_ZOOM = False


def target_zoom_extent(
    map_hw: float,
    map_hh: float,
    wall_half: float = TGT_WALL_HALF_MM,
) -> tuple:
    """Allowed data limits when zooming the target plane (virtual wall)."""
    hw = max(float(map_hw), float(wall_half))
    hh = max(float(map_hh), float(wall_half))
    return (-hw, hw, -hh, hh)


def map_half_to_cover_view(
    xlim: tuple,
    ylim: tuple,
    cur_hw: float,
    cur_hh: float,
    *,
    pad: float = 1.05,
    max_half: float = MAP_HALF_MAX_MM,
) -> tuple:
    """
    Grow recorded map half-sizes so the current view is inside the heatmap.

    Never shrinks. Returns (new_hw, new_hh, grew).
    """
    vx = max(abs(float(xlim[0])), abs(float(xlim[1]))) * float(pad)
    vy = max(abs(float(ylim[0])), abs(float(ylim[1]))) * float(pad)
    new_w = min(float(max_half), max(float(cur_hw), vx))
    new_h = min(float(max_half), max(float(cur_hh), vy))
    grew = new_w > float(cur_hw) + 0.5 or new_h > float(cur_hh) + 0.5
    return new_w, new_h, grew

IRRAD_CMAP = LinearSegmentedColormap.from_list(
    "optiflux",
    [
        (0.0, "#080818"),
        (0.25, "#302090"),
        (0.5, "#20b0d0"),
        (0.75, "#f0d030"),
        (1.0, "#ffffff"),
    ],
)


class LiveEditGate:
    """Tracks a held slider / plot handle so traces start only on release."""

    def __init__(self):
        self.depth = 0
        self.cancel_gen = 0

    @property
    def live(self) -> bool:
        return self.depth > 0

    def begin(self) -> int:
        self.depth += 1
        self.cancel_gen += 1
        return self.cancel_gen

    def end(self) -> bool:
        """Return True if this call closed the last live interaction."""
        if self.depth <= 0:
            return False
        self.depth -= 1
        return self.depth == 0


def should_start_trace(*, auto_run: bool, live: bool, handle_drag: bool) -> bool:
    """Ray-trace only when Auto-run is on and the pointer is not held down."""
    return bool(auto_run) and not bool(live) and not bool(handle_drag)


def group_slider_limits(params: Dict[str, Any]) -> tuple:
    """Z range for the under-plot group bar: stack stays between source and target."""
    return group_z_bounds(params)


def should_draw_display_rays(*, auto_run: bool, dragging: bool, preview: bool) -> bool:
    """
    Individual ray polylines are the side-view cost (thousands of ax.plot calls).

    * Dragging a handle never draws them.
    * A live preview with Auto-run off never draws them.
    * A full redraw after an explicit Trace still draws them (preview=False).
    """
    if dragging:
        return False
    if preview:
        return bool(auto_run)
    return True


def side_full_extent(z0: float, z1: float, t_y: float, t_x: float) -> tuple:
    """(zmin, zmax, ymin, ymax, xmin, xmax) for the two stacked side views."""
    ty = max(float(t_y), 1e-6)
    tx = max(float(t_x), 1e-6)
    return (float(z0), float(z1), -ty, ty, -tx, tx)


def resolve_applied_side_limits(
    *,
    stored_xlim: Optional[tuple],
    stored_ylim: Optional[tuple],
    zmin: float,
    zmax: float,
    tmin: float,
    tmax: float,
    clamp_fn=None,
) -> tuple:
    """
    Limits for one side-view plane.

    A missing ylim must not reset the shared Z window — that was collapsing
    Y after every progressive batch (X–Z ylim stayed None, set_xlim(full)
    ran on the shared axis, and the zoomed Y window looked paper-thin).
    """
    clamp = clamp_fn
    if stored_xlim is None:
        xlim = (float(zmin), float(zmax))
    else:
        x0, x1 = float(stored_xlim[0]), float(stored_xlim[1])
        if clamp is not None:
            x0, x1 = clamp(x0, x1, zmin, zmax)
        xlim = (x0, x1)
    if stored_ylim is None:
        ylim = (float(tmin), float(tmax))
    else:
        y0, y1 = float(stored_ylim[0]), float(stored_ylim[1])
        if clamp is not None:
            y0, y1 = clamp(y0, y1, tmin, tmax)
        ylim = (y0, y1)
    return xlim, ylim


FOV_SIZE_MIN_MM = 5.0
FOV_SIZE_MAX_MM = 500.0  # matches engine.FOV_SIZE_MAX_MM


def clamp_fov_size(v: float) -> float:
    return max(FOV_SIZE_MIN_MM, min(FOV_SIZE_MAX_MM, float(v)))


def resize_fov_from_end(center: float, size: float, new_end: float) -> tuple:
    """Keep the FOV centre; set the span from a dragged bar end. Returns (center, size)."""
    return float(center), clamp_fov_size(2.0 * abs(float(new_end) - float(center)))


def move_fov_center(cx: float, cy: float, dx: float, dy: float) -> tuple:
    return float(cx) + float(dx), float(cy) + float(dy)


def resize_fov_rect(
    cx: float,
    cy: float,
    w: float,
    h: float,
    *,
    edge: Optional[str] = None,
    corner: Optional[str] = None,
    x: float = 0.0,
    y: float = 0.0,
    lock_aspect: bool = False,
) -> tuple:
    """
    Resize the target FOV rectangle.

    Edge drags keep the opposite edge. Corner drags keep the opposite corner.
    Returns (cx, cy, w, h).
    """
    x0 = float(cx) - float(w) / 2.0
    x1 = float(cx) + float(w) / 2.0
    y0 = float(cy) - float(h) / 2.0
    y1 = float(cy) + float(h) / 2.0
    aspect = (x1 - x0) / max(y1 - y0, 1e-9)
    kind = (corner or edge or "").lower()
    if kind in ("left", "nw", "sw"):
        x0 = float(x)
    if kind in ("right", "ne", "se"):
        x1 = float(x)
    if kind in ("bottom", "sw", "se"):
        y0 = float(y)
    if kind in ("top", "nw", "ne"):
        y1 = float(y)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    nw = clamp_fov_size(x1 - x0)
    nh = clamp_fov_size(y1 - y0)
    ncx = 0.5 * (x0 + x1)
    ncy = 0.5 * (y0 + y1)
    # After clamp, re-anchor so the unmoved edge/corner stays put.
    if kind in ("right", "ne", "se"):
        ncx = x0 + 0.5 * nw
    elif kind in ("left", "nw", "sw"):
        ncx = x1 - 0.5 * nw
    if kind in ("top", "nw", "ne"):
        ncy = y0 + 0.5 * nh
    elif kind in ("bottom", "sw", "se"):
        ncy = y1 - 0.5 * nh
    if lock_aspect and aspect > 1e-9 and kind in ("left", "right", "top", "bottom"):
        if kind in ("left", "right"):
            nh = clamp_fov_size(nw / aspect)
        else:
            nw = clamp_fov_size(nh * aspect)
    return ncx, ncy, nw, nh


def pick_fov_rect_handle(
    x: float,
    y: float,
    cx: float,
    cy: float,
    w: float,
    h: float,
    *,
    hit_r: float = 2.0,
) -> Optional[tuple]:
    """
    Return ('corner', name), ('edge', name), ('move', None), or None.
    Corners win over edges.
    """
    x0 = float(cx) - float(w) / 2.0
    x1 = float(cx) + float(w) / 2.0
    y0 = float(cy) - float(h) / 2.0
    y1 = float(cy) + float(h) / 2.0
    r = max(float(hit_r), 0.4)
    corners = {
        "sw": (x0, y0),
        "se": (x1, y0),
        "nw": (x0, y1),
        "ne": (x1, y1),
    }
    best = None
    best_d = r
    for name, (px, py) in corners.items():
        d = math.hypot(float(x) - px, float(y) - py)
        if d <= best_d:
            best_d = d
            best = ("corner", name)
    if best is not None:
        return best
    # Edges: near the rectangle boundary, away from the opposite side
    if abs(float(x) - x1) <= r and y0 - r <= float(y) <= y1 + r:
        return ("edge", "right")
    if abs(float(x) - x0) <= r and y0 - r <= float(y) <= y1 + r:
        return ("edge", "left")
    if abs(float(y) - y1) <= r and x0 - r <= float(x) <= x1 + r:
        return ("edge", "top")
    if abs(float(y) - y0) <= r and x0 - r <= float(x) <= x1 + r:
        return ("edge", "bottom")
    if x0 < float(x) < x1 and y0 < float(y) < y1:
        return ("move", None)
    return None


class SliderRow(ttk.Frame):
    """Label + [−] scale [+] + numeric entry bound to a DoubleVar / IntVar."""

    def __init__(
        self,
        parent,
        label: str,
        var: tk.Variable,
        from_: float,
        to: float,
        resolution: float = 0.1,
        is_int: bool = False,
        command=None,
        affects_trace: bool = True,
    ):
        super().__init__(parent)
        self.var = var
        self.is_int = is_int
        self.from_ = float(from_)
        self.to = float(to)
        self.resolution = float(resolution) if resolution else 1.0
        self._command = command
        self.affects_trace = bool(affects_trace)
        self._dragging = False
        self.columnconfigure(1, weight=1)

        ttk.Label(self, text=label, style="Dim.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(4, 0)
        )
        ttk.Button(self, text="−", width=3, style="Step.TButton", command=self._nudge_down).grid(
            row=1, column=0, sticky="w", padx=(0, 2)
        )
        self.scale = ttk.Scale(
            self,
            from_=from_,
            to=to,
            variable=var,
            orient="horizontal",
            command=self._on_scale,
        )
        self.scale.grid(row=1, column=1, sticky="ew", padx=2)
        self.scale.bind("<ButtonPress-1>", self._on_press, add="+")
        self.scale.bind("<ButtonRelease-1>", self._on_release, add="+")
        ttk.Button(self, text="+", width=3, style="Step.TButton", command=self._nudge_up).grid(
            row=1, column=2, sticky="w", padx=(2, 4)
        )
        self.entry = ttk.Entry(self, width=8, justify="right")
        self.entry.grid(row=1, column=3, sticky="e")
        self.entry.insert(0, self._fmt(var.get()))
        self.entry.bind("<Return>", self._on_entry)
        self.entry.bind("<FocusOut>", self._on_entry)
        var.trace_add("write", self._on_var)

    def _fmt(self, v) -> str:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return "0"
        if self.is_int:
            return str(int(round(v)))
        if abs(v) >= 100 or (abs(v) >= 1 and abs(v - round(v)) < 1e-9):
            return f"{v:.4g}"
        return f"{v:.4g}"

    def _step(self) -> float:
        """
        One nudge step. Default is 1 unit (mm, degrees, …).
        Very fine controls (asphere coeffs, n over a small span) keep their resolution.
        """
        if self.is_int:
            return max(1.0, float(self.resolution))
        span = self.to - self.from_
        if self.resolution < 0.05 and span <= 5.0:
            return self.resolution
        if self.resolution >= 1.0:
            return self.resolution
        return 1.0

    def _nudge(self, direction: int):
        step = self._step() * direction
        try:
            cur = float(self.var.get())
        except (TypeError, ValueError, tk.TclError):
            cur = self.from_
        v = cur + step
        v = max(self.from_, min(self.to, v))
        if self.is_int:
            v = int(round(v))
        self.var.set(v)
        self.entry.delete(0, tk.END)
        self.entry.insert(0, self._fmt(v))
        if self._command:
            self._command()

    def _nudge_down(self):
        self._nudge(-1)

    def _nudge_up(self):
        self._nudge(1)

    def _host_app(self):
        w = self.winfo_toplevel()
        return w if hasattr(w, "_begin_live_edit") else None

    def _on_press(self, _event=None):
        self._dragging = True
        if not self.affects_trace:
            return
        app = self._host_app()
        if app is not None:
            app._live_slider = self
            app._begin_live_edit()

    def _on_release(self, _event=None):
        self._finish_drag()

    def _finish_drag(self):
        if not self._dragging:
            return
        self._dragging = False
        app = self._host_app()
        if app is not None and getattr(app, "_live_slider", None) is self:
            app._live_slider = None
        if self.affects_trace and app is not None:
            app._end_live_edit()
        elif self._command:
            self._command()

    def _on_scale(self, _=None):
        self.entry.delete(0, tk.END)
        self.entry.insert(0, self._fmt(self.var.get()))
        if self._command:
            self._command()

    def _on_entry(self, _=None):
        try:
            v = float(self.entry.get())
            if self.is_int:
                v = int(round(v))
            v = max(self.from_, min(self.to, v))
            self.var.set(v)
        except ValueError:
            self.entry.delete(0, tk.END)
            self.entry.insert(0, self._fmt(self.var.get()))
        if self._command:
            self._command()

    def _on_var(self, *_):
        cur = self.entry.get()
        want = self._fmt(self.var.get())
        if cur != want:
            self.entry.delete(0, tk.END)
            self.entry.insert(0, want)


class OptiFluxApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OptiFlux — LED / COB Lens Designer")
        self.geometry("1480x900")
        self.minsize(1100, 700)
        self.configure(bg=BG)

        self.params = default_params()
        self.blockers: list = list(self.params.get("blockers") or [])
        self.result: Optional[SimResult] = None
        self._run_lock = threading.Lock()
        self._running = False
        self._trace_gen = 0  # bumps on every new progressive run (cancels prior)
        self._debounce_id = None
        self._live_edit = LiveEditGate()
        self._live_slider = None
        self._opt_gen = 0  # bumps to cancel an in-flight optimizer
        self._optimizing = False
        self._focusing = False
        # Progressive defaults: 5 batches × 5000 map rays + 500 side paths
        self.prog_batches = 5
        self.prog_rays_batch = 5000
        self.prog_disp_batch = 500
        self.auto_run = tk.BooleanVar(value=True)
        self.log_scale = tk.BooleanVar(value=False)
        # Side-view lens drag (recalculate only on mouse release)
        self._drag: Optional[Dict[str, Any]] = None
        self._element_handles: list = []  # pick targets from last side-view draw
        self._blocker_handles: list = []  # absorb panel pick targets
        self._fov_side_handles: list = []  # FOV bar ends / body on side views
        self._fov_tgt_handles: list = []
        self._fov_tgt_handle_arts: list = []
        self._fov_rect_patch = None
        self._side_layout_done = False
        self._draw_side_preview = False  # True while Auto-run-off geometry refresh
        self._side_cid = {}  # matplotlib event connection ids
        self._view3d_win = None
        # Side-view zoom/pan (Z–Y mm)
        self._side_xlim: Optional[tuple] = None
        self._side_ylim: Optional[tuple] = None
        self._side_full_extent: Optional[tuple] = None  # (zmin,zmax,ymin,ymax,xmin,xmax)
        self._side_pan: Optional[Dict[str, Any]] = None
        # Target-plane view zoom/pan (data limits in mm)
        self._tgt_xlim: Optional[tuple] = None
        self._tgt_ylim: Optional[tuple] = None
        self._tgt_full_extent: Optional[tuple] = None  # heatmap (xmin,xmax,ymin,ymax)
        self._tgt_zoom_extent: Optional[tuple] = None  # virtual wall for zoom-out
        self._tgt_pan: Optional[Dict[str, Any]] = None
        self._tgt_expand_id = None

        self._init_style()
        self._build_vars()
        self._build_ui()
        # If the pointer leaves a ttk.Scale while dragging, the scale itself
        # may never see ButtonRelease — catch it globally.
        self.bind_all("<ButtonRelease-1>", self._on_global_slider_release, add="+")
        self.after(100, self._first_run)

    # ── Style ────────────────────────────────────────────────────────────

    def _init_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=BG2, foreground=FG, fieldbackground=BG3)
        style.configure("TFrame", background=BG2)
        style.configure("TLabel", background=BG2, foreground=FG)
        style.configure("Dim.TLabel", background=BG2, foreground=FG, font=("Segoe UI", 8))
        style.configure("Title.TLabel", background=BG, foreground=FG_BRIGHT, font=("Segoe UI", 14, "bold"))
        style.configure("Head.TLabel", background=BG2, foreground=FG_BRIGHT, font=("Segoe UI", 10, "bold"))
        style.configure("Metric.TLabel", background=BG3, foreground=FG_BRIGHT, font=("Consolas", 12, "bold"))
        style.configure("MetricDim.TLabel", background=BG3, foreground=FG, font=("Segoe UI", 8))
        style.configure("TLabelframe", background=BG2, foreground=FG_BRIGHT)
        style.configure("TLabelframe.Label", background=BG2, foreground=ACCENT, font=("Segoe UI", 9, "bold"))
        style.configure("TButton", background=BG3, foreground=FG_BRIGHT, padding=6)
        style.configure("Accent.TButton", background=ACCENT, foreground="#041018", padding=6)
        style.configure("TCheckbutton", background=BG2, foreground=FG)
        style.configure("TRadiobutton", background=BG2, foreground=FG)
        style.configure("TNotebook", background=BG2, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG3, foreground=FG, padding=(10, 4))
        style.map("TNotebook.Tab", background=[("selected", BG)], foreground=[("selected", ACCENT)])
        style.configure("Horizontal.TScale", background=BG2)
        style.configure("TEntry", fieldbackground=BG, foreground=FG_BRIGHT)
        # Readable combobox: black text on light field (named style — avoid breaking default maps)
        combo_bg = "#f3f4f6"
        combo_fg = "#000000"
        style.configure(
            "Readable.TCombobox",
            fieldbackground=combo_bg,
            background=combo_bg,
            foreground=combo_fg,
            arrowcolor="#111827",
            insertcolor=combo_fg,
        )
        style.map(
            "Readable.TCombobox",
            fieldbackground=[
                ("readonly", combo_bg),
                ("!disabled", combo_bg),
                ("disabled", "#d1d5db"),
            ],
            foreground=[
                ("readonly", combo_fg),
                ("!disabled", combo_fg),
                ("disabled", "#6b7280"),
            ],
            selectbackground=[("readonly", combo_bg), ("!disabled", combo_bg)],
            selectforeground=[("readonly", combo_fg), ("!disabled", combo_fg)],
            background=[("readonly", combo_bg), ("!disabled", combo_bg)],
            arrowcolor=[("readonly", "#111827"), ("!disabled", "#111827")],
        )
        # Also style default TCombobox (header preset, etc.)
        style.configure(
            "TCombobox",
            fieldbackground=combo_bg,
            background=combo_bg,
            foreground=combo_fg,
            arrowcolor="#111827",
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", combo_bg), ("!disabled", combo_bg)],
            foreground=[("readonly", combo_fg), ("!disabled", combo_fg)],
            selectbackground=[("readonly", combo_bg)],
            selectforeground=[("readonly", combo_fg)],
            background=[("readonly", combo_bg)],
        )
        # Popup list: black on white
        self.option_add("*TCombobox*Listbox.foreground", "#000000")
        self.option_add("*TCombobox*Listbox.background", "#ffffff")
        self.option_add("*TCombobox*Listbox.selectForeground", "#000000")
        self.option_add("*TCombobox*Listbox.selectBackground", "#93c5fd")
        style.configure(
            "Step.TButton",
            background=BG3,
            foreground=FG_BRIGHT,
            padding=(4, 1),
            font=("Segoe UI", 10, "bold"),
        )
        # Visible scrollbars on dark UI (default clam trough matches bg → invisible)
        style.configure(
            "Vertical.TScrollbar",
            background="#64748b",
            troughcolor="#020617",
            bordercolor="#1e293b",
            arrowcolor="#f1f5f9",
            lightcolor="#94a3b8",
            darkcolor="#334155",
            gripcount=0,
            width=14,
        )
        style.map(
            "Vertical.TScrollbar",
            background=[("active", ACCENT), ("pressed", "#0ea5e9"), ("!disabled", "#64748b")],
            arrowcolor=[("active", "#0b0f14"), ("!disabled", "#f1f5f9")],
        )
        style.configure(
            "Horizontal.TScrollbar",
            background="#64748b",
            troughcolor="#020617",
            arrowcolor="#f1f5f9",
            width=12,
        )

    def _make_combobox(self, parent, textvariable, values, width=28, command=None):
        """
        Create a readonly combobox that works reliably inside a scrollable Canvas.
        """
        cb = ttk.Combobox(
            parent,
            textvariable=textvariable,
            values=list(values),
            width=width,
            state="readonly",
            style="Readable.TCombobox",
        )
        if command is not None:
            cb.bind("<<ComboboxSelected>>", command)

        def _on_click(_event=None):
            # Ensure focus so the dropdown opens under the cursor on Windows
            cb.focus_set()

        cb.bind("<ButtonPress-1>", _on_click, add="+")
        return cb

    def _build_vars(self):
        s = self.params["source"]
        self.v_mode = tk.StringVar(value=s["mode"])
        self.v_rows = tk.IntVar(value=s["rows"])
        self.v_cols = tk.IntVar(value=s["cols"])
        self.v_pitch_x = tk.DoubleVar(value=s["pitch_x"])
        self.v_pitch_y = tk.DoubleVar(value=s["pitch_y"])
        self.v_die_w = tk.DoubleVar(value=s["die_width"])
        self.v_die_h = tk.DoubleVar(value=s["die_height"])
        self.v_source_z = tk.DoubleVar(value=s["source_z"])
        self.v_flux = tk.DoubleVar(value=s["flux_per_die"])
        self.v_wl = tk.DoubleVar(value=s["wavelength_nm"])
        self.v_half = tk.DoubleVar(value=s["half_angle_deg"])
        self.v_tilt_x = tk.DoubleVar(value=s["tilt_x"])
        self.v_tilt_y = tk.DoubleVar(value=s["tilt_y"])
        self.v_off_x = tk.DoubleVar(value=s["offset_x"])
        self.v_off_y = tk.DoubleVar(value=s["offset_y"])
        self.v_rot_z = tk.DoubleVar(value=s["die_rot_z"])
        self.v_stagger = tk.BooleanVar(value=s["stagger"])
        self.v_circ = tk.BooleanVar(value=s["circular_mask"])
        self.v_mask_r = tk.DoubleVar(value=s["mask_radius"])

        self.v_lens_z = tk.DoubleVar(value=self.params["lens_z_start"])
        self.v_custom_n = tk.DoubleVar(value=self.params["custom_n"])
        self.v_fresnel = tk.BooleanVar(value=self.params["apply_fresnel"])
        self.v_tir_abs = tk.BooleanVar(value=self.params.get("absorb_on_tir", True))
        self.v_kill_back = tk.BooleanVar(value=self.params.get("kill_backward", True))
        self.v_mla = tk.BooleanVar(value=self.params.get("mla", {}).get("enabled", False))
        self.v_mla_fill = tk.DoubleVar(value=self.params.get("mla", {}).get("fill_factor", 0.88))
        self.v_mla_ap = tk.DoubleVar(value=self.params.get("mla", {}).get("lenslet_aperture", 0.0))
        self.v_export_plate = tk.BooleanVar(value=self.params.get("mla", {}).get("export_plate", True))
        self.v_cad_edge = tk.DoubleVar(value=float(self.params.get("cad_max_edge_mm", 0.25)))
        self.v_cad_angle = tk.DoubleVar(value=float(self.params.get("cad_max_angle_deg", 2.0)))
        self.v_cad_flange_r = tk.DoubleVar(value=float(self.params.get("cad_flange_radial_mm", 2.0)))
        self.v_cad_flange_t = tk.DoubleVar(value=float(self.params.get("cad_flange_thickness_mm", 1.5)))
        self.v_cad_halves = tk.BooleanVar(value=bool(self.params.get("cad_export_halves", False)))
        self.v_pol_dia = tk.DoubleVar(value=float(self.params.get("cad_polish_tool_dia_mm", 10.0)))
        self.v_pol_step = tk.DoubleVar(value=float(self.params.get("cad_polish_stepover_mm", 0.3)))
        self.v_pol_allow = tk.DoubleVar(value=float(self.params.get("cad_polish_allowance_mm", 0.0)))
        self.v_pol_feed = tk.DoubleVar(value=float(self.params.get("cad_polish_feed_mm_min", 100.0)))
        self.v_pol_retr = tk.DoubleVar(value=float(self.params.get("cad_polish_retract_mm", 5.0)))
        self.v_pol_rpm = tk.DoubleVar(value=float(self.params.get("cad_polish_rpm", 5000.0)))
        self.v_pol_revs = tk.DoubleVar(value=float(self.params.get("cad_polish_revs", 2.0)))
        self.v_pol_surf = tk.StringVar(value=str(self.params.get("cad_polish_surface", "front")))
        self.v_pol_strat = tk.StringVar(value=str(self.params.get("cad_polish_strategy", "helical")))
        self.v_lap_surf = tk.StringVar(value=str(self.params.get("cad_lap_surface", "front")))
        self.v_lap_wall = tk.DoubleVar(value=float(self.params.get("cad_lap_wall_mm", 6.0)))
        self.v_las_wl = tk.DoubleVar(value=float(self.params.get("source", {}).get("wavelength_nm", 650.0)))
        if self.v_las_wl.get() < 400:
            self.v_las_wl.set(650.0)
        self.v_las_z = tk.DoubleVar(value=float(self.params.get("source", {}).get("source_z", 0.0)))
        self.v_las_screen = tk.DoubleVar(value=float(self.params.get("target_z", 80.0)))
        self.v_las_hx = tk.DoubleVar(value=0.0)
        self.v_las_hy = tk.DoubleVar(value=5.0)
        self.v_las_cx = tk.DoubleVar(value=0.0)
        self.v_las_cy = tk.DoubleVar(value=0.0)
        self.v_las_sx = tk.DoubleVar(value=0.0)
        self.v_las_sy = tk.DoubleVar(value=0.0)
        self.laser_cal_status = tk.StringVar(
            value="On-axis first (alignment), then shift X/Y and enter the new dot."
        )
        self._laser_fit: dict = {}

        self.params["elements"] = pad_elements(self.params.get("elements") or [], MAX_ELEMENTS)
        self.elem_vars = []
        self.elem_ui = []  # collapsible panel state per element
        for e in self.params["elements"]:
            self.elem_vars.append(self._make_elem_vars(e))

        self.v_target_z = tk.DoubleVar(value=self.params["target_z"])
        self.v_fov_w = tk.DoubleVar(value=self.params["fov_width"])
        self.v_fov_h = tk.DoubleVar(value=self.params["fov_height"])
        self.v_fov_aspect = tk.DoubleVar(
            value=fov_aspect(self.params["fov_width"], self.params["fov_height"])
        )
        self.v_fov_lock = tk.BooleanVar(value=self.params.get("fov_aspect_lock", True))
        self.v_fov_cx = tk.DoubleVar(value=self.params["fov_cx"])
        self.v_fov_cy = tk.DoubleVar(value=self.params["fov_cy"])
        self.v_map_w = tk.DoubleVar(value=self.params["map_half_w"])
        self.v_map_h = tk.DoubleVar(value=self.params["map_half_h"])
        self.v_map_res = tk.IntVar(value=self.params["map_res"])
        self.v_rays = tk.IntVar(value=self.params["total_rays"])
        self.v_disp = tk.IntVar(value=self.params["display_rays"])

    # ── Layout ───────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        header = tk.Frame(self, bg=BG, height=48)
        header.pack(side="top", fill="x")
        header.pack_propagate(False)
        ttk.Label(header, text="OptiFlux", style="Title.TLabel").pack(side="left", padx=14, pady=10)
        ttk.Label(
            header,
            text="LED / COB Lens Designer  ·  physics ray tracer",
            background=BG,
            foreground=FG,
        ).pack(side="left", padx=4)

        ttk.Button(header, text="Trace rays", style="Accent.TButton", command=self.run_trace).pack(
            side="right", padx=10, pady=8
        )
        ttk.Button(header, text="3D view", command=self.open_3d_view).pack(
            side="right", padx=4, pady=8
        )
        ttk.Checkbutton(
            header,
            text="Auto-run",
            variable=self.auto_run,
            command=self._on_auto_run_toggle,
        ).pack(side="right", padx=6)
        ttk.Button(header, text="Export STL…", command=lambda: self.export_cad("stl")).pack(
            side="right", padx=4
        )
        ttk.Button(header, text="Export STEP…", command=lambda: self.export_cad("step")).pack(
            side="right", padx=4
        )
        # Design I/O — keep lens groups / full parameter sets
        design_bar = ttk.Frame(header)
        design_bar.pack(side="right", padx=6)
        ttk.Label(design_bar, text="Design:", style="Dim.TLabel").pack(side="left", padx=(0, 4))
        ttk.Button(design_bar, text="Save…", command=self.save_design).pack(side="left", padx=2)
        ttk.Button(design_bar, text="Load…", command=self.load_design).pack(side="left", padx=2)
        self._last_design_path: Optional[Path] = None
        ttk.Button(header, text="Reset defaults", command=self.reset_defaults).pack(side="right", padx=4)
        ttk.Button(header, text="Buy list…", command=self._show_buy_list_window).pack(
            side="right", padx=4
        )
        ttk.Button(header, text="Help", command=self._show_help).pack(side="right", padx=4)

        self.preset = tk.StringVar(value="")
        preset_cb = self._make_combobox(
            header,
            self.preset,
            [
                "",
                "Single LED",
                "COB 4×4",
                "Visible COB acrylic",
                "Formlabs Clear MLA",
                "Collimator",
                "Rect FOV · crossed cylinders",
                "Rect FOV · biconic singlet",
            ],
            width=22,
            command=self._on_preset,
        )
        preset_cb.pack(side="right", padx=8)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(header, textvariable=self.status_var, background=BG, foreground=FG).pack(
            side="right", padx=12
        )

        self.progress = ttk.Progressbar(header, mode="determinate", length=120)
        self.progress.pack(side="right", padx=4)

        # Body
        body = tk.Frame(self, bg=BG)
        body.pack(side="top", fill="both", expand=True)

        # Left controls (scrollable) — wheel only while pointer is over this panel
        # (bind_all MouseWheel breaks ttk.Combobox dropdowns on Windows)
        left_wrap = tk.Frame(body, bg=BG2, width=360)
        left_wrap.pack(side="left", fill="y")
        left_wrap.pack_propagate(False)
        self._left_wrap = left_wrap

        canvas = tk.Canvas(left_wrap, bg=BG2, highlightthickness=0, borderwidth=0)
        # tk.Scrollbar (not ttk) so trough/thumb stay visible on dark themes
        scroll = tk.Scrollbar(
            left_wrap,
            orient="vertical",
            command=canvas.yview,
            bg="#64748b",
            troughcolor="#020617",
            activebackground=ACCENT,
            highlightthickness=0,
            bd=0,
            width=14,
            relief="flat",
        )
        self.ctrl_frame = ttk.Frame(canvas)
        self._ctrl_canvas = canvas
        self._ctrl_scroll = scroll
        self._ctrl_window = canvas.create_window((0, 0), window=self.ctrl_frame, anchor="nw")
        self._section_ui: list = []

        def _on_frame_configure(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            # Keep embedded frame full width; leave room for the scrollbar strip
            canvas.itemconfigure(self._ctrl_window, width=max(event.width - 2, 200))

        self.ctrl_frame.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.configure(yscrollcommand=scroll.set)
        # Pack scrollbar first so it is never clipped by the canvas
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        def _wheel(event):
            # Only scroll the control panel (not when over plots)
            canvas.yview_scroll(int(-event.delta / 120), "units")
            return "break"

        def _pointer_in_left_panel(event) -> bool:
            try:
                w = left_wrap.winfo_containing(event.x_root, event.y_root)
            except tk.TclError:
                return False
            while w is not None:
                if w == left_wrap:
                    return True
                w = getattr(w, "master", None)
            return False

        def _bind_wheel(_event=None):
            canvas.bind_all("<MouseWheel>", _wheel)

        def _unbind_wheel(event=None):
            # Don't unbind when moving into a child of the left panel
            if event is not None and _pointer_in_left_panel(event):
                return
            canvas.unbind_all("<MouseWheel>")

        left_wrap.bind("<Enter>", _bind_wheel)
        left_wrap.bind("<Leave>", _unbind_wheel)
        canvas.bind("<MouseWheel>", _wheel)
        self.ctrl_frame.bind("<MouseWheel>", _wheel)

        self._build_controls(self.ctrl_frame)

        def _bind_wheel_recursive(widget):
            widget.bind("<MouseWheel>", _wheel)
            for child in widget.winfo_children():
                _bind_wheel_recursive(child)

        _bind_wheel_recursive(self.ctrl_frame)

        # Center views
        center = tk.Frame(body, bg=BG)
        center.pack(side="left", fill="both", expand=True)

        side_tools = ttk.Frame(center)
        side_tools.pack(side="top", fill="x", padx=8)
        ttk.Label(
            side_tools,
            text="Y–Z + X–Z: lens=centre/rim/purple · group bar=whole stack · FOV=violet · pan=right-drag · dbl-click=reset",
            style="Dim.TLabel",
        ).pack(side="left")
        ttk.Button(side_tools, text="Reset side zoom", command=self._reset_side_zoom).pack(
            side="right", padx=4
        )

        # Two stacked meridional cuts: Y–Z (top) and orthogonal X–Z (bottom).
        # cylinder_y curvature appears in Y–Z; cylinder_x in X–Z.
        self.fig_side = Figure(figsize=(6, 4.6), dpi=100, facecolor=BG)
        self.ax_side = self.fig_side.add_subplot(211)  # Y–Z
        self.ax_side_xz = self.fig_side.add_subplot(212, sharex=self.ax_side)  # X–Z
        self.canvas_side = FigureCanvasTkAgg(self.fig_side, master=center)
        side_widget = self.canvas_side.get_tk_widget()
        side_widget.pack(side="top", fill="both", expand=True, padx=2, pady=2)
        side_widget.configure(cursor="hand2")
        self._connect_side_mouse()
        # Optional separate zoom for the X–Z transverse axis
        self._side_xz_ylim: Optional[Tuple[float, float]] = None
        self._build_group_z_bar(center)

        bot = tk.Frame(center, bg=BG)
        bot.pack(side="top", fill="both", expand=True)

        tgt_tools = ttk.Frame(bot)
        tgt_tools.pack(side="top", fill="x", padx=8)
        ttk.Checkbutton(
            tgt_tools, text="Log irradiance", variable=self.log_scale, command=self._redraw
        ).pack(side="left")
        ttk.Button(tgt_tools, text="Reset zoom", command=self._reset_tgt_zoom).pack(
            side="right", padx=4
        )
        ttk.Label(
            tgt_tools,
            text="Scroll = zoom  ·  right-drag = pan  ·  double-click = reset",
            style="Dim.TLabel",
        ).pack(side="right", padx=8)

        # Bottom plots: profiles (left) | target plane (right)
        bot_plots = tk.Frame(bot, bg=BG)
        bot_plots.pack(side="top", fill="both", expand=True)

        prof_frame = tk.Frame(bot_plots, bg=BG)
        prof_frame.pack(side="left", fill="both", expand=True)
        self.fig_prof = Figure(figsize=(4.2, 3.2), dpi=100, facecolor=BG)
        self.ax_prof = self.fig_prof.add_subplot(111)
        self.canvas_prof = FigureCanvasTkAgg(self.fig_prof, master=prof_frame)
        self.canvas_prof.get_tk_widget().pack(side="top", fill="both", expand=True, padx=2, pady=2)

        tgt_frame = tk.Frame(bot_plots, bg=BG)
        tgt_frame.pack(side="left", fill="both", expand=True)
        self.fig_tgt = Figure(figsize=(5.0, 3.2), dpi=100, facecolor=BG)
        self.ax_tgt = self.fig_tgt.add_subplot(111)
        self.canvas_tgt = FigureCanvasTkAgg(self.fig_tgt, master=tgt_frame)
        tgt_widget = self.canvas_tgt.get_tk_widget()
        tgt_widget.pack(side="top", fill="both", expand=True, padx=2, pady=2)
        self._connect_target_mouse()

        # Right metrics
        right = tk.Frame(body, bg=BG2, width=260)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        self._build_metrics(right)

        self._style_axes()
        self._redraw_empty()

    def _make_elem_vars(self, e: dict) -> dict:
        r1y = e.get("R1y")
        r2y = e.get("R2y")
        apy = e.get("aperture_y")
        sid = e.get("shape_id", "custom")
        r_mag = max(abs(float(e.get("R1", 25))), abs(float(e.get("R2", 25))), 2.0)
        if r_mag < 1e-6:
            r_mag = 25.0
        return {
            "enabled": tk.BooleanVar(value=bool(e.get("enabled", False))),
            "shape": tk.StringVar(value=shape_label_from_id(sid)),
            "R_mag": tk.DoubleVar(value=r_mag),
            "R1": tk.DoubleVar(value=float(e.get("R1", 40.0))),
            "R2": tk.DoubleVar(value=float(e.get("R2", -50.0))),
            "R1y": tk.DoubleVar(
                value=float(r1y) if r1y is not None else float(e.get("R1", 40.0))
            ),
            "R2y": tk.DoubleVar(
                value=float(r2y) if r2y is not None else float(e.get("R2", -50.0))
            ),
            "thickness": tk.DoubleVar(value=float(e.get("thickness", 6.0))),
            "air_after": tk.DoubleVar(value=float(e.get("air_after", 2.0))),
            "aperture": tk.DoubleVar(value=float(e.get("aperture", 10.0))),
            "aperture_y": tk.DoubleVar(
                value=float(apy) if apy is not None else float(e.get("aperture", 10.0))
            ),
            "material": tk.StringVar(
                value=material_name_from_id(str(e.get("material", "ACRYLIC_PMMA")))
            ),
            "surface_mode": tk.StringVar(value=e.get("surface_mode", "rotational")),
            "k1": tk.DoubleVar(value=float(e.get("k1", 0.0))),
            "k2": tk.DoubleVar(value=float(e.get("k2", 0.0))),
            "A4_1": tk.DoubleVar(value=float(e.get("A4_1", 0.0))),
            "A4_2": tk.DoubleVar(value=float(e.get("A4_2", 0.0))),
            "circular_lock": tk.BooleanVar(value=bool(e.get("circular_lock", True))),
            "use_elliptical_ap": tk.BooleanVar(
                value=(apy is not None) and not bool(e.get("circular_lock", True))
            ),
            "use_biconic_radii": tk.BooleanVar(
                value=e.get("surface_mode", "rotational") != "rotational" or r1y is not None
            ),
            "copy_from": tk.StringVar(value="(none)"),
        }

    def _element_header_text(self, i: int, ev: dict) -> str:
        on = bool(ev["enabled"].get())
        mode = str(ev["surface_mode"].get())
        if on:
            return (
                f"Element {i + 1}  ·  ON  ·  {mode}  ·  "
                f"R1={float(ev['R1'].get()):.1f}  t={float(ev['thickness'].get()):.1f}"
            )
        return f"Element {i + 1}  ·  off  ·  click ▶ to expand"

    def _set_element_collapsed(self, i: int, collapsed: bool):
        if i < 0 or i >= len(self.elem_ui):
            return
        ui = self.elem_ui[i]
        ui["collapsed"] = bool(collapsed)
        if collapsed:
            ui["body"].pack_forget()
            ui["toggle_btn"].configure(text="▶")
        else:
            ui["body"].pack(fill="x", padx=2, pady=(0, 4))
            ui["toggle_btn"].configure(text="▼")
        ui["title"].configure(text=self._element_header_text(i, self.elem_vars[i]))

    def _toggle_element_collapsed(self, i: int):
        if i < 0 or i >= len(self.elem_ui):
            return
        self._set_element_collapsed(i, not self.elem_ui[i]["collapsed"])

    def _collapse_disabled_elements(self):
        for i, ev in enumerate(self.elem_vars):
            if not bool(ev["enabled"].get()):
                self._set_element_collapsed(i, True)

    def _expand_all_elements(self):
        for i in range(len(self.elem_ui)):
            self._set_element_collapsed(i, False)

    def _on_circular_lock(self, i: int):
        if i < 0 or i >= len(self.elem_vars):
            return
        ev = self.elem_vars[i]
        if ev.get("circular_lock") is not None and bool(ev["circular_lock"].get()):
            ev["use_elliptical_ap"].set(False)
            ev["aperture_y"].set(float(ev["aperture"].get()))
        self._on_param_change()

    def _on_elliptical_ap(self, i: int):
        if i < 0 or i >= len(self.elem_vars):
            return
        ev = self.elem_vars[i]
        if bool(ev["use_elliptical_ap"].get()) and ev.get("circular_lock") is not None:
            ev["circular_lock"].set(False)
        self._on_param_change()

    def _on_aperture_x(self, i: int):
        if i < 0 or i >= len(self.elem_vars):
            return
        ev = self.elem_vars[i]
        if ev.get("circular_lock") is not None and bool(ev["circular_lock"].get()):
            ev["use_elliptical_ap"].set(False)
            ev["aperture_y"].set(float(ev["aperture"].get()))
        self._on_param_change()

    def _on_copy_from_element(self, dest_index: int):
        """Copy another stack slot onto this element and enable it."""
        if dest_index < 0 or dest_index >= len(self.elem_vars):
            return
        ev = self.elem_vars[dest_index]
        label = str(ev.get("copy_from").get()) if ev.get("copy_from") else "(none)"
        if not label or label == "(none)":
            return
        try:
            src_index = int(label.split()[-1]) - 1
        except (ValueError, IndexError):
            ev["copy_from"].set("(none)")
            return
        if src_index == dest_index or src_index < 0 or src_index >= len(self.elem_vars):
            ev["copy_from"].set("(none)")
            return
        params = self.collect_params()
        elems = copy_element(params["elements"], src_index, dest_index)
        self._apply_element_dict_to_vars(dest_index, elems[dest_index])
        ev["enabled"].set(True)
        ev["copy_from"].set("(none)")
        self._set_element_collapsed(dest_index, collapsed=False)
        gap = float(elems[src_index].get("air_after", 0.0) or 0.0)
        self.status_var.set(
            f"Element {dest_index + 1} copied from Element {src_index + 1} "
            f"(air gap after source = {gap:.2f} mm)"
        )
        self._on_param_change()

    def _on_element_enabled(self, i: int):
        """Enable toggles: expand when turning on, collapse when turning off."""
        if i < 0 or i >= len(self.elem_vars):
            return
        on = bool(self.elem_vars[i]["enabled"].get())
        self._set_element_collapsed(i, collapsed=not on)
        self._on_param_change()

    def _on_material_selected(self, i: int):
        """Normalize material label to the single canonical catalog name."""
        if i < 0 or i >= len(self.elem_vars):
            return
        ev = self.elem_vars[i]
        raw = str(ev["material"].get())
        canon = material_name_from_id(material_id_from_name(raw))
        if canon != raw:
            ev["material"].set(canon)
        self._on_param_change()

    def _build_element_panel(self, parent, i: int, ev: dict):
        el = ttk.LabelFrame(parent, text="")
        el.pack(fill="x", padx=6, pady=3)

        header = ttk.Frame(el)
        header.pack(fill="x", padx=2, pady=2)
        toggle_btn = ttk.Button(
            header,
            text="▼",
            width=3,
            command=lambda idx=i: self._toggle_element_collapsed(idx),
        )
        toggle_btn.pack(side="left", padx=(2, 4))
        title = ttk.Label(header, text=self._element_header_text(i, ev), style="Dim.TLabel")
        title.pack(side="left", fill="x", expand=True)

        # Enable checkbox on its own row so the "Enabled" label is never clipped
        en_row = ttk.Frame(el)
        en_row.pack(fill="x", padx=6, pady=(0, 2))
        ttk.Checkbutton(
            en_row,
            text="Enabled  (include this element in the stack)",
            variable=ev["enabled"],
            command=lambda idx=i: self._on_element_enabled(idx),
        ).pack(anchor="w")

        body = ttk.Frame(el)
        ttk.Label(body, text="Copy from", style="Dim.TLabel").pack(
            anchor="w", padx=4, pady=(4, 0)
        )
        if "copy_from" not in ev:
            ev["copy_from"] = tk.StringVar(value="(none)")
        ev["copy_from"].set("(none)")
        copy_vals = ["(none)"] + [
            f"Element {j + 1}" for j in range(MAX_ELEMENTS) if j != i
        ]
        copy_cb = self._make_combobox(
            body,
            ev["copy_from"],
            copy_vals,
            width=32,
            command=lambda _e, idx=i: self._on_copy_from_element(idx),
        )
        copy_cb.pack(fill="x", padx=4, pady=2)
        ttk.Label(
            body,
            text="Copies shape, material, and apertures. The new lens sits after the previous air gap.",
            style="Dim.TLabel",
            wraplength=280,
        ).pack(anchor="w", padx=4, pady=(0, 4))
        body.pack(fill="x", padx=2, pady=(0, 4))

        ttk.Label(body, text="Lens type", style="Dim.TLabel").pack(anchor="w", padx=4, pady=(4, 0))
        shape_cb = self._make_combobox(
            body,
            ev["shape"],
            shape_dropdown_values(),
            width=32,
            command=lambda e, idx=i: self._on_element_shape_selected(idx),
        )
        shape_cb.pack(fill="x", padx=4, pady=2)

        SliderRow(
            body,
            "|R| magnitude (mm) for type",
            ev["R_mag"],
            2,
            200,
            0.5,
            command=lambda idx=i: self._on_element_r_mag(idx),
        ).pack(fill="x", padx=8, pady=1)

        ttk.Label(body, text="Material", style="Dim.TLabel").pack(anchor="w", padx=4, pady=(4, 0))
        # Canonical display name only — never a raw id or alternate spelling
        ev["material"].set(material_name_from_id(material_id_from_name(ev["material"].get())))
        mat_cb = self._make_combobox(
            body,
            ev["material"],
            MATERIAL_NAMES,
            width=32,
            command=lambda e, idx=i: self._on_material_selected(idx),
        )
        try:
            mat_cb.configure(state="readonly")
        except Exception:
            pass
        mat_cb.pack(fill="x", padx=4, pady=2)

        ttk.Label(body, text="Surface mode", style="Dim.TLabel").pack(anchor="w", padx=4, pady=(4, 0))
        mode_cb = self._make_combobox(
            body,
            ev["surface_mode"],
            ["rotational", "biconic", "cylinder_x", "cylinder_y"],
            width=32,
            command=lambda e: self._on_param_change(),
        )
        try:
            mode_cb.configure(state="readonly")
        except Exception:
            pass
        mode_cb.pack(fill="x", padx=4, pady=2)

        self._add_slider(body, "R₁ / Rₓ front (mm)", ev["R1"], -200, 200, 0.5)
        self._add_slider(body, "R₂ / Rₓ rear (mm)", ev["R2"], -200, 200, 0.5)
        self._add_slider(body, "R₁ᵧ front (mm) · biconic/cyl Y", ev["R1y"], -200, 200, 0.5)
        self._add_slider(body, "R₂ᵧ rear (mm) · biconic/cyl Y", ev["R2y"], -200, 200, 0.5)
        self._add_slider(body, "Thickness (mm)", ev["thickness"], 0.2, 30, 0.1)
        self._add_slider(body, "Air gap after (mm)", ev["air_after"], 0, AIR_AFTER_MAX_MM, 0.1)
        self._add_slider(
            body,
            "Semi-aperture X (mm)",
            ev["aperture"],
            1,
            80,
            0.1,
            command=lambda idx=i: self._on_aperture_x(idx),
        )
        self._add_slider(body, "Semi-aperture Y (mm)", ev["aperture_y"], 1, 80, 0.1)
        ttk.Checkbutton(
            body,
            text="Lock circular outline (Y = X) — tube can hold this lens",
            variable=ev["circular_lock"],
            command=lambda idx=i: self._on_circular_lock(idx),
        ).pack(anchor="w", padx=4)
        ttk.Checkbutton(
            body,
            text="Elliptical clear aperture (use X & Y)",
            variable=ev["use_elliptical_ap"],
            command=lambda idx=i: self._on_elliptical_ap(idx),
        ).pack(anchor="w", padx=4)
        self._add_slider(body, "Conic k₁", ev["k1"], -5, 5, 0.01)
        self._add_slider(body, "Conic k₂", ev["k2"], -5, 5, 0.01)
        self._add_slider(body, "Asphere A4₁", ev["A4_1"], -0.001, 0.001, 1e-6)
        self._add_slider(body, "Asphere A4₂", ev["A4_2"], -0.001, 0.001, 1e-6)

        self.elem_ui.append(
            {
                "frame": el,
                "header": header,
                "body": body,
                "title": title,
                "toggle_btn": toggle_btn,
                "collapsed": False,
            }
        )
        # Start collapsed when disabled
        if not bool(ev["enabled"].get()):
            self._set_element_collapsed(i, True)
        else:
            self._set_element_collapsed(i, False)

    def _add_slider(
        self,
        parent,
        label,
        var,
        lo,
        hi,
        res=0.1,
        is_int=False,
        command=None,
        affects_trace: bool = True,
    ):
        if command is False:
            cmd = None
            affects_trace = False
        elif command is None:
            cmd = self._on_param_change
        else:
            cmd = command
        row = SliderRow(
            parent,
            label,
            var,
            lo,
            hi,
            res,
            is_int,
            command=cmd,
            affects_trace=affects_trace,
        )
        row.pack(fill="x", padx=8, pady=1)
        return row

    def _refresh_ctrl_scrollregion(self):
        if hasattr(self, "_ctrl_canvas") and hasattr(self, "ctrl_frame"):
            try:
                self.ctrl_frame.update_idletasks()
                self._ctrl_canvas.configure(scrollregion=self._ctrl_canvas.bbox("all"))
            except tk.TclError:
                pass

    def _make_collapsible_section(
        self,
        parent,
        title: str,
        *,
        start_collapsed: bool = True,
    ):
        """
        Section header with ▶/▼ toggle (like lens elements). Returns the body frame.
        Defaults to collapsed so the left panel stays short at launch.
        """
        shell = ttk.Frame(parent)
        shell.pack(fill="x", padx=6, pady=3)
        hdr = tk.Frame(shell, bg=BG3, highlightbackground=BORDER, highlightthickness=1)
        hdr.pack(fill="x")
        toggle = ttk.Button(hdr, text="▶" if start_collapsed else "▼", width=3, style="Step.TButton")
        toggle.pack(side="left", padx=4, pady=4)
        ttk.Label(hdr, text=title, style="Head.TLabel").pack(
            side="left", fill="x", expand=True, padx=4, pady=4
        )
        body = ttk.Frame(shell)
        state = {
            "collapsed": bool(start_collapsed),
            "body": body,
            "btn": toggle,
            "title": title,
        }

        def _toggle(_event=None, st=state):
            st["collapsed"] = not st["collapsed"]
            if st["collapsed"]:
                st["body"].pack_forget()
                st["btn"].configure(text="▶")
            else:
                st["body"].pack(fill="x", padx=2, pady=(2, 4))
                st["btn"].configure(text="▼")
            self._refresh_ctrl_scrollregion()

        toggle.configure(command=_toggle)
        # Click header bar to toggle as well
        hdr.bind("<Button-1>", _toggle)
        for child in hdr.winfo_children():
            if child is not toggle:
                child.bind("<Button-1>", _toggle)

        if not start_collapsed:
            body.pack(fill="x", padx=2, pady=(2, 4))
        if not hasattr(self, "_section_ui") or self._section_ui is None:
            self._section_ui = []
        self._section_ui.append(state)
        return body

    def _build_blockers_panel(self, parent):
        """Left-panel UI for absorbing panels / tubes / aperture stops."""
        box = parent  # already a collapsible body
        ttk.Label(
            box,
            text=(
                "Absorbing enclosure geometry. "
                "Rect default = horizontal body (top/bottom/sides along Z — lens barrel). "
                "Circle = tube/pipe along the optical axis. "
                "Vertical = face-on stop / iris (normal to Z)."
            ),
            style="Dim.TLabel",
            wraplength=300,
        ).pack(anchor="w", padx=6, pady=2)

        list_fr = ttk.Frame(box)
        list_fr.pack(fill="x", padx=6, pady=2)
        self.blk_list = tk.Listbox(
            list_fr,
            height=4,
            bg=BG3,
            fg=FG_BRIGHT,
            selectbackground=ACCENT,
            selectforeground="#0b0f14",
            highlightthickness=0,
            borderwidth=1,
            font=("Segoe UI", 9),
        )
        self.blk_list.pack(side="left", fill="x", expand=True)
        self.blk_list.bind("<<ListboxSelect>>", self._on_blocker_select)
        scr = ttk.Scrollbar(list_fr, orient="vertical", command=self.blk_list.yview)
        scr.pack(side="right", fill="y")
        self.blk_list.configure(yscrollcommand=scr.set)

        btn_fr = ttk.Frame(box)
        btn_fr.pack(fill="x", padx=6, pady=2)
        ttk.Button(btn_fr, text="Add body", command=self._add_blocker_solid).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(btn_fr, text="Add tube", command=self._add_blocker_tube).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(btn_fr, text="Add stop", command=self._add_blocker_stop).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(btn_fr, text="Delete", command=self._delete_blocker).pack(side="left")

        # Edit fields for selected blocker
        self.v_blk_enabled = tk.BooleanVar(value=True)
        self.v_blk_label = tk.StringVar(value="")
        self.v_blk_shape = tk.StringVar(value="rect")
        self.v_blk_orient = tk.StringVar(value="horizontal")
        self.v_blk_z = tk.DoubleVar(value=20.0)
        self.v_blk_length = tk.DoubleVar(value=40.0)
        self.v_blk_ow = tk.DoubleVar(value=15.0)
        self.v_blk_oh = tk.DoubleVar(value=15.0)
        self.v_blk_iw = tk.DoubleVar(value=0.0)
        self.v_blk_ih = tk.DoubleVar(value=0.0)
        self.v_blk_x0 = tk.DoubleVar(value=0.0)
        self.v_blk_y0 = tk.DoubleVar(value=0.0)
        self._blk_syncing = False

        ttk.Checkbutton(
            box, text="Enabled", variable=self.v_blk_enabled, command=self._on_blocker_field_change
        ).pack(anchor="w", padx=6)
        lab_row = ttk.Frame(box)
        lab_row.pack(fill="x", padx=6, pady=1)
        ttk.Label(lab_row, text="Label", style="Dim.TLabel").pack(side="left")
        ent = ttk.Entry(lab_row, textvariable=self.v_blk_label, width=18)
        ent.pack(side="right", fill="x", expand=True, padx=4)
        ent.bind("<FocusOut>", lambda _e: self._on_blocker_field_change())
        ent.bind("<Return>", lambda _e: self._on_blocker_field_change())

        shape_row = ttk.Frame(box)
        shape_row.pack(fill="x", padx=6, pady=1)
        ttk.Label(shape_row, text="Shape", style="Dim.TLabel").pack(side="left")
        shape_cb = self._make_combobox(
            shape_row, self.v_blk_shape, ["rect", "circle"], width=10,
            command=self._on_blocker_field_change,
        )
        shape_cb.pack(side="right")

        orient_row = ttk.Frame(box)
        orient_row.pack(fill="x", padx=6, pady=1)
        ttk.Label(orient_row, text="Orient", style="Dim.TLabel").pack(side="left")
        orient_cb = self._make_combobox(
            orient_row,
            self.v_blk_orient,
            ["horizontal", "vertical", "tube"],
            width=12,
            command=self._on_blocker_field_change,
        )
        orient_cb.pack(side="right")
        ttk.Label(
            box,
            text="horizontal = body walls ‖ Z · vertical = face-on stop · tube = circular barrel",
            style="Dim.TLabel",
            wraplength=300,
        ).pack(anchor="w", padx=6)

        for lab, var, lo, hi in (
            ("Center Z (mm)", self.v_blk_z, -5, 500),
            ("Length along Z (mm)", self.v_blk_length, 1, 400),
            ("Outer half-W / radius (mm)", self.v_blk_ow, 0.5, 200),
            ("Outer half-H (mm, rect body)", self.v_blk_oh, 0.5, 200),
            ("Inner radius / hole half-W (mm)", self.v_blk_iw, 0, 150),
            ("Hole half-H (mm, vertical rect)", self.v_blk_ih, 0, 150),
            ("Decenter X (mm)", self.v_blk_x0, -50, 50),
            ("Decenter Y (mm)", self.v_blk_y0, -50, 50),
        ):
            row = SliderRow(
                box, lab, var, lo, hi, 0.1, False, command=self._on_blocker_field_change
            )
            row.pack(fill="x", padx=8, pady=1)

        self._refresh_blocker_listbox()
        if self.blockers:
            self.blk_list.selection_set(0)
            self._load_blocker_to_vars(0)

    def _blocker_list_label(self, b: dict, i: int) -> str:
        shape = str(b.get("shape", "rect"))
        orient = str(b.get("orient") or ("tube" if shape == "circle" else "horizontal"))
        lab = str(b.get("label") or f"Blocker {i + 1}")
        en = "ON" if b.get("enabled", True) else "off"
        L = float(b.get("length", b.get("thickness", 0)) or 0)
        return (
            f"{i + 1}. {lab}  [{shape}/{orient}] "
            f"Z={float(b.get('z', 0)):.1f} L={L:.0f}  {en}"
        )

    def _refresh_blocker_listbox(self):
        if not hasattr(self, "blk_list"):
            return
        sel = self.blk_list.curselection()
        idx = int(sel[0]) if sel else 0
        self.blk_list.delete(0, tk.END)
        for i, b in enumerate(self.blockers):
            self.blk_list.insert(tk.END, self._blocker_list_label(b, i))
        if self.blockers:
            idx = max(0, min(idx, len(self.blockers) - 1))
            self.blk_list.selection_set(idx)

    def _selected_blocker_index(self) -> Optional[int]:
        if not hasattr(self, "blk_list"):
            return None
        sel = self.blk_list.curselection()
        if not sel:
            return None
        i = int(sel[0])
        if 0 <= i < len(self.blockers):
            return i
        return None

    def _load_blocker_to_vars(self, i: int):
        if i < 0 or i >= len(self.blockers):
            return
        b = self.blockers[i]
        shape = str(b.get("shape") or "rect")
        orient = str(b.get("orient") or ("tube" if shape == "circle" else "horizontal"))
        self._blk_syncing = True
        try:
            self.v_blk_enabled.set(bool(b.get("enabled", True)))
            self.v_blk_label.set(str(b.get("label") or f"Blocker {i + 1}"))
            self.v_blk_shape.set(shape)
            if hasattr(self, "v_blk_orient"):
                self.v_blk_orient.set(orient)
            self.v_blk_z.set(float(b.get("z", 20.0)))
            if hasattr(self, "v_blk_length"):
                self.v_blk_length.set(
                    float(b.get("length", b.get("thickness", 40.0)) or 40.0)
                )
            self.v_blk_ow.set(float(b.get("outer_w", 15.0)))
            self.v_blk_oh.set(float(b.get("outer_h", 15.0)))
            self.v_blk_iw.set(float(b.get("inner_w", 0.0)))
            self.v_blk_ih.set(float(b.get("inner_h", 0.0)))
            self.v_blk_x0.set(float(b.get("x0", 0.0)))
            self.v_blk_y0.set(float(b.get("y0", 0.0)))
        finally:
            self._blk_syncing = False

    def _write_vars_to_blocker(self, i: int):
        if i < 0 or i >= len(self.blockers):
            return
        shape = str(self.v_blk_shape.get() or "rect")
        orient = (
            str(self.v_blk_orient.get() or "horizontal")
            if hasattr(self, "v_blk_orient")
            else ("tube" if shape == "circle" else "horizontal")
        )
        length = (
            float(self.v_blk_length.get())
            if hasattr(self, "v_blk_length")
            else float(self.blockers[i].get("length", 40.0) or 40.0)
        )
        self.blockers[i] = {
            "enabled": bool(self.v_blk_enabled.get()),
            "label": str(self.v_blk_label.get() or f"Blocker {i + 1}"),
            "z": float(self.v_blk_z.get()),
            "shape": shape,
            "orient": orient,
            "length": length,
            "outer_w": float(self.v_blk_ow.get()),
            "outer_h": float(self.v_blk_oh.get()),
            "inner_w": float(self.v_blk_iw.get()),
            "inner_h": float(self.v_blk_ih.get()),
            "x0": float(self.v_blk_x0.get()),
            "y0": float(self.v_blk_y0.get()),
            "thickness": float(self.blockers[i].get("thickness", 1.0) or 1.0),
        }

    def _on_blocker_select(self, *_):
        i = self._selected_blocker_index()
        if i is not None:
            self._load_blocker_to_vars(i)

    def _on_blocker_field_change(self, *_):
        if getattr(self, "_blk_syncing", False):
            return
        i = self._selected_blocker_index()
        if i is None and self.blockers:
            i = 0
            self.blk_list.selection_set(0)
        if i is None:
            return
        self._write_vars_to_blocker(i)
        self._refresh_blocker_listbox()
        self.blk_list.selection_set(i)
        self._on_param_change()

    def _add_blocker_solid(self):
        """Rectangular enclosure body (horizontal walls along Z)."""
        z = float(self.v_target_z.get()) * 0.35 if hasattr(self, "v_target_z") else 25.0
        b = default_blocker(
            z=z,
            shape="rect",
            orient="horizontal",
            outer_w=18,
            outer_h=14,
            length=50,
            label=f"Body {len(self.blockers) + 1}",
        )
        self.blockers.append(b)
        self._refresh_blocker_listbox()
        self.blk_list.selection_clear(0, tk.END)
        self.blk_list.selection_set(len(self.blockers) - 1)
        self._load_blocker_to_vars(len(self.blockers) - 1)
        self._on_param_change()

    def _add_blocker_tube(self):
        """Circular tube / lens barrel / snoot along optical axis."""
        z = float(self.v_lens_z.get()) + 20.0 if hasattr(self, "v_lens_z") else 30.0
        b = default_blocker(
            z=z,
            shape="circle",
            orient="tube",
            outer_w=16,
            outer_h=16,
            inner_w=0,
            length=60,
            label=f"Tube {len(self.blockers) + 1}",
        )
        self.blockers.append(b)
        self._refresh_blocker_listbox()
        self.blk_list.selection_clear(0, tk.END)
        self.blk_list.selection_set(len(self.blockers) - 1)
        self._load_blocker_to_vars(len(self.blockers) - 1)
        self._on_param_change()

    def _add_blocker_stop(self):
        """Face-on vertical aperture stop (normal to Z)."""
        z = float(self.v_lens_z.get()) + 12.0 if hasattr(self, "v_lens_z") else 25.0
        b = default_blocker(
            z=z,
            shape="circle",
            orient="vertical",
            outer_w=20,
            outer_h=20,
            inner_w=8,
            inner_h=8,
            length=1,
            label=f"Stop {len(self.blockers) + 1}",
        )
        b["thickness"] = 1.0
        self.blockers.append(b)
        self._refresh_blocker_listbox()
        self.blk_list.selection_clear(0, tk.END)
        self.blk_list.selection_set(len(self.blockers) - 1)
        self._load_blocker_to_vars(len(self.blockers) - 1)
        self._on_param_change()

    def _delete_blocker(self):
        i = self._selected_blocker_index()
        if i is None:
            return
        del self.blockers[i]
        self._refresh_blocker_listbox()
        if self.blockers:
            ni = min(i, len(self.blockers) - 1)
            self.blk_list.selection_set(ni)
            self._load_blocker_to_vars(ni)
        self._on_param_change()

    def open_3d_view(self):
        """Open / focus the interactive isometric 3D layout window."""
        try:
            if self._view3d_win is not None and self._view3d_win.winfo_exists():
                self._view3d_win.lift()
                if hasattr(self._view3d_win, "_optiflux_3d"):
                    self._view3d_win._optiflux_3d["refresh"]()
                return
        except Exception:
            self._view3d_win = None
        self._view3d_win = open_isometric_view(
            self,
            self.collect_params(),
            self.result,
            get_params=self.collect_params,
            get_result=lambda: self.result,
        )

    def _build_controls(self, parent):
        self._section_ui = []

        # Source
        src = self._make_collapsible_section(
            parent, "SOURCE  ·  LED / COB", start_collapsed=True
        )

        mode_f = ttk.Frame(src)
        mode_f.pack(fill="x", padx=6, pady=4)
        ttk.Radiobutton(
            mode_f, text="Single LED", value="single", variable=self.v_mode, command=self._on_param_change
        ).pack(side="left")
        ttk.Radiobutton(
            mode_f, text="COB array", value="cob", variable=self.v_mode, command=self._on_param_change
        ).pack(side="left", padx=8)

        ttk.Label(
            src,
            text="Dies are rectangular surface emitters (not point sources).",
            style="Dim.TLabel",
            wraplength=290,
        ).pack(anchor="w", padx=8)

        self._add_slider(src, "Rows", self.v_rows, 1, 16, 1, True)
        self._add_slider(src, "Columns", self.v_cols, 1, 16, 1, True)
        self._add_slider(src, "Pitch X (mm)", self.v_pitch_x, 0.3, 40, 0.05)
        self._add_slider(src, "Pitch Y (mm)", self.v_pitch_y, 0.3, 40, 0.05)
        self._add_slider(src, "Die width (mm)", self.v_die_w, SOURCE_DIE_MIN_MM, SOURCE_DIE_MAX_MM, 0.1)
        self._add_slider(src, "Die height (mm)", self.v_die_h, SOURCE_DIE_MIN_MM, SOURCE_DIE_MAX_MM, 0.1)
        self._add_slider(src, "Source plane Z (mm)", self.v_source_z, -20, 20, 0.1)
        self._add_slider(src, "Flux per die (a.u.)", self.v_flux, 0.1, 10, 0.1)
        ttk.Label(
            src,
            text=f"Visible band only: {VISIBLE_NM_MIN:.0f}–{VISIBLE_NM_MAX:.0f} nm (CIE-style).",
            style="Dim.TLabel",
            wraplength=290,
        ).pack(anchor="w", padx=8)
        self._add_slider(src, "Wavelength (nm, visible)", self.v_wl, VISIBLE_NM_MIN, VISIBLE_NM_MAX, 1)
        self._add_slider(src, "Emission half-angle (°)", self.v_half, 5, 90, 1)
        self._add_slider(src, "Array tilt X (°)", self.v_tilt_x, -30, 30, 0.5)
        self._add_slider(src, "Array tilt Y (°)", self.v_tilt_y, -30, 30, 0.5)
        self._add_slider(src, "Array offset X (mm)", self.v_off_x, -15, 15, 0.1)
        self._add_slider(src, "Array offset Y (mm)", self.v_off_y, -15, 15, 0.1)
        self._add_slider(src, "Die rotation Z (°)", self.v_rot_z, -45, 45, 1)
        ttk.Checkbutton(src, text="Stagger odd rows", variable=self.v_stagger, command=self._on_param_change).pack(
            anchor="w", padx=8
        )
        ttk.Checkbutton(
            src, text="Circular COB mask", variable=self.v_circ, command=self._on_param_change
        ).pack(anchor="w", padx=8)
        self._add_slider(src, "Mask radius (mm)", self.v_mask_r, 0.5, 40, 0.1)

        # Optics
        opt = self._make_collapsible_section(
            parent, "OPTICS  ·  lens stack", start_collapsed=True
        )

        mla_box = ttk.LabelFrame(opt, text="Micro-lens array (COB match)")
        mla_box.pack(fill="x", padx=6, pady=4)
        ttk.Label(
            mla_box,
            text="Replicate Element 1 (type, R, material) as one lenslet per COB die.",
            style="Dim.TLabel",
            wraplength=290,
        ).pack(anchor="w", padx=6, pady=2)
        ttk.Label(
            mla_box,
            text="Element 1 type/R/material define each micro-lens. Geometry is scaled "
            "to die pitch so a bi-convex stays bi-convex (not flat cylinders).",
            style="Dim.TLabel",
            wraplength=300,
        ).pack(anchor="w", padx=6, pady=2)
        ttk.Checkbutton(
            mla_box,
            text="Enable MLA (one lens per die)",
            variable=self.v_mla,
            command=self._on_param_change,
        ).pack(anchor="w", padx=6)
        self.v_mla_scale = tk.BooleanVar(
            value=self.params.get("mla", {}).get("scale_to_pitch", True)
        )
        ttk.Checkbutton(
            mla_box,
            text="Scale Element 1 geometry to die pitch (recommended)",
            variable=self.v_mla_scale,
            command=self._on_param_change,
        ).pack(anchor="w", padx=6)
        self.v_mla_aim = tk.BooleanVar(
            value=self.params.get("mla", {}).get("aim_to_fov", True)
        )
        self.v_mla_aim_s = tk.DoubleVar(
            value=float(self.params.get("mla", {}).get("aim_strength", 1.0))
        )
        ttk.Checkbutton(
            mla_box,
            text="Aim each lenslet at FOV center (common spot)",
            variable=self.v_mla_aim,
            command=self._on_param_change,
        ).pack(anchor="w", padx=6)
        self._add_slider(
            mla_box, "Aim strength (0=none, 1=full)", self.v_mla_aim_s, 0.0, 1.5, 0.05
        )
        ttk.Label(
            mla_box,
            text=(
                "Per-die emission tilt + optical-center offset toward FOV "
                "center; offset clamped so lenslets stay inside pitch cells."
            ),
            style="Dim.TLabel",
            wraplength=300,
        ).pack(anchor="w", padx=6, pady=2)
        self._add_slider(mla_box, "Fill factor (aperture / pitch)", self.v_mla_fill, 0.4, 0.99, 0.01)
        self._add_slider(mla_box, "Lenslet semi-aperture mm (0=auto)", self.v_mla_ap, 0, 10, 0.05)
        ttk.Checkbutton(
            mla_box,
            text="Thicken MLA plate slightly on CAD export",
            variable=self.v_export_plate,
        ).pack(anchor="w", padx=6)

        cad_box = ttk.LabelFrame(opt, text="CAD export (units: mm)")
        cad_box.pack(fill="x", padx=6, pady=4)
        ttk.Label(
            cad_box,
            text=(
                "STL / STEP use the same aspheric sag as the ray tracer (mm). "
                "Finer vertex length / smaller facet angle = smoother 3D prints "
                "(larger files). Flanges are CAD-only (not traced). A "
                "_tube.txt file lists OD, flange thickness, and groove spacing."
            ),
            style="Dim.TLabel",
            wraplength=290,
        ).pack(anchor="w", padx=6, pady=2)
        self._add_slider(
            cad_box,
            "Max vertex length (mm)",
            self.v_cad_edge,
            0.05,
            2.0,
            0.01,
            command=self._refresh_cad_mesh_label,
            affects_trace=False,
        )
        self._add_slider(
            cad_box,
            "Max facet angle (°)",
            self.v_cad_angle,
            0.5,
            15.0,
            0.1,
            command=self._refresh_cad_mesh_label,
            affects_trace=False,
        )
        self._add_slider(
            cad_box,
            "Flange radial width (mm)",
            self.v_cad_flange_r,
            0.0,
            15.0,
            0.1,
            command=False,
        )
        self._add_slider(
            cad_box,
            "Flange thickness (mm)",
            self.v_cad_flange_t,
            0.4,
            8.0,
            0.1,
            command=False,
        )
        self.cad_mesh_status = tk.StringVar(value="")
        ttk.Label(
            cad_box,
            textvariable=self.cad_mesh_status,
            style="Dim.TLabel",
            wraplength=290,
        ).pack(anchor="w", padx=6, pady=(0, 4))
        self._refresh_cad_mesh_label()
        ttk.Checkbutton(
            cad_box,
            text="STL as print halves (cut at flange mid-plane)",
            variable=self.v_cad_halves,
        ).pack(anchor="w", padx=6, pady=(0, 2))
        ttk.Button(cad_box, text="Export STL (binary, mm)…", command=lambda: self.export_cad("stl")).pack(
            fill="x", padx=6, pady=2
        )
        ttk.Button(cad_box, text="Export STEP (mm)…", command=lambda: self.export_cad("step")).pack(
            fill="x", padx=6, pady=2
        )
        pol = ttk.LabelFrame(cad_box, text="4-axis polish (GRBL / Candle)")
        pol.pack(fill="x", padx=4, pady=4)
        ttk.Label(
            pol,
            text=(
                "Printed half on A (optical axis along X). "
                "Spherical abrasive on the vertical spindle approaches radially in Z. "
                "X0 = flange mid-plane; X+ toward the optical surface. Y = 0."
            ),
            style="Dim.TLabel",
            wraplength=270,
        ).pack(anchor="w", padx=6, pady=2)
        row_s = ttk.Frame(pol)
        row_s.pack(fill="x", padx=6, pady=2)
        ttk.Label(row_s, text="Face", style="Dim.TLabel").pack(side="left")
        ttk.Combobox(
            row_s,
            textvariable=self.v_pol_surf,
            values=("front", "rear", "both"),
            state="readonly",
            width=8,
        ).pack(side="left", padx=4)
        ttk.Label(row_s, text="Path", style="Dim.TLabel").pack(side="left", padx=(8, 0))
        ttk.Combobox(
            row_s,
            textvariable=self.v_pol_strat,
            values=("helical", "rings"),
            state="readonly",
            width=8,
        ).pack(side="left", padx=4)
        self._add_slider(pol, "Ball tool diameter (mm)", self.v_pol_dia, 2.0, 30.0, 0.1, command=False)
        self._add_slider(pol, "Stepover along meridian (mm)", self.v_pol_step, 0.05, 2.0, 0.05, command=False)
        self._add_slider(pol, "Allowance / extra offset (mm)", self.v_pol_allow, -0.2, 0.5, 0.01, command=False)
        self._add_slider(pol, "Feed (mm/min)", self.v_pol_feed, 20, 800, 5, command=False)
        self._add_slider(pol, "A revolutions per ring", self.v_pol_revs, 0.5, 8.0, 0.5, command=False)
        self._add_slider(pol, "Retract (mm, radial Z)", self.v_pol_retr, 1.0, 30.0, 0.5, command=False)
        self._add_slider(pol, "Spindle RPM (0 = M3 only)", self.v_pol_rpm, 0, 24000, 100, command=False)
        ttk.Button(
            pol,
            text="Export 4-axis polish (.nc)…",
            command=self.export_cnc_profile,
        ).pack(fill="x", padx=6, pady=4)

        lap = ttk.LabelFrame(cad_box, text="Polish lap (cylindrical abrasive tool)")
        lap.pack(fill="x", padx=4, pady=4)
        ttk.Label(
            lap,
            text=(
                "Cylinder with the negative of the optical face — coat with "
                "abrasive and spin the lens or the tool. Opposite end is a "
                "blind 1/4-20 tap (#7 / 5.10 mm drill, tap after printing)."
            ),
            style="Dim.TLabel",
            wraplength=270,
        ).pack(anchor="w", padx=6, pady=2)
        row_l = ttk.Frame(lap)
        row_l.pack(fill="x", padx=6, pady=2)
        ttk.Label(row_l, text="Face", style="Dim.TLabel").pack(side="left")
        ttk.Combobox(
            row_l,
            textvariable=self.v_lap_surf,
            values=("front", "rear", "both"),
            state="readonly",
            width=8,
        ).pack(side="left", padx=4)
        self._add_slider(
            lap,
            "Wall beyond clear aperture (mm)",
            self.v_lap_wall,
            2.0,
            20.0,
            0.1,
            command=False,
        )
        ttk.Button(
            lap,
            text="Export polish lap STL…",
            command=self.export_polish_lap_stl,
        ).pack(fill="x", padx=6, pady=4)

        self._add_slider(opt, "First vertex Z (mm)", self.v_lens_z, LENS_Z_MIN_MM, LENS_Z_MAX_MM, 0.1)
        self._add_slider(opt, "Custom material n", self.v_custom_n, 1.3, 2.5, 0.001)

        las = ttk.LabelFrame(opt, text="Laser n calibration (printed resin)")
        las.pack(fill="x", padx=6, pady=6)
        ttk.Label(
            las,
            text=(
                "Coaxial shot = card origin. Shift the pointer in X/Y "
                "(do not tilt), mark the new dot, then Fit n. "
                "Press How to calibrate… for the full bench procedure."
            ),
            style="Dim.TLabel",
            wraplength=290,
        ).pack(anchor="w", padx=6, pady=2)
        ttk.Button(las, text="How to calibrate…", command=self._show_laser_cal_guide).pack(
            fill="x", padx=6, pady=(0, 4)
        )
        self._add_slider(las, "Laser wavelength (nm)", self.v_las_wl, 400, 780, 1, command=False)
        self._add_slider(las, "Laser Z (mm)", self.v_las_z, -20, 40, 0.1, command=False)
        self._add_slider(las, "Screen / card Z (mm)", self.v_las_screen, 10, 400, 0.5, command=False)
        self._add_slider(las, "Laser offset X (mm)", self.v_las_hx, -20, 20, 0.05, command=False)
        self._add_slider(las, "Laser offset Y (mm)", self.v_las_hy, -20, 20, 0.05, command=False)
        self._add_slider(las, "Coaxial dot X (mm)", self.v_las_cx, -20, 20, 0.05, command=False)
        self._add_slider(las, "Coaxial dot Y (mm)", self.v_las_cy, -20, 20, 0.05, command=False)
        self._add_slider(las, "Shifted dot X (mm)", self.v_las_sx, -40, 40, 0.05, command=False)
        self._add_slider(las, "Shifted dot Y (mm)", self.v_las_sy, -40, 40, 0.05, command=False)
        ttk.Label(
            las, textvariable=self.laser_cal_status, style="Dim.TLabel", wraplength=290
        ).pack(anchor="w", padx=6, pady=2)
        ttk.Button(las, text="Fit n from laser dots", command=self._calibrate_laser_n).pack(
            fill="x", padx=6, pady=2
        )
        ttk.Button(las, text="Apply fitted n to Custom / Formlabs", command=self._apply_laser_n).pack(
            fill="x", padx=6, pady=(0, 4)
        )
        ttk.Checkbutton(
            opt, text="Fresnel transmission", variable=self.v_fresnel, command=self._on_param_change
        ).pack(anchor="w", padx=8)
        ttk.Checkbutton(
            opt,
            text="Absorb on TIR (recommended — no bounce)",
            variable=self.v_tir_abs,
            command=self._on_param_change,
        ).pack(anchor="w", padx=8)
        ttk.Checkbutton(
            opt,
            text="Kill rays going backward (−Z)",
            variable=self.v_kill_back,
            command=self._on_param_change,
        ).pack(anchor="w", padx=8)

        ttk.Label(
            opt,
            text=f"Up to {MAX_ELEMENTS} elements. Collapse unused slots to save space.",
            style="Dim.TLabel",
            wraplength=290,
        ).pack(anchor="w", padx=8, pady=(2, 0))
        btn_row = ttk.Frame(opt)
        btn_row.pack(fill="x", padx=6, pady=2)
        ttk.Button(
            btn_row, text="Collapse disabled", command=self._collapse_disabled_elements
        ).pack(side="left", padx=(0, 4))
        ttk.Button(
            btn_row, text="Expand all", command=self._expand_all_elements
        ).pack(side="left")

        self.elem_ui = []
        for i, ev in enumerate(self.elem_vars):
            self._build_element_panel(opt, i, ev)

        # Absorbing blockers / enclosure panels
        blk = self._make_collapsible_section(
            parent, "BLOCKERS  ·  absorbing panels / stops", start_collapsed=True
        )
        self._build_blockers_panel(blk)

        # Target — rectangular FOV (camera field)
        tgt = self._make_collapsible_section(
            parent, "TARGET  ·  rectangular FOV (camera field)", start_collapsed=True
        )
        ttk.Label(
            tgt,
            text="Define the rectangular region to fill (aspect ratio like a camera FOV). "
            "Use Design tools below to generate anamorphic optics that reshape LED/COB light into that rectangle.",
            style="Dim.TLabel",
            wraplength=290,
        ).pack(anchor="w", padx=6, pady=2)
        self._add_slider(tgt, "Target distance Z (mm)", self.v_target_z, 10, 2000, 1)
        self._add_slider(tgt, "FOV width (mm)", self.v_fov_w, 5, FOV_SIZE_MAX_MM, 1)
        self._add_slider(tgt, "FOV height (mm)", self.v_fov_h, 5, FOV_SIZE_MAX_MM, 1)
        self._add_slider(tgt, "FOV aspect W/H", self.v_fov_aspect, 0.25, 4.0, 0.01)
        ttk.Checkbutton(
            tgt,
            text="Lock aspect (edit W or aspect → updates H)",
            variable=self.v_fov_lock,
        ).pack(anchor="w", padx=6)
        # Sync aspect when W/H change
        self.v_fov_w.trace_add("write", self._on_fov_dim_change)
        self.v_fov_h.trace_add("write", self._on_fov_dim_change)
        self.v_fov_aspect.trace_add("write", self._on_fov_aspect_change)
        self._add_slider(tgt, "FOV center X (mm)", self.v_fov_cx, -200, 200, 0.5)
        self._add_slider(tgt, "FOV center Y (mm)", self.v_fov_cy, -200, 200, 0.5)
        ttk.Button(
            tgt,
            text="Focus group on target (image the source)",
            command=self._focus_group_on_target,
        ).pack(fill="x", padx=8, pady=(4, 2))
        ttk.Label(
            tgt,
            text=(
                "Moves the whole lens group along Z so each point on the source "
                "is imaged as sharply as possible on the FOV plane — like focusing "
                "a flashlight until the emitter shape appears."
            ),
            style="Dim.TLabel",
            wraplength=290,
        ).pack(anchor="w", padx=8, pady=(0, 4))

        design = ttk.LabelFrame(tgt, text="Design optics for rectangular FOV")
        design.pack(fill="x", padx=6, pady=6)
        ttk.Label(
            design,
            text="Generates cylindrical or biconic powers so the beam footprint "
            "matches the FOV aspect ratio (non-imaging estimate; refine with Trace).",
            style="Dim.TLabel",
            wraplength=280,
        ).pack(anchor="w", padx=4, pady=2)
        ttk.Button(
            design,
            text="Crossed cylinders (X then Y)",
            command=lambda: self._design_rect_fov("crossed"),
        ).pack(fill="x", padx=4, pady=2)
        ttk.Button(
            design,
            text="Biconic singlet (Rx ≠ Ry)",
            command=lambda: self._design_rect_fov("biconic"),
        ).pack(fill="x", padx=4, pady=2)
        ttk.Button(
            design,
            text="Swap anamorphic X ↔ Y (fix 90° rotation)",
            command=self._swap_anamorphic_xy,
        ).pack(fill="x", padx=4, pady=2)
        ttk.Button(
            design,
            text="Rotate optics 90° vs FOV (swap W↔H + axes)",
            command=self._rotate_optics_90_vs_fov,
        ).pack(fill="x", padx=4, pady=2)
        self.design_status = tk.StringVar(value="")
        ttk.Label(design, textvariable=self.design_status, style="Dim.TLabel", wraplength=280).pack(
            anchor="w", padx=4, pady=2
        )
        self._add_slider(tgt, "Map half-width (mm)", self.v_map_w, 10, MAP_HALF_MAX_MM, 1)
        self._add_slider(tgt, "Map half-height (mm)", self.v_map_h, 10, MAP_HALF_MAX_MM, 1)
        self._add_slider(tgt, "Map resolution (bins)", self.v_map_res, 32, 256, 16, True)

        # Sim
        sim = self._make_collapsible_section(
            parent, "SIMULATION", start_collapsed=True
        )
        self._add_slider(sim, "Rays to trace", self.v_rays, 500, 100000, 500, True)
        ttk.Label(
            sim,
            text=(
                "Target-plane map uses this count (split across progressive batches). "
                "Raise map resolution as well if the heatmap still looks blocky."
            ),
            style="Dim.TLabel",
            wraplength=290,
        ).pack(anchor="w", padx=8, pady=(0, 4))
        self._add_slider(sim, "Side-view ray paths", self.v_disp, 20, 5000, 10, True)
        self.v_color_partial = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            sim,
            text="Color partial-lens rays on target (off by default)",
            variable=self.v_color_partial,
            command=self._on_partial_ray_color_toggle,
        ).pack(anchor="w", padx=8, pady=2)
        ttk.Label(
            sim,
            text=(
                "When on: lime = full stack, red = partial stack, "
                "pink = missed lenses (colors avoid the irradiance colorbar)."
            ),
            style="Dim.TLabel",
            wraplength=290,
        ).pack(anchor="w", padx=8, pady=2)

        # Optimizer — all search goals live here (no header optimize button)
        optz = self._make_collapsible_section(
            parent, "OPTIMIZER", start_collapsed=True
        )
        ttk.Label(
            optz,
            text=(
                "All optimization runs from this panel. Pick a goal, then decide "
                "whether extra lenses may be added (rectangular fill only). "
                "Extra lenses are capped by empty slots in the 8-element stack."
            ),
            style="Dim.TLabel",
            wraplength=300,
        ).pack(anchor="w", padx=6, pady=2)
        self.v_opt_rays = tk.IntVar(value=2500)
        self.v_opt_evals = tk.IntVar(value=80)
        self.v_opt_uni_w = tk.DoubleVar(value=0.9)
        self.v_opt_aspect_w = tk.DoubleVar(value=1.5)
        self.v_opt_fill_w = tk.DoubleVar(value=2.0)
        self.v_opt_goal = tk.StringVar(value="Rectangular FOV fill")
        self.v_opt_allow_extra = tk.BooleanVar(value=False)
        self.v_opt_two_phase = tk.BooleanVar(value=False)
        self.v_opt_extra = tk.IntVar(value=2)
        self.v_opt_ana_mode = tk.StringVar(value="crossed")
        self.v_opt_asphere = tk.BooleanVar(value=False)
        self.v_opt_polish = tk.BooleanVar(value=True)

        ttk.Label(optz, text="Goal", style="Dim.TLabel").pack(
            anchor="w", padx=6, pady=(4, 0)
        )
        goal_cb = self._make_combobox(
            optz,
            self.v_opt_goal,
            [
                "Rectangular FOV fill",
                "Sharpest focus",
                "Maximum evenness",
            ],
            width=28,
        )
        goal_cb.pack(fill="x", padx=6, pady=2)
        ttk.Label(
            optz,
            text=(
                "Rectangular FOV fill — pack a sharp rectangle into the FOV "
                "(anamorphic extras only if allowed below).  "
                "Sharpest focus — crispest illumination border.  "
                "Maximum evenness — flattest light from edge to edge of the FOV."
            ),
            style="Dim.TLabel",
            wraplength=300,
        ).pack(anchor="w", padx=6, pady=(0, 4))

        ttk.Checkbutton(
            optz,
            text="Allow adding lenses (rectangular fill only)",
            variable=self.v_opt_allow_extra,
        ).pack(anchor="w", padx=6)
        self._add_slider(
            optz,
            "Max extra lenses (0–8; stack limit 8)",
            self.v_opt_extra,
            0,
            8,
            1,
            True,
            command=False,
        )
        ttk.Label(optz, text="Anamorphic form (when extras are added)", style="Dim.TLabel").pack(
            anchor="w", padx=6, pady=(4, 0)
        )
        ana_cb = self._make_combobox(
            optz,
            self.v_opt_ana_mode,
            ["crossed", "biconic"],
            width=28,
        )
        ana_cb.pack(fill="x", padx=6, pady=2)
        ttk.Label(
            optz,
            text="crossed = cylinder X + cylinder Y · biconic = one Rx≠Ry singlet",
            style="Dim.TLabel",
            wraplength=300,
        ).pack(anchor="w", padx=6)

        self._add_slider(
            optz, "Rays per evaluation", self.v_opt_rays, 500, 15000, 500, True, command=False
        )
        self._add_slider(
            optz, "Max evaluations (approx.)", self.v_opt_evals, 20, 300, 10, True, command=False
        )
        self._add_slider(
            optz, "Uniformity weight", self.v_opt_uni_w, 0.0, 3.0, 0.05, command=False
        )
        self._add_slider(
            optz,
            "FOV-fill weight (size match; under-fill hurts)",
            self.v_opt_fill_w,
            0.0,
            5.0,
            0.1,
            command=False,
        )
        self._add_slider(
            optz,
            "Aspect-match weight (rectangular fill)",
            self.v_opt_aspect_w,
            0.0,
            4.0,
            0.1,
            command=False,
        )
        ttk.Checkbutton(
            optz,
            text="Also optimize conic k & A4 asphere terms",
            variable=self.v_opt_asphere,
        ).pack(anchor="w", padx=6)
        ttk.Checkbutton(
            optz,
            text="Local polish after global search (Nelder–Mead)",
            variable=self.v_opt_polish,
        ).pack(anchor="w", padx=6)
        btn_row = ttk.Frame(optz)
        btn_row.pack(fill="x", padx=6, pady=4)
        ttk.Button(
            btn_row,
            text="Optimize",
            style="Accent.TButton",
            command=self.run_optimize_rectangular,
        ).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text="Cancel", command=self.cancel_optimize).pack(side="left")
        self.opt_status = tk.StringVar(
            value="Choose a goal, then Optimize. Extra lenses stay off until you allow them."
        )
        ttk.Label(
            optz, textvariable=self.opt_status, style="Dim.TLabel", wraplength=300
        ).pack(anchor="w", padx=6, pady=2)

    def _build_metrics(self, parent):
        ttk.Label(parent, text="SPOT & FLUX METRICS", style="Head.TLabel").pack(
            anchor="w", padx=12, pady=(12, 8)
        )
        self.metric_labels: Dict[str, tk.StringVar] = {}
        items = [
            ("collection", "Collection in FOV"),
            ("rms", "RMS spot radius"),
            ("ee50", "Encircled energy R(50%)"),
            ("ee86", "Encircled energy R(86%)"),
            ("peak", "Peak irradiance"),
            ("fov_frac", "Flux in rectangular FOV"),
            ("uniform", "FOV uniformity Emin/Emax"),
            ("edge", "Illumination edge sharpness"),
            ("cv", "FOV CV (σ/μ)"),
            ("aspect", "Footprint aspect σx/σy"),
            ("aspect_tgt", "Target FOV aspect W/H"),
            ("aspect_err", "Aspect error |fp−tgt|/tgt"),
            ("sig", "Spot RMS half-widths σx, σy"),
            ("centroid", "Centroid (X, Y)"),
            ("efl", "Element 1 EFL (lensmaker)"),
            ("dies", "Active dies / surfaces"),
            ("tir", "TIR absorbed / reflections"),
            ("absorb", "Blocker absorbed rays"),
            ("backend", "Trace backend"),
        ]
        for key, title in items:
            card = tk.Frame(parent, bg=BG3, highlightbackground=BORDER, highlightthickness=1)
            card.pack(fill="x", padx=10, pady=3)
            ttk.Label(card, text=title, style="MetricDim.TLabel").pack(anchor="w", padx=8, pady=(6, 0))
            sv = tk.StringVar(value="—")
            self.metric_labels[key] = sv
            ttk.Label(card, textvariable=sv, style="Metric.TLabel").pack(anchor="w", padx=8, pady=(0, 6))

        note = tk.Label(
            parent,
            text=(
                "Physics: rectangular Lambertian dies, Snell refraction, "
                "aspheric sag, Fresnel T, Sellmeier/Cauchy n(λ), Monte Carlo irradiance."
            ),
            bg=BG2,
            fg=FG,
            font=("Segoe UI", 8),
            wraplength=230,
            justify="left",
        )
        note.pack(side="bottom", padx=12, pady=12)

    # ── Params ───────────────────────────────────────────────────────────

    def collect_params(self) -> Dict[str, Any]:
        # Keep selected blocker fields in sync with the list model
        if hasattr(self, "v_blk_z") and not getattr(self, "_blk_syncing", False):
            bi = self._selected_blocker_index()
            if bi is not None:
                self._write_vars_to_blocker(bi)
        p = copy.deepcopy(self.params)
        p["source"] = {
            "mode": self.v_mode.get(),
            "rows": int(self.v_rows.get()),
            "cols": int(self.v_cols.get()),
            "pitch_x": float(self.v_pitch_x.get()),
            "pitch_y": float(self.v_pitch_y.get()),
            "die_width": float(self.v_die_w.get()),
            "die_height": float(self.v_die_h.get()),
            "source_z": float(self.v_source_z.get()),
            "flux_per_die": float(self.v_flux.get()),
            "wavelength_nm": max(VISIBLE_NM_MIN, min(VISIBLE_NM_MAX, float(self.v_wl.get()))),
            "half_angle_deg": float(self.v_half.get()),
            "tilt_x": float(self.v_tilt_x.get()),
            "tilt_y": float(self.v_tilt_y.get()),
            "offset_x": float(self.v_off_x.get()),
            "offset_y": float(self.v_off_y.get()),
            "die_rot_z": float(self.v_rot_z.get()),
            "stagger": bool(self.v_stagger.get()),
            "circular_mask": bool(self.v_circ.get()),
            "mask_radius": float(self.v_mask_r.get()),
        }
        p["elements"] = []
        for ev in self.elem_vars:
            mode = str(ev["surface_mode"].get())
            use_ell = bool(ev["use_elliptical_ap"].get())
            el = {
                "enabled": bool(ev["enabled"].get()),
                "shape_id": shape_id_from_label(ev["shape"].get()),
                "R1": float(ev["R1"].get()),
                "R2": float(ev["R2"].get()),
                "thickness": float(ev["thickness"].get()),
                "air_after": float(ev["air_after"].get()),
                "aperture": float(ev["aperture"].get()),
                "material": material_id_from_name(ev["material"].get()),
                "surface_mode": mode,
                "mode_s1": mode,
                "mode_s2": mode,
                "k1": float(ev["k1"].get()),
                "k2": float(ev["k2"].get()),
                "A4_1": float(ev["A4_1"].get()),
                "A4_2": float(ev["A4_2"].get()),
            }
            # Radii convention:
            #   rotational / cylinder_x / cylinder_y → R1, R2 are the powered radii
            #     (cylinder_x: power in X; cylinder_y: power in Y)
            #   biconic → R1/R2 = Rx, R1y/R2y = Ry (independent)
            # Leaving R1y/R2y as None for cylinders makes the engine use R1/R2
            # for the powered meridian so front AND rear both curve when |R|≠0.
            if mode == "biconic":
                el["R1y"] = float(ev["R1y"].get())
                el["R2y"] = float(ev["R2y"].get())
            else:
                el["R1y"] = None
                el["R2y"] = None
            circ = True
            if ev.get("circular_lock") is not None:
                circ = bool(ev["circular_lock"].get())
            el["circular_lock"] = circ
            if circ:
                el = apply_circular_outline(el, locked=True)
            elif use_ell:
                el["aperture_y"] = float(ev["aperture_y"].get())
            else:
                el["aperture_y"] = None
            p["elements"].append(el)
        p["lens_z_start"] = float(self.v_lens_z.get())
        p["custom_n"] = float(self.v_custom_n.get())
        p["apply_fresnel"] = bool(self.v_fresnel.get())
        p["absorb_on_tir"] = bool(self.v_tir_abs.get())
        p["kill_backward"] = bool(self.v_kill_back.get())
        p["max_reflections"] = 0 if p["absorb_on_tir"] else 3
        p["mla"] = {
            "enabled": bool(self.v_mla.get()),
            "fill_factor": float(self.v_mla_fill.get()),
            "lenslet_aperture": float(self.v_mla_ap.get()),
            "export_plate": bool(self.v_export_plate.get()),
            "scale_to_pitch": bool(self.v_mla_scale.get()) if hasattr(self, "v_mla_scale") else True,
            "aim_to_fov": bool(self.v_mla_aim.get()) if hasattr(self, "v_mla_aim") else True,
            "aim_strength": float(self.v_mla_aim_s.get()) if hasattr(self, "v_mla_aim_s") else 1.0,
        }
        p["target_z"] = float(self.v_target_z.get())
        p["fov_width"] = float(self.v_fov_w.get())
        p["fov_height"] = float(self.v_fov_h.get())
        p["fov_aspect_lock"] = bool(self.v_fov_lock.get())
        p["fov_cx"] = float(self.v_fov_cx.get())
        p["fov_cy"] = float(self.v_fov_cy.get())
        p["map_half_w"] = float(self.v_map_w.get())
        p["map_half_h"] = float(self.v_map_h.get())
        # FOV must sit inside the recorded map or collection / fill read empty
        cov_w, cov_h = map_half_covering_fov(
            p["fov_width"], p["fov_height"], p["fov_cx"], p["fov_cy"],
            p["map_half_w"], p["map_half_h"],
        )
        p["map_half_w"], p["map_half_h"] = cov_w, cov_h
        p["map_res"] = int(self.v_map_res.get())
        p["total_rays"] = int(self.v_rays.get())
        p["display_rays"] = int(self.v_disp.get())
        if hasattr(self, "v_cad_edge"):
            p["cad_max_edge_mm"] = float(self.v_cad_edge.get())
        if hasattr(self, "v_cad_angle"):
            p["cad_max_angle_deg"] = float(self.v_cad_angle.get())
        if hasattr(self, "v_cad_flange_r"):
            p["cad_flange_radial_mm"] = float(self.v_cad_flange_r.get())
        if hasattr(self, "v_cad_flange_t"):
            p["cad_flange_thickness_mm"] = float(self.v_cad_flange_t.get())
        if hasattr(self, "v_cad_halves"):
            p["cad_export_halves"] = bool(self.v_cad_halves.get())
        if hasattr(self, "v_pol_dia"):
            p["cad_polish_tool_dia_mm"] = float(self.v_pol_dia.get())
            p["cad_polish_stepover_mm"] = float(self.v_pol_step.get())
            p["cad_polish_allowance_mm"] = float(self.v_pol_allow.get())
            p["cad_polish_feed_mm_min"] = float(self.v_pol_feed.get())
            p["cad_polish_retract_mm"] = float(self.v_pol_retr.get())
            p["cad_polish_rpm"] = float(self.v_pol_rpm.get())
            p["cad_polish_revs"] = float(self.v_pol_revs.get())
            p["cad_polish_surface"] = str(self.v_pol_surf.get())
            p["cad_polish_strategy"] = str(self.v_pol_strat.get())
        if hasattr(self, "v_lap_surf"):
            p["cad_lap_surface"] = str(self.v_lap_surf.get())
            p["cad_lap_wall_mm"] = float(self.v_lap_wall.get())
        p["blockers"] = [dict(b) for b in self.blockers]
        return p

    def _preview_optics(self):
        """Geometry-only refresh — no Monte Carlo.

        With Auto-run off this skips ray polylines and CAD remesh estimates
        so dragging many lenses stays interactive.
        """
        auto = bool(self.auto_run.get()) if hasattr(self, "auto_run") else True
        if self.result is not None:
            try:
                self._draw_side_preview = not auto
                self._draw_side()
            except Exception:
                pass
            finally:
                self._draw_side_preview = False
        if auto and hasattr(self, "cad_mesh_status"):
            try:
                self._refresh_cad_mesh_label()
            except Exception:
                pass
        try:
            self._refresh_fov_overlay()
        except Exception:
            pass
        try:
            self._sync_group_z_bar()
        except Exception:
            pass

    def _on_auto_run_toggle(self):
        """Auto-run off must actually stop work — cancel an in-flight trace."""
        if not bool(self.auto_run.get()):
            self._cancel_trace_work()
            self._preview_optics()
            self.status_var.set("Auto-run off — geometry only. Press Trace rays to simulate.")
            return
        self.status_var.set("Auto-run on — traces after edits. Press Trace rays for a map now.")

    def _request_trace_after_edit(self, status: str = "") -> None:
        """Start a trace only when Auto-run is on and nothing is held down."""
        if should_start_trace(
            auto_run=bool(self.auto_run.get()),
            live=self._live_edit.live,
            handle_drag=self._drag is not None,
        ):
            if status:
                self.status_var.set(status)
            self.run_trace()
            return
        self._preview_optics()
        if status:
            note = status.split("—")[0].strip(" .")
            if not bool(self.auto_run.get()):
                note = f"{note} · Auto-run off" if note else "Auto-run off"
            self.status_var.set(note or "Ready")

    def _cancel_trace_work(self):
        """Drop a pending debounce and abort any in-flight progressive trace."""
        if self._debounce_id is not None:
            try:
                self.after_cancel(self._debounce_id)
            except Exception:
                pass
            self._debounce_id = None
        self._trace_gen += 1

    def _begin_live_edit(self):
        self._live_edit.begin()
        self._cancel_trace_work()

    def _end_live_edit(self):
        if not self._live_edit.end():
            return
        self._preview_optics()
        if should_start_trace(
            auto_run=bool(self.auto_run.get()),
            live=self._live_edit.live,
            handle_drag=self._drag is not None,
        ):
            self.run_trace()

    def _on_global_slider_release(self, _event=None):
        sl = getattr(self, "_live_slider", None)
        if sl is None:
            return
        sl._finish_drag()

    def _on_param_change(self, *_):
        # Live geometry (convexity, Z, aperture) updates immediately.
        # Ray-tracing waits until the pointer is up — see LiveEditGate.
        self._preview_optics()
        if not should_start_trace(
            auto_run=bool(self.auto_run.get()),
            live=self._live_edit.live,
            handle_drag=self._drag is not None,
        ):
            self._cancel_trace_work()
            return
        # Discrete clicks (buttons, comboboxes): coalesce only, do not wait
        # long enough that a held slider could sneak a trace in.
        if self._debounce_id is not None:
            self.after_cancel(self._debounce_id)
        self._debounce_id = self.after(50, self._run_trace_if_idle)

    def _run_trace_if_idle(self):
        self._debounce_id = None
        if not should_start_trace(
            auto_run=bool(self.auto_run.get()),
            live=self._live_edit.live,
            handle_drag=self._drag is not None,
        ):
            return
        self.run_trace()

    def _first_run(self):
        self.run_trace()

    def run_trace(self, *, _resume_gen: Optional[int] = None):
        """Start progressive Monte Carlo (cancels any in-flight refinement).

        _resume_gen: internal — reuse an existing generation after waiting for
        a previous worker to release the lock (do not bump counter again).
        """
        if _resume_gen is None:
            self._trace_gen += 1
            gen = self._trace_gen
        else:
            if _resume_gen != self._trace_gen:
                return  # superseded while waiting
            gen = _resume_gen
        params = self.collect_params()
        n_batches, rpb = map_ray_budget(
            int(params.get("total_rays", 6000)),
            int(self.prog_batches),
        )

        acquired = self._run_lock.acquire(blocking=False)
        if not acquired:
            # Previous worker still holds the lock — it will exit on cancel;
            # schedule a retry shortly so the new gen actually runs.
            self.after(80, lambda g=gen: self._retry_trace_if_current(g))
            self.status_var.set(f"Waiting to start batch 1/{n_batches}…")
            return

        self._running = True
        self.status_var.set(f"Tracing batch 1/{n_batches} · {rpb} rays/batch…")
        self.progress["value"] = 0

        def work():
            def prog(f):
                self.after(0, lambda v=f: self.progress.configure(value=v * 100))

            def should_cancel():
                return gen != self._trace_gen

            def on_batch(result, bi, n_b):
                if gen != self._trace_gen:
                    return
                self.after(0, lambda r=result, b=bi, n=n_b: self._on_batch(r, b, n, gen))

            try:
                # Side-view / color-overlay path count follows the UI slider
                n_disp = max(20, int(params.get("display_rays", self.prog_disp_batch)))
                run_simulation_progressive(
                    params,
                    batch_cb=on_batch,
                    n_batches=n_batches,
                    rays_per_batch=rpb,
                    display_per_batch=n_disp,
                    progress_cb=prog,
                    should_cancel=should_cancel,
                )
                self.after(0, lambda: self._on_progressive_finished(gen, None))
            except Exception as e:
                self.after(0, lambda: self._on_progressive_finished(gen, e))

        threading.Thread(target=work, daemon=True).start()

    def _retry_trace_if_current(self, gen: int):
        if gen != self._trace_gen:
            return
        self.run_trace(_resume_gen=gen)

    def _on_batch(self, result: SimResult, batch_i: int, n_batches: int, gen: int):
        """Intermediate progressive update — only apply if still current gen."""
        if gen != self._trace_gen:
            return
        self.result = result
        st = result.stats
        more = batch_i < n_batches
        if more:
            self.status_var.set(
                f"Batch {batch_i}/{n_batches} · {st['hit']:,} hits / {st['launched']:,} rays · {st.get('backend', 'cpu')}"
            )
        else:
            self.status_var.set(
                f"Done · {st['hit']:,} hits / {st['launched']:,} rays · {st['n_dies']} dies · {st.get('backend', 'cpu')}"
            )
        self.progress["value"] = 100.0 * batch_i / max(n_batches, 1)
        self._update_metrics(st)
        self._redraw()

    def _on_progressive_finished(self, gen: int, err: Optional[BaseException]):
        # Always release the lock this worker acquired, even if superseded.
        self._running = False
        try:
            self._run_lock.release()
        except Exception:
            pass
        if gen != self._trace_gen:
            return  # superseded — ignore error/UI for this gen
        if err is not None:
            self.status_var.set(f"Error: {err}")
            messagebox.showerror("Simulation error", str(err))
        else:
            self.progress["value"] = 100

    def _on_done(self, result: Optional[SimResult], err: Optional[BaseException]):
        """Legacy single-shot completion (kept for safety)."""
        self._running = False
        try:
            self._run_lock.release()
        except Exception:
            pass
        self.progress["value"] = 100
        if err is not None:
            self.status_var.set(f"Error: {err}")
            messagebox.showerror("Simulation error", str(err))
            return
        self.result = result
        st = result.stats
        self.status_var.set(
            f"Done · {st['hit']:,} hits / {st['launched']:,} rays · {st['n_dies']} dies · {st.get('backend', 'cpu')}"
        )
        self._update_metrics(st)
        self._redraw()

    def _update_metrics(self, st: Dict[str, Any]):
        fov = st["fov"]
        efl = st["efl"]
        self.metric_labels["collection"].set(f"{st['collection'] * 100:.1f} %")
        self.metric_labels["rms"].set(f"{st['rms']:.2f} mm")
        self.metric_labels["ee50"].set(f"{st['ee50']:.2f} mm")
        self.metric_labels["ee86"].set(f"{st['ee86']:.2f} mm")
        pe = st["peak_e"]
        self.metric_labels["peak"].set(f"{pe:.3g} a.u./mm²")
        self.metric_labels["fov_frac"].set(f"{fov['fraction'] * 100:.1f} %")
        self.metric_labels["uniform"].set(f"{fov['uniformity'] * 100:.1f} %")
        if "edge" in self.metric_labels:
            self.metric_labels["edge"].set(f"{float(fov.get('edge_sharpness', 0.0)):.3f}")
        self.metric_labels["cv"].set(f"{fov['cv']:.3f}")
        self.metric_labels["aspect"].set(f"{fov.get('footprint_aspect', 0):.3f}")
        self.metric_labels["aspect_tgt"].set(f"{fov.get('target_aspect', 0):.3f}")
        self.metric_labels["aspect_err"].set(f"{fov.get('aspect_error', 0) * 100:.1f} %")
        self.metric_labels["sig"].set(
            f"{fov.get('sig_x', 0):.2f}, {fov.get('sig_y', 0):.2f} mm"
        )
        cx, cy = st["centroid"]
        self.metric_labels["centroid"].set(f"{cx:.2f}, {cy:.2f} mm")
        if math.isfinite(efl):
            self.metric_labels["efl"].set(f"{efl:.2f} mm")
        else:
            self.metric_labels["efl"].set("∞")
        self.metric_labels["dies"].set(f"{st['n_dies']} / {st['n_surfaces']}")
        self.metric_labels["tir"].set(
            f"{st.get('n_tir_absorb', 0)} / {st.get('n_reflections', 0)}"
        )
        if "absorb" in self.metric_labels:
            self.metric_labels["absorb"].set(str(st.get("n_absorb", 0)))
        self.metric_labels["backend"].set(str(st.get("backend", "cpu")))

    # ── Drawing ──────────────────────────────────────────────────────────

    def _style_axes(self):
        axes = [self.ax_side, self.ax_tgt]
        if hasattr(self, "ax_side_xz"):
            axes.append(self.ax_side_xz)
        if hasattr(self, "ax_prof"):
            axes.append(self.ax_prof)
        for ax in axes:
            ax.set_facecolor(BG)
            ax.tick_params(colors="#f8fafc", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(BORDER)
            ax.xaxis.label.set_color("#f8fafc")
            ax.yaxis.label.set_color("#f8fafc")
            ax.title.set_color(FG_BRIGHT)

    def _redraw_empty(self):
        self.ax_side.clear()
        if hasattr(self, "ax_side_xz"):
            self.ax_side_xz.clear()
        self.ax_tgt.clear()
        if hasattr(self, "ax_prof"):
            self.ax_prof.clear()
        self._style_axes()
        self.ax_side.set_title(
            "SIDE VIEW  ·  Y–Z  ·  cylinder_y / rotational",
            loc="left",
            fontsize=9,
            color=FG_BRIGHT,
        )
        self.ax_side.set_ylabel("Y (mm)", color="#f8fafc")
        if hasattr(self, "ax_side_xz"):
            self.ax_side_xz.set_title(
                "SIDE VIEW  ·  X–Z  ·  cylinder_x / rotational (orthogonal)",
                loc="left",
                fontsize=9,
                color=FG_BRIGHT,
            )
            self.ax_side_xz.set_xlabel("Z (mm)", color="#f8fafc")
            self.ax_side_xz.set_ylabel("X (mm)", color="#f8fafc")
        self.ax_tgt.set_title(
            "TARGET PLANE  ·  irradiance (source → field)",
            loc="left",
            fontsize=10,
            color=FG_BRIGHT,
        )
        self.ax_tgt.set_xlabel("X (mm)", color="#f8fafc")
        self.ax_tgt.set_ylabel("Y (mm)", color="#f8fafc")
        if hasattr(self, "ax_prof"):
            self.ax_prof.set_title(
                "PROFILES  ·  X / Y / diagonal",
                loc="left",
                fontsize=10,
                color=FG_BRIGHT,
            )
            self.ax_prof.set_xlabel("Position along cut (mm)", color="#f8fafc")
            self.ax_prof.set_ylabel("Normalized irradiance", color="#f8fafc")
        for ax in self._side_axes() + [self.ax_tgt] + (
            [self.ax_prof] if hasattr(self, "ax_prof") else []
        ):
            ax.title.set_color(FG_BRIGHT)
            ax.xaxis.label.set_color("#f8fafc")
            ax.yaxis.label.set_color("#f8fafc")
            ax.tick_params(colors="#f8fafc")
        self.fig_side.tight_layout()
        self.fig_tgt.tight_layout()
        if hasattr(self, "fig_prof"):
            self.fig_prof.tight_layout()
            self.canvas_prof.draw_idle()
        self.canvas_side.draw_idle()
        self.canvas_tgt.draw_idle()

    def _redraw(self):
        if self.result is None:
            self._redraw_empty()
            return
        self._draw_side()
        if self._drag is None:
            self._draw_target()
            self._draw_profiles()

    def _connect_side_mouse(self):
        c = self.canvas_side
        self._side_cid["press"] = c.mpl_connect("button_press_event", self._on_side_press)
        self._side_cid["motion"] = c.mpl_connect("motion_notify_event", self._on_side_motion)
        self._side_cid["release"] = c.mpl_connect("button_release_event", self._on_side_release)
        self._side_cid["scroll"] = c.mpl_connect("scroll_event", self._on_side_scroll)

    def _build_group_z_bar(self, parent):
        """Horizontal bar under both side views: move the whole lens group on Z."""
        bar = tk.Frame(parent, bg=BG2, highlightbackground=BORDER, highlightthickness=1)
        bar.pack(side="top", fill="x", padx=8, pady=(0, 4))
        ttk.Label(
            bar,
            text="Lens group Z",
            style="Dim.TLabel",
        ).pack(side="left", padx=(8, 6), pady=4)
        self._group_z_label = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self._group_z_label, style="Dim.TLabel").pack(
            side="right", padx=(4, 8)
        )
        lo, hi = 0.5, 100.0
        try:
            lo, hi = group_slider_limits(self.params)
        except Exception:
            pass
        self._group_z_scale = tk.Scale(
            bar,
            from_=lo,
            to=hi,
            orient="horizontal",
            variable=self.v_lens_z,
            resolution=0.1,
            showvalue=False,
            bg=BG2,
            fg=FG_BRIGHT,
            troughcolor="#020617",
            activebackground=ACCENT,
            highlightthickness=0,
            bd=0,
            sliderrelief="flat",
            width=12,
        )
        self._group_z_scale.pack(side="left", fill="x", expand=True, padx=4, pady=2)
        self._group_z_scale.bind("<ButtonPress-1>", self._on_group_z_press, add="+")
        self._group_z_scale.bind("<ButtonRelease-1>", self._on_group_z_release, add="+")
        self._group_z_dragging = False
        self._syncing_group_z = False
        self._sync_group_z_bar()
        self._group_z_scale.configure(command=self._on_group_z_scale)

    def _on_group_z_press(self, _event=None):
        self._group_z_dragging = True
        self._begin_live_edit()

    def _on_group_z_release(self, _event=None):
        if not getattr(self, "_group_z_dragging", False):
            return
        self._group_z_dragging = False
        self._end_live_edit()

    def _on_group_z_scale(self, _=None):
        if getattr(self, "_syncing_group_z", False):
            return
        self._update_group_z_label()
        self._on_param_change()

    def _update_group_z_label(self):
        if not hasattr(self, "_group_z_label"):
            return
        try:
            z = float(self.v_lens_z.get())
        except (TypeError, ValueError, tk.TclError):
            return
        self._group_z_label.set(f"{z:.2f} mm  ·  source ← → target")

    def _sync_group_z_bar(self):
        if not hasattr(self, "_group_z_scale"):
            return
        if getattr(self, "_syncing_group_z", False):
            return
        try:
            p = self.collect_params()
            lo, hi = group_slider_limits(p)
        except Exception:
            return
        try:
            z = float(self.v_lens_z.get())
        except (TypeError, ValueError, tk.TclError):
            z = lo
        lo_s = min(float(lo), z)
        hi_s = max(float(hi), z)
        if hi_s < lo_s + 0.1:
            hi_s = lo_s + 0.1
        self._syncing_group_z = True
        try:
            self._group_z_scale.configure(from_=lo_s, to=hi_s)
            self._update_group_z_label()
        finally:
            self._syncing_group_z = False

    def _side_axes(self):
        """Axes that participate in side-view interaction (Y–Z and X–Z)."""
        axes = []
        if getattr(self, "ax_side", None) is not None:
            axes.append(self.ax_side)
        if getattr(self, "ax_side_xz", None) is not None:
            axes.append(self.ax_side_xz)
        return axes

    def _reset_side_zoom(self):
        self._side_xlim = None
        self._side_ylim = None
        self._side_xz_ylim = None
        self._side_pan = None
        if self.result is not None:
            self._draw_side()
        self.status_var.set("Side view zoom reset")

    @staticmethod
    def _clamp_view_window(x0, x1, full_lo, full_hi, *, max_out: float = ZOOM_OUT_MAX):
        """
        Keep a zoom/pan window valid.

        - Window may be up to ``max_out`` × the full scene size (zoom out).
        - When zoomed *in* (window smaller than scene), pan is clamped so the
          window stays inside the scene.
        - When zoomed *out*, only soft-clamp so the scene remains partly visible.
        """
        full_w = max(full_hi - full_lo, 1e-9)
        w = x1 - x0
        max_w = full_w * max_out
        min_w = max(0.5, 0.02 * full_w)
        if w < min_w:
            cx = 0.5 * (x0 + x1)
            w = min_w
            x0, x1 = cx - 0.5 * w, cx + 0.5 * w
        elif w > max_w:
            cx = 0.5 * (x0 + x1)
            w = max_w
            x0, x1 = cx - 0.5 * w, cx + 0.5 * w
        if w <= full_w + 1e-12:
            # Zoomed in: keep window inside the scene frame
            if x0 < full_lo:
                x1 += full_lo - x0
                x0 = full_lo
            if x1 > full_hi:
                x0 -= x1 - full_hi
                x1 = full_hi
            # Re-clamp if full scene smaller than window (shouldn't happen)
            if x0 < full_lo:
                x0 = full_lo
            if x1 > full_hi:
                x1 = full_hi
        else:
            # Zoomed out: allow margin around the scene, but don't drift away forever
            margin = 0.5 * (w - full_w)
            lo = full_lo - margin
            hi = full_hi + margin
            if x0 < lo:
                x1 += lo - x0
                x0 = lo
            if x1 > hi:
                x0 -= x1 - hi
                x1 = hi
        return x0, x1

    def _apply_side_limits(self, ax, plane: str = "yz"):
        """Restore side-view zoom after redraw; allow zoom-out beyond fit view."""
        if self._side_full_extent is None:
            return
        ext = self._side_full_extent
        zmin, zmax, ymin, ymax = ext[0], ext[1], ext[2], ext[3]
        xmin, xmax = (ext[4], ext[5]) if len(ext) > 4 else (ymin, ymax)
        use_xz = str(plane).lower().startswith("x") or ax is getattr(self, "ax_side_xz", None)
        tmin, tmax = (xmin, xmax) if use_xz else (ymin, ymax)
        stored_y = self._side_xz_ylim if use_xz else self._side_ylim
        xlim, ylim = resolve_applied_side_limits(
            stored_xlim=self._side_xlim,
            stored_ylim=stored_y,
            zmin=zmin,
            zmax=zmax,
            tmin=tmin,
            tmax=tmax,
            clamp_fn=self._clamp_view_window,
        )
        if self._side_xlim is not None:
            self._side_xlim = xlim
        if stored_y is not None:
            if use_xz:
                self._side_xz_ylim = ylim
            else:
                self._side_ylim = ylim
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_autoscalex_on(False)
        ax.set_autoscaley_on(False)

    def _on_side_scroll(self, event):
        """Mouse wheel zoom on either side view, centered on cursor."""
        if self._drag is not None:
            return
        if event.inaxes not in self._side_axes():
            return
        if event.xdata is None or event.ydata is None:
            return
        if event.button == "up":
            scale = 0.8
        elif event.button == "down":
            scale = 1.25
        else:
            return
        ax = event.inaxes
        use_xz = ax is getattr(self, "ax_side_xz", None)
        zc, yc = float(event.xdata), float(event.ydata)
        z0, z1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        new_w = (z1 - z0) * scale
        new_h = (y1 - y0) * scale
        if self._side_full_extent is not None:
            ext = self._side_full_extent
            fz0, fz1, fy0, fy1 = ext[0], ext[1], ext[2], ext[3]
            fx0, fx1 = (ext[4], ext[5]) if len(ext) > 4 else (fy0, fy1)
            t0, t1 = (fx0, fx1) if use_xz else (fy0, fy1)
            min_w = max(0.5, 0.02 * (fz1 - fz0))
            min_h = max(0.2, 0.02 * (t1 - t0))
            max_w = (fz1 - fz0) * ZOOM_OUT_MAX
            max_h = (t1 - t0) * ZOOM_OUT_MAX
            new_w = max(min_w, min(new_w, max_w))
            new_h = max(min_h, min(new_h, max_h))
        else:
            new_w = max(0.5, new_w)
            new_h = max(0.2, new_h)
        relz = (zc - z0) / max(z1 - z0, 1e-12)
        rely = (yc - y0) / max(y1 - y0, 1e-12)
        nz0 = zc - relz * new_w
        nz1 = nz0 + new_w
        ny0 = yc - rely * new_h
        ny1 = ny0 + new_h
        self._side_xlim = (nz0, nz1)
        if use_xz:
            self._side_xz_ylim = (ny0, ny1)
        else:
            self._side_ylim = (ny0, ny1)
        self._apply_side_limits(ax, plane="xz" if use_xz else "yz")
        # Keep shared Z on the other panel
        other = self.ax_side_xz if not use_xz and hasattr(self, "ax_side_xz") else self.ax_side
        if other is not None and other is not ax and self._side_xlim is not None:
            other.set_xlim(*self._side_xlim)
        self.canvas_side.draw_idle()

    def _element_layout(self, params: Optional[Dict[str, Any]] = None) -> list:
        """Front/rear Z of each enabled lens element along the stack.

        When MLA is on, element 0 uses scaled lenslet thickness/aperture so the
        drag handle matches the real micro-plate, not the full Element 1 design.
        """
        p = params if params is not None else self.collect_params()
        z = float(p["lens_z_start"])
        mla = p.get("mla") or {}
        mla_on = bool(mla.get("enabled", False))
        mla_meta = None
        if mla_on:
            try:
                from mla_geometry import build_mla_lens_specs

                _specs, mla_meta = build_mla_lens_specs(p)
            except Exception:
                mla_meta = None

        layout = []
        first_enabled = True
        for i, e in enumerate(p["elements"]):
            if not e.get("enabled", True):
                continue
            thick = float(e["thickness"])
            ap = float(e.get("aperture", 10))
            ap_y = e.get("aperture_y")
            ap_y = float(ap_y) if ap_y is not None else ap
            label = f"E{i + 1}"
            if mla_on and first_enabled and mla_meta:
                thick = float(mla_meta.get("thickness", thick))
                ap = float(mla_meta.get("aperture", ap))
                ap_y = ap
                label = "MLA"
            first_enabled = False
            layout.append(
                {
                    "index": i,
                    "front_z": z,
                    "rear_z": z + thick,
                    "thickness": thick,
                    "air_after": float(e.get("air_after", 0)),
                    "aperture": max(ap, ap_y),
                    "label": label,
                    "mla": mla_on and label == "MLA",
                }
            )
            z = z + thick + float(e.get("air_after", 0))
        return layout

    def _pick_element(self, z: float, y: float) -> Optional[Dict[str, Any]]:
        hit = self._pick_element_interaction(z, y)
        return hit[0] if hit else None

    def _pick_element_interaction(
        self, z: float, y: float
    ) -> Optional[tuple]:
        """
        Return (handle_dict, mode) where mode is one of:
          'move' | 'resize' | 'radius_front' | 'radius_rear'

        Front/rear vertex (near axis) → radius of that surface.
        Top/bottom rim → clear aperture.
        Centre body → move along Z (other elements stay fixed).
        """
        if not self._element_handles:
            return None
        best = None
        best_d = 1e9
        best_mode = "move"
        res = self.result
        for h in self._element_handles:
            ap = max(float(h["aperture"]), 0.5)
            zf, zr = float(h["front_z"]), float(h["rear_z"])
            z_mid = 0.5 * (zf + zr)
            # Visible top/bottom of the optic in side view
            y_top, y_bot = ap, -ap
            if h.get("mla") and res and res.dies:
                ys = [d.cy for d in res.dies]
                y_top = max(ys) + ap
                y_bot = min(ys) - ap
            y_lim = max(abs(y_top), abs(y_bot)) * 1.15
            in_z = (zf - 2.5) <= z <= (zr + 2.5)

            # Radius handles: near optical axis at front / rear vertices
            # Prefer these when |y| is small so body-move does not steal the grab.
            rad_hit = max(1.4, 0.12 * ap)
            if abs(y) <= rad_hit * 1.35:
                d_front = math.hypot(z - zf, y)
                d_rear = math.hypot(z - zr, y)
                if d_front <= rad_hit and d_front < best_d:
                    best_d = d_front
                    best = h
                    best_mode = "radius_front"
                if d_rear <= rad_hit and d_rear < best_d:
                    best_d = d_rear
                    best = h
                    best_mode = "radius_rear"
                if best is h and best_mode.startswith("radius"):
                    continue

            # Distance to top / bottom resize grips
            d_top = math.hypot(z - z_mid, y - y_top)
            d_bot = math.hypot(z - z_mid, y - y_bot)
            d_edge = min(d_top, d_bot)
            if in_z:
                d_edge = min(d_edge, abs(y - y_top), abs(y - y_bot))
            edge_hit_r = max(1.2, 0.18 * ap, 0.08 * max(zr - zf, 1.0))
            if d_edge <= edge_hit_r and in_z:
                if d_edge < best_d:
                    best_d = d_edge
                    best = h
                    best_mode = "resize"
                continue
            # Body / centre → move
            if in_z and abs(y) <= y_lim:
                d = abs(z - z_mid) + 0.15 * abs(y)
                if d < best_d:
                    best_d = d
                    best = h
                    best_mode = "move"
            else:
                d = math.hypot(z - zf, max(0.0, abs(y) - y_lim))
                if d < 3.0 and d < best_d:
                    best_d = d
                    best = h
                    best_mode = "move"
        if best is None:
            return None
        return best, best_mode

    def _max_drawable_aperture(self, elem_index: int, ap_request: float = LENS_SEMI_MAX_MM) -> float:
        """
        Largest semi-aperture that still keeps a positive edge thickness for
        this element's current R1/R2/thickness. Matches what _draw_lens_body
        will actually render (handles must not drift past the glass outline).
        """
        from engine import build_surfaces, max_aperture_positive_edge

        cap = LENS_SEMI_MAX_MM
        if elem_index < 0 or elem_index >= len(self.elem_vars):
            return max(1.0, min(cap, float(ap_request)))
        p = self.collect_params()
        # Temporarily request a large aperture so the clamp is geometry-limited
        els = [dict(e) for e in p["elements"]]
        if elem_index >= len(els) or not els[elem_index].get("enabled"):
            return max(1.0, min(cap, float(ap_request)))
        els[elem_index] = dict(els[elem_index])
        els[elem_index]["aperture"] = max(float(ap_request), cap)
        try:
            surfs = build_surfaces(
                els,
                float(p["lens_z_start"]),
                None,
                None,
            )
        except Exception:
            return max(1.0, min(cap, float(ap_request)))
        # Surfaces for this element are labeled E{n}S1 / E{n}S2
        label = f"E{elem_index + 1}"
        pair = [s for s in surfs if getattr(s, "label", "").startswith(label)]
        if len(pair) < 2:
            return max(1.0, min(cap, float(ap_request)))
        s1, s2 = pair[0], pair[1]
        ap = max_aperture_positive_edge(s1, s2, max(float(ap_request), cap), min_edge=0.25)
        return max(1.0, min(cap, float(ap)))

    def _apply_element_aperture(self, elem_index: int, new_ap: float) -> None:
        """Set semi-aperture (mm) for an element; keep Y in sync when elliptical."""
        if elem_index < 0 or elem_index >= len(self.elem_vars):
            return
        # Clamp to physically drawable size so stored value matches the side view
        ap_req = max(1.0, min(LENS_SEMI_MAX_MM, float(new_ap)))
        ap = self._max_drawable_aperture(elem_index, ap_req)
        ap = max(1.0, min(ap_req, ap))
        ev = self.elem_vars[elem_index]
        # MLA: driving aperture is fill factor / lenslet size — update design aperture
        # so scaled MLA follows after re-trace
        old_ap = max(float(ev["aperture"].get()), 1e-6)
        ev["aperture"].set(round(ap, 3))
        if bool(ev.get("use_elliptical_ap") and ev["use_elliptical_ap"].get()):
            old_apy = max(float(ev["aperture_y"].get()), 1e-6)
            ratio = ap / old_ap
            ev["aperture_y"].set(round(max(1.0, min(LENS_SEMI_MAX_MM, old_apy * ratio)), 3))
        else:
            ev["aperture_y"].set(round(ap, 3))
        self.status_var.set(f"Element {elem_index + 1} semi-aperture → {ap:.2f} mm")

    def _clamp_element_front_z(self, elem_index: int, new_front: float) -> float:
        """Clamp so this element does not overlap neighbours; others stay put."""
        p = self.collect_params()
        layout = self._element_layout(p)
        by_idx = {h["index"]: h for h in layout}
        if elem_index not in by_idx:
            return new_front
        h = by_idx[elem_index]
        thick = h["thickness"]
        source_z = float(p["source"]["source_z"])
        target_z = float(p["target_z"])
        z_min = source_z + 0.3
        z_max = target_z - thick - 0.5
        enabled = [item for item in layout]
        pos = next((i for i, item in enumerate(enabled) if item["index"] == elem_index), None)
        if pos is None:
            return new_front
        if pos > 0:
            z_min = max(z_min, enabled[pos - 1]["rear_z"] + 0.2)
        if pos + 1 < len(enabled):
            # Keep a minimum air gap before the next element (which stays fixed)
            z_max = min(z_max, enabled[pos + 1]["front_z"] - thick - 0.2)
        if z_max < z_min:
            z_max = z_min
        return max(z_min, min(z_max, new_front))

    def _apply_element_front_z(self, elem_index: int, new_front: float) -> None:
        """
        Move one element in Z without shifting the others.

        Stack spacing is still stored as lens_z_start + air_after chain, but
        when element i moves we compensate air_after of i so element i+1 keeps
        its absolute front Z.
        """
        new_front = self._clamp_element_front_z(elem_index, new_front)
        layout = self._element_layout()
        enabled = list(layout)
        enabled_indices = [h["index"] for h in enabled]
        if elem_index not in enabled_indices:
            return
        pos = enabled_indices.index(elem_index)
        h = enabled[pos]
        old_front = float(h["front_z"])
        thick = float(h["thickness"])
        # Absolute front of the next enabled element (held fixed)
        next_front = None
        if pos + 1 < len(enabled):
            next_front = float(enabled[pos + 1]["front_z"])

        if pos == 0:
            self.v_lens_z.set(round(new_front, 3))
        else:
            prev_idx = enabled_indices[pos - 1]
            prev_ev = self.elem_vars[prev_idx]
            prev_h = enabled[pos - 1]
            prev_rear = float(prev_h["rear_z"])
            air_before = max(0.05, new_front - prev_rear)
            prev_ev["air_after"].set(round(air_before, 3))

        # Compensate this element's air_after so the next stays put
        if next_front is not None:
            air_after = max(0.05, next_front - (new_front + thick))
            self.elem_vars[elem_index]["air_after"].set(round(air_after, 3))

        self.status_var.set(
            f"Moved element {elem_index + 1} front → Z = {new_front:.2f} mm "
            f"(neighbours held fixed)"
        )

    def _apply_element_radius(self, elem_index: int, which: str, new_R: float) -> None:
        """Set R1 (front) or R2 (rear). which is 'front' or 'rear'."""
        if elem_index < 0 or elem_index >= len(self.elem_vars):
            return
        ev = self.elem_vars[elem_index]
        R = float(new_R)
        # Keep |R| in a practical range; allow plano via large |R| ≈ 0 set as 0
        if abs(R) < 0.5:
            R = 0.0
        else:
            R = max(-500.0, min(500.0, R))
        key = "R1" if which == "front" else "R2"
        ev[key].set(round(R, 3))
        # Keep Y-radius sliders in step for rotational and cylinder_y so the
        # UI and the powered meridian never diverge. Biconic keeps R*y independent.
        mode = str(ev["surface_mode"].get())
        if mode in ("rotational", "cylinder_y"):
            ykey = "R1y" if which == "front" else "R2y"
            if ykey in ev:
                ev[ykey].set(round(R, 3))
        self.status_var.set(f"Element {elem_index + 1} {key} → {R:.2f} mm")

    @staticmethod
    def _radius_from_vertex_drag(
        ap: float, rim_z: float, vertex_z: float, sign_hint: float
    ) -> float:
        """
        Sphere radius from axial vertex Z and fixed rim Z.
        sag = rim_z - vertex_z; R = (ap² + sag²) / (2 sag) with sign.
        """
        sag = float(rim_z) - float(vertex_z)
        ap = max(float(ap), 0.5)
        if abs(sag) < 1e-4:
            # Nearly plano
            return 0.0
        R = (ap * ap + sag * sag) / (2.0 * sag)
        # Prefer continuity with previous sign when |sag| is tiny near flip
        if sign_hint != 0.0 and abs(R) > 1e-6:
            if R * sign_hint < 0 and abs(sag) < 0.05:
                R = -R
        return max(-500.0, min(500.0, R))

    def _on_side_press(self, event):
        if event.inaxes not in self._side_axes():
            return
        # Double-click = reset zoom (any button except while dragging lens)
        if getattr(event, "dblclick", False) and self._drag is None:
            self._reset_side_zoom()
            return
        # Right / middle = pan
        if event.button in (2, 3) and event.x is not None and event.y is not None:
            self._side_pan = {
                "xpress": event.x,
                "ypress": event.y,
                "xlim": event.inaxes.get_xlim(),
                "ylim": event.inaxes.get_ylim(),
                "axes": event.inaxes,
            }
            self.canvas_side.get_tk_widget().configure(cursor="fleur")
            return
        # Left = drag lens element (move) or top/bottom rim (resize aperture)
        # or absorbing blocker (Z move)
        if event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        # Cancel an in-flight trace so the grab stays live (geometry preview
        # only). Re-trace runs on mouse release.
        self._cancel_trace_work()
        use_xz = event.inaxes is getattr(self, "ax_side_xz", None)
        fov_hit = self._pick_fov_side(
            float(event.xdata), float(event.ydata), use_xz=bool(use_xz)
        )
        if fov_hit is not None and fov_hit[0] == "resize":
            _mode, h, which = fov_hit
            end_t = float(h["t_lo"] if which == "lo" else h["t_hi"])
            self._drag = {
                "mode": "fov_resize",
                "elem_index": -1,
                "use_xz": bool(use_xz),
                "which": which,
                "orig_center": float(h["t_c"]),
                "orig_size": float(h["size"]),
                "current_end": end_t,
                "label": "FOV width" if use_xz else "FOV height",
            }
            self.canvas_side.get_tk_widget().configure(cursor="sb_v_double_arrow")
            self.status_var.set(
                f"Resizing {self._drag['label']}  ·  {float(h['size']):.1f} mm"
            )
            return
        # Prefer blockers near the click (small pick radius on Z)
        blk = self._pick_blocker(float(event.xdata), float(event.ydata))
        if blk is not None:
            bi, bz = blk
            self._drag = {
                "mode": "blocker_move",
                "blocker_index": bi,
                "elem_index": -1,
                "orig_z": bz,
                "current_z": bz,
                "press_z": float(event.xdata),
                "press_y": float(event.ydata),
                "label": self.blockers[bi].get("label", f"Blocker {bi + 1}"),
            }
            self.canvas_side.get_tk_widget().configure(cursor="sb_h_double_arrow")
            self.status_var.set(
                f"Dragging blocker  ·  Z = {bz:.2f} mm  ·  release to re-trace"
            )
            return
        picked = self._pick_element_interaction(float(event.xdata), float(event.ydata))
        if picked is None:
            if fov_hit is not None and fov_hit[0] == "move":
                h = fov_hit[1]
                self._drag = {
                    "mode": "fov_move",
                    "elem_index": -1,
                    "use_xz": bool(use_xz),
                    "orig_center": float(h["t_c"]),
                    "current_center": float(h["t_c"]),
                    "press_t": float(event.ydata),
                    "label": "FOV center X" if use_xz else "FOV center Y",
                }
                self.canvas_side.get_tk_widget().configure(cursor="fleur")
                self.status_var.set(f"Moving {self._drag['label']}")
                return
            return
        hit, mode = picked
        idx = int(hit["index"])
        ev = self.elem_vars[idx] if 0 <= idx < len(self.elem_vars) else None
        R1 = float(ev["R1"].get()) if ev else 0.0
        R2 = float(ev["R2"].get()) if ev else 0.0
        ap = float(hit["aperture"])
        zf, zr = float(hit["front_z"]), float(hit["rear_z"])
        # Rim Z from current radius (sag at aperture) — held fixed while vertex moves
        def _rim_z(z_vertex: float, R: float) -> float:
            if abs(R) < 1e-9:
                return z_vertex
            c = 1.0 / R
            r2 = ap * ap
            disc = 1.0 - c * c * r2
            if disc < 0:
                return z_vertex
            sag = (c * r2) / (1.0 + math.sqrt(max(0.0, disc)))
            return z_vertex + sag

        max_ap = None
        if mode == "resize":
            max_ap = self._max_drawable_aperture(idx, LENS_SEMI_MAX_MM)
            # Never start past the drawable limit
            ap = min(ap, max_ap)
        self._drag = {
            "mode": mode,
            "elem_index": idx,
            "label": hit["label"],
            "orig_front": zf,
            "press_z": float(event.xdata),
            "press_y": float(event.ydata),
            "current_front": zf,
            "aperture": ap,
            "orig_aperture": ap,
            "current_aperture": ap,
            "max_aperture": max_ap,
            "thickness": hit["thickness"],
            "orig_R1": R1,
            "orig_R2": R2,
            "current_R1": R1,
            "current_R2": R2,
            "rim_front_z": _rim_z(zf, R1),
            "rim_rear_z": _rim_z(zr, R2),
        }
        if mode == "resize":
            self.canvas_side.get_tk_widget().configure(cursor="sb_v_double_arrow")
            self.status_var.set(
                f"Resizing {hit['label']}  ·  semi-aperture = {ap:.2f} mm  ·  release to re-trace"
            )
        elif mode == "radius_front":
            self.canvas_side.get_tk_widget().configure(cursor="sb_h_double_arrow")
            self.status_var.set(
                f"Front radius {hit['label']}  ·  R₁ = {R1:.2f} mm  ·  drag vertex · release to re-trace"
            )
        elif mode == "radius_rear":
            self.canvas_side.get_tk_widget().configure(cursor="sb_h_double_arrow")
            self.status_var.set(
                f"Rear radius {hit['label']}  ·  R₂ = {R2:.2f} mm  ·  drag vertex · release to re-trace"
            )
        else:
            self.canvas_side.get_tk_widget().configure(cursor="sb_h_double_arrow")
            self.status_var.set(
                f"Dragging {hit['label']}  ·  Z = {zf:.2f} mm  ·  neighbours stay fixed"
            )

    def _on_side_motion(self, event):
        # Pan takes priority when active
        if self._side_pan is not None:
            if event.x is None or event.y is None:
                return
            ax = self._side_pan.get("axes") or self.ax_side
            if ax is None:
                return
            try:
                inv = ax.transData.inverted()
                p0 = inv.transform((self._side_pan["xpress"], self._side_pan["ypress"]))
                p1 = inv.transform((event.x, event.y))
            except Exception:
                return
            dx = p0[0] - p1[0]
            dy = p0[1] - p1[1]
            x0, x1 = self._side_pan["xlim"]
            y0, y1 = self._side_pan["ylim"]
            self._side_xlim = (x0 + dx, x1 + dx)
            use_xz = ax is getattr(self, "ax_side_xz", None)
            if use_xz:
                self._side_xz_ylim = (y0 + dy, y1 + dy)
            else:
                self._side_ylim = (y0 + dy, y1 + dy)
            self._apply_side_limits(ax, plane="xz" if use_xz else "yz")
            # Shared Z on the sibling panel
            for other in self._side_axes():
                if other is not ax:
                    other.set_xlim(self._side_xlim)
            self.canvas_side.draw_idle()
            return

        if self._drag is None:
            if event.inaxes in self._side_axes() and event.xdata is not None and event.ydata is not None:
                hover_xz = event.inaxes is getattr(self, "ax_side_xz", None)
                fov_h = self._pick_fov_side(
                    float(event.xdata), float(event.ydata), use_xz=bool(hover_xz)
                )
                if fov_h is not None and fov_h[0] == "resize":
                    cur = "sb_v_double_arrow"
                    self.canvas_side.get_tk_widget().configure(cursor=cur)
                    return
                if self._pick_blocker(float(event.xdata), float(event.ydata)) is not None:
                    cur = "sb_h_double_arrow"
                else:
                    picked = self._pick_element_interaction(
                        float(event.xdata), float(event.ydata)
                    )
                    if picked is None:
                        cur = "hand2"
                    elif picked[1] == "resize":
                        cur = "sb_v_double_arrow"
                    elif picked[1] in ("radius_front", "radius_rear"):
                        cur = "sb_h_double_arrow"
                    else:
                        cur = "sb_h_double_arrow"
                self.canvas_side.get_tk_widget().configure(cursor=cur)
            return
        if event.inaxes not in self._side_axes():
            return
        mode = self._drag.get("mode", "move")
        if mode == "fov_resize":
            if event.ydata is None:
                return
            self._drag["current_end"] = float(event.ydata)
            _c, new_size = resize_fov_from_end(
                float(self._drag["orig_center"]),
                float(self._drag["orig_size"]),
                float(event.ydata),
            )
            self._set_fov_span(new_size, use_xz=bool(self._drag.get("use_xz")))
            self.status_var.set(
                f"Resizing {self._drag['label']}  ·  {new_size:.1f} mm"
            )
            self._draw_side()
            return
        if mode == "fov_move":
            if event.ydata is None:
                return
            dt = float(event.ydata) - float(self._drag["press_t"])
            new_c = float(self._drag["orig_center"]) + dt
            self._drag["current_center"] = new_c
            self._set_fov_center_axis(new_c, use_xz=bool(self._drag.get("use_xz")))
            self.status_var.set(
                f"Moving {self._drag['label']}  ·  {new_c:.2f} mm"
            )
            self._draw_side()
            return
        if mode == "blocker_move":
            if event.xdata is None:
                return
            dz = float(event.xdata) - self._drag["press_z"]
            target_z = float(self.v_target_z.get()) if hasattr(self, "v_target_z") else 200.0
            new_z = max(-5.0, min(target_z - 0.5, self._drag["orig_z"] + dz))
            self._drag["current_z"] = new_z
            self.status_var.set(
                f"Dragging blocker  ·  Z = {new_z:.2f} mm  ·  release to re-trace"
            )
            self._draw_side()
            return
        if mode == "resize":
            if event.ydata is None:
                return
            # Clamp to the max aperture the glass can still support so the
            # grey handles never leave the lens outline (element 3 symptom).
            requested = max(1.0, min(LENS_SEMI_MAX_MM, abs(float(event.ydata))))
            max_ap = self._drag.get("max_aperture")
            if max_ap is None:
                max_ap = self._max_drawable_aperture(
                    self._drag["elem_index"], requested
                )
                self._drag["max_aperture"] = max_ap
            new_ap = max(1.0, min(float(max_ap), requested))
            self._drag["current_aperture"] = new_ap
            note = ""
            if requested > float(max_ap) + 0.05:
                note = f"  ·  limited by edge thickness (max {max_ap:.2f} mm)"
            self.status_var.set(
                f"Resizing {self._drag['label']}  ·  semi-aperture = {new_ap:.2f} mm"
                f"{note}  ·  release to re-trace"
            )
            self._draw_side()
            return
        if mode in ("radius_front", "radius_rear"):
            if event.xdata is None:
                return
            ap = float(self._drag["aperture"])
            if mode == "radius_front":
                rim_z = float(self._drag["rim_front_z"])
                sign_hint = float(self._drag.get("orig_R1", 0.0))
                new_R = self._radius_from_vertex_drag(
                    ap, rim_z, float(event.xdata), sign_hint
                )
                self._drag["current_R1"] = new_R
                # Live UI update so side-view geometry rebuilds with new R
                self._apply_element_radius(self._drag["elem_index"], "front", new_R)
                self.status_var.set(
                    f"Front radius {self._drag['label']}  ·  R₁ = {new_R:.2f} mm  ·  release to re-trace"
                )
            else:
                rim_z = float(self._drag["rim_rear_z"])
                sign_hint = float(self._drag.get("orig_R2", 0.0))
                new_R = self._radius_from_vertex_drag(
                    ap, rim_z, float(event.xdata), sign_hint
                )
                self._drag["current_R2"] = new_R
                self._apply_element_radius(self._drag["elem_index"], "rear", new_R)
                self.status_var.set(
                    f"Rear radius {self._drag['label']}  ·  R₂ = {new_R:.2f} mm  ·  release to re-trace"
                )
            self._draw_side()
            return
        if event.xdata is None:
            return
        dz = float(event.xdata) - self._drag["press_z"]
        new_front = self._clamp_element_front_z(
            self._drag["elem_index"], self._drag["orig_front"] + dz
        )
        self._drag["current_front"] = new_front
        self.status_var.set(
            f"Dragging {self._drag['label']}  ·  Z = {new_front:.2f} mm  ·  neighbours stay fixed"
        )
        self._draw_side()

    def _on_side_release(self, event):
        if self._side_pan is not None:
            self._side_pan = None
            self.canvas_side.get_tk_widget().configure(cursor="hand2")
            return
        if self._drag is None:
            return
        drag = self._drag
        self._drag = None
        self.canvas_side.get_tk_widget().configure(cursor="hand2")
        mode = drag.get("mode", "move")
        if mode in ("fov_resize", "fov_move"):
            if mode == "fov_resize":
                _c, new_size = resize_fov_from_end(
                    float(drag.get("orig_center", 0.0)),
                    float(drag.get("orig_size", 40.0)),
                    float(drag.get("current_end", 0.0)),
                )
                self._set_fov_span(new_size, use_xz=bool(drag.get("use_xz")))
                msg = f"{drag.get('label', 'FOV')} = {new_size:.1f} mm — tracing…"
            else:
                self._set_fov_center_axis(
                    float(drag.get("current_center", 0.0)),
                    use_xz=bool(drag.get("use_xz")),
                )
                msg = f"{drag.get('label', 'FOV')} — tracing…"
            self._refresh_fov_overlay()
            self._request_trace_after_edit(msg)
            return
        if mode == "blocker_move":
            bi = int(drag.get("blocker_index", -1))
            new_z = float(drag.get("current_z", drag.get("orig_z", 0)))
            if bi < 0 or bi >= len(self.blockers):
                self._draw_side()
                return
            if abs(new_z - float(drag.get("orig_z", 0))) < 1e-4:
                self.status_var.set("Ready")
                self._draw_side()
                return
            self.blockers[bi] = dict(self.blockers[bi])
            self.blockers[bi]["z"] = round(new_z, 3)
            if hasattr(self, "blk_list"):
                self.blk_list.selection_clear(0, tk.END)
                self.blk_list.selection_set(bi)
                self._load_blocker_to_vars(bi)
            self._refresh_blocker_listbox()
            if self._debounce_id is not None:
                self.after_cancel(self._debounce_id)
                self._debounce_id = None
            if self._debounce_id is not None:
                self.after_cancel(self._debounce_id)
                self._debounce_id = None
            self._request_trace_after_edit(
                f"{drag.get('label', 'Blocker')} Z = {new_z:.2f} mm — tracing…"
            )
            return
        if mode == "resize":
            new_ap = float(drag.get("current_aperture", drag["orig_aperture"]))
            if abs(new_ap - float(drag["orig_aperture"])) < 1e-3:
                self.status_var.set("Ready")
                self._draw_side()
                return
            self._apply_element_aperture(drag["elem_index"], new_ap)
            if self._debounce_id is not None:
                self.after_cancel(self._debounce_id)
                self._debounce_id = None
            self._request_trace_after_edit(
                f"{drag['label']} semi-aperture = {new_ap:.2f} mm — tracing…"
            )
            return
        if mode in ("radius_front", "radius_rear"):
            which = "front" if mode == "radius_front" else "rear"
            key = "current_R1" if which == "front" else "current_R2"
            orig_key = "orig_R1" if which == "front" else "orig_R2"
            new_R = float(drag.get(key, drag.get(orig_key, 0.0)))
            orig_R = float(drag.get(orig_key, 0.0))
            if abs(new_R - orig_R) < 1e-3:
                self.status_var.set("Ready")
                self._draw_side()
                return
            # Already applied live during motion; just re-trace
            if self._debounce_id is not None:
                self.after_cancel(self._debounce_id)
                self._debounce_id = None
            rlabel = "R₁" if which == "front" else "R₂"
            self._request_trace_after_edit(
                f"{drag['label']} {rlabel} = {new_R:.2f} mm — tracing…"
            )
            return
        new_front = drag["current_front"]
        if abs(new_front - drag["orig_front"]) < 1e-4:
            self.status_var.set("Ready")
            self._draw_side()
            return
        self._apply_element_front_z(drag["elem_index"], new_front)
        if self._debounce_id is not None:
            self.after_cancel(self._debounce_id)
            self._debounce_id = None
        self._request_trace_after_edit(
            f"{drag['label']} at Z = {new_front:.2f} mm — tracing…"
        )

    @staticmethod
    def _elem_index_from_label(label: str) -> Optional[int]:
        if label.startswith("MLA"):
            return 0
        if label.startswith("E") and "S" in label:
            try:
                return int(label[1 : label.index("S")]) - 1
            except ValueError:
                return None
        return None

    def _surface_pairs(self, surfaces) -> list:
        """Group consecutive front/rear surfaces of each element (and MLA lenslets)."""
        pairs = []
        i = 0
        while i < len(surfaces) - 1:
            s1, s2 = surfaces[i], surfaces[i + 1]
            # Front then rear: same element label family
            e1 = self._elem_index_from_label(s1.label)
            e2 = self._elem_index_from_label(s2.label)
            same_mla = s1.label.startswith("MLA") and s2.label.startswith("MLA")
            same_mla = same_mla and s1.label.rsplit("S", 1)[0] == s2.label.rsplit("S", 1)[0]
            same_e = e1 is not None and e1 == e2 and s1.label.endswith("S1") and s2.label.endswith("S2")
            if same_e or same_mla:
                pairs.append((s1, s2, e1 if e1 is not None else 0))
                i += 2
            else:
                i += 1
        return pairs

    def _draw_lens_body(
        self,
        ax,
        s1,
        s2,
        z_off: float = 0.0,
        highlight: bool = False,
        compact: bool = False,
        plane: str = "yz",
    ):
        """
        Draw one lens as a closed meridional section.

        plane='yz' — cut along Y (shows cylinder_y / rotational Y power)
        plane='xz' — cut along X (shows cylinder_x / rotational X power)
        Decentered MLA lenslets use the lenslet centre in that plane.
        """
        from engine import max_aperture_positive_edge

        # Tiny lenslets: allow thinner edge for drawing/trace clamp
        min_edge = 0.05 if compact else 0.25
        ap_req = min(s1.aperture, s2.aperture)
        ap = max_aperture_positive_edge(s1, s2, ap_req, min_edge=min_edge)
        if ap < 1e-4:
            # Still show a minimal plate so MLA is visible
            ap = max(ap_req, 0.15)
        nseg = 12 if compact else 36
        use_xz = str(plane).lower().startswith("x")
        t0 = float(s1.x0) if use_xz else float(s1.y0)
        z_front_u, t_front_u = [], []
        z_rear_u, t_rear_u = [], []
        for i in range(nseg + 1):
            r = ap * i / nseg
            if use_xz:
                sag1 = s1.sag_xy(r, 0.0)
                sag2 = s2.sag_xy(r, 0.0)
            else:
                sag1 = s1.sag_xy(0.0, r)
                sag2 = s2.sag_xy(0.0, r)
            if sag1 is None or sag2 is None:
                break
            zf = s1.z_vertex + sag1 + z_off
            zr = s2.z_vertex + sag2 + z_off
            if zr < zf + min_edge * 0.5:
                break
            z_front_u.append(zf)
            t_front_u.append(t0 + r)
            z_rear_u.append(zr)
            t_rear_u.append(t0 + r)
        if len(z_front_u) < 2:
            return

        # Lower rim about lenslet axis
        z_front_l = list(z_front_u)
        t_front_l = [t0 - (t - t0) for t in t_front_u]
        z_rear_l = list(z_rear_u)
        t_rear_l = [t0 - (t - t0) for t in t_rear_u]

        poly_z = (
            z_front_u
            + list(reversed(z_rear_u))
            + z_rear_l
            + list(reversed(z_front_l))
        )
        poly_t = (
            t_front_u
            + list(reversed(t_rear_u))
            + t_rear_l
            + list(reversed(t_front_l))
        )

        col = "#fbbf24" if highlight else LENS
        lw = 1.2 if compact else (2.0 if highlight else 1.8)
        ax.fill(poly_z, poly_t, color=col, alpha=0.22 if not highlight else 0.35, zorder=3)
        ax.plot(z_front_u, t_front_u, color=col, lw=lw, zorder=4)
        ax.plot(z_front_l, t_front_l, color=col, lw=lw, zorder=4)
        ax.plot(z_rear_u, t_rear_u, color=col, lw=lw, zorder=4)
        ax.plot(z_rear_l, t_rear_l, color=col, lw=lw, zorder=4)
        ax.plot(
            [z_front_u[-1], z_rear_u[-1]],
            [t_front_u[-1], t_rear_u[-1]],
            color=col,
            lw=max(1.0, lw - 0.2),
            zorder=4,
        )
        ax.plot(
            [z_front_l[-1], z_rear_l[-1]],
            [t_front_l[-1], t_rear_l[-1]],
            color=col,
            lw=max(1.0, lw - 0.2),
            zorder=4,
        )

    def _draw_mla_plate_section(
        self,
        ax,
        params: Dict[str, Any],
        dies,
        z_off: float = 0.0,
        highlight: bool = False,
    ) -> bool:
        """
        Continuous Y–Z cut of the monolithic MLA plate (lenslet sags + land).
        Cut plane is the die column nearest X=0 so the side view shows a real MLA.
        """
        try:
            from mla_geometry import (
                build_mla_lens_specs,
                front_z_at,
                rear_z_at,
                land_sags,
            )
        except Exception:
            return False

        specs, meta = build_mla_lens_specs(params, dies=list(dies) if dies else None)
        if not specs or not meta:
            return False

        # Choose cut column closest to X = 0
        xs = sorted({round(s.x0, 6) for s in specs})
        x_cut = min(xs, key=lambda x: abs(x)) if xs else 0.0

        ys = [s.y0 for s in specs if abs(s.x0 - x_cut) < 1e-6] or [s.y0 for s in specs]
        ap = float(meta["aperture"])
        y_min = min(ys) - ap - 0.15
        y_max = max(ys) + ap + 0.15
        n = max(80, int((y_max - y_min) / max(ap * 0.08, 0.02)))
        n = min(n, 400)

        land_f, land_r = land_sags(specs)
        y_samples = [y_min + (y_max - y_min) * i / n for i in range(n + 1)]
        zf, zr, yy = [], [], []
        for y in y_samples:
            z1 = front_z_at(x_cut, y, specs, land_f) + z_off
            z2 = rear_z_at(x_cut, y, specs, land_r) + z_off
            if z2 < z1 + 0.05:
                z2 = z1 + 0.05
            zf.append(z1)
            zr.append(z2)
            yy.append(y)

        if len(yy) < 3:
            return False

        col = "#fbbf24" if highlight else LENS
        poly_z = zf + list(reversed(zr))
        poly_y = yy + list(reversed(yy))
        ax.fill(poly_z, poly_y, color=col, alpha=0.28 if not highlight else 0.4, zorder=3)
        ax.plot(zf, yy, color=col, lw=1.8 if not highlight else 2.2, zorder=4)
        ax.plot(zr, yy, color=col, lw=1.6 if not highlight else 2.0, zorder=4)
        # End caps
        ax.plot([zf[0], zr[0]], [yy[0], yy[0]], color=col, lw=1.2, zorder=4)
        ax.plot([zf[-1], zr[-1]], [yy[-1], yy[-1]], color=col, lw=1.2, zorder=4)

        # Mark lenslet centers on this cut
        for s in specs:
            if abs(s.x0 - x_cut) < 1e-6:
                ax.plot(
                    [s.z_front + z_off],
                    [s.y0],
                    "o",
                    color=col,
                    ms=3.5,
                    zorder=5,
                    alpha=0.9,
                )
        return True

    def _draw_side(self):
        """Draw Y–Z and orthogonal X–Z side views (stacked subplots)."""
        res = self.result
        if res is None:
            return
        p = self.collect_params()
        target_z = p["target_z"]
        layout = self._element_layout(p)
        self._element_handles = layout
        self._blocker_handles = []
        self._fov_side_handles = []

        drag_idx = None
        drag_dz = 0.0
        drag_mode = "move"
        drag_ap = None
        if self._drag is not None:
            drag_idx = self._drag["elem_index"]
            drag_mode = self._drag.get("mode", "move")
            drag_dz = (
                self._drag["current_front"] - self._drag["orig_front"]
                if drag_mode == "move"
                else 0.0
            )
            if drag_mode == "resize":
                drag_ap = float(self._drag.get("current_aperture", self._drag["orig_aperture"]))

        try:
            from engine import assemble_surfaces, build_source_array

            live_dies = build_source_array(p["source"])
            mla_p = p.get("mla") or {}
            # During resize, inject the live aperture so the glass body grows
            # with the handles (otherwise body stays at the last-committed size).
            els = [dict(e) for e in p["elements"]]
            if (
                drag_idx is not None
                and drag_mode == "resize"
                and drag_ap is not None
                and 0 <= drag_idx < len(els)
            ):
                els[drag_idx] = dict(els[drag_idx])
                els[drag_idx]["aperture"] = float(drag_ap)
                if els[drag_idx].get("aperture_y") is not None:
                    els[drag_idx]["aperture_y"] = float(drag_ap)
            # Live Z for a dragged blocker
            live_blockers = [dict(b) for b in (p.get("blockers") or [])]
            if (
                self._drag is not None
                and self._drag.get("mode") == "blocker_move"
                and "blocker_index" in self._drag
            ):
                bi = int(self._drag["blocker_index"])
                if 0 <= bi < len(live_blockers):
                    live_blockers[bi] = dict(live_blockers[bi])
                    live_blockers[bi]["z"] = float(self._drag.get("current_z", live_blockers[bi].get("z", 0)))
            live_surfs = assemble_surfaces(
                els,
                float(p["lens_z_start"]),
                mla_p if mla_p.get("enabled") else None,
                live_dies if mla_p.get("enabled") else None,
                blockers=live_blockers,
            )
            if not live_surfs:
                live_surfs = res.surfaces
        except Exception:
            live_surfs = res.surfaces
            live_dies = list(res.dies)
        pairs = self._surface_pairs(live_surfs)
        mla_mode = any(s.label.startswith("MLA") for s in live_surfs)

        # Shared Z extent; independent transverse extents per plane
        t_ext_y = 12.0
        t_ext_x = 12.0
        live_src = live_dies if live_dies else list(res.dies)
        for d in live_src:
            t_ext_y = max(t_ext_y, abs(d.cy) + d.height / 2 + 2)
            t_ext_x = max(t_ext_x, abs(d.cx) + d.width / 2 + 2)
        for s in res.surfaces:
            t_ext_y = max(t_ext_y, abs(s.y0) + s.aperture * 1.2)
            t_ext_x = max(t_ext_x, abs(s.x0) + s.aperture * 1.2)
        for h in layout:
            ap_h = (
                drag_ap
                if (drag_idx is not None and h["index"] == drag_idx and drag_ap is not None)
                else h["aperture"]
            )
            t_ext_y = max(t_ext_y, float(ap_h) * 1.15)
            t_ext_x = max(t_ext_x, float(ap_h) * 1.15)
        if drag_ap is not None:
            t_ext_y = max(t_ext_y, drag_ap * 1.2)
            t_ext_x = max(t_ext_x, drag_ap * 1.2)
        # Include the FOV bar so a tall/wide field does not look like a Y collapse
        try:
            t_ext_y = max(t_ext_y, abs(float(p["fov_cy"])) + float(p["fov_height"]) * 0.5 + 2.0)
            t_ext_x = max(t_ext_x, abs(float(p["fov_cx"])) + float(p["fov_width"]) * 0.5 + 2.0)
        except (TypeError, ValueError, KeyError):
            pass
        z0 = min((d.cz for d in res.dies), default=0) - 5
        z_max_optics = max((s.z_vertex for s in res.surfaces), default=10)
        if self._drag is not None and drag_mode == "move":
            z_max_optics = max(
                z_max_optics, self._drag["current_front"] + self._drag["thickness"]
            )
        z1 = max(target_z, z_max_optics) + 10
        self._side_full_extent = side_full_extent(z0, z1, t_ext_y, t_ext_x)

        ctx = dict(
            res=res,
            p=p,
            target_z=target_z,
            live_src=live_src,
            layout=layout,
            pairs=pairs,
            mla_mode=mla_mode,
            drag_idx=drag_idx,
            drag_dz=drag_dz,
            drag_mode=drag_mode,
            drag_ap=drag_ap,
            z0=z0,
            z1=z1,
            live_surfs=live_surfs,
        )
        self._paint_side_plane(self.ax_side, plane="yz", t_ext=t_ext_y, **ctx)
        if hasattr(self, "ax_side_xz"):
            self._paint_side_plane(self.ax_side_xz, plane="xz", t_ext=t_ext_x, **ctx)

        n_lenslets = sum(
            1 for s in res.surfaces if s.label.endswith("S1") and s.label.startswith("MLA")
        )
        title_y = "SIDE VIEW  ·  Y–Z  ·  cylinder_y / rotational"
        title_x = "SIDE VIEW  ·  X–Z  ·  cylinder_x / rotational (orthogonal)"
        if n_lenslets:
            title_y = f"Y–Z  ·  MLA {n_lenslets} lenslets"
            title_x = f"X–Z  ·  MLA {n_lenslets} lenslets"
        if self._drag is not None:
            mode = self._drag.get("mode", "move")
            lab = self._drag["label"]
            if mode == "move":
                note = f"moving {lab} → Z={self._drag['current_front']:.2f}"
            elif mode == "resize":
                note = f"resize {lab} → ap={self._drag.get('current_aperture', 0):.2f}"
            elif mode == "radius_front":
                note = f"R₁ {lab} → {self._drag.get('current_R1', 0):.2f}"
            elif mode == "radius_rear":
                note = f"R₂ {lab} → {self._drag.get('current_R2', 0):.2f}"
            else:
                note = lab
            title_y = f"Y–Z  ·  {note}"
            title_x = f"X–Z  ·  {note}"
        self.ax_side.set_title(title_y, loc="left", fontsize=9, color=FG_BRIGHT)
        self.ax_side.set_ylabel("Y (mm)", color="#f8fafc")
        if hasattr(self, "ax_side_xz"):
            self.ax_side_xz.set_title(title_x, loc="left", fontsize=9, color=FG_BRIGHT)
            self.ax_side_xz.set_xlabel("Z (mm)", color="#f8fafc")
            self.ax_side_xz.set_ylabel("X (mm)", color="#f8fafc")
        # Re-assert light text — set_title can reset to matplotlib default (black)
        for ax in self._side_axes():
            ax.title.set_color(FG_BRIGHT)
            ax.xaxis.label.set_color("#f8fafc")
            ax.yaxis.label.set_color("#f8fafc")
            ax.tick_params(colors="#f8fafc")
        self._apply_side_limits(self.ax_side)
        if hasattr(self, "ax_side_xz"):
            self._apply_side_limits(self.ax_side_xz, plane="xz")
        if not getattr(self, "_side_layout_done", False):
            try:
                self.fig_side.tight_layout()
            except Exception:
                pass
            self._side_layout_done = True
        self.canvas_side.draw_idle()

    def _draw_blockers_side(self, ax, p, *, plane: str, use_xz: bool):
        """Draw absorbing bodies / tubes / stops in the side-view cut."""
        handles = []
        blockers = p.get("blockers") or []
        drag_bi = None
        drag_z = None
        if self._drag is not None and self._drag.get("mode") == "blocker_move":
            drag_bi = int(self._drag.get("blocker_index", -1))
            drag_z = float(self._drag.get("current_z", 0))
        for i, b in enumerate(blockers):
            if not b.get("enabled", True):
                continue
            z = float(b.get("z", 0))
            if drag_bi is not None and i == drag_bi and drag_z is not None:
                z = drag_z
            shape = str(b.get("shape", "rect") or "rect").lower()
            orient = str(
                b.get("orient") or ("tube" if shape == "circle" else "horizontal")
            ).lower()
            if orient in ("face", "stop", "z"):
                orient = "vertical"
            ow = float(b.get("outer_w", 15) or 15)
            oh = float(b.get("outer_h", ow) or ow)
            iw = float(b.get("inner_w", 0) or 0)
            ih = float(b.get("inner_h", 0) or 0)
            x0 = float(b.get("x0", 0) or 0)
            y0 = float(b.get("y0", 0) or 0)
            length = float(b.get("length", 0) or 0)
            if length <= 1e-9:
                length = max(float(b.get("thickness", 40) or 40), 1.0)
            thick = max(float(b.get("thickness", 1.0) or 1.0), 0.6)
            hi = 0.55 if (drag_bi is not None and i == drag_bi) else 0.38
            c0 = x0 if use_xz else y0
            # Transverse outer radius/half-size for this cut
            if orient == "tube" or shape == "circle" and orient != "vertical":
                outer_t = ow
            elif use_xz:
                outer_t = ow  # half-width in X
            else:
                outer_t = oh  # half-height in Y

            if orient == "vertical":
                # Face-on stop: short vertical slab at Z
                z_lo, z_hi = z - thick / 2, z + thick / 2
                ax.fill(
                    [z_lo, z_hi, z_hi, z_lo],
                    [c0 - outer_t, c0 - outer_t, c0 + outer_t, c0 + outer_t],
                    color=BLOCKER, alpha=hi, zorder=3.5,
                    edgecolor="#94a3b8", linewidth=0.8,
                )
                inner_t = iw if (shape == "circle" or use_xz) else ih
                if inner_t > 1e-6:
                    ax.fill(
                        [z_lo - 0.01, z_hi + 0.01, z_hi + 0.01, z_lo - 0.01],
                        [c0 - inner_t, c0 - inner_t, c0 + inner_t, c0 + inner_t],
                        color=BG, alpha=0.95, zorder=3.6,
                        edgecolor="#cbd5e1", linewidth=0.6,
                    )
            else:
                # Tube or horizontal body: long walls ‖ optical axis
                z_lo, z_hi = z - length / 2, z + length / 2
                # Top & bottom walls (horizontal lines in side view)
                for t_sign in (+1.0, -1.0):
                    t_wall = c0 + t_sign * outer_t
                    ax.plot(
                        [z_lo, z_hi], [t_wall, t_wall],
                        color=BLOCKER, lw=2.2, alpha=min(1.0, hi + 0.35), zorder=3.5,
                        solid_capstyle="butt",
                    )
                # End rings / rectangles at z_lo, z_hi
                for ze in (z_lo, z_hi):
                    ax.plot(
                        [ze, ze], [c0 - outer_t, c0 + outer_t],
                        color="#94a3b8", lw=0.9, alpha=0.7, zorder=3.4,
                    )
                if iw > 1e-6 and (orient == "tube" or shape == "circle"):
                    # Bore outline
                    for t_sign in (+1.0, -1.0):
                        ax.plot(
                            [z_lo, z_hi],
                            [c0 + t_sign * iw, c0 + t_sign * iw],
                            color="#cbd5e1", lw=0.8, alpha=0.5, ls="--", zorder=3.6,
                        )

            # Move handle at center
            ax.plot(
                [z], [c0],
                marker="s", markersize=7,
                color="#f87171",
                markeredgecolor="#fecaca",
                zorder=8,
            )
            lab = str(b.get("label") or f"BLK{i}")
            ax.text(
                z, c0 + outer_t * 1.05, f" {lab}",
                color="#94a3b8", fontsize=7, va="bottom", zorder=8,
            )
            handles.append({
                "index": i,
                "z": z,
                "t_lo": c0 - outer_t,
                "t_hi": c0 + outer_t,
                "label": lab,
                "plane": plane,
            })
        self._blocker_handles = list(getattr(self, "_blocker_handles", []) or []) + handles

    def _paint_side_plane(
        self,
        ax,
        *,
        plane: str,
        t_ext: float,
        res,
        p,
        target_z,
        layout,
        pairs,
        mla_mode,
        drag_idx,
        drag_dz,
        drag_mode,
        drag_ap,
        z0,
        z1,
        live_surfs,
        live_src=None,
    ):
        """Paint one meridional cut: plane 'yz' (transverse=Y) or 'xz' (transverse=X)."""
        ax.clear()
        self._style_axes()
        use_xz = str(plane).lower().startswith("x")
        # transverse index into history / die: 0=X, 1=Y
        ti = 0 if use_xz else 1

        ax.axhline(0, color="#3d5a73", lw=1.2, zorder=1)
        ax.set_xlim(z0, z1)
        ax.set_ylim(-t_ext, t_ext)

        # Source dies — live params so die size sliders update the gold bar immediately
        dies_draw = live_src if live_src else res.dies
        for die in dies_draw:
            if use_xz:
                t0 = die.cx - die.width / 2
                t1 = die.cx + die.width / 2
                tc = die.cx
            else:
                t0 = die.cy - die.height / 2
                t1 = die.cy + die.height / 2
                tc = die.cy
            ax.plot([die.cz, die.cz], [t0, t1], color=SOURCE, lw=3, solid_capstyle="round", zorder=5)
            ax.fill(
                [die.cz, die.cz + max(0.4, 0.15 * (z1 - z0) * 0.02), die.cz],
                [t0, tc, t1],
                color=SOURCE,
                alpha=0.25,
                zorder=4,
            )

        n_mla = sum(1 for s1, s2, _ in pairs if s1.label.startswith("MLA"))
        mla_drawn = False
        if mla_mode and n_mla > 0 and not use_xz:
            # MLA continuous plate section currently drawn in Y–Z only
            z_off_mla = drag_dz if (drag_idx is not None and drag_idx == 0) else 0.0
            mla_drawn = self._draw_mla_plate_section(
                ax,
                p,
                res.dies,
                z_off=z_off_mla,
                highlight=abs(z_off_mla) > 1e-9,
            )
        for s1, s2, eidx in pairs:
            is_mla = s1.label.startswith("MLA")
            if is_mla and mla_drawn:
                continue
            z_off = drag_dz if (drag_idx is not None and eidx == drag_idx) else 0.0
            dragging_this = abs(z_off) > 1e-9 or (
                drag_idx is not None
                and eidx == drag_idx
                and drag_mode in ("radius_front", "radius_rear", "resize")
            )
            self._draw_lens_body(
                ax,
                s1,
                s2,
                z_off,
                highlight=dragging_this,
                compact=is_mla or n_mla > 4,
                plane=plane,
            )

        # Absorbing blockers / aperture stops
        self._draw_blockers_side(ax, p, plane=plane, use_xz=use_xz)

        # Handles
        for h in layout:
            zf = h["front_z"]
            zr = h["rear_z"]
            ap = float(h["aperture"])
            if drag_idx is not None and h["index"] == drag_idx:
                if drag_mode == "move":
                    zf = self._drag["current_front"]
                    zr = zf + h["thickness"]
                if drag_ap is not None:
                    ap = drag_ap
            if mla_mode and h["index"] == 0 and res.dies:
                if use_xz:
                    ts = [d.cx for d in res.dies]
                else:
                    ts = [d.cy for d in res.dies]
                t_lo = min(ts) - ap
                t_hi = max(ts) + ap
                t_top, t_bot = t_hi, t_lo
            else:
                t_lo, t_hi = -ap * 0.2, ap * 0.2
                t_top, t_bot = ap, -ap
            active = drag_idx is not None and h["index"] == drag_idx
            resizing = active and drag_mode == "resize"
            moving = active and drag_mode == "move"
            rad_front = active and drag_mode == "radius_front"
            rad_rear = active and drag_mode == "radius_rear"
            z_handle = 0.5 * (zf + zr)
            ax.plot(
                [z_handle, z_handle],
                [t_lo, t_hi],
                color="#fbbf24" if moving else "#94a3b8",
                lw=1.8,
                solid_capstyle="round",
                zorder=6,
                linestyle="--" if moving else ":",
                alpha=0.9,
            )
            t_ax = 0 if not (mla_mode and h["index"] == 0) else 0.5 * (t_lo + t_hi)
            ax.plot(
                z_handle,
                t_ax,
                "s",
                color="#fbbf24",
                ms=6,
                zorder=7,
                markeredgecolor="#0b0f14",
            )
            for z_v, active_r in ((zf, rad_front), (zr, rad_rear)):
                ax.plot(
                    z_v,
                    t_ax,
                    "o",
                    color="#a78bfa" if active_r else "#c4b5fd",
                    ms=8 if active_r else 6,
                    zorder=9,
                    markeredgecolor="#0b0f14",
                    markeredgewidth=0.7,
                    alpha=0.95,
                )
            grip_col = "#38bdf8" if resizing else "#64748b"
            for tg in (t_top, t_bot):
                ax.plot(
                    [zf, zr],
                    [tg, tg],
                    color=grip_col,
                    lw=1.4 if resizing else 1.0,
                    solid_capstyle="round",
                    zorder=6,
                    linestyle="-" if resizing else "--",
                    alpha=0.95,
                )
                ax.plot(
                    z_handle,
                    tg,
                    "D",
                    color="#38bdf8" if resizing else "#94a3b8",
                    ms=7 if resizing else 5,
                    zorder=8,
                    markeredgecolor="#0b0f14",
                    markeredgewidth=0.6,
                )
            if resizing:
                ax.plot(
                    [zf, zr, zr, zf, zf],
                    [t_bot, t_bot, t_top, t_top, t_bot],
                    color="#38bdf8",
                    lw=1.2,
                    alpha=0.55,
                    zorder=5,
                    linestyle=":",
                )
            label = h["label"]
            if mla_mode and h["index"] == 0 and "MLA" not in label:
                label = label + " MLA"
            ax.text(
                z_handle,
                (t_hi if mla_mode and h["index"] == 0 else ap) * 1.05 + 0.4,
                label,
                color=FG_BRIGHT,
                fontsize=8,
                ha="center",
                va="bottom",
                zorder=7,
            )

        draw_rays = should_draw_display_rays(
            auto_run=bool(self.auto_run.get()) if hasattr(self, "auto_run") else True,
            dragging=self._drag is not None,
            preview=bool(getattr(self, "_draw_side_preview", False)),
        )
        if draw_rays:
            color_cls = bool(getattr(self, "v_color_partial", None) and self.v_color_partial.get())
            stack_need, only_mla = self._stack_element_counts(res)
            max_ap = max((float(h["aperture"]) for h in layout), default=10.0)
            slice_full = max(2.5, 0.35 * max_ap)
            slice_cut = max(8.0, 1.10 * max_ap)
            # Off-plane coordinate: |X| for Y–Z view, |Y| for X–Z view
            oi = 0 if not use_xz else 1
            for path in res.paths:
                if len(path.history) < 2:
                    continue
                max_off = max(abs(float(pt[oi])) for pt in path.history)
                if max_off > slice_cut:
                    continue
                if max_off <= slice_full:
                    off_scale = 1.0
                else:
                    off_scale = max(
                        0.08,
                        1.0 - (max_off - slice_full) / max(slice_cut - slice_full, 1e-6),
                    )
                ev = getattr(path, "events", None) or []
                through_lens = int(getattr(path, "n_refractions", 0) or 0) > 0 or any(
                    e == "refract" for e in ev
                )
                miss_scale = 1.0 if through_lens else 0.12
                cls = self._classify_path(path, stack_need, only_mla) if color_cls else None
                for j in range(len(path.history) - 1):
                    p0, p1 = path.history[j], path.history[j + 1]
                    kind = ev[j] if j < len(ev) else "refract"
                    is_launch = j == 0
                    if kind == "reflect":
                        col, al, lw = "#f97316", 0.85, 1.2
                    elif kind in ("tir_absorb", "kill", "absorb"):
                        col, al, lw = "#ef4444", 0.65, 1.0
                    elif kind == "ghost":
                        col, al, lw = "#64748b", 0.25, 0.5
                    elif cls == "miss":
                        col, al, lw = "#ff1493", 0.55, 0.75  # deep pink
                    elif cls == "partial":
                        col, al, lw = "#ff0000", 0.55, 0.85  # pure red
                    elif cls == "full":
                        col, al, lw = "#c6ff00", 0.50, 0.80  # electric lime
                    else:
                        col, al, lw = "#7dd3fc", 0.40, 0.75
                    # First segment is the emission cone — never fade it, or
                    # wide-angle rays that miss the lens vanish and the
                    # half-angle looks broken.
                    scale = off_scale if is_launch else miss_scale * off_scale
                    al = max(0.12 if is_launch else 0.04, al * scale)
                    if not through_lens and not color_cls and not is_launch:
                        lw = min(lw, 0.55)
                    elif is_launch:
                        lw = max(lw, 0.75)
                    ax.plot(
                        [p0[2], p1[2]],
                        [p0[ti], p1[ti]],
                        color=col,
                        alpha=al,
                        lw=lw,
                        zorder=2 if through_lens else 1,
                    )

        ax.axvline(target_z, color=TARGET, ls="--", lw=1.4, zorder=4)
        if use_xz:
            f_c = float(p["fov_cx"])
            f_span = float(p["fov_width"])
        else:
            f_c = float(p["fov_cy"])
            f_span = float(p["fov_height"])
        if (
            self._drag is not None
            and self._drag.get("mode") in ("fov_resize", "fov_move")
            and bool(self._drag.get("use_xz", False)) == bool(use_xz)
        ):
            if self._drag.get("mode") == "fov_resize":
                f_c, f_span = resize_fov_from_end(
                    float(self._drag.get("orig_center", f_c)),
                    float(self._drag.get("orig_size", f_span)),
                    float(self._drag.get("current_end", f_c + f_span / 2)),
                )
            else:
                f_c = float(self._drag.get("current_center", f_c))
        f0 = f_c - f_span / 2
        f1 = f_c + f_span / 2
        ax.plot([target_z, target_z], [f0, f1], color=FOV, lw=3, zorder=5)
        # Grab handles at the FOV bar ends (size) and mid-span (move)
        for end_t, which in ((f0, "lo"), (f1, "hi")):
            ax.plot(
                target_z,
                end_t,
                "D",
                color="#c4b5fd",
                ms=7,
                zorder=9,
                markeredgecolor="#0b0f14",
                markeredgewidth=0.6,
            )
        ax.plot(
            target_z,
            f_c,
            marker="s",
            markersize=6,
            color="#a78bfa",
            markeredgecolor="#ede9fe",
            zorder=9,
        )
        ax.text(target_z, t_ext * 0.92, " TARGET", color=TARGET, fontsize=8, va="top")
        ax.text(target_z, f1, " FOV", color=FOV, fontsize=8, va="bottom")
        self._fov_side_handles = list(getattr(self, "_fov_side_handles", []) or []) + [
            {
                "plane": "xz" if use_xz else "yz",
                "z": float(target_z),
                "t_lo": float(f0),
                "t_hi": float(f1),
                "t_c": float(f_c),
                "size": float(f_span),
            }
        ]


    def _connect_target_mouse(self):
        c = self.canvas_tgt
        c.mpl_connect("scroll_event", self._on_tgt_scroll)
        c.mpl_connect("button_press_event", self._on_tgt_press)
        c.mpl_connect("button_release_event", self._on_tgt_release)
        c.mpl_connect("motion_notify_event", self._on_tgt_motion)

    def _reset_tgt_zoom(self):
        self._tgt_xlim = None
        self._tgt_ylim = None
        self._tgt_pan = None
        if self.result is not None:
            self._draw_target()
        self.status_var.set("Target plane zoom reset")

    def _apply_tgt_limits(self, ax):
        """Restore zoom/pan after a full redraw; allow zoom-out onto a large wall."""
        if self._tgt_full_extent is None:
            return
        hx0, hx1, hy0, hy1 = self._tgt_full_extent
        if self._tgt_zoom_extent is None:
            self._tgt_zoom_extent = target_zoom_extent(
                (hx1 - hx0) * 0.5, (hy1 - hy0) * 0.5
            )
        zx0, zx1, zy0, zy1 = self._tgt_zoom_extent
        if self._tgt_xlim is None or self._tgt_ylim is None:
            ax.set_xlim(hx0, hx1)
            ax.set_ylim(hy0, hy1)
            ax.set_aspect("equal", adjustable="box")
            return
        x0, x1 = self._tgt_xlim
        y0, y1 = self._tgt_ylim
        x0, x1 = self._clamp_view_window(x0, x1, zx0, zx1, max_out=1.0)
        y0, y1 = self._clamp_view_window(y0, y1, zy0, zy1, max_out=1.0)
        self._tgt_xlim = (x0, x1)
        self._tgt_ylim = (y0, y1)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        # Keep data limits (the wall); shrink the axes box instead of
        # snapping back to the heatmap rectangle.
        ax.set_aspect("equal", adjustable="box")

    def _on_tgt_scroll(self, event):
        """Mouse wheel zoom on target plane, centered on cursor."""
        if self.ax_tgt is None or event.inaxes != self.ax_tgt:
            return
        if event.xdata is None or event.ydata is None:
            return
        ax = self.ax_tgt
        # button 'up' = zoom in, 'down' = zoom out (matplotlib)
        if event.button == "up":
            scale = 0.8
        elif event.button == "down":
            scale = 1.25
        else:
            return

        xdata, ydata = float(event.xdata), float(event.ydata)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        # Zoom toward mouse position
        new_w = (x1 - x0) * scale
        new_h = (y1 - y0) * scale
        # Keep aspect roughly equal (square zoom)
        side = max(new_w, new_h)
        # Minimum zoom window ~ 1% of heatmap or 0.5 mm; max = virtual wall
        if self._tgt_zoom_extent is not None:
            fx0, fx1, fy0, fy1 = self._tgt_zoom_extent
        elif self._tgt_full_extent is not None:
            fx0, fx1, fy0, fy1 = self._tgt_full_extent
        else:
            fx0 = fy0 = -TGT_WALL_HALF_MM
            fx1 = fy1 = TGT_WALL_HALF_MM
        min_side = max(0.5, 0.01 * max(fx1 - fx0, fy1 - fy0))
        max_side = max(fx1 - fx0, fy1 - fy0)
        side = max(min_side, min(side, max_side))

        relx = (xdata - x0) / max(x1 - x0, 1e-12)
        rely = (ydata - y0) / max(y1 - y0, 1e-12)
        nx0 = xdata - relx * side
        nx1 = nx0 + side
        ny0 = ydata - rely * side
        ny1 = ny0 + side

        self._tgt_xlim = (nx0, nx1)
        self._tgt_ylim = (ny0, ny1)
        self._apply_tgt_limits(ax)
        self.canvas_tgt.draw_idle()
        self._schedule_target_map_expand()

    def _on_tgt_press(self, event):
        if self.ax_tgt is None or event.inaxes != self.ax_tgt:
            return
        if event.button == 1 and getattr(event, "dblclick", False):
            self._reset_tgt_zoom()
            return
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            try:
                w = float(self.v_fov_w.get())
                h = float(self.v_fov_h.get())
                cx = float(self.v_fov_cx.get())
                cy = float(self.v_fov_cy.get())
            except (tk.TclError, ValueError):
                w = h = cx = cy = 0.0
            span = max(w, h, 1.0)
            hit = pick_fov_rect_handle(
                float(event.xdata),
                float(event.ydata),
                cx,
                cy,
                w,
                h,
                hit_r=max(1.2, 0.04 * span),
            )
            if hit is not None:
                self._cancel_trace_work()
                self._drag = {
                    "mode": "fov_tgt",
                    "elem_index": -1,
                    "kind": hit[0],
                    "name": hit[1],
                    "orig": (cx, cy, w, h),
                    "press": (float(event.xdata), float(event.ydata)),
                    "label": "FOV",
                }
                self.canvas_tgt.get_tk_widget().configure(
                    cursor="fleur" if hit[0] == "move" else "plus"
                )
                return
        if event.button in (2, 3) and event.x is not None and event.y is not None:
            self._tgt_pan = {
                "xpress": event.x,
                "ypress": event.y,
                "xlim": self.ax_tgt.get_xlim(),
                "ylim": self.ax_tgt.get_ylim(),
            }
            self.canvas_tgt.get_tk_widget().configure(cursor="fleur")

    def _schedule_target_map_expand(self):
        """After zoom/pan settles, grow the recorded map to cover the view."""
        if self._tgt_expand_id is not None:
            try:
                self.after_cancel(self._tgt_expand_id)
            except Exception:
                pass
        self._tgt_expand_id = self.after(280, self._maybe_expand_target_map)

    def _maybe_expand_target_map(self):
        self._tgt_expand_id = None
        if not AUTO_EXPAND_MAP_ON_ZOOM:
            return
        if self._tgt_xlim is None or self._tgt_ylim is None:
            return
        if self._tgt_pan is not None or self._live_edit.live or self._drag is not None:
            return
        try:
            cur_w = float(self.v_map_w.get())
            cur_h = float(self.v_map_h.get())
        except (tk.TclError, ValueError):
            return
        new_w, new_h, grew = map_half_to_cover_view(
            self._tgt_xlim, self._tgt_ylim, cur_w, cur_h
        )
        if not grew:
            return
        self.v_map_w.set(round(new_w, 1))
        self.v_map_h.set(round(new_h, 1))
        self.status_var.set(
            f"Target wall expanded to ±{new_w:.0f} × ±{new_h:.0f} mm — recording full illumination…"
        )
        if should_start_trace(
            auto_run=bool(self.auto_run.get()),
            live=self._live_edit.live,
            handle_drag=self._drag is not None,
        ):
            self.run_trace()

    def _on_tgt_release(self, event):
        if self._tgt_pan is not None:
            self._tgt_pan = None
            self._schedule_target_map_expand()
            self.canvas_tgt.get_tk_widget().configure(cursor="")
            return
        if self._drag is not None and self._drag.get("mode") == "fov_tgt":
            self._drag = None
            self.canvas_tgt.get_tk_widget().configure(cursor="")
            try:
                w = float(self.v_fov_w.get())
                h = float(self.v_fov_h.get())
            except (tk.TclError, ValueError):
                w = h = 0.0
            self._request_trace_after_edit(
                f"FOV {w:.1f} × {h:.1f} mm — tracing…"
            )

    def _on_tgt_motion(self, event):
        if (
            self._drag is not None
            and self._drag.get("mode") == "fov_tgt"
            and event.xdata is not None
            and event.ydata is not None
        ):
            cx0, cy0, w0, h0 = self._drag["orig"]
            px, py = self._drag["press"]
            kind = self._drag.get("kind")
            name = self._drag.get("name")
            lock = bool(self.v_fov_lock.get()) if hasattr(self, "v_fov_lock") else False
            if kind == "move":
                cx, cy = move_fov_center(
                    cx0, cy0, float(event.xdata) - px, float(event.ydata) - py
                )
                w, h = w0, h0
            else:
                cx, cy, w, h = resize_fov_rect(
                    cx0,
                    cy0,
                    w0,
                    h0,
                    edge=name if kind == "edge" else None,
                    corner=name if kind == "corner" else None,
                    x=float(event.xdata),
                    y=float(event.ydata),
                    lock_aspect=lock and kind == "edge",
                )
            self._set_fov_rect(cx, cy, w, h)
            self._refresh_fov_overlay()
            self.status_var.set(f"FOV {w:.1f} × {h:.1f} mm @ ({cx:.1f}, {cy:.1f})")
            return
        if self._tgt_pan is None or event.x is None or event.y is None:
            return
        if self.ax_tgt is None:
            return
        ax = self.ax_tgt
        try:
            inv = ax.transData.inverted()
            p0 = inv.transform((self._tgt_pan["xpress"], self._tgt_pan["ypress"]))
            p1 = inv.transform((event.x, event.y))
        except Exception:
            return
        dx = p0[0] - p1[0]
        dy = p0[1] - p1[1]
        x0, x1 = self._tgt_pan["xlim"]
        y0, y1 = self._tgt_pan["ylim"]
        self._tgt_xlim = (x0 + dx, x1 + dx)
        self._tgt_ylim = (y0 + dy, y1 + dy)
        self._apply_tgt_limits(ax)
        self.canvas_tgt.draw_idle()

    def _stack_element_counts(self, res) -> tuple:
        """Return (n_elements_required_for_full, only_mla)."""
        from engine import element_id_from_label

        stack_ids = []
        seen = set()
        for s in res.surfaces:
            eid = element_id_from_label(getattr(s, "label", "") or "")
            if not eid or eid in seen:
                continue
            seen.add(eid)
            stack_ids.append(eid)
        only_mla = bool(stack_ids) and all(x.startswith("MLA") for x in stack_ids)
        if only_mla:
            return 1, True
        bulk = [x for x in stack_ids if not x.startswith("MLA")]
        return max(1, len(bulk)), False

    @staticmethod
    def _path_in_meridional_slice(path, slice_half_mm: float) -> bool:
        from engine import path_in_meridional_slice

        return path_in_meridional_slice(path, slice_half_mm)

    def _classify_path(self, path, stack_need: int, only_mla: bool) -> str:
        """
        Classify by *refracted* element ids (true surface hits), not by the
        Y–Z silhouette. A ray can cross a lens outline in the side view while
        missing the circular clear aperture in 3D (large |X|).
        """
        hit = list(getattr(path, "elements_hit", None) or [])
        if only_mla:
            n_hit = 1 if hit else 0
            need = 1
        else:
            n_hit = len([x for x in hit if not str(x).startswith("MLA")])
            need = max(1, int(stack_need))
        if n_hit <= 0:
            return "miss"
        if n_hit < need:
            return "partial"
        return "full"

    def _pick_blocker(self, z: float, t: float):
        """Return (blocker_index, z) if click is near a blocker handle/slab."""
        best = None
        best_d = 1e9
        for h in getattr(self, "_blocker_handles", []) or []:
            dz = abs(float(h["z"]) - z)
            # Accept if near Z and within transverse extent (+ margin)
            t_lo = float(h["t_lo"]) - 0.5
            t_hi = float(h["t_hi"]) + 0.5
            if t < t_lo or t > t_hi:
                continue
            if dz < best_d and dz < 2.5:
                best_d = dz
                best = (int(h["index"]), float(h["z"]))
        return best

    def _pick_fov_side(self, z: float, t: float, *, use_xz: bool):
        """Return ('resize', handle, which) or ('move', handle, None) or None."""
        want = "xz" if use_xz else "yz"
        best = None
        best_d = 1e9
        for h in getattr(self, "_fov_side_handles", []) or []:
            if h.get("plane") != want:
                continue
            hz = float(h["z"])
            if abs(float(z) - hz) > 4.0:
                continue
            t_lo, t_hi, t_c = float(h["t_lo"]), float(h["t_hi"]), float(h["t_c"])
            d_lo = math.hypot(float(z) - hz, float(t) - t_lo)
            d_hi = math.hypot(float(z) - hz, float(t) - t_hi)
            hit_r = max(2.2, 0.08 * max(abs(t_hi - t_lo), 8.0))
            if d_lo < best_d and d_lo <= hit_r:
                best_d = d_lo
                best = ("resize", h, "lo")
            if d_hi < best_d and d_hi <= hit_r:
                best_d = d_hi
                best = ("resize", h, "hi")
            if best is None and t_lo <= float(t) <= t_hi and abs(float(z) - hz) < 2.8:
                best = ("move", h, None)
        return best

    def _draw_profiles(self):
        """Always-visible X / Y / diagonal irradiance cuts with FOV markers."""
        if not hasattr(self, "fig_prof"):
            return
        self.fig_prof.clf()
        ax = self.fig_prof.add_subplot(111)
        self.ax_prof = ax
        self._style_axes()
        res = self.result
        if res is None:
            self.canvas_prof.draw_idle()
            return
        p = self.collect_params()
        grid = np.asarray(res.map.as_grid(), dtype=float)
        ny, nx = grid.shape
        hw, hh = float(res.map.half_w), float(res.map.half_h)
        xs = np.linspace(-hw, hw, nx)
        ys = np.linspace(hh, -hh, ny)
        ix0 = int(np.argmin(np.abs(xs)))
        iy0 = int(np.argmin(np.abs(ys)))
        prof_x = grid[iy0, :]
        prof_y = grid[:, ix0]
        n_diag = min(nx, ny)
        t = np.linspace(-min(hw, hh), min(hw, hh), n_diag)

        def sample(xx, yy):
            fx = (xx + hw) / max(2 * hw, 1e-9) * (nx - 1)
            fy = (hh - yy) / max(2 * hh, 1e-9) * (ny - 1)
            i0 = int(np.clip(np.floor(fx), 0, nx - 2))
            j0 = int(np.clip(np.floor(fy), 0, ny - 2))
            tx, ty = fx - i0, fy - j0
            return (
                (1 - tx) * (1 - ty) * grid[j0, i0]
                + tx * (1 - ty) * grid[j0, i0 + 1]
                + (1 - tx) * ty * grid[j0 + 1, i0]
                + tx * ty * grid[j0 + 1, i0 + 1]
            )

        prof_d = np.array([sample(tt, tt) for tt in t])

        def norm(a):
            a = np.asarray(a, dtype=float)
            m = float(np.max(a)) if a.size else 0.0
            return a / m if m > 0 else a

        def smooth_profile(a: np.ndarray) -> np.ndarray:
            """
            Light digital filter for Monte-Carlo bin noise.
            Prefer Savitzky–Golay (preserves peaks) with a short window
            (~6–8% of samples, odd, ≥5). Falls back to a 5-tap moving mean.
            Intentionally mild — not a heavy low-pass.
            """
            y = np.asarray(a, dtype=float)
            n = int(y.size)
            if n < 7:
                return y
            # Window ~7% of length, odd, clamped
            w = max(5, int(round(n * 0.07)) | 1)
            if w >= n:
                w = n - 1 if (n % 2 == 0) else n
            if w < 5:
                return y
            try:
                from scipy.signal import savgol_filter

                poly = 2 if w >= 7 else 1
                return savgol_filter(y, window_length=w, polyorder=poly, mode="nearest")
            except Exception:
                k = np.ones(w, dtype=float) / float(w)
                # Reflect pad then convolve to avoid edge bias
                pad = w // 2
                yp = np.pad(y, pad, mode="edge")
                return np.convolve(yp, k, mode="valid")

        # Faint raw + solid smoothed: noise reduced without hiding structure
        nx_s = smooth_profile(prof_x)
        ny_s = smooth_profile(prof_y)
        nd_s = smooth_profile(prof_d)

        fov_w = float(p["fov_width"])
        fov_h = float(p["fov_height"])
        fov_cx = float(p.get("fov_cx", 0.0))
        fov_cy = float(p.get("fov_cy", 0.0))
        ax.plot(xs, norm(prof_x), color="#38bdf8", lw=0.7, alpha=0.25)
        ax.plot(ys, norm(prof_y), color="#a78bfa", lw=0.7, alpha=0.25)
        ax.plot(t, norm(prof_d), color="#fbbf24", lw=0.7, alpha=0.25)
        ax.plot(xs, norm(nx_s), color="#38bdf8", lw=1.9, label="X-axis cut (Y = 0)")
        ax.plot(ys, norm(ny_s), color="#a78bfa", lw=1.9, label="Y-axis cut (X = 0)")
        ax.plot(t, norm(nd_s), color="#fbbf24", lw=1.9, label="XY diagonal")
        ax.axvline(fov_cx - fov_w / 2, color="#c084fc", ls="--", lw=1.1, alpha=0.9, label="FOV X edge")
        ax.axvline(fov_cx + fov_w / 2, color="#c084fc", ls="--", lw=1.1, alpha=0.9)
        ax.axvline(fov_cy - fov_h / 2, color="#e879f9", ls="--", lw=1.1, alpha=0.9, label="FOV Y edge")
        ax.axvline(fov_cy + fov_h / 2, color="#e879f9", ls="--", lw=1.1, alpha=0.9)
        d_lim = min(fov_w / 2, fov_h / 2)
        ax.axvline(-d_lim, color="#94a3b8", ls=":", lw=1.0, alpha=0.7, label="FOV on diagonal")
        ax.axvline(d_lim, color="#94a3b8", ls=":", lw=1.0, alpha=0.7)
        ax.set_xlabel("Position along cut (mm)", color="#f8fafc")
        ax.set_ylabel("Normalized irradiance", color="#f8fafc")
        ax.set_title(
            "PROFILES  ·  X / Y / diagonal  ·  light Savitzky–Golay",
            loc="left",
            fontsize=10,
            color=FG_BRIGHT,
        )
        ax.tick_params(colors="#f8fafc", labelsize=8)
        leg = ax.legend(
            fontsize=7,
            loc="upper right",
            facecolor=BG2,
            edgecolor=BORDER,
            labelcolor="#f8fafc",
        )
        ax.grid(True, alpha=0.25, color=BORDER)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        self.fig_prof.tight_layout()
        self.canvas_prof.draw_idle()

    def _draw_target(self):
        self._fov_rect_patch = None
        self._fov_tgt_handle_arts = []
        self.fig_tgt.clf()
        ax = self.fig_tgt.add_subplot(111)
        self.ax_tgt = ax
        self._style_axes()
        res = self.result
        if res is None:
            return
        p = self.collect_params()
        grid = res.map.as_grid()
        if self.log_scale.get() and grid.max() > 0:
            data = np.log1p(grid * 50 / grid.max())
            data = data / (data.max() + 1e-30)
        else:
            data = grid / (grid.max() + 1e-30) if grid.max() > 0 else grid

        hw, hh = res.map.half_w, res.map.half_h
        extent = [-hw, hw, -hh, hh]
        self._tgt_full_extent = (-hw, hw, -hh, hh)
        self._tgt_zoom_extent = target_zoom_extent(hw, hh)
        im = ax.imshow(
            data,
            extent=extent,
            origin="upper",
            cmap=IRRAD_CMAP,
            aspect="equal",
            interpolation="bilinear",
        )
        ax.set_aspect("equal", adjustable="box")
        from matplotlib.patches import Rectangle

        fov_w, fov_h = p["fov_width"], p["fov_height"]
        fov_cx, fov_cy = p["fov_cx"], p["fov_cy"]
        rect = Rectangle(
            (fov_cx - fov_w / 2, fov_cy - fov_h / 2),
            fov_w,
            fov_h,
            fill=False,
            edgecolor=FOV,
            lw=2,
            label="Camera FOV",
        )
        ax.add_patch(rect)
        self._fov_rect_patch = rect
        self._paint_fov_tgt_handles(ax, fov_cx, fov_cy, fov_w, fov_h)
        cx, cy = res.stats["centroid"]
        ax.plot(cx, cy, "+", color="white", ms=10, mew=1.5)
        ax.axhline(0, color="#64748b", ls=":", lw=0.7, alpha=0.5)
        ax.axvline(0, color="#64748b", ls=":", lw=0.7, alpha=0.5)

        # Every display-path hit on the plane — keep individual rays visible
        # when zoomed in; they still read as a spray when zoomed out.
        tgt_z = float(p.get("target_z", 80.0))
        hits_xy = target_plane_hits(res.paths, tgt_z)
        if hits_xy:
            ax.scatter(
                [h[0] for h in hits_xy],
                [h[1] for h in hits_xy],
                s=8,
                c="#e2e8f0",
                alpha=0.40,
                zorder=5,
                linewidths=0,
            )

        # Optional overlay: color ray endpoints by how many elements they hit
        if getattr(self, "v_color_partial", None) is not None and bool(self.v_color_partial.get()):
            stack_need, only_mla = self._stack_element_counts(res)
            xs_m, ys_m = [], []
            xs_p, ys_p = [], []
            xs_f, ys_f = [], []
            target_z = float(p.get("target_z", 80.0))
            for path in res.paths:
                if len(path.history) < 1:
                    continue
                pt = path.history[-1]
                if abs(pt[2] - target_z) > 1.5 and path.terminated != "target":
                    continue
                cls = self._classify_path(path, stack_need, only_mla)
                if cls == "miss":
                    xs_m.append(pt[0])
                    ys_m.append(pt[1])
                elif cls == "partial":
                    xs_p.append(pt[0])
                    ys_p.append(pt[1])
                else:
                    xs_f.append(pt[0])
                    ys_f.append(pt[1])
            # Overlay colors chosen to avoid the irradiance colorbar
            # (dark → purple → cyan → yellow → white). Use high-chroma markers
            # with dark edges so they stay visible on both dark and bright bins.
            if xs_m:
                ax.scatter(
                    xs_m,
                    ys_m,
                    s=18,
                    c="#ff1493",  # deep pink — not in colorbar
                    alpha=0.9,
                    zorder=6,
                    edgecolors="#1a1a1a",
                    linewidths=0.45,
                    label="Missed lenses",
                )
            if xs_p:
                ax.scatter(
                    xs_p,
                    ys_p,
                    s=20,
                    c="#ff0000",  # pure red
                    alpha=0.9,
                    zorder=7,
                    edgecolors="#1a1a1a",
                    linewidths=0.45,
                    label="Partial stack",
                )
            if xs_f:
                ax.scatter(
                    xs_f,
                    ys_f,
                    s=18,
                    c="#c6ff00",  # electric lime — outside purple/cyan/yellow ramp
                    alpha=0.9,
                    zorder=8,
                    edgecolors="#1a1a1a",
                    linewidths=0.45,
                    label="Full stack",
                )
            if xs_m or xs_p or xs_f:
                ax.legend(
                    fontsize=7,
                    loc="upper right",
                    facecolor=BG2,
                    edgecolor=BORDER,
                    labelcolor="#f8fafc",
                )

        ax.set_title(
            "TARGET PLANE  ·  FOV handles  ·  scroll zoom  ·  right-drag pan",
            loc="left",
            fontsize=10,
            color=FG_BRIGHT,
        )
        ax.set_xlabel("X (mm)", color="#f8fafc")
        ax.set_ylabel("Y (mm)", color="#f8fafc")
        ax.tick_params(colors="#f8fafc", labelsize=8)
        cbar = self.fig_tgt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="#f8fafc", labelsize=7)
        cbar.outline.set_edgecolor(BORDER)
        for spine in cbar.ax.spines.values():
            spine.set_color(BORDER)
        self._apply_tgt_limits(ax)
        self.fig_tgt.tight_layout()
        self.canvas_tgt.draw_idle()

    # ── Presets / reset ──────────────────────────────────────────────────

    def reset_defaults(self):
        self.params = default_params()
        self._apply_params_to_vars(self.params)
        self.run_trace()

    def cancel_optimize(self):
        """Signal the background optimizer to stop after the current evaluation."""
        if self._optimizing:
            self._opt_gen += 1
            self.opt_status.set("Cancel requested — finishing current evaluation…")
            self.status_var.set("Optimizer cancel requested…")

    def _optimizer_objective_key(self) -> str:
        label = str(self.v_opt_goal.get()) if hasattr(self, "v_opt_goal") else ""
        if label.startswith("Sharpest"):
            return "focus"
        if label.startswith("Maximum"):
            return "evenness"
        return "rect_fill"

    def run_optimize_current(self):
        """Tune the current stack only (no extra lenses). Kept for API compat."""
        if not hasattr(self, "v_opt_rays"):
            return
        cfg = config_from_panel(
            objective="evenness",
            allow_extra_lenses=False,
            extra_lenses=0,
            rays_per_eval=int(self.v_opt_rays.get()),
            max_evals=int(self.v_opt_evals.get()),
            uniformity_weight=float(self.v_opt_uni_w.get()),
            fill_weight=float(self.v_opt_fill_w.get()),
            polish=bool(self.v_opt_polish.get()),
            optimize_asphere=bool(self.v_opt_asphere.get()),
        )
        self._start_optimize(
            cfg,
            status_msg="Optimizing current lens group for evenness…",
            panel_msg="Current stack only (no extra lenses)…",
        )

    def run_optimize_rectangular(self):
        """Left-panel Optimize button — runs the selected goal."""
        if not hasattr(self, "v_opt_extra"):
            return
        objective = self._optimizer_objective_key()
        allow = bool(self.v_opt_allow_extra.get()) if hasattr(self, "v_opt_allow_extra") else False
        extra = max(0, min(8, int(self.v_opt_extra.get())))
        cfg = config_from_panel(
            objective=objective,
            allow_extra_lenses=allow,
            extra_lenses=extra,
            anamorphic_mode=str(self.v_opt_ana_mode.get() or "crossed"),
            rays_per_eval=int(self.v_opt_rays.get()),
            max_evals=int(self.v_opt_evals.get()),
            uniformity_weight=float(self.v_opt_uni_w.get()),
            aspect_weight=float(self.v_opt_aspect_w.get()),
            fill_weight=float(self.v_opt_fill_w.get()),
            polish=bool(self.v_opt_polish.get()),
            optimize_asphere=bool(self.v_opt_asphere.get()),
        )
        if hasattr(self, "v_opt_two_phase"):
            self.v_opt_two_phase.set(bool(cfg.two_phase))
        if cfg.two_phase and cfg.extra_anamorphic_lenses > 0:
            panel = (
                f"Rectangular fill · may add up to {cfg.extra_anamorphic_lenses} "
                f"lenses ({cfg.anamorphic_mode})…"
            )
            status = "Rectangular FOV optimize…"
        elif objective == "focus":
            panel = "Sharpest focus: maximizing illumination-border crispness…"
            status = "Optimizing focus…"
        elif objective == "evenness":
            panel = "Maximum evenness: flattening FOV irradiance edge-to-edge…"
            status = "Optimizing evenness…"
        else:
            panel = "Rectangular fill on the current stack (no extra lenses)…"
            status = "Optimizing rectangular FOV…"
        self._start_optimize(cfg, status_msg=status, panel_msg=panel)

    def _start_optimize(self, cfg: OptimizeConfig, *, status_msg: str, panel_msg: str):
        """Shared background optimizer launcher."""
        if self._optimizing:
            messagebox.showinfo(
                "Optimizer",
                "An optimization is already running. Press Cancel first, or wait.",
            )
            return
        try:
            from scipy.optimize import differential_evolution  # noqa: F401
        except ImportError:
            messagebox.showerror(
                "Missing dependency",
                "The optimizer needs SciPy.\n\nInstall with:\n  pip install scipy",
            )
            return

        self._opt_gen += 1
        gen = self._opt_gen
        self._optimizing = True
        was_auto = bool(self.auto_run.get())
        self.auto_run.set(False)

        params = self.collect_params()
        if hasattr(self, "opt_status"):
            self.opt_status.set(panel_msg)
        self.status_var.set(status_msg)
        self.progress["value"] = 0

        def progress_cb(frac: float, msg: str, best: float):
            if gen != self._opt_gen:
                return
            self.after(
                0,
                lambda: (
                    self.progress.configure(value=frac * 100),
                    self.opt_status.set(msg) if hasattr(self, "opt_status") else None,
                    self.status_var.set(msg),
                ),
            )

        def should_cancel():
            return gen != self._opt_gen

        cfg.progress_cb = progress_cb
        cfg.should_cancel = should_cancel

        def work():
            err = None
            result = None
            try:
                result = optimize_fov_flux(params, cfg)
            except Exception as e:
                err = e
            self.after(0, lambda: self._on_optimize_done(gen, result, err, was_auto))

        threading.Thread(target=work, daemon=True).start()

    # Back-compat alias
    def run_optimize(self):
        self.run_optimize_rectangular()

    def _on_optimize_done(self, gen: int, result, err, was_auto: bool):
        self._optimizing = False
        self.progress["value"] = 100
        if gen != self._opt_gen and result is None and err is None:
            self.auto_run.set(was_auto)
            self.opt_status.set("Optimizer cancelled.")
            self.status_var.set("Ready")
            return
        if err is not None:
            self.auto_run.set(was_auto)
            self.opt_status.set(f"Error: {err}")
            self.status_var.set(f"Optimizer error: {err}")
            messagebox.showerror("Optimizer error", str(err))
            return
        if result is None:
            self.auto_run.set(was_auto)
            self.opt_status.set("Optimizer returned no result.")
            return

        self._apply_params_to_vars(result.params)
        self.params = result.params
        msg = (
            f"FOV flux {result.fov_flux * 100:.1f}% · "
            f"uniform {result.uniformity * 100:.1f}% · "
            f"aspect err {result.aspect_error * 100:.1f}% · "
            f"{result.n_evals} evals in {result.elapsed_s:.0f}s"
        )
        self.opt_status.set(msg)
        self.status_var.set(result.message or msg)
        self.auto_run.set(was_auto)
        self.run_trace()

    def _apply_params_to_vars(self, p: Dict[str, Any]):
        s = p["source"]
        self.v_mode.set(s["mode"])
        self.v_rows.set(s["rows"])
        self.v_cols.set(s["cols"])
        self.v_pitch_x.set(s["pitch_x"])
        self.v_pitch_y.set(s["pitch_y"])
        self.v_die_w.set(s["die_width"])
        self.v_die_h.set(s["die_height"])
        self.v_source_z.set(s["source_z"])
        self.v_flux.set(s["flux_per_die"])
        self.v_wl.set(s["wavelength_nm"])
        self.v_half.set(s["half_angle_deg"])
        self.v_tilt_x.set(s["tilt_x"])
        self.v_tilt_y.set(s["tilt_y"])
        self.v_off_x.set(s["offset_x"])
        self.v_off_y.set(s["offset_y"])
        self.v_rot_z.set(s["die_rot_z"])
        self.v_stagger.set(s["stagger"])
        self.v_circ.set(s["circular_mask"])
        self.v_mask_r.set(s["mask_radius"])
        self.v_lens_z.set(p["lens_z_start"])
        self.v_custom_n.set(p["custom_n"])
        self.v_fresnel.set(p["apply_fresnel"])
        self.v_tir_abs.set(p.get("absorb_on_tir", True))
        if hasattr(self, "v_kill_back"):
            self.v_kill_back.set(p.get("kill_backward", True))
        elems = pad_elements(p.get("elements") or [], MAX_ELEMENTS)
        p["elements"] = elems
        for i, e in enumerate(elems):
            if i >= len(self.elem_vars):
                break
            self._apply_element_dict_to_vars(i, e)
            # Collapse disabled slots so the side panel stays compact
            if hasattr(self, "elem_ui") and i < len(self.elem_ui):
                self._set_element_collapsed(i, collapsed=not bool(e.get("enabled", False)))
        self.v_target_z.set(p["target_z"])
        self._fov_syncing = True
        try:
            self.v_fov_w.set(p["fov_width"])
            self.v_fov_h.set(p["fov_height"])
            self.v_fov_aspect.set(round(fov_aspect(p["fov_width"], p["fov_height"]), 4))
        finally:
            self._fov_syncing = False
        self.v_fov_cx.set(p["fov_cx"])
        self.v_fov_cy.set(p["fov_cy"])
        if "fov_aspect_lock" in p:
            self.v_fov_lock.set(p["fov_aspect_lock"])
        self.v_map_w.set(p["map_half_w"])
        self.v_map_h.set(p["map_half_h"])
        self._grow_map_sliders_to_fov()
        self.v_map_res.set(p["map_res"])
        self.v_rays.set(p["total_rays"])
        self.v_disp.set(p["display_rays"])
        mla = p.get("mla") or {}
        self.v_mla.set(bool(mla.get("enabled", False)))
        self.v_mla_fill.set(float(mla.get("fill_factor", 0.88)))
        self.v_mla_ap.set(float(mla.get("lenslet_aperture", 0.0)))
        self.v_export_plate.set(bool(mla.get("export_plate", True)))
        if hasattr(self, "v_mla_scale"):
            self.v_mla_scale.set(bool(mla.get("scale_to_pitch", True)))
        if hasattr(self, "v_mla_aim"):
            self.v_mla_aim.set(bool(mla.get("aim_to_fov", True)))
        if hasattr(self, "v_mla_aim_s"):
            self.v_mla_aim_s.set(float(mla.get("aim_strength", 1.0)))
        if hasattr(self, "v_cad_edge") and "cad_max_edge_mm" in p:
            self.v_cad_edge.set(float(p["cad_max_edge_mm"]))
        if hasattr(self, "v_cad_angle") and "cad_max_angle_deg" in p:
            self.v_cad_angle.set(float(p["cad_max_angle_deg"]))
        if hasattr(self, "v_cad_flange_r") and "cad_flange_radial_mm" in p:
            self.v_cad_flange_r.set(float(p["cad_flange_radial_mm"]))
        if hasattr(self, "v_cad_flange_t") and "cad_flange_thickness_mm" in p:
            self.v_cad_flange_t.set(float(p["cad_flange_thickness_mm"]))
        if hasattr(self, "v_cad_halves"):
            self.v_cad_halves.set(bool(p.get("cad_export_halves", False)))
        if hasattr(self, "v_pol_dia"):
            self.v_pol_dia.set(float(p.get("cad_polish_tool_dia_mm", 10.0)))
            self.v_pol_step.set(float(p.get("cad_polish_stepover_mm", 0.3)))
            self.v_pol_allow.set(float(p.get("cad_polish_allowance_mm", 0.0)))
            self.v_pol_feed.set(float(p.get("cad_polish_feed_mm_min", 100.0)))
            self.v_pol_retr.set(float(p.get("cad_polish_retract_mm", 5.0)))
            self.v_pol_rpm.set(float(p.get("cad_polish_rpm", 5000.0)))
            self.v_pol_revs.set(float(p.get("cad_polish_revs", 2.0)))
            self.v_pol_surf.set(str(p.get("cad_polish_surface", "front")))
            self.v_pol_strat.set(str(p.get("cad_polish_strategy", "helical")))
        if hasattr(self, "v_lap_surf"):
            self.v_lap_surf.set(str(p.get("cad_lap_surface", p.get("cad_polish_surface", "front"))))
            self.v_lap_wall.set(float(p.get("cad_lap_wall_mm", 6.0)))
        if hasattr(self, "cad_mesh_status"):
            self._refresh_cad_mesh_label()
        # Absorbing panels
        blks = p.get("blockers")
        if not isinstance(blks, list):
            blks = []
        self.blockers = [dict(b) for b in blks if isinstance(b, dict)]
        self._refresh_blocker_listbox()
        if self.blockers and hasattr(self, "blk_list"):
            self.blk_list.selection_set(0)
            self._load_blocker_to_vars(0)

    def _on_element_shape_selected(self, elem_index: int):
        """User picked a library type for one element — apply immediately."""
        self._apply_shape_to_element(elem_index)

    def _on_element_r_mag(self, elem_index: int):
        """|R| changed: if a library type is selected, rebuild R₁/R₂ from it."""
        ev = self.elem_vars[elem_index]
        if shape_id_from_label(ev["shape"].get()) == "custom":
            self._on_param_change()
            return
        self._apply_shape_to_element(elem_index)

    def _set_fov_span(self, size: float, *, use_xz: bool) -> None:
        """Set FOV width (X–Z) or height (Y–Z), honouring aspect lock."""
        size = clamp_fov_size(size)
        lock = bool(self.v_fov_lock.get())
        self._fov_syncing = True
        try:
            if use_xz:
                old_w = max(float(self.v_fov_w.get()), 1e-9)
                self.v_fov_w.set(round(size, 3))
                if lock:
                    self.v_fov_h.set(
                        round(clamp_fov_size(float(self.v_fov_h.get()) * (size / old_w)), 3)
                    )
            else:
                old_h = max(float(self.v_fov_h.get()), 1e-9)
                self.v_fov_h.set(round(size, 3))
                if lock:
                    self.v_fov_w.set(
                        round(clamp_fov_size(float(self.v_fov_w.get()) * (size / old_h)), 3)
                    )
            w = float(self.v_fov_w.get())
            h = float(self.v_fov_h.get())
            if h > 1e-9:
                self.v_fov_aspect.set(round(w / h, 4))
        finally:
            self._fov_syncing = False

    def _set_fov_center_axis(self, value: float, *, use_xz: bool) -> None:
        value = max(-200.0, min(200.0, float(value)))
        if use_xz:
            self.v_fov_cx.set(round(value, 3))
        else:
            self.v_fov_cy.set(round(value, 3))

    def _set_fov_rect(self, cx: float, cy: float, w: float, h: float) -> None:
        self._fov_syncing = True
        try:
            self.v_fov_cx.set(round(float(cx), 3))
            self.v_fov_cy.set(round(float(cy), 3))
            self.v_fov_w.set(round(clamp_fov_size(w), 3))
            self.v_fov_h.set(round(clamp_fov_size(h), 3))
            hh = max(float(self.v_fov_h.get()), 1e-9)
            self.v_fov_aspect.set(round(float(self.v_fov_w.get()) / hh, 4))
        finally:
            self._fov_syncing = False

    def _refresh_fov_overlay(self):
        """Move the target-plane FOV rectangle without rebuilding the heatmap."""
        ax = getattr(self, "ax_tgt", None)
        patch = getattr(self, "_fov_rect_patch", None)
        if ax is None or patch is None:
            return
        try:
            w = float(self.v_fov_w.get())
            h = float(self.v_fov_h.get())
            cx = float(self.v_fov_cx.get())
            cy = float(self.v_fov_cy.get())
        except (tk.TclError, ValueError):
            return
        patch.set_xy((cx - w / 2.0, cy - h / 2.0))
        patch.set_width(w)
        patch.set_height(h)
        self._paint_fov_tgt_handles(ax, cx, cy, w, h)
        try:
            self.canvas_tgt.draw_idle()
        except Exception:
            pass

    def _paint_fov_tgt_handles(self, ax, cx, cy, w, h):
        """Corner / edge grips on the target-plane FOV rectangle."""
        for art in getattr(self, "_fov_tgt_handle_arts", []) or []:
            try:
                art.remove()
            except Exception:
                pass
        self._fov_tgt_handle_arts = []
        x0, x1 = cx - w / 2.0, cx + w / 2.0
        y0, y1 = cy - h / 2.0, cy + h / 2.0
        pts = {
            "nw": (x0, y1),
            "ne": (x1, y1),
            "sw": (x0, y0),
            "se": (x1, y0),
            "left": (x0, cy),
            "right": (x1, cy),
            "top": (cx, y1),
            "bottom": (cx, y0),
        }
        xs, ys = zip(*pts.values()) if pts else ([], [])
        (ln,) = ax.plot(
            list(xs),
            list(ys),
            "D",
            color="#c4b5fd",
            ms=6,
            zorder=10,
            markeredgecolor="#0b0f14",
            markeredgewidth=0.5,
        )
        self._fov_tgt_handle_arts = [ln]
        self._fov_tgt_handles = [
            {"kind": "corner" if k in ("nw", "ne", "sw", "se") else "edge", "name": k, "x": px, "y": py}
            for k, (px, py) in pts.items()
        ]

    def _on_fov_dim_change(self, *_):
        if getattr(self, "_fov_syncing", False):
            return
        try:
            w = float(self.v_fov_w.get())
            h = float(self.v_fov_h.get())
        except (tk.TclError, ValueError):
            return
        if h <= 0:
            return
        self._fov_syncing = True
        try:
            self.v_fov_aspect.set(round(w / h, 4))
        finally:
            self._fov_syncing = False
        self._grow_map_sliders_to_fov()

    def _grow_map_sliders_to_fov(self) -> None:
        """Keep the recorded map at least as large as the FOV rectangle."""
        if not hasattr(self, "v_map_w"):
            return
        try:
            w = float(self.v_fov_w.get())
            h = float(self.v_fov_h.get())
            cx = float(self.v_fov_cx.get())
            cy = float(self.v_fov_cy.get())
            hw = float(self.v_map_w.get())
            hh = float(self.v_map_h.get())
        except (tk.TclError, ValueError):
            return
        nw, nh = map_half_covering_fov(w, h, cx, cy, hw, hh)
        if nw > hw + 0.05:
            self.v_map_w.set(round(nw, 1))
        if nh > hh + 0.05:
            self.v_map_h.set(round(nh, 1))

    def _focus_group_on_target(self):
        """Slide the lens group along Z until the source images onto the FOV plane."""
        if getattr(self, "_optimizing", False) or getattr(self, "_focusing", False):
            messagebox.showinfo("Focus", "A search is already running.")
            return
        params = self.collect_params()
        if not any(e.get("enabled") for e in params.get("elements") or []):
            messagebox.showwarning("Focus", "Enable at least one lens element.")
            return
        self._focusing = True
        self._opt_gen += 1
        gen = self._opt_gen
        was_auto = bool(self.auto_run.get())
        self.auto_run.set(False)
        self.status_var.set("Focusing group — imaging the source onto the target…")
        self.progress["value"] = 0

        def work():
            err = None
            result = None
            try:
                result = focus_group_on_target(
                    params,
                    n_scan=21,
                    n_refine=11,
                    rays_per_point=36,
                    should_cancel=lambda: gen != self._opt_gen,
                    progress_cb=lambda f, m: self.after(
                        0,
                        lambda: (
                            self.progress.configure(value=f * 100),
                            self.status_var.set(m),
                        ),
                    ),
                )
            except Exception as e:
                err = e
            self.after(0, lambda: self._on_focus_group_done(gen, result, err, was_auto))

        threading.Thread(target=work, daemon=True).start()

    def _on_focus_group_done(self, gen: int, result, err, was_auto: bool):
        self._focusing = False
        self.progress["value"] = 100
        self.auto_run.set(was_auto)
        if gen != self._opt_gen:
            self.status_var.set("Focus search cancelled.")
            return
        if err is not None:
            self.status_var.set(f"Focus error: {err}")
            messagebox.showerror("Focus failed", str(err))
            return
        result = result or {}
        if not result.get("ok"):
            msg = str(result.get("message") or "Focus search failed.")
            self.status_var.set(msg)
            messagebox.showwarning("Focus", msg)
            return
        z = float(result["lens_z_start"])
        self.v_lens_z.set(round(z, 3))
        msg = (
            f"{result.get('message', 'Focused.')}  "
            f"Group front Z = {z:.2f} mm."
        )
        self.status_var.set(msg)
        if hasattr(self, "opt_status"):
            self.opt_status.set(
                f"Source image focus · Z={z:.2f} mm · blur {float(result.get('blur_mm', 0)):.2f} mm"
            )
        self._request_trace_after_edit(msg + " — tracing…")

    def _on_fov_aspect_change(self, *_):
        if getattr(self, "_fov_syncing", False) or not self.v_fov_lock.get():
            return
        try:
            a = float(self.v_fov_aspect.get())
            h = float(self.v_fov_h.get())
        except (tk.TclError, ValueError):
            return
        if a <= 0 or h <= 0:
            return
        w, _h2 = set_fov_from_aspect(a, h)
        self._fov_syncing = True
        try:
            self.v_fov_w.set(round(w, 3))
        finally:
            self._fov_syncing = False
        self._grow_map_sliders_to_fov()
        self._on_param_change()

    def _swap_anamorphic_xy(self):
        """Swap cylinder/biconic X↔Y powers — fixes a 90° rotated footprint."""
        p = self.collect_params()
        p2 = swap_anamorphic_xy_params(p)
        n = 0
        for i, el in enumerate(p2.get("elements", [])):
            if i >= len(self.elem_vars):
                break
            if not el.get("enabled", True):
                continue
            mode = str(el.get("surface_mode", "")).lower()
            if mode in ("cylinder_x", "cylinder_y", "biconic") or el.get("R1y") is not None:
                self._apply_element_dict_to_vars(i, el)
                n += 1
        if n == 0:
            self.design_status.set("No anamorphic elements to swap (need cylinder/biconic).")
            messagebox.showinfo(
                "Swap X/Y",
                "No cylindrical or biconic elements are enabled.\n"
                "Use “Crossed cylinders” or “Biconic singlet” first, or set "
                "an element surface mode to cylinder/biconic.",
            )
            return
        self.design_status.set(f"Swapped X↔Y on {n} anamorphic element(s). Re-trace to verify.")
        self.status_var.set(self.design_status.get())
        self._on_param_change()

    def _rotate_optics_90_vs_fov(self):
        """
        Rotate the rectangular FOV 90° and swap anamorphic X/Y so the lens
        powers track the new orientation.
        """
        try:
            w = float(self.v_fov_w.get())
            h = float(self.v_fov_h.get())
        except (tk.TclError, ValueError):
            return
        self._fov_syncing = True
        try:
            self.v_fov_w.set(h)
            self.v_fov_h.set(w)
            if h > 1e-9:
                self.v_fov_aspect.set(round(h / w, 4) if w > 1e-9 else 1.0)
            else:
                self.v_fov_aspect.set(1.0)
        finally:
            self._fov_syncing = False
        # Swap lens axes to match the rotated FOV
        p = self.collect_params()
        p2 = swap_anamorphic_xy_params(p)
        n = 0
        for i, el in enumerate(p2.get("elements", [])):
            if i >= len(self.elem_vars):
                break
            mode = str(el.get("surface_mode", "")).lower()
            if mode in ("cylinder_x", "cylinder_y", "biconic") or el.get("R1y") is not None:
                self._apply_element_dict_to_vars(i, el)
                n += 1
        self.design_status.set(
            f"FOV rotated 90° (now {h:.1f}×{w:.1f} mm)"
            + (f"; swapped X↔Y on {n} element(s)" if n else "")
            + ". Re-trace."
        )
        self.status_var.set(self.design_status.get())
        self._on_param_change()

    def _on_partial_ray_color_toggle(self):
        if self.result is not None:
            self._draw_target()
            self._draw_side()

    def _show_help(self):
        """Load HELP.txt from the program folder (not hardcoded)."""
        help_path = Path(__file__).resolve().parent / "HELP.txt"
        win = tk.Toplevel(self)
        win.title("OptiFlux — Help")
        win.geometry("720x560")
        win.configure(bg=BG)
        frm = ttk.Frame(win)
        frm.pack(fill="both", expand=True, padx=8, pady=8)
        txt = tk.Text(
            frm,
            wrap="word",
            bg=BG2,
            fg=FG,
            insertbackground=FG,
            relief="flat",
            font=("Segoe UI", 10),
            padx=8,
            pady=8,
        )
        sb = ttk.Scrollbar(frm, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        txt.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        try:
            body = help_path.read_text(encoding="utf-8")
        except Exception as e:
            body = (
                f"Could not load help file:\n  {help_path}\n\n{e}\n\n"
                "Place HELP.txt next to app.py."
            )
        txt.insert("1.0", body)
        txt.configure(state="disabled")
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=6)

    def _commercial_lens_report(self, params: dict) -> str:
        """Human-readable buy-sheet for enabled elements (Edmund-style fields)."""
        from materials_catalog import refractive_index, material_id_from_name, material_name_from_id
        from mla_geometry import thin_lens_focal_length_mm

        lines = []
        lines.append("COMMERCIAL LENS PARAMETER SHEET")
        lines.append("=" * 56)
        lines.append("Units: millimetres (mm). Quote these to catalog suppliers")
        lines.append("(Edmund Optics, Thorlabs, Newport, etc.) or for custom optic RFQs.")
        lines.append("")
        src = params.get("source") or {}
        wl = float(src.get("wavelength_nm", 550.0))
        lines.append(f"Design wavelength: {wl:.1f} nm")
        lines.append(f"Target plane Z:    {float(params.get('target_z', 80)):.2f} mm")
        lines.append(
            f"FOV (W×H):         {float(params.get('fov_width', 40)):.2f} × "
            f"{float(params.get('fov_height', 32)):.2f} mm"
        )
        lines.append(f"Lens group Z start:{float(params.get('lens_z_start', 3)):.2f} mm")
        lines.append("")
        z = float(params.get("lens_z_start", 3.0))
        any_on = False
        for i, e in enumerate(params.get("elements") or []):
            if not e.get("enabled", True):
                continue
            any_on = True
            mid = material_id_from_name(str(e.get("material", "N_BK7")))
            mat_name = material_name_from_id(mid)
            n = refractive_index(mid, wl, float(params.get("custom_n", 1.5)))
            R1 = float(e.get("R1", 0.0) or 0.0)
            R2 = float(e.get("R2", 0.0) or 0.0)
            R1y = e.get("R1y")
            R2y = e.get("R2y")
            t = float(e.get("thickness", 3.0))
            ap = float(e.get("aperture", 10.0))
            apy = e.get("aperture_y")
            mode = str(e.get("surface_mode", "rotational"))
            efl = thin_lens_focal_length_mm(R1, R2, n, t)
            # Edge thickness estimate at circular aperture (rotational sag)
            from engine import OpticalSurface

            s1 = OpticalSurface(z_vertex=0.0, radius=R1, aperture=ap, material_after=mid)
            s2 = OpticalSurface(z_vertex=t, radius=R2, aperture=ap, material_after="AIR", material_before=mid)
            try:
                from engine import lens_edge_thickness

                et_raw = lens_edge_thickness(s1, s2, ap * 0.98)
                et = float(et_raw) if et_raw is not None else t
            except Exception:
                et = t
            diam = 2.0 * ap
            lines.append(f"--- Element {i + 1}  [{mode}] ---")
            lines.append(f"  Material / catalog glass : {mat_name}  (n≈{n:.4f} @ {wl:.0f} nm)")
            lines.append(f"  Clear diameter (CA)      : {diam:.3f} mm  (semi-aperture {ap:.3f})")
            if apy is not None:
                lines.append(f"  Clear aperture Y         : {2.0 * float(apy):.3f} mm (elliptical)")
            lines.append(f"  Centre thickness (CT)    : {t:.3f} mm")
            lines.append(f"  Edge thickness (est.)    : {et:.3f} mm @ r={ap * 0.98:.2f}")
            lines.append(f"  Front radius R1 (Rx)     : {R1:.4f} mm")
            lines.append(f"  Rear radius  R2 (Rx)     : {R2:.4f} mm")
            if R1y is not None or mode in ("biconic", "cylinder_y"):
                lines.append(f"  Front radius R1y         : {float(R1y) if R1y is not None else 0.0:.4f} mm")
            if R2y is not None or mode in ("biconic", "cylinder_y"):
                lines.append(f"  Rear radius  R2y         : {float(R2y) if R2y is not None else 0.0:.4f} mm")
            lines.append(f"  Conic k1 / k2            : {float(e.get('k1', 0)):.4f} / {float(e.get('k2', 0)):.4f}")
            lines.append(f"  Asphere A4_1 / A4_2      : {float(e.get('A4_1', 0)):.6g} / {float(e.get('A4_2', 0)):.6g}")
            lines.append(f"  EFL (thin lensmaker)     : {efl:.3f} mm")
            lines.append(f"  Vertex Z (front)         : {z:.3f} mm")
            lines.append(f"  Air gap after element    : {float(e.get('air_after', 0)):.3f} mm")
            lines.append("")
            z += t + float(e.get("air_after", 0.0))
        if not any_on:
            lines.append("(No enabled lens elements.)")
        lines.append("Notes for purchasing:")
        lines.append("  • Prefer stock PCX/DCX with closest |R| and diameter ≥ CA.")
        lines.append("  • Cylinder / biconic parts are usually custom; specify Rx, Ry, CT, CA.")
        lines.append("  • Coatings: uncoated Fresnel losses are modeled; AR coats reduce them.")
        lines.append("  • Sign convention: +R1 = convex toward source; check vendor drawings.")
        return "\n".join(lines)

    def _show_buy_list_window(self):
        """Commercial / RFQ lens parameter sheet (profiles live beside the target plane)."""
        p = self.collect_params()
        report = self._commercial_lens_report(p)
        win = tk.Toplevel(self)
        win.title("OptiFlux — Commercial lens list")
        win.geometry("640x560")
        win.configure(bg=BG)
        bot = ttk.Frame(win)
        bot.pack(fill="both", expand=True, padx=6, pady=4)
        ttk.Label(bot, text="Commercial / RFQ lens parameters", style="Dim.TLabel").pack(anchor="w")
        txt = tk.Text(
            bot,
            wrap="word",
            bg=BG2,
            fg="#f8fafc",
            insertbackground="#f8fafc",
            relief="flat",
            font=("Consolas", 9),
            padx=6,
            pady=6,
        )
        sb = ttk.Scrollbar(bot, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        txt.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        txt.insert("1.0", report)
        txt.configure(state="disabled")

        def _copy():
            self.clipboard_clear()
            self.clipboard_append(report)
            self.status_var.set("Commercial lens list copied to clipboard")

        row = ttk.Frame(win)
        row.pack(fill="x", padx=6, pady=4)
        ttk.Button(row, text="Copy parameters", command=_copy).pack(side="left", padx=4)
        ttk.Button(row, text="Close", command=win.destroy).pack(side="right", padx=4)

    def _design_rect_fov(self, kind: str = "crossed"):
        """Auto-generate anamorphic optics for the current rectangular FOV."""
        mat = material_id_from_name(self.elem_vars[0]["material"].get())
        kwargs = dict(
            fov_width=float(self.v_fov_w.get()),
            fov_height=float(self.v_fov_h.get()),
            target_z=float(self.v_target_z.get()),
            lens_z_start=float(self.v_lens_z.get()),
            source_z=float(self.v_source_z.get()),
            half_angle_deg=float(self.v_half.get()),
            material=mat,
            wavelength_nm=float(self.v_wl.get()),
            aperture=float(self.elem_vars[0]["aperture"].get()),
            thickness=float(self.elem_vars[0]["thickness"].get()),
            custom_n=float(self.v_custom_n.get()),
        )
        if kind == "biconic":
            design = design_biconic_singlet_for_rect_fov(**{
                k: kwargs[k]
                for k in (
                    "fov_width", "fov_height", "target_z", "lens_z_start", "source_z",
                    "material", "wavelength_nm", "aperture", "thickness", "custom_n",
                )
            })
        else:
            design = design_crossed_cylinders_for_rect_fov(**kwargs)

        for i, el in enumerate(design["elements"]):
            if i >= len(self.elem_vars):
                break
            self._apply_element_dict_to_vars(i, el)
        self.v_lens_z.set(design.get("lens_z_start", self.v_lens_z.get()))
        self.v_mla.set(False)
        meta = design.get("meta", {})
        self.design_status.set(meta.get("description", "Design applied"))
        self.status_var.set(meta.get("description", "Rectangular FOV design applied"))
        self._on_param_change()

    def _apply_element_dict_to_vars(self, i: int, el: dict):
        ev = self.elem_vars[i]
        ev["enabled"].set(bool(el.get("enabled", True)))
        for key in ("R1", "R2", "thickness", "air_after", "aperture", "k1", "k2", "A4_1", "A4_2"):
            if key in el and key in ev:
                ev[key].set(el[key])
        mode = el.get("surface_mode", "rotational")
        if "surface_mode" in ev:
            ev["surface_mode"].set(mode)
        # Y radii: explicit values for biconic; otherwise mirror R1/R2
        if el.get("R1y") is not None and "R1y" in ev:
            ev["R1y"].set(el["R1y"])
        elif "R1y" in ev and "R1" in el:
            ev["R1y"].set(float(el["R1"]))
        if el.get("R2y") is not None and "R2y" in ev:
            ev["R2y"].set(el["R2y"])
        elif "R2y" in ev and "R2" in el:
            ev["R2y"].set(float(el["R2"]))
        if el.get("aperture_y") is not None and "aperture_y" in ev:
            ev["aperture_y"].set(el["aperture_y"])
            ev["use_elliptical_ap"].set(True)
        else:
            if "aperture_y" in ev and "aperture" in el:
                ev["aperture_y"].set(float(el["aperture"]))
            if "use_elliptical_ap" in ev:
                ev["use_elliptical_ap"].set(False)
        lock = bool(el.get("circular_lock", True))
        if "circular_lock" in ev:
            ev["circular_lock"].set(lock)
        if lock:
            if "use_elliptical_ap" in ev:
                ev["use_elliptical_ap"].set(False)
            if "aperture_y" in ev and "aperture" in el:
                ev["aperture_y"].set(float(el["aperture"]))
        if "material" in el and "material" in ev:
            ev["material"].set(material_name_from_id(str(el["material"])))
        if "shape_id" in el and "shape" in ev:
            ev["shape"].set(shape_label_from_id(str(el["shape_id"])))
        elif "shape" in ev and mode in ("cylinder_x", "cylinder_y", "biconic"):
            # Anamorphic designs are custom forms
            ev["shape"].set(shape_label_from_id("custom"))
        # Sync |R| magnitude from radii when present
        if "R_mag" in ev and "R1" in el and "R2" in el:
            rm = max(abs(float(el["R1"])), abs(float(el["R2"])), 2.0)
            if rm < 1e-6:
                rm = 25.0
            ev["R_mag"].set(rm)
        if "copy_from" in ev:
            ev["copy_from"].set("(none)")
        if hasattr(self, "elem_ui") and i < len(self.elem_ui):
            self.elem_ui[i]["title"].configure(text=self._element_header_text(i, ev))

    def _apply_shape_to_element(self, elem_index: int = 0):
        """Apply library lens type to the given element index."""
        if elem_index < 0 or elem_index >= len(self.elem_vars):
            return
        ev = self.elem_vars[elem_index]
        sid = shape_id_from_label(ev["shape"].get())
        if sid == "custom":
            self.status_var.set(
                f"Element {elem_index + 1}: custom — set R₁/R₂ (and mode) manually"
            )
            self._on_param_change()
            return

        r_mag = float(ev["R_mag"].get())
        el = apply_shape(
            sid,
            R_mag=r_mag,
            thickness=float(ev["thickness"].get()),
            aperture=float(ev["aperture"].get()),
            material=material_id_from_name(ev["material"].get()),
            k1=float(ev["k1"].get()),
            k2=float(ev["k2"].get()),
            A4_1=float(ev["A4_1"].get()),
            A4_2=float(ev["A4_2"].get()),
            air_after=float(ev["air_after"].get()),
        )
        # Library forms are rotational singlets
        el["surface_mode"] = "rotational"
        el["mode_s1"] = "rotational"
        el["mode_s2"] = "rotational"
        el["R1y"] = None
        el["R2y"] = None
        el["shape_id"] = sid

        # Keep Element 1 design aperture full-size. MLA maps it to die pitch via
        # scale_to_pitch — shrinking here left R macro-sized → flat cylinders.

        for k, v in el.items():
            if k in ev and k not in ("shape_id", "surface_mode", "mode_s1", "mode_s2", "R1y", "R2y"):
                ev[k].set(v)
        ev["surface_mode"].set("rotational")
        ev["enabled"].set(True)
        ev["use_elliptical_ap"].set(False)
        desc = SHAPE_DESCRIPTIONS.get(sid, "")
        self.status_var.set(f"Element {elem_index + 1}: {ev['shape'].get()}" + (f" — {desc}" if desc else ""))
        self._on_param_change()

    def save_design(self):
        """Write all current parameters (source, lens stack, FOV, MLA, sim) to JSON."""
        from design_io import default_designs_dir, save_design

        params = self.collect_params()
        initial = default_designs_dir()
        if self._last_design_path is not None and self._last_design_path.parent.is_dir():
            initial = self._last_design_path.parent
            initialfile = self._last_design_path.name
        else:
            initialfile = "my_lens_group.json"
        path = filedialog.asksaveasfilename(
            title="Save OptiFlux design (all parameters)",
            initialdir=str(initial),
            initialfile=initialfile,
            defaultextension=".json",
            filetypes=[
                ("OptiFlux design", "*.json"),
                ("OptiFlux design", "*.optiflux"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        notes = ""
        # Optional short note stored inside the file
        try:
            from tkinter import simpledialog

            notes = simpledialog.askstring(
                "Design notes",
                "Optional note for this lens group (shown when loading):",
                parent=self,
            ) or ""
        except Exception:
            notes = ""
        try:
            out = save_design(path, params, name=Path(path).stem, notes=notes)
            self._last_design_path = out
            n_on = sum(1 for e in params.get("elements", []) if e.get("enabled"))
            self.status_var.set(f"Saved design → {out}")
            messagebox.showinfo(
                "Design saved",
                f"Wrote:\n{out}\n\n"
                f"Includes: source, {n_on} enabled lens element(s), "
                f"FOV, MLA, materials, and simulation settings.\n\n"
                f"Reload anytime with Design → Load…",
            )
        except Exception as e:
            messagebox.showerror("Save design failed", str(e))
            self.status_var.set("Save design failed")

    def load_design(self):
        """Load a previously saved design and apply it to the UI."""
        from design_io import default_designs_dir, load_design

        initial = default_designs_dir()
        if self._last_design_path is not None and self._last_design_path.parent.is_dir():
            initial = self._last_design_path.parent
        path = filedialog.askopenfilename(
            title="Load OptiFlux design",
            initialdir=str(initial),
            filetypes=[
                ("OptiFlux design", "*.json"),
                ("OptiFlux design", "*.optiflux"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            params, meta = load_design(path)
        except Exception as e:
            messagebox.showerror("Load design failed", str(e))
            self.status_var.set("Load design failed")
            return
        was_auto = bool(self.auto_run.get())
        self.auto_run.set(False)
        try:
            self.params = params
            self._apply_params_to_vars(params)
            self.preset.set("")
            self._last_design_path = Path(path)
            name = meta.get("name") or Path(path).stem
            notes = (meta.get("notes") or "").strip()
            self.status_var.set(f"Loaded design “{name}” — tracing…")
            if notes:
                messagebox.showinfo(
                    f"Loaded: {name}",
                    f"{notes}\n\nSaved: {meta.get('saved_at') or '—'}",
                )
        finally:
            self.auto_run.set(was_auto)
        self.run_trace()

    def _refresh_cad_mesh_label(self, *_):
        """Update the CAD panel estimate of rings / rim spacing / triangles."""
        if not hasattr(self, "cad_mesh_status"):
            return
        try:
            edge = float(self.v_cad_edge.get())
            ang = float(self.v_cad_angle.get())
        except (tk.TclError, ValueError, AttributeError):
            return
        specs = []
        ap = 10.0
        try:
            params = self.collect_params()
            specs, _ = build_lens_specs_from_params(params)
        except Exception:
            specs = []
        if specs:
            n_r, n_t = tessellation_for_specs(
                specs, max_edge_mm=edge, max_angle_deg=ang
            )
            ap = max(float(s.aperture) for s in specs)
        else:
            n_r, n_t = tessellation_from_tolerance(
                ap, [40.0, -50.0], max_edge_mm=edge, max_angle_deg=ang
            )
        rim = 2.0 * math.pi * ap / max(n_t, 1)
        n_faces = 2 * (n_t + 2 * n_t * max(n_r - 1, 0)) + 2 * n_t
        self.cad_mesh_status.set(
            f"≈ {n_r} rings × {n_t} segs · rim {rim:.2f} mm · ~{n_faces:,} triangles"
        )

    def _show_laser_cal_guide(self):
        """Scrollable bench procedure for laser n calibration."""
        win = tk.Toplevel(self)
        win.title("OptiFlux — Laser n calibration")
        win.geometry("640x640")
        win.configure(bg=BG)
        win.minsize(480, 400)
        frm = ttk.Frame(win)
        frm.pack(fill="both", expand=True, padx=8, pady=8)
        txt = tk.Text(
            frm,
            wrap="word",
            bg=BG2,
            fg=FG,
            insertbackground=FG,
            relief="flat",
            font=("Segoe UI", 10),
            padx=10,
            pady=10,
        )
        sb = ttk.Scrollbar(frm, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        txt.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        txt.insert("1.0", laser_calibration_guide())
        txt.configure(state="disabled")
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=6)

    def _calibrate_laser_n(self):
        try:
            p = self.collect_params()
            fit = calibrate_n_from_laser(
                p,
                laser_x=float(self.v_las_hx.get()),
                laser_y=float(self.v_las_hy.get()),
                laser_z=float(self.v_las_z.get()),
                screen_z=float(self.v_las_screen.get()),
                spot_x=float(self.v_las_sx.get()),
                spot_y=float(self.v_las_sy.get()),
                coaxial_spot_x=float(self.v_las_cx.get()),
                coaxial_spot_y=float(self.v_las_cy.get()),
                wavelength_nm=float(self.v_las_wl.get()),
            )
        except (tk.TclError, ValueError) as exc:
            self.laser_cal_status.set(f"Bad input: {exc}")
            return
        self._laser_fit = fit
        extra = ""
        efl = fit.get("efl_mm")
        if efl is not None and math.isfinite(float(efl)):
            extra = f" · E1 lensmaker EFL {float(efl):.2f} mm"
        self.laser_cal_status.set(str(fit.get("message", "")) + extra)
        self.status_var.set(str(fit.get("message", "Laser n fit done")))

    def _apply_laser_n(self):
        fit = getattr(self, "_laser_fit", None) or {}
        if not fit.get("ok"):
            self.laser_cal_status.set("Fit n first (shifted laser, then Fit).")
            return
        p = apply_calibrated_n(self.collect_params(), float(fit["n"]))
        self._apply_params_to_vars(p)
        self.v_custom_n.set(float(fit["n"]))
        self.laser_cal_status.set(
            f"Applied n = {float(fit['n']):.4f} to Custom n and Formlabs elements."
        )
        self._on_param_change()

    def export_cad(self, fmt: str = "stl"):
        params = self.collect_params()
        if not any(e.get("enabled") for e in params["elements"]):
            messagebox.showwarning("Export", "Enable at least one lens element.")
            return
        ext = ".stl" if fmt == "stl" else ".step"
        halves = bool(getattr(self, "v_cad_halves", None) and self.v_cad_halves.get()) and fmt == "stl"
        title = "Export STL (mm)" if fmt == "stl" else "Export STEP (mm)"
        if halves:
            title = "Export STL print halves (mm)"
        path = filedialog.asksaveasfilename(
            title=title,
            defaultextension=ext,
            filetypes=[(f"{fmt.upper()} file", f"*{ext}"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            dies = self.result.dies if self.result else None
            edge = float(self.v_cad_edge.get()) if hasattr(self, "v_cad_edge") else 0.25
            ang = float(self.v_cad_angle.get()) if hasattr(self, "v_cad_angle") else 2.0
            fr = float(self.v_cad_flange_r.get()) if hasattr(self, "v_cad_flange_r") else 0.0
            ft = float(self.v_cad_flange_t.get()) if hasattr(self, "v_cad_flange_t") else 0.0
            out = export_lens(
                params,
                path,
                fmt=fmt,
                dies=dies,
                max_edge_mm=edge,
                max_angle_deg=ang,
                include_plate=bool(self.v_export_plate.get()),
                flange_radial_mm=fr,
                flange_thickness_mm=ft,
                split_halves=halves,
            )
            mla_on = bool(params.get("mla", {}).get("enabled"))
            n_en = sum(1 for e in params.get("elements", []) if e.get("enabled", True))
            if mla_on:
                geom_note = (
                    "MLA: monolithic plate with Element-1 form lenslets (scaled to die pitch)"
                )
            elif n_en > 1:
                geom_note = (
                    f"Lens stack: {n_en} separate elements at correct Z spacings "
                    f"(STEP multi-body; STL multi-shell)"
                )
            else:
                geom_note = "Single lens export"
            if halves:
                geom_note += (
                    "\nPrint halves: each element is cut at the flange mid-plane "
                    "(*_front.stl / *_rear.stl). Build on the flat cut and glue with the same resin."
                )
            try:
                specs, _ = build_lens_specs_from_params(params, dies)
                n_r, n_t = tessellation_for_specs(
                    specs, max_edge_mm=edge, max_angle_deg=ang
                )
                tess_note = (
                    f"Tessellation: max vertex {edge:.3g} mm, max angle {ang:.2g}° "
                    f"→ {n_r} rings × {n_t} segments"
                )
            except Exception:
                tess_note = f"Tessellation: max vertex {edge:.3g} mm, max angle {ang:.2g}°"
            notes = out.with_name(out.stem + "_tube.txt")
            note_line = f"\nTube dimensions: {notes.name}" if notes.is_file() else ""
            messagebox.showinfo(
                "Export complete",
                f"Wrote {out}\n\nUnits: millimetres (mm)\n"
                f"Surfaces use the same aspheric sag model as the ray tracer.\n"
                f"{geom_note}\n{tess_note}"
                f"{note_line}\n"
                f"Flange radial {fr:.2f} mm · thickness {ft:.2f} mm (CAD only).",
            )
            self.status_var.set(f"Exported {out.name}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def export_cnc_profile(self):
        """Write 4-axis GRBL polish G-code. Default extension is .nc for Candle."""
        params = self.collect_params()
        if not any(e.get("enabled") for e in params["elements"]):
            messagebox.showwarning("Export", "Enable at least one lens element.")
            return
        if bool((params.get("mla") or {}).get("enabled")):
            messagebox.showwarning(
                "Export",
                "4-axis polish is for singlet halves. Turn MLA off.",
            )
            return
        path = filedialog.asksaveasfilename(
            title="Export 4-axis polish (mm, GRBL / Candle)",
            defaultextension=CNC_PROFILE_EXT,
            filetypes=[
                ("CNC program", "*.nc"),
                ("G-code", "*.gcode"),
                ("All", "*.*"),
            ],
        )
        if not path:
            return
        try:
            out = export_lens_cnc_profile(params, path)
            extra = ""
            if str(params.get("cad_polish_surface", "front")).lower() == "both":
                extra = (
                    f"\nAlso wrote {out.with_name(out.stem.replace('_front', '') + '_rear.nc').name}"
                    if "_front" in out.stem
                    else "\nFront and rear halves are separate .nc files."
                )
            messagebox.showinfo(
                "Export complete",
                f"Wrote {out}{extra}\n\n"
                "4-axis polish, millimetres.\n"
                "A = optical axis (along X). Z = radial (spherical tool).\n"
                "X0 = printed flat / flange mid-plane. Y = 0.\n"
                "Zero X on the cut face, YZ on the A-axis centre.\n"
                f"Extension is {CNC_PROFILE_EXT} for Candle.",
            )
            self.status_var.set(f"Exported {out.name}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def export_polish_lap_stl(self):
        """Write cylindrical abrasive-lap STL (optical negative + 1/4-20 tap)."""
        params = self.collect_params()
        if not any(e.get("enabled") for e in params["elements"]):
            messagebox.showwarning("Export", "Enable at least one lens element.")
            return
        path = filedialog.asksaveasfilename(
            title="Export polish lap STL (mm)",
            defaultextension=".stl",
            filetypes=[("STL file", "*.stl"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            edge = float(self.v_cad_edge.get()) if hasattr(self, "v_cad_edge") else 0.25
            ang = float(self.v_cad_angle.get()) if hasattr(self, "v_cad_angle") else 2.0
            wall = float(self.v_lap_wall.get()) if hasattr(self, "v_lap_wall") else 6.0
            surf = str(self.v_lap_surf.get()) if hasattr(self, "v_lap_surf") else "front"
            out = export_polish_lap(
                params,
                path,
                surface=surf,
                max_edge_mm=edge,
                max_angle_deg=ang,
                wall_mm=wall,
            )
            notes = Path(path)
            if notes.suffix.lower() != ".stl":
                notes = notes.with_suffix(".stl")
            note_path = notes.with_name(notes.stem + "_tool.txt")
            note_line = f"\nTool dimensions: {note_path.name}" if note_path.is_file() else ""
            extra = ""
            if surf.lower() == "both":
                extra = "\nFront and rear faces are separate STL files."
            messagebox.showinfo(
                "Export complete",
                f"Wrote {out}{extra}{note_line}\n\n"
                "Cylindrical polish lap, millimetres.\n"
                "Working face = negative of the lens surface (coat with abrasive).\n"
                "Opposite end = 1/4-20 UNC tap, #7 drill (5.10 mm), 12.7 mm deep.\n"
                "Tap the hole after printing. Do not break through the optical face.",
            )
            self.status_var.set(f"Exported {out.name}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def _on_preset(self, _=None):
        name = self.preset.get()
        if not name:
            return
        p = default_params()
        if name == "Single LED":
            p["source"]["mode"] = "single"
            p["source"]["die_width"] = 1.2
            p["source"]["die_height"] = 1.2
            p["source"]["half_angle_deg"] = 60
            p["source"]["wavelength_nm"] = VISIBLE_NM_DEFAULT
            p["elements"][0].update(
                {
                    "enabled": True,
                    "R1": 12,
                    "R2": -18,
                    "thickness": 5,
                    "air_after": 1,
                    "aperture": 8,
                    "material": "ACRYLIC_PMMA",
                    "k1": -0.8,
                    "k2": -0.5,
                }
            )
            p["elements"][1]["enabled"] = False
            p["elements"][2]["enabled"] = False
            p["lens_z_start"] = 1.5
            p["target_z"] = 80
            p["fov_width"] = 25
            p["fov_height"] = 20
            p["map_half_w"] = 30
            p["map_half_h"] = 25
        elif name == "COB 4×4":
            p["source"]["mode"] = "cob"
            p["source"]["rows"] = 4
            p["source"]["cols"] = 4
            p["source"]["wavelength_nm"] = VISIBLE_NM_DEFAULT
            p["elements"][0].update(
                {
                    "enabled": True,
                    "R1": 30,
                    "R2": -50,
                    "thickness": 6,
                    "aperture": 14,
                    "material": "N_BK7",
                    "k1": 0,
                    "k2": 0,
                }
            )
            p["target_z"] = 120
            p["fov_width"] = 50
            p["fov_height"] = 40
        elif name == "Visible COB acrylic":
            p["source"].update(
                {
                    "mode": "cob",
                    "rows": 5,
                    "cols": 6,
                    "pitch_x": 1.4,
                    "pitch_y": 1.4,
                    "die_width": 1.0,
                    "die_height": 1.0,
                    "wavelength_nm": VISIBLE_NM_DEFAULT,
                    "half_angle_deg": 50,
                    "circular_mask": True,
                    "mask_radius": 4.2,
                }
            )
            p["elements"][0].update(
                {
                    "enabled": True,
                    "R1": 35,
                    "R2": -45,
                    "thickness": 5,
                    "air_after": 1.5,
                    "aperture": 16,
                    "material": "ACRYLIC_PMMA",
                    "k1": -1,
                    "A4_1": 2e-5,
                }
            )
            p["elements"][1].update(
                {
                    "enabled": True,
                    "R1": 40,
                    "R2": -60,
                    "thickness": 4,
                    "aperture": 15,
                    "material": "ACRYLIC_PMMA",
                    "k2": -0.5,
                    "A4_2": 1e-5,
                }
            )
            p["elements"][2]["enabled"] = False
            p["lens_z_start"] = 3
            p["target_z"] = 200
            p["fov_width"] = 80
            p["fov_height"] = 60
            p["map_half_w"] = 70
            p["map_half_h"] = 55
            p["total_rays"] = 10000
        elif name == "Formlabs Clear MLA":
            p["source"].update(
                {
                    "mode": "cob",
                    "rows": 4,
                    "cols": 4,
                    "pitch_x": 1.6,
                    "pitch_y": 1.6,
                    "wavelength_nm": VISIBLE_NM_DEFAULT,
                    "half_angle_deg": 55,
                }
            )
            # Full-size Element 1 design — MLA scales R/t/ap to die pitch
            p["elements"][0] = apply_shape(
                "convex_plano_PCX",
                R_mag=18,
                thickness=4.0,
                aperture=10.0,
                material="FORMLABS_CLEAR",
            )
            p["elements"][1]["enabled"] = False
            p["elements"][2]["enabled"] = False
            p["mla"]["enabled"] = True
            p["mla"]["fill_factor"] = 0.88
            p["mla"]["scale_to_pitch"] = True
            p["lens_z_start"] = 1.2
            p["target_z"] = 100
            p["total_rays"] = 8000
        elif name == "Collimator":
            p["source"]["mode"] = "single"
            p["source"]["half_angle_deg"] = 45
            p["source"]["wavelength_nm"] = VISIBLE_NM_DEFAULT
            p["elements"][0].update(
                {
                    "enabled": True,
                    "R1": 0,
                    "R2": -15,
                    "thickness": 8,
                    "aperture": 10,
                    "material": "ACRYLIC_PMMA",
                    "k2": -1,
                }
            )
            p["elements"][1]["enabled"] = False
            p["elements"][2]["enabled"] = False
            p["lens_z_start"] = 2
            p["target_z"] = 500
            p["fov_width"] = 30
            p["fov_height"] = 30
            p["map_half_w"] = 40
            p["map_half_h"] = 40
        elif name == "Rect FOV · crossed cylinders":
            p["source"]["mode"] = "single"
            p["source"]["die_width"] = 1.2
            p["source"]["die_height"] = 1.2
            p["source"]["half_angle_deg"] = 55
            p["source"]["wavelength_nm"] = VISIBLE_NM_DEFAULT
            p["fov_width"] = 48
            p["fov_height"] = 32  # 3:2 aspect
            p["target_z"] = 100
            p["map_half_w"] = 50
            p["map_half_h"] = 40
            design = design_crossed_cylinders_for_rect_fov(
                fov_width=48,
                fov_height=32,
                target_z=100,
                lens_z_start=3.0,
                material="ACRYLIC_PMMA",
                wavelength_nm=VISIBLE_NM_DEFAULT,
                aperture=14,
                thickness=4,
            )
            p["elements"] = design["elements"]
            p["lens_z_start"] = design["lens_z_start"]
            p["mla"]["enabled"] = False
        elif name == "Rect FOV · biconic singlet":
            p["source"]["mode"] = "cob"
            p["source"]["rows"] = 3
            p["source"]["cols"] = 3
            p["source"]["wavelength_nm"] = VISIBLE_NM_DEFAULT
            p["fov_width"] = 60
            p["fov_height"] = 40  # 3:2
            p["target_z"] = 120
            p["map_half_w"] = 55
            p["map_half_h"] = 45
            design = design_biconic_singlet_for_rect_fov(
                fov_width=60,
                fov_height=40,
                target_z=120,
                lens_z_start=4.0,
                material="ACRYLIC_PMMA",
                aperture=16,
                thickness=5,
            )
            p["elements"] = design["elements"]
            p["lens_z_start"] = design["lens_z_start"]
            p["mla"]["enabled"] = False

        self._apply_params_to_vars(p)
        self.preset.set("")
        self.run_trace()


def main():
    app = OptiFluxApp()
    app.mainloop()


if __name__ == "__main__":
    main()
