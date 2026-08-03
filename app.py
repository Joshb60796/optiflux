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
from tkinter import ttk, messagebox, filedialog
from typing import Any, Dict, Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap

from engine import (
    MATERIAL_NAMES,
    default_params,
    run_simulation,
    SimResult,
)
from progressive import run_simulation_progressive
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
from export_cad import export_lens
from rect_fov import (
    design_biconic_singlet_for_rect_fov,
    design_crossed_cylinders_for_rect_fov,
    fov_aspect,
    set_fov_from_aspect,
)
from optimizer import OptimizeConfig, optimize_fov_flux


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
BORDER = "#1e2a3a"

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
    ):
        super().__init__(parent)
        self.var = var
        self.is_int = is_int
        self.from_ = float(from_)
        self.to = float(to)
        self.resolution = float(resolution) if resolution else 1.0
        self._command = command
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
        self.result: Optional[SimResult] = None
        self._run_lock = threading.Lock()
        self._running = False
        self._trace_gen = 0  # bumps on every new progressive run (cancels prior)
        self._debounce_id = None
        self._opt_gen = 0  # bumps to cancel an in-flight optimizer
        self._optimizing = False
        # Progressive defaults: 5 batches × 5000 map rays + 500 side paths
        self.prog_batches = 5
        self.prog_rays_batch = 5000
        self.prog_disp_batch = 500
        self.auto_run = tk.BooleanVar(value=True)
        self.log_scale = tk.BooleanVar(value=False)
        # Side-view lens drag (recalculate only on mouse release)
        self._drag: Optional[Dict[str, Any]] = None
        self._element_handles: list = []  # pick targets from last side-view draw
        self._side_cid = {}  # matplotlib event connection ids
        # Side-view zoom/pan (Z–Y mm)
        self._side_xlim: Optional[tuple] = None
        self._side_ylim: Optional[tuple] = None
        self._side_full_extent: Optional[tuple] = None  # (zmin,zmax,ymin,ymax)
        self._side_pan: Optional[Dict[str, Any]] = None
        # Target-plane view zoom/pan (data limits in mm)
        self._tgt_xlim: Optional[tuple] = None
        self._tgt_ylim: Optional[tuple] = None
        self._tgt_full_extent: Optional[tuple] = None  # (xmin,xmax,ymin,ymax)
        self._tgt_pan: Optional[Dict[str, Any]] = None

        self._init_style()
        self._build_vars()
        self._build_ui()
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
        style.configure("Vertical.TScrollbar", background=BG3)

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
        self.v_mesh_res = tk.IntVar(value=48)

        self.elem_vars = []
        for e in self.params["elements"]:
            r1y = e.get("R1y")
            r2y = e.get("R2y")
            apy = e.get("aperture_y")
            sid = e.get("shape_id", "custom")
            r_mag = max(abs(float(e.get("R1", 25))), abs(float(e.get("R2", 25))), 2.0)
            if r_mag < 1e-6:
                r_mag = 25.0
            self.elem_vars.append(
                {
                    "enabled": tk.BooleanVar(value=e["enabled"]),
                    "shape": tk.StringVar(value=shape_label_from_id(sid)),
                    "R_mag": tk.DoubleVar(value=r_mag),
                    "R1": tk.DoubleVar(value=e["R1"]),
                    "R2": tk.DoubleVar(value=e["R2"]),
                    "R1y": tk.DoubleVar(value=float(r1y) if r1y is not None else e["R1"]),
                    "R2y": tk.DoubleVar(value=float(r2y) if r2y is not None else e["R2"]),
                    "thickness": tk.DoubleVar(value=e["thickness"]),
                    "air_after": tk.DoubleVar(value=e["air_after"]),
                    "aperture": tk.DoubleVar(value=e["aperture"]),
                    "aperture_y": tk.DoubleVar(value=float(apy) if apy is not None else e["aperture"]),
                    "material": tk.StringVar(value=material_name_from_id(e["material"])),
                    "surface_mode": tk.StringVar(value=e.get("surface_mode", "rotational")),
                    "k1": tk.DoubleVar(value=e["k1"]),
                    "k2": tk.DoubleVar(value=e["k2"]),
                    "A4_1": tk.DoubleVar(value=e["A4_1"]),
                    "A4_2": tk.DoubleVar(value=e["A4_2"]),
                    "use_elliptical_ap": tk.BooleanVar(value=apy is not None),
                    "use_biconic_radii": tk.BooleanVar(
                        value=e.get("surface_mode", "rotational") != "rotational"
                        or r1y is not None
                    ),
                }
            )

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
        ttk.Button(header, text="Optimize FOV", command=self.run_optimize).pack(
            side="right", padx=4, pady=8
        )
        ttk.Checkbutton(header, text="Auto-run", variable=self.auto_run).pack(side="right", padx=6)
        ttk.Button(header, text="Export STL…", command=lambda: self.export_cad("stl")).pack(
            side="right", padx=4
        )
        ttk.Button(header, text="Export STEP…", command=lambda: self.export_cad("step")).pack(
            side="right", padx=4
        )
        ttk.Button(header, text="Reset defaults", command=self.reset_defaults).pack(side="right", padx=4)

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
        scroll = ttk.Scrollbar(left_wrap, orient="vertical", command=canvas.yview)
        self.ctrl_frame = ttk.Frame(canvas)
        self._ctrl_canvas = canvas
        self._ctrl_window = canvas.create_window((0, 0), window=self.ctrl_frame, anchor="nw")

        def _on_frame_configure(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            # Keep embedded frame full width of the canvas
            canvas.itemconfigure(self._ctrl_window, width=max(event.width - 4, 200))

        self.ctrl_frame.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

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
            text="Side: scroll=zoom · right-drag=pan · left-drag lens · double-click=reset",
            style="Dim.TLabel",
        ).pack(side="left")
        ttk.Button(side_tools, text="Reset side zoom", command=self._reset_side_zoom).pack(
            side="right", padx=4
        )

        self.fig_side = Figure(figsize=(6, 3.2), dpi=100, facecolor=BG)
        self.ax_side = self.fig_side.add_subplot(111)
        self.canvas_side = FigureCanvasTkAgg(self.fig_side, master=center)
        side_widget = self.canvas_side.get_tk_widget()
        side_widget.pack(side="top", fill="both", expand=True, padx=2, pady=2)
        side_widget.configure(cursor="hand2")
        self._connect_side_mouse()

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

        self.fig_tgt = Figure(figsize=(6, 3.2), dpi=100, facecolor=BG)
        self.ax_tgt = self.fig_tgt.add_subplot(111)
        self.canvas_tgt = FigureCanvasTkAgg(self.fig_tgt, master=bot)
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

    def _add_slider(self, parent, label, var, lo, hi, res=0.1, is_int=False):
        row = SliderRow(parent, label, var, lo, hi, res, is_int, command=self._on_param_change)
        row.pack(fill="x", padx=8, pady=1)
        return row

    def _build_controls(self, parent):
        # Source
        src = ttk.LabelFrame(parent, text="SOURCE  ·  LED / COB")
        src.pack(fill="x", padx=8, pady=6)

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
        self._add_slider(src, "Pitch X (mm)", self.v_pitch_x, 0.3, 8, 0.05)
        self._add_slider(src, "Pitch Y (mm)", self.v_pitch_y, 0.3, 8, 0.05)
        self._add_slider(src, "Die width (mm)", self.v_die_w, 0.1, 5, 0.05)
        self._add_slider(src, "Die height (mm)", self.v_die_h, 0.1, 5, 0.05)
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
        self._add_slider(src, "Mask radius (mm)", self.v_mask_r, 0.5, 20, 0.1)

        # Optics
        opt = ttk.LabelFrame(parent, text="OPTICS  ·  lens stack")
        opt.pack(fill="x", padx=8, pady=6)

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
            text="STL / STEP surfaces use the same aspheric sag as the ray tracer. Coordinates in millimetres.",
            style="Dim.TLabel",
            wraplength=290,
        ).pack(anchor="w", padx=6, pady=2)
        self._add_slider(cad_box, "Mesh radial rings", self.v_mesh_res, 16, 96, 4, True)
        ttk.Button(cad_box, text="Export STL (binary, mm)…", command=lambda: self.export_cad("stl")).pack(
            fill="x", padx=6, pady=2
        )
        ttk.Button(cad_box, text="Export STEP (mm)…", command=lambda: self.export_cad("step")).pack(
            fill="x", padx=6, pady=2
        )

        self._add_slider(opt, "First vertex Z (mm)", self.v_lens_z, 0.5, 40, 0.1)
        self._add_slider(opt, "Custom material n", self.v_custom_n, 1.3, 2.5, 0.001)
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

        for i, ev in enumerate(self.elem_vars):
            el = ttk.LabelFrame(opt, text=f"Element {i + 1}")
            el.pack(fill="x", padx=6, pady=4)
            ttk.Checkbutton(
                el, text="Enabled", variable=ev["enabled"], command=self._on_param_change
            ).pack(anchor="w", padx=4)

            # Per-element lens type / material / mode (full-width for reliable clicks)
            ttk.Label(el, text="Lens type", style="Dim.TLabel").pack(anchor="w", padx=4, pady=(4, 0))
            shape_cb = self._make_combobox(
                el,
                ev["shape"],
                shape_dropdown_values(),
                width=32,
                command=lambda e, idx=i: self._on_element_shape_selected(idx),
            )
            shape_cb.pack(fill="x", padx=4, pady=2)

            SliderRow(
                el,
                "|R| magnitude (mm) for type",
                ev["R_mag"],
                2,
                200,
                0.5,
                command=lambda idx=i: self._on_element_r_mag(idx),
            ).pack(fill="x", padx=8, pady=1)

            ttk.Label(el, text="Material", style="Dim.TLabel").pack(anchor="w", padx=4, pady=(4, 0))
            mat_cb = self._make_combobox(
                el,
                ev["material"],
                MATERIAL_NAMES,
                width=32,
                command=lambda e: self._on_param_change(),
            )
            mat_cb.pack(fill="x", padx=4, pady=2)

            ttk.Label(el, text="Surface mode", style="Dim.TLabel").pack(anchor="w", padx=4, pady=(4, 0))
            mode_cb = self._make_combobox(
                el,
                ev["surface_mode"],
                ["rotational", "biconic", "cylinder_x", "cylinder_y"],
                width=32,
                command=lambda e: self._on_param_change(),
            )
            mode_cb.pack(fill="x", padx=4, pady=2)

            self._add_slider(el, "R₁ / Rₓ front (mm)", ev["R1"], -200, 200, 0.5)
            self._add_slider(el, "R₂ / Rₓ rear (mm)", ev["R2"], -200, 200, 0.5)
            self._add_slider(el, "R₁ᵧ front (mm) · biconic/cyl Y", ev["R1y"], -200, 200, 0.5)
            self._add_slider(el, "R₂ᵧ rear (mm) · biconic/cyl Y", ev["R2y"], -200, 200, 0.5)
            self._add_slider(el, "Thickness (mm)", ev["thickness"], 0.2, 30, 0.1)
            self._add_slider(el, "Air gap after (mm)", ev["air_after"], 0, 50, 0.1)
            self._add_slider(el, "Semi-aperture X (mm)", ev["aperture"], 1, 50, 0.1)
            self._add_slider(el, "Semi-aperture Y (mm)", ev["aperture_y"], 1, 50, 0.1)
            ttk.Checkbutton(
                el,
                text="Elliptical clear aperture (use X & Y)",
                variable=ev["use_elliptical_ap"],
                command=self._on_param_change,
            ).pack(anchor="w", padx=4)
            self._add_slider(el, "Conic k₁", ev["k1"], -5, 5, 0.01)
            self._add_slider(el, "Conic k₂", ev["k2"], -5, 5, 0.01)
            self._add_slider(el, "Asphere A4₁", ev["A4_1"], -0.001, 0.001, 1e-6)
            self._add_slider(el, "Asphere A4₂", ev["A4_2"], -0.001, 0.001, 1e-6)

        # Target — rectangular FOV (camera field)
        tgt = ttk.LabelFrame(parent, text="TARGET  ·  rectangular FOV (camera field)")
        tgt.pack(fill="x", padx=8, pady=6)
        ttk.Label(
            tgt,
            text="Define the rectangular region to fill (aspect ratio like a camera FOV). "
            "Use Design tools below to generate anamorphic optics that reshape LED/COB light into that rectangle.",
            style="Dim.TLabel",
            wraplength=290,
        ).pack(anchor="w", padx=6, pady=2)
        self._add_slider(tgt, "Target distance Z (mm)", self.v_target_z, 20, 1000, 1)
        self._add_slider(tgt, "FOV width (mm)", self.v_fov_w, 5, 300, 1)
        self._add_slider(tgt, "FOV height (mm)", self.v_fov_h, 5, 300, 1)
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
        self._add_slider(tgt, "FOV center X (mm)", self.v_fov_cx, -50, 50, 0.5)
        self._add_slider(tgt, "FOV center Y (mm)", self.v_fov_cy, -50, 50, 0.5)

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
        self.design_status = tk.StringVar(value="")
        ttk.Label(design, textvariable=self.design_status, style="Dim.TLabel", wraplength=280).pack(
            anchor="w", padx=4, pady=2
        )
        self._add_slider(tgt, "Map half-width (mm)", self.v_map_w, 10, 200, 1)
        self._add_slider(tgt, "Map half-height (mm)", self.v_map_h, 10, 200, 1)
        self._add_slider(tgt, "Map resolution (bins)", self.v_map_res, 32, 256, 16, True)

        # Sim
        sim = ttk.LabelFrame(parent, text="SIMULATION")
        sim.pack(fill="x", padx=8, pady=6)
        self._add_slider(sim, "Rays to trace", self.v_rays, 500, 100000, 500, True)
        self._add_slider(sim, "Side-view ray paths", self.v_disp, 20, 5000, 10, True)
        ttk.Label(
            sim,
            text="More rays → cleaner flux map. Use 15k–30k for final uniformity.",
            style="Dim.TLabel",
            wraplength=290,
        ).pack(anchor="w", padx=8, pady=4)

        # Optimizer — maximize power into the rectangular FOV
        optz = ttk.LabelFrame(parent, text="OPTIMIZER  ·  rectangular FOV flux")
        optz.pack(fill="x", padx=8, pady=6)
        ttk.Label(
            optz,
            text=(
                "Objective fills the rectangular FOV: flux × coverage × uniformity, "
                "penalizing under-size and wrong aspect. Multi-starts several group "
                "distances (near LED → farther) so conjugate scale is explored. "
                "Phase 2 adds anamorphic lenses for a rectangular footprint."
            ),
            style="Dim.TLabel",
            wraplength=300,
        ).pack(anchor="w", padx=6, pady=2)
        self.v_opt_rays = tk.IntVar(value=2500)
        self.v_opt_evals = tk.IntVar(value=80)
        self.v_opt_uni_w = tk.DoubleVar(value=0.35)
        self.v_opt_aspect_w = tk.DoubleVar(value=1.5)
        self.v_opt_fill_w = tk.DoubleVar(value=1.5)
        self.v_opt_two_phase = tk.BooleanVar(value=True)
        self.v_opt_extra = tk.IntVar(value=2)
        self.v_opt_ana_mode = tk.StringVar(value="crossed")
        self.v_opt_asphere = tk.BooleanVar(value=False)
        self.v_opt_polish = tk.BooleanVar(value=True)
        self._add_slider(optz, "Rays per evaluation", self.v_opt_rays, 500, 15000, 500, True)
        self._add_slider(optz, "Max evaluations (approx.)", self.v_opt_evals, 20, 300, 10, True)
        self._add_slider(optz, "Uniformity weight", self.v_opt_uni_w, 0.0, 2.0, 0.05)
        self._add_slider(
            optz,
            "FOV-fill weight (size match; under-fill hurts)",
            self.v_opt_fill_w,
            0.0,
            4.0,
            0.1,
        )
        self._add_slider(optz, "Aspect-match weight (phase 2)", self.v_opt_aspect_w, 0.0, 4.0, 0.1)
        ttk.Checkbutton(
            optz,
            text="Two-phase: even FOV → add anamorphic lenses",
            variable=self.v_opt_two_phase,
        ).pack(anchor="w", padx=6)
        self._add_slider(
            optz,
            "Extra anamorphic lenses (phase 2)",
            self.v_opt_extra,
            0,
            2,
            1,
            True,
        )
        ttk.Label(optz, text="Anamorphic form", style="Dim.TLabel").pack(
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
            text="Optimize rectangular FOV",
            style="Accent.TButton",
            command=self.run_optimize,
        ).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text="Cancel", command=self.cancel_optimize).pack(side="left")
        self.opt_status = tk.StringVar(
            value="Two-phase on: P1 even illumination, P2 reshape to FOV rectangle."
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
            ("collection", "Collection efficiency"),
            ("rms", "RMS spot radius"),
            ("ee50", "Encircled energy R(50%)"),
            ("ee86", "Encircled energy R(86%)"),
            ("peak", "Peak irradiance"),
            ("fov_frac", "Flux in rectangular FOV"),
            ("uniform", "FOV uniformity Emin/Emax"),
            ("cv", "FOV CV (σ/μ)"),
            ("aspect", "Footprint aspect σx/σy"),
            ("aspect_tgt", "Target FOV aspect W/H"),
            ("aspect_err", "Aspect error |fp−tgt|/tgt"),
            ("sig", "Spot RMS half-widths σx, σy"),
            ("centroid", "Centroid (X, Y)"),
            ("efl", "Element 1 EFL (lensmaker)"),
            ("dies", "Active dies / surfaces"),
            ("tir", "TIR absorbed / reflections"),
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
            if mode in ("biconic", "cylinder_x", "cylinder_y"):
                el["R1y"] = float(ev["R1y"].get())
                el["R2y"] = float(ev["R2y"].get())
            else:
                el["R1y"] = None
                el["R2y"] = None
            if use_ell:
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
        }
        p["target_z"] = float(self.v_target_z.get())
        p["fov_width"] = float(self.v_fov_w.get())
        p["fov_height"] = float(self.v_fov_h.get())
        p["fov_aspect_lock"] = bool(self.v_fov_lock.get())
        p["fov_cx"] = float(self.v_fov_cx.get())
        p["fov_cy"] = float(self.v_fov_cy.get())
        p["map_half_w"] = float(self.v_map_w.get())
        p["map_half_h"] = float(self.v_map_h.get())
        p["map_res"] = int(self.v_map_res.get())
        p["total_rays"] = int(self.v_rays.get())
        p["display_rays"] = int(self.v_disp.get())
        return p

    def _on_param_change(self, *_):
        if not self.auto_run.get():
            return
        if self._debounce_id is not None:
            self.after_cancel(self._debounce_id)
        self._debounce_id = self.after(350, self.run_trace)

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

        acquired = self._run_lock.acquire(blocking=False)
        if not acquired:
            # Previous worker still holds the lock — it will exit on cancel;
            # schedule a retry shortly so the new gen actually runs.
            self.after(80, lambda g=gen: self._retry_trace_if_current(g))
            self.status_var.set(f"Waiting to start batch 1/{self.prog_batches}…")
            return

        self._running = True
        self.status_var.set(f"Tracing batch 1/{self.prog_batches}…")
        self.progress["value"] = 0

        def work():
            def prog(f):
                self.after(0, lambda v=f: self.progress.configure(value=v * 100))

            def should_cancel():
                return gen != self._trace_gen

            def on_batch(result, bi, n_batches):
                if gen != self._trace_gen:
                    return
                self.after(0, lambda r=result, b=bi, n=n_batches: self._on_batch(r, b, n, gen))

            try:
                run_simulation_progressive(
                    params,
                    batch_cb=on_batch,
                    n_batches=self.prog_batches,
                    rays_per_batch=self.prog_rays_batch,
                    display_per_batch=self.prog_disp_batch,
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
        self.metric_labels["backend"].set(str(st.get("backend", "cpu")))

    # ── Drawing ──────────────────────────────────────────────────────────

    def _style_axes(self):
        for ax in (self.ax_side, self.ax_tgt):
            ax.set_facecolor(BG)
            ax.tick_params(colors=FG, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(BORDER)
            ax.xaxis.label.set_color(FG)
            ax.yaxis.label.set_color(FG)
            ax.title.set_color(FG_BRIGHT)

    def _redraw_empty(self):
        self.ax_side.clear()
        self.ax_tgt.clear()
        self._style_axes()
        self.ax_side.set_title("SIDE VIEW  ·  Y–Z meridional", loc="left", fontsize=10)
        self.ax_side.set_xlabel("Z (mm)")
        self.ax_side.set_ylabel("Y (mm)")
        self.ax_tgt.set_title("TARGET PLANE  ·  irradiance (source → field)", loc="left", fontsize=10)
        self.ax_tgt.set_xlabel("X (mm)")
        self.ax_tgt.set_ylabel("Y (mm)")
        self.fig_side.tight_layout()
        self.fig_tgt.tight_layout()
        self.canvas_side.draw_idle()
        self.canvas_tgt.draw_idle()

    def _redraw(self):
        if self.result is None:
            self._redraw_empty()
            return
        self._draw_side()
        if self._drag is None:
            self._draw_target()

    def _connect_side_mouse(self):
        c = self.canvas_side
        self._side_cid["press"] = c.mpl_connect("button_press_event", self._on_side_press)
        self._side_cid["motion"] = c.mpl_connect("motion_notify_event", self._on_side_motion)
        self._side_cid["release"] = c.mpl_connect("button_release_event", self._on_side_release)
        self._side_cid["scroll"] = c.mpl_connect("scroll_event", self._on_side_scroll)

    def _reset_side_zoom(self):
        self._side_xlim = None
        self._side_ylim = None
        self._side_pan = None
        if self.result is not None:
            self._draw_side()
        self.status_var.set("Side view zoom reset")

    def _apply_side_limits(self, ax):
        """Restore side-view zoom after redraw; clamp to full scene extent."""
        if self._side_full_extent is None:
            return
        zmin, zmax, ymin, ymax = self._side_full_extent
        if self._side_xlim is None or self._side_ylim is None:
            ax.set_xlim(zmin, zmax)
            ax.set_ylim(ymin, ymax)
            return
        x0, x1 = self._side_xlim
        y0, y1 = self._side_ylim
        w, h = x1 - x0, y1 - y0
        full_w, full_h = zmax - zmin, ymax - ymin
        if w >= full_w * 0.999:
            x0, x1 = zmin, zmax
        else:
            if x0 < zmin:
                x1 += zmin - x0
                x0 = zmin
            if x1 > zmax:
                x0 -= x1 - zmax
                x1 = zmax
        if h >= full_h * 0.999:
            y0, y1 = ymin, ymax
        else:
            if y0 < ymin:
                y1 += ymin - y0
                y0 = ymin
            if y1 > ymax:
                y0 -= y1 - ymax
                y1 = ymax
        self._side_xlim = (x0, x1)
        self._side_ylim = (y0, y1)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

    def _on_side_scroll(self, event):
        """Mouse wheel zoom on side view (Z–Y), centered on cursor."""
        if self._drag is not None:
            return
        if self.ax_side is None or event.inaxes != self.ax_side:
            return
        if event.xdata is None or event.ydata is None:
            return
        if event.button == "up":
            scale = 0.8
        elif event.button == "down":
            scale = 1.25
        else:
            return
        ax = self.ax_side
        zc, yc = float(event.xdata), float(event.ydata)
        z0, z1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        new_w = (z1 - z0) * scale
        new_h = (y1 - y0) * scale
        if self._side_full_extent is not None:
            fz0, fz1, fy0, fy1 = self._side_full_extent
            min_w = max(0.5, 0.02 * (fz1 - fz0))
            min_h = max(0.2, 0.02 * (fy1 - fy0))
            max_w = (fz1 - fz0) * 1.05
            max_h = (fy1 - fy0) * 1.05
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
        self._side_ylim = (ny0, ny1)
        self._apply_side_limits(ax)
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
        if not self._element_handles:
            return None
        best = None
        best_d = 1e9
        res = self.result
        for h in self._element_handles:
            ap = h["aperture"]
            y_lim = ap * 1.15
            if h.get("mla") and res and res.dies:
                ys = [d.cy for d in res.dies]
                y_lim = max(abs(min(ys)), abs(max(ys))) + ap * 1.2
            if h["front_z"] - 1.5 <= z <= h["rear_z"] + 1.5 and abs(y) <= y_lim:
                d = abs(z - 0.5 * (h["front_z"] + h["rear_z"]))
                if d < best_d:
                    best_d = d
                    best = h
            else:
                d = math.hypot(z - h["front_z"], max(0.0, abs(y) - y_lim))
                if d < 3.0 and d < best_d:
                    best_d = d
                    best = h
        return best

    def _clamp_element_front_z(self, elem_index: int, new_front: float) -> float:
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
        prev = None
        for item in layout:
            if item["index"] == elem_index:
                break
            prev = item
        if prev is not None:
            z_min = max(z_min, prev["rear_z"] + 0.2)
        return max(z_min, min(z_max, new_front))

    def _apply_element_front_z(self, elem_index: int, new_front: float) -> None:
        new_front = self._clamp_element_front_z(elem_index, new_front)
        layout = self._element_layout()
        enabled_indices = [h["index"] for h in layout]
        if elem_index not in enabled_indices:
            return
        pos = enabled_indices.index(elem_index)
        if pos == 0:
            self.v_lens_z.set(round(new_front, 3))
        else:
            prev_idx = enabled_indices[pos - 1]
            prev_ev = self.elem_vars[prev_idx]
            prev_front = next(h["front_z"] for h in layout if h["index"] == prev_idx)
            prev_thick = float(prev_ev["thickness"].get())
            prev_rear = prev_front + prev_thick
            air = max(0.05, new_front - prev_rear)
            prev_ev["air_after"].set(round(air, 3))
        self.status_var.set(f"Moved element {elem_index + 1} front → Z = {new_front:.2f} mm")

    def _on_side_press(self, event):
        if event.inaxes != self.ax_side:
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
                "xlim": self.ax_side.get_xlim(),
                "ylim": self.ax_side.get_ylim(),
            }
            self.canvas_side.get_tk_widget().configure(cursor="fleur")
            return
        # Left = drag lens element (if hit)
        if event.button != 1:
            return
        if event.xdata is None or event.ydata is None or self._running:
            return
        hit = self._pick_element(float(event.xdata), float(event.ydata))
        if hit is None:
            return
        self._drag = {
            "elem_index": hit["index"],
            "label": hit["label"],
            "orig_front": hit["front_z"],
            "press_z": float(event.xdata),
            "current_front": hit["front_z"],
            "aperture": hit["aperture"],
            "thickness": hit["thickness"],
        }
        self.canvas_side.get_tk_widget().configure(cursor="sb_h_double_arrow")
        self.status_var.set(
            f"Dragging {hit['label']}  ·  release to re-trace  ·  Z = {hit['front_z']:.2f} mm"
        )

    def _on_side_motion(self, event):
        # Pan takes priority when active
        if self._side_pan is not None:
            if event.x is None or event.y is None or self.ax_side is None:
                return
            ax = self.ax_side
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
            self._side_ylim = (y0 + dy, y1 + dy)
            self._apply_side_limits(ax)
            self.canvas_side.draw_idle()
            return

        if self._drag is None:
            if event.inaxes == self.ax_side and event.xdata is not None and event.ydata is not None:
                hit = self._pick_element(float(event.xdata), float(event.ydata))
                self.canvas_side.get_tk_widget().configure(
                    cursor="sb_h_double_arrow" if hit else "hand2"
                )
            return
        if event.inaxes != self.ax_side or event.xdata is None:
            return
        dz = float(event.xdata) - self._drag["press_z"]
        new_front = self._clamp_element_front_z(
            self._drag["elem_index"], self._drag["orig_front"] + dz
        )
        self._drag["current_front"] = new_front
        self.status_var.set(
            f"Dragging {self._drag['label']}  ·  Z = {new_front:.2f} mm  ·  release to re-trace"
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
        new_front = drag["current_front"]
        if abs(new_front - drag["orig_front"]) < 1e-4:
            self.status_var.set("Ready")
            self._draw_side()
            return
        self._apply_element_front_z(drag["elem_index"], new_front)
        if self._debounce_id is not None:
            self.after_cancel(self._debounce_id)
            self._debounce_id = None
        self.status_var.set(f"{drag['label']} at Z = {new_front:.2f} mm — tracing…")
        self.run_trace()

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
    ):
        """
        Draw one lens as a closed Y–Z section about the lenslet axis (y0).
        Decentered MLA lenslets use y = y0 ± r (not mirrored through y=0).
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
        y0 = float(s1.y0)
        z_front_u, y_front_u = [], []
        z_rear_u, y_rear_u = [], []
        for i in range(nseg + 1):
            r = ap * i / nseg
            sag1 = s1.sag_xy(0.0, r)
            sag2 = s2.sag_xy(0.0, r)
            if sag1 is None or sag2 is None:
                break
            zf = s1.z_vertex + sag1 + z_off
            zr = s2.z_vertex + sag2 + z_off
            if zr < zf + min_edge * 0.5:
                break
            z_front_u.append(zf)
            y_front_u.append(y0 + r)
            z_rear_u.append(zr)
            y_rear_u.append(y0 + r)
        if len(z_front_u) < 2:
            return

        # Lower rim about lenslet axis
        z_front_l = list(z_front_u)
        y_front_l = [y0 - (y - y0) for y in y_front_u]
        z_rear_l = list(z_rear_u)
        y_rear_l = [y0 - (y - y0) for y in y_rear_u]

        poly_z = (
            z_front_u
            + list(reversed(z_rear_u))
            + z_rear_l
            + list(reversed(z_front_l))
        )
        poly_y = (
            y_front_u
            + list(reversed(y_rear_u))
            + y_rear_l
            + list(reversed(y_front_l))
        )

        col = "#fbbf24" if highlight else LENS
        lw = 1.2 if compact else (2.0 if highlight else 1.8)
        ax.fill(poly_z, poly_y, color=col, alpha=0.22 if not highlight else 0.35, zorder=3)
        ax.plot(z_front_u, y_front_u, color=col, lw=lw, zorder=4)
        ax.plot(z_front_l, y_front_l, color=col, lw=lw, zorder=4)
        ax.plot(z_rear_u, y_rear_u, color=col, lw=lw, zorder=4)
        ax.plot(z_rear_l, y_rear_l, color=col, lw=lw, zorder=4)
        ax.plot(
            [z_front_u[-1], z_rear_u[-1]],
            [y_front_u[-1], y_rear_u[-1]],
            color=col,
            lw=max(1.0, lw - 0.2),
            zorder=4,
        )
        ax.plot(
            [z_front_l[-1], z_rear_l[-1]],
            [y_front_l[-1], y_rear_l[-1]],
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
        ax = self.ax_side
        ax.clear()
        self._style_axes()
        res = self.result
        if res is None:
            return
        p = self.collect_params()
        target_z = p["target_z"]
        layout = self._element_layout(p)
        self._element_handles = layout

        drag_idx = None
        drag_dz = 0.0
        if self._drag is not None:
            drag_idx = self._drag["elem_index"]
            drag_dz = self._drag["current_front"] - self._drag["orig_front"]

        pairs = self._surface_pairs(res.surfaces)
        mla_mode = any(s.label.startswith("MLA") for s in res.surfaces)

        y_ext = 12.0
        for d in res.dies:
            y_ext = max(y_ext, abs(d.cy) + d.height / 2 + 2)
        for s in res.surfaces:
            y_ext = max(y_ext, abs(s.y0) + s.aperture * 1.2)
        for h in layout:
            y_ext = max(y_ext, h["aperture"] * 1.15)
        z0 = min((d.cz for d in res.dies), default=0) - 5
        z_max_optics = max((s.z_vertex for s in res.surfaces), default=10)
        if self._drag is not None:
            z_max_optics = max(
                z_max_optics, self._drag["current_front"] + self._drag["thickness"]
            )
        z1 = max(target_z, z_max_optics) + 10

        ax.axhline(0, color="#3d5a73", lw=1.2, zorder=1)
        self._side_full_extent = (z0, z1, -y_ext, y_ext)
        # Default full frame; zoom/pan applied after drawing via _apply_side_limits
        ax.set_xlim(z0, z1)
        ax.set_ylim(-y_ext, y_ext)

        # LED / COB dies (side view: Y extent at source Z)
        for die in res.dies:
            y0 = die.cy - die.height / 2
            y1 = die.cy + die.height / 2
            ax.plot([die.cz, die.cz], [y0, y1], color=SOURCE, lw=3, solid_capstyle="round", zorder=5)
            ax.fill(
                [die.cz, die.cz + max(0.4, 0.15 * (z1 - z0) * 0.02), die.cz],
                [y0, die.cy, y1],
                color=SOURCE,
                alpha=0.25,
                zorder=4,
            )

        # Optics bodies: continuous MLA plate section, or discrete singlets
        n_mla = sum(1 for s1, s2, _ in pairs if s1.label.startswith("MLA"))
        mla_drawn = False
        if mla_mode and n_mla > 0:
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
                continue  # already rendered as continuous plate
            z_off = drag_dz if (drag_idx is not None and eidx == drag_idx) else 0.0
            dragging_this = abs(z_off) > 1e-9
            self._draw_lens_body(
                ax,
                s1,
                s2,
                z_off,
                highlight=dragging_this,
                compact=is_mla or n_mla > 4,
            )

        # Drag handles: for MLA, one handle for the whole array (element 0)
        for h in layout:
            zf = h["front_z"]
            zr = h["rear_z"]
            if drag_idx is not None and h["index"] == drag_idx:
                zf = self._drag["current_front"]
                zr = zf + h["thickness"]
            ap = h["aperture"]
            # Span handle over array height when MLA is active
            if mla_mode and h["index"] == 0 and res.dies:
                ys = [d.cy for d in res.dies]
                y_lo = min(ys) - ap
                y_hi = max(ys) + ap
            else:
                y_lo, y_hi = -ap * 0.2, ap * 0.2
            active = drag_idx is not None and h["index"] == drag_idx
            z_handle = 0.5 * (zf + zr)
            ax.plot(
                [z_handle, z_handle],
                [y_lo, y_hi],
                color="#fbbf24" if active else "#94a3b8",
                lw=1.8,
                solid_capstyle="round",
                zorder=6,
                linestyle="--" if active else ":",
                alpha=0.9,
            )
            ax.plot(
                z_handle,
                0 if not mla_mode else 0.5 * (y_lo + y_hi),
                "s",
                color="#fbbf24",
                ms=6,
                zorder=7,
                markeredgecolor="#0b0f14",
            )
            label = h["label"]
            if mla_mode and h["index"] == 0 and "MLA" not in label:
                label = label + " MLA"
            ax.text(
                z_handle,
                (y_hi if mla_mode and h["index"] == 0 else ap) * 1.05 + 0.4,
                label,
                color=FG_BRIGHT,
                fontsize=8,
                ha="center",
                va="bottom",
                zorder=7,
            )

        if self._drag is None:
            for path in res.paths:
                if len(path.history) < 2:
                    continue
                ev = getattr(path, "events", None) or []
                # Rays that never refracted miss the clear aperture / optics —
                # draw them nearly transparent so in-lens paths stay readable.
                through_lens = int(getattr(path, "n_refractions", 0) or 0) > 0 or any(
                    e == "refract" for e in ev
                )
                miss_scale = 1.0 if through_lens else 0.12
                for j in range(len(path.history) - 1):
                    p0, p1 = path.history[j], path.history[j + 1]
                    kind = ev[j] if j < len(ev) else "refract"
                    if kind == "reflect":
                        col, al, lw = "#f97316", 0.85, 1.2
                    elif kind in ("tir_absorb", "kill"):
                        col, al, lw = "#ef4444", 0.65, 1.0
                    elif kind == "ghost":
                        col, al, lw = "#64748b", 0.25, 0.5
                    else:
                        col, al, lw = "#7dd3fc", 0.35, 0.7
                    al = max(0.04, al * miss_scale)
                    if not through_lens:
                        lw = min(lw, 0.55)
                    ax.plot(
                        [p0[2], p1[2]],
                        [p0[1], p1[1]],
                        color=col,
                        alpha=al,
                        lw=lw,
                        zorder=2 if through_lens else 1,
                    )

        ax.axvline(target_z, color=TARGET, ls="--", lw=1.4, zorder=4)
        fy0 = p["fov_cy"] - p["fov_height"] / 2
        fy1 = p["fov_cy"] + p["fov_height"] / 2
        ax.plot([target_z, target_z], [fy0, fy1], color=FOV, lw=3, zorder=5)
        ax.text(target_z, y_ext * 0.92, " TARGET", color=TARGET, fontsize=8, va="top")
        ax.text(target_z, fy1, " FOV", color=FOV, fontsize=8, va="bottom")

        n_lenslets = sum(1 for s in res.surfaces if s.label.endswith("S1") and s.label.startswith("MLA"))
        title = "SIDE VIEW  ·  scroll zoom · right-drag pan · left-drag lens"
        if n_lenslets:
            title = f"SIDE VIEW  ·  MLA {n_lenslets} lenslets  ·  scroll/pan/drag"
        if self._drag is not None:
            title = (
                f"SIDE VIEW  ·  moving {self._drag['label']} → "
                f"Z={self._drag['current_front']:.2f} mm"
            )
        ax.set_title(title, loc="left", fontsize=10, color=FG_BRIGHT)
        ax.set_xlabel("Z (mm)")
        ax.set_ylabel("Y (mm)")
        ax.grid(True, color="#1a2332", lw=0.6)
        self._apply_side_limits(ax)
        self.fig_side.tight_layout()
        self.canvas_side.draw_idle()

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
        """Restore zoom/pan after a full redraw, clamped to data extent."""
        if self._tgt_full_extent is None:
            return
        xmin, xmax, ymin, ymax = self._tgt_full_extent
        if self._tgt_xlim is None or self._tgt_ylim is None:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            return
        x0, x1 = self._tgt_xlim
        y0, y1 = self._tgt_ylim
        # Keep window size; clamp center inside full extent
        w = x1 - x0
        h = y1 - y0
        full_w = xmax - xmin
        full_h = ymax - ymin
        if w > full_w:
            x0, x1 = xmin, xmax
        else:
            if x0 < xmin:
                x1 += xmin - x0
                x0 = xmin
            if x1 > xmax:
                x0 -= x1 - xmax
                x1 = xmax
        if h > full_h:
            y0, y1 = ymin, ymax
        else:
            if y0 < ymin:
                y1 += ymin - y0
                y0 = ymin
            if y1 > ymax:
                y0 -= y1 - ymax
                y1 = ymax
        self._tgt_xlim = (x0, x1)
        self._tgt_ylim = (y0, y1)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

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
        # Minimum zoom window ~ 1% of full map or 0.5 mm
        if self._tgt_full_extent is not None:
            fx0, fx1, fy0, fy1 = self._tgt_full_extent
            min_side = max(0.5, 0.02 * max(fx1 - fx0, fy1 - fy0))
            max_side = max(fx1 - fx0, fy1 - fy0) * 1.05
            side = max(min_side, min(side, max_side))
        else:
            side = max(0.5, side)

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

    def _on_tgt_press(self, event):
        if self.ax_tgt is None or event.inaxes != self.ax_tgt:
            return
        if event.button == 1 and getattr(event, "dblclick", False):
            self._reset_tgt_zoom()
            return
        if event.button in (2, 3) and event.x is not None and event.y is not None:
            self._tgt_pan = {
                "xpress": event.x,
                "ypress": event.y,
                "xlim": self.ax_tgt.get_xlim(),
                "ylim": self.ax_tgt.get_ylim(),
            }
            self.canvas_tgt.get_tk_widget().configure(cursor="fleur")

    def _on_tgt_release(self, event):
        if self._tgt_pan is not None:
            self._tgt_pan = None
            self.canvas_tgt.get_tk_widget().configure(cursor="")

    def _on_tgt_motion(self, event):
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

    def _draw_target(self):
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
        im = ax.imshow(
            data,
            extent=extent,
            origin="upper",
            cmap=IRRAD_CMAP,
            aspect="equal",
            interpolation="bilinear",
        )
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
        cx, cy = res.stats["centroid"]
        ax.plot(cx, cy, "+", color="white", ms=10, mew=1.5)
        ax.axhline(0, color="#64748b", ls=":", lw=0.7, alpha=0.5)
        ax.axvline(0, color="#64748b", ls=":", lw=0.7, alpha=0.5)
        ax.set_title(
            "TARGET PLANE  ·  scroll zoom  ·  right-drag pan",
            loc="left",
            fontsize=10,
            color=FG_BRIGHT,
        )
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        cbar = self.fig_tgt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color=FG, labelsize=7)
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

    def run_optimize(self):
        """
        Background search for rectangular FOV illumination.

        Single-phase: tune current elements for FOV flux (+ uniformity / aspect).
        Two-phase: (1) even light in FOV, (2) add N anamorphic lenses and match
        footprint aspect to the rectangular FOV (not limited to a circular zone).
        """
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
        two_phase = bool(self.v_opt_two_phase.get())
        extra = int(self.v_opt_extra.get())
        cfg = OptimizeConfig(
            rays_per_eval=int(self.v_opt_rays.get()),
            max_evals=int(self.v_opt_evals.get()),
            uniformity_weight=float(self.v_opt_uni_w.get()),
            aspect_weight=float(self.v_opt_aspect_w.get()),
            fill_weight=float(self.v_opt_fill_w.get()),
            coverage_mix=0.75,
            two_phase=two_phase,
            extra_anamorphic_lenses=extra,
            anamorphic_mode=str(self.v_opt_ana_mode.get() or "crossed"),
            polish=bool(self.v_opt_polish.get()),
            optimize_asphere=bool(self.v_opt_asphere.get()),
            optimize_lens_z=True,
            force_cpu=True,
            seed=42,
        )

        if two_phase and extra > 0:
            self.opt_status.set(
                f"Phase 1 → even FOV light; Phase 2 → +{extra} anamorphic lens(es)…"
            )
            self.status_var.set("Two-phase rectangular FOV optimize…")
        else:
            self.opt_status.set("Starting FOV-flux optimizer…")
            self.status_var.set("Optimizing FOV flux…")
        self.progress["value"] = 0

        def progress_cb(frac: float, msg: str, best: float):
            if gen != self._opt_gen:
                return
            self.after(
                0,
                lambda: (
                    self.progress.configure(value=frac * 100),
                    self.opt_status.set(msg),
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
        for i, e in enumerate(p["elements"]):
            if i >= len(self.elem_vars):
                break
            self._apply_element_dict_to_vars(i, e)
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
        self._on_param_change()

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
        if el.get("R1y") is not None and "R1y" in ev:
            ev["R1y"].set(el["R1y"])
        if el.get("R2y") is not None and "R2y" in ev:
            ev["R2y"].set(el["R2y"])
        if el.get("aperture_y") is not None and "aperture_y" in ev:
            ev["aperture_y"].set(el["aperture_y"])
            ev["use_elliptical_ap"].set(True)
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

    def export_cad(self, fmt: str = "stl"):
        params = self.collect_params()
        if not any(e.get("enabled") for e in params["elements"]):
            messagebox.showwarning("Export", "Enable at least one lens element.")
            return
        ext = ".stl" if fmt == "stl" else ".step"
        title = "Export STL (mm)" if fmt == "stl" else "Export STEP (mm)"
        path = filedialog.asksaveasfilename(
            title=title,
            defaultextension=ext,
            filetypes=[(f"{fmt.upper()} file", f"*{ext}"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            dies = self.result.dies if self.result else None
            n_r = int(self.v_mesh_res.get())
            out = export_lens(
                params,
                path,
                fmt=fmt,
                dies=dies,
                n_radial=n_r,
                n_theta=max(36, n_r * 2),
                include_plate=bool(self.v_export_plate.get()),
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
            messagebox.showinfo(
                "Export complete",
                f"Wrote {out}\n\nUnits: millimetres (mm)\n"
                f"Surfaces use the same aspheric sag model as the ray tracer.\n"
                f"{geom_note}",
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
