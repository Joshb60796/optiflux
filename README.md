# OptiFlux — LED / COB Lens Designer

Physics-based Monte Carlo ray tracer for designing lenses that collect light from **single LEDs** or **COB arrays** and deliver a controlled spot / uniform irradiance field (e.g. for a camera FOV).

**Python desktop app only** (tkinter + matplotlib). Optional **NVIDIA Warp** GPU acceleration for the Monte-Carlo irradiance map.

## Quick start

```bash
pip install matplotlib numpy scipy
# Optional (highly recommended for ≥10k rays):
pip install warp-lang
python app.py
```

Or double-click **`start.bat`**.

In PyCharm: right-click `app.py` → **Run**.

## Features

### Sources
- Rectangular surface emitters (not point sources)
- Single LED or COB grid (pitch, stagger, circular mask, tilt)

### Optics
- Up to 3 elements; **per-element** lens type library (PCX, bi-convex, meniscus, …)
- Spherical / conic / A4, biconic and cylindrical (anamorphic) modes
- Materials: N-BK7, flints, fused silica, acrylic, Formlabs resins, etc. (visible 380–780 nm)
- Snell, Fresnel T, TIR absorb (default), positive edge-thickness clamping

### Target & metrics
- Rectangular FOV, irradiance map, RMS / encircled energy, FOV uniformity, aspect match

### Views
- Side view (drag lenses on Z, release to re-trace)
- Target-plane irradiance heatmap

### CAD
- Export **STL** / **STEP** in **mm**

### Rectangular FOV design
- Crossed cylinders or biconic singlet helpers for camera-like fields

### Optimizer (rectangular FOV flux)
- Derivative-free search (SciPy differential evolution + optional Nelder–Mead polish)
- **Two-phase rectangular mode** (default in GUI):
  1. Even light in the FOV (flux + uniformity; circular footprint OK)
  2. Add 1–2 anamorphic elements (crossed cylinders or biconic) and match footprint aspect to FOV W/H — not limited to a circular zone
- Free variables: radii (incl. Ry), thickness, air gaps, apertures, first-vertex Z; optional conic k & A4
- Objective: FOV power/source × (1 + w_u·uniformity) / (1 + w_a·aspect_error)
- GUI: **Optimize FOV** / OPTIMIZER panel · CLI: `python optimizer.py --two-phase --extra-lenses 2 --anamorphic crossed`

### Performance (NVIDIA Warp)
- When `warp-lang` is installed and a CUDA GPU is present, the bulk Monte-Carlo deposit runs as a parallel `@wp.kernel`.
- Side-view ray paths and TIR accounting stay on the pure-Python tracer for exact event history.
- Automatic fallback to pure Python (or Warp CPU) if CUDA / Warp is unavailable.
- Typical speed-up: 10–50× on a modern RTX / Quadro for 30k–100k rays.
- Control via params: `"use_warp": True` (default). Set `False` to force CPU.

## Validation

```bash
python validate_physics.py
```

## Project layout

```
app.py                 Desktop GUI  ← run this
engine.py              Ray tracer + surfaces (CPU + Warp dispatch)
optimizer.py           FOV-flux parameter optimizer (SciPy DE)
warp_backend.py        NVIDIA Warp Monte-Carlo kernel
materials_catalog.py   Visible-band materials
lens_shapes.py         Per-element form catalog
rect_fov.py            Rectangular FOV design helpers
export_cad.py          STL / STEP export
mla_geometry.py        Shared MLA geometry for tracer + CAD
tests/                 Physics & path-integrity tests
validate_physics.py    Test runner
start.bat              Launch GUI
requirements.txt
```

## License

Use freely for design and education.
