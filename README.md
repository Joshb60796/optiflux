# OptiFlux — LED / COB Lens Designer

Physics-based Monte Carlo ray tracer for designing lenses that collect light from **single LEDs** or **COB arrays** and deliver a controlled spot / uniform irradiance field (e.g. for a camera FOV).

**Python desktop app only** (tkinter + matplotlib).

## Quick start

```bash
pip install matplotlib numpy
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

## Validation

```bash
python validate_physics.py
```

## Project layout

```
app.py                 Desktop GUI  ← run this
engine.py              Ray tracer + surfaces
materials_catalog.py   Visible-band materials
lens_shapes.py         Per-element form catalog
rect_fov.py            Rectangular FOV design helpers
export_cad.py          STL / STEP export
tests/                 Physics & path-integrity tests
validate_physics.py    Test runner
start.bat              Launch GUI
requirements.txt
```

## License

Use freely for design and education.
