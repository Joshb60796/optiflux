"""
Optical materials catalog for visible-band design (380–780 nm).

Includes:
  - Common catalog glasses (Schott / Edmund Optics style: N-BK7, N-SF11, …)
  - Plastics used for molded LED optics (acrylic/PMMA, polycarbonate, COC)
  - Formlabs photopolymer resins used for SLA-printed optics prototypes
  - Crystals often stocked by optics suppliers (fused silica, CaF₂, sapphire)

Dispersion models:
  - Sellmeier (preferred for glasses): n² = 1 + Σ Bᵢ λ²/(λ² − Cᵢ), λ in µm
  - Cauchy: n = A + B/λ² + C/λ⁴
  - Constant n (prototype resins with sparse published data)

References: Schott catalog Sellmeier coefficients; refractiveindex.info;
manufacturer data sheets (approximate for resins).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

# CIE-style visible window used by the app
VISIBLE_NM_MIN = 380.0
VISIBLE_NM_MAX = 780.0
VISIBLE_NM_DEFAULT = 550.0  # green, near photopic peak / common LED design λ

# Fraunhofer lines used for validation (nm)
LAMBDA_F = 486.1
LAMBDA_D = 587.6
LAMBDA_C = 656.3
LAMBDA_E = 546.1
LAMBDA_HE_NE = 632.8


def clamp_visible_nm(wavelength_nm: float) -> float:
    return max(VISIBLE_NM_MIN, min(VISIBLE_NM_MAX, float(wavelength_nm)))


# ── Material definitions ─────────────────────────────────────────────────────
# Keys are stable IDs used in params / save files.

MATERIALS: Dict[str, Dict[str, Any]] = {
    "AIR": {
        "name": "Air",
        "type": "const",
        "n": 1.000293,
        "category": "medium",
        "notes": "Standard air at STP (approx.).",
    },
    # ── Catalog glasses (Edmund / Schott style) ─────────────────────────────
    "N_BK7": {
        "name": "N-BK7 (Schott / Edmund)",
        "type": "sellmeier",
        "B": [1.03961212, 0.231792344, 1.01046945],
        "C": [0.00600069867, 0.0200179144, 103.560653],
        "category": "glass",
        # Known n_d ≈ 1.5168
        "n_d_ref": 1.5168,
        "notes": "Workhorse crown glass; visible AR coatings common.",
    },
    "N_SF11": {
        "name": "N-SF11 (dense flint)",
        "type": "sellmeier",
        "B": [1.73759695, 0.313747346, 1.89878101],
        "C": [0.013188707, 0.0623068142, 155.23629],
        "category": "glass",
        "n_d_ref": 1.7847,
        "notes": "High-index flint; Edmund stock singlets / achromats.",
    },
    "N_SF5": {
        "name": "N-SF5 (flint)",
        "type": "sellmeier",
        "B": [1.52481889, 0.187085527, 1.42729015],
        "C": [0.011254756, 0.0588995392, 129.141675],
        "category": "glass",
        "n_d_ref": 1.6727,
    },
    "N_F2": {
        "name": "N-F2 (flint)",
        "type": "sellmeier",
        "B": [1.39757037, 0.159201403, 1.2686543],
        "C": [0.00995906143, 0.0546931752, 119.248346],
        "category": "glass",
        "n_d_ref": 1.6200,
    },
    "N_BAF10": {
        "name": "N-BAF10 (barium dense flint)",
        "type": "sellmeier",
        "B": [1.5851495, 0.143559385, 1.08521269],
        "C": [0.00926681282, 0.0424489805, 105.613573],
        "category": "glass",
        "n_d_ref": 1.6700,
    },
    "N_SK16": {
        "name": "N-SK16 (dense crown)",
        "type": "sellmeier",
        "B": [1.34317774, 0.241144399, 0.994317969],
        "C": [0.00704687339, 0.0229005, 92.7508526],
        "category": "glass",
        "n_d_ref": 1.6204,
    },
    "N_LAK22": {
        "name": "N-LAK22 (lanthanum crown)",
        "type": "sellmeier",
        "B": [1.14229781, 0.535138441, 1.04088385],
        "C": [0.00585778594, 0.01985461423, 100.834826],
        "category": "glass",
        "n_d_ref": 1.6511,
    },
    "FUSED_SILICA": {
        "name": "Fused silica (SiO₂)",
        "type": "sellmeier",
        "B": [0.6961663, 0.4079426, 0.8974794],
        "C": [0.00467914826, 0.0135120631, 97.9340025],
        "category": "glass",
        "n_d_ref": 1.4585,
        "notes": "UV–vis; Edmund UVFS / Corning 7980 style.",
    },
    "CAF2": {
        "name": "Calcium fluoride (CaF₂)",
        "type": "sellmeier",
        # Malitson-style (approx. for visible–NIR)
        "B": [0.5675888, 0.4710914, 3.8484723],
        "C": [0.00252643, 0.010078333, 1200.555973],
        "category": "crystal",
        "n_d_ref": 1.4338,
        "notes": "Low dispersion; IR/UV windows and achromats.",
    },
    "SAPPHIRE": {
        "name": "Sapphire (Al₂O₃, ordinary)",
        "type": "sellmeier",
        "B": [1.4313493, 0.65054713, 5.3414021],
        "C": [0.0052799261, 0.014238265, 325.017834],
        "category": "crystal",
        "n_d_ref": 1.7682,
        "notes": "Hard window / high-power; birefringent (ordinary ray model).",
    },
    # ── Plastics (LED secondary optics) ────────────────────────────────────
    "ACRYLIC_PMMA": {
        "name": "Acrylic / PMMA",
        "type": "cauchy",
        # Fitted so n_d ≈ 1.491 (optical-grade acrylic / Plexiglas)
        "A": 1.481,
        "B": 0.00318,
        "C": 0.0001,
        "category": "plastic",
        "n_d_ref": 1.491,
        "notes": "Standard molded LED TIR / collimator material.",
    },
    "POLYCARBONATE": {
        "name": "Polycarbonate (PC)",
        "type": "cauchy",
        # Fitted so n_d ≈ 1.585 and V_d ≈ 30 (typical optical PC)
        "A": 1.556496,
        "B": 0.009552,
        "C": 0.000100,
        "category": "plastic",
        "n_d_ref": 1.585,
        "notes": "Impact-resistant molded optics; higher n and dispersion than PMMA.",
    },
    "COC_ZEONEX": {
        "name": "COC / COP (Zeonex-class)",
        "type": "cauchy",
        # Fitted so n_d ≈ 1.530 and V_d ≈ 56 (typical optical COP / Zeonex-class)
        "A": 1.516711,
        "B": 0.004299,
        "C": 0.000100,
        "category": "plastic",
        "n_d_ref": 1.53,
        "notes": "Low birefringence precision plastic optics.",
    },
    "STYRENE": {
        "name": "Polystyrene (PS)",
        "type": "cauchy",
        # Fitted so n_d ≈ 1.590 and V_d ≈ 31
        "A": 1.562205,
        "B": 0.009307,
        "C": 0.000100,
        "category": "plastic",
        "n_d_ref": 1.590,
    },
    # ── Formlabs photopolymers (SLA prototypes) ────────────────────────────
    # Published full Sellmeier data is sparse; use constant / mild Cauchy
    # anchored near manufacturer-reported visible n ≈ 1.53–1.55 for Clear.
    "FORMLABS_CLEAR": {
        "name": "Formlabs Clear Resin",
        "type": "cauchy",
        # Mild dispersion around n_d ≈ 1.54 (approx. SLA resin data)
        "A": 1.524686,
        "B": 0.004998,
        "C": 0.000100,
        "category": "photopolymer",
        "n_d_ref": 1.54,
        "notes": "SLA prototype optics; n≈1.53–1.55 (approx.). Polish/coat as needed.",
    },
    "FORMLABS_TOUGH": {
        "name": "Formlabs Tough / Durable (approx.)",
        "type": "const",
        "n": 1.54,
        "category": "photopolymer",
        "n_d_ref": 1.54,
        "notes": "Approximate visible n; not a precision optical grade.",
    },
    "FORMLABS_HIGH_TEMP": {
        "name": "Formlabs High Temp (approx.)",
        "type": "const",
        "n": 1.55,
        "category": "photopolymer",
        "n_d_ref": 1.55,
        "notes": "Approximate; use for mechanical mock-ups more than precision EFL.",
    },
    "FORMLABS_RIGID": {
        "name": "Formlabs Rigid (approx.)",
        "type": "const",
        "n": 1.54,
        "category": "photopolymer",
        "n_d_ref": 1.54,
    },
    "CUSTOM": {
        "name": "Custom constant n",
        "type": "const",
        "n": 1.5,
        "category": "custom",
        "notes": "Override with the Custom n slider.",
    },
}

# Backward-compatible aliases used in older params / presets / UI free-text
_ALIASES = {
    "BK7": "N_BK7",
    "SF11": "N_SF11",
    "PMMA": "ACRYLIC_PMMA",
    "ACRYLIC": "ACRYLIC_PMMA",
    # Common short / alternate labels → canonical Formlabs Clear
    "CLEAR RESIN": "FORMLABS_CLEAR",
    "CLEAR": "FORMLABS_CLEAR",
    "FORMLABS CLEAR": "FORMLABS_CLEAR",
    "FORMLABS CLEAR RESIN": "FORMLABS_CLEAR",
    "FORMLABS_CLEAR_RESIN": "FORMLABS_CLEAR",
}


def resolve_material_id(mat_id: str) -> str:
    if not mat_id:
        return "N_BK7"
    s = str(mat_id).strip()
    if s in MATERIALS:
        return s
    # Case-insensitive alias / id match
    up = s.upper().replace("-", " ").replace("_", " ")
    up_key = s.upper().replace(" ", "_").replace("-", "_")
    if up_key in MATERIALS:
        return up_key
    if up_key in _ALIASES:
        return _ALIASES[up_key]
    if up in _ALIASES:
        return _ALIASES[up]
    # Compact spaces for alias table
    compact = " ".join(up.split())
    if compact in _ALIASES:
        return _ALIASES[compact]
    return _ALIASES.get(s, s)


def material_display_names() -> List[str]:
    """Ordered labels for UI dropdowns (exactly one label per material)."""
    order = [
        "AIR",
        "N_BK7",
        "N_SF5",
        "N_SF11",
        "N_F2",
        "N_BAF10",
        "N_SK16",
        "N_LAK22",
        "FUSED_SILICA",
        "CAF2",
        "SAPPHIRE",
        "ACRYLIC_PMMA",
        "POLYCARBONATE",
        "COC_ZEONEX",
        "STYRENE",
        "FORMLABS_CLEAR",
        "FORMLABS_TOUGH",
        "FORMLABS_HIGH_TEMP",
        "FORMLABS_RIGID",
        "CUSTOM",
    ]
    names = [MATERIALS[k]["name"] for k in order if k in MATERIALS]
    # Deduplicate while preserving order (guards against catalog typos)
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def material_id_from_name(name: str) -> str:
    """Map a UI label or raw id to a canonical MATERIALS key."""
    if not name:
        return "N_BK7"
    s = str(name).strip()
    # Exact display-name match
    for k, m in MATERIALS.items():
        if m["name"] == s:
            return k
    # Case-insensitive display-name match
    low = s.casefold()
    for k, m in MATERIALS.items():
        if str(m["name"]).casefold() == low:
            return k
    # Partial match for common short forms ("Clear Resin" → Formlabs Clear Resin)
    for k, m in MATERIALS.items():
        nm = str(m["name"]).casefold()
        if low in nm or nm in low:
            if "formlabs" in nm or "clear" in low:
                return k
    # Raw id / alias
    rid = resolve_material_id(s)
    if rid in MATERIALS:
        return rid
    return "N_BK7"


def material_name_from_id(mat_id: str) -> str:
    """Always return the canonical catalog display name (never a raw id)."""
    mid = material_id_from_name(str(mat_id))
    m = MATERIALS.get(mid)
    return m["name"] if m else MATERIALS["N_BK7"]["name"]


def material_ids() -> List[str]:
    return list(MATERIALS.keys())


def refractive_index(
    mat_id: str,
    wavelength_nm: float,
    custom_n: float = 1.5,
    *,
    clamp_visible: bool = True,
) -> float:
    """
    Refractive index at wavelength_nm.
    By default clamps λ into the visible design window.
    """
    mid = resolve_material_id(mat_id)
    m = MATERIALS.get(mid, MATERIALS["AIR"])
    wl = clamp_visible_nm(wavelength_nm) if clamp_visible else float(wavelength_nm)

    if m["type"] == "const":
        if mid == "CUSTOM":
            return float(custom_n)
        return float(m["n"])

    lam = wl / 1000.0  # µm
    lam2 = lam * lam
    if m["type"] == "sellmeier":
        n2 = 1.0
        for b, c in zip(m["B"], m["C"]):
            denom = lam2 - c
            if abs(denom) < 1e-18:
                return float("nan")
            n2 += (b * lam2) / denom
        return math.sqrt(max(1.0, n2))

    if m["type"] == "cauchy":
        return float(m["A"] + m["B"] / lam2 + m["C"] / (lam2 * lam2))

    return 1.5


def abbe_number(mat_id: str, custom_n: float = 1.5) -> Optional[float]:
    """V_d = (n_d − 1)/(n_F − n_C) using model n(λ)."""
    nd = refractive_index(mat_id, LAMBDA_D, custom_n)
    nF = refractive_index(mat_id, LAMBDA_F, custom_n)
    nC = refractive_index(mat_id, LAMBDA_C, custom_n)
    denom = nF - nC
    if abs(denom) < 1e-12:
        return None
    return (nd - 1.0) / denom


def materials_by_category() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for k, m in MATERIALS.items():
        cat = m.get("category", "other")
        out.setdefault(cat, []).append(k)
    return out
