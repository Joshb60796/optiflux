"""
Save / load OptiFlux design files (.json).

A design captures the full parameter dict from collect_params() so lens stacks,
source, FOV, MLA, and sim settings can be restored later.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from engine import MAX_ELEMENTS, default_params, pad_elements

DESIGN_FORMAT = "optiflux_design"
DESIGN_VERSION = 1


def design_document(
    params: Dict[str, Any],
    *,
    name: str = "",
    notes: str = "",
) -> Dict[str, Any]:
    """Wrap a params dict in a versioned design document."""
    return {
        "format": DESIGN_FORMAT,
        "version": DESIGN_VERSION,
        "name": str(name or ""),
        "notes": str(notes or ""),
        "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "params": _sanitize_params(params),
    }


def _sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-copy params into JSON-friendly plain types; pad elements to MAX."""
    p = default_params()
    if not isinstance(params, dict):
        return p
    # Shallow-merge known top-level keys from default, then overlay user values
    for k, v in params.items():
        if k in ("elements", "blockers"):
            continue
        p[k] = _jsonable(v)
    elems = params.get("elements") or []
    if not isinstance(elems, list):
        elems = []
    cleaned = []
    for e in elems:
        if not isinstance(e, dict):
            continue
        cleaned.append({str(kk): _jsonable(vv) for kk, vv in e.items()})
    p["elements"] = pad_elements(cleaned, MAX_ELEMENTS)
    # Absorbing panels / aperture stops
    blockers_in = params.get("blockers")
    if not isinstance(blockers_in, list):
        blockers_in = []
    blks = []
    for b in blockers_in:
        if not isinstance(b, dict):
            continue
        blks.append({str(kk): _jsonable(vv) for kk, vv in b.items()})
    p["blockers"] = blks
    return p


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    # numpy scalars etc.
    try:
        import numpy as np

        if isinstance(v, np.generic):
            return v.item()
    except Exception:
        pass
    return str(v)


def save_design(
    path: Path | str,
    params: Dict[str, Any],
    *,
    name: str = "",
    notes: str = "",
) -> Path:
    """Write a design file. Returns the resolved path."""
    path = Path(path)
    if path.suffix.lower() not in (".json", ".optiflux"):
        path = path.with_suffix(".json")
    doc = design_document(params, name=name or path.stem, notes=notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def load_design(path: Path | str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load a design file.

    Returns (params, meta) where meta has format/version/name/notes/saved_at.
    Raises ValueError on bad format.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Design file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Design file must be a JSON object")

    # Accept both wrapped documents and bare params dicts (legacy / hand-edited)
    if raw.get("format") == DESIGN_FORMAT or "params" in raw:
        params_in = raw.get("params")
        if not isinstance(params_in, dict):
            raise ValueError("Design document missing 'params' object")
        meta = {
            "format": raw.get("format", DESIGN_FORMAT),
            "version": int(raw.get("version", 1)),
            "name": str(raw.get("name") or path.stem),
            "notes": str(raw.get("notes") or ""),
            "saved_at": str(raw.get("saved_at") or ""),
            "path": str(path),
        }
    elif "source" in raw and "elements" in raw:
        params_in = raw
        meta = {
            "format": "bare_params",
            "version": 0,
            "name": path.stem,
            "notes": "",
            "saved_at": "",
            "path": str(path),
        }
    else:
        raise ValueError(
            "Unrecognized design file — expected OptiFlux design JSON "
            f"(format={DESIGN_FORMAT!r}) or a bare params object"
        )

    params = _sanitize_params(params_in)
    return params, meta


def default_designs_dir() -> Path:
    """Preferred folder for user designs (next to the app if writable)."""
    here = Path(__file__).resolve().parent
    d = here / "designs"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        d = Path.home() / "OptiFlux_designs"
        d.mkdir(parents=True, exist_ok=True)
    return d
