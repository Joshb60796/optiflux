"""
Physics engine for OptiFlux desktop GUI.
Extended rectangular LED/COB emitters, aspheric surfaces, Snell + Fresnel, MC irradiance.
Units: millimeters throughout. Optical axis = +Z.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

Vec3 = Tuple[float, float, float]


def v_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def v_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_len(a: Vec3) -> float:
    return math.hypot(a[0], a[1], a[2])


def v_norm(a: Vec3) -> Vec3:
    L = v_len(a)
    if L < 1e-15:
        return (0.0, 0.0, 0.0)
    return (a[0] / L, a[1] / L, a[2] / L)


# ── Materials (visible-band catalog) ─────────────────────────────────────────
from materials_catalog import (  # noqa: E402
    MATERIALS,
    VISIBLE_NM_DEFAULT,
    VISIBLE_NM_MAX,
    VISIBLE_NM_MIN,
    abbe_number,
    clamp_visible_nm,
    material_display_names,
    material_id_from_name,
    material_ids,
    material_name_from_id,
    refractive_index,
    resolve_material_id,
)

# UI list: human-readable names (Edmund-style + acrylic + Formlabs)
MATERIAL_NAMES = material_display_names()


def fresnel_T(n1: float, n2: float, cos_i: float) -> Tuple[float, bool]:
    """Return (power transmittance, tir)."""
    cos_i = min(1.0, max(0.0, abs(cos_i)))
    eta = n1 / n2
    sin_t2 = eta * eta * (1.0 - cos_i * cos_i)
    if sin_t2 > 1.0:
        return 0.0, True
    cos_t = math.sqrt(1.0 - sin_t2)
    rs = ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)) ** 2
    rp = ((n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)) ** 2
    R = 0.5 * (rs + rp)
    return max(0.0, 1.0 - R), False


def snell_refract(I: Vec3, N: Vec3, n1: float, n2: float) -> Tuple[Vec3, bool]:
    cosi = -v_dot(I, N)
    Nx, Ny, Nz = N
    if cosi < 0:
        Nx, Ny, Nz = -Nx, -Ny, -Nz
        cosi = -v_dot(I, (Nx, Ny, Nz))
    cosi = min(1.0, max(0.0, cosi))
    eta = n1 / n2
    k = 1.0 - eta * eta * (1.0 - cosi * cosi)
    if k < 0:
        # TIR → reflect
        return v_norm(v_add(I, v_scale((Nx, Ny, Nz), 2.0 * cosi))), True
    cost = math.sqrt(k)
    T = v_add(v_scale(I, eta), v_scale((Nx, Ny, Nz), eta * cosi - cost))
    return v_norm(T), False


def lensmaker_f(R1: float, R2: float, n: float, thickness: float) -> float:
    c1 = 0.0 if abs(R1) < 1e-12 else 1.0 / R1
    c2 = 0.0 if abs(R2) < 1e-12 else 1.0 / R2
    invf = (n - 1.0) * (c1 - c2 + (n - 1.0) * thickness * c1 * c2 / n)
    if abs(invf) < 1e-15:
        return float("inf")
    return 1.0 / invf


# ── Sampling ─────────────────────────────────────────────────────────────────

def sample_rect(cx: float, cy: float, cz: float, w: float, h: float, rot_z_deg: float = 0.0) -> Vec3:
    u = (random.random() - 0.5) * w
    v = (random.random() - 0.5) * h
    th = math.radians(rot_z_deg)
    c, s = math.cos(th), math.sin(th)
    return (cx + u * c - v * s, cy + u * s + v * c, cz)


def sample_lambertian_cone(half_angle_deg: float) -> Vec3:
    alpha = math.radians(max(0.1, min(90.0, half_angle_deg)))
    if alpha >= math.pi / 2 - 1e-6:
        xi1, xi2 = random.random(), random.random()
        cos_t = math.sqrt(xi1)
        sin_t = math.sqrt(max(0.0, 1.0 - cos_t * cos_t))
        phi = 2.0 * math.pi * xi2
        return (sin_t * math.cos(phi), sin_t * math.sin(phi), cos_t)
    cos_a = math.cos(alpha)
    cos_a2 = cos_a * cos_a
    xi1 = random.random()
    cos_t2 = 1.0 - xi1 * (1.0 - cos_a2)
    cos_t = math.sqrt(max(0.0, cos_t2))
    sin_t = math.sqrt(max(0.0, 1.0 - cos_t2))
    phi = 2.0 * math.pi * random.random()
    return (sin_t * math.cos(phi), sin_t * math.sin(phi), cos_t)


def apply_tilt(d: Vec3, tilt_x_deg: float, tilt_y_deg: float) -> Vec3:
    ax, ay = math.radians(tilt_x_deg), math.radians(tilt_y_deg)
    x, y, z = d
    y1 = y * math.cos(ax) - z * math.sin(ax)
    z1 = y * math.sin(ax) + z * math.cos(ax)
    y, z = y1, z1
    x2 = x * math.cos(ay) + z * math.sin(ay)
    z2 = -x * math.sin(ay) + z * math.cos(ay)
    return v_norm((x2, y, z2))


# ── Surfaces ─────────────────────────────────────────────────────────────────

def _curv(R: float) -> float:
    if not math.isfinite(R) or abs(R) < 1e-12:
        return 0.0
    return 1.0 / R


@dataclass
class OpticalSurface:
    """
    Optical surface. Modes:
      rotational — classic asphere of radial r (radius = Rx = Ry)
      biconic    — independent Rx, Ry (anamorphic; rectangular FOV shaping)
      cylinder_x — power only in X (Ry = ∞)
      cylinder_y — power only in Y (Rx = ∞)

    Clear aperture: circular (default) or elliptical (aperture_x, aperture_y).
    """
    z_vertex: float
    radius: float  # Rx (mm); 0 = plano in X
    k: float = 0.0
    aperture: float = 10.0  # circular semi-diameter if aperture_y is None
    material_after: str = "AIR"   # medium on the +Z side of the surface
    material_before: str = "AIR"  # medium on the −Z side of the surface
    a4: float = 0.0
    label: str = ""
    active: bool = True
    x0: float = 0.0
    y0: float = 0.0
    # Anamorphic / biconic
    radius_y: Optional[float] = None  # Ry; None → same as radius (rotational)
    k_y: float = 0.0
    a4_y: float = 0.0
    mode: str = "rotational"  # rotational | biconic | cylinder_x | cylinder_y
    aperture_y: Optional[float] = None  # if set with aperture as ap_x → ellipse

    def curvature(self) -> float:
        return _curv(self.radius)

    def curvature_y(self) -> float:
        if self.mode == "cylinder_x":
            return 0.0
        if self.mode == "cylinder_y":
            return _curv(self.radius_y if self.radius_y is not None else self.radius)
        if self.radius_y is None and self.mode == "rotational":
            return self.curvature()
        return _curv(self.radius_y if self.radius_y is not None else self.radius)

    def is_anamorphic(self) -> bool:
        if self.mode in ("biconic", "cylinder_x", "cylinder_y"):
            return True
        if self.radius_y is None:
            return False
        return abs(self.radius - self.radius_y) > 1e-9 or abs(self.k - self.k_y) > 1e-12

    def sag(self, r: float) -> Optional[float]:
        """Rotationally symmetric sag (for profiles / legacy)."""
        return self.sag_xy(r, 0.0) if abs(r) < 1e-15 else self.sag_xy(r, 0.0)

    def sag_xy(self, x: float, y: float) -> Optional[float]:
        """
        Surface sag relative to vertex.
        Biconic:
          z = (cx x² + cy y²) / (1 + sqrt(1 − (1+kx) cx² x² − (1+ky) cy² y²))
            + A4x x⁴ + A4y y⁴
        """
        mode = self.mode
        if mode == "cylinder_x":
            cx, cy = _curv(self.radius), 0.0
            kx, ky = self.k, 0.0
            a4x, a4y = self.a4, 0.0
        elif mode == "cylinder_y":
            Ry = self.radius_y if self.radius_y is not None else self.radius
            cx, cy = 0.0, _curv(Ry)
            kx, ky = 0.0, self.k_y
            a4x, a4y = 0.0, self.a4_y if abs(self.a4_y) > 0 else self.a4
        elif mode == "biconic" or self.is_anamorphic():
            cx = _curv(self.radius)
            Ry = self.radius_y if self.radius_y is not None else self.radius
            cy = _curv(Ry)
            kx, ky = self.k, self.k_y
            a4x, a4y = self.a4, (self.a4_y if self.radius_y is not None else self.a4)
        else:
            # rotational
            c = _curv(self.radius)
            r2 = x * x + y * y
            z = 0.0
            if abs(c) > 1e-14:
                disc = 1.0 - (1.0 + self.k) * c * c * r2
                if disc < 0:
                    return None
                z = (c * r2) / (1.0 + math.sqrt(max(0.0, disc)))
            z += self.a4 * r2 * r2
            return z

        x2, y2 = x * x, y * y
        disc = 1.0 - (1.0 + kx) * cx * cx * x2 - (1.0 + ky) * cy * cy * y2
        if disc < 0:
            return None
        num = cx * x2 + cy * y2
        z = num / (1.0 + math.sqrt(max(0.0, disc))) if abs(num) > 1e-18 or abs(cx) + abs(cy) > 0 else 0.0
        z += a4x * x2 * x2 + a4y * y2 * y2
        return z

    def local_xy(self, x: float, y: float) -> Tuple[float, float]:
        return x - self.x0, y - self.y0

    def in_aperture(self, lx: float, ly: float) -> bool:
        if self.aperture_y is not None and self.aperture_y > 0:
            ax = max(self.aperture, 1e-9)
            ay = max(self.aperture_y, 1e-9)
            return (lx / ax) ** 2 + (ly / ay) ** 2 <= 1.0 + 1e-9
        return math.hypot(lx, ly) <= self.aperture + 1e-6

    def surface_z(self, x: float, y: float) -> Optional[float]:
        lx, ly = self.local_xy(x, y)
        s = self.sag_xy(lx, ly)
        if s is None:
            return None
        return self.z_vertex + s

    def normal_at(self, x: float, y: float) -> Optional[Vec3]:
        eps = 1e-5
        zc = self.surface_z(x, y)
        if zc is None:
            return None
        zp = self.surface_z(x + eps, y)
        zm = self.surface_z(x - eps, y)
        yp = self.surface_z(x, y + eps)
        ym = self.surface_z(x, y - eps)
        if None in (zp, zm, yp, ym):
            return (0.0, 0.0, 1.0)
        dzdx = (zp - zm) / (2 * eps)
        dzdy = (yp - ym) / (2 * eps)
        return v_norm((-dzdx, -dzdy, 1.0))

    def intersect(self, o: Vec3, d: Vec3, t_min: float = 1e-6, t_max: float = 1e6):
        if not self.active:
            return None
        t = (self.z_vertex - o[2]) / d[2] if abs(d[2]) > 1e-12 else t_min + 0.1
        t = max(t, t_min)
        for _ in range(30):
            p = v_add(o, v_scale(d, t))
            zs = self.surface_z(p[0], p[1])
            if zs is None:
                return None
            f = p[2] - zs
            if abs(f) < 1e-7:
                lx, ly = self.local_xy(p[0], p[1])
                if not self.in_aperture(lx, ly):
                    return None
                if t < t_min or t > t_max:
                    return None
                n = self.normal_at(p[0], p[1])
                if n is None:
                    return None
                return t, p, n
            n_approx = self.normal_at(p[0], p[1])
            if n_approx is None:
                return None
            if abs(n_approx[2]) > 1e-8:
                dzdx = -n_approx[0] / n_approx[2]
                dzdy = -n_approx[1] / n_approx[2]
                dfdt = d[2] - (dzdx * d[0] + dzdy * d[1])
            else:
                dfdt = d[2]
            if abs(dfdt) < 1e-14:
                return None
            t = t - f / dfdt
            if t < t_min * 0.1 or t > t_max:
                return None
        return None

    def profile(self, n_pts: int = 40, along_y: bool = True) -> List[Tuple[float, float, float]]:
        """Meridional profile (z, x, y) along +local Y or +local X."""
        pts = []
        ap = self.aperture_y if (along_y and self.aperture_y) else self.aperture
        ap = max(ap, self.aperture)
        for i in range(n_pts + 1):
            r = (i / n_pts) * ap
            if along_y:
                s = self.sag_xy(0.0, r)
            else:
                s = self.sag_xy(r, 0.0)
            if s is None:
                break
            z = self.z_vertex + s
            if along_y:
                pts.append((z, self.x0, self.y0 + r))
            else:
                pts.append((z, self.x0 + r, self.y0))
        return pts


# ── Sources ──────────────────────────────────────────────────────────────────

@dataclass
class EmitterDie:
    cx: float
    cy: float
    cz: float
    width: float
    height: float
    flux: float = 1.0
    wavelength_nm: float = VISIBLE_NM_DEFAULT
    half_angle_deg: float = 60.0
    tilt_x_deg: float = 0.0
    tilt_y_deg: float = 0.0
    rot_z_deg: float = 0.0
    enabled: bool = True

    def spawn_ray(self, power: float) -> Tuple[Vec3, Vec3, float, float]:
        """Returns origin, direction, power, wavelength_nm."""
        o = sample_rect(self.cx, self.cy, self.cz, self.width, self.height, self.rot_z_deg)
        d = sample_lambertian_cone(self.half_angle_deg)
        d = apply_tilt(d, self.tilt_x_deg, self.tilt_y_deg)
        if d[2] < 0:
            d = v_norm((d[0], d[1], -d[2]))
        return o, d, power, self.wavelength_nm

    def corners(self) -> List[Vec3]:
        hw, hh = self.width / 2, self.height / 2
        th = math.radians(self.rot_z_deg)
        c, s = math.cos(th), math.sin(th)
        out = []
        for u, v in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
            out.append((self.cx + u * c - v * s, self.cy + u * s + v * c, self.cz))
        return out


def build_source_array(cfg: Dict[str, Any]) -> List[EmitterDie]:
    mode = cfg.get("mode", "cob")
    rows = 1 if mode == "single" else max(1, int(cfg.get("rows", 4)))
    cols = 1 if mode == "single" else max(1, int(cfg.get("cols", 4)))
    pitch_x = float(cfg.get("pitch_x", 1.6))
    pitch_y = float(cfg.get("pitch_y", 1.6))
    die_w = float(cfg.get("die_width", 1.0))
    die_h = float(cfg.get("die_height", 1.0))
    z = float(cfg.get("source_z", 0.0))
    ox = float(cfg.get("offset_x", 0.0))
    oy = float(cfg.get("offset_y", 0.0))
    x0 = -((cols - 1) * pitch_x) / 2 + ox
    y0 = -((rows - 1) * pitch_y) / 2 + oy
    dies: List[EmitterDie] = []
    for r in range(rows):
        stagger = pitch_x * 0.5 if cfg.get("stagger") and (r % 2 == 1) else 0.0
        for c in range(cols):
            cx = x0 + c * pitch_x + stagger
            cy = y0 + r * pitch_y
            if cfg.get("circular_mask") and cfg.get("mask_radius", 0) > 0:
                if math.hypot(cx - ox, cy - oy) > float(cfg["mask_radius"]):
                    continue
            dies.append(
                EmitterDie(
                    cx=cx,
                    cy=cy,
                    cz=z,
                    width=die_w,
                    height=die_h,
                    flux=float(cfg.get("flux_per_die", 1.0)),
                    wavelength_nm=clamp_visible_nm(float(cfg.get("wavelength_nm", VISIBLE_NM_DEFAULT))),
                    half_angle_deg=float(cfg.get("half_angle_deg", 60.0)),
                    tilt_x_deg=float(cfg.get("tilt_x", 0.0)),
                    tilt_y_deg=float(cfg.get("tilt_y", 0.0)),
                    rot_z_deg=float(cfg.get("die_rot_z", 0.0)),
                )
            )
    return dies


def build_surfaces(
    elements: List[Dict[str, Any]],
    z_start: float,
    mla: Optional[Dict[str, Any]] = None,
    dies: Optional[List[EmitterDie]] = None,
) -> List[OpticalSurface]:
    """
    Build optical surfaces. If mla['enabled'], element 0 is replicated as a
    micro-lens array centered on each COB die (or custom grid).
    """
    surfaces: List[OpticalSurface] = []
    mla = mla or {}
    mla_on = bool(mla.get("enabled", False))

    if mla_on and dies:
        e0 = next((e for e in elements if e.get("enabled", True)), None)
        if e0 is None:
            return surfaces
        # Shared geometry with CAD: Element 1 form scaled to die pitch
        from mla_geometry import (
            lenslet_semi_aperture,
            scale_element_to_lenslet,
        )

        ap = lenslet_semi_aperture(mla, dies)
        scale_geo = bool(mla.get("scale_to_pitch", True))
        g = scale_element_to_lenslet(e0, ap, scale_geometry=scale_geo)
        # Build a temporary element dict with scaled R/thickness for each lenslet
        e_lens = dict(e0)
        e_lens.update(
            {
                "R1": g["R1"],
                "R2": g["R2"],
                "R1y": g["R1y"],
                "R2y": g["R2y"],
                "thickness": g["thickness"],
                "aperture": g["aperture"],
                "k1": g["k1"],
                "k2": g["k2"],
                "A4_1": g["A4_1"],
                "A4_2": g["A4_2"],
                "surface_mode": g["mode"],
                "mode_s1": g["mode"],
                "mode_s2": g["mode"],
            }
        )
        mat = material_id_from_name(str(e0.get("material", "ACRYLIC_PMMA")))
        z1 = z_start
        z2 = z_start + float(g["thickness"])
        ap_use = float(g["aperture"])
        for li, die in enumerate(dies):
            s1 = _surface_from_element(
                e_lens, side=1, z=z1, ap=ap_use, glass=mat, label=f"MLA{li}S1", x0=die.cx, y0=die.cy
            )
            s2 = _surface_from_element(
                e_lens, side=2, z=z2, ap=ap_use, glass=mat, label=f"MLA{li}S2", x0=die.cx, y0=die.cy
            )
            _clamp_pair_aperture(s1, s2, ap_use, min_edge=0.08)
            surfaces.extend([s1, s2])
        # Optional additional on-axis stack elements after MLA air gap
        z = z2 + float(e0.get("air_after", 1.0))
        for i, e in enumerate(elements[1:], start=1):
            if not e.get("enabled", True):
                continue
            ap2 = float(e.get("aperture", 12.0))
            mat2 = material_id_from_name(str(e.get("material", "N_BK7")))
            s1 = _surface_from_element(e, side=1, z=z, ap=ap2, glass=mat2, label=f"E{i+1}S1")
            z += float(e.get("thickness", 3.0))
            s2 = _surface_from_element(e, side=2, z=z, ap=ap2, glass=mat2, label=f"E{i+1}S2")
            _clamp_pair_aperture(s1, s2, ap2)
            surfaces.extend([s1, s2])
            z += float(e.get("air_after", 2.0))
        return surfaces

    # Conventional centered stack
    z = z_start
    for i, e in enumerate(elements):
        if not e.get("enabled", True):
            continue
        ap = float(e.get("aperture", 12.0))
        mat = material_id_from_name(str(e.get("material", "N_BK7")))
        s1 = _surface_from_element(e, side=1, z=z, ap=ap, glass=mat, label=f"E{i+1}S1")
        z += float(e.get("thickness", 3.0))
        s2 = _surface_from_element(e, side=2, z=z, ap=ap, glass=mat, label=f"E{i+1}S2")
        _clamp_pair_aperture(s1, s2, ap)
        surfaces.extend([s1, s2])
        z += float(e.get("air_after", 2.0))
    return surfaces


def _clamp_pair_aperture(
    s1: OpticalSurface,
    s2: OpticalSurface,
    ap_request: float,
    min_edge: float = 0.3,
) -> None:
    """Shrink clear aperture so front/rear never cross (positive edge thickness)."""
    ap = max_aperture_positive_edge(s1, s2, ap_request, min_edge=min_edge)
    s1.aperture = ap
    s2.aperture = ap
    if s1.aperture_y is not None:
        aspect = s1.aperture_y / max(ap_request, 1e-9)
        s1.aperture_y = ap * aspect
        s2.aperture_y = ap * aspect


def lens_edge_thickness(
    s_front: "OpticalSurface",
    s_rear: "OpticalSurface",
    r: float,
) -> Optional[float]:
    """Glass thickness along +Z at radial height r (local to both surfaces)."""
    z1 = s_front.surface_z(s_front.x0, s_front.y0 + r)
    z2 = s_rear.surface_z(s_rear.x0, s_rear.y0 + r)
    if z1 is None or z2 is None:
        return None
    return z2 - z1


def max_aperture_positive_edge(
    s_front: "OpticalSurface",
    s_rear: "OpticalSurface",
    ap_request: float,
    min_edge: float = 0.25,
    samples: int = 48,
) -> float:
    """
    Largest semi-aperture ≤ ap_request with edge thickness ≥ min_edge (mm).
    Prevents self-intersecting lens drawings / unphysical rims.
    """
    ap_request = max(float(ap_request), 1e-3)
    best = 0.0
    for i in range(1, samples + 1):
        r = ap_request * i / samples
        t = lens_edge_thickness(s_front, s_rear, r)
        if t is None or t < min_edge:
            break
        best = r
    # If even tiny r fails, keep a minimal mechanical radius at vertex thickness
    if best < 1e-6:
        t0 = lens_edge_thickness(s_front, s_rear, 0.0)
        if t0 is not None and t0 >= min_edge:
            return min(ap_request, 0.5)
    return best if best > 1e-6 else min(ap_request, 0.5)


def _surface_from_element(
    e: Dict[str, Any],
    side: int,
    z: float,
    ap: float,
    glass: str,
    label: str,
    x0: float = 0.0,
    y0: float = 0.0,
) -> OpticalSurface:
    """
    Build OpticalSurface from element dict; side=1 front, side=2 rear.
    Front: air → glass.  Rear: glass → air.  (light travels +Z)
    """
    mode = str(e.get("surface_mode", "rotational"))
    ap_y = e.get("aperture_y", None)
    ap_y_f = float(ap_y) if ap_y is not None and float(ap_y) > 0 else None
    glass = material_id_from_name(str(glass))

    if side == 1:
        Rx = float(e.get("R1", 20.0))
        Ry = e.get("R1y", None)
        Ry_f = float(Ry) if Ry is not None else None
        k = float(e.get("k1", 0.0))
        ky = float(e.get("k1y", e.get("k1", 0.0)))
        a4 = float(e.get("A4_1", 0.0))
        a4y = float(e.get("A4_1y", e.get("A4_1", 0.0)))
        mat_before, mat_after = "AIR", glass
    else:
        Rx = float(e.get("R2", -20.0))
        Ry = e.get("R2y", None)
        Ry_f = float(Ry) if Ry is not None else None
        k = float(e.get("k2", 0.0))
        ky = float(e.get("k2y", e.get("k2", 0.0)))
        a4 = float(e.get("A4_2", 0.0))
        a4y = float(e.get("A4_2y", e.get("A4_2", 0.0)))
        mat_before, mat_after = glass, "AIR"

    sm = str(e.get(f"mode_s{side}", mode))

    return OpticalSurface(
        z_vertex=z,
        radius=Rx,
        k=k,
        aperture=ap,
        material_before=mat_before,
        material_after=mat_after,
        a4=a4,
        label=label,
        x0=x0,
        y0=y0,
        radius_y=Ry_f,
        k_y=ky,
        a4_y=a4y,
        mode=sm,
        aperture_y=ap_y_f,
    )


# ── Ray / irradiance ─────────────────────────────────────────────────────────

@dataclass
class RayPath:
    history: List[Vec3] = field(default_factory=list)
    power: float = 1.0
    # Parallel to history segments (len = len(history)-1): 'launch'|'refract'|'reflect'|'target'|'kill'
    events: List[str] = field(default_factory=list)
    n_reflections: int = 0
    n_refractions: int = 0
    terminated: str = ""  # target | tir_absorb | miss | power | bounce_limit | backward


@dataclass
class IrradianceMap:
    half_w: float
    half_h: float
    nx: int
    ny: int
    bins: List[float] = field(default_factory=list)
    hit_count: int = 0
    total_power: float = 0.0
    missed_power: float = 0.0

    def __post_init__(self):
        self.bins = [0.0] * (self.nx * self.ny)

    def reset(self):
        self.bins = [0.0] * (self.nx * self.ny)
        self.hit_count = 0
        self.total_power = 0.0
        self.missed_power = 0.0

    @property
    def bin_area(self) -> float:
        return (2 * self.half_w / self.nx) * (2 * self.half_h / self.ny)

    def deposit(self, x: float, y: float, power: float) -> bool:
        u = (x + self.half_w) / (2 * self.half_w)
        v = (y + self.half_h) / (2 * self.half_h)
        if u < 0 or u >= 1 or v < 0 or v >= 1:
            self.missed_power += power
            return False
        ix = min(self.nx - 1, int(u * self.nx))
        iy = min(self.ny - 1, int(v * self.ny))
        row = self.ny - 1 - iy
        self.bins[row * self.nx + ix] += power
        self.hit_count += 1
        self.total_power += power
        return True

    def peak(self) -> float:
        return max(self.bins) if self.bins else 0.0

    def max_irradiance(self) -> float:
        return self.peak() / max(self.bin_area, 1e-30)

    def as_grid(self):
        import numpy as np

        g = np.array(self.bins, dtype=float).reshape(self.ny, self.nx)
        return g

    def centroid(self) -> Tuple[float, float, float]:
        dx = 2 * self.half_w / self.nx
        dy = 2 * self.half_h / self.ny
        sx = sy = sp = 0.0
        for iy in range(self.ny):
            for ix in range(self.nx):
                p = self.bins[iy * self.nx + ix]
                if p <= 0:
                    continue
                x = -self.half_w + (ix + 0.5) * dx
                y = self.half_h - (iy + 0.5) * dy
                sx += x * p
                sy += y * p
                sp += p
        if sp < 1e-30:
            return 0.0, 0.0, 0.0
        return sx / sp, sy / sp, sp

    def rms_radius(self) -> float:
        cx, cy, sp = self.centroid()
        if sp < 1e-30:
            return 0.0
        dx = 2 * self.half_w / self.nx
        dy = 2 * self.half_h / self.ny
        s = 0.0
        for iy in range(self.ny):
            for ix in range(self.nx):
                p = self.bins[iy * self.nx + ix]
                if p <= 0:
                    continue
                x = -self.half_w + (ix + 0.5) * dx
                y = self.half_h - (iy + 0.5) * dy
                s += p * ((x - cx) ** 2 + (y - cy) ** 2)
        return math.sqrt(s / sp)

    def encircled_radius(self, fraction: float = 0.86) -> float:
        cx, cy, sp = self.centroid()
        if sp < 1e-30:
            return 0.0
        dx = 2 * self.half_w / self.nx
        dy = 2 * self.half_h / self.ny
        parts = []
        for iy in range(self.ny):
            for ix in range(self.nx):
                p = self.bins[iy * self.nx + ix]
                if p <= 0:
                    continue
                x = -self.half_w + (ix + 0.5) * dx
                y = self.half_h - (iy + 0.5) * dy
                parts.append((math.hypot(x - cx, y - cy), p))
        parts.sort(key=lambda t: t[0])
        target = fraction * sp
        acc = 0.0
        for r, p in parts:
            acc += p
            if acc >= target:
                return r
        return parts[-1][0] if parts else 0.0

    def fov_metrics(self, fov_w: float, fov_h: float, fov_cx: float = 0.0, fov_cy: float = 0.0) -> Dict[str, float]:
        dx = 2 * self.half_w / self.nx
        dy = 2 * self.half_h / self.ny
        power_in = 0.0
        min_e = float("inf")
        max_e = 0.0
        sum_e = 0.0
        sum_e2 = 0.0
        n_bins = 0
        area = self.bin_area
        # power-weighted second moments for footprint aspect
        sum_p = 0.0
        sum_x2 = 0.0
        sum_y2 = 0.0
        for iy in range(self.ny):
            for ix in range(self.nx):
                x = -self.half_w + (ix + 0.5) * dx
                y = self.half_h - (iy + 0.5) * dy
                p = self.bins[iy * self.nx + ix]
                if p > 0:
                    sum_p += p
                    sum_x2 += p * (x - fov_cx) ** 2
                    sum_y2 += p * (y - fov_cy) ** 2
                if abs(x - fov_cx) > fov_w / 2 or abs(y - fov_cy) > fov_h / 2:
                    continue
                power_in += p
                e = p / area
                min_e = min(min_e, e)
                max_e = max(max_e, e)
                sum_e += e
                sum_e2 += e * e
                n_bins += 1
        # RMS half-widths of whole map (spot shape)
        sig_x = math.sqrt(sum_x2 / sum_p) if sum_p > 1e-30 else 0.0
        sig_y = math.sqrt(sum_y2 / sum_p) if sum_p > 1e-30 else 0.0
        footprint_aspect = sig_x / max(sig_y, 1e-12)
        target_aspect = float(fov_w) / max(float(fov_h), 1e-12)
        aspect_error = abs(footprint_aspect - target_aspect) / max(target_aspect, 1e-12)

        if n_bins == 0:
            return {
                "power_in": 0.0,
                "fraction": 0.0,
                "min_e": 0.0,
                "max_e": 0.0,
                "mean_e": 0.0,
                "uniformity": 0.0,
                "cv": 0.0,
                "footprint_aspect": footprint_aspect,
                "target_aspect": target_aspect,
                "aspect_error": aspect_error,
                "sig_x": sig_x,
                "sig_y": sig_y,
            }
        mean_e = sum_e / n_bins
        var_e = sum_e2 / n_bins - mean_e * mean_e
        std_e = math.sqrt(max(0.0, var_e))
        if min_e == float("inf"):
            min_e = 0.0
        return {
            "power_in": power_in,
            "fraction": power_in / self.total_power if self.total_power > 0 else 0.0,
            "min_e": min_e,
            "max_e": max_e,
            "mean_e": mean_e,
            "uniformity": min_e / max_e if max_e > 1e-30 else 0.0,
            "cv": std_e / mean_e if mean_e > 1e-30 else 0.0,
            "footprint_aspect": footprint_aspect,
            "target_aspect": target_aspect,
            "aspect_error": aspect_error,
            "sig_x": sig_x,
            "sig_y": sig_y,
        }


def _media_for_hit(
    surf: OpticalSurface,
    d: Vec3,
    nrm: Vec3,
    wavelength_nm: float,
    custom_n: float,
) -> Tuple[float, float, Vec3]:
    """
    Determine (n1, n2, N_toward_incident) for a hit.
    Surfaces store material_before (−Z side) and material_after (+Z side).
    Geometric normal is oriented with positive Z component.
    """
    n_before = refractive_index(surf.material_before, wavelength_nm, custom_n)
    n_after = refractive_index(surf.material_after, wavelength_nm, custom_n)
    # N points generally +Z
    N = nrm
    if N[2] < 0:
        N = v_scale(N, -1.0)
    # Ray traveling with +N (d·N > 0) comes from the −Z side → before → after
    if v_dot(d, N) > 0:
        n1, n2 = n_before, n_after
        # Incident is from −Z; N for Snell should face incident → −N? 
        # Standard: N points toward incident medium.
        # Incident medium is before (−Z), so N should point −Z (toward before).
        N_inc = v_scale(N, -1.0)
    else:
        # traveling toward −Z, coming from +Z side
        n1, n2 = n_after, n_before
        N_inc = N  # points +Z toward after (incident)
    return n1, n2, N_inc


def trace_ray(
    o: Vec3,
    d: Vec3,
    power: float,
    wavelength_nm: float,
    surfaces: List[OpticalSurface],
    target_z: float,
    custom_n: float = 1.5,
    apply_fresnel: bool = True,
    absorb_on_tir: bool = True,
    store_path: bool = False,
    max_reflections: int = 0,
    kill_backward: bool = True,
    max_interactions: int = 32,
) -> Tuple[bool, Optional[Vec3], float, Optional[RayPath]]:
    """
    Closest-hit surface tracing with two-sided media.

    Defaults suited to illumination design:
      - absorb_on_tir=True  → no TIR bounce (rays that TIR are removed)
      - max_reflections=0   → same; set >0 only if you want ghost paths
      - kill_backward=True  → terminate if direction has d_z < 0 after an event

    Supports micro-lens arrays (many coplanar decentered surfaces).
    """
    n_med = refractive_index("AIR", wavelength_nm, custom_n)
    history: List[Vec3] = [o] if store_path else []
    events: List[str] = []
    last_hit_i: Optional[int] = None
    n_refl = 0
    n_refr = 0
    guard = 0

    def _path(term: str, pwr: float = power) -> Optional[RayPath]:
        if not store_path:
            return None
        return RayPath(
            history=history,
            power=pwr,
            events=list(events),
            n_reflections=n_refl,
            n_refractions=n_refr,
            terminated=term,
        )

    while guard < max_interactions:
        guard += 1

        # Optional: do not continue rays going toward −Z (back toward source)
        if kill_backward and d[2] < -1e-9:
            return False, None, 0.0, _path("backward", 0.0)

        best = None
        best_s = None
        best_i = None
        for i, s in enumerate(surfaces):
            if i == last_hit_i:
                continue
            hit = s.intersect(o, d, t_min=1e-5)
            if hit is None:
                continue
            t, p, nrm = hit
            if best is None or t < best[0]:
                best = (t, p, nrm)
                best_s = s
                best_i = i

        t_target = None
        if abs(d[2]) > 1e-14:
            tt = (target_z - o[2]) / d[2]
            if tt > 1e-5:
                t_target = tt

        if t_target is not None and (best is None or t_target < best[0]):
            p = v_add(o, v_scale(d, t_target))
            if store_path:
                history.append(p)
                events.append("target")
            return True, p, power, _path("target")

        if best is None or best_s is None:
            if t_target is not None and d[2] > 0:
                p = v_add(o, v_scale(d, t_target))
                if store_path:
                    history.append(p)
                    events.append("target")
                return True, p, power, _path("target")
            return False, None, power, _path("miss")

        t, p, nrm = best
        if store_path:
            history.append(p)

        # Two-sided media from surface definition (not only material_after)
        n1_geom, n2_geom, N_inc = _media_for_hit(best_s, d, nrm, wavelength_nm, custom_n)
        # Trust geometric n1/n2 for Snell (consistent interface); keep n_med for validation
        n1, n2 = n1_geom, n2_geom

        # Skip null interfaces (same medium both sides) — advance without event noise
        if abs(n1 - n2) < 1e-10:
            o = v_add(p, v_scale(d, 1e-4))
            last_hit_i = best_i
            if store_path and events:
                events.append("ghost")
            elif store_path:
                events.append("ghost")
            continue

        cosi = -v_dot(d, N_inc)
        if cosi < 0:
            N_inc = v_scale(N_inc, -1.0)
            cosi = -v_dot(d, N_inc)
            n1, n2 = n2, n1  # flipped interface orientation

        new_d, tir = snell_refract(d, N_inc, n1, n2)
        if tir:
            if absorb_on_tir or n_refl >= max_reflections:
                if store_path:
                    events.append("tir_absorb")
                return False, None, 0.0, _path("tir_absorb", 0.0)
            # Specular TIR bounce (ghost path) — only when explicitly allowed
            n_refl += 1
            d = new_d
            o = v_add(p, v_scale(d, 1e-4))
            last_hit_i = best_i
            if store_path:
                events.append("reflect")
            if kill_backward and d[2] < -1e-9:
                return False, None, 0.0, _path("backward", 0.0)
            continue

        n_refr += 1
        if apply_fresnel:
            T, fr_tir = fresnel_T(n1, n2, cosi)
            if fr_tir:
                # Should be handled by snell; belt-and-suspenders
                if absorb_on_tir:
                    if store_path:
                        events.append("tir_absorb")
                    return False, None, 0.0, _path("tir_absorb", 0.0)
            power *= T
        if power < 1e-8:
            if store_path:
                events.append("kill")
            return False, None, 0.0, _path("power", 0.0)

        d = new_d
        o = v_add(p, v_scale(d, 1e-4))
        n_med = n2
        last_hit_i = best_i
        if store_path:
            events.append("refract")

    return False, None, power, _path("bounce_limit")


@dataclass
class SimResult:
    map: IrradianceMap
    paths: List[RayPath]
    stats: Dict[str, Any]
    dies: List[EmitterDie]
    surfaces: List[OpticalSurface]


def run_simulation(params: Dict[str, Any], progress_cb=None) -> SimResult:
    dies = build_source_array(params["source"])
    surfaces = build_surfaces(
        params["elements"],
        float(params.get("lens_z_start", 3.0)),
        mla=params.get("mla"),
        dies=dies,
    )
    target_z = float(params.get("target_z", 80.0))
    half_w = float(params.get("map_half_w", 50.0))
    half_h = float(params.get("map_half_h", 40.0))
    res = int(params.get("map_res", 96))
    ny = max(16, int(res * half_h / half_w))
    imap = IrradianceMap(half_w, half_h, res, ny)
    total_rays = int(params.get("total_rays", 6000))
    max_display = int(params.get("display_rays", 300))
    custom_n = float(params.get("custom_n", 1.5))
    apply_fresnel = bool(params.get("apply_fresnel", True))
    absorb_tir = bool(params.get("absorb_on_tir", True))
    max_refl = int(params.get("max_reflections", 0))
    kill_backward = bool(params.get("kill_backward", True))
    fov_w = float(params.get("fov_width", 40.0))
    fov_h = float(params.get("fov_height", 32.0))
    fov_cx = float(params.get("fov_cx", 0.0))
    fov_cy = float(params.get("fov_cy", 0.0))

    active = [d for d in dies if d.enabled and d.flux > 0]
    paths: List[RayPath] = []
    if not active or total_rays < 1:
        stats = _empty_stats()
        return SimResult(imap, paths, stats, dies, surfaces)

    total_f = sum(d.flux for d in active)
    power_per = total_f / total_rays
    launched = hit = 0
    n_tir = n_reflect = n_backward = n_miss = 0
    batch = max(50, total_rays // 30)

    for i in range(total_rays):
        r = random.random() * total_f
        die = active[0]
        for dd in active:
            r -= dd.flux
            if r <= 0:
                die = dd
                break
        o, d, pwr, wl = die.spawn_ray(power_per)
        store = len(paths) < max_display and (
            random.random() < (max_display / max(total_rays, 1)) * 1.4 or len(paths) < 40
        )
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
            if path.terminated == "tir_absorb":
                n_tir += 1
            elif path.terminated == "backward":
                n_backward += 1
            elif path.terminated == "miss":
                n_miss += 1
            n_reflect += path.n_reflections
        if ok and pt is not None:
            hit += 1
            imap.deposit(pt[0], pt[1], pwr_out)
        if store and path is not None and len(path.history) >= 2:
            paths.append(path)
        if progress_cb and (i % batch == 0 or i == total_rays - 1):
            progress_cb((i + 1) / total_rays)

    cx, cy, _ = imap.centroid()
    fov = imap.fov_metrics(fov_w, fov_h, fov_cx, fov_cy)
    e0 = next((e for e in params["elements"] if e.get("enabled")), None)
    efl = float("nan")
    if e0:
        n_use = refractive_index(
            e0.get("material", "N_BK7"),
            float(params["source"].get("wavelength_nm", VISIBLE_NM_DEFAULT)),
            custom_n,
        )
        efl = lensmaker_f(float(e0["R1"]), float(e0["R2"]), n_use, float(e0["thickness"]))

    stats = {
        "launched": launched,
        "hit": hit,
        "collection": imap.total_power / total_f if total_f > 0 else 0.0,
        "rms": imap.rms_radius(),
        "ee50": imap.encircled_radius(0.5),
        "ee86": imap.encircled_radius(0.86),
        "peak_e": imap.max_irradiance(),
        "centroid": (cx, cy),
        "fov": fov,
        "source_power": total_f,
        "map_power": imap.total_power,
        "efl": efl,
        "n_dies": len(active),
        "n_surfaces": len(surfaces),
        "n_tir_absorb": n_tir,
        "n_reflections": n_reflect,
        "n_backward": n_backward,
        "n_miss": n_miss,
    }
    return SimResult(imap, paths, stats, dies, surfaces)


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
        "n_reflections": 0,
        "n_backward": 0,
        "n_miss": 0,
    }


def default_params() -> Dict[str, Any]:
    return {
        "source": {
            "mode": "cob",
            "rows": 4,
            "cols": 4,
            "pitch_x": 1.6,
            "pitch_y": 1.6,
            "die_width": 1.0,
            "die_height": 1.0,
            "source_z": 0.0,
            "flux_per_die": 1.0,
            "wavelength_nm": VISIBLE_NM_DEFAULT,
            "half_angle_deg": 60.0,
            "tilt_x": 0.0,
            "tilt_y": 0.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "die_rot_z": 0.0,
            "stagger": False,
            "circular_mask": False,
            "mask_radius": 4.0,
        },
        "elements": [
            {
                # Mild bi-convex with positive edge thickness across clear aperture
                "enabled": True,
                "R1": 40.0,
                "R2": -50.0,
                "thickness": 6.0,
                "air_after": 2.0,
                "aperture": 10.0,
                "material": "ACRYLIC_PMMA",
                "shape_id": "biconvex",
                "surface_mode": "rotational",
                "k1": 0.0,
                "k2": 0.0,
                "A4_1": 0.0,
                "A4_2": 0.0,
                "R1y": None,
                "R2y": None,
                "aperture_y": None,
            },
            {
                "enabled": False,
                "R1": 30.0,
                "R2": -30.0,
                "thickness": 3.0,
                "air_after": 2.0,
                "aperture": 12.0,
                "material": "N_BK7",
                "shape_id": "custom",
                "surface_mode": "rotational",
                "k1": 0.0,
                "k2": 0.0,
                "A4_1": 0.0,
                "A4_2": 0.0,
                "R1y": None,
                "R2y": None,
                "aperture_y": None,
            },
            {
                "enabled": False,
                "R1": 40.0,
                "R2": -25.0,
                "thickness": 3.0,
                "air_after": 1.0,
                "aperture": 11.0,
                "material": "FORMLABS_CLEAR",
                "shape_id": "custom",
                "surface_mode": "rotational",
                "k1": 0.0,
                "k2": 0.0,
                "A4_1": 0.0,
                "A4_2": 0.0,
                "R1y": None,
                "R2y": None,
                "aperture_y": None,
            },
        ],
        "lens_z_start": 3.0,
        "custom_n": 1.5,
        "apply_fresnel": True,
        "absorb_on_tir": True,
        "max_reflections": 0,
        "kill_backward": True,
        "target_z": 80.0,
        "fov_width": 40.0,
        "fov_height": 32.0,
        "fov_cx": 0.0,
        "fov_cy": 0.0,
        "fov_aspect_lock": True,
        "map_half_w": 50.0,
        "map_half_h": 40.0,
        "map_res": 96,
        "total_rays": 6000,
        "display_rays": 300,
        "mla": {
            "enabled": False,
            "fill_factor": 0.88,
            "lenslet_aperture": 0.0,  # 0 = auto from COB pitch
            "export_plate": True,
            "scale_to_pitch": True,  # scale Element 1 R/t to die size (real micro-lenses)
        },
    }
