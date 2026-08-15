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
    # Interaction: "refract" (default lenses) or "absorb" (opaque blockers / stops)
    interaction: str = "refract"
    # Outer aperture shape for hit test: "circle" | "rect" (axis-aligned)
    aperture_shape: str = "circle"
    # Central hole (aperture stop). None / ≤0 → solid panel. Same shape family as outer.
    inner_aperture: Optional[float] = None
    inner_aperture_y: Optional[float] = None
    # Display-only thickness (mm) for Z-facing stops; for baffles/tubes = axial length hint
    display_thickness: float = 1.0
    # Geometry for intersection (lenses use asphere sag; blockers use planes / tube)
    #   asphere  — standard optical surface z = z_vertex + sag(x,y)
    #   plane_z  — face-on stop: plane z = z_vertex (vertical in side view)
    #   plane_y  — horizontal baffle: plane y = y0 + plane_offset
    #   plane_x  — side baffle: plane x = x0 + plane_offset
    #   cylinder_z — tube / pipe / lens barrel along optical +Z
    geom: str = "asphere"
    plane_offset: float = 0.0  # signed offset from x0/y0/z_vertex for plane_* geoms
    # Half-length along optical Z for baffles & tubes (mm); full length = 2 * extent_z
    extent_z: float = 10.0

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
            # Power in X only — both front and rear use self.radius (R1 / R2)
            cx, cy = _curv(self.radius), 0.0
            kx, ky = self.k, 0.0
            a4x, a4y = self.a4, 0.0
        elif mode == "cylinder_y":
            # Power in Y only — prefer explicit radius_y, else self.radius (R1 / R2)
            # so a biconvex cylinder has curved front AND rear when |R1| and |R2| ≠ 0
            Ry = self.radius_y if self.radius_y is not None else self.radius
            cx, cy = 0.0, _curv(Ry)
            kx, ky = 0.0, self.k_y if self.radius_y is not None else self.k
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

    def _in_region(
        self,
        lx: float,
        ly: float,
        semi_x: float,
        semi_y: Optional[float],
        *,
        shape: Optional[str] = None,
    ) -> bool:
        """True if (lx, ly) is inside a circle/ellipse/rect of the given semi-sizes."""
        shape = (shape or self.aperture_shape or "circle").lower()
        ax = max(float(semi_x), 1e-12)
        if shape == "rect":
            ay = max(float(semi_y if semi_y is not None and semi_y > 0 else semi_x), 1e-12)
            return abs(lx) <= ax + 1e-9 and abs(ly) <= ay + 1e-9
        # circle / ellipse
        if semi_y is not None and float(semi_y) > 0 and abs(float(semi_y) - ax) > 1e-12:
            ay = max(float(semi_y), 1e-12)
            return (lx / ax) ** 2 + (ly / ay) ** 2 <= 1.0 + 1e-9
        return math.hypot(lx, ly) <= ax + 1e-6

    def in_aperture(self, lx: float, ly: float) -> bool:
        """Outer clear aperture / panel outer extent."""
        return self._in_region(lx, ly, self.aperture, self.aperture_y)

    def in_hole(self, lx: float, ly: float) -> bool:
        """True if inside the central hole (aperture stop opening)."""
        inn = self.inner_aperture
        if inn is None or float(inn) <= 1e-12:
            return False
        return self._in_region(lx, ly, float(inn), self.inner_aperture_y)

    def in_hit_region(self, lx: float, ly: float) -> bool:
        """
        Region where the surface actually interacts with the ray.
        Lenses: outer clear aperture (no hole).
        Absorbing panels: outer AND NOT hole (annulus / frame).
        """
        if not self.in_aperture(lx, ly):
            return False
        if self.interaction == "absorb" and self.in_hole(lx, ly):
            return False
        return True

    def surface_z(self, x: float, y: float) -> Optional[float]:
        lx, ly = self.local_xy(x, y)
        s = self.sag_xy(lx, ly)
        if s is None:
            return None
        return self.z_vertex + s

    def normal_at(self, x: float, y: float) -> Optional[Vec3]:
        g = (self.geom or "asphere").lower()
        if g == "plane_y":
            return (0.0, 1.0, 0.0)
        if g == "plane_x":
            return (1.0, 0.0, 0.0)
        if g in ("plane_z", "asphere") and abs(self.radius) < 1e-14 and g == "plane_z":
            return (0.0, 0.0, 1.0)
        if g == "cylinder_z":
            lx, ly = self.local_xy(x, y)
            return v_norm((lx, ly, 0.0)) if math.hypot(lx, ly) > 1e-15 else (1.0, 0.0, 0.0)
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

    def _z_span(self) -> Tuple[float, float]:
        half = max(float(self.extent_z), 1e-6)
        return self.z_vertex - half, self.z_vertex + half

    def _intersect_plane_z(self, o: Vec3, d: Vec3, t_min: float, t_max: float):
        if abs(d[2]) < 1e-14:
            return None
        t = (self.z_vertex - o[2]) / d[2]
        if t < t_min or t > t_max:
            return None
        p = v_add(o, v_scale(d, t))
        lx, ly = self.local_xy(p[0], p[1])
        if not self.in_hit_region(lx, ly):
            return None
        return t, p, (0.0, 0.0, 1.0)

    def _intersect_plane_y(self, o: Vec3, d: Vec3, t_min: float, t_max: float):
        """Horizontal baffle: y = y0 + plane_offset, strip in X and Z."""
        if abs(d[1]) < 1e-14:
            return None
        y_plane = self.y0 + self.plane_offset
        t = (y_plane - o[1]) / d[1]
        if t < t_min or t > t_max:
            return None
        p = v_add(o, v_scale(d, t))
        z0, z1 = self._z_span()
        if p[2] < z0 - 1e-9 or p[2] > z1 + 1e-9:
            return None
        # Bounds in X (half-width = aperture)
        if abs(p[0] - self.x0) > max(float(self.aperture), 1e-9) + 1e-9:
            return None
        n = (0.0, 1.0 if d[1] < 0 else -1.0, 0.0)  # face incident
        # Standard: geometric normal +Y; snell path will flip if needed
        return t, p, (0.0, 1.0, 0.0)

    def _intersect_plane_x(self, o: Vec3, d: Vec3, t_min: float, t_max: float):
        """Side baffle: x = x0 + plane_offset, strip in Y and Z."""
        if abs(d[0]) < 1e-14:
            return None
        x_plane = self.x0 + self.plane_offset
        t = (x_plane - o[0]) / d[0]
        if t < t_min or t > t_max:
            return None
        p = v_add(o, v_scale(d, t))
        z0, z1 = self._z_span()
        if p[2] < z0 - 1e-9 or p[2] > z1 + 1e-9:
            return None
        half_y = max(
            float(self.aperture_y if self.aperture_y is not None else self.aperture),
            1e-9,
        )
        if abs(p[1] - self.y0) > half_y + 1e-9:
            return None
        return t, p, (1.0, 0.0, 0.0)

    def _intersect_cylinder_z(self, o: Vec3, d: Vec3, t_min: float, t_max: float):
        """
        Absorbing tube / pipe / lens barrel along +Z.
        Thin shell at r = aperture (outer radius). Optional inner radius ignores
        hits with local r < inner (not used for thin shell; both shells if inner set).
        """
        # Ray in XY relative to axis (x0, y0)
        ox, oy = o[0] - self.x0, o[1] - self.y0
        dx, dy = d[0], d[1]
        R = max(float(self.aperture), 1e-9)
        # Quadratic: |o_xy + t d_xy|^2 = R^2
        a = dx * dx + dy * dy
        if a < 1e-16:
            return None  # ray parallel to axis
        b = 2.0 * (ox * dx + oy * dy)
        c = ox * ox + oy * oy - R * R
        disc = b * b - 4.0 * a * c
        if disc < 0:
            return None
        sdisc = math.sqrt(disc)
        candidates = [(-b - sdisc) / (2.0 * a), (-b + sdisc) / (2.0 * a)]
        z0, z1 = self._z_span()
        best = None
        for t in sorted(candidates):
            if t < t_min or t > t_max:
                continue
            p = v_add(o, v_scale(d, t))
            if p[2] < z0 - 1e-9 or p[2] > z1 + 1e-9:
                continue
            # Optional thick wall: also absorb on inner cylinder
            n = v_norm((p[0] - self.x0, p[1] - self.y0, 0.0))
            best = (t, p, n if n != (0.0, 0.0, 0.0) else (1.0, 0.0, 0.0))
            break
        if best is not None:
            return best
        # Inner shell (bore wall of thick tube)
        r_in = self.inner_aperture
        if r_in is None or float(r_in) <= 1e-12 or float(r_in) >= R:
            return None
        Ri = float(r_in)
        c2 = ox * ox + oy * oy - Ri * Ri
        disc2 = b * b - 4.0 * a * c2
        if disc2 < 0:
            return None
        s2 = math.sqrt(disc2)
        for t in sorted([(-b - s2) / (2.0 * a), (-b + s2) / (2.0 * a)]):
            if t < t_min or t > t_max:
                continue
            p = v_add(o, v_scale(d, t))
            if p[2] < z0 - 1e-9 or p[2] > z1 + 1e-9:
                continue
            n = v_norm((p[0] - self.x0, p[1] - self.y0, 0.0))
            return t, p, n if n != (0.0, 0.0, 0.0) else (1.0, 0.0, 0.0)
        return None

    def intersect(self, o: Vec3, d: Vec3, t_min: float = 1e-6, t_max: float = 1e6):
        if not self.active:
            return None
        g = (self.geom or "asphere").lower()
        if g == "plane_z":
            return self._intersect_plane_z(o, d, t_min, t_max)
        if g == "plane_y":
            return self._intersect_plane_y(o, d, t_min, t_max)
        if g == "plane_x":
            return self._intersect_plane_x(o, d, t_min, t_max)
        if g == "cylinder_z":
            return self._intersect_cylinder_z(o, d, t_min, t_max)

        # Default: asphere Newton intersect
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
                if not self.in_hit_region(lx, ly):
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
            channel_aim_to_fov,
            die_pitch_mm,
            lenslet_semi_aperture,
            scale_element_to_lenslet,
            thin_lens_focal_length_mm,
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
        # Per-channel aim toward FOV center (lens optical-center offset)
        aim_on = bool(mla.get("aim_to_fov", True))
        strength = float(mla.get("aim_strength", 1.0)) if aim_on else 0.0
        n_glass = refractive_index(mat, VISIBLE_NM_DEFAULT, 1.5)
        f_mm = thin_lens_focal_length_mm(g["R1"], g["R2"], n_glass, g["thickness"])
        pitch = die_pitch_mm(dies) if len(dies) >= 2 else max(2.0 * ap_use, 1.6)
        # FOV / throw come from optional keys stashed on mla dict by run_simulation
        target_z = float(mla.get("_target_z", 80.0))
        fov_cx = float(mla.get("_fov_cx", 0.0))
        fov_cy = float(mla.get("_fov_cy", 0.0))
        for li, die in enumerate(dies):
            if aim_on and strength > 0:
                x0, y0, _tx, _ty = channel_aim_to_fov(
                    die.cx,
                    die.cy,
                    die.cz,
                    lens_z=z1,
                    target_z=target_z,
                    fov_cx=fov_cx,
                    fov_cy=fov_cy,
                    focal_length=f_mm,
                    aperture=ap_use,
                    pitch=pitch,
                    aim_strength=strength,
                )
            else:
                x0, y0 = die.cx, die.cy
            s1 = _surface_from_element(
                e_lens, side=1, z=z1, ap=ap_use, glass=mat, label=f"MLA{li}S1", x0=x0, y0=y0
            )
            s2 = _surface_from_element(
                e_lens, side=2, z=z2, ap=ap_use, glass=mat, label=f"MLA{li}S2", x0=x0, y0=y0
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


def default_blocker(
    *,
    z: float = 20.0,
    shape: str = "rect",
    orient: Optional[str] = None,
    outer_w: float = 15.0,
    outer_h: float = 15.0,
    inner_w: float = 0.0,
    inner_h: float = 0.0,
    length: float = 40.0,
    label: str = "Blocker",
    enabled: bool = True,
) -> Dict[str, Any]:
    """
    Factory for an absorbing panel / tube / aperture-stop dict.

    orient
      - ``horizontal`` — rect body: top/bottom (+ sides) along Z (default for rect)
      - ``vertical`` / ``face`` — Z-normal stop / aperture (face the beam)
      - ``tube`` — circular pipe / lens barrel along Z (default for circle)
    """
    shape = str(shape or "rect").lower()
    if orient is None:
        orient = "tube" if shape == "circle" else "horizontal"
    return {
        "enabled": bool(enabled),
        "label": str(label),
        "z": float(z),
        "shape": shape,  # "rect" | "circle"
        "orient": str(orient),  # horizontal | vertical | face | tube
        "outer_w": float(outer_w),
        "outer_h": float(outer_h),
        "inner_w": float(inner_w),
        "inner_h": float(inner_h),
        "length": float(length),  # axial span for baffles / tubes (mm)
        "x0": 0.0,
        "y0": 0.0,
        "thickness": 1.0,  # face-stop display thickness (mm)
    }


def build_blockers(blockers: Optional[List[Dict[str, Any]]] = None) -> List[OpticalSurface]:
    """
    Build opaque absorbing geometry.

    Orientations
    ------------
    * **vertical / face** — Z-normal plane (aperture stop); vertical line in side view
    * **horizontal** — rectangular tube body: top & bottom (and left & right) walls
      parallel to the optical axis — horizontal lines in the side view
    * **tube** — circular cylinder along +Z (camera barrel / snoot / pipe)

    Axial ``length`` is physical for baffles and tubes. Face-stop ``thickness``
    is display-only (hit is still zero-thickness at Z).
    """
    out: List[OpticalSurface] = []
    if not blockers:
        return out
    for i, b in enumerate(blockers):
        if not isinstance(b, dict) or not b.get("enabled", True):
            continue
        shape = str(b.get("shape", "rect") or "rect").lower()
        if shape not in ("rect", "circle"):
            shape = "rect"
        orient = str(b.get("orient") or ("tube" if shape == "circle" else "horizontal")).lower()
        # Legacy files without orient: circle → tube, rect with hole → face stop
        if "orient" not in b:
            if shape == "circle" and float(b.get("inner_w", 0) or 0) <= 1e-12 and float(b.get("length", 0) or 0) <= 1e-12:
                # Old solid disks were face stops; old circle with only z → treat as tube if length set later
                if float(b.get("outer_w", 0) or 0) > 0 and float(b.get("thickness", 1) or 1) <= 2.0:
                    # Prefer tube for circle default going forward; keep face if explicitly a stop hole
                    orient = "tube"
            if shape == "rect" and float(b.get("inner_w", 0) or 0) > 1e-12:
                orient = "vertical"  # old aperture stop
            elif shape == "rect" and orient not in ("horizontal", "vertical", "face", "tube"):
                orient = "horizontal"
        if orient in ("face", "stop", "z"):
            orient = "vertical"
        if orient in ("pipe", "barrel", "cylinder"):
            orient = "tube"
        if shape == "circle" and orient == "horizontal":
            orient = "tube"  # circle is never a flat horizontal plate in our model

        outer_w = max(float(b.get("outer_w", 15.0) or 15.0), 1e-3)
        outer_h = max(float(b.get("outer_h", outer_w) or outer_w), 1e-3)
        inner_w = max(float(b.get("inner_w", 0.0) or 0.0), 0.0)
        inner_h = max(float(b.get("inner_h", 0.0) or 0.0), 0.0)
        z = float(b.get("z", 20.0))
        x0 = float(b.get("x0", 0.0) or 0.0)
        y0 = float(b.get("y0", 0.0) or 0.0)
        lab = str(b.get("label") or f"BLK{i}")
        # Axial length: prefer length; fall back to thickness for older tube-like entries
        length = float(b.get("length", 0.0) or 0.0)
        if length <= 1e-9:
            length = max(float(b.get("thickness", 40.0) or 40.0), 1.0) if orient != "vertical" else 1.0
        half_z = 0.5 * max(length, 1e-3)
        thick_disp = max(float(b.get("thickness", 1.0) or 1.0), 1e-3)
        base_lab = f"BLK{i}:{lab}"

        if orient == "vertical":
            # Face-on aperture stop / solid disk (legacy Z plane)
            if shape == "circle":
                if inner_w >= outer_w:
                    inner_w = max(0.0, outer_w * 0.95)
                ap_y, inn_y = None, None
            else:
                if inner_w >= outer_w:
                    inner_w = max(0.0, outer_w * 0.95)
                if inner_h >= outer_h:
                    inner_h = max(0.0, outer_h * 0.95)
                ap_y = outer_h
                inn_y = inner_h if inner_w > 1e-12 or inner_h > 1e-12 else None
            out.append(
                OpticalSurface(
                    z_vertex=z,
                    radius=0.0,
                    aperture=outer_w,
                    aperture_y=ap_y,
                    material_before="AIR",
                    material_after="AIR",
                    label=base_lab,
                    x0=x0,
                    y0=y0,
                    mode="rotational",
                    interaction="absorb",
                    aperture_shape=shape,
                    inner_aperture=inner_w if inner_w > 1e-12 else None,
                    inner_aperture_y=(
                        inn_y if (inn_y is not None and inn_y > 1e-12) else (
                            None if shape == "circle" else (inner_h if inner_h > 1e-12 else None)
                        )
                    ),
                    display_thickness=thick_disp,
                    geom="plane_z",
                    extent_z=0.5 * thick_disp,
                )
            )
            continue

        if orient == "tube" or shape == "circle":
            # Circular pipe / lens barrel along +Z
            if inner_w >= outer_w:
                inner_w = max(0.0, outer_w * 0.95)
            out.append(
                OpticalSurface(
                    z_vertex=z,
                    radius=0.0,
                    aperture=outer_w,  # outer radius
                    material_before="AIR",
                    material_after="AIR",
                    label=base_lab,
                    x0=x0,
                    y0=y0,
                    mode="rotational",
                    interaction="absorb",
                    aperture_shape="circle",
                    inner_aperture=inner_w if inner_w > 1e-12 else None,
                    display_thickness=length,
                    geom="cylinder_z",
                    extent_z=half_z,
                )
            )
            continue

        # horizontal rect body: four walls of a rectangular tube (camera body)
        # Top / bottom at y = y0 ± outer_h, half-width outer_w in X
        for sign, tag in ((+1.0, "top"), (-1.0, "bot")):
            out.append(
                OpticalSurface(
                    z_vertex=z,
                    radius=0.0,
                    aperture=outer_w,  # half-width in X
                    material_before="AIR",
                    material_after="AIR",
                    label=f"{base_lab}:{tag}",
                    x0=x0,
                    y0=y0,
                    mode="rotational",
                    interaction="absorb",
                    aperture_shape="rect",
                    display_thickness=length,
                    geom="plane_y",
                    plane_offset=sign * outer_h,
                    extent_z=half_z,
                )
            )
        # Left / right at x = x0 ± outer_w, half-height outer_h in Y
        for sign, tag in ((+1.0, "right"), (-1.0, "left")):
            out.append(
                OpticalSurface(
                    z_vertex=z,
                    radius=0.0,
                    aperture=outer_w,
                    aperture_y=outer_h,
                    material_before="AIR",
                    material_after="AIR",
                    label=f"{base_lab}:{tag}",
                    x0=x0,
                    y0=y0,
                    mode="rotational",
                    interaction="absorb",
                    aperture_shape="rect",
                    display_thickness=length,
                    geom="plane_x",
                    plane_offset=sign * outer_w,
                    extent_z=half_z,
                )
            )
    return out


def blockers_need_cpu(surfaces: List[OpticalSurface]) -> bool:
    """True if any absorber uses non-asphere geometry (Warp lacks full support)."""
    for s in surfaces:
        if getattr(s, "interaction", "refract") != "absorb":
            continue
        g = (getattr(s, "geom", "asphere") or "asphere").lower()
        if g not in ("asphere", "plane_z"):
            return True
    return False


def assemble_surfaces(
    elements: List[Dict[str, Any]],
    z_start: float,
    mla: Optional[Dict[str, Any]] = None,
    dies: Optional[List[EmitterDie]] = None,
    blockers: Optional[List[Dict[str, Any]]] = None,
) -> List[OpticalSurface]:
    """Refractive stack (+ optional MLA) plus absorbing blockers."""
    surfs = build_surfaces(elements, z_start, mla=mla, dies=dies)
    surfs.extend(build_blockers(blockers))
    return surfs


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
    *,
    along_x: bool = False,
) -> Optional[float]:
    """
    Glass thickness along +Z at height r from the local optical axis.
    along_x=False → sample the Y meridian; True → the X meridian.
    Both meridians are needed for cylinders / biconics so the clear aperture
    is not limited only in the flat direction.
    """
    if along_x:
        z1 = s_front.surface_z(s_front.x0 + r, s_front.y0)
        z2 = s_rear.surface_z(s_rear.x0 + r, s_rear.y0)
    else:
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
    Largest semi-aperture ≤ ap_request with edge thickness ≥ min_edge (mm)
    in *both* the X and Y meridians. Prevents self-intersecting lens drawings
    and keeps resize handles locked to the drawable lens body.
    """
    ap_request = max(float(ap_request), 1e-3)
    best = 0.0
    for i in range(1, samples + 1):
        r = ap_request * i / samples
        t_y = lens_edge_thickness(s_front, s_rear, r, along_x=False)
        t_x = lens_edge_thickness(s_front, s_rear, r, along_x=True)
        if t_y is None or t_x is None or t_y < min_edge or t_x < min_edge:
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

def element_id_from_label(label: str) -> str:
    """Map surface label (E1S1, MLA3S2) to element id (E1, MLA3)."""
    lab = str(label or "")
    if not lab:
        return ""
    if lab.startswith("MLA"):
        return lab.rsplit("S", 1)[0]
    if "S" in lab:
        return lab.rsplit("S", 1)[0]
    return lab


def path_in_meridional_slice(path: "RayPath", slice_half_mm: float) -> bool:
    """
    True if the path stays near the Y–Z plane (|X| ≤ slice_half_mm).

    Used by the side-view plot so rays with large |X| are not projected onto
    the lens silhouette (they can miss a circular aperture while looking like
    they cross every element in Y–Z).
    """
    if slice_half_mm <= 0:
        return True
    hist = getattr(path, "history", None) or []
    if not hist:
        return False
    return max(abs(float(pt[0])) for pt in hist) <= float(slice_half_mm)


@dataclass
class RayPath:
    history: List[Vec3] = field(default_factory=list)
    power: float = 1.0
    # Parallel to history segments (len = len(history)-1): 'launch'|'refract'|'reflect'|'target'|'kill'
    events: List[str] = field(default_factory=list)
    n_reflections: int = 0
    n_refractions: int = 0
    terminated: str = ""  # target | tir_absorb | absorb | miss | power | bounce_limit | backward
    # Distinct element ids (E1, E2, MLA0, …) that this ray refracted through
    elements_hit: List[str] = field(default_factory=list)


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

    def edge_sharpness(self) -> float:
        """
        Mean irradiance gradient on the half-max contour of the whole map,
        divided by peak so a one-bin step is ~1 and a soft blob is lower.
        Used to focus so the illumination border is as crisp as possible.
        """
        import numpy as np

        g = self.as_grid()
        if g.size == 0:
            return 0.0
        peak = float(np.max(g))
        if peak <= 1e-30:
            return 0.0
        gy, gx = np.gradient(g)
        mag = np.hypot(gx, gy)
        norm = g / peak
        edge = (norm >= 0.25) & (norm <= 0.75)
        if not np.any(edge):
            return float(np.max(mag) / peak)
        return float(np.mean(mag[edge]) / peak)

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
        n_lit = 0  # FOV bins above noise floor
        area = self.bin_area
        # power-weighted second moments for footprint aspect / size
        sum_p = 0.0
        sum_x2 = 0.0
        sum_y2 = 0.0
        fov_irrad: List[float] = []
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
                fov_irrad.append(e)
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

        # Landscape FOV (W>H) must not accept a portrait beam (σx<σy), and vice
        # versa. |aspect−target| alone is not enough: DE can keep an inverted
        # beam if flux is higher. Flag flips and inflate aspect_error.
        orientation_flipped = 0.0
        if abs(target_aspect - 1.0) > 0.02 and abs(footprint_aspect - 1.0) > 0.02:
            if (target_aspect - 1.0) * (footprint_aspect - 1.0) < 0.0:
                orientation_flipped = 1.0
                aspect_error = max(
                    aspect_error,
                    abs(1.0 / max(footprint_aspect, 1e-6) - target_aspect)
                    / max(target_aspect, 1e-12)
                    + 0.5,
                )

        # Ideal RMS half-widths for a *uniform* rectangle of size fov_w × fov_h:
        #   σ = (half-width) / √3
        # Axis-resolved so X and Y cannot be swapped without cost.
        # Under-fill (σ too small) is penalized harder than mild over-fill.
        sig_tgt_x = (0.5 * float(fov_w)) / math.sqrt(3.0)
        sig_tgt_y = (0.5 * float(fov_h)) / math.sqrt(3.0)

        def _size_err(sig: float, tgt: float) -> float:
            if tgt < 1e-9:
                return 0.0
            ratio = sig / tgt
            if ratio < 1.0:
                return (1.0 - ratio) ** 2  # under-fill
            return 0.35 * (ratio - 1.0) ** 2  # over-fill (softer)

        size_error = 0.5 * (_size_err(sig_x, sig_tgt_x) + _size_err(sig_y, sig_tgt_y))
        if orientation_flipped > 0.0:
            size_error = size_error + 0.5

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
                "coverage": 0.0,
                "size_error": 1.0,
                "orientation_flipped": 1.0,
                "profile_fill": 0.0,
                "profile_fill_x": 0.0,
                "profile_fill_y": 0.0,
                "edge_sharpness": self.edge_sharpness(),
            }
        mean_e = sum_e / n_bins
        var_e = sum_e2 / n_bins - mean_e * mean_e
        std_e = math.sqrt(max(0.0, var_e))
        if min_e == float("inf"):
            min_e = 0.0
        # Coverage: fraction of FOV bins with irradiance ≥ 10% of peak-in-FOV
        # (empty bins from under-fill count against coverage)
        peak_fov = max_e if max_e > 1e-30 else 0.0
        thr = 0.10 * peak_fov
        if peak_fov > 0:
            n_lit = sum(1 for e in fov_irrad if e >= thr)
            coverage = n_lit / max(n_bins, 1)
        else:
            coverage = 0.0
        # Uniformity only over lit bins so a small hot-spot is not "0% uniform"
        # solely because empty FOV corners dominate min_e.
        lit = [e for e in fov_irrad if e >= thr] if thr > 0 else []
        if len(lit) >= 2:
            uni = min(lit) / max(lit) if max(lit) > 1e-30 else 0.0
        else:
            uni = 0.0

        # Line-cut (profile) fill — same quantities shown in the profile plot.
        # X-cut at FOV centre Y, Y-cut at FOV centre X: fraction of FOV span
        # where the cut is ≥ 10% of that cut's peak.
        def _profile_fill_1d(along_x: bool) -> float:
            if along_x:
                # row nearest fov_cy
                row = int(round((self.half_h - fov_cy) / max(dy, 1e-12) - 0.5))
                row = max(0, min(self.ny - 1, row))
                vals = [self.bins[row * self.nx + ix] for ix in range(self.nx)]
                coords = [-self.half_w + (ix + 0.5) * dx for ix in range(self.nx)]
                half = 0.5 * float(fov_w)
                c0 = float(fov_cx)
            else:
                col = int(round((fov_cx + self.half_w) / max(dx, 1e-12) - 0.5))
                col = max(0, min(self.nx - 1, col))
                vals = [self.bins[iy * self.nx + col] for iy in range(self.ny)]
                coords = [self.half_h - (iy + 0.5) * dy for iy in range(self.ny)]
                half = 0.5 * float(fov_h)
                c0 = float(fov_cy)
            in_fov = [
                (v, c) for v, c in zip(vals, coords) if abs(c - c0) <= half + 1e-9
            ]
            if not in_fov:
                return 0.0
            peak = max(v for v, _ in in_fov)
            if peak <= 1e-30:
                return 0.0
            thr_p = 0.10 * peak
            lit_n = sum(1 for v, _ in in_fov if v >= thr_p)
            return lit_n / max(len(in_fov), 1)

        profile_fill_x = _profile_fill_1d(True)
        profile_fill_y = _profile_fill_1d(False)
        profile_fill = 0.5 * (profile_fill_x + profile_fill_y)

        return {
            "power_in": power_in,
            "fraction": power_in / self.total_power if self.total_power > 0 else 0.0,
            "min_e": min_e,
            "max_e": max_e,
            "mean_e": mean_e,
            "uniformity": uni,
            "cv": std_e / mean_e if mean_e > 1e-30 else 0.0,
            "footprint_aspect": footprint_aspect,
            "target_aspect": target_aspect,
            "aspect_error": aspect_error,
            "sig_x": sig_x,
            "sig_y": sig_y,
            "coverage": coverage,
            "size_error": size_error,
            "orientation_flipped": orientation_flipped,
            "profile_fill": profile_fill,
            "profile_fill_x": profile_fill_x,
            "profile_fill_y": profile_fill_y,
            "edge_sharpness": self.edge_sharpness(),
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
    elements_hit: List[str] = []
    elements_hit_set: set = set()
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
            elements_hit=list(elements_hit),
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

        # Opaque panels / aperture stops: kill before media / Snell
        # (blockers are AIR|AIR so they would otherwise ghost-pass)
        if getattr(best_s, "interaction", "refract") == "absorb":
            if store_path:
                events.append("absorb")
            return False, None, 0.0, _path("absorb", 0.0)

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
        if store_path and best_s is not None:
            eid = element_id_from_label(getattr(best_s, "label", "") or "")
            if eid and eid not in elements_hit_set:
                elements_hit_set.add(eid)
                elements_hit.append(eid)
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


def target_plane_hits(
    paths: List[RayPath],
    target_z: float,
    z_tol: float = 1.5,
) -> List[Tuple[float, float]]:
    """(X, Y) of every display-path endpoint that reached the target plane."""
    hits: List[Tuple[float, float]] = []
    tz = float(target_z)
    for path in paths:
        hist = getattr(path, "history", None) or []
        if len(hist) < 1:
            continue
        pt = hist[-1]
        if abs(float(pt[2]) - tz) > z_tol and getattr(path, "terminated", "") != "target":
            continue
        hits.append((float(pt[0]), float(pt[1])))
    return hits


@dataclass
class SimResult:
    map: IrradianceMap
    paths: List[RayPath]
    stats: Dict[str, Any]
    dies: List[EmitterDie]
    surfaces: List[OpticalSurface]


def run_simulation(params: Dict[str, Any], progress_cb=None) -> SimResult:
    dies = build_source_array(params["source"])
    target_z = float(params.get("target_z", 80.0))
    fov_cx = float(params.get("fov_cx", 0.0))
    fov_cy = float(params.get("fov_cy", 0.0))
    # Stash FOV / throw on a copy of mla so build_surfaces can aim channels
    mla = dict(params.get("mla") or {})
    mla["_target_z"] = target_z
    mla["_fov_cx"] = fov_cx
    mla["_fov_cy"] = fov_cy
    if bool(mla.get("enabled", False)) and bool(mla.get("aim_to_fov", True)):
        from mla_geometry import apply_mla_die_aim, lenslet_semi_aperture, scale_element_to_lenslet, thin_lens_focal_length_mm

        e0 = next((e for e in params.get("elements", []) if e.get("enabled", True)), None)
        if e0 is not None:
            ap = lenslet_semi_aperture(mla, dies, params.get("source"))
            g = scale_element_to_lenslet(e0, ap, scale_geometry=bool(mla.get("scale_to_pitch", True)))
            mat = material_id_from_name(str(e0.get("material", "ACRYLIC_PMMA")))
            n_g = refractive_index(mat, VISIBLE_NM_DEFAULT, float(params.get("custom_n", 1.5)))
            f_mm = thin_lens_focal_length_mm(g["R1"], g["R2"], n_g, g["thickness"])
            apply_mla_die_aim(dies, {**params, "mla": mla}, focal_length=f_mm, aperture=g["aperture"])
    surfaces = assemble_surfaces(
        params["elements"],
        float(params.get("lens_z_start", 3.0)),
        mla=mla,
        dies=dies,
        blockers=params.get("blockers"),
    )
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
    use_warp = bool(params.get("use_warp", True))
    # Cylinders / horizontal baffles are CPU-traced (Warp path is Z-plane only)
    if use_warp and blockers_need_cpu(surfaces):
        use_warp = False

    active = [d for d in dies if d.enabled and d.flux > 0]
    paths: List[RayPath] = []
    if not active or total_rays < 1:
        stats = _empty_stats()
        return SimResult(imap, paths, stats, dies, surfaces)

    total_f = sum(d.flux for d in active)
    power_per = total_f / total_rays
    launched = hit = 0
    n_tir = n_reflect = n_backward = n_miss = n_absorb = 0
    batch = max(50, total_rays // 30)
    backend = "cpu"

    # ── Optional NVIDIA Warp acceleration for the irradiance map ──────────
    # Display paths (side-view) always use the pure-Python tracer so history
    # and event labels remain exact. The bulk Monte-Carlo deposit runs on
    # GPU when Warp + CUDA are available (or on Warp's CPU backend).
    warp_grid = None
    warp_stats = None
    if use_warp and total_rays >= 2000:
        try:
            from warp_backend import try_accelerate, warp_available, warp_device_info
            if warp_available() or True:  # allow Warp CPU backend too
                def _prog(f):
                    if progress_cb:
                        progress_cb(0.05 + 0.85 * f)
                warp_grid, warp_stats = try_accelerate(
                    params, dies, surfaces, progress_cb=_prog
                )
                if warp_grid is not None:
                    backend = warp_stats.get("backend", "warp")
                    # Inject Warp grid into IrradianceMap.
                    # Warp kernel deposits with iy increasing with +Y; IrradianceMap
                    # stores row 0 at +Y (origin=upper for imshow). Flip vertically.
                    import numpy as _np
                    g = _np.asarray(warp_grid, dtype=float)
                    if g.ndim == 2 and g.shape == (imap.ny, imap.nx):
                        g = _np.flipud(g)
                    flat = g.ravel()
                    if len(flat) == len(imap.bins):
                        imap.bins = list(flat)
                        imap.total_power = float(flat.sum())
                        imap.hit_count = int(warp_stats.get("hit", 0))
                    launched = int(warp_stats.get("launched", total_rays))
                    hit = int(warp_stats.get("hit", 0))
        except Exception as exc:
            # Keep going with CPU; print once for diagnostics
            print(f"[OptiFlux] Warp path skipped: {exc}")

    # Always run a modest Python batch so side-view paths & TIR counters stay accurate
    n_python = max_display * 3 if warp_grid is not None else total_rays
    n_python = min(n_python, total_rays)
    for i in range(n_python):
        r = random.random() * total_f
        die = active[0]
        for dd in active:
            r -= dd.flux
            if r <= 0:
                die = dd
                break
        o, d, pwr, wl = die.spawn_ray(power_per if warp_grid is None else total_f / n_python)
        store = len(paths) < max_display and (
            random.random() < (max_display / max(n_python, 1)) * 1.4 or len(paths) < 40
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
        if warp_grid is None:
            launched += 1
        if path is not None:
            if path.terminated == "tir_absorb":
                n_tir += 1
            elif path.terminated == "absorb":
                n_absorb += 1
            elif path.terminated == "backward":
                n_backward += 1
            elif path.terminated == "miss":
                n_miss += 1
            n_reflect += path.n_reflections
        if warp_grid is None and ok and pt is not None:
            hit += 1
            imap.deposit(pt[0], pt[1], pwr_out)
        if store and path is not None and len(path.history) >= 2:
            paths.append(path)
        if progress_cb and warp_grid is None and (i % batch == 0 or i == n_python - 1):
            progress_cb((i + 1) / n_python)
        elif progress_cb and warp_grid is not None and i == n_python - 1:
            progress_cb(1.0)

    if warp_grid is not None and launched == 0:
        launched = total_rays

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

    # Collection = power inside the rectangular FOV / source power.
    # Hits anywhere else on the infinite target plane (including the
    # zoomed-out wall) stay in plane_power / missed_power, not collection.
    plane_power = float(imap.total_power) + float(getattr(imap, "missed_power", 0.0) or 0.0)
    fov_power = float(fov.get("power_in", 0.0) or 0.0)
    stats = {
        "launched": launched,
        "hit": hit,
        "collection": fov_power / total_f if total_f > 0 else 0.0,
        "rms": imap.rms_radius(),
        "ee50": imap.encircled_radius(0.5),
        "ee86": imap.encircled_radius(0.86),
        "peak_e": imap.max_irradiance(),
        "centroid": (cx, cy),
        "fov": fov,
        "source_power": total_f,
        "map_power": imap.total_power,
        "plane_power": plane_power,
        "missed_power": float(getattr(imap, "missed_power", 0.0) or 0.0),
        "efl": efl,
        "n_dies": len(active),
        "n_surfaces": len(surfaces),
        "n_tir_absorb": n_tir,
        "n_absorb": n_absorb,
        "n_reflections": n_reflect,
        "n_backward": n_backward,
        "n_miss": n_miss,
        "backend": backend,
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
            "edge_sharpness": 0.0,
        },
        "source_power": 0.0,
        "map_power": 0.0,
        "efl": float("nan"),
        "n_dies": 0,
        "n_surfaces": 0,
        "n_tir_absorb": 0,
        "n_absorb": 0,
        "n_reflections": 0,
        "n_backward": 0,
        "n_miss": 0,
    }


MAX_ELEMENTS = 8
MAX_EXTRA_LENSES = 8  # optimizer may add this many; stack still caps at MAX_ELEMENTS
SOURCE_DIE_MIN_MM = 0.1
SOURCE_DIE_MAX_MM = 50.0  # large COB / flood LED modules (e.g. 35×35 mm)
LENS_Z_MIN_MM = 0.2
LENS_Z_MAX_MM = 1000.0  # first vertex; must reach mid-throw of long benches
LENS_SEMI_MAX_MM = 80.0
AIR_AFTER_MAX_MM = 250.0
FOV_SIZE_MAX_MM = 500.0


# Canonical starter optic — Element 1 and every unused slot share this geometry
# so enabling/adding lenses one-by-one always begins from the same size & radii.
DEFAULT_ELEMENT: Dict[str, Any] = {
    "enabled": False,
    "R1": 40.0,
    "R2": -50.0,
    "thickness": 6.0,
    "air_after": 2.0,
    "aperture": 10.0,
    "material": "FORMLABS_CLEAR",
    "shape_id": "biconvex",
    "surface_mode": "rotational",
    "k1": 0.0,
    "k2": 0.0,
    "A4_1": 0.0,
    "A4_2": 0.0,
    "R1y": None,
    "R2y": None,
    "aperture_y": None,
    "circular_lock": True,
}


def apply_circular_outline(element: Dict[str, Any], locked: Optional[bool] = None) -> Dict[str, Any]:
    """
    Force a round clear aperture so a circular tube can hold the lens.

    When locked, ``aperture_y`` is cleared (engine then uses aperture for both
    meridians). Optical surface mode is unchanged — only the outline is round.
    """
    e = dict(element)
    lock = bool(e.get("circular_lock", True) if locked is None else locked)
    e["circular_lock"] = lock
    if lock:
        e["aperture_y"] = None
    return e


def blank_element(
    *,
    enabled: bool = False,
    material: Optional[str] = None,
    shape_id: Optional[str] = None,
    surface_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Default lens-stack slot — same size, radii, thickness, and material as Element 1.
    Only ``enabled`` differs (False until the user turns the slot on).
    """
    e = dict(DEFAULT_ELEMENT)
    e["enabled"] = bool(enabled)
    if material is not None:
        e["material"] = material
    if shape_id is not None:
        e["shape_id"] = shape_id
    if surface_mode is not None:
        e["surface_mode"] = surface_mode
    return e


def pad_elements(elements: List[Dict[str, Any]], n: int = MAX_ELEMENTS) -> List[Dict[str, Any]]:
    """Ensure the stack has exactly ``n`` element slots (all default geometry)."""
    out = [dict(e) for e in (elements or [])]
    while len(out) < n:
        out.append(blank_element())
    return out[:n]


_COPY_ELEMENT_KEYS = (
    "R1",
    "R2",
    "R1y",
    "R2y",
    "thickness",
    "air_after",
    "aperture",
    "aperture_y",
    "circular_lock",
    "material",
    "shape_id",
    "surface_mode",
    "mode_s1",
    "mode_s2",
    "k1",
    "k2",
    "k1y",
    "k2y",
    "A4_1",
    "A4_2",
    "A4_1y",
    "A4_2y",
)


def copy_element(elements: List[Dict[str, Any]], src: int, dst: int) -> List[Dict[str, Any]]:
    """
    Return a new stack where slot ``dst`` is a copy of slot ``src``.

    The destination is enabled. Axial placement is not stored on the element;
    it follows the stack: dest sits after the previous *enabled* element's
    ``air_after`` (typically the source's air gap when dest is the next slot).
    """
    out = [dict(e) for e in (elements or [])]
    if src < 0 or dst < 0 or src >= len(out) or dst >= len(out):
        raise IndexError(f"copy_element src={src} dst={dst} n={len(out)}")
    if src == dst:
        return out
    src_e = out[src]
    dest = dict(out[dst])
    for key in _COPY_ELEMENT_KEYS:
        if key in src_e:
            val = src_e[key]
            dest[key] = dict(val) if isinstance(val, dict) else val
        elif key in dest:
            # Drop dest-only optional fields the source does not have
            if key in ("R1y", "R2y", "aperture_y", "k1y", "k2y", "A4_1y", "A4_2y"):
                dest[key] = src_e.get(key)
    dest["enabled"] = True
    out[dst] = dest
    return out


def default_params() -> Dict[str, Any]:
    return {
        "source": {
            "mode": "single",
            "rows": 1,
            "cols": 1,
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
            # Element 1 only enabled; remaining slots match the same starter optic
            {**dict(DEFAULT_ELEMENT), "enabled": True},
            *[blank_element() for _ in range(MAX_ELEMENTS - 1)],
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
        "cad_flange_radial_mm": 2.0,
        "cad_flange_thickness_mm": 1.5,
        "cad_export_halves": False,
        "cad_polish_tool_dia_mm": 10.0,
        "cad_polish_stepover_mm": 0.3,
        "cad_polish_allowance_mm": 0.0,
        "cad_polish_feed_mm_min": 100.0,
        "cad_polish_retract_mm": 5.0,
        "cad_polish_rpm": 5000.0,
        "cad_polish_revs": 2.0,
        "cad_polish_surface": "front",
        "cad_polish_strategy": "helical",
        "cad_polish_x_origin": "cut",
        "cad_polish_y_offset_mm": 0.0,
        "cad_lap_surface": "front",
        "cad_lap_wall_mm": 6.0,
        "map_res": 96,
        "total_rays": 6000,
        "display_rays": 300,
        "cad_max_edge_mm": 0.25,
        "cad_max_angle_deg": 2.0,
        "mla": {
            "enabled": False,
            "fill_factor": 0.88,
            "lenslet_aperture": 0.0,  # 0 = auto from COB pitch
            "export_plate": True,
            "scale_to_pitch": True,  # scale Element 1 R/t to die size (real micro-lenses)
            "aim_to_fov": True,  # each channel steers toward FOV center
            "aim_strength": 1.0,  # 0 = none, 1 = full geometric aim
        },
        # Opaque absorbing panels / aperture stops (enclosure simulation)
        "blockers": [],
    }
