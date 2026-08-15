"""
Export optical lens solids to STL and STEP.

Units: millimeters (mm) throughout — encoded in STEP header and STL is unitless
but comments/README state mm (standard for optics CAM).

Surfaces use the same aspheric sag as the ray tracer:
  z(r) = c r²/(1+sqrt(1-(1+k)c²r²)) + A4 r⁴

Exports:
  - Single centered singlet
  - Micro-lens array (MLA) matched to COB die positions on a substrate plate
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Dict, Any

import numpy as np

Vec3 = Tuple[float, float, float]


def sag_z(r: float, radius: float, k: float = 0.0, a4: float = 0.0) -> Optional[float]:
    if not math.isfinite(radius) or abs(radius) < 1e-12:
        z = 0.0
    else:
        c = 1.0 / radius
        r2 = r * r
        disc = 1.0 - (1.0 + k) * c * c * r2
        if disc < 0:
            return None
        z = (c * r2) / (1.0 + math.sqrt(max(0.0, disc)))
    z += a4 * (r ** 4)
    return z


def sag_xy(
    x: float,
    y: float,
    Rx: float,
    Ry: float,
    kx: float = 0.0,
    ky: float = 0.0,
    a4x: float = 0.0,
    a4y: float = 0.0,
    mode: str = "rotational",
) -> Optional[float]:
    """Match engine biconic / cylinder sag (mm)."""
    def c_of(R):
        return 0.0 if (not math.isfinite(R) or abs(R) < 1e-12) else 1.0 / R

    if mode == "cylinder_x":
        cx, cy, kx, ky, a4x, a4y = c_of(Rx), 0.0, kx, 0.0, a4x, 0.0
    elif mode == "cylinder_y":
        cx, cy, kx, ky, a4x, a4y = 0.0, c_of(Ry), 0.0, ky, 0.0, a4y
    elif mode == "biconic":
        cx, cy = c_of(Rx), c_of(Ry)
    else:
        return sag_z(math.hypot(x, y), Rx, kx, a4x)

    x2, y2 = x * x, y * y
    disc = 1.0 - (1.0 + kx) * cx * cx * x2 - (1.0 + ky) * cy * cy * y2
    if disc < 0:
        return None
    num = cx * x2 + cy * y2
    z = num / (1.0 + math.sqrt(max(0.0, disc))) if abs(num) > 1e-18 or abs(cx) + abs(cy) > 0 else 0.0
    z += a4x * x2 * x2 + a4y * y2 * y2
    return z


@dataclass
class LensSpec:
    """One singlet (possibly decentered for MLA). Supports biconic / cylindrical."""
    R1: float
    R2: float
    thickness: float
    aperture: float  # semi-diameter clear aperture X (mm)
    k1: float = 0.0
    k2: float = 0.0
    A4_1: float = 0.0
    A4_2: float = 0.0
    x0: float = 0.0
    y0: float = 0.0
    z_front: float = 0.0  # world Z of front vertex
    R1y: Optional[float] = None
    R2y: Optional[float] = None
    mode: str = "rotational"
    aperture_y: Optional[float] = None


@dataclass
class Mesh:
    vertices: np.ndarray  # (N, 3) float64, mm
    faces: np.ndarray     # (M, 3) int32

    def merge(self, other: "Mesh") -> "Mesh":
        if other.vertices.size == 0:
            return self
        if self.vertices.size == 0:
            return other
        off = len(self.vertices)
        return Mesh(
            vertices=np.vstack([self.vertices, other.vertices]),
            faces=np.vstack([self.faces, other.faces + off]),
        )

    def translate(self, dx: float, dy: float, dz: float) -> "Mesh":
        v = self.vertices.copy()
        v[:, 0] += dx
        v[:, 1] += dy
        v[:, 2] += dz
        return Mesh(v, self.faces.copy())


# Mesh density caps so a tiny tolerance on a large lens cannot explode RAM.
CAD_N_RADIAL_MAX = 1024
CAD_N_THETA_MAX = 2048
CAD_DEFAULT_MAX_EDGE_MM = 0.25
CAD_DEFAULT_MAX_ANGLE_DEG = 2.0


def tessellation_from_tolerance(
    aperture: float,
    radii: Optional[Sequence[Optional[float]]] = None,
    *,
    max_edge_mm: float = CAD_DEFAULT_MAX_EDGE_MM,
    max_angle_deg: float = CAD_DEFAULT_MAX_ANGLE_DEG,
    n_radial_min: int = 8,
    n_theta_min: int = 24,
    n_radial_max: int = CAD_N_RADIAL_MAX,
    n_theta_max: int = CAD_N_THETA_MAX,
) -> Tuple[int, int]:
    """
    Convert print-oriented tessellation tolerances into polar-grid counts.

    *max_edge_mm*
        Maximum vertex-to-vertex spacing along the clear aperture (mm).
        Smaller → finer mesh, fewer visible print facets.
    *max_angle_deg*
        Maximum central / facet angle (degrees) on curved surfaces.
        Ignored for plano (R ≈ 0) so a flat face is not over-tessellated.
    """
    ap = max(float(aperture), 1e-6)
    edge = max(float(max_edge_mm), 1e-4)
    ang = max(float(max_angle_deg), 0.05)
    d_rad = math.radians(ang)

    n_r = int(math.ceil(ap / edge))
    n_t = int(math.ceil(2.0 * math.pi * ap / edge))

    for R in radii or []:
        if R is None:
            continue
        try:
            Rf = abs(float(R))
        except (TypeError, ValueError):
            continue
        if Rf < 1e-9:
            continue
        # Meridian span from vertex to rim on a sphere of radius |R|.
        beta = math.asin(min(0.999999, ap / Rf))
        n_r = max(n_r, int(math.ceil(beta / d_rad)))
        # Azimuthal normal change at the rim ≈ Δθ · sin(β).
        sin_b = max(math.sin(beta), 1e-6)
        n_t = max(n_t, int(math.ceil(2.0 * math.pi * sin_b / d_rad)))

    n_r = min(int(n_radial_max), max(int(n_radial_min), n_r))
    n_t = min(int(n_theta_max), max(int(n_theta_min), n_t))
    if n_t % 2:
        n_t = min(int(n_theta_max), n_t + 1)
    return n_r, n_t


def tessellation_for_specs(
    specs: Sequence[LensSpec],
    *,
    max_edge_mm: float = CAD_DEFAULT_MAX_EDGE_MM,
    max_angle_deg: float = CAD_DEFAULT_MAX_ANGLE_DEG,
) -> Tuple[int, int]:
    """Finest (n_radial, n_theta) required by any spec in the export set."""
    if not specs:
        return tessellation_from_tolerance(
            10.0, [], max_edge_mm=max_edge_mm, max_angle_deg=max_angle_deg
        )
    n_r = 8
    n_t = 24
    for s in specs:
        ap = max(float(s.aperture), float(s.aperture_y or 0.0), 1e-6)
        radii = [s.R1, s.R2, s.R1y, s.R2y]
        cr, ct = tessellation_from_tolerance(
            ap, radii, max_edge_mm=max_edge_mm, max_angle_deg=max_angle_deg
        )
        n_r = max(n_r, cr)
        n_t = max(n_t, ct)
    return n_r, n_t


def _polar_grid(semi: float, n_radial: int, n_theta: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X, Y, linear index grid including center. Faces via rings."""
    # Rings: 0 = center, 1..n_radial
    n_r = max(4, n_radial)
    n_t = max(12, n_theta)
    # vertices: 1 center + n_r * n_t
    verts = []
    verts.append((0.0, 0.0))  # local
    for ir in range(1, n_r + 1):
        r = semi * (ir / n_r)
        for it in range(n_t):
            th = 2.0 * math.pi * it / n_t
            verts.append((r * math.cos(th), r * math.sin(th)))
    return np.array(verts, dtype=np.float64), n_r, n_t


def _surface_points(
    semi: float,
    radius: float,
    k: float,
    a4: float,
    z_vertex: float,
    n_radial: int,
    n_theta: int,
    x0: float = 0.0,
    y0: float = 0.0,
    radius_y: Optional[float] = None,
    k_y: float = 0.0,
    a4_y: float = 0.0,
    mode: str = "rotational",
    semi_y: Optional[float] = None,
) -> Tuple[np.ndarray, int, int]:
    """Polar sample; for elliptical apertures scale Y by semi_y/semi."""
    local, n_r, n_t = _polar_grid(semi, n_radial, n_theta)
    sy = float(semi_y) if semi_y and semi_y > 0 else semi
    scale_y = sy / max(semi, 1e-9)
    Ry = radius if radius_y is None else radius_y
    pts = np.zeros((len(local), 3), dtype=np.float64)
    for i, (x, y) in enumerate(local):
        yy = y * scale_y
        s = sag_xy(x, yy, radius, Ry, k, k_y, a4, a4_y, mode)
        if s is None:
            s = 0.0
        pts[i, 0] = x + x0
        pts[i, 1] = yy + y0
        pts[i, 2] = z_vertex + s
    return pts, n_r, n_t


def _disk_faces(n_r: int, n_t: int, flip: bool = False) -> np.ndarray:
    """Triangulate polar grid: center + rings."""
    faces = []
    # center fan to first ring: verts 1..n_t
    for it in range(n_t):
        a, b = 1 + it, 1 + (it + 1) % n_t
        if flip:
            faces.append((0, b, a))
        else:
            faces.append((0, a, b))
    # rings
    for ir in range(1, n_r):
        base = 1 + (ir - 1) * n_t
        nxt = 1 + ir * n_t
        for it in range(n_t):
            i0 = base + it
            i1 = base + (it + 1) % n_t
            j0 = nxt + it
            j1 = nxt + (it + 1) % n_t
            if flip:
                faces.append((i0, j1, j0))
                faces.append((i0, i1, j1))
            else:
                faces.append((i0, j0, j1))
                faces.append((i0, j1, i1))
    return np.array(faces, dtype=np.int32)


def _edge_faces(n_t: int, front_rim_start: int, back_rim_start: int, flip: bool = False) -> np.ndarray:
    faces = []
    for it in range(n_t):
        f0 = front_rim_start + it
        f1 = front_rim_start + (it + 1) % n_t
        b0 = back_rim_start + it
        b1 = back_rim_start + (it + 1) % n_t
        if flip:
            faces.append((f0, b0, b1))
            faces.append((f0, b1, f1))
        else:
            faces.append((f0, b1, b0))
            faces.append((f0, f1, b1))
    return np.array(faces, dtype=np.int32)


def _circle_ring(
    radius: float,
    z: float,
    n_t: int,
    x0: float = 0.0,
    y0: float = 0.0,
) -> np.ndarray:
    pts = np.zeros((n_t, 3), dtype=np.float64)
    for it in range(n_t):
        th = 2.0 * math.pi * it / n_t
        pts[it, 0] = x0 + radius * math.cos(th)
        pts[it, 1] = y0 + radius * math.sin(th)
        pts[it, 2] = z
    return pts


def mesh_singlet(
    spec: LensSpec,
    n_radial: int = 48,
    n_theta: int = 96,
    flange_radial_mm: float = 0.0,
    flange_thickness_mm: float = 0.0,
) -> Mesh:
    """
    Closed solid mesh of a decentered singlet.
    Front surface outward normals ≈ −Z, rear ≈ +Z, rim radial.
    """
    ap = max(1e-3, float(spec.aperture))
    z1 = float(spec.z_front)
    # rear vertex
    z2 = z1 + float(spec.thickness)

    ap_y = spec.aperture_y if spec.aperture_y else ap
    mode = spec.mode or "rotational"
    R1y = spec.R1y if spec.R1y is not None else spec.R1
    R2y = spec.R2y if spec.R2y is not None else spec.R2
    # Ensure positive edge thickness: rear rim Z should be > front rim Z for solid
    s1 = sag_xy(ap, 0.0, spec.R1, R1y, spec.k1, 0.0, spec.A4_1, 0.0, mode) or 0.0
    s2 = sag_xy(ap, 0.0, spec.R2, R2y, spec.k2, 0.0, spec.A4_2, 0.0, mode) or 0.0
    z_front_rim = z1 + s1
    z_rear_rim = z2 + s2
    if z_rear_rim <= z_front_rim + 0.05:
        z2 += (z_front_rim + 0.05) - z_rear_rim + 0.1
        z_rear_rim = z2 + s2

    front, n_r, n_t = _surface_points(
        ap, spec.R1, spec.k1, spec.A4_1, z1, n_radial, n_theta, spec.x0, spec.y0,
        radius_y=R1y, mode=mode, semi_y=ap_y,
    )
    back, _, _ = _surface_points(
        ap, spec.R2, spec.k2, spec.A4_2, z2, n_radial, n_theta, spec.x0, spec.y0,
        radius_y=R2y, mode=mode, semi_y=ap_y,
    )

    # Merge vertices: front then back
    verts = np.vstack([front, back])
    n_front = len(front)

    # Front faces: outward toward −Z → flip winding relative to +Z view
    f_front = _disk_faces(n_r, n_t, flip=True)
    # Back faces: outward +Z
    f_back = _disk_faces(n_r, n_t, flip=False) + n_front

    # Rim: front outer ring indices and back outer ring
    front_rim = 1 + (n_r - 1) * n_t  # start index of outer ring on front
    back_rim = n_front + 1 + (n_r - 1) * n_t

    face_blocks = [f_front, f_back]
    fr = max(0.0, float(flange_radial_mm or 0.0))
    ft = max(0.0, float(flange_thickness_mm or 0.0))
    if fr < 1e-6 or ft < 1e-6:
        face_blocks.append(_edge_faces(n_t, front_rim, back_rim, flip=False))
    else:
        inner_r = _clear_semi_mm(ap, ap_y)
        outer_r = inner_r + fr
        z_c = z1 + float(spec.thickness) * 0.5
        z_ff = z_c - 0.5 * ft
        z_rf = z_c + 0.5 * ft
        x0, y0 = float(spec.x0), float(spec.y0)
        ca_ff = _circle_ring(inner_r, z_ff, n_t, x0, y0)
        ca_rf = _circle_ring(inner_r, z_rf, n_t, x0, y0)
        od_ff = _circle_ring(outer_r, z_ff, n_t, x0, y0)
        od_rf = _circle_ring(outer_r, z_rf, n_t, x0, y0)
        i_ca_ff = len(verts)
        verts = np.vstack([verts, ca_ff])
        i_ca_rf = len(verts)
        verts = np.vstack([verts, ca_rf])
        i_od_ff = len(verts)
        verts = np.vstack([verts, od_ff])
        i_od_rf = len(verts)
        verts = np.vstack([verts, od_rf])
        # Optical rim → flange inner rings → outer cylinder
        face_blocks.append(_edge_faces(n_t, front_rim, i_ca_ff, flip=False))
        face_blocks.append(_edge_faces(n_t, i_ca_ff, i_od_ff, flip=False))
        face_blocks.append(_edge_faces(n_t, i_od_ff, i_od_rf, flip=False))
        face_blocks.append(_edge_faces(n_t, i_od_rf, i_ca_rf, flip=False))
        face_blocks.append(_edge_faces(n_t, i_ca_rf, back_rim, flip=False))

    faces = np.vstack(face_blocks)
    return Mesh(verts, faces)


def mesh_substrate_plate(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z0: float,
    z1: float,
    margin: float = 0.5,
) -> Mesh:
    """Rectangular box plate (mm) for MLA carrier."""
    x0, x1 = x_min - margin, x_max + margin
    y0, y1 = y_min - margin, y_max + margin
    # 8 corners
    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2], [0, 2, 3],  # bottom −Z
            [4, 6, 5], [4, 7, 6],  # top +Z
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0],
        ],
        dtype=np.int32,
    )
    return Mesh(corners, faces)


def mesh_mla(
    lenslets: Sequence[LensSpec],
    include_plate: bool = True,
    plate_extra_z: float = 0.0,
    n_radial: int = 24,
    n_theta: int = 48,
    grid_pitch: Optional[float] = None,
) -> Mesh:
    """
    Monolithic MLA: one solid plate whose front/rear faces carry Element-1
    lenslet sags at each die center (not separate cylinders on a slab).
    """
    if not lenslets:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32))
    # Dense enough that micro-dome curvature is obvious in CAD viewers
    spp = max(12, n_radial // 2)
    return mesh_mla_monolithic(
        list(lenslets),
        margin=0.2,
        samples_per_pitch=spp,
        include_skirt=include_plate,
        skirt_z=plate_extra_z,
    )


def mesh_mla_monolithic(
    lenslets: List[LensSpec],
    margin: float = 0.25,
    samples_per_pitch: int = 12,
    include_skirt: bool = False,
    skirt_z: float = 0.0,
) -> Mesh:
    """
    Single closed solid: rectangular footprint, front and rear height fields
    from nearest-lenslet sag (flat land between clear apertures at rim sag).
    """
    from mla_geometry import front_z_at, rear_z_at, land_sags

    xs = [s.x0 for s in lenslets]
    ys = [s.y0 for s in lenslets]
    ap = float(lenslets[0].aperture)
    xmin = min(xs) - ap - margin
    xmax = max(xs) + ap + margin
    ymin = min(ys) - ap - margin
    ymax = max(ys) + ap + margin

    # Grid resolution from pitch estimate
    if len(lenslets) >= 2:
        pitch = min(
            math.hypot(lenslets[i].x0 - lenslets[0].x0, lenslets[i].y0 - lenslets[0].y0)
            for i in range(1, len(lenslets))
            if math.hypot(lenslets[i].x0 - lenslets[0].x0, lenslets[i].y0 - lenslets[0].y0) > 1e-9
        ) if any(
            math.hypot(lenslets[i].x0 - lenslets[0].x0, lenslets[i].y0 - lenslets[0].y0) > 1e-9
            for i in range(1, len(lenslets))
        ) else 2 * ap
    else:
        pitch = 2 * ap
    pitch = max(pitch, 2 * ap, 0.5)
    dx = pitch / max(samples_per_pitch, 4)
    nx = max(8, int(math.ceil((xmax - xmin) / dx)) + 1)
    ny = max(8, int(math.ceil((ymax - ymin) / dx)) + 1)
    # Cap for export size
    nx = min(nx, 220)
    ny = min(ny, 220)

    land_f, land_r = land_sags(lenslets)

    # Sample front and rear grids
    xs_g = np.linspace(xmin, xmax, nx)
    ys_g = np.linspace(ymin, ymax, ny)
    zf = np.zeros((ny, nx), dtype=np.float64)
    zr = np.zeros((ny, nx), dtype=np.float64)
    for j, y in enumerate(ys_g):
        for i, x in enumerate(xs_g):
            zf[j, i] = front_z_at(float(x), float(y), lenslets, land_f)
            zr[j, i] = rear_z_at(float(x), float(y), lenslets, land_r)
            # Enforce positive local thickness
            if zr[j, i] < zf[j, i] + 0.08:
                zr[j, i] = zf[j, i] + 0.08

    if include_skirt and skirt_z > 0:
        zr = zr + skirt_z  # push rear further if skirt requested (thicker plate)

    # Vertices: front layer then rear layer, row-major
    n = nx * ny
    verts = np.zeros((2 * n, 3), dtype=np.float64)
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            verts[idx, 0] = xs_g[i]
            verts[idx, 1] = ys_g[j]
            verts[idx, 2] = zf[j, i]
            verts[n + idx, 0] = xs_g[i]
            verts[n + idx, 1] = ys_g[j]
            verts[n + idx, 2] = zr[j, i]

    faces: List[Tuple[int, int, int]] = []

    def add_quad(a, b, c, d, flip=False):
        # triangle a-b-c and a-c-d
        if flip:
            faces.append((a, c, b))
            faces.append((a, d, c))
        else:
            faces.append((a, b, c))
            faces.append((a, c, d))

    # Front surface (outward −Z → flip)
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            b = j * nx + i + 1
            c = (j + 1) * nx + i + 1
            d = (j + 1) * nx + i
            add_quad(a, b, c, d, flip=True)

    # Rear surface (outward +Z)
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = n + j * nx + i
            b = n + j * nx + i + 1
            c = n + (j + 1) * nx + i + 1
            d = n + (j + 1) * nx + i
            add_quad(a, b, c, d, flip=False)

    # Side walls (boundary of rectangle)
    # bottom j=0, top j=ny-1, left i=0, right i=nx-1
    for i in range(nx - 1):
        # bottom edge y=ymin
        a, b = i, i + 1
        add_quad(a, b, n + b, n + a, flip=False)
        # top edge
        a, b = (ny - 1) * nx + i, (ny - 1) * nx + i + 1
        add_quad(a, b, n + b, n + a, flip=True)
    for j in range(ny - 1):
        # left
        a, b = j * nx, (j + 1) * nx
        add_quad(a, b, n + b, n + a, flip=True)
        # right
        a, b = j * nx + (nx - 1), (j + 1) * nx + (nx - 1)
        add_quad(a, b, n + b, n + a, flip=False)

    return Mesh(verts, np.array(faces, dtype=np.int32))


def write_stl_binary(path: str | Path, mesh: Mesh, solid_name: str = "OptiFlux_Lens_mm") -> Path:
    """Binary STL. Coordinates in millimeters (STL itself has no unit field)."""
    path = Path(path)
    verts = mesh.vertices
    faces = mesh.faces
    # recompute normals
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    nrms = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(nrms, axis=1, keepdims=True)
    ln = np.maximum(ln, 1e-30)
    nrms = nrms / ln

    n = len(faces)
    header = f"OptiFlux lens solid; units=mm; {solid_name}".encode("ascii", errors="replace")
    header = header[:80].ljust(80, b"\0")

    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", n))
        for i in range(n):
            nx, ny, nz = nrms[i]
            f.write(struct.pack("<3f", float(nx), float(ny), float(nz)))
            for j in range(3):
                x, y, z = verts[faces[i, j]]
                f.write(struct.pack("<3f", float(x), float(y), float(z)))
            f.write(struct.pack("<H", 0))
    return path


def write_stl_ascii(path: str | Path, mesh: Mesh, solid_name: str = "OptiFlux_Lens_mm") -> Path:
    path = Path(path)
    verts = mesh.vertices
    faces = mesh.faces
    lines = [f"solid {solid_name} ; units = mm"]
    for face in faces:
        v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
        n = np.cross(v1 - v0, v2 - v0)
        ln = np.linalg.norm(n)
        if ln < 1e-30:
            n = np.array([0.0, 0.0, 1.0])
        else:
            n = n / ln
        lines.append(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}")
        lines.append("    outer loop")
        for idx in face:
            x, y, z = verts[idx]
            lines.append(f"      vertex {x:.6e} {y:.6e} {z:.6e}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {solid_name}")
    path.write_text("\n".join(lines), encoding="ascii")
    return path


def write_step_mesh(path: str | Path, mesh: Mesh, name: str = "OptiFlux_Lens") -> Path:
    """
    Write a STEP AP214 file with a faceted MANIFOLD_SOLID_BREP.
    Units: millimetre (SI_UNIT .MILLI. .METRE.).

    Each optical surface triangle uses the same aspheric sample points as the
    ray-tracer mesh — raise mesh density for smoother freeform surfaces.
    """
    path = Path(path)
    verts = mesh.vertices
    faces = mesh.faces
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)[:40]

    lines: List[str] = []
    eid = 1

    def add(s: str) -> int:
        nonlocal eid
        lines.append(f"#{eid}={s};")
        i = eid
        eid += 1
        return i

    lines.append("ISO-10303-21;")
    lines.append("HEADER;")
    lines.append("FILE_DESCRIPTION(('OptiFlux optical lens - units millimetre'),'2;1');")
    lines.append(
        f"FILE_NAME('{path.name.replace(chr(39), '')}','',('OptiFlux'),(''),"
        f"'OptiFlux','OptiFlux','');"
    )
    lines.append("FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));")
    lines.append("ENDSEC;")
    lines.append("DATA;")

    app_ctx = add("APPLICATION_CONTEXT('core data for automotive mechanical design processes')")
    add(
        f"APPLICATION_PROTOCOL_DEFINITION('international standard',"
        f"'automotive_design',2000,#{app_ctx})"
    )
    product_ctx = add(f"PRODUCT_CONTEXT('',#{app_ctx},'mechanical')")
    product = add(f"PRODUCT('{safe_name}','{safe_name}','optical lens solid mm',(#{product_ctx}))")
    pdf_ctx = add(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{app_ctx},'design')")
    pdf_form = add(f"PRODUCT_DEFINITION_FORMATION('',#{product})")
    pdf = add(f"PRODUCT_DEFINITION('design','',#{pdf_form},#{pdf_ctx})")
    pdf_shape = add(f"PRODUCT_DEFINITION_SHAPE('','',#{pdf})")

    si_unit = add("(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))")
    plane_angle = add("(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))")
    solid_angle = add("(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())")
    uncertainty = add(
        f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-6),#{si_unit},"
        f"'distance_accuracy_value','maximum gap')"
    )
    geom_ctx = add(
        f"(GEOMETRIC_REPRESENTATION_CONTEXT(3)"
        f"GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{uncertainty}))"
        f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{si_unit},#{plane_angle},#{solid_angle}))"
        f"REPRESENTATION_CONTEXT('Context1','3D mm'))"
    )

    pt_ids = []
    for x, y, z in verts:
        pt_ids.append(add(f"CARTESIAN_POINT('',({x:.8f},{y:.8f},{z:.8f}))"))

    face_ids = []
    for ia, ib, ic in faces:
        ia, ib, ic = int(ia), int(ib), int(ic)
        va, vb, vc = verts[ia], verts[ib], verts[ic]
        e1 = vb - va
        e2 = vc - va
        n = np.cross(e1, e2)
        ln = float(np.linalg.norm(n))
        if ln < 1e-18:
            continue
        n = n / ln
        el = float(np.linalg.norm(e1))
        if el < 1e-18:
            continue
        d1 = e1 / el
        pa, pb, pc = pt_ids[ia], pt_ids[ib], pt_ids[ic]
        loop = add(f"POLY_LOOP('',(#{pa},#{pb},#{pc}))")
        bound = add(f"FACE_OUTER_BOUND('',#{loop},.T.)")
        origin = add(f"CARTESIAN_POINT('',({va[0]:.8f},{va[1]:.8f},{va[2]:.8f}))")
        axis = add(f"DIRECTION('',({n[0]:.8f},{n[1]:.8f},{n[2]:.8f}))")
        refd = add(f"DIRECTION('',({d1[0]:.8f},{d1[1]:.8f},{d1[2]:.8f}))")
        place = add(f"AXIS2_PLACEMENT_3D('',#{origin},#{axis},#{refd})")
        plane = add(f"PLANE('',#{place})")
        face = add(f"ADVANCED_FACE('',(#{bound}),#{plane},.T.)")
        face_ids.append(face)

    if not face_ids:
        raise RuntimeError("No faces to export")

    # Limit STEP face count for huge MLAs — already meshed; warn via size
    face_list = ",".join(f"#{i}" for i in face_ids)
    shell = add(f"CLOSED_SHELL('',({face_list}))")
    solid = add(f"MANIFOLD_SOLID_BREP('{safe_name}',#{shell})")
    shape_rep = add(f"ADVANCED_BREP_SHAPE_REPRESENTATION('',(#{solid}),#{geom_ctx})")
    add(f"SHAPE_DEFINITION_REPRESENTATION(#{pdf_shape},#{shape_rep})")

    lines.append("ENDSEC;")
    lines.append("END-ISO-10303-21;")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_lens_specs_from_params(params: Dict[str, Any], dies: Optional[list] = None) -> Tuple[List[LensSpec], str]:
    """
    Build lens specs for export. MLA uses shared geometry helper so CAD matches
    the ray-tracer (Element 1 form scaled to die pitch).

    Non-MLA stacks return one ``LensSpec`` per enabled element, with ``z_front``
    advanced along the optical axis (same layout as the ray tracer).
    """
    elements = [e for e in params.get("elements", []) if e.get("enabled", True)]
    if not elements:
        return [], "empty"
    z0 = float(params.get("lens_z_start", 3.0))
    mla = params.get("mla", {}) or {}
    mla_on = bool(mla.get("enabled", False))

    if mla_on:
        from mla_geometry import build_mla_lens_specs

        specs, _meta = build_mla_lens_specs(params, dies=dies)
        return specs, "mla"

    specs: List[LensSpec] = []
    z = z0
    for e in elements:
        r1y = e.get("R1y", None)
        r2y = e.get("R2y", None)
        apy = e.get("aperture_y", None)
        thick = float(e["thickness"])
        specs.append(
            LensSpec(
                R1=float(e["R1"]),
                R2=float(e["R2"]),
                thickness=thick,
                aperture=float(e["aperture"]),
                k1=float(e.get("k1", 0.0)),
                k2=float(e.get("k2", 0.0)),
                A4_1=float(e.get("A4_1", 0.0)),
                A4_2=float(e.get("A4_2", 0.0)),
                x0=0.0,
                y0=0.0,
                z_front=z,
                R1y=float(r1y) if r1y is not None else None,
                R2y=float(r2y) if r2y is not None else None,
                mode=str(e.get("surface_mode", "rotational")),
                aperture_y=float(apy) if apy is not None else None,
            )
        )
        z += thick + float(e.get("air_after", 0.0))

    mode = "stack" if len(specs) > 1 else "singlet"
    return specs, mode


def write_step_multibody(path: str | Path, meshes: Sequence[Mesh], name: str = "OptiFlux_Optics") -> Path:
    """
    Multi-body STEP (one MANIFOLD_SOLID_BREP per mesh) — correct for MLA
    (separate lenslets + optional plate). Units: millimetre.
    """
    path = Path(path)
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)[:40]
    lines: List[str] = []
    eid = 1

    def add(s: str) -> int:
        nonlocal eid
        lines.append(f"#{eid}={s};")
        i = eid
        eid += 1
        return i

    lines.append("ISO-10303-21;")
    lines.append("HEADER;")
    lines.append("FILE_DESCRIPTION(('OptiFlux multi-body optics - units millimetre'),'2;1');")
    lines.append(
        f"FILE_NAME('{path.name.replace(chr(39), '')}','',('OptiFlux'),(''),"
        f"'OptiFlux','OptiFlux','');"
    )
    lines.append("FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));")
    lines.append("ENDSEC;")
    lines.append("DATA;")

    app_ctx = add("APPLICATION_CONTEXT('core data for automotive mechanical design processes')")
    add(
        f"APPLICATION_PROTOCOL_DEFINITION('international standard',"
        f"'automotive_design',2000,#{app_ctx})"
    )
    product_ctx = add(f"PRODUCT_CONTEXT('',#{app_ctx},'mechanical')")
    product = add(f"PRODUCT('{safe}','{safe}','optical array mm',(#{product_ctx}))")
    pdf_ctx = add(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{app_ctx},'design')")
    pdf_form = add(f"PRODUCT_DEFINITION_FORMATION('',#{product})")
    pdf = add(f"PRODUCT_DEFINITION('design','',#{pdf_form},#{pdf_ctx})")
    pdf_shape = add(f"PRODUCT_DEFINITION_SHAPE('','',#{pdf})")

    si_unit = add("(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))")
    plane_angle = add("(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))")
    solid_angle = add("(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())")
    uncertainty = add(
        f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-6),#{si_unit},"
        f"'distance_accuracy_value','maximum gap')"
    )
    geom_ctx = add(
        f"(GEOMETRIC_REPRESENTATION_CONTEXT(3)"
        f"GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{uncertainty}))"
        f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{si_unit},#{plane_angle},#{solid_angle}))"
        f"REPRESENTATION_CONTEXT('Context1','3D mm'))"
    )

    solid_ids = []
    for mi, mesh in enumerate(meshes):
        if mesh.vertices.size == 0 or mesh.faces.size == 0:
            continue
        verts = mesh.vertices
        faces = mesh.faces
        pt_ids = []
        for x, y, z in verts:
            pt_ids.append(add(f"CARTESIAN_POINT('',({x:.8f},{y:.8f},{z:.8f}))"))
        face_ids = []
        for ia, ib, ic in faces:
            ia, ib, ic = int(ia), int(ib), int(ic)
            va, vb, vc = verts[ia], verts[ib], verts[ic]
            e1 = vb - va
            e2 = vc - va
            n = np.cross(e1, e2)
            ln = float(np.linalg.norm(n))
            if ln < 1e-18:
                continue
            n = n / ln
            el = float(np.linalg.norm(e1))
            if el < 1e-18:
                continue
            d1 = e1 / el
            pa, pb, pc = pt_ids[ia], pt_ids[ib], pt_ids[ic]
            loop = add(f"POLY_LOOP('',(#{pa},#{pb},#{pc}))")
            bound = add(f"FACE_OUTER_BOUND('',#{loop},.T.)")
            origin = add(f"CARTESIAN_POINT('',({va[0]:.8f},{va[1]:.8f},{va[2]:.8f}))")
            axis = add(f"DIRECTION('',({n[0]:.8f},{n[1]:.8f},{n[2]:.8f}))")
            refd = add(f"DIRECTION('',({d1[0]:.8f},{d1[1]:.8f},{d1[2]:.8f}))")
            place = add(f"AXIS2_PLACEMENT_3D('',#{origin},#{axis},#{refd})")
            plane = add(f"PLANE('',#{place})")
            face = add(f"ADVANCED_FACE('',(#{bound}),#{plane},.T.)")
            face_ids.append(face)
        if not face_ids:
            continue
        face_list = ",".join(f"#{i}" for i in face_ids)
        shell = add(f"CLOSED_SHELL('',({face_list}))")
        solid = add(f"MANIFOLD_SOLID_BREP('body_{mi}',#{shell})")
        solid_ids.append(solid)

    if not solid_ids:
        raise RuntimeError("No solids to export")
    solid_list = ",".join(f"#{i}" for i in solid_ids)
    shape_rep = add(f"ADVANCED_BREP_SHAPE_REPRESENTATION('',({solid_list}),#{geom_ctx})")
    add(f"SHAPE_DEFINITION_REPRESENTATION(#{pdf_shape},#{shape_rep})")
    lines.append("ENDSEC;")
    lines.append("END-ISO-10303-21;")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _clear_semi_mm(aperture: float, aperture_y: Optional[float] = None) -> float:
    ax = max(float(aperture), 1e-6)
    if aperture_y is None:
        return ax
    try:
        ay = float(aperture_y)
    except (TypeError, ValueError):
        return ax
    if ay <= 0:
        return ax
    return max(ax, ay)


def flange_od_mm(
    aperture: float,
    flange_radial_mm: float,
    aperture_y: Optional[float] = None,
) -> float:
    """Outside diameter of the printed flange (mm)."""
    semi = _clear_semi_mm(aperture, aperture_y)
    return 2.0 * (semi + max(0.0, float(flange_radial_mm or 0.0)))


def _groove_center_z(z_front: float, thickness: float) -> float:
    return float(z_front) + 0.5 * float(thickness)


def tube_layout(
    params: Dict[str, Any],
    *,
    flange_radial_mm: float = 0.0,
    flange_thickness_mm: float = 0.0,
) -> Dict[str, Any]:
    """
    Seat / groove numbers for a printed lens tube (mm).

    Groove center is the vertex mid-plane of each enabled element
    (z_front + thickness/2). Flange front face = groove − thickness/2.
    Focus plane is the design Target Z.
    """
    from engine import default_params

    p = params if params else default_params()
    specs, mode = build_lens_specs_from_params(p)
    fr = max(0.0, float(flange_radial_mm or 0.0))
    ft = max(0.0, float(flange_thickness_mm or 0.0))
    src_z = float((p.get("source") or {}).get("source_z", 0.0))
    target_z = float(p.get("target_z", 80.0))

    seats: List[Dict[str, Any]] = []
    if mode != "mla":
        for i, s in enumerate(specs):
            apy = s.aperture_y
            od = flange_od_mm(s.aperture, fr, apy)
            gz = _groove_center_z(s.z_front, s.thickness)
            seats.append(
                {
                    "index": i + 1,
                    "od_mm": od,
                    "clear_diameter_mm": 2.0 * _clear_semi_mm(s.aperture, apy),
                    "flange_radial_mm": fr,
                    "flange_thickness_mm": ft,
                    "groove_center_z_mm": gz,
                    "z_front_vertex_mm": float(s.z_front),
                    "thickness_mm": float(s.thickness),
                }
            )

    spacings = [
        seats[i + 1]["groove_center_z_mm"] - seats[i]["groove_center_z_mm"]
        for i in range(max(0, len(seats) - 1))
    ]
    if seats:
        tube_front = seats[0]["groove_center_z_mm"] - 0.5 * ft
        last_g = seats[-1]["groove_center_z_mm"]
    else:
        tube_front = float(p.get("lens_z_start", 3.0))
        last_g = tube_front

    return {
        "mode": mode,
        "units": "mm",
        "source_z_mm": src_z,
        "target_z_mm": target_z,
        "tube_front_z_mm": tube_front,
        "source_to_tube_front_mm": tube_front - src_z,
        "last_groove_to_focus_mm": target_z - last_g,
        "flange_radial_mm": fr,
        "flange_thickness_mm": ft,
        "seats": seats,
        "groove_center_to_center_mm": spacings,
    }


def format_tube_notes(layout: Dict[str, Any]) -> str:
    """Human-readable lens-tube dimensions for CAD."""
    seats = list(layout.get("seats") or [])
    lines = [
        "OptiFlux lens-tube dimensions",
        "All values in millimetres (mm).",
        "Flanges are on the exported STL only — not in the ray-trace.",
        "Groove center = lens vertex mid-plane (z_front + thickness/2).",
        "Tube front = front face of the first flange.",
        "Focus = design Target Z (illumination / FOV plane).",
        "",
        f"Source Z                         {float(layout.get('source_z_mm', 0.0)):8.3f}",
        f"Tube front Z                     {float(layout.get('tube_front_z_mm', 0.0)):8.3f}",
        f"Source to tube front             {float(layout.get('source_to_tube_front_mm', 0.0)):8.3f}",
        f"Target Z (focus plane)           {float(layout.get('target_z_mm', 0.0)):8.3f}",
        f"Last groove center to focus      {float(layout.get('last_groove_to_focus_mm', 0.0)):8.3f}",
        f"Flange radial width              {float(layout.get('flange_radial_mm', 0.0)):8.3f}",
        f"Flange thickness (axial)         {float(layout.get('flange_thickness_mm', 0.0)):8.3f}",
        "",
        f"{'E':>3}  {'OD':>8}  {'CA dia':>8}  {'Flange t':>8}  "
        f"{'Groove Z':>9}  {'t':>7}  {'Front Z':>8}",
    ]
    for s in seats:
        lines.append(
            f"{int(s['index']):>3}  {s['od_mm']:8.2f}  {s['clear_diameter_mm']:8.2f}  "
            f"{s['flange_thickness_mm']:8.2f}  {s['groove_center_z_mm']:9.3f}  "
            f"{s['thickness_mm']:7.3f}  {s['z_front_vertex_mm']:8.3f}"
        )
    spacings = list(layout.get("groove_center_to_center_mm") or [])
    if spacings:
        lines.append("")
        lines.append("Groove center-to-center (along +Z):")
        for i, d in enumerate(spacings):
            lines.append(f"  E{i + 1} → E{i + 2}                    {float(d):8.3f}")
    if not seats:
        lines.append("")
        lines.append("No enabled lens seats (MLA plate or empty stack).")
    lines.append("")
    return "\n".join(lines)


def write_tube_notes(path: Path | str, layout: Dict[str, Any]) -> Path:
    path = Path(path)
    path.write_text(format_tube_notes(layout), encoding="utf-8")
    return path


CNC_PROFILE_EXT = ".nc"


def flange_cut_z(spec: LensSpec) -> float:
    """Axial mid-plane of the print flange (build / glue face)."""
    return float(spec.z_front) + 0.5 * float(spec.thickness)


def half_stl_paths(path: Path | str, n_elements: int) -> List[Path]:
    """Companion filenames for front/rear print halves."""
    path = Path(path)
    if path.suffix.lower() != ".stl":
        path = path.with_suffix(".stl")
    n = max(1, int(n_elements))
    out: List[Path] = []
    if n == 1:
        out.append(path.with_name(f"{path.stem}_front.stl"))
        out.append(path.with_name(f"{path.stem}_rear.stl"))
        return out
    for i in range(n):
        out.append(path.with_name(f"{path.stem}_E{i + 1}_front.stl"))
        out.append(path.with_name(f"{path.stem}_E{i + 1}_rear.stl"))
    return out


def _interp_xyz(a: np.ndarray, b: np.ndarray, z_cut: float) -> np.ndarray:
    dz = float(b[2] - a[2])
    if abs(dz) < 1e-15:
        p = a.copy()
        p[2] = z_cut
        return p
    t = (float(z_cut) - float(a[2])) / dz
    t = min(1.0, max(0.0, t))
    return a + t * (b - a)


def _clip_mesh_keep(mesh: Mesh, z_cut: float, *, keep_front: bool) -> tuple:
    """
    Clip a closed mesh at z = z_cut.

    keep_front=True keeps z <= z_cut. Returns (verts, faces, cap_edge_pairs).
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if verts.size == 0 or faces.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.int32),
            [],
        )
    eps = 1e-9
    on_keep = np.zeros(len(verts), dtype=bool)
    if keep_front:
        on_keep = verts[:, 2] <= float(z_cut) + eps
    else:
        on_keep = verts[:, 2] >= float(z_cut) - eps

    new_verts: List[np.ndarray] = []
    remap = {}

    def add_orig(i: int) -> int:
        i = int(i)
        if i in remap:
            return remap[i]
        remap[i] = len(new_verts)
        new_verts.append(verts[i].copy())
        return remap[i]

    split_cache: Dict[Tuple[int, int], int] = {}

    def add_split(i: int, j: int) -> int:
        key = (i, j) if i < j else (j, i)
        if key in split_cache:
            return split_cache[key]
        p = _interp_xyz(verts[i], verts[j], z_cut)
        idx = len(new_verts)
        new_verts.append(p)
        split_cache[key] = idx
        return idx

    new_faces: List[Tuple[int, int, int]] = []
    cap_edges: List[Tuple[int, int]] = []

    for tri in faces:
        ia, ib, ic = int(tri[0]), int(tri[1]), int(tri[2])
        keep = [bool(on_keep[ia]), bool(on_keep[ib]), bool(on_keep[ic])]
        ids = [ia, ib, ic]
        n_keep = sum(keep)
        if n_keep == 3:
            new_faces.append((add_orig(ia), add_orig(ib), add_orig(ic)))
            continue
        if n_keep == 0:
            continue
        if n_keep == 2:
            # Quad → two tris; the cut edge is between the two split points
            # Walk edges to preserve winding.
            pts: List[int] = []
            cut: List[int] = []
            for k in range(3):
                i0, i1 = ids[k], ids[(k + 1) % 3]
                k0, k1 = keep[k], keep[(k + 1) % 3]
                if k0:
                    pts.append(add_orig(i0))
                if k0 != k1:
                    s = add_split(i0, i1)
                    pts.append(s)
                    cut.append(s)
            if len(pts) == 4:
                new_faces.append((pts[0], pts[1], pts[2]))
                new_faces.append((pts[0], pts[2], pts[3]))
            elif len(pts) == 3:
                new_faces.append((pts[0], pts[1], pts[2]))
            if len(cut) == 2:
                cap_edges.append((cut[0], cut[1]))
            continue
        # n_keep == 1: one triangle, cut edge between the two split points
        pts = []
        cut = []
        for k in range(3):
            i0, i1 = ids[k], ids[(k + 1) % 3]
            k0, k1 = keep[k], keep[(k + 1) % 3]
            if k0:
                pts.append(add_orig(i0))
            if k0 != k1:
                s = add_split(i0, i1)
                pts.append(s)
                cut.append(s)
        if len(pts) >= 3:
            new_faces.append((pts[0], pts[1], pts[2]))
        if len(cut) == 2:
            cap_edges.append((cut[0], cut[1]))

    if not new_verts:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.int32),
            [],
        )
    return np.vstack(new_verts), np.array(new_faces, dtype=np.int32), cap_edges


def _cap_loops_from_edges(
    verts: np.ndarray,
    edges: Sequence[Tuple[int, int]],
    *,
    flip: bool,
) -> np.ndarray:
    """Fan-triangulate closed intersection loops in the cut plane."""
    if not edges or len(verts) == 0:
        return np.zeros((0, 3), dtype=np.int32)
    # Build undirected adjacency
    adj: Dict[int, List[int]] = {}
    for a, b in edges:
        if a == b:
            continue
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    unused = set()
    for a, b in edges:
        unused.add((min(int(a), int(b)), max(int(a), int(b))))
    faces: List[Tuple[int, int, int]] = []
    visited_nodes = set()
    for start in list(adj.keys()):
        if start in visited_nodes:
            continue
        loop = [start]
        prev = None
        cur = start
        guard = 0
        while guard < len(adj) + 3:
            guard += 1
            nxts = [n for n in adj.get(cur, []) if n != prev]
            if not nxts:
                break
            # Prefer unused edge
            nxt = None
            for cand in nxts:
                e = (min(cur, cand), max(cur, cand))
                if e in unused:
                    nxt = cand
                    unused.discard(e)
                    break
            if nxt is None:
                nxt = nxts[0]
            if nxt == start:
                break
            if nxt in loop:
                break
            loop.append(nxt)
            prev, cur = cur, nxt
        if len(loop) < 3:
            continue
        for i in loop:
            visited_nodes.add(i)
        # Order by polar angle so the fan is a disk
        pts = verts[np.array(loop, dtype=np.int32)]
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        order = [loop[i] for i in np.argsort(ang)]
        # Fan from the first vertex (on the circle, not a new center — avoids
        # adding a vertex). For a near-circle this is a valid triangulation.
        a0 = order[0]
        for i in range(1, len(order) - 1):
            b, c = order[i], order[i + 1]
            if flip:
                faces.append((a0, c, b))
            else:
                faces.append((a0, b, c))
    if not faces:
        return np.zeros((0, 3), dtype=np.int32)
    return np.array(faces, dtype=np.int32)


def split_mesh_at_z(mesh: Mesh, z_cut: float) -> Tuple[Mesh, Mesh]:
    """
    Split a closed solid at z = z_cut and cap each half with a planar face.

    The cap is the resin-printer build / glue face (middle of the flange
    when z_cut is the flange mid-plane).
    """
    zc = float(z_cut)
    fv, ff, fe = _clip_mesh_keep(mesh, zc, keep_front=True)
    rv, rf, re = _clip_mesh_keep(mesh, zc, keep_front=False)
    # Front half cap faces +Z (toward the mate); rear half faces −Z.
    if len(fv) and len(fe):
        cap_f = _cap_loops_from_edges(fv, fe, flip=False)
        if cap_f.size:
            ff = np.vstack([ff, cap_f]) if len(ff) else cap_f
    if len(rv) and len(re):
        cap_r = _cap_loops_from_edges(rv, re, flip=True)
        if cap_r.size:
            rf = np.vstack([rf, cap_r]) if len(rf) else cap_r
    if fv.size == 0:
        fv = np.zeros((0, 3), dtype=np.float64)
        ff = np.zeros((0, 3), dtype=np.int32)
    if rv.size == 0:
        rv = np.zeros((0, 3), dtype=np.float64)
        rf = np.zeros((0, 3), dtype=np.int32)
    return Mesh(fv, ff), Mesh(rv, rf)


def _meridian_outline(
    spec: LensSpec,
    *,
    flange_radial_mm: float = 0.0,
    flange_thickness_mm: float = 0.0,
    n: int = 48,
) -> np.ndarray:
    """Closed Y–Z outline of one singlet (mm), winding +Y then −Y."""
    ap = max(1e-3, float(spec.aperture))
    ap_y = float(spec.aperture_y) if spec.aperture_y else ap
    mode = spec.mode or "rotational"
    R1y = spec.R1y if spec.R1y is not None else spec.R1
    R2y = spec.R2y if spec.R2y is not None else spec.R2
    z1 = float(spec.z_front)
    z2 = z1 + float(spec.thickness)
    fr = max(0.0, float(flange_radial_mm or 0.0))
    ft = max(0.0, float(flange_thickness_mm or 0.0))
    inner_r = _clear_semi_mm(ap, ap_y)
    outer_r = inner_r + fr if fr > 1e-6 and ft > 1e-6 else inner_r
    z_c = flange_cut_z(spec)
    z_ff = z_c - 0.5 * ft if ft > 1e-6 and fr > 1e-6 else None
    z_rf = z_c + 0.5 * ft if ft > 1e-6 and fr > 1e-6 else None

    ys = np.linspace(0.0, inner_r, max(8, int(n)))
    front = []
    rear = []
    for y in ys:
        s1 = sag_xy(0.0, float(y), spec.R1, R1y, spec.k1, 0.0, spec.A4_1, 0.0, mode)
        s2 = sag_xy(0.0, float(y), spec.R2, R2y, spec.k2, 0.0, spec.A4_2, 0.0, mode)
        front.append((z1 + (0.0 if s1 is None else s1), float(y)))
        rear.append((z2 + (0.0 if s2 is None else s2), float(y)))
    # +Y walk: front vertex → front rim → flange → rear rim → rear vertex
    pts: List[Tuple[float, float]] = list(front)
    if z_ff is not None and z_rf is not None:
        pts.append((float(z_ff), inner_r))
        pts.append((float(z_ff), outer_r))
        pts.append((float(z_rf), outer_r))
        pts.append((float(z_rf), inner_r))
    pts.extend(reversed(rear))
    # Mirror to −Y (skip the on-axis rear vertex duplicate)
    minus = [(z, -y) for (z, y) in reversed(pts[1:-1])]
    pts.extend(minus)
    if pts:
        pts.append(pts[0])
    return np.array(pts, dtype=np.float64)


def optical_meridian(
    spec: LensSpec,
    which: str,
    n: int = 48,
) -> np.ndarray:
    """
    Meridian of one optical face in design coordinates.

    Returns (N, 2) array of (r, z) from the vertex (r=0) to the clear
    aperture, millimetres. ``which`` is 'front' or 'rear'.
    """
    which = str(which or "front").lower()
    ap = max(1e-3, float(spec.aperture))
    ap_y = float(spec.aperture_y) if spec.aperture_y else ap
    semi = _clear_semi_mm(ap, ap_y)
    mode = spec.mode or "rotational"
    n = max(8, int(n))
    rs = np.linspace(0.0, semi, n)
    z_v = float(spec.z_front) if which == "front" else float(spec.z_front) + float(spec.thickness)
    if which == "front":
        Rx, Ry = float(spec.R1), (spec.R1y if spec.R1y is not None else spec.R1)
        kx, a4x = float(spec.k1), float(spec.A4_1)
    else:
        Rx, Ry = float(spec.R2), (spec.R2y if spec.R2y is not None else spec.R2)
        kx, a4x = float(spec.k2), float(spec.A4_2)
    pts = np.zeros((n, 2), dtype=np.float64)
    for i, r in enumerate(rs):
        s = sag_xy(float(r), 0.0, Rx, Ry, kx, 0.0, a4x, 0.0, mode)
        pts[i, 0] = float(r)
        pts[i, 1] = z_v + (0.0 if s is None else float(s))
    return pts


def meridian_air_normals(r: np.ndarray, z: np.ndarray, which: str) -> np.ndarray:
    """
    Unit normals in the (r, z) plane pointing into air.

    Front face: air is -Z (toward the source). Rear face: air is +Z.
    """
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    n = len(r)
    nrm = np.zeros((n, 2), dtype=np.float64)
    if n == 0:
        return nrm
    # Central differences along the meridian
    dr = np.gradient(r)
    dz = np.gradient(z)
    # Tangent (dr, dz); rotate to a normal and pick the air side
    # N = (dz, -dr) has N·(+Z) = -dr; we flip as needed.
    nr = dz.copy()
    nz = -dr.copy()
    front = str(which or "front").lower() == "front"
    for i in range(n):
        # Want Nz < 0 on the front (air -Z), Nz > 0 on the rear
        if front and nz[i] > 0:
            nr[i] = -nr[i]
            nz[i] = -nz[i]
        if (not front) and nz[i] < 0:
            nr[i] = -nr[i]
            nz[i] = -nz[i]
        ln = math.hypot(float(nr[i]), float(nz[i]))
        if ln < 1e-15:
            nr[i], nz[i] = 0.0, (-1.0 if front else 1.0)
        else:
            nr[i] /= ln
            nz[i] /= ln
    # Axis of revolution: the vertex normal is purely axial
    if n > 0 and abs(float(r[0])) < 1e-9:
        nr[0], nz[0] = 0.0, (-1.0 if front else 1.0)
    nrm[:, 0] = nr
    nrm[:, 1] = nz
    return nrm


def ball_tool_centers(
    r: np.ndarray,
    z: np.ndarray,
    nr: np.ndarray,
    nz: np.ndarray,
    tool_radius_mm: float,
    allowance_mm: float = 0.0,
) -> np.ndarray:
    """Ball-center meridian: surface + (R_tool + allowance) * air normal."""
    off = float(tool_radius_mm) + float(allowance_mm)
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    nr = np.asarray(nr, dtype=np.float64).reshape(-1)
    nz = np.asarray(nz, dtype=np.float64).reshape(-1)
    tc = np.zeros((len(r), 2), dtype=np.float64)
    tc[:, 0] = r + off * nr
    tc[:, 1] = z + off * nz
    return tc


def polish_machine_coords(
    spec: LensSpec,
    which: str,
    r_t: np.ndarray,
    z_t: np.ndarray,
    *,
    x_origin: str = "cut",
) -> np.ndarray:
    """
    Map tool-center (r, z_design) to machine (X, Y, Z) millimetres.

    Genmitsu-style 4-axis: A along X (optical axis). Vertical spindle, so
    Z is the radial approach. Y is held at 0.

    X0 at the flange mid-plane (printed flat) by default; X+ toward the
    optical surface so the half sticks out of the fixture.
    """
    r_t = np.asarray(r_t, dtype=np.float64).reshape(-1)
    z_t = np.asarray(z_t, dtype=np.float64).reshape(-1)
    z_cut = flange_cut_z(spec)
    which = str(which or "front").lower()
    xyz = np.zeros((len(r_t), 3), dtype=np.float64)
    if str(x_origin or "cut").lower() == "vertex":
        z_v = float(spec.z_front) if which == "front" else float(spec.z_front) + float(spec.thickness)
        if which == "front":
            xyz[:, 0] = z_v - z_t
        else:
            xyz[:, 0] = z_t - z_v
    else:
        if which == "front":
            xyz[:, 0] = z_cut - z_t
        else:
            xyz[:, 0] = z_t - z_cut
    xyz[:, 1] = 0.0
    xyz[:, 2] = np.maximum(r_t, 0.0)
    return xyz


def polish_toolpath(
    spec: LensSpec,
    which: str,
    *,
    tool_radius_mm: float = 5.0,
    allowance_mm: float = 0.0,
    stepover_mm: float = 0.3,
    strategy: str = "helical",
    revs_per_ring: float = 2.0,
    n_min: int = 16,
    x_origin: str = "cut",
) -> Dict[str, Any]:
    """
    Build a 4-axis polish path for one optical face.

    Returns dict with machine xyz (N,3), a_deg (N,), n_spins, and metadata.
    Points run rim -> vertex so the tool starts at the OD (clear of the chuck).
    """
    step = max(float(stepover_mm), 0.05)
    semi = _clear_semi_mm(float(spec.aperture), spec.aperture_y)
    n = max(int(n_min), int(math.ceil(semi / step)) + 1)
    mer = optical_meridian(spec, which, n=n)
    nrm = meridian_air_normals(mer[:, 0], mer[:, 1], which)
    tc = ball_tool_centers(
        mer[:, 0], mer[:, 1], nrm[:, 0], nrm[:, 1], tool_radius_mm, allowance_mm
    )
    # Rim first
    mer = mer[::-1]
    tc = tc[::-1]
    xyz = polish_machine_coords(spec, which, tc[:, 0], tc[:, 1], x_origin=x_origin)
    n_pts = len(xyz)
    a = np.zeros(n_pts, dtype=np.float64)
    n_spins = 0
    strat = str(strategy or "helical").lower()
    if strat == "rings":
        revs = max(0.25, float(revs_per_ring))
        # At each station the spindle dwells in XZ and A turns.
        # Encode as duplicate XZ with A advanced (export expands to G1 A).
        xs, ys, zs, aa = [], [], [], []
        a_now = 0.0
        for i in range(n_pts):
            xs.append(xyz[i, 0])
            ys.append(xyz[i, 1])
            zs.append(xyz[i, 2])
            aa.append(a_now)
            a_now += 360.0 * revs
            xs.append(xyz[i, 0])
            ys.append(xyz[i, 1])
            zs.append(xyz[i, 2])
            aa.append(a_now)
            n_spins += 1
        xyz = np.column_stack([xs, ys, zs])
        a = np.array(aa, dtype=np.float64)
    else:
        # Helical: A advances so the pitch along the tool-center path ~ stepover
        a_now = 0.0
        a[0] = 0.0
        for i in range(1, n_pts):
            ds = math.hypot(
                float(xyz[i, 0] - xyz[i - 1, 0]),
                float(xyz[i, 2] - xyz[i - 1, 2]),
            )
            a_now += (ds / step) * 360.0
            a[i] = a_now
        n_spins = int(round(a_now / 360.0))
    return {
        "xyz": xyz,
        "a_deg": a,
        "n_spins": n_spins,
        "which": which,
        "tool_radius_mm": float(tool_radius_mm),
        "stepover_mm": step,
        "strategy": strat,
    }


def _format_polish_gcode(
    paths: Sequence[Dict[str, Any]],
    *,
    feed_mm_min: float = 100.0,
    retract_mm: float = 5.0,
    spindle_rpm: float = 5000.0,
    y_offset_mm: float = 0.0,
) -> str:
    feed = max(1.0, float(feed_mm_min))
    retr = max(0.5, float(retract_mm))
    rpm = float(spindle_rpm)
    y0 = float(y_offset_mm)
    lines = [
        "(OptiFlux 4-axis lens polish)",
        "(Genmitsu-style: A = optical axis along X, Z = radial, Y = 0)",
        "(Spherical abrasive: tool center is offset along the air normal)",
        "(X0 = flange mid-plane / printed flat; X+ toward the optical surface)",
        "(Mount the printed half with its axis on A. Candle: use .nc)",
        "G21",
        "G90",
        "G94",
        "G17",
        f"G0 Z{retr:.4f}",
    ]
    if rpm > 1.0:
        lines.append(f"M3 S{rpm:.0f}")
    else:
        lines.append("M3")
    for pi, path in enumerate(paths):
        xyz = np.asarray(path["xyz"], dtype=np.float64)
        a = np.asarray(path["a_deg"], dtype=np.float64)
        if len(xyz) == 0:
            continue
        which = path.get("which", "front")
        lines.append(
            f"(Face {which}  tool R={float(path.get('tool_radius_mm', 0)):.3f} mm  "
            f"{path.get('strategy', 'helical')}  radial approach)"
        )
        x, y, z = float(xyz[0, 0]), float(xyz[0, 1]) + y0, float(xyz[0, 2])
        z_safe = z + retr
        lines.append(f"G0 X{x:.4f} Y{y:.4f} A{float(a[0]):.3f}")
        lines.append(f"G0 Z{z_safe:.4f}")
        lines.append(f"G1 Z{z:.4f} F{feed:.2f}")
        for i in range(1, len(xyz)):
            x, y, z = float(xyz[i, 0]), float(xyz[i, 1]) + y0, float(xyz[i, 2])
            lines.append(f"G1 X{x:.4f} Y{y:.4f} Z{z:.4f} A{float(a[i]):.3f}")
        lines.append(f"G0 Z{z + retr:.4f}")
        if pi + 1 < len(paths):
            lines.append("(Reclamp the mating half before the next face)")
            lines.append("M0")
    lines.append(f"G0 Z{retr:.4f}")
    lines.append("M5")
    lines.append("M30")
    return "\n".join(lines) + "\n"


def _force_nc_path(path: Path) -> Path:
    path = Path(path)
    if path.suffix.lower() not in (".nc", ".gcode", ".cnc", ".tap"):
        return path.with_suffix(CNC_PROFILE_EXT)
    if path.suffix.lower() == ".gcode":
        return path.with_suffix(CNC_PROFILE_EXT)
    return path


def export_lens_cnc_profile(
    params: Dict[str, Any],
    path: str | Path,
    *,
    flange_radial_mm: float = 0.0,
    flange_thickness_mm: float = 0.0,
    feed_mm_min: Optional[float] = None,
    safe_z_mm: Optional[float] = None,
    tool_dia_mm: Optional[float] = None,
    stepover_mm: Optional[float] = None,
    surface: Optional[str] = None,
    strategy: Optional[str] = None,
    revs_per_ring: Optional[float] = None,
    allowance_mm: Optional[float] = None,
    spindle_rpm: Optional[float] = None,
) -> Path:
    """
    Write 4-axis GRBL polish G-code for printed lens halves.

    Machine: A rotary = optical axis (along X), spherical tool on the
    vertical spindle approaching radially in Z. Default suffix .nc so
    Candle / GRBL senders open the file.
    """
    path = _force_nc_path(Path(path))
    specs, mode = build_lens_specs_from_params(params)
    if not specs:
        raise ValueError("No enabled lens element to export")
    if mode == "mla":
        raise ValueError("4-axis polish is for singlet halves, not an MLA plate")

    p = params or {}
    tool_dia = float(tool_dia_mm if tool_dia_mm is not None else p.get("cad_polish_tool_dia_mm", 10.0))
    tool_r = max(0.25, tool_dia * 0.5)
    step = float(stepover_mm if stepover_mm is not None else p.get("cad_polish_stepover_mm", 0.3))
    surf = str(surface if surface is not None else p.get("cad_polish_surface", "front")).lower()
    strat = str(strategy if strategy is not None else p.get("cad_polish_strategy", "helical")).lower()
    revs = float(revs_per_ring if revs_per_ring is not None else p.get("cad_polish_revs", 2.0))
    allow = float(allowance_mm if allowance_mm is not None else p.get("cad_polish_allowance_mm", 0.0))
    feed = float(feed_mm_min if feed_mm_min is not None else p.get("cad_polish_feed_mm_min", 100.0))
    retr = float(safe_z_mm if safe_z_mm is not None else p.get("cad_polish_retract_mm", 5.0))
    rpm = float(spindle_rpm if spindle_rpm is not None else p.get("cad_polish_rpm", 5000.0))
    y_off = float(p.get("cad_polish_y_offset_mm", 0.0))
    x_origin = str(p.get("cad_polish_x_origin", "cut"))

    if surf in ("both", "all"):
        faces = ["front", "rear"]
    elif surf == "rear":
        faces = ["rear"]
    else:
        faces = ["front"]

    written: List[Path] = []
    if len(faces) == 2:
        dests = [
            path.with_name(f"{path.stem}_front{path.suffix}"),
            path.with_name(f"{path.stem}_rear{path.suffix}"),
        ]
    else:
        dests = [path]

    for face, dest in zip(faces, dests):
        jobs = []
        for ei, spec in enumerate(specs):
            job = polish_toolpath(
                spec,
                face,
                tool_radius_mm=tool_r,
                allowance_mm=allow,
                stepover_mm=step,
                strategy=strat,
                revs_per_ring=revs,
                x_origin=x_origin,
            )
            job["which"] = f"E{ei + 1} {face}"
            jobs.append(job)
        dest.write_text(
            _format_polish_gcode(
                jobs,
                feed_mm_min=feed,
                retract_mm=retr,
                spindle_rpm=rpm,
                y_offset_mm=y_off,
            ),
            encoding="ascii",
        )
        written.append(dest)

    return written[0]


def export_lens(
    params: Dict[str, Any],
    path: str | Path,
    fmt: str = "stl",
    dies=None,
    n_radial: int = 40,
    n_theta: int = 72,
    include_plate: bool = True,
    max_edge_mm: Optional[float] = None,
    max_angle_deg: Optional[float] = None,
    flange_radial_mm: float = 0.0,
    flange_thickness_mm: float = 0.0,
    emit_tube_notes: bool = True,
    split_halves: bool = False,
) -> Path:
    """
    Export lens geometry. fmt in {'stl','stl_ascii','step'}.
    All coordinates in millimetres.

    - MLA → one monolithic plate solid
    - Single element → one solid
    - Multi-element stack → separate solid per element
      (STEP multi-body; STL merges shells into one file with air gaps preserved in Z)

    If ``split_halves`` is set (STL only), each body is cut at the flange
    mid-plane and written as ``*_front.stl`` / ``*_rear.stl`` so a resin
    printer can build on that flat face and the halves can be glued.

    If ``max_edge_mm`` or ``max_angle_deg`` is set, polar-grid density is
    computed from those print tolerances (and overrides n_radial / n_theta).
    """
    path = Path(path)
    specs, mode = build_lens_specs_from_params(params, dies)
    if not specs:
        raise ValueError("No enabled lens element to export")

    fmt = fmt.lower().replace(".", "")
    fr = max(0.0, float(flange_radial_mm or 0.0))
    ft = max(0.0, float(flange_thickness_mm or 0.0))
    do_halves = bool(split_halves) and fmt in ("stl", "stl_binary", "bin", "stl_ascii", "ascii")

    if max_edge_mm is not None or max_angle_deg is not None:
        n_radial, n_theta = tessellation_for_specs(
            specs,
            max_edge_mm=CAD_DEFAULT_MAX_EDGE_MM if max_edge_mm is None else float(max_edge_mm),
            max_angle_deg=CAD_DEFAULT_MAX_ANGLE_DEG if max_angle_deg is None else float(max_angle_deg),
        )

    if mode == "mla":
        # One solid MLA plate with Element-1 form lenslets (not cylinders on a slab)
        mesh = mesh_mla(
            specs,
            include_plate=include_plate,
            plate_extra_z=0.12 if include_plate else 0.0,
            n_radial=max(24, n_radial),
            n_theta=n_theta,
        )
        bodies = [mesh]
        cut_zs = [0.5 * (float(np.min(mesh.vertices[:, 2])) + float(np.max(mesh.vertices[:, 2])))]
    else:
        # One mesh body per enabled element (correct Z along the stack)
        bodies = [
            mesh_singlet(
                s,
                n_radial=n_radial,
                n_theta=n_theta,
                flange_radial_mm=fr,
                flange_thickness_mm=ft,
            )
            for s in specs
        ]
        cut_zs = [flange_cut_z(s) for s in specs]
        mesh = bodies[0]
        for extra in bodies[1:]:
            mesh = mesh.merge(extra)

    if do_halves:
        if path.suffix.lower() != ".stl":
            path = path.with_suffix(".stl")
        writer = write_stl_ascii if fmt in ("stl_ascii", "ascii") else write_stl_binary
        dests = half_stl_paths(path, len(bodies))
        written: Optional[Path] = None
        for i, body in enumerate(bodies):
            zc = cut_zs[i] if i < len(cut_zs) else flange_cut_z(specs[min(i, len(specs) - 1)])
            front, rear = split_mesh_at_z(body, zc)
            fp = dests[2 * i]
            rp = dests[2 * i + 1]
            writer(fp, front, solid_name=f"OptiFlux_E{i + 1}_front_mm")
            writer(rp, rear, solid_name=f"OptiFlux_E{i + 1}_rear_mm")
            if written is None:
                written = fp
        if written is None:
            raise RuntimeError("No halves to export")
        if emit_tube_notes:
            notes = path.with_name(path.stem + "_tube.txt")
            write_tube_notes(
                notes,
                tube_layout(params, flange_radial_mm=fr, flange_thickness_mm=ft),
            )
        return written

    if fmt in ("stl", "stl_binary", "bin"):
        if path.suffix.lower() != ".stl":
            path = path.with_suffix(".stl")
        written = write_stl_binary(path, mesh)
    elif fmt in ("stl_ascii", "ascii"):
        if path.suffix.lower() != ".stl":
            path = path.with_suffix(".stl")
        written = write_stl_ascii(path, mesh)
    elif fmt in ("step", "stp"):
        if path.suffix.lower() not in (".step", ".stp"):
            path = path.with_suffix(".step")
        if len(bodies) > 1:
            written = write_step_multibody(path, bodies, name="OptiFlux_LensStack")
        else:
            written = write_step_mesh(path, bodies[0])
    else:
        raise ValueError(f"Unknown format: {fmt}")

    if emit_tube_notes:
        notes = written.with_name(written.stem + "_tube.txt")
        write_tube_notes(
            notes,
            tube_layout(params, flange_radial_mm=fr, flange_thickness_mm=ft),
        )
    return written
