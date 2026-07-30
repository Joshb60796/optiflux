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


def mesh_singlet(
    spec: LensSpec,
    n_radial: int = 48,
    n_theta: int = 96,
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
    f_edge = _edge_faces(n_t, front_rim, back_rim, flip=False)

    faces = np.vstack([f_front, f_back, f_edge])
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
    """
    elements = [e for e in params.get("elements", []) if e.get("enabled", True)]
    if not elements:
        return [], "empty"
    e0 = elements[0]
    z0 = float(params.get("lens_z_start", 3.0))
    mla = params.get("mla", {}) or {}
    mla_on = bool(mla.get("enabled", False))

    if not mla_on:
        r1y = e0.get("R1y", None)
        r2y = e0.get("R2y", None)
        apy = e0.get("aperture_y", None)
        return [
            LensSpec(
                R1=float(e0["R1"]),
                R2=float(e0["R2"]),
                thickness=float(e0["thickness"]),
                aperture=float(e0["aperture"]),
                k1=float(e0.get("k1", 0.0)),
                k2=float(e0.get("k2", 0.0)),
                A4_1=float(e0.get("A4_1", 0.0)),
                A4_2=float(e0.get("A4_2", 0.0)),
                x0=0.0,
                y0=0.0,
                z_front=z0,
                R1y=float(r1y) if r1y is not None else None,
                R2y=float(r2y) if r2y is not None else None,
                mode=str(e0.get("surface_mode", "rotational")),
                aperture_y=float(apy) if apy is not None else None,
            )
        ], "singlet"

    from mla_geometry import build_mla_lens_specs

    specs, _meta = build_mla_lens_specs(params, dies=dies)
    return specs, "mla"


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


def export_lens(
    params: Dict[str, Any],
    path: str | Path,
    fmt: str = "stl",
    dies=None,
    n_radial: int = 40,
    n_theta: int = 72,
    include_plate: bool = True,
) -> Path:
    """
    Export lens geometry. fmt in {'stl','stl_ascii','step'}.
    All coordinates in millimetres.
    """
    path = Path(path)
    specs, mode = build_lens_specs_from_params(params, dies)
    if not specs:
        raise ValueError("No enabled lens element to export")

    fmt = fmt.lower().replace(".", "")

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
    else:
        mesh = mesh_singlet(specs[0], n_radial=n_radial, n_theta=n_theta)
        bodies = [mesh]

    if fmt in ("stl", "stl_binary", "bin"):
        if path.suffix.lower() != ".stl":
            path = path.with_suffix(".stl")
        return write_stl_binary(path, mesh)
    if fmt in ("stl_ascii", "ascii"):
        if path.suffix.lower() != ".stl":
            path = path.with_suffix(".stl")
        return write_stl_ascii(path, mesh)
    if fmt in ("step", "stp"):
        if path.suffix.lower() not in (".step", ".stp"):
            path = path.with_suffix(".step")
        return write_step_mesh(path, mesh)
    raise ValueError(f"Unknown format: {fmt}")
