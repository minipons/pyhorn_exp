"""
pyhorn_wavefront — 2D Wavefront Simulator for pyhorn.

Models 2D sound-wave propagation through horn geometry using the Helmholtz
equation (∇²p + k²p = 0) via a finite-difference discretization solved with
scipy.sparse / scipy.sparse.linalg.

Key outputs:
  - 2D complex pressure field at a given frequency
  - Wavefront visualisations with horn-geometry overlay
  - k·a warning when the 1-D horn assumption breaks down (k·a > ~0.5)

Physics reference: Hornresp manual pp. 87–88; Morse & Ingard §9.3.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# ─── Physical constants ───────────────────────────────────────────────────────
C = 343.0  # Speed of sound m/s

# ─── Dataclass ────────────────────────────────────────────────────────────────


@dataclass
class WavefrontGrid:
    """
    Container for a 2D rectangular simulation grid.

    Attributes
    ----------
    nx, ny : int
        Number of grid points in x and y directions.
    dx, dy : float
        Grid spacing in metres.
    frequency : float
        Drive frequency in Hz.
    k : float
        Wave number (2πf / C).
    pressure_field : NDArray[complex]
        2-D complex pressure field (shape ny × nx).  Real part is the
        steady-state pressure; imaginary part carries phase information.
    source_x, source_y : float | None
        Point-source position in metres.  Set via ``set_point_source()``.
    """

    nx: int
    ny: int
    dx: float
    dy: float
    frequency: float
    k: float
    pressure_field: NDArray[np.complex128] = field(default=None)
    source_x: float | None = field(default=None)
    source_y: float | None = field(default=None)
    boundary_mask: NDArray[np.bool_] | None = field(default=None)
    horn_coords: NDArray[np.float64] | None = field(default=None)

    def __post_init__(self):
        if self.pressure_field is None:
            self.pressure_field = np.zeros((self.ny, self.nx), dtype=np.complex128)

    def set_point_source(self, x: float, y: float) -> None:
        """
        Place the point source (driver) at the given coordinates.

        If the grid has already been solved, the caller should re-run
        ``solve_2d_wave`` with the updated source position.

        Parameters
        ----------
        x, y : float
            Source location in metres.
        """
        self.source_x = float(x)
        self.source_y = float(y)

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path,
        nx: int = 200,
        ny: int = 100,
        frequency: float = 500.0,
        wall_thickness: float = 0.005,
        margin: float = 0.02,
    ) -> tuple["WavefrontGrid", NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]]:
        """
        Build a ready-to-solve ``WavefrontGrid`` from a pyhorn YAML file.

        This is a convenience factory that:
        1. Loads geometry from the YAML via ``load_horn_geometry``.
        2. Constructs a rectangular mesh that tightly bounds the horn + margin.
        3. Builds the wall boundary mask.
        4. Returns the grid (with source placed at ``driver_coord``), the mask,
           and the mesh arrays.

        Parameters
        ----------
        yaml_path : str or Path
            Path to a pyhorn project or source YAML.
        nx, ny : int
            Target grid points in x and y.  The mesh spacing is derived from
            the geometry bounding box so that the full horn fits in the grid.
        frequency : float
            Drive frequency in Hz (stored in the grid; does not affect the mesh).
        wall_thickness : float
            Half-width of the wall exclusion band in metres (passed to
            ``boundary_condition_mask``).
        margin : float
            Extra margin in metres added around the geometry bounding box
            when constructing the mesh.

        Returns
        -------
        tuple
            (grid, boundary_mask, mesh_x, mesh_y) where:
            - ``grid``: WavefrontGrid with source placed at driver_coord.
            - ``boundary_mask``: boolean wall mask.
            - ``mesh_x``, ``mesh_y``: 2-D coordinate arrays (metres).
        """
        geo = load_horn_geometry(yaml_path)
        coords = geo["coords"]

        # Bounding box with margin
        x_min, x_max = coords[:, 0].min() - margin, coords[:, 0].max() + margin
        y_min, y_max = coords[:, 1].min() - margin, coords[:, 1].max() + margin

        mesh_x = np.linspace(x_min, x_max, nx)
        mesh_y = np.linspace(y_min, y_max, ny)
        mesh_xx, mesh_yy = np.meshgrid(mesh_x, mesh_y)

        walls = boundary_condition_mask(coords, mesh_xx, mesh_yy, wall_thickness)

        dx = float(mesh_x[1] - mesh_x[0])
        dy = float(mesh_y[1] - mesh_y[0])
        k = 2.0 * np.pi * frequency / C

        grid = cls(
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            frequency=frequency,
            k=k,
            pressure_field=None,
            boundary_mask=walls,
            horn_coords=coords,
        )

        if geo["source_x"] is not None and geo["source_y"] is not None:
            grid.set_point_source(geo["source_x"], geo["source_y"])

        return grid, walls, mesh_xx, mesh_yy

    def plot_pressure_colormap(
        self,
        mesh_x: NDArray[np.float64],
        mesh_y: NDArray[np.float64],
        horn_path: list[tuple[float, float]] | None = None,
        output_path: str = "pressure_colormap.png",
        figsize: tuple[float, float] = (12, 9),
        p_ref: float = 2e-5,
    ) -> None:
        """
        Convenience wrapper — renders the two-panel **pressure amplitude
        colormap** (sign-encoded real part + dB SPL amplitude) directly from
        this grid's solved ``pressure_field``.

        Panel 1: Real-part pressure (Pa) — diverging colormap centred at zero.
                 Blue = rarefaction, white = zero, red = compression.
        Panel 2: |p| in dB SPL — sequential colormap showing acoustic energy.

        Parameters
        ----------
        mesh_x, mesh_y : NDArray
            Physical grid coordinates (metres) — same arrays used when calling
            ``solve_2d_wave`` or ``solve_2d_wave_pml``.
        horn_path : list of (x, y) or None
            Optional horn polygon vertices for the outline overlay.
        output_path : str
            Destination PNG path.  Default ``pressure_colormap.png``.
        figsize : tuple
            Matplotlib figure (w, h) in inches.
        p_ref : float
            Reference pressure for dB SPL (default 20 µPa).

        Returns
        -------
        None
        """
        plot_pressure_amplitude(
            pressure_field=self.pressure_field,
            mesh_x=mesh_x,
            mesh_y=mesh_y,
            horn_path=horn_path,
            frequency=self.frequency,
            output_path=output_path,
            figsize=figsize,
            p_ref=p_ref,
        )


# ─── Solver ──────────────────────────────────────────────────────────────────


def solve_2d_wave(
    mesh_x: NDArray[np.float64],
    mesh_y: NDArray[np.float64],
    source_x: float,
    source_y: float,
    frequency: float,
    boundary_mask: NDArray[np.bool_],
) -> WavefrontGrid:
    """
    Solve the 2-D Helmholtz equation on a rectangular grid with a point source
    and hard-wall (Neumann) boundary conditions.

    Discretisation
    --------------
    Central-difference approximation of the Laplacian::

        ∇²p ≈ (p_{i+1,j} - 2p_{i,j} + p_{i-1,j}) / dx²
            + (p_{i,j+1} - 2p_{i,j} + p_{i,j-1}) / dy²

    The Helmholtz equation ∇²p + k²p = 0 becomes::

        (2/dx² + 2/dy² - k²) · p_{i,j}
            - p_{i+1,j}/dx² - p_{i-1,j}/dx²
            - p_{i,j+1}/dy² - p_{i,j-1}/dy² = 0

    Boundary nodes (wall cells in ``boundary_mask``) use a zero-field
    extrapolation ghost-node approach to enforce ∂p/∂n = 0.

    Parameters
    ----------
    mesh_x, mesh_y : NDArray (2-D, shape ny × nx)
        Physical x/y coordinates of each grid point (metres).
    source_x, source_y : float
        Point-source location in metres.
    frequency : float
        Drive frequency in Hz.
    boundary_mask : NDArray (2-D, shape ny × nx)
        Boolean mask.  True = wall (hard boundary), False = interior/fluid.

    Returns
    -------
    WavefrontGrid
        Object containing the solved complex pressure field.
    """
    ny, nx = boundary_mask.shape
    dx = float(mesh_x[0, 1] - mesh_x[0, 0])
    dy = float(mesh_y[1, 0] - mesh_y[0, 0])
    k = 2.0 * np.pi * frequency / C

    # ── 1. Build interior index list ──────────────────────────────────────────
    interior = ~boundary_mask
    n_interior = int(np.sum(interior))

    # Map (i, j) global grid index → row in the sparse system
    idx_map = np.full((ny, nx), -1, dtype=np.int32)
    idx_map[interior] = np.arange(n_interior, dtype=np.int32)

    # ── 2. Construct sparse tridiagonal / pentadiagonal system ───────────────
    row: list[int] = []
    col: list[int] = []
    data: list[float] = []
    rhs: list[complex] = []

    # Source delta (complex) — placed at nearest interior node to source position
    src_j = int(np.argmin(np.abs(mesh_x[0, :] - source_x)))
    src_i = int(np.argmin(np.abs(mesh_y[:, 0] - source_y)))
    src_linear = src_j + src_i * nx  # column-major flatten index

    A_sparsity = 5 * n_interior  # approximate non-zeros

    for i in range(ny):
        for j in range(nx):
            if boundary_mask[i, j]:
                continue

            row_idx = idx_map[i, j]

            # Diagonal coefficient (interior stencil)
            diag = 2.0 / dx**2 + 2.0 / dy**2 - k**2
            row.append(row_idx)
            col.append(row_idx)
            data.append(diag)

            # Neighbours — skip out-of-bounds or wall cells
            neighbours = [
                (i - 1, j, 1.0 / dy**2),  # below
                (i + 1, j, 1.0 / dy**2),  # above
                (i, j - 1, 1.0 / dx**2),  # left
                (i, j + 1, 1.0 / dx**2),  # right
            ]
            for ni, nj, coeff in neighbours:
                if 0 <= ni < ny and 0 <= nj < nx:
                    if not boundary_mask[ni, nj]:
                        row.append(row_idx)
                        col.append(idx_map[ni, nj])
                        data.append(-coeff)

            # RHS: point source (Laplacian of a point dipole ≈ 0, so we
            # inject amplitude directly at the source node; the sign convention
            # follows the Helmholtz RHS = -Q for a point source).
            if (i == src_i and j == src_j) or (i * nx + j == src_linear):
                rhs.append(1.0 + 0.0j)
            else:
                rhs.append(0.0 + 0.0j)

    # ── 3. Assemble and solve ─────────────────────────────────────────────────
    A = csr_matrix((data, (row, col)), shape=(n_interior, n_interior))
    b = np.array(rhs, dtype=np.complex128)

    # Solve sparse linear system
    p_interior = spsolve(A, b)

    # ── 4. Scatter back to full grid ──────────────────────────────────────────
    pressure_field = np.zeros((ny, nx), dtype=np.complex128)
    pressure_field[interior] = p_interior

    return WavefrontGrid(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        frequency=frequency,
        k=k,
        pressure_field=pressure_field,
    )


def boundary_condition_mask(
    horn_coords: list[tuple[float, float]] | NDArray[np.float64],
    mesh_x: NDArray[np.float64],
    mesh_y: NDArray[np.float64],
    wall_thickness: float = 0.005,
) -> NDArray[np.bool_]:
    """
    Build a boolean wall mask from horn geometry coordinates.

    The horn outline is provided as an ordered list of (x, y) vertices in
    physical metres.  Any grid point within ``wall_thickness`` of the polygon
    edges is marked as a hard wall.

    Parameters
    ----------
    horn_coords : list of (x, y) or ndarray
        Vertices of the horn polygon (closed or open — this function closes it).
    mesh_x, mesh_y : NDArray
        Grid coordinate arrays (shape ny × nx).
    wall_thickness : float
        Half-width of the wall exclusion band around the horn polygon
        edges (metres).  Default 5 mm.

    Returns
    -------
    boundary_mask : NDArray[bool] (shape ny × nx)
        True where the cell is inside the horn wall band, False in fluid.
    """
    from shapely.geometry import Polygon, LineString, Point
    from shapely.ops import unary_union

    coords = np.asarray(horn_coords)
    if coords.ndim == 1:
        # Already a flat list of alternating x, y — reshape to (N, 2)
        coords = coords.reshape(-1, 2)

    # Ensure the polygon is closed
    if np.all(coords[0] != coords[-1]):
        coords = np.vstack([coords, coords[0]])

    poly = Polygon(coords)

    # Compute distance of every grid point to the horn polygon boundary
    pts = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
    dists = np.array([poly.boundary.distance(Point(pt)) for pt in pts])
    dists = dists.reshape(mesh_x.shape)

    return dists <= wall_thickness


# ─── YAML geometry loader ─────────────────────────────────────────────────────


def load_horn_geometry(
    yaml_path: str | Path,
) -> dict[str, Any]:
    """
    Load horn geometry from a pyhorn project or source YAML file.

    Supports two formats:

    **Project YAML** (``projects/*.yaml``):
        Contains a ``geometry_path`` pointing to the source YAML, plus
        ``driver_coord: [x, y]`` for the source position.

    **Source YAML** (``source/*.yaml``):
        Contains ``coordinates: [[x, y], ...]`` directly.

    Parameters
    ----------
    yaml_path : str or Path
        Path to a pyhorn project YAML or source YAML.

    Returns
    -------
    dict with keys:
        ``coords``      — ndarray shape (N, 2): polygon vertices in metres.
        ``source_x``     — float: driver x position in metres (or None).
        ``source_y``     — float: driver y position in metres (or None).
        ``enclosure_dims`` — list [width, height] in metres (or None).
        ``source_yaml`` — Path: resolved path to the source YAML.
        ``name``        — str: project/source name from the YAML.

    Raises
    ------
    FileNotFoundError
        If the YAML file or its referenced geometry_path does not exist.
    ValueError
        If the YAML contains no recognised geometry keys.
    """
    import yaml

    yaml_path = Path(yaml_path).expanduser()

    with open(yaml_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if data is None:
        raise ValueError(f"Empty YAML file: {yaml_path}")

    name = str(data.get("name", yaml_path.stem))

    # ── Resolve coordinates ───────────────────────────────────────────────────
    coords: NDArray[np.float64] | None = None

    # Source YAML format: coordinates listed directly
    raw_coords = data.get("coordinates")
    if raw_coords is not None:
        coords = np.asarray(raw_coords, dtype=np.float64)
        source_yaml = yaml_path
        driver_x = float(data["source_x"]) if data.get("source_x") is not None else None
        driver_y = float(data["source_y"]) if data.get("source_y") is not None else None
        enclosure_dims: list | None = data.get("enclosure_dims")

    # Project YAML format: references a source YAML via geometry_path
    elif "geometry_path" in data:
        geometry_rel = str(data["geometry_path"])
        source_yaml = (yaml_path.parent / geometry_rel).resolve()
        with open(source_yaml, encoding="utf-8") as fh:
            src_data = yaml.safe_load(fh)

        raw_coords = src_data.get("coordinates")
        if raw_coords is None:
            raise ValueError(
                f"Referenced source YAML has no 'coordinates' key: {source_yaml}"
            )
        coords = np.asarray(raw_coords, dtype=np.float64)

        driver_raw = data.get("driver_coord")
        if driver_raw is not None:
            driver_x, driver_y = float(driver_raw[0]), float(driver_raw[1])
        else:
            driver_x, driver_y = None, None

        enclosure_dims = data.get("enclosure_dims") or src_data.get(
            "enclosure_dims"
        )
        name = str(data.get("name", src_data.get("name", source_yaml.stem)))

    else:
        raise ValueError(
            f"YAML must contain 'coordinates' (source YAML) or "
            f"'geometry_path' (project YAML). Neither found in: {yaml_path}"
        )

    if coords is None:
        raise ValueError(f"No geometry coordinates found in: {yaml_path}")

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            f"'coordinates' must be a list of [x, y] pairs; "
            f"got shape {coords.shape}: {yaml_path}"
        )

    return {
        "coords": coords,
        "source_x": driver_x,
        "source_y": driver_y,
        "enclosure_dims": enclosure_dims,
        "source_yaml": source_yaml,
        "name": name,
    }


# ─── PML absorbing boundary ──────────────────────────────────────────────────


def pml_damping_mask(
    ny: int,
    nx: int,
    pml_width: int = 15,
    kappa_max: float = 2.0,
) -> NDArray[np.float64]:
    """
    Build a 2-D PML (Perfectly Matched Layer) damping coefficient array.

    The PML is a strip of absorbing material applied at all four grid edges.
    Inside the PML the wave is exponentially attenuated, greatly reducing
    reflections from the domain boundary.  This lets the solver use a finite
    grid while approximately simulating an infinite domain.

    The damping profile follows a quadratic ramp::

        sigma(s) = kappa_max * (s / pml_width)²

    where ``s`` is the normalised distance from the inner PML face (0–1).

    Parameters
    ----------
    ny, nx : int
        Grid dimensions.
    pml_width : int
        Number of cells thick for each PML layer.  Default 15.
    kappa_max : float
        Peak damping coefficient (dimensionless).  Values of 1–3 are typical;
        higher values absorb more but may cause minor numerical reflection at
        the PML inner face.

    Returns
    -------
    sigma : NDArray (shape ny × nx)
        2-D array of damping coefficients.  Zero inside the computational
        domain, rising quadratically toward the edges.  Multiply the
        Helmholtz diagonal term by ``1 + 1j * sigma`` to apply the PML.
    """
    sigma = np.zeros((ny, nx), dtype=np.float64)

    w = min(pml_width, ny // 4, nx // 4)
    if w < 2:
        return sigma

    ramp = (np.arange(w, dtype=np.float64) + 1) / w
    profile = kappa_max * ramp**2

    for s in range(w):
        sigma[s, :] = profile[s]
        sigma[ny - 1 - s, :] = profile[s]

    for s in range(w):
        sigma[:, s] = np.maximum(sigma[:, s], profile[s])
        sigma[:, nx - 1 - s] = np.maximum(sigma[:, nx - 1 - s], profile[s])

    return sigma


def solve_2d_wave_pml(
    mesh_x: NDArray[np.float64],
    mesh_y: NDArray[np.float64],
    source_x: float,
    source_y: float,
    frequency: float,
    boundary_mask: NDArray[np.bool_],
    pml_width: int = 15,
    kappa_max: float = 2.0,
) -> WavefrontGrid:
    """
    Same as ``solve_2d_wave`` but with a PML absorbing boundary layer applied
    at all four grid edges.

    Use this when the horn mouth is close to the grid boundary and reflections
    from the domain edge would contaminate the solution.

    Parameters
    ----------
    mesh_x, mesh_y, source_x, source_y, frequency, boundary_mask
        Same as ``solve_2d_wave``.
    pml_width, kappa_max
        Passed to ``pml_damping_mask``; control PML thickness and strength.

    Returns
    -------
    WavefrontGrid
        Solved pressure field with PML absorption applied.
    """
    ny, nx = boundary_mask.shape
    dx = float(mesh_x[0, 1] - mesh_x[0, 0])
    dy = float(mesh_y[1, 0] - mesh_y[0, 0])
    k = 2.0 * np.pi * frequency / C

    interior = ~boundary_mask
    n_interior = int(np.sum(interior))

    idx_map = np.full((ny, nx), -1, dtype=np.int32)
    idx_map[interior] = np.arange(n_interior, dtype=np.int32)

    sigma = pml_damping_mask(ny, nx, pml_width=pml_width, kappa_max=kappa_max)

    src_j = int(np.argmin(np.abs(mesh_x[0, :] - source_x)))
    src_i = int(np.argmin(np.abs(mesh_y[:, 0] - source_y)))
    src_linear = src_j + src_i * nx

    row: list[int] = []
    col: list[int] = []
    data: list[complex] = []
    rhs: list[complex] = []

    for i in range(ny):
        for j in range(nx):
            if boundary_mask[i, j]:
                continue

            row_idx = idx_map[i, j]
            sig = sigma[i, j]

            # PML: multiply diagonal by (1 + 1j*sigma)
            diag = (2.0 / dx**2 + 2.0 / dy**2 - k**2) * (1 + 1j * sig)
            row.append(row_idx)
            col.append(row_idx)
            data.append(diag)

            neighbours = [
                (i - 1, j, 1.0 / dy**2),
                (i + 1, j, 1.0 / dy**2),
                (i, j - 1, 1.0 / dx**2),
                (i, j + 1, 1.0 / dx**2),
            ]
            for ni, nj, coeff in neighbours:
                if 0 <= ni < ny and 0 <= nj < nx:
                    if not boundary_mask[ni, nj]:
                        sig_n = sigma[ni, nj]
                        row.append(row_idx)
                        col.append(idx_map[ni, nj])
                        data.append(-coeff * (1 + 1j * (sig + sig_n) / 2))

            if (i == src_i and j == src_j) or (i * nx + j == src_linear):
                rhs.append(1.0 + 0.0j)
            else:
                rhs.append(0.0 + 0.0j)

    A = csr_matrix((data, (row, col)), shape=(n_interior, n_interior))
    b = np.array(rhs, dtype=np.complex128)

    p_interior = spsolve(A, b)

    pressure_field = np.zeros((ny, nx), dtype=np.complex128)
    pressure_field[interior] = p_interior

    return WavefrontGrid(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        frequency=frequency,
        k=k,
        pressure_field=pressure_field,
        source_x=source_x,
        source_y=source_y,
    )


# ─── Convenience wrapper ──────────────────────────────────────────────────────


def compute_pressure_field(
    mesh_x: NDArray[np.float64],
    mesh_y: NDArray[np.float64],
    source_x: float,
    source_y: float,
    frequency: float,
    walls: NDArray[np.bool_],
) -> WavefrontGrid:
    """
    One-liner to compute the steady-state 2-D pressure field for a given
    frequency and set of wall boundaries.

    Parameters
    ----------
    mesh_x, mesh_y : NDArray
        Grid coordinate arrays (metres).
    source_x, source_y : float
        Driver position in metres.
    frequency : float
        Drive frequency in Hz.
    walls : NDArray[bool]
        Wall mask returned by ``boundary_condition_mask``.

    Returns
    -------
    WavefrontGrid
        Contains the solved complex pressure field and grid metadata.
    """
    return solve_2d_wave(mesh_x, mesh_y, source_x, source_y, frequency, walls)


# ─── Plotting ─────────────────────────────────────────────────────────────────


def plot_wavefront(
    pressure_field: NDArray[np.complex128],
    mesh_x: NDArray[np.float64],
    mesh_y: NDArray[np.float64],
    horn_path: list[tuple[float, float]] | None,
    frequency: float,
    output_path: str,
    figsize: tuple[float, float] = (12, 6),
) -> None:
    """
    Plot the magnitude of the 2-D pressure field with the horn outline overlaid.

    Parameters
    ----------
    pressure_field : NDArray (complex)
        2-D complex pressure field (ny × nx).
    mesh_x, mesh_y : NDArray
        Physical grid coordinates (metres).
    horn_path : list of (x, y) or None
        Horn polygon vertices to draw on top of the pressure field.  Pass None
        to skip the overlay.
    frequency : float
        Drive frequency in Hz (used in the title).
    output_path : str
        Path to save the figure (PNG recommended).
    figsize : tuple
        Matplotlib figure size in inches (w, h).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    p_mag = np.abs(pressure_field)
    # Use a symmetric log scale so both positive and negative pressure lobes
    # are visible; clip zeros to avoid log(0) warnings.
    vmax = np.percentile(p_mag[p_mag > 0], 99) if p_mag.max() > 0 else 1.0
    norm = SymLogNorm(linthresh=vmax * 0.01, vmin=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(
        mesh_x,
        mesh_y,
        p_mag,
        cmap="RdBu_r",
        norm=norm,
        shading="auto",
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"2-D Wavefront — {frequency:.1f} Hz  (k = {2*np.pi*frequency/C:.2f} rad/m)")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label="|p| (Pa)")

    # Horn outline overlay
    if horn_path is not None:
        from matplotlib.patches import Polygon as MplPolygon
        from matplotlib.collections import PatchCollection

        coords = np.asarray(horn_path)
        if coords.ndim == 1:
            coords = coords.reshape(-1, 2)
        if not np.all(coords[0] == coords[-1]):
            coords = np.vstack([coords, coords[0]])

        horn_patch = MplPolygon(coords, closed=True, fill=False,
                                edgecolor="white", linewidth=1.5, linestyle="--")
        ax.add_patch(horn_patch)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ─── k·a validity check ──────────────────────────────────────────────────────


def ka_warning(
    pressure_field: NDArray[np.complex128],
    mesh_x: NDArray[np.float64],
    mesh_y: NDArray[np.float64],
    frequency: float,
    a_key: float | None = None,
) -> str | None:
    """
    Check whether the 1-D horn assumption remains valid at the given frequency.

    The 1-D approximation (plane-wave propagation inside the horn) holds while
    ``k · a ≪ 1``, where ``a`` is the characteristic horn radius (half the
    mouth width or throat radius). 业界经验阈值: k·a > 0.5 signals that
    transverse wave modes can exist and the 1-D model becomes inaccurate.

    This function estimates the largest local ``k·a`` using the pressure-field
    envelope: regions where |p| decays to < 10 % of its maximum are treated as
    the horn interior; the half-width of that region is used as ``a``.

    Parameters
    ----------
    pressure_field : NDArray
        Complex pressure field.
    mesh_x, mesh_y : NDArray
        Grid coordinates (metres).
    frequency : float
        Drive frequency in Hz.
    a_key : float | None
        If provided, use this explicit radius in metres instead of estimating
        from the pressure field.

    Returns
    -------
    str | None
        None if k·a < 0.5 (all clear).  Returns a warning string if k·a ≥ 0.5.
    """
    k = 2.0 * np.pi * frequency / C

    if a_key is not None:
        a = a_key
    else:
        p_mag = np.abs(pressure_field)
        # Simple heuristic: characteristic half-width = half the grid span
        # where pressure is above 10 % of its global maximum.
        threshold = 0.1 * p_mag.max()
        active = p_mag > threshold
        if not np.any(active):
            return None  # can't estimate; skip warning
        # Half-height of the active region along y
        rows = np.any(active, axis=1)
        if not np.any(rows):
            return None
        y_spread = np.sum(rows) * (mesh_y[1, 0] - mesh_y[0, 0])
        a = y_spread / 2.0

    ka = k * a

    if ka >= 0.5:
        msg = (
            f"[ka_warning] k·a = {ka:.3f} ≥ 0.5 at {frequency:.1f} Hz.\n"
            f"  Transverse wave effects are significant — 1-D horn model "
            f"accuracy is reduced.\n"
            f"  Consider a 2-D or 3-D treatment for this frequency."
        )
        warnings.warn(msg)
        return msg
    return None


# ─── Phase 3: Pressure amplitude colormap ────────────────────────────────────


def plot_pressure_amplitude(
    pressure_field: NDArray[np.complex128],
    mesh_x: NDArray[np.float64],
    mesh_y: NDArray[np.float64],
    horn_path: list[tuple[float, float]] | None,
    frequency: float,
    output_path: str,
    figsize: tuple[float, float] = (12, 9),
    p_ref: float = 2e-5,
) -> None:
    """
    Two-panel **pressure amplitude colormap** showing both the standing-wave
    pattern (real part of p) and the acoustic energy distribution (|p| in dB).

    Panel 1 — Real-part pressure (Pa)
        Diverging colormap centred at zero: blue = rarefaction, white = zero,
        red = compression.  Reveals the spatial structure of the standing-
        wave pattern at the chosen frequency.

    Panel 2 — Amplitude |p| in dB SPL
        |p| normalised to ``p_ref`` (default 20 µPa) on a sequential colormap.
        Shows acoustic energy independent of standing-wave phase.

    Parameters
    ----------
    pressure_field : NDArray (complex)
        2-D complex pressure field (ny × nx) from ``solve_2d_wave``.
    mesh_x, mesh_y : NDArray
        Physical grid coordinates (metres).
    horn_path : list of (x, y) or None
        Horn polygon vertices for outline overlay.  Pass None to skip.
    frequency : float
        Drive frequency in Hz (used in titles and k annotation).
    output_path : str
        Destination path for the PNG figure.
    figsize : tuple
        Matplotlib figure (w, h) in inches.  Default (12, 9) fits two panels.
    p_ref : float
        Reference pressure for dB SPL conversion.  Default 2×10⁻⁵ Pa.

    Returns
    -------
    None
    """
    import matplotlib
    matplotlib.use("Agg")  # headless-safe: no display required
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm, Normalize
    from matplotlib.patches import Polygon as MplPolygon

    p_real = pressure_field.real  # steady-state pressure at phase ωt=0
    p_mag = np.abs(pressure_field)
    k_val = 2.0 * np.pi * frequency / C

    # ── Panel 1: Real-part (sign-encoded, diverging) ─────────────────────────
    vmax_real = float(np.abs(p_real[~np.isnan(p_real)]).max())
    if vmax_real == 0.0:
        vmax_real = 1.0
    norm_real = TwoSlopeNorm(vmin=-vmax_real, vcenter=0.0, vmax=vmax_real)

    fig, (ax1, ax2) = plt.subplots(
        nrows=2, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.28},
    )

    im1 = ax1.pcolormesh(
        mesh_x, mesh_y, p_real,
        cmap="RdBu", norm=norm_real, shading="auto",
    )
    ax1.set_ylabel("y (m)")
    ax1.set_title(
        f"Pressure — real part  |  {frequency:.1f} Hz  "
        f"(k = {k_val:.2f} rad/m)"
    )
    ax1.set_aspect("equal")
    fig.colorbar(im1, ax=ax1, orientation="vertical", label="p (Pa)")

    # ── Panel 2: |p| in dB SPL ───────────────────────────────────────────────
    spl_db = 20.0 * np.log10(np.maximum(p_mag, p_ref) / p_ref)
    spl_min = float(np.nanmin(spl_db))
    spl_max = float(np.nanmax(spl_db))
    if spl_max - spl_min < 10.0:
        mid = (spl_max + spl_min) / 2.0
        spl_min, spl_max = mid - 10.0, mid + 10.0

    im2 = ax2.pcolormesh(
        mesh_x, mesh_y, spl_db,
        cmap="hot", norm=Normalize(vmin=spl_min, vmax=spl_max),
        shading="auto",
    )
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_title(
        f"Pressure amplitude |p|  |  {frequency:.1f} Hz  "
        f"(dB SPL, p_ref = {p_ref:.0e} Pa)"
    )
    ax2.set_aspect("equal")
    fig.colorbar(im2, ax=ax2, orientation="vertical", label="dB SPL")

    # ── Horn outline overlay ─────────────────────────────────────────────────
    def _overlay(ax):
        if horn_path is None:
            return
        coords = np.asarray(horn_path)
        if coords.ndim == 1:
            coords = coords.reshape(-1, 2)
        if not np.all(coords[0] == coords[-1]):
            coords = np.vstack([coords, coords[0]])
        ax.add_patch(MplPolygon(
            coords, closed=True, fill=False,
            edgecolor="cyan", linewidth=1.8, linestyle="--",
        ))

    _overlay(ax1)
    _overlay(ax2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_amplitude_db(
    pressure_field: NDArray[np.complex128],
    mesh_x: NDArray[np.float64],
    mesh_y: NDArray[np.float64],
    horn_path: list[tuple[float, float]] | None,
    frequency: float,
    output_path: str,
    p_ref: float = 2e-5,
    figsize: tuple[float, float] = (10, 5),
) -> None:
    """
    Single-panel **dB SPL amplitude colormap** of the 2-D pressure field —
    the acoustic energy distribution independent of standing-wave phase.

    Renders ``|p|`` as SPL (dB re 20 µPa) on a sequential colormap.
    This is the "energy view" counterpart to Panel 2 of
    ``plot_pressure_amplitude``.

    Parameters
    ----------
    pressure_field, mesh_x, mesh_y, horn_path, frequency, output_path
        Same as ``plot_pressure_amplitude``.
    p_ref : float
        Reference pressure for dB SPL.  Default 2×10⁻⁵ Pa.
    figsize : tuple
        Matplotlib figure (w, h) in inches.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Polygon as MplPolygon

    p_mag = np.abs(pressure_field)
    spl_db = 20.0 * np.log10(np.maximum(p_mag, p_ref) / p_ref)

    spl_min = float(np.nanmin(spl_db))
    spl_max = float(np.nanmax(spl_db))
    if spl_max - spl_min < 10.0:
        mid = (spl_max + spl_min) / 2.0
        spl_min, spl_max = mid - 10.0, mid + 10.0

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(
        mesh_x, mesh_y, spl_db,
        cmap="magma", norm=Normalize(vmin=spl_min, vmax=spl_max),
        shading="auto",
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"Pressure amplitude |p|  —  {frequency:.1f} Hz  "
        f"(dB SPL, p_ref = {p_ref:.0e} Pa)"
    )
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label="dB SPL")

    if horn_path is not None:
        coords = np.asarray(horn_path)
        if coords.ndim == 1:
            coords = coords.reshape(-1, 2)
        if not np.all(coords[0] == coords[-1]):
            coords = np.vstack([coords, coords[0]])
        ax.add_patch(MplPolygon(
            coords, closed=True, fill=False,
            edgecolor="cyan", linewidth=1.8, linestyle="--",
        ))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ─── Phase 3: Animated wave propagation ─────────────────────────────────────


def animate_wave_propagation(
    mesh_x: NDArray[np.float64],
    mesh_y: NDArray[np.float64],
    pressure_field: NDArray[np.complex128],
    horn_path: list[tuple[float, float]] | None,
    frequency: float,
    filename: str = "wavefront_animation.gif",
    n_frames: int = 48,
    interval_ms: int = 80,
    figsize: tuple[float, float] = (12, 6),
    p_ref: float = 2e-5,
) -> "matplotlib.animation.FuncAnimation":
    """
    Animate the steady-state pressure field by cycling through phase.

    The solver computes the complex phasor ``p(x,y)``.  This function renders
    the real pressure at each time step as::

        p(x,y,t) = Re[ p_complex(x,y) · exp(j · ω · t) ]

    The animation cycles through one full period (2π) in ``n_frames`` steps.

    Parameters
    ----------
    mesh_x, mesh_y : NDArray
        Physical grid coordinates (metres).
    pressure_field : NDArray (complex)
        2-D complex pressure field from ``solve_2d_wave_pml``.
    horn_path : list of (x, y) or None
        Horn polygon vertices for outline overlay.
    frequency : float
        Drive frequency in Hz (used to compute ω = 2πf).
    filename : str
        Output GIF path.  Default ``wavefront_animation.gif``.
    n_frames : int
        Number of frames per full cycle.  Default 48.
    interval_ms : int
        Delay between frames in milliseconds.  Default 80 (~12.5 fps).
    figsize : tuple
        Matplotlib figure (w, h) in inches.
    p_ref : float
        Reference pressure for dB SPL colour scale.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation object (also saved to ``filename``).
    """
    import matplotlib
    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt
    import matplotlib.animation as manimation
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Polygon as MplPolygon

    ny, nx = mesh_x.shape
    omega = 2.0 * np.pi * frequency

    # Real-part pressure at t=0
    p0 = pressure_field.real

    # Colour scale: fixed across all frames so it doesn't flicker
    vmax = float(np.abs(p0[~np.isnan(p0)]).max())
    if vmax == 0.0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize)

    def _horn_overlay():
        if horn_path is None:
            return
        coords = np.asarray(horn_path)
        if coords.ndim == 1:
            coords = coords.reshape(-1, 2)
        if not np.all(coords[0] == coords[-1]):
            coords = np.vstack([coords, coords[0]])
        ax.add_patch(MplPolygon(
            coords, closed=True, fill=False,
            edgecolor="white", linewidth=1.8, linestyle="--",
        ))

    im = ax.pcolormesh(
        mesh_x, mesh_y,
        p0,
        cmap="RdBu",
        norm=norm,
        shading="auto",
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"2-D Wavefront — {frequency:.1f} Hz  "
        f"(phase cycle  ωt = 0 → 2π)"
    )
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label="p (Pa)")
    _horn_overlay()

    frames: list[NDArray] = []
    for n in range(n_frames):
        phi = n * 2.0 * np.pi / n_frames
        p_frame = np.real(pressure_field * np.exp(1j * phi))
        frames.append(p_frame)

    def init():
        im.set_array(frames[0])
        return [im]

    def update(frame_idx):
        im.set_array(frames[frame_idx])
        phi_deg = int(round(frame_idx * 360.0 / n_frames))
        ax.set_title(
            f"2-D Wavefront — {frequency:.1f} Hz  "
            f"(ωt = {phi_deg}°)"
        )
        return [im]

    ani = manimation.FuncAnimation(
        fig, update,
        frames=range(n_frames),
        init_func=init,
        interval=interval_ms,
        blit=False,
    )
    ani.save(filename, writer="pillow", fps=1000 // interval_ms)
    plt.close(fig)
    return ani


def plot_animation_frames(
    snapshots: list[NDArray[np.float64]],
    mesh_x: NDArray[np.float64],
    mesh_y: NDArray[np.float64],
    time_steps: NDArray[np.float64],
    horn_path: list[tuple[float, float]] | None,
    frequency: float,
    filename: str = "wavefront_td_animation.gif",
    interval_ms: int = 60,
    figsize: tuple[float, float] = (12, 6),
) -> "matplotlib.animation.FuncAnimation":
    """
    Build an animated GIF from a list of time-domain pressure snapshots.

    Each snapshot is a 2-D real pressure field captured at a given time step.
    This is the visual output of ``solve_2d_wave_time_domain``.

    Parameters
    ----------
    snapshots : list of NDArray
        List of 2-D real pressure fields (Pa) captured at each output step.
    mesh_x, mesh_y : NDArray
        Physical grid coordinates (metres).
    time_steps : NDArray
        Physical time in seconds for each snapshot (same length as snapshots).
    horn_path : list of (x, y) or None
        Horn polygon vertices for outline overlay.
    frequency : float
        Drive / centre frequency in Hz (used in the title).
    filename : str
        Output GIF path.  Default ``wavefront_td_animation.gif``.
    interval_ms : int
        Delay between frames in milliseconds.
    figsize : tuple
        Matplotlib figure (w, h) in inches.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as manimation
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Polygon as MplPolygon

    # Collect all values to set a fixed colour scale across frames
    all_vals = np.concatenate([s.ravel() for s in snapshots])
    vmax = float(np.percentile(np.abs(all_vals), 99))
    if vmax == 0.0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize)

    def _horn_overlay():
        if horn_path is None:
            return
        coords = np.asarray(horn_path)
        if coords.ndim == 1:
            coords = coords.reshape(-1, 2)
        if not np.all(coords[0] == coords[-1]):
            coords = np.vstack([coords, coords[0]])
        ax.add_patch(MplPolygon(
            coords, closed=True, fill=False,
            edgecolor="white", linewidth=1.8, linestyle="--",
        ))

    im = ax.pcolormesh(
        mesh_x, mesh_y,
        snapshots[0],
        cmap="RdBu",
        norm=norm,
        shading="auto",
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label="p (Pa)")
    _horn_overlay()

    def init():
        im.set_array(snapshots[0])
        ax.set_title(f"{frequency:.1f} Hz  —  t = {time_steps[0]*1000:.1f} ms")
        return [im]

    def update(frame_idx):
        im.set_array(snapshots[frame_idx])
        t_ms = time_steps[frame_idx] * 1000.0
        ax.set_title(f"{frequency:.1f} Hz  —  t = {t_ms:.1f} ms")
        return [im]

    ani = manimation.FuncAnimation(
        fig, update,
        frames=range(len(snapshots)),
        init_func=init,
        interval=interval_ms,
        blit=False,
    )
    ani.save(filename, writer="pillow", fps=1000 // interval_ms)
    plt.close(fig)
    return ani


def solve_2d_wave_time_domain(
    mesh_x: NDArray[np.float64],
    mesh_y: NDArray[np.float64],
    source_x: float,
    source_y: float,
    frequency: float,
    boundary_mask: NDArray[np.bool_],
    n_steps: int = 600,
    output_interval: int = 6,
    pml_width: int = 15,
    kappa_max: float = 2.0,
    c0: float = C,
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
    """
    Time-domain 2-D FDTD (leapfrog) simulation with PML and a Gaussian-pulse
    source.

    Uses the first-order wave-system form::

        ∂p/∂t = -c² ∇·v          (pressure update)
        ∂v/∂t = -∇p / ρ₀          (velocity update, ρ₀ = 1.225 kg/m³)

    Discretised with a staggered leapfrog (FDTD) scheme.  The result is a
    broadband impulse response that shows wave energy propagating from the
    driver through the horn and out of the mouth.

    Parameters
    ----------
    mesh_x, mesh_y : NDArray
        Physical grid coordinates (metres).
    source_x, source_y : float
        Source (driver) position in metres.
    frequency : float
        Centre frequency of the Gaussian pulse in Hz.
    boundary_mask : NDArray[bool]
        True = wall (hard boundary), False = fluid.
    n_steps : int
        Number of time steps to run.  Default 600.
    output_interval : int
        Capture a snapshot every ``output_interval`` steps.  Default 6.
    pml_width, kappa_max : int, float
        PML layer parameters (passed to ``pml_damping_mask``).
    c0 : float
        Speed of sound.  Default 343 m/s.

    Returns
    -------
    tuple
        (time_steps, snapshots) where:
        - ``time_steps``: 1-D array of physical time (seconds) for each snapshot.
        - ``snapshots``: list of 2-D real pressure fields (Pa).
    """
    ny, nx = mesh_x.shape
    dx = float(mesh_x[0, 1] - mesh_x[0, 0])
    dy = float(mesh_y[1, 0] - mesh_y[0, 0])
    dt = 0.45 * min(dx, dy) / c0  # CFL-stable timestep

    # Source: Gaussian-enveloped sinusoid
    tau = 1.0 / (2.0 * np.pi * frequency)
    src_j = int(np.argmin(np.abs(mesh_x[0, :] - source_x)))
    src_i = int(np.argmin(np.abs(mesh_y[:, 0] - source_y)))

    # Pressure and velocity fields
    p = np.zeros((ny, nx), dtype=np.float64)
    vx = np.zeros((ny, nx), dtype=np.float64)  # x-velocity
    vy = np.zeros((ny, nx), dtype=np.float64)  # y-velocity

    # PML damping
    sigma = pml_damping_mask(ny, nx, pml_width=pml_width, kappa_max=kappa_max)
    rho0 = 1.225  # kg/m³

    # Walls: zero pressure (hard wall)
    interior = ~boundary_mask

    snapshots: list[NDArray[np.float64]] = []
    time_steps_out: list[float] = []

    for step in range(n_steps):
        # Gaussian pulse source
        tPhys = step * dt
        amp = np.exp(-((tPhys - 6.0 * tau) ** 2) / (2.0 * tau**2))
        source = amp * np.sin(2.0 * np.pi * frequency * tPhys)
        p[src_i, src_j] = source

        # Enforce wall BCs (zero pressure inside walls every step)
        p[boundary_mask] = 0.0

        # Velocity update (leapfrog)
        dvx = -(dt / rho0 / dx) * (p[:, 1:] - p[:, :-1])
        dvy = -(dt / rho0 / dy) * (p[1:, :] - p[:-1, :])
        vx[:, :-1] += dvx
        vy[:-1, :] += dvy

        # Apply PML damping to velocity
        vx *= np.exp(-sigma * c0 * dt / (rho0 * dx))
        vy *= np.exp(-sigma * c0 * dt / (rho0 * dy))

        # Re-enforce wall BCs after velocity update
        vx[boundary_mask] = 0.0
        vy[boundary_mask] = 0.0

        # Pressure update
        # vx has shape (ny, nx); interior x-velocity faces are vx[:, 1:-1]
        # vy has shape (ny, nx); interior y-velocity faces are vy[1:-1, :]
        # dp has shape (ny-2, nx-2); p[1:-1, 1:-1] is the interior pressure
        dp = -(rho0 * c0**2 * dt) * (
            (vx[:, 1:-1] - vx[:, :-2]) / dx
            + (vy[1:-1, :] - vy[:-2, :]) / dy
        )
        p[1:-1, 1:-1] += dp

        # Re-enforce wall BCs
        p[boundary_mask] = 0.0

        # Output snapshot
        if step % output_interval == 0:
            snapshots.append(p.copy())
            time_steps_out.append(tPhys)

    return (
        np.array(time_steps_out, dtype=np.float64),
        snapshots,
    )


def solve_2d_wave_time_domain_pml(
    mesh_x: NDArray[np.float64],
    mesh_y: NDArray[np.float64],
    source_x: float,
    source_y: float,
    frequency: float,
    boundary_mask: NDArray[np.bool_],
    n_steps: int = 600,
    output_interval: int = 6,
    pml_width: int = 15,
    kappa_max: float = 2.0,
    c0: float = C,
    rho0: float = 1.225,
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
    """
    Time-domain 2-D FDTD with a split-field PML (more accurate than the
    simplified sigma-damping of ``solve_2d_wave_time_domain``).

    Parameters
    ----------
    mesh_x, mesh_y, source_x, source_y, frequency, boundary_mask
        Same as ``solve_2d_wave_time_domain``.
    n_steps, output_interval
        Same as ``solve_2d_wave_time_domain``.
    pml_width, kappa_max
        PML layer parameters.
    c0 : float
        Speed of sound.  Default 343 m/s.
    rho0 : float
        Air density.  Default 1.225 kg/m³.

    Returns
    -------
    tuple
        (time_steps, snapshots) — same as ``solve_2d_wave_time_domain``.
    """
    ny, nx = mesh_x.shape
    dx = float(mesh_x[0, 1] - mesh_x[0, 0])
    dy = float(mesh_y[1, 0] - mesh_y[0, 0])
    dt = 0.45 * min(dx, dy) / c0

    # CPML parameters
    w = min(pml_width, ny // 4, nx // 4)
    ramp = (np.arange(w, dtype=np.float64) + 1) / w
    profile = kappa_max * ramp**2

    sigma_x = np.zeros((ny, nx), dtype=np.float64)
    sigma_y = np.zeros((ny, nx), dtype=np.float64)
    for s in range(w):
        sigma_x[:, s] = profile[s]
        sigma_x[:, nx - 1 - s] = profile[s]
        sigma_y[s, :] = np.maximum(sigma_y[s, :], profile[s])
        sigma_y[ny - 1 - s, :] = np.maximum(sigma_y[ny - 1 - s, :], profile[s])

    # CPML convolved correction arrays
    psi_vx_y = np.zeros((ny, nx - 1), dtype=np.float64)
    psi_vy_x = np.zeros((ny - 1, nx), dtype=np.float64)

    # Source
    tau = 1.0 / (2.0 * np.pi * frequency)
    src_j = int(np.argmin(np.abs(mesh_x[0, :] - source_x)))
    src_i = int(np.argmin(np.abs(mesh_y[:, 0] - source_y)))

    p = np.zeros((ny, nx), dtype=np.float64)
    vx = np.zeros((ny, nx), dtype=np.float64)
    vy = np.zeros((ny, nx), dtype=np.float64)

    snapshots: list[NDArray[np.float64]] = []
    time_steps_out: list[float] = []

    for step in range(n_steps):
        tPhys = step * dt
        amp = np.exp(-((tPhys - 6.0 * tau) ** 2) / (2.0 * tau**2))
        source = amp * np.sin(2.0 * np.pi * frequency * tPhys)
        p[src_i, src_j] = source
        p[boundary_mask] = 0.0

        # vx update with CPML
        for i in range(ny):
            for j in range(nx - 1):
                if boundary_mask[i, j] or boundary_mask[i, j + 1]:
                    continue
                sig_x = sigma_x[i, j]
                if sig_x > 0.0:
                    psi_vx_y[i, j] = (
                        (sig_x * dt / rho0) * (p[i, j + 1] - p[i, j]) / dx
                        + np.exp(-sig_x * dt / rho0) * psi_vx_y[i, j]
                    )
                    d_vx = -(dt / rho0) * ((p[i, j + 1] - p[i, j]) / dx + psi_vx_y[i, j])
                else:
                    d_vx = -(dt / rho0) * ((p[i, j + 1] - p[i, j]) / dx)
                vx[i, j] += d_vx

        # vy update with CPML
        for i in range(ny - 1):
            for j in range(nx):
                if boundary_mask[i, j] or boundary_mask[i + 1, j]:
                    continue
                sig_y = sigma_y[i, j]
                if sig_y > 0.0:
                    psi_vy_x[i, j] = (
                        (sig_y * dt / rho0) * (p[i + 1, j] - p[i, j]) / dy
                        + np.exp(-sig_y * dt / rho0) * psi_vy_x[i, j]
                    )
                    d_vy = -(dt / rho0) * ((p[i + 1, j] - p[i, j]) / dy + psi_vy_x[i, j])
                else:
                    d_vy = -(dt / rho0) * ((p[i + 1, j] - p[i, j]) / dy)
                vy[i, j] += d_vy

        # Enforce wall BCs on velocity
        vx[boundary_mask] = 0.0
        vy[boundary_mask] = 0.0

        # Pressure update
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                if boundary_mask[i, j]:
                    continue
                dp = -(rho0 * c0**2 * dt) * (
                    (vx[i, j] - vx[i, j - 1]) / dx
                    + (vy[i, j] - vy[i - 1, j]) / dy
                )
                p[i, j] += dp

        p[boundary_mask] = 0.0

        if step % output_interval == 0:
            snapshots.append(p.copy())
            time_steps_out.append(tPhys)

    return (
        np.array(time_steps_out, dtype=np.float64),
        snapshots,
    )


# ─── WavefrontGrid.animate() ─────────────────────────────────────────────────


def WavefrontGrid_animate(
    self,
    mesh_x: NDArray[np.float64],
    mesh_y: NDArray[np.float64],
    horn_path: list[tuple[float, float]] | None = None,
    filename: str = "wavefront_animation.gif",
    n_frames: int = 48,
    interval_ms: int = 80,
    figsize: tuple[float, float] = (12, 6),
    p_ref: float = 2e-5,
    use_pml: bool = True,
    pml_width: int = 15,
    kappa_max: float = 2.0,
    source_x: float | None = None,
    source_y: float | None = None,
) -> "matplotlib.animation.FuncAnimation":
    """
    One-liner convenience method: solve the 2-D wave equation and animate
    the steady-state standing-wave pattern by cycling through phase.

    This method:
      1. Resolves the source position (``source_x/y`` or ``self.source_x/y``).
      2. Calls ``solve_2d_wave_pml`` (or ``solve_2d_wave``) to compute the
         complex pressure field.
      3. Calls ``animate_wave_propagation`` to produce the GIF.

    Parameters
    ----------
    mesh_x, mesh_y : NDArray
        Physical grid coordinates (metres) — same arrays used to build the
        boundary mask.
    horn_path : list of (x, y) or None
        Optional horn polygon vertices for the outline overlay.
    filename : str
        Output GIF path.  Default ``wavefront_animation.gif``.
    n_frames : int
        Frames per full phase cycle.  Default 48.
    interval_ms : int
        Milliseconds between frames.  Default 80.
    figsize : tuple
        Matplotlib figure (w, h) in inches.
    p_ref : float
        Reference pressure for dB SPL.  Default 2×10⁻⁵ Pa.
    use_pml : bool
        Use PML absorbing boundary (``solve_2d_wave_pml``).  Default True.
    pml_width, kappa_max : int, float
        PML parameters when ``use_pml`` is True.
    source_x, source_y : float | None
        Override the driver position for this run.  If None the method uses
        ``self.source_x`` / ``self.source_y``.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation object (also saved to ``filename``).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as manimation

    sx = source_x if source_x is not None else self.source_x
    sy = source_y if source_y is not None else self.source_y

    if sx is None or sy is None:
        raise ValueError(
            "WavefrontGrid.animate() requires a source position. "
            "Either set source_x/source_y on the grid or pass them explicitly."
        )

    # Use stored boundary mask if available; fall back to an empty (no-wall) mask
    bmask = self.boundary_mask if self.boundary_mask is not None else np.zeros((self.ny, self.nx), dtype=bool)
    # Resolve horn outline — stored coords take priority over the horn_path arg
    hpath: list[tuple[float, float]] | None = None
    if self.horn_coords is not None:
        hpath = [tuple(c) for c in self.horn_coords]
    if horn_path is not None:
        hpath = horn_path  # explicit arg overrides stored coords

    if use_pml:
        result = solve_2d_wave_pml(
            mesh_x, mesh_y, sx, sy,
            self.frequency, bmask,
            pml_width=pml_width, kappa_max=kappa_max,
        )
    else:
        result = solve_2d_wave(
            mesh_x, mesh_y, sx, sy, self.frequency, bmask,
        )

    self.pressure_field = result.pressure_field

    return animate_wave_propagation(
        mesh_x=mesh_x,
        mesh_y=mesh_y,
        pressure_field=self.pressure_field,
        horn_path=hpath,
        frequency=self.frequency,
        filename=filename,
        n_frames=n_frames,
        interval_ms=interval_ms,
        figsize=figsize,
        p_ref=p_ref,
    )


# ─── Phase 2: Interactive wall geometry editor ─────────────────────────────────


def edit_horn_geometry(
    coords: list[tuple[float, float]] | NDArray[np.float64],
    title: str = "Horn Wall Editor — click to add, drag to move, right-click to delete",
    figsize: tuple[float, float] = (12, 8),
    wall_color: str = "#e63946",
    vertex_color: str = "#f4a261",
    vertex_size: float = 120,
    line_width: float = 2.0,
    grid_color: str = "#cccccc",
    background_color: str = "#0d1b2a",
) -> list[tuple[float, float]]:
    """
    Open an interactive matplotlib figure that lets the user edit horn-wall
    vertices.

    Controls
    --------
    - **Left-click** on empty space → add a new vertex there.
    - **Left-click + drag** an existing vertex → move it.
    - **Right-click** on a vertex → delete that vertex.
    - **Enter** in the figure window → confirm and close.
    - **Escape** → cancel and return the original coordinates.

    Parameters
    ----------
    coords : list of (x, y) or ndarray
        Initial wall polygon vertices in metres.
    title : str
        Figure window title.
    figsize, wall_color, vertex_color, vertex_size, line_width
        Visual styling passed to matplotlib.
    grid_color, background_color
        Axes grid and figure background colours.

    Returns
    -------
    list of (x, y)
        Updated vertex list.  Cancelling (Escape) returns the original list.

    Example
    -------
    >>> coords = [(0,0), (0.5,0), (0.5,0.3), (0,0.3)]
    >>> new_coords = edit_horn_geometry(coords)
    >>> # new_coords may have different vertices after user editing
    """
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import MouseEvent, KeyEvent

    # Convert to mutable list of [x, y] floats
    _verts: list[list[float]] = [list(c) for c in coords]

    # Hit-test radius in display pixels
    HIT_RADIUS_PX = 12

    # ── Setup figure ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, facecolor=background_color)
    ax.set_facecolor(background_color)
    ax.set_title(title, color="white", fontsize=11)
    ax.set_xlabel("x (m)", color="white")
    ax.set_ylabel("y (m)", color="white")
    ax.tick_params(colors="white")
    ax.spines["top"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.spines["left"].set_color("#444444")
    ax.spines["right"].set_color("#444444")
    ax.grid(True, color=grid_color, alpha=0.3, linewidth=0.5)
    ax.set_aspect("equal")

    if _verts:
        xs = [v[0] for v in _verts]
        ys = [v[1] for v in _verts]
        x_pad = (max(xs) - min(xs)) * 0.1 if xs else 0.1
        y_pad = (max(ys) - min(ys)) * 0.1 if ys else 0.1
        ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
        ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)

    # ── Drawing state ─────────────────────────────────────────────────────────
    _drag_idx: int | None = None
    _pending_vertex: list[float] | None = None
    _done = False

    # ── Redraw function ───────────────────────────────────────────────────────
    def _redraw():
        # Remove old editor artists
        for artist in list(ax.get_children()):
            if getattr(artist, "_is_horn_editor", False):
                artist.remove()

        xs_list = [v[0] for v in _verts]
        ys_list = [v[1] for v in _verts]

        if len(_verts) >= 2:
            closed_xs = xs_list + [xs_list[0]]
            closed_ys = ys_list + [ys_list[0]]
            line, = ax.plot(
                closed_xs, closed_ys,
                color=wall_color, linewidth=line_width,
                zorder=1,
            )
            line._is_horn_editor = True  # type: ignore[attr-defined]

        if _verts:
            scat = ax.scatter(
                xs_list, ys_list,
                s=vertex_size, c=vertex_color,
                zorder=2, edgecolors="white", linewidths=0.8,
            )
            scat._is_horn_editor = True  # type: ignore[attr-defined]
            for i, (vx, vy) in enumerate(_verts):
                ax.text(
                    vx, vy, str(i),
                    color="white", fontsize=7,
                    ha="center", va="center",
                    zorder=3,
                )
        fig.canvas.draw_idle()

    _redraw()

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _display_to_data(ex: float, ey: float) -> tuple[float, float]:
        return ax.transData.inverted().transform((ex, ey))

    def _nearest_vertex(
        mx: float, my: float,
    ) -> tuple[int | None, float]:
        if not _verts:
            return None, float("inf")
        dists = [np.hypot(v[0] - mx, v[1] - my) for v in _verts]
        min_idx = int(np.argmin(dists))
        hit = _px_to_data_radius(HIT_RADIUS_PX)
        return (min_idx, dists[min_idx]) if dists[min_idx] < hit else (None, float("inf"))

    def _px_to_data_radius(px: int) -> float:
        xlim = ax.get_xlim()
        x_range = xlim[1] - xlim[0]
        fig_w = fig.get_size_inches()[0] * fig.dpi
        return px * x_range / fig_w

    # ── Event handlers ─────────────────────────────────────────────────────────
    def _on_press(event: MouseEvent):
        nonlocal _drag_idx, _pending_vertex
        if event.button == 3:  # right-click → delete
            mx, my = _display_to_data(event.x, event.y)
            idx, _ = _nearest_vertex(mx, my)
            if idx is not None:
                _verts.pop(idx)
                _redraw()
            return
        if event.button != 1:
            return
        mx, my = _display_to_data(event.x, event.y)
        idx, _ = _nearest_vertex(mx, my)
        if idx is not None:
            _drag_idx = idx
        else:
            _pending_vertex = [mx, my]

    def _on_motion(event: MouseEvent):
        nonlocal _drag_idx
        if _drag_idx is None:
            return
        mx, my = _display_to_data(event.x, event.y)
        if _drag_idx < len(_verts):
            _verts[_drag_idx] = [mx, my]
            _redraw()

    def _on_release(event: MouseEvent):
        nonlocal _drag_idx, _pending_vertex
        if event.button != 1:
            return
        if _drag_idx is not None:
            mx, my = _display_to_data(event.x, event.y)
            if _drag_idx < len(_verts):
                _verts[_drag_idx] = [mx, my]
                _redraw()
            _drag_idx = None
        elif _pending_vertex is not None:
            _verts.append(_pending_vertex)
            _pending_vertex = None
            _redraw()

    def _on_key(event: KeyEvent):
        nonlocal _done
        if event.key in ("enter", "return"):
            _done = True
            plt.close(fig)
        elif event.key == "escape":
            _done = False
            plt.close(fig)

    cid_press = fig.canvas.mpl_connect("button_press_event", _on_press)
    cid_release = fig.canvas.mpl_connect("button_release_event", _on_release)
    cid_motion = fig.canvas.mpl_connect("motion_notify_event", _on_motion)
    cid_key = fig.canvas.mpl_connect("key_press_event", _on_key)

    # Info overlay
    info = ax.text(
        0.01, 0.99,
        "L-click: add  |  drag: move  |  R-click: delete  |  Enter: done  |  Esc: cancel",
        transform=ax.transAxes,
        color="#aaaaaa", fontsize=8,
        va="top", ha="left",
    )
    info._is_horn_editor = True  # type: ignore[attr-defined]

    plt.show(block=True)

    # Clean up connections
    fig.canvas.mpl_disconnect(cid_press)
    fig.canvas.mpl_disconnect(cid_release)
    fig.canvas.mpl_disconnect(cid_motion)
    fig.canvas.mpl_disconnect(cid_key)

    if _done:
        return [tuple(v) for v in _verts]
    return [tuple(v) for v in coords]  # cancelled → original


def edit_horn_geometry_from_yaml(
    yaml_path: str | Path,
    output_yaml_path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 8),
) -> dict[str, Any]:
    """
    Load horn geometry from a YAML file, open the interactive editor, and
    optionally save the modified geometry back to a YAML file.

    Parameters
    ----------
    yaml_path : str or Path
        Path to a pyhorn source or project YAML (see ``load_horn_geometry``).
    output_yaml_path : str or Path, optional
        Destination YAML path.  If None the original file is **not** overwritten;
        the caller receives the updated geometry dict and decides what to do
        with it.
    title : str, optional
        Figure window title.  If None a default is used.
    figsize : tuple
        Matplotlib figure (w, h) in inches.

    Returns
    -------
    dict
        Updated geometry dict with keys:
        ``coords``  — list of (x, y) vertices in metres.
        ``source_x``, ``source_y`` — driver position (unchanged).
        ``name``    — geometry name.
        ``saved``   — bool indicating whether the file was written to disk.

    Example
    -------
    >>> result = edit_horn_geometry_from_yaml("source/my_horn.yaml")
    >>> if result["saved"]:
    ...     print("Saved!")
    >>> new_coords = result["coords"]
    """
    import yaml

    geo = load_horn_geometry(yaml_path)
    coords = geo["coords"]

    if title is None:
        title = f"Horn Wall Editor — {geo.get('name', Path(yaml_path).stem)}"

    updated_coords = edit_horn_geometry(
        coords=coords,
        title=title,
        figsize=figsize,
    )

    geo["coords"] = updated_coords

    saved = False
    if output_yaml_path is not None:
        out_path = Path(output_yaml_path)
        out_data: dict[str, Any] = {
            "name": geo.get("name", out_path.stem),
            "coordinates": [[float(x), float(y)] for x, y in updated_coords],
        }
        if geo.get("source_x") is not None:
            out_data["source_x"] = geo["source_x"]
        if geo.get("source_y") is not None:
            out_data["source_y"] = geo["source_y"]
        if geo.get("enclosure_dims") is not None:
            out_data["enclosure_dims"] = geo["enclosure_dims"]

        with open(out_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(out_data, fh, default_flow_style=False, sort_keys=False)
        saved = True

    geo["saved"] = saved
    return geo


# ─── Interactive Edit + Wave Simulation ───────────────────────────────────────


def edit_horn_geometry_and_simulate(
    yaml_path: str | Path,
    output_yaml_path: str | Path | None = None,
    output_png_path: str | Path | None = None,
    frequency: float = 1000.0,
    grid_size: int = 200,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 8),
) -> dict[str, Any]:
    """
    Load horn geometry from YAML, open the interactive wall editor, run the
    2-D Helmholtz wave solver on the edited geometry, and save the pressure-
    field plot as a PNG.

    This gives a tight edit→simulate→view loop: the user moves wall vertices
    interactively, presses Enter to confirm, and immediately sees the acoustic
    standing-wave pattern at ``frequency`` Hz for the updated geometry.

    Parameters
    ----------
    yaml_path : str or Path
        Path to a pyhorn source or project YAML.
    output_yaml_path : str or Path, optional
        Destination for the edited geometry YAML.  If None the geometry is not
        written to disk (only returned in the result dict).
    output_png_path : str or Path, optional
        Destination for the pressure-field PNG.  If None defaults to
        ``{yaml_stem}_wavefront_{freq}Hz.png`` in the same directory as
        ``yaml_path``.
    frequency : float
        Drive frequency in Hz for the wave simulation.  Default 1000 Hz.
    grid_size : int
        Square grid dimension (cells per side).  Higher = more resolution but
        slower.  Default 200.  Typical run time: 0.5–3 s per solve.
    title : str, optional
        Matplotlib editor window title.  If None a default is used.
    figsize : tuple
        Matplotlib figure (w, h) in inches for the editor window.

    Returns
    -------
    dict with keys:
        ``coords``    — edited polygon vertices in metres.
        ``source_x``  — driver x position (metres, or None).
        ``source_y``  — driver y position (metres, or None).
        ``name``      — geometry name.
        ``saved``     — bool: whether the YAML was written to disk.
        ``png_path``  — Path to the saved PNG (always returned, even if None
                        was passed — resolved to the default path).

    Example
    -------
    >>> result = edit_horn_geometry_and_simulate(
    ...     "source/my_horn.yaml",
    ...     frequency=800.0,
    ...     output_png_path="wavefront_800Hz.png",
    ... )
    >>> print(f"Saved to {result['png_path']}")
    """
    import matplotlib
    matplotlib.use("Agg")  # must be set BEFORE pyplot import
    import matplotlib.pyplot as plt

    yaml_path = Path(yaml_path)
    geo = load_horn_geometry(yaml_path)
    coords = geo["coords"]

    if title is None:
        title = f"Edit walls → then press Enter to simulate @ {frequency} Hz"

    # ── Interactive geometry edit ────────────────────────────────────────────
    updated_coords = edit_horn_geometry(
        coords=coords,
        title=title,
        figsize=figsize,
    )

    # ── Build simulation grid ────────────────────────────────────────────────
    all_x = [c[0] for c in updated_coords]
    all_y = [c[1] for c in updated_coords]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_pad = (x_max - x_min) * 0.2 + 0.05
    y_pad = (y_max - y_min) * 0.2 + 0.05

    x_edges = np.linspace(x_min - x_pad, x_max + x_pad, grid_size + 1)
    y_edges = np.linspace(y_min - y_pad, y_max + y_pad, grid_size + 1)
    mesh_x, mesh_y = np.meshgrid(x_edges, y_edges)

    # ── Boundary mask ──────────────────────────────────────────────────────
    walls = boundary_condition_mask(updated_coords, mesh_x, mesh_y)

    # ── Source position ─────────────────────────────────────────────────────
    src_x = geo.get("source_x", (x_min + x_max) / 2)
    src_y = geo.get("source_y", (y_min + y_max) / 2)

    # ── Solve + plot ────────────────────────────────────────────────────────
    grid = solve_2d_wave_pml(
        mesh_x, mesh_y, src_x, src_y, frequency, walls,
    )

    if output_png_path is None:
        output_png_path = str(yaml_path.with_suffix("").parent / (
            f"{yaml_path.stem}_wavefront_{int(frequency)}Hz.png"
        ))
    else:
        output_png_path = str(output_png_path)

    plot_pressure_amplitude(
        pressure_field=grid.pressure_field,
        mesh_x=mesh_x,
        mesh_y=mesh_y,
        horn_path=updated_coords if updated_coords else None,
        frequency=frequency,
        output_path=output_png_path,
        figsize=(12, 9),
    )

    # ── Save YAML if requested ──────────────────────────────────────────────
    saved = False
    if output_yaml_path is not None:
        import yaml
        out_data: dict[str, Any] = {
            "name": geo.get("name", Path(output_yaml_path).stem),
            "coordinates": [[float(x), float(y)] for x, y in updated_coords],
        }
        if geo.get("source_x") is not None:
            out_data["source_x"] = geo["source_x"]
        if geo.get("source_y") is not None:
            out_data["source_y"] = geo["source_y"]
        if geo.get("enclosure_dims") is not None:
            out_data["enclosure_dims"] = geo["enclosure_dims"]
        with open(output_yaml_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(out_data, fh, default_flow_style=False, sort_keys=False)
        saved = True

    result = dict(geo)
    result["coords"] = updated_coords
    result["saved"] = saved
    result["png_path"] = Path(output_png_path)
    return result


# ─── Wavefront-derived Polar Directivity Panel ───────────────────────────────


def plot_wavefront_polar(
    pressure_field: NDArray[np.complex128],
    mesh_x: NDArray[np.float64],
    mesh_y: NDArray[np.float64],
    boundary_mask: NDArray[np.bool_],
    frequency: float,
    ax: "plt.Axes",
    p_ref: float = 2e-5,
    n_bins: int = 72,
) -> None:
    """
    Extract the mouth radiation pattern from a 2-D wavefront pressure field
    and render it as a polar directivity panel.

    The mouth is identified as the exterior wall/fluid boundary — wall cells
    that have at least one fluid neighbour AND sit at the topmost extreme of the
    horn geometry (highest y).  The mouth centre is taken as the centroid of
    these cells.  Pressure amplitudes at each cell are converted to dB SPL
    relative to the on-axis maximum and plotted at the corresponding polar
    angle.

    This gives the simulated directivity radiation pattern of the horn mouth
    at the chosen frequency — as opposed to the Levine/Inglis piston model used
    in ``plot_polar_response``.

    Parameters
    ----------
    pressure_field  : complex 2-D array (ny × nx) — solved pressure field
    mesh_x, mesh_y  : 2-D coordinate arrays (metres)
    boundary_mask   : boolean wall mask (True = wall)
    frequency       : drive frequency in Hz
    ax              : matplotlib Axes with ``projection='polar'``
    p_ref           : reference pressure for dB SPL (default 20 µPa)
    n_bins          : number of angular bins for averaging (default 72 → 5° bins)
    """
    import matplotlib.pyplot as plt

    ny, nx = pressure_field.shape
    p_mag = np.abs(pressure_field)

    # ── Identify exterior wall/fluid boundary cells ────────────────────────────
    # Wall cells that have at least one fluid neighbour
    exterior_mask = np.zeros_like(boundary_mask)
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            if not boundary_mask[i, j]:
                continue
            # Check if any neighbour is fluid
            if (not boundary_mask[i - 1, j] or not boundary_mask[i + 1, j] or
                    not boundary_mask[i, j - 1] or not boundary_mask[i, j + 1]):
                exterior_mask[i, j] = True

    exterior_y = mesh_y[exterior_mask]
    if exterior_y.size == 0:
        ax.set_title("Polar (wavefront) — no exterior boundary found", fontsize=6)
        return

    # Mouth = topmost exterior cells (highest y ≈ open end of horn)
    y_threshold = np.percentile(exterior_y, 90)  # top 10% of exterior cells
    mouth_mask = exterior_mask & (mesh_y >= y_threshold)

    # Fallback: if mouth region is too small, use all exterior cells
    if np.sum(mouth_mask) < 5:
        mouth_mask = exterior_mask

    mouth_x = mesh_x[mouth_mask]
    mouth_y = mesh_y[mouth_mask]
    mouth_p = p_mag[mouth_mask]

    if mouth_x.size == 0:
        ax.set_title("Polar (wavefront) — mouth region not found", fontsize=6)
        return

    # Mouth centre
    cx = float(np.mean(mouth_x))
    cy = float(np.mean(mouth_y))

    # Polar angles from mouth centre (0° = right, 90° = top in data coords)
    dx = mouth_x - cx
    dy = mouth_y - cy
    angles_raw = np.arctan2(dy, dx)          # radians, -π to π
    angles_deg = np.rad2deg(angles_raw)       # -180 to 180

    # Convert to acoustic polar convention:
    #   0° = forward (right, +x), 90° = up, 180°/-180° = backward
    # We use 0° at top (north), clockwise → match _COLORS / plot_polar_response
    # Shift: 0° at top, clockwise: theta = 90° - angle_deg
    angles_polar = 90.0 - angles_deg        # now roughly 0-360 with top=0

    # Normalise to [0, 360)
    angles_polar = angles_polar % 360.0

    # ── Bin by angle, compute mean SPL ────────────────────────────────────────
    bin_edges = np.linspace(0, 360, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    spl_vals = 20.0 * np.log10(np.maximum(mouth_p, p_ref) / p_ref)

    # Clip to reasonable range relative to the data
    spl_min = np.min(spl_vals)
    spl_max = np.max(spl_vals)
    on_axis_db = spl_max  # approximate on-axis level

    rel_db = spl_vals - on_axis_db   # dB relative to approximate on-axis

    # Bin-average
    binned_db = np.zeros(n_bins)
    for k in range(n_bins):
        in_bin = (angles_polar >= bin_edges[k]) & (angles_polar < bin_edges[k + 1])
        if np.any(in_bin):
            binned_db[k] = np.mean(rel_db[in_bin])
        else:
            binned_db[k] = np.nan

    # Fill NaNs by interpolation
    valid = np.isfinite(binned_db)
    if np.any(valid):
        binned_db = np.interp(bin_centers, bin_centers[valid], binned_db[valid])

    # Mirror to full 360° (2-D planar radiation is symmetric about the horn axis)
    all_angles = np.concatenate([bin_centers, bin_centers + 180.0]) % 360.0
    all_db = np.concatenate([binned_db, binned_db[::-1]])

    # Sort by angle for clean polar plot
    sort_idx = np.argsort(all_angles)
    all_angles = all_angles[sort_idx]
    all_db = all_db[sort_idx]

    # Clip to physical range
    all_db = np.clip(all_db, -40.0, 5.0)

    theta = np.deg2rad(all_angles)
    r = -all_db   # 0 dB rel → centre, -30 dB → outer ring

    ax.fill(theta, r, color="#0f766e", alpha=0.15, zorder=1)
    ax.plot(theta, r, color="#0f766e", linewidth=1.0, zorder=2)

    # Reference circles
    for db_ref in [-10.0, -20.0, -30.0]:
        ax.plot(theta, [-db_ref] * len(theta), color="#9ca3af",
                linewidth=0.3, linestyle="--", alpha=0.6, zorder=0)
        ax.text(np.deg2rad(90), -db_ref + 0.3, f"{db_ref:.0f}",
                fontsize=5, color="#9ca3af", ha="left", va="center", zorder=3)

    # On-axis marker
    ax.plot(0, 0, "o", color="#dc2626", markersize=3, zorder=4)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(270)
    ax.set_rlim(0, 30)
    ax.set_rgrids([0, 5, 10, 15, 20, 25, 30],
                  ["0", "−5", "−10", "−15", "−20", "−25", "−30"],
                  fontsize=5, color="#6b7280")
    ax.tick_params(labelsize=5, pad=1)
    ax.set_title(
        f"Polar (Wavefront) @ {frequency:.0f} Hz\n"
        f"(dB rel to on-axis, 2-D)",
        fontsize=6, pad=8, fontweight="medium",
    )


# Monkey-patch onto WavefrontGrid so it reads naturally as a method
WavefrontGrid.animate = WavefrontGrid_animate
