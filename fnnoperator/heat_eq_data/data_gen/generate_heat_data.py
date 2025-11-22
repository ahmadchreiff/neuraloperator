"""
Synthetic heat equation data generator for 1D, 2D, and 3D domains.

This script solves the heat equation on 1D, 2D, or 3D domains with Dirichlet
boundary conditions for many randomly sampled initial states and thermal
diffusivities. The generated trajectories are split into train (80%),
test (10%), and validate (10%) datasets, stored as NumPy arrays in
`heat_eq_data/data` as train.npz, test.npz, and validate.npz.

The heat equation in d dimensions:
    ∂u/∂t = α ∇²u

Where ∇² is the Laplacian operator.

Example:
    # 1D
    python heat_eq_data/data_gen/generate_heat_data.py --dimension 1 --nx 128
    
    # 2D
    python heat_eq_data/data_gen/generate_heat_data.py --dimension 2 --nx 64 --ny 64
    
    # 3D
    python heat_eq_data/data_gen/generate_heat_data.py --dimension 3 --nx 32 --ny 32 --nz 32
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class HeatEqConfig:
    dimension: int = 1  # 1, 2, or 3
    length: float = 1.0  # Domain length in each dimension
    nx: int = 128  # Spatial resolution in x-direction
    ny: int = 64  # Spatial resolution in y-direction (2D/3D)
    nz: int = 32  # Spatial resolution in z-direction (3D)
    nt: int = 64  # Temporal resolution
    total_time: float = 0.008  # Adjusted for stability (CFL < 0.5)
    diffusivity_min: float = 0.05
    diffusivity_max: float = 0.2
    dataset_size: int = 256
    noise_std: float = 0.0
    seed: int | None = None
    output_dir: Path = Path("heat_eq_data/data")

    @property
    def dt(self) -> float:
        if self.nt < 2:
            raise ValueError("nt must be at least 2 to define a time step.")
        return self.total_time / (self.nt - 1)

    @property
    def dx(self) -> float:
        if self.nx < 2:
            raise ValueError("nx must be at least 2 to define a spatial step.")
        return self.length / (self.nx - 1)
    
    @property
    def dy(self) -> float:
        if self.ny < 2:
            raise ValueError("ny must be at least 2 to define a spatial step.")
        return self.length / (self.ny - 1)
    
    @property
    def dz(self) -> float:
        if self.nz < 2:
            raise ValueError("nz must be at least 2 to define a spatial step.")
        return self.length / (self.nz - 1)
    
    @property
    def spatial_shape(self) -> tuple[int, ...]:
        """Return spatial shape tuple based on dimension."""
        if self.dimension == 1:
            return (self.nx,)
        elif self.dimension == 2:
            return (self.ny, self.nx)  # (y, x) for 2D array indexing
        elif self.dimension == 3:
            return (self.nz, self.ny, self.nx)  # (z, y, x) for 3D array indexing
        else:
            raise ValueError(f"Dimension must be 1, 2, or 3, got {self.dimension}")

    def validate(self) -> None:
        if self.dimension not in [1, 2, 3]:
            raise ValueError(f"Dimension must be 1, 2, or 3, got {self.dimension}.")
        if self.diffusivity_min <= 0 or self.diffusivity_max <= 0:
            raise ValueError("Diffusivity bounds must be positive.")
        if self.diffusivity_min > self.diffusivity_max:
            raise ValueError("diffusivity_min cannot exceed diffusivity_max.")
        if self.dataset_size <= 0:
            raise ValueError("dataset_size must be positive.")
        if self.noise_std < 0:
            raise ValueError("noise_std cannot be negative.")
        
        # CFL condition for explicit scheme: depends on dimension
        # For d-dimensional heat equation: CFL = α * dt / (dx²) * d < 0.5
        # We use the minimum grid spacing for stability check
        if self.dimension == 1:
            min_dx = self.dx
        elif self.dimension == 2:
            min_dx = min(self.dx, self.dy)
        else:  # 3D
            min_dx = min(self.dx, self.dy, self.dz)
        
        # Multi-dimensional CFL: more restrictive
        cfl_factor = self.dimension  # Need more restrictive CFL for higher dimensions
        cfl = self.diffusivity_max * self.dt / (min_dx ** 2) * cfl_factor
        cfl_limit = 0.5 / self.dimension  # More restrictive for higher dimensions
        
        if cfl >= cfl_limit:
            raise ValueError(
                f"Unstable explicit scheme for {self.dimension}D: "
                "decrease total_time, increase spatial resolution, increase nt, "
                "or reduce diffusivity_max.\n"
                f"Computed CFL={cfl:.3f} must be < {cfl_limit:.3f} for stability."
            )


def sample_initial_condition_1d(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate a smooth random 1D initial profile satisfying boundary conditions."""
    num_modes = rng.integers(2, 6)
    coeffs = rng.uniform(-1.0, 1.0, size=num_modes)
    phases = rng.uniform(0.0, np.pi, size=num_modes)
    u0 = np.zeros_like(x)
    for k in range(1, num_modes + 1):
        u0 += coeffs[k - 1] * np.sin(k * np.pi * x + phases[k - 1])
    u0 -= u0.mean()
    u0 /= np.max(np.abs(u0)) + 1e-8
    u0[0] = 0.0
    u0[-1] = 0.0
    return u0.astype(np.float32)


def sample_initial_condition_2d(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Generate a smooth random 2D initial profile satisfying boundary conditions."""
    X, Y = np.meshgrid(x, y, indexing='xy')
    num_modes = rng.integers(2, 5)
    u0 = np.zeros_like(X)
    
    for _ in range(num_modes):
        kx = rng.integers(1, 5)
        ky = rng.integers(1, 5)
        coeff = rng.uniform(-1.0, 1.0)
        phase_x = rng.uniform(0.0, np.pi)
        phase_y = rng.uniform(0.0, np.pi)
        u0 += coeff * np.sin(kx * np.pi * X + phase_x) * np.sin(ky * np.pi * Y + phase_y)
    
    u0 -= u0.mean()
    u0 /= np.max(np.abs(u0)) + 1e-8
    
    # Enforce Dirichlet boundary conditions: u = 0 on all boundaries
    u0[0, :] = 0.0  # Bottom boundary
    u0[-1, :] = 0.0  # Top boundary
    u0[:, 0] = 0.0  # Left boundary
    u0[:, -1] = 0.0  # Right boundary
    
    return u0.astype(np.float32)


def sample_initial_condition_3d(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Generate a smooth random 3D initial profile satisfying boundary conditions."""
    X, Y, Z = np.meshgrid(x, y, z, indexing='xy')
    num_modes = rng.integers(2, 4)
    u0 = np.zeros_like(X)
    
    for _ in range(num_modes):
        kx = rng.integers(1, 4)
        ky = rng.integers(1, 4)
        kz = rng.integers(1, 4)
        coeff = rng.uniform(-1.0, 1.0)
        phase_x = rng.uniform(0.0, np.pi)
        phase_y = rng.uniform(0.0, np.pi)
        phase_z = rng.uniform(0.0, np.pi)
        u0 += (
            coeff
            * np.sin(kx * np.pi * X + phase_x)
            * np.sin(ky * np.pi * Y + phase_y)
            * np.sin(kz * np.pi * Z + phase_z)
        )
    
    u0 -= u0.mean()
    u0 /= np.max(np.abs(u0)) + 1e-8
    
    # Enforce Dirichlet boundary conditions: u = 0 on all faces
    u0[0, :, :] = 0.0  # Front face (z=0)
    u0[-1, :, :] = 0.0  # Back face (z=L)
    u0[:, 0, :] = 0.0  # Bottom face (y=0)
    u0[:, -1, :] = 0.0  # Top face (y=L)
    u0[:, :, 0] = 0.0  # Left face (x=0)
    u0[:, :, -1] = 0.0  # Right face (x=L)
    
    return u0.astype(np.float32)


def sample_initial_condition(
    cfg: HeatEqConfig,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    z: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate initial condition based on dimension."""
    if rng is None:
        rng = np.random.default_rng()
    
    if cfg.dimension == 1:
        if x is None:
            x = np.linspace(0.0, cfg.length, cfg.nx, dtype=np.float32)
        return sample_initial_condition_1d(x, rng)
    elif cfg.dimension == 2:
        if x is None:
            x = np.linspace(0.0, cfg.length, cfg.nx, dtype=np.float32)
        if y is None:
            y = np.linspace(0.0, cfg.length, cfg.ny, dtype=np.float32)
        return sample_initial_condition_2d(x, y, rng)
    elif cfg.dimension == 3:
        if x is None:
            x = np.linspace(0.0, cfg.length, cfg.nx, dtype=np.float32)
        if y is None:
            y = np.linspace(0.0, cfg.length, cfg.ny, dtype=np.float32)
        if z is None:
            z = np.linspace(0.0, cfg.length, cfg.nz, dtype=np.float32)
        return sample_initial_condition_3d(x, y, z, rng)
    else:
        raise ValueError(f"Unsupported dimension: {cfg.dimension}")


def solve_heat_1d(
    u0: np.ndarray,
    diffusivity: float,
    cfg: HeatEqConfig,
) -> np.ndarray:
    """Solve the 1-D heat equation with an explicit finite-difference scheme."""
    r = diffusivity * cfg.dt / (cfg.dx ** 2)
    nx = cfg.nx
    nt = cfg.nt

    solution = np.zeros((nt, nx), dtype=np.float32)
    solution[0] = u0

    for n in range(1, nt):
        u_prev = solution[n - 1]
        interior = u_prev[1:-1] + r * (u_prev[0:-2] - 2 * u_prev[1:-1] + u_prev[2:])
        solution[n, 1:-1] = interior
        solution[n, 0] = 0.0
        solution[n, -1] = 0.0

    return solution


def solve_heat_2d(
    u0: np.ndarray,
    diffusivity: float,
    cfg: HeatEqConfig,
) -> np.ndarray:
    """Solve the 2-D heat equation with an explicit finite-difference scheme.
    
    The 2D heat equation: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
    Uses vectorized operations for efficiency.
    """
    rx = diffusivity * cfg.dt / (cfg.dx ** 2)
    ry = diffusivity * cfg.dt / (cfg.dy ** 2)
    ny, nx = u0.shape
    nt = cfg.nt

    solution = np.zeros((nt, ny, nx), dtype=np.float32)
    solution[0] = u0

    for n in range(1, nt):
        u_prev = solution[n - 1]
        u_new = u_prev.copy()
        
        # Vectorized Laplacian computation
        # Second derivative in x (axis 1): u[:, j+1] - 2u[:, j] + u[:, j-1]
        d2u_dx2 = np.zeros_like(u_prev)
        d2u_dx2[:, 1:-1] = u_prev[:, 2:] - 2 * u_prev[:, 1:-1] + u_prev[:, :-2]
        
        # Second derivative in y (axis 0): u[i+1, :] - 2u[i, :] + u[i-1, :]
        d2u_dy2 = np.zeros_like(u_prev)
        d2u_dy2[1:-1, :] = u_prev[2:, :] - 2 * u_prev[1:-1, :] + u_prev[:-2, :]
        
        # Update interior points: u_new = u_old + dt * α * (d²u/dx² + d²u/dy²)
        u_new[1:-1, 1:-1] = (
            u_prev[1:-1, 1:-1]
            + rx * d2u_dx2[1:-1, 1:-1]
            + ry * d2u_dy2[1:-1, 1:-1]
        )
        
        # Enforce boundary conditions (already 0 from initialization)
        u_new[0, :] = 0.0
        u_new[-1, :] = 0.0
        u_new[:, 0] = 0.0
        u_new[:, -1] = 0.0
        
        solution[n] = u_new

    return solution


def solve_heat_3d(
    u0: np.ndarray,
    diffusivity: float,
    cfg: HeatEqConfig,
) -> np.ndarray:
    """Solve the 3-D heat equation with an explicit finite-difference scheme.
    
    The 3D heat equation: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)
    Uses vectorized operations for efficiency.
    """
    rx = diffusivity * cfg.dt / (cfg.dx ** 2)
    ry = diffusivity * cfg.dt / (cfg.dy ** 2)
    rz = diffusivity * cfg.dt / (cfg.dz ** 2)
    nz, ny, nx = u0.shape
    nt = cfg.nt

    solution = np.zeros((nt, nz, ny, nx), dtype=np.float32)
    solution[0] = u0

    for n in range(1, nt):
        u_prev = solution[n - 1]
        u_new = u_prev.copy()
        
        # Vectorized Laplacian computation
        # Second derivative in x (axis 2): u[:, :, j+1] - 2u[:, :, j] + u[:, :, j-1]
        d2u_dx2 = np.zeros_like(u_prev)
        d2u_dx2[:, :, 1:-1] = u_prev[:, :, 2:] - 2 * u_prev[:, :, 1:-1] + u_prev[:, :, :-2]
        
        # Second derivative in y (axis 1): u[:, i+1, :] - 2u[:, i, :] + u[:, i-1, :]
        d2u_dy2 = np.zeros_like(u_prev)
        d2u_dy2[:, 1:-1, :] = u_prev[:, 2:, :] - 2 * u_prev[:, 1:-1, :] + u_prev[:, :-2, :]
        
        # Second derivative in z (axis 0): u[k+1, :, :] - 2u[k, :, :] + u[k-1, :, :]
        d2u_dz2 = np.zeros_like(u_prev)
        d2u_dz2[1:-1, :, :] = u_prev[2:, :, :] - 2 * u_prev[1:-1, :, :] + u_prev[:-2, :, :]
        
        # Update interior points: u_new = u_old + dt * α * (d²u/dx² + d²u/dy² + d²u/dz²)
        u_new[1:-1, 1:-1, 1:-1] = (
            u_prev[1:-1, 1:-1, 1:-1]
            + rx * d2u_dx2[1:-1, 1:-1, 1:-1]
            + ry * d2u_dy2[1:-1, 1:-1, 1:-1]
            + rz * d2u_dz2[1:-1, 1:-1, 1:-1]
        )
        
        # Enforce boundary conditions on all faces
        u_new[0, :, :] = 0.0   # Front face (z=0)
        u_new[-1, :, :] = 0.0  # Back face (z=L)
        u_new[:, 0, :] = 0.0   # Bottom face (y=0)
        u_new[:, -1, :] = 0.0  # Top face (y=L)
        u_new[:, :, 0] = 0.0   # Left face (x=0)
        u_new[:, :, -1] = 0.0  # Right face (x=L)
        
        solution[n] = u_new

    return solution


def solve_heat(
    u0: np.ndarray,
    diffusivity: float,
    cfg: HeatEqConfig,
) -> np.ndarray:
    """Solve the heat equation with dimension-dependent solver."""
    if cfg.dimension == 1:
        return solve_heat_1d(u0, diffusivity, cfg)
    elif cfg.dimension == 2:
        return solve_heat_2d(u0, diffusivity, cfg)
    elif cfg.dimension == 3:
        return solve_heat_3d(u0, diffusivity, cfg)
    else:
        raise ValueError(f"Unsupported dimension: {cfg.dimension}")


def generate_dataset(cfg: HeatEqConfig) -> tuple[Path, Path, Path]:
    """Generate train/test/validate datasets with 80%/10%/10% split."""
    cfg.validate()
    rng = np.random.default_rng(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Create spatial grids based on dimension
    x = np.linspace(0.0, cfg.length, cfg.nx, dtype=np.float32)
    if cfg.dimension >= 2:
        y = np.linspace(0.0, cfg.length, cfg.ny, dtype=np.float32)
    else:
        y = None
    if cfg.dimension == 3:
        z = np.linspace(0.0, cfg.length, cfg.nz, dtype=np.float32)
    else:
        z = None
    
    t = np.linspace(0.0, cfg.total_time, cfg.nt, dtype=np.float32)

    # Determine spatial shape
    spatial_shape = cfg.spatial_shape
    if cfg.dimension == 1:
        solutions = np.zeros((cfg.dataset_size, cfg.nt, cfg.nx), dtype=np.float32)
        initials = np.zeros((cfg.dataset_size, cfg.nx), dtype=np.float32)
    elif cfg.dimension == 2:
        solutions = np.zeros((cfg.dataset_size, cfg.nt, cfg.ny, cfg.nx), dtype=np.float32)
        initials = np.zeros((cfg.dataset_size, cfg.ny, cfg.nx), dtype=np.float32)
    else:  # 3D
        solutions = np.zeros((cfg.dataset_size, cfg.nt, cfg.nz, cfg.ny, cfg.nx), dtype=np.float32)
        initials = np.zeros((cfg.dataset_size, cfg.nz, cfg.ny, cfg.nx), dtype=np.float32)
    
    diffusivities = np.zeros(cfg.dataset_size, dtype=np.float32)

    # Progress printing
    print("Generating trajectories...")
    print_progress_every = max(1, cfg.dataset_size // 20)  # Print ~20 times total
    
    import time
    start_time = time.time()
    
    for idx in range(cfg.dataset_size):
        diffusivity = rng.uniform(cfg.diffusivity_min, cfg.diffusivity_max)
        u0 = sample_initial_condition(cfg, x=x, y=y, z=z, rng=rng)
        sol = solve_heat(u0, diffusivity, cfg)

        if cfg.noise_std > 0:
            sol += rng.normal(scale=cfg.noise_std, size=sol.shape).astype(np.float32)

        solutions[idx] = sol
        initials[idx] = u0
        diffusivities[idx] = diffusivity
        
        # Print progress
        if (idx + 1) % print_progress_every == 0 or (idx + 1) == cfg.dataset_size:
            progress = (idx + 1) / cfg.dataset_size * 100
            elapsed = time.time() - start_time
            if idx > 0:
                rate = (idx + 1) / elapsed
                remaining = (cfg.dataset_size - idx - 1) / rate
                print(f"  Progress: {idx + 1}/{cfg.dataset_size} ({progress:.1f}%) - "
                      f"Elapsed: {elapsed:.1f}s - "
                      f"Remaining: ~{remaining:.1f}s")
            else:
                print(f"  Progress: {idx + 1}/{cfg.dataset_size} ({progress:.1f}%)")

    # Shuffle indices before splitting to ensure random distribution
    print("\nSplitting data into train/test/validate sets...")
    indices = np.arange(cfg.dataset_size)
    rng.shuffle(indices)
    
    # Calculate split sizes: 80% train, 10% test, 10% validate
    train_size = int(0.8 * cfg.dataset_size)
    test_size = int(0.1 * cfg.dataset_size)
    
    train_idx = indices[:train_size]
    test_idx = indices[train_size:train_size + test_size]
    validate_idx = indices[train_size + test_size:]
    
    print(f"  Train: {train_size} samples")
    print(f"  Test: {test_size} samples")
    print(f"  Validate: {len(validate_idx)} samples")
    
    # Split the data
    train_solutions = solutions[train_idx]
    train_initials = initials[train_idx]
    train_diffusivities = diffusivities[train_idx]
    
    test_solutions = solutions[test_idx]
    test_initials = initials[test_idx]
    test_diffusivities = diffusivities[test_idx]
    
    validate_solutions = solutions[validate_idx]
    validate_initials = initials[validate_idx]
    validate_diffusivities = diffusivities[validate_idx]
    
    # Store spatial coordinates based on dimension
    if cfg.dimension == 1:
        coords = {"x": x}
    elif cfg.dimension == 2:
        coords = {"x": x, "y": y}
    else:  # 3D
        coords = {"x": x, "y": y, "z": z}
    
    metadata = np.array(
        [
            ("dimension", cfg.dimension),
            ("length", cfg.length),
            ("nx", cfg.nx),
            ("ny", cfg.ny if cfg.dimension >= 2 else 0),
            ("nz", cfg.nz if cfg.dimension == 3 else 0),
            ("nt", cfg.nt),
            ("total_time", cfg.total_time),
            ("diffusivity_min", cfg.diffusivity_min),
            ("diffusivity_max", cfg.diffusivity_max),
            ("noise_std", cfg.noise_std),
        ],
        dtype=object,
    )
    
    # Save train dataset
    print("\nSaving datasets...")
    save_start = time.time()
    
    # For large datasets, use uncompressed saving to reduce memory pressure
    # Compressed saving requires extra memory for compression and can be very slow when memory is low
    use_compressed = True
    # Estimate dataset size in MB
    if cfg.dimension == 1:
        estimated_size_mb = (train_solutions.nbytes + train_initials.nbytes) / (1024 * 1024)
    elif cfg.dimension == 2:
        estimated_size_mb = (train_solutions.nbytes + train_initials.nbytes) / (1024 * 1024)
    else:  # 3D
        estimated_size_mb = (train_solutions.nbytes + train_initials.nbytes) / (1024 * 1024)
    
    # Use uncompressed if dataset is very large (>500MB) to save memory and time
    if estimated_size_mb > 500:
        use_compressed = False
        print(f"  Large dataset detected (~{estimated_size_mb:.0f} MB). Using uncompressed format for faster saving.")
    
    save_func = np.savez_compressed if use_compressed else np.savez
    
    train_path = cfg.output_dir / "train.npz"
    save_dict = {
        "t": t,
        "initial": train_initials,
        "solution": train_solutions,
        "diffusivity": train_diffusivities,
        "metadata": metadata,
        **coords,  # Add spatial coordinates (x, y, z based on dimension)
    }
    
    print(f"  Saving train dataset to {train_path}...", end="", flush=True)
    save_func(train_path, **save_dict)
    save_time = time.time() - save_start
    file_size_mb = train_path.stat().st_size / (1024 * 1024)
    print(f" Done ({save_time:.1f}s, {file_size_mb:.1f} MB)")
    
    # Save test dataset
    save_start = time.time()
    test_path = cfg.output_dir / "test.npz"
    save_dict = {
        "t": t,
        "initial": test_initials,
        "solution": test_solutions,
        "diffusivity": test_diffusivities,
        "metadata": metadata,
        **coords,
    }
    print(f"  Saving test dataset to {test_path}...", end="", flush=True)
    save_func(test_path, **save_dict)
    save_time = time.time() - save_start
    file_size_mb = test_path.stat().st_size / (1024 * 1024)
    print(f" Done ({save_time:.1f}s, {file_size_mb:.1f} MB)")
    
    # Save validate dataset
    save_start = time.time()
    validate_path = cfg.output_dir / "validate.npz"
    save_dict = {
        "t": t,
        "initial": validate_initials,
        "solution": validate_solutions,
        "diffusivity": validate_diffusivities,
        "metadata": metadata,
        **coords,
    }
    print(f"  Saving validate dataset to {validate_path}...", end="", flush=True)
    save_func(validate_path, **save_dict)
    save_time = time.time() - save_start
    file_size_mb = validate_path.stat().st_size / (1024 * 1024)
    print(f" Done ({save_time:.1f}s, {file_size_mb:.1f} MB)")
    
    total_time = time.time() - start_time
    print(f"\nData generation complete! Total time: {total_time:.1f}s")
    
    return train_path, test_path, validate_path


def parse_args() -> HeatEqConfig:
    parser = argparse.ArgumentParser(
        description="Generate synthetic solutions of the 1D, 2D, or 3D heat equation."
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Spatial dimension (1, 2, or 3).",
    )
    parser.add_argument("--length", type=float, default=1.0, help="Domain length in each dimension.")
    parser.add_argument("--nx", type=int, default=128, help="Number of spatial points in x-direction.")
    parser.add_argument(
        "--ny",
        type=int,
        default=64,
        help="Number of spatial points in y-direction (2D/3D only).",
    )
    parser.add_argument(
        "--nz",
        type=int,
        default=32,
        help="Number of spatial points in z-direction (3D only).",
    )
    parser.add_argument("--nt", type=int, default=64, help="Number of time steps.")
    parser.add_argument("--total-time", type=float, default=0.008, help="Simulation time.")
    parser.add_argument(
        "--diffusivity-min",
        type=float,
        default=0.05,
        help="Minimum thermal diffusivity to sample.",
    )
    parser.add_argument(
        "--diffusivity-max",
        type=float,
        default=0.2,
        help="Maximum thermal diffusivity to sample.",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=256,
        help="Number of trajectories to generate.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Optional Gaussian noise level added to trajectories.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("heat_eq_data/data"),
        help="Directory to store the generated dataset.",
    )

    args = parser.parse_args()
    return HeatEqConfig(
        dimension=args.dimension,
        length=args.length,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        nt=args.nt,
        total_time=args.total_time,
        diffusivity_min=args.diffusivity_min,
        diffusivity_max=args.diffusivity_max,
        dataset_size=args.dataset_size,
        noise_std=args.noise_std,
        seed=args.seed,
        output_dir=args.output_dir,
    )


def main() -> None:
    cfg = parse_args()
    print(f"Generating {cfg.dimension}D heat equation dataset")
    print(f"  Spatial resolution: {cfg.spatial_shape}")
    print(f"  Time steps: {cfg.nt}")
    print(f"  Dataset size: {cfg.dataset_size}")
    print()
    
    train_path, test_path, validate_path = generate_dataset(cfg)
    print(f"Wrote train dataset to {train_path}")
    print(f"Wrote test dataset to {test_path}")
    print(f"Wrote validate dataset to {validate_path}")


if __name__ == "__main__":
    main()


