"""
Heat equation data visualization utilities.

This module provides functions to visualize the heat equation dataset:
1. Plot a single trajectory over time (snapshots at different time steps)
2. Plot trajectory as a heat map (space-time visualization)
3. Plot multiple trajectories (space snapshots)
4. Animate the evolution over time
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def load_data(
    data_path: str | Path,
    split: Literal["train", "test", "validate"] = "train",
) -> dict[str, np.ndarray]:
    """Load heat equation data from NPZ file.
    
    Args:
        data_path: Path to the data directory containing train.npz, test.npz, validate.npz
        split: Which dataset split to load
        
    Returns:
        Dictionary containing:
        - 'x': spatial coordinates (nx,)
        - 't': time coordinates (nt,)
        - 'initial': initial conditions (nsamples, nx)
        - 'solution': solutions over time (nsamples, nt, nx)
        - 'diffusivity': thermal diffusivity values (nsamples,)
        - 'metadata': metadata array
    """
    data_path = Path(data_path)
    file_path = data_path / f"{split}.npz"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Need allow_pickle=True because metadata is stored as object array
    data = np.load(file_path, allow_pickle=True)
    return {key: data[key] for key in data.keys()}


def plot_single_trajectory(
    x: np.ndarray,
    t: np.ndarray,
    solution: np.ndarray,
    sample_idx: int = 0,
    num_time_steps: int = 5,
    ax: plt.Axes | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a single trajectory showing u(x) at different time steps.
    
    Args:
        x: Spatial coordinates (nx,)
        t: Time coordinates (nt,)
        solution: Solution array (nt, nx) or (nsamples, nt, nx)
        sample_idx: Index of sample to plot (if solution is 3D)
        num_time_steps: Number of time steps to plot
        ax: Optional axes to plot on
        figsize: Figure size
        
    Returns:
        Figure and axes objects
    """
    # Handle 3D solution array
    if solution.ndim == 3:
        solution = solution[sample_idx]
    
    nt, nx = solution.shape
    
    # Select time steps to plot (evenly spaced)
    if num_time_steps > nt:
        num_time_steps = nt
    time_indices = np.linspace(0, nt - 1, num_time_steps, dtype=int)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Plot solution at different time steps
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, num_time_steps))
    
    for i, t_idx in enumerate(time_indices):
        ax.plot(
            x,
            solution[t_idx],
            label=f"t = {t[t_idx]:.4f}",
            color=colors[i],
            linewidth=2,
        )
    
    ax.set_xlabel("Position x", fontsize=12, fontweight="bold")
    ax.set_ylabel("Temperature u(x,t)", fontsize=12, fontweight="bold")
    ax.set_title(f"Heat Equation Solution Over Time (Sample {sample_idx})", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_heatmap(
    x: np.ndarray,
    t: np.ndarray,
    solution: np.ndarray,
    sample_idx: int = 0,
    ax: plt.Axes | None = None,
    figsize: tuple[int, int] = (10, 6),
    cmap: str = "hot",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot trajectory as a 2D heat map (space-time visualization).
    
    Args:
        x: Spatial coordinates (nx,)
        t: Time coordinates (nt,)
        solution: Solution array (nt, nx) or (nsamples, nt, nx)
        sample_idx: Index of sample to plot (if solution is 3D)
        ax: Optional axes to plot on
        figsize: Figure size
        cmap: Colormap name
        
    Returns:
        Figure and axes objects
    """
    # Handle 3D solution array
    if solution.ndim == 3:
        solution = solution[sample_idx]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Create heatmap with space on x-axis, time on y-axis
    im = ax.imshow(
        solution,
        aspect="auto",
        origin="lower",
        extent=[x.min(), x.max(), t.min(), t.max()],
        cmap=cmap,
        interpolation="bilinear",
    )
    
    ax.set_xlabel("Position x", fontsize=12, fontweight="bold")
    ax.set_ylabel("Time t", fontsize=12, fontweight="bold")
    ax.set_title(f"Heat Equation Solution Heatmap (Sample {sample_idx})", fontsize=14, fontweight="bold")
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Temperature u(x,t)", fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    return fig, ax


def plot_multiple_trajectories(
    x: np.ndarray,
    solution: np.ndarray,
    time_idx: int | None = None,
    sample_indices: list[int] | None = None,
    num_samples: int = 5,
    ax: plt.Axes | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot multiple trajectories showing u(x) for different samples.
    
    Args:
        x: Spatial coordinates (nx,)
        solution: Solution array (nsamples, nt, nx) or (nt, nx)
        time_idx: Time index to plot. If None, plots at final time step
        sample_indices: Indices of samples to plot. If None, plots first num_samples
        num_samples: Number of samples to plot if sample_indices is None
        ax: Optional axes to plot on
        figsize: Figure size
        
    Returns:
        Figure and axes objects
    """
    # Handle 2D solution array (single sample, already indexed)
    if solution.ndim == 2:
        solution = solution[np.newaxis, ...]
    
    nsamples, nt, nx = solution.shape
    
    # Default to final time step
    if time_idx is None:
        time_idx = nt - 1
    
    # Select samples to plot
    if sample_indices is None:
        sample_indices = list(range(min(num_samples, nsamples)))
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Plot multiple trajectories
    cmap = plt.cm.tab10
    colors = cmap(np.linspace(0, 1, len(sample_indices)))
    
    for i, sample_idx in enumerate(sample_indices):
        ax.plot(
            x,
            solution[sample_idx, time_idx],
            label=f"Sample {sample_idx}",
            color=colors[i],
            linewidth=2,
            alpha=0.8,
        )
    
    ax.set_xlabel("Position x", fontsize=12, fontweight="bold")
    ax.set_ylabel("Temperature u(x,t)", fontsize=12, fontweight="bold")
    ax.set_title(f"Multiple Trajectories at Time Step {time_idx}", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def animate_trajectory(
    x: np.ndarray,
    t: np.ndarray,
    solution: np.ndarray,
    sample_idx: int = 0,
    interval: int = 50,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> animation.FuncAnimation:
    """Animate the evolution of a single trajectory over time.
    
    Args:
        x: Spatial coordinates (nx,)
        t: Time coordinates (nt,)
        solution: Solution array (nt, nx) or (nsamples, nt, nx)
        sample_idx: Index of sample to animate (if solution is 3D)
        interval: Animation frame interval in milliseconds
        save_path: Optional path to save animation as GIF or MP4
        figsize: Figure size
        
    Returns:
        Animation object
    """
    # Handle 3D solution array
    if solution.ndim == 3:
        solution = solution[sample_idx]
    
    nt, nx = solution.shape
    
    fig, ax = plt.subplots(figsize=figsize)
    line, = ax.plot([], [], linewidth=2, color="blue")
    
    # Set axis limits
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(solution.min(), solution.max())
    ax.set_xlabel("Position x", fontsize=12, fontweight="bold")
    ax.set_ylabel("Temperature u(x,t)", fontsize=12, fontweight="bold")
    ax.set_title(f"Heat Equation Evolution Over Time (Sample {sample_idx})", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=12,
                        verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    def animate(frame):
        line.set_data(x, solution[frame])
        time_text.set_text(f"Time: t = {t[frame]:.4f}")
        return line, time_text
    
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=nt,
        interval=interval,
        blit=True,
        repeat=True,
    )
    
    if save_path:
        save_path = Path(save_path)
        if save_path.suffix.lower() == ".gif":
            anim.save(save_path, writer="pillow", fps=1000 // interval)
        elif save_path.suffix.lower() in [".mp4", ".avi"]:
            anim.save(save_path, writer="ffmpeg", fps=1000 // interval)
        else:
            raise ValueError(f"Unsupported file format: {save_path.suffix}")
        print(f"Animation saved to {save_path}")
    
    plt.tight_layout()
    return anim


def main():
    """Example usage of the plotting functions."""
    # Example: Load and plot data
    data_dir = Path("heat_eq_data/data")
    
    # Load train data
    data = load_data(data_dir, split="train")
    
    x = data["x"]
    t = data["t"]
    solutions = data["solution"]
    
    print(f"Loaded data: {solutions.shape[0]} samples, {solutions.shape[1]} time steps, {solutions.shape[2]} spatial points")
    
    # Plot 1: Single trajectory over time
    print("Plotting single trajectory...")
    fig1, ax1 = plot_single_trajectory(x, t, solutions, sample_idx=0, num_time_steps=6)
    plt.show()
    
    # Plot 2: Heat map
    print("Plotting heat map...")
    fig2, ax2 = plot_heatmap(x, t, solutions, sample_idx=0)
    plt.show()
    
    # Plot 3: Multiple trajectories
    print("Plotting multiple trajectories...")
    fig3, ax3 = plot_multiple_trajectories(x, solutions, time_idx=-1, num_samples=5)
    plt.show()
    
    # Plot 4: Animation (uncomment to save)
    print("Creating animation...")
    anim = animate_trajectory(x, t, solutions, sample_idx=0, interval=100)
    # anim.save("heat_eq_animation.gif", writer="pillow", fps=10)  # Uncomment to save
    plt.show()


if __name__ == "__main__":
    main()

