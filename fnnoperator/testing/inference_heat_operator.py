"""
Inference script for FNNOperator on heat equation data.

This script loads a trained model and generates predictions on test data,
then creates heatmap comparisons between ground truth and predictions.

Usage:
    python fnnoperator/inference_heat_operator.py --checkpoint fnnoperator/outputs/heat_eq_2d_20241215_143022/checkpoints/final_model.pt --data-dir fnnoperator/data/32x32 --output-dir fnnoperator/outputs/heat_eq_2d_20241215_143022/inference
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from fnnoperator.FNNOperator import FNNOperator


def load_model(checkpoint_path: Path, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model parameters from checkpoint
    grid_shape = tuple(checkpoint['grid_shape'])
    in_channels = checkpoint['in_channels']
    out_channels = checkpoint['out_channels']
    width = checkpoint['width']
    depth = checkpoint['depth']
    n_time_steps = checkpoint.get('n_time_steps', 1)  # Default to 1 for backward compatibility
    
    # Create model
    model = FNNOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        grid_shape=grid_shape,
        width=width,
        depth=depth,
        n_time_steps=n_time_steps,
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"  Grid shape: {grid_shape}")
    print(f"  In channels: {in_channels}, Out channels: {out_channels}")
    print(f"  Width: {width}, Depth: {depth}")
    print(f"  Time steps: {n_time_steps}")
    if 'test_loss' in checkpoint:
        print(f"  Test loss: {checkpoint['test_loss']:.6f}")
    
    return model, checkpoint


def load_test_data(data_dir: Path, return_full_trajectory=False):
    """Load test data.
    
    Args:
        data_dir: Directory containing test.npz
        return_full_trajectory: If True, also return full solution trajectory
    
    Returns:
        If return_full_trajectory=False: (X, Y, grid_shape)
        If return_full_trajectory=True: (X, Y, grid_shape, solution_trajectory, time_steps)
    """
    import numpy as np
    
    file_path = data_dir / "test.npz"
    if not file_path.exists():
        raise FileNotFoundError(f"Test data not found: {file_path}")
    
    data = np.load(file_path, allow_pickle=True)
    initial_raw = data["initial"]  # (n_samples, *spatial_dims)
    solution_raw = data["solution"]  # (n_samples, nt, *spatial_dims)
    time_steps = data.get("t", None)  # (nt,) if available
    
    # Extract final state (last time step)
    final_states = solution_raw[:, -1, ...]  # (n_samples, *spatial_dims)
    
    # Convert to tensors and add channel dimension
    initial_conditions = torch.tensor(initial_raw, dtype=torch.float32)
    final_states = torch.tensor(final_states, dtype=torch.float32)
    
    # Add channel dimension: (n_samples, *spatial_dims) -> (n_samples, 1, *spatial_dims)
    X = initial_conditions.unsqueeze(1)
    Y = final_states.unsqueeze(1)
    
    if return_full_trajectory:
        return X, Y, initial_raw.shape[1:], solution_raw, time_steps
    else:
        return X, Y, initial_raw.shape[1:]  # Return grid shape


def predict(model, X, device, batch_size=32):
    """Generate predictions in batches."""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size].to(device)
            batch_pred = model(batch_X)
            predictions.append(batch_pred.cpu().numpy())
    
    return np.concatenate(predictions, axis=0)


def create_heatmap_comparison(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    initial: np.ndarray,
    sample_idx: int,
    output_path: Path,
    grid_shape: tuple,
):
    """Create side-by-side heatmap comparison."""
    # Remove channel dimension if present
    if len(ground_truth.shape) == len(grid_shape) + 2:  # (batch, channels, *grid)
        gt = ground_truth[sample_idx, 0, ...]
        pred = prediction[sample_idx, 0, ...]
        init = initial[sample_idx, ...] if len(initial.shape) == len(grid_shape) + 1 else initial[sample_idx, 0, ...]
    else:  # (batch, *grid)
        gt = ground_truth[sample_idx, ...]
        pred = prediction[sample_idx, ...]
        init = initial[sample_idx, ...]
    
    # Calculate error
    error = np.abs(gt - pred)
    
    # Determine dimension
    if len(grid_shape) == 1:
        # 1D: line plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        x = np.linspace(0, 1, grid_shape[0])
        
        axes[0, 0].plot(x, init, 'b-', label='Initial', linewidth=2)
        axes[0, 0].set_title('Initial Condition')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('u(x)')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        axes[0, 1].plot(x, gt, 'g-', label='Ground Truth', linewidth=2)
        axes[0, 1].plot(x, pred, 'r--', label='Prediction', linewidth=2)
        axes[0, 1].set_title('Final State Comparison')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('u(x)')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        
        axes[1, 0].plot(x, error, 'r-', linewidth=2)
        axes[1, 0].set_title('Absolute Error')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('|Error|')
        axes[1, 0].grid(True)
        
        axes[1, 1].axis('off')
        axes[1, 1].text(0.5, 0.5, f'Max Error: {error.max():.6f}\nMean Error: {error.mean():.6f}\nRMSE: {np.sqrt(np.mean(error**2)):.6f}',
                       ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
    elif len(grid_shape) == 2:
        # 2D: heatmaps
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Get common vmin/vmax for consistent color scales
        vmin = min(gt.min(), pred.min(), init.min())
        vmax = max(gt.max(), pred.max(), init.max())
        error_max = error.max()
        
        # Initial condition
        im0 = axes[0, 0].imshow(init, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        axes[0, 0].set_title('Initial Condition')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        plt.colorbar(im0, ax=axes[0, 0])
        
        # Ground truth
        im1 = axes[0, 1].imshow(gt, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        axes[0, 1].set_title('Ground Truth (Final State)')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Prediction
        im2 = axes[0, 2].imshow(pred, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        axes[0, 2].set_title('Prediction (Final State)')
        axes[0, 2].set_xlabel('x')
        axes[0, 2].set_ylabel('y')
        plt.colorbar(im2, ax=axes[0, 2])
        
        # Error
        im3 = axes[1, 0].imshow(error, cmap='hot', origin='lower', vmin=0, vmax=error_max)
        axes[1, 0].set_title('Absolute Error')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('y')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Difference (prediction - ground truth)
        diff = pred - gt
        diff_max = np.abs(diff).max()
        im4 = axes[1, 1].imshow(diff, cmap='RdBu_r', origin='lower', vmin=-diff_max, vmax=diff_max)
        axes[1, 1].set_title('Difference (Pred - GT)')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('y')
        plt.colorbar(im4, ax=axes[1, 1])
        
        # Statistics
        axes[1, 2].axis('off')
        stats_text = f"""Statistics:
Max Error: {error.max():.6f}
Mean Error: {error.mean():.6f}
RMSE: {np.sqrt(np.mean(error**2)):.6f}
Relative Error: {np.mean(error) / (np.abs(gt).mean() + 1e-8) * 100:.2f}%
"""
        axes[1, 2].text(0.5, 0.5, stats_text, ha='center', va='center',
                       fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
    else:
        # 3D: show middle slice
        mid_z = grid_shape[0] // 2
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        vmin = min(gt.min(), pred.min(), init.min())
        vmax = max(gt.max(), pred.max(), init.max())
        error_max = error.max()
        
        im0 = axes[0, 0].imshow(init[mid_z, ...], cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        axes[0, 0].set_title(f'Initial Condition (z={mid_z})')
        plt.colorbar(im0, ax=axes[0, 0])
        
        im1 = axes[0, 1].imshow(gt[mid_z, ...], cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        axes[0, 1].set_title(f'Ground Truth (z={mid_z})')
        plt.colorbar(im1, ax=axes[0, 1])
        
        im2 = axes[0, 2].imshow(pred[mid_z, ...], cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        axes[0, 2].set_title(f'Prediction (z={mid_z})')
        plt.colorbar(im2, ax=axes[0, 2])
        
        im3 = axes[1, 0].imshow(error[mid_z, ...], cmap='hot', origin='lower', vmin=0, vmax=error_max)
        axes[1, 0].set_title(f'Absolute Error (z={mid_z})')
        plt.colorbar(im3, ax=axes[1, 0])
        
        diff = pred - gt
        diff_max = np.abs(diff).max()
        im4 = axes[1, 1].imshow(diff[mid_z, ...], cmap='RdBu_r', origin='lower', vmin=-diff_max, vmax=diff_max)
        axes[1, 1].set_title(f'Difference (z={mid_z})')
        plt.colorbar(im4, ax=axes[1, 1])
        
        axes[1, 2].axis('off')
        stats_text = f"""Statistics:
Max Error: {error.max():.6f}
Mean Error: {error.mean():.6f}
RMSE: {np.sqrt(np.mean(error**2)):.6f}
"""
        axes[1, 2].text(0.5, 0.5, stats_text, ha='center', va='center',
                       fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def create_temporal_evolution(
    solution_trajectory: np.ndarray,
    prediction: np.ndarray,
    initial: np.ndarray,
    sample_idx: int,
    time_steps: np.ndarray,
    output_path: Path,
    grid_shape: tuple,
    n_time_steps_to_show: int = 6,
):
    """Create temporal evolution heatmap showing ground truth and predictions at different time steps.
    
    Args:
        solution_trajectory: (n_samples, nt, *spatial_dims) - full ground truth trajectory
        prediction: If single time step: (n_samples, channels, *spatial_dims)
                    If multiple time steps: (n_samples, nt, channels, *spatial_dims)
        initial: (n_samples, *spatial_dims) - initial conditions
        sample_idx: Index of sample to visualize
        time_steps: (nt,) array of time values
        output_path: Where to save the figure
        grid_shape: Spatial grid shape
        n_time_steps_to_show: Number of time steps to display (default: 6)
    """
    # Handle prediction shape - check if it has time dimension
    if len(prediction.shape) == len(grid_shape) + 3:  # (batch, nt, channels, *grid)
        # Multiple time steps predicted
        pred_trajectory = prediction[sample_idx, ...]  # (nt, channels, *grid)
        # Remove channel dimension
        if pred_trajectory.shape[1] == 1:
            pred_trajectory = pred_trajectory[:, 0, ...]  # (nt, *grid)
        else:
            pred_trajectory = pred_trajectory  # Keep channels if > 1
        has_multiple_time_steps = True
    elif len(prediction.shape) == len(grid_shape) + 2:  # (batch, channels, *grid) - single time step
        pred = prediction[sample_idx, 0, ...] if prediction.shape[1] == 1 else prediction[sample_idx, ...]
        has_multiple_time_steps = False
    else:  # (batch, *grid) - single time step, no channel
        pred = prediction[sample_idx, ...]
        has_multiple_time_steps = False
    
    if len(initial.shape) == len(grid_shape) + 1:  # (batch, *grid)
        init = initial[sample_idx, ...]
    else:  # (batch, channels, *grid)
        init = initial[sample_idx, 0, ...]
    
    # Get trajectory for this sample
    traj = solution_trajectory[sample_idx, ...]  # (nt, *spatial_dims)
    nt = traj.shape[0]
    
    # Select time steps to show (evenly spaced)
    if n_time_steps_to_show > nt:
        n_time_steps_to_show = nt
    time_indices = np.linspace(0, nt - 1, n_time_steps_to_show, dtype=int)
    
    # Get prediction at each time step
    if has_multiple_time_steps:
        # Get predictions at selected time indices
        pred_at_times = [pred_trajectory[t_idx, ...] for t_idx in time_indices]
        # Update vmin/vmax to include all predictions
        all_pred_vals = np.concatenate([p.flatten() for p in pred_at_times])
    else:
        pred_at_times = [pred] * len(time_indices)  # Same prediction for all
    
    # Determine dimension
    if len(grid_shape) == 1:
        # 1D: line plots over time
        fig, axes = plt.subplots(2, n_time_steps_to_show, figsize=(4 * n_time_steps_to_show, 8))
        if n_time_steps_to_show == 1:
            axes = axes.reshape(2, 1)
        
        x = np.linspace(0, 1, grid_shape[0])
        # Update vmin/vmax to include all predictions if multiple time steps
        if has_multiple_time_steps:
            vmin = min(traj.min(), all_pred_vals.min(), init.min())
            vmax = max(traj.max(), all_pred_vals.max(), init.max())
        else:
            vmin = min(traj.min(), pred.min(), init.min())
            vmax = max(traj.max(), pred.max(), init.max())
        
        for col, t_idx in enumerate(time_indices):
            t_val = time_steps[t_idx] if time_steps is not None else t_idx
            gt_at_t = traj[t_idx, ...]
            pred_at_t = pred_at_times[col]
            
            # Ground truth at this time
            axes[0, col].plot(x, gt_at_t, 'b-', linewidth=2, label='GT')
            axes[0, col].set_title(f'GT: t={t_val:.4f}')
            axes[0, col].set_ylim(vmin, vmax)
            axes[0, col].grid(True)
            if col == 0:
                axes[0, col].legend()
            
            # Prediction at this time step
            axes[1, col].plot(x, pred_at_t, 'r--', linewidth=2, label='Pred')
            axes[1, col].plot(x, gt_at_t, 'b-', linewidth=2, label='GT')
            axes[1, col].set_title(f'Pred vs GT: t={t_val:.4f}')
            axes[1, col].set_ylim(vmin, vmax)
            axes[1, col].grid(True)
            axes[1, col].legend()
    
    elif len(grid_shape) == 2:
        # 2D: heatmaps over time
        fig, axes = plt.subplots(2, n_time_steps_to_show, figsize=(4 * n_time_steps_to_show, 8))
        if n_time_steps_to_show == 1:
            axes = axes.reshape(2, 1)
        
        # Get common color scale
        if has_multiple_time_steps:
            vmin = min(traj.min(), all_pred_vals.min(), init.min())
            vmax = max(traj.max(), all_pred_vals.max(), init.max())
        else:
            vmin = min(traj.min(), pred.min(), init.min())
            vmax = max(traj.max(), pred.max(), init.max())
        
        for col, t_idx in enumerate(time_indices):
            t_val = time_steps[t_idx] if time_steps is not None else t_idx
            gt_at_t = traj[t_idx, ...]
            pred_at_t = pred_at_times[col]
            
            # Ground truth at this time step
            im0 = axes[0, col].imshow(gt_at_t, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
            axes[0, col].set_title(f'GT: t={t_val:.4f}')
            axes[0, col].set_xlabel('x')
            if col == 0:
                axes[0, col].set_ylabel('y')
            plt.colorbar(im0, ax=axes[0, col])
            
            # Prediction at this time step
            im1 = axes[1, col].imshow(pred_at_t, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
            axes[1, col].set_title(f'Pred: t={t_val:.4f}')
            axes[1, col].set_xlabel('x')
            if col == 0:
                axes[1, col].set_ylabel('y')
            plt.colorbar(im1, ax=axes[1, col])
    
    else:
        # 3D: show middle slice over time
        mid_z = grid_shape[0] // 2
        fig, axes = plt.subplots(2, n_time_steps_to_show, figsize=(4 * n_time_steps_to_show, 8))
        if n_time_steps_to_show == 1:
            axes = axes.reshape(2, 1)
        
        if has_multiple_time_steps:
            vmin = min(traj.min(), all_pred_vals.min(), init.min())
            vmax = max(traj.max(), all_pred_vals.max(), init.max())
        else:
            vmin = min(traj.min(), pred.min(), init.min())
            vmax = max(traj.max(), pred.max(), init.max())
        
        for col, t_idx in enumerate(time_indices):
            t_val = time_steps[t_idx] if time_steps is not None else t_idx
            gt_at_t = traj[t_idx, mid_z, ...]
            pred_at_t = pred_at_times[col]
            
            # Extract middle slice for 3D predictions
            if len(pred_at_t.shape) == 3:  # (z, y, x)
                pred_slice = pred_at_t[mid_z, ...]
            else:
                pred_slice = pred_at_t
            
            im0 = axes[0, col].imshow(gt_at_t, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
            axes[0, col].set_title(f'GT: t={t_val:.4f} (z={mid_z})')
            plt.colorbar(im0, ax=axes[0, col])
            
            im1 = axes[1, col].imshow(pred_slice, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
            axes[1, col].set_title(f'Pred: t={t_val:.4f} (z={mid_z})')
            plt.colorbar(im1, ax=axes[1, col])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on trained FNNOperator model"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing test.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save inference results (default: checkpoint_dir/../inference)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of samples to visualize (default: 10)",
    )
    parser.add_argument(
        "--n-time-steps",
        type=int,
        default=6,
        help="Number of time steps to show in temporal evolution (default: 6)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    
    args = parser.parse_args()
    
    # Resolve all paths relative to project root
    # Handle both "outputs/..." and "fnnoperator/outputs/..." paths
    if not args.checkpoint.is_absolute():
        checkpoint_str = str(args.checkpoint).replace("\\", "/")  # Normalize path separators
        # Try fnnoperator/outputs/ first if path starts with "outputs/"
        if checkpoint_str.startswith("outputs/"):
            # Try with fnnoperator prefix first (most common case)
            checkpoint_path1 = (project_root / "fnnoperator" / args.checkpoint).resolve()
            # Try without fnnoperator prefix as fallback
            checkpoint_path2 = (project_root / args.checkpoint).resolve()
            
            if checkpoint_path1.exists():
                args.checkpoint = checkpoint_path1
            elif checkpoint_path2.exists():
                args.checkpoint = checkpoint_path2
            else:
                # Neither exists, use the fnnoperator path (will show error later)
                args.checkpoint = checkpoint_path1
        else:
            args.checkpoint = (project_root / args.checkpoint).resolve()
    else:
        args.checkpoint = Path(args.checkpoint).resolve()
    
    if not args.data_dir.is_absolute():
        data_str = str(args.data_dir).replace("\\", "/")  # Normalize path separators
        # Try fnnoperator/data/ first if path starts with "data/"
        if data_str.startswith("data/"):
            # Try with fnnoperator prefix first (most common case)
            data_path1 = (project_root / "fnnoperator" / args.data_dir).resolve()
            # Try without fnnoperator prefix as fallback
            data_path2 = (project_root / args.data_dir).resolve()
            
            if data_path1.exists():
                args.data_dir = data_path1
            elif data_path2.exists():
                args.data_dir = data_path2
            else:
                # Neither exists, use the fnnoperator path (will show error later)
                args.data_dir = data_path1
        else:
            args.data_dir = (project_root / args.data_dir).resolve()
    else:
        args.data_dir = Path(args.data_dir).resolve()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.checkpoint.parent.parent / "inference"
    elif not args.output_dir.is_absolute():
        args.output_dir = (project_root / args.output_dir).resolve()
    else:
        args.output_dir = Path(args.output_dir).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify checkpoint exists
    if not args.checkpoint.exists():
        print(f"\nERROR: Checkpoint file not found: {args.checkpoint}")
        print(f"\nMake sure the checkpoint path is correct.")
        print(f"Expected location: fnnoperator/outputs/{{run_name}}/checkpoints/final_model.pt")
        print(f"\nTip: Use full path from project root:")
        print(f"  fnnoperator/outputs/heat_eq_2d_.../checkpoints/final_model.pt")
        return
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading model...")
    model, checkpoint = load_model(args.checkpoint, device)
    
    # Verify data directory exists
    if not args.data_dir.exists():
        print(f"\nERROR: Data directory not found: {args.data_dir}")
        print(f"\nMake sure the data directory path is correct.")
        print(f"Expected location: fnnoperator/data/{{grid_size}}/")
        return
    
    # Load test data (with full trajectory for temporal visualization)
    print("\nLoading test data...")
    X_test, Y_test, grid_shape, solution_trajectory, time_steps = load_test_data(
        args.data_dir, return_full_trajectory=True
    )
    print(f"  Test samples: {len(X_test)}")
    print(f"  Grid shape: {grid_shape}")
    print(f"  Input shape: {X_test.shape}")
    print(f"  Output shape: {Y_test.shape}")
    print(f"  Time steps: {solution_trajectory.shape[1]}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = predict(model, X_test, device, batch_size=args.batch_size)
    print(f"  Predictions shape: {predictions.shape}")
    
    # Check if model predicts multiple time steps
    n_time_steps_model = checkpoint.get('n_time_steps', 1)
    has_multiple_time_steps = n_time_steps_model > 1
    
    # Calculate overall metrics
    if has_multiple_time_steps:
        # Predictions: (batch, nt, channels, *grid)
        # Need to compare with full trajectory: (batch, nt, *spatial_dims)
        # Add channel dimension to trajectory for comparison
        Y_test_full = solution_trajectory[:, :, np.newaxis, ...]  # (batch, nt, 1, *grid)
        
        # Remove channel dimension from predictions if it's 1
        if predictions.shape[2] == 1:
            pred_compare = predictions[:, :, 0, ...]  # (batch, nt, *grid)
            gt_compare = solution_trajectory  # (batch, nt, *grid)
        else:
            pred_compare = predictions
            gt_compare = Y_test_full
        
        mse = np.mean((pred_compare - gt_compare) ** 2)
        mae = np.mean(np.abs(pred_compare - gt_compare))
        rmse = np.sqrt(mse)
        relative_error = mae / (np.abs(gt_compare).mean() + 1e-8) * 100
        
        print(f"\nOverall Metrics (all {n_time_steps_model} time steps):")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  Relative Error: {relative_error:.2f}%")
        
        # Also calculate metrics for final time step only
        final_pred = pred_compare[:, -1, ...]  # (batch, *grid)
        final_gt = gt_compare[:, -1, ...]  # (batch, *grid)
        final_mse = np.mean((final_pred - final_gt) ** 2)
        final_mae = np.mean(np.abs(final_pred - final_gt))
        final_rmse = np.sqrt(final_mse)
        print(f"\nFinal Time Step Metrics:")
        print(f"  MSE: {final_mse:.6f}")
        print(f"  MAE: {final_mae:.6f}")
        print(f"  RMSE: {final_rmse:.6f}")
    else:
        # Single time step (final state only)
        Y_test_np = Y_test.numpy()
        mse = np.mean((predictions - Y_test_np) ** 2)
        mae = np.mean(np.abs(predictions - Y_test_np))
        rmse = np.sqrt(mse)
        relative_error = mae / (np.abs(Y_test_np).mean() + 1e-8) * 100
        
        print(f"\nOverall Metrics (final state only):")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  Relative Error: {relative_error:.2f}%")
    
    # Load initial conditions for visualization
    test_data = np.load(args.data_dir / "test.npz", allow_pickle=True)
    initial_conditions = test_data["initial"]
    
    # Prepare predictions and ground truth for visualization
    if has_multiple_time_steps:
        # For comparison, use final time step
        if predictions.shape[2] == 1:  # Remove channel dimension if it's 1
            predictions_final = predictions[:, -1, 0, ...]  # (batch, *grid)
        else:
            predictions_final = predictions[:, -1, ...]  # (batch, channels, *grid)
        # Add channel dimension back for comparison function
        if len(predictions_final.shape) == len(grid_shape) + 1:
            predictions_final = predictions_final[:, np.newaxis, ...]  # (batch, 1, *grid)
        Y_test_np = solution_trajectory[:, -1, np.newaxis, ...]  # (batch, 1, *grid) - final state
    else:
        Y_test_np = Y_test.numpy()
        predictions_final = predictions
    
    # Create visualizations
    print(f"\nCreating visualizations for {min(args.n_samples, len(X_test))} samples...")
    print(f"  - Final state comparisons")
    print(f"  - Temporal evolution ({args.n_time_steps} time steps)")
    vis_dir = args.output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(min(args.n_samples, len(X_test))):
        # Final state comparison
        output_path = vis_dir / f"sample_{i:03d}_comparison.png"
        create_heatmap_comparison(
            Y_test_np,
            predictions_final,
            initial_conditions,
            i,
            output_path,
            grid_shape,
        )
        
        # Temporal evolution
        temporal_path = vis_dir / f"sample_{i:03d}_temporal_evolution.png"
        create_temporal_evolution(
            solution_trajectory,
            predictions,
            initial_conditions,
            i,
            time_steps,
            temporal_path,
            grid_shape,
            n_time_steps_to_show=args.n_time_steps,
        )
    
    # Save predictions
    predictions_file = args.output_dir / "predictions.npy"
    np.save(predictions_file, predictions)
    print(f"\nPredictions saved to {predictions_file}")
    
    # Save metrics
    import json
    metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'relative_error_percent': float(relative_error),
        'n_samples': len(X_test),
        'n_visualized': min(args.n_samples, len(X_test)),
    }
    metrics_file = args.output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_file}")
    
    print(f"\nAll inference results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

