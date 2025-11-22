"""
Inference script for trained FNO model on heat equation.

This script loads a trained FNO model and uses it to make predictions
on test/validation data, with visualization of results.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch

from neuralop.models import FNO
from neuralop.training.training_state import load_training_state
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer

# Import from train_fno module in the same directory
train_fno_path = Path(__file__).parent / "train_fno.py"
spec = importlib.util.spec_from_file_location("train_fno", train_fno_path)
train_fno = importlib.util.module_from_spec(spec)
sys.modules["train_fno"] = train_fno
spec.loader.exec_module(train_fno)

HeatEquationDataset = train_fno.HeatEquationDataset
create_model = train_fno.create_model


def load_model(
    checkpoint_dir: str | Path,
    checkpoint_name: str = "best_model",
    dimension: int = 1,
    n_modes: int = 16,
    hidden_channels: int = 64,
    n_layers: int = 4,
    device: str = "cpu",
):
    """Load trained FNO model from checkpoint.
    
    Args:
        checkpoint_dir: Directory containing saved checkpoints
        checkpoint_name: Name of checkpoint to load ('model', 'best_model', etc.)
        dimension: Spatial dimension (1, 2, or 3) - must match training config
        n_modes: Number of Fourier modes per dimension (must match training config)
        hidden_channels: Hidden channels (must match training config)
        n_layers: Number of layers (must match training config)
        device: Device to load model on
        
    Returns:
        Loaded model, epoch number (if available)
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir}\n"
            "To save checkpoints during training, use:\n"
            "  python heat_eq_data/train_fno.py --save-dir checkpoints\n"
            "Or specify the correct path to your checkpoint directory."
        )
    
    # Check if checkpoint exists
    checkpoint_path = checkpoint_dir / f"{checkpoint_name}_state_dict.pt"
    if not checkpoint_path.exists():
        # Try 'model' if 'best_model' doesn't exist
        if checkpoint_name == "best_model":
            checkpoint_path = checkpoint_dir / "model_state_dict.pt"
            checkpoint_name = "model"
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"No checkpoint found in {checkpoint_dir}. "
                    "Expected 'best_model_state_dict.pt' or 'model_state_dict.pt'"
                )
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create model with same architecture as training
    model = create_model(
        dimension=dimension,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        device=device,
    )
    
    # Load model weights
    print(f"Loading model from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load epoch from manifest if available
    epoch = None
    manifest_path = checkpoint_dir / "manifest.pt"
    if manifest_path.exists():
        manifest = torch.load(manifest_path, weights_only=False)
        epoch = manifest.get("epoch", None)
        if epoch is not None:
            print(f"Checkpoint from epoch {epoch}")
    
    return model, epoch


def load_data_processor(
    data_dir: str | Path,
    dimension: int | None = None,
    device: str = "cpu",
):
    """Load and recreate data processor from training data.
    
    Args:
        data_dir: Directory containing train.npz
        dimension: Spatial dimension (1, 2, or 3). If None, auto-detect from data.
        device: Device to place processor on
        
    Returns:
        Data processor, detected dimension
    """
    # Load training data to fit normalizers (dimension will be auto-detected if not provided)
    train_dataset = HeatEquationDataset(
        data_dir, split="train", use_final_state=True, dimension=dimension
    )
    
    # Use detected dimension
    dimension = train_dataset.dimension
    
    # Create normalizers - normalize over batch and all spatial dimensions
    # Shape: (batch, channels, *spatial_dims)
    # Normalize over batch (0) and all spatial dims (2, 3, ...)
    ndim = train_dataset.initial.ndim
    spatial_dims = list(range(2, ndim))
    normalize_dims = [0] + spatial_dims  # Normalize over batch + all spatial dims
    
    normalizer_x = UnitGaussianNormalizer(dim=normalize_dims)
    normalizer_y = UnitGaussianNormalizer(dim=normalize_dims)
    
    x_train_batch = train_dataset.initial
    y_train_batch = train_dataset.target
    normalizer_x.fit(x_train_batch)
    normalizer_y.fit(y_train_batch)
    
    data_processor = DefaultDataProcessor(
        in_normalizer=normalizer_x,
        out_normalizer=normalizer_y,
    )
    data_processor = data_processor.to(device)
    
    return data_processor, dimension


def predict(
    model: torch.nn.Module,
    data_processor: DefaultDataProcessor,
    initial_condition: torch.Tensor,
    device: str = "cpu",
):
    """Make prediction with trained model.
    
    Args:
        model: Trained FNO model
        data_processor: Data processor for normalization
        initial_condition: Initial condition tensor
            - 1D: (nx,) or (1, nx) or (1, 1, nx)
            - 2D: (ny, nx) or (1, ny, nx) or (1, 1, ny, nx)
            - 3D: (nz, ny, nx) or (1, nz, ny, nx) or (1, 1, nz, ny, nx)
        device: Device to run inference on
        
    Returns:
        Predicted final state (same spatial shape as input, without batch/channel dims)
    """
    model.eval()
    
    # Ensure correct shape: (1, 1, *spatial_dims)
    original_shape = initial_condition.shape
    
    if initial_condition.ndim == 1:
        # 1D: (nx,) -> (1, 1, nx)
        initial_condition = initial_condition.unsqueeze(0).unsqueeze(0)
    elif initial_condition.ndim == 2:
        if initial_condition.shape[0] == 1:
            # Already has channel dim: (1, nx) -> (1, 1, nx)
            initial_condition = initial_condition.unsqueeze(0)
        else:
            # 2D spatial: (ny, nx) -> (1, 1, ny, nx)
            initial_condition = initial_condition.unsqueeze(0).unsqueeze(0)
    elif initial_condition.ndim == 3:
        if initial_condition.shape[0] == 1:
            # Has batch dim: (1, ny, nx) -> (1, 1, ny, nx)
            initial_condition = initial_condition.unsqueeze(1)
        else:
            # 3D spatial: (nz, ny, nx) -> (1, 1, nz, ny, nx)
            initial_condition = initial_condition.unsqueeze(0).unsqueeze(0)
    elif initial_condition.ndim == 4:
        if initial_condition.shape[1] == 1:
            # Has batch and channel: (1, 1, nz, ny, nx) - already correct
            pass
        else:
            # 4D spatial: (1, nz, ny, nx) -> (1, 1, nz, ny, nx)
            initial_condition = initial_condition.unsqueeze(1)
    # If already (1, 1, ...), leave as is
    
    # Move to device
    initial_condition = initial_condition.to(device)
    
    # Preprocess
    data_dict = {"x": initial_condition, "y": torch.zeros_like(initial_condition)}
    data_dict = data_processor.preprocess(data_dict, batched=True)
    
    # Predict
    with torch.no_grad():
        pred = model(data_dict["x"])
    
    # Postprocess
    pred, _ = data_processor.postprocess(pred, data_dict)
    
    # Remove batch and channel dimensions
    return pred.squeeze().cpu()


def evaluate_model(
    model: torch.nn.Module,
    data_processor: DefaultDataProcessor,
    dataset: HeatEquationDataset,
    num_samples: int = 5,
    num_time_steps: int = 5,
    device: str = "cpu",
):
    """Evaluate model on dataset and return predictions at multiple time steps.
    
    Uses autoregressive prediction: predict t1 from t0, then t2 from t1, etc.
    
    Args:
        model: Trained FNO model
        data_processor: Data processor
        dataset: Dataset to evaluate on
        num_samples: Number of samples to evaluate
        num_time_steps: Number of time steps to predict
        device: Device to run on
        
    Returns:
        Dictionary with predictions, ground truth trajectories, and inputs
    """
    model.eval()
    # Ensure data processor is in eval mode so it denormalizes predictions
    if hasattr(data_processor, 'eval'):
        data_processor.eval()
    # Also set training=False explicitly to ensure eval behavior
    if hasattr(data_processor, 'training'):
        data_processor.training = False
    
    predictions = []  # List of trajectories: [sample][time_step]
    ground_truth_trajectories = []  # List of trajectories: [sample][time_step]
    inputs = []
    
    num_samples = min(num_samples, len(dataset))
    # Get time step indices to extract - use much larger spacing to show significant differences
    nt = len(dataset.t)
    
    # With total_time=0.008 and nt=64, dt ≈ 0.000127 seconds per step
    # To show visible differences, we need to select timestamps that are far apart
    # Strategy: Select evenly spaced indices with large gaps (e.g., every 12-15 steps for 5 timestamps)
    if num_time_steps == 1:
        time_indices = np.array([nt - 1], dtype=int)  # Just final time step
    elif num_time_steps == 2:
        time_indices = np.array([0, nt - 1], dtype=int)  # Initial and final
    else:
        # Calculate step size to get evenly spaced timestamps with large gaps
        # For 5 timestamps in 64 steps: step = 64/4 = 16, giving [0, 16, 32, 48, 63]
        step_size = (nt - 1) // (num_time_steps - 1)
        time_indices = []
        for i in range(num_time_steps - 1):
            idx = i * step_size
            time_indices.append(idx)
        # Always include the final time step
        time_indices.append(nt - 1)
        time_indices = np.array(time_indices, dtype=int)
        
        # Print selected indices and their time values for debugging
        if len(dataset.t) > 0:
            selected_times = dataset.t[time_indices].numpy()
            print(f"Selected time indices: {time_indices}")
            print(f"Selected time values: {selected_times}")
            print(f"Time differences between consecutive timestamps: {np.diff(selected_times)}")
    
    for i in range(num_samples):
        sample = dataset[i]
        x_raw = sample["x"]  # (1, *spatial_dims) from dataset - initial condition
        
        # Get full trajectory from raw solution data
        solution_raw = dataset.solution_raw[i]  # (nt, *spatial_dims)
        
        # Extract ground truth at selected time steps and process through data processor
        gt_trajectory = []
        for t_idx in time_indices:
            gt_at_t_raw = solution_raw[t_idx]  # (*spatial_dims) - raw tensor
            
            # Process through data processor for consistency with predictions
            # Shape: (*spatial_dims) -> (1, 1, *spatial_dims) for processing
            gt_at_t_tensor = gt_at_t_raw.unsqueeze(0).unsqueeze(0).to(device)
            data_dict_gt = {"x": torch.zeros_like(gt_at_t_tensor), "y": gt_at_t_tensor}
            data_dict_gt = data_processor.preprocess(data_dict_gt, batched=True)
            # Postprocess to get denormalized version
            dummy_out = torch.zeros_like(data_dict_gt["y"])
            _, data_dict_gt_post = data_processor.postprocess(dummy_out, data_dict_gt)
            gt_at_t_processed = data_dict_gt_post["y"].squeeze().cpu().numpy()
            gt_trajectory.append(gt_at_t_processed)
        
        # Predict each time step using the model
        # The model was trained to predict the final state (t=total_time) from initial condition.
        # To predict intermediate states, we use autoregressive prediction where each model call
        # is scaled to advance by the actual time difference between consecutive selected timestamps.
        pred_trajectory = []
        t_values = dataset.t.numpy()
        t_final = t_values[-1]
        
        # Start with initial condition
        current_state = x_raw.squeeze()
        prev_t_idx = time_indices[0]
        
        for i, t_idx in enumerate(time_indices):
            if i == 0:
                # First time step is the initial condition
                pred_trajectory.append(current_state.numpy().copy() if isinstance(current_state, torch.Tensor) else current_state.copy())
            else:
                # Calculate the actual time difference between this timestamp and the previous one
                t_prev = t_values[prev_t_idx]
                t_current = t_values[t_idx]
                delta_t = t_current - t_prev
                
                # Use model to predict: model(current) predicts total_time forward
                # But we only want to advance by delta_t, so scale: pred = current + (model(current) - current) * (delta_t / total_time)
                pred_final_from_current = predict(model, data_processor, current_state, device)
                pred_final_np = pred_final_from_current.numpy() if isinstance(pred_final_from_current, torch.Tensor) else pred_final_from_current
                current_np = current_state.numpy() if isinstance(current_state, torch.Tensor) else current_state
                
                # Scale the prediction to only advance by delta_t
                time_scale = delta_t / t_final
                pred_next = current_np + time_scale * (pred_final_np - current_np)
                pred_trajectory.append(pred_next)
                
                # Update current state for next iteration (autoregressive)
                current_state = torch.tensor(pred_next, dtype=torch.float32)
                prev_t_idx = t_idx
        
        # Convert initial condition to numpy
        x_final = x_raw.squeeze().numpy()
        
        predictions.append(pred_trajectory)
        ground_truth_trajectories.append(gt_trajectory)
        inputs.append(x_final)
    
    return {
        "predictions": predictions,  # List of trajectories: [sample][time_step]
        "ground_truth_trajectories": ground_truth_trajectories,  # List of lists: [sample][time_step]
        "inputs": inputs,
        "time_indices": time_indices,
        "time_values": dataset.t[time_indices].numpy(),
    }


def visualize_predictions(
    dataset: HeatEquationDataset,
    inputs: np.ndarray | list,
    predictions: np.ndarray | list,
    ground_truth_trajectories: list,
    time_values: np.ndarray,
    dimension: int,
    save_path: str | Path | None = None,
):
    """Visualize model predictions vs ground truth across time steps.
    
    Each row represents one sample.
    Column 0: Initial condition (single plot)
    Columns 1+: Time steps, each with two stacked plots:
        - Top: Ground truth heatmap at that time step
        - Bottom: Prediction heatmap at that time step
    
    Args:
        dataset: Dataset object (contains spatial coordinates)
        inputs: Initial conditions (list/array of samples)
        predictions: Model predictions - list of trajectories [sample][time_step] -> spatial data
        ground_truth_trajectories: List of trajectories [sample][time_step] -> spatial data
        time_values: Time values corresponding to each time step
        dimension: Spatial dimension (1, 2, or 3)
        save_path: Optional path to save figure
    """
    num_samples = len(predictions)
    num_time_steps = len(time_values)
    
    # Total columns: 1 (initial) + num_time_steps
    num_cols = 1 + num_time_steps
    
    # Use GridSpec for flexible layout with nested subplots
    # For each sample, we need: 1 full-height column (initial) + num_time_steps columns (each with 2 stacked plots)
    # Total grid: num_samples rows x num_cols columns, but time step columns need 2 subplots each
    from matplotlib.gridspec import GridSpec
    
    # Create figure with enough rows: each sample row actually needs 2 subplot rows for stacked plots
    fig = plt.figure(figsize=(4 * num_cols, 6 * num_samples))
    
    # Use subplot2grid for nested layouts - simpler approach
    # For each sample (row), create subplots
    for i in range(num_samples):
        input_i = inputs[i] if isinstance(inputs, (list, np.ndarray)) else inputs[i]
        pred_traj = predictions[i]  # List of predictions at each time step
        gt_traj = ground_truth_trajectories[i]  # List of ground truth at each time step
        
        if dimension == 1:
            x = dataset.x.numpy()
            input_i = input_i.flatten()
            
            # Column 0: Initial condition (spans both subplot rows)
            ax = plt.subplot2grid((num_samples * 2, num_cols), (i * 2, 0), rowspan=2, colspan=1)
            ax.plot(x, input_i, 'b-', linewidth=2, label='Initial condition')
            ax.set_xlabel('Position x', fontsize=10, fontweight='bold')
            ax.set_ylabel('Temperature u(x,0)', fontsize=10, fontweight='bold')
            ax.set_title(f'Sample {i+1}: t=0', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Columns 1+: Time steps with stacked subplots
            for t_idx, t_val in enumerate(time_values):
                col = 1 + t_idx
                gt_at_t = gt_traj[t_idx].flatten()
                pred_at_t = pred_traj[t_idx].flatten()
                
                # Top subplot: Ground truth (row i*2+1, col)
                ax_top = plt.subplot2grid((num_samples * 2, num_cols), (i * 2 + 1, col), rowspan=1, colspan=1)
                ax_top.plot(x, gt_at_t, 'g-', linewidth=2, label='Ground truth')
                ax_top.set_xlabel('Position x', fontsize=9)
                ax_top.set_ylabel('Temperature', fontsize=9)
                ax_top.set_title(f'GT: t={t_val:.4f}', fontsize=10, fontweight='bold')
                ax_top.grid(True, alpha=0.3)
                
                # Bottom subplot: Prediction (row i*2, col)
                ax_bottom = plt.subplot2grid((num_samples * 2, num_cols), (i * 2, col), rowspan=1, colspan=1)
                ax_bottom.plot(x, pred_at_t, 'r--', linewidth=2, label='Prediction')
                ax_bottom.plot(x, gt_at_t, 'g-', linewidth=1, alpha=0.5, label='Ground truth')
                ax_bottom.set_xlabel('Position x', fontsize=9)
                ax_bottom.set_ylabel('Temperature', fontsize=9)
                ax_bottom.set_title(f'Pred: t={t_val:.4f}', fontsize=10, fontweight='bold')
                ax_bottom.legend(fontsize=8)
                ax_bottom.grid(True, alpha=0.3)
                
                # Compute error
                error = np.mean((pred_at_t - gt_at_t)**2)
                ax_bottom.text(0.05, 0.95, f'MSE: {error:.6f}', transform=ax_bottom.transAxes,
                        fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        elif dimension == 2:
            x = dataset.x.numpy()
            y = dataset.y.numpy()
            X, Y = np.meshgrid(x, y)
            
            # Reshape to 2D if needed
            if input_i.ndim > 2:
                input_i = input_i.squeeze()
            
            # Column 0: Initial condition (spans both subplot rows)
            ax = plt.subplot2grid((num_samples * 2, num_cols), (i * 2, 0), rowspan=2, colspan=1)
            im = ax.contourf(X, Y, input_i, levels=20, cmap='viridis')
            ax.set_xlabel('Position x', fontsize=10, fontweight='bold')
            ax.set_ylabel('Position y', fontsize=10, fontweight='bold')
            ax.set_title(f'Sample {i+1}: t=0', fontsize=11, fontweight='bold')
            plt.colorbar(im, ax=ax)
            
            # Columns 1+: Time steps with stacked subplots
            for t_idx, t_val in enumerate(time_values):
                col = 1 + t_idx
                gt_at_t = gt_traj[t_idx]
                pred_at_t = pred_traj[t_idx]
                if gt_at_t.ndim > 2:
                    gt_at_t = gt_at_t.squeeze()
                if pred_at_t.ndim > 2:
                    pred_at_t = pred_at_t.squeeze()
                
                # Top subplot: Ground truth heatmap (row i*2+1, col)
                ax_top = plt.subplot2grid((num_samples * 2, num_cols), (i * 2 + 1, col), rowspan=1, colspan=1)
                im = ax_top.contourf(X, Y, gt_at_t, levels=20, cmap='viridis')
                ax_top.set_xlabel('Position x', fontsize=9)
                ax_top.set_ylabel('Position y', fontsize=9)
                ax_top.set_title(f'GT: t={t_val:.4f}', fontsize=10, fontweight='bold')
                plt.colorbar(im, ax=ax_top)
                
                # Bottom subplot: Prediction (row i*2, col)
                ax_bottom = plt.subplot2grid((num_samples * 2, num_cols), (i * 2, col), rowspan=1, colspan=1)
                im = ax_bottom.contourf(X, Y, pred_at_t, levels=20, cmap='viridis')
                ax_bottom.set_xlabel('Position x', fontsize=9)
                ax_bottom.set_ylabel('Position y', fontsize=9)
                ax_bottom.set_title(f'Pred: t={t_val:.4f}', fontsize=10, fontweight='bold')
                plt.colorbar(im, ax=ax_bottom)
                
                # Compute error
                error = np.mean((pred_at_t - gt_at_t)**2)
                ax_bottom.text(0.05, 0.95, f'MSE: {error:.6f}', transform=ax_bottom.transAxes,
                        fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        else:  # dimension == 3
            x = dataset.x.numpy()
            y = dataset.y.numpy()
            z = dataset.z.numpy()
            X, Y = np.meshgrid(x, y)
            z_mid_idx = len(z) // 2
            
            # Reshape to 3D if needed
            if input_i.ndim > 3:
                input_i = input_i.squeeze()
            
            # Extract middle slice for initial condition
            input_slice = input_i[z_mid_idx, :, :]
            
            # Column 0: Initial condition (spans both subplot rows)
            ax = plt.subplot2grid((num_samples * 2, num_cols), (i * 2, 0), rowspan=2, colspan=1)
            im = ax.contourf(X, Y, input_slice, levels=20, cmap='viridis')
            ax.set_xlabel('Position x', fontsize=10, fontweight='bold')
            ax.set_ylabel('Position y', fontsize=10, fontweight='bold')
            ax.set_title(f'Sample {i+1}: t=0 (z={z[z_mid_idx]:.3f})', fontsize=11, fontweight='bold')
            plt.colorbar(im, ax=ax)
            
            # Columns 1+: Time steps with stacked subplots
            for t_idx, t_val in enumerate(time_values):
                col = 1 + t_idx
                gt_at_t = gt_traj[t_idx]
                pred_at_t = pred_traj[t_idx]
                if gt_at_t.ndim > 3:
                    gt_at_t = gt_at_t.squeeze()
                if pred_at_t.ndim > 3:
                    pred_at_t = pred_at_t.squeeze()
                gt_slice = gt_at_t[z_mid_idx, :, :]
                pred_slice = pred_at_t[z_mid_idx, :, :]
                
                # Top subplot: Ground truth heatmap (row i*2+1, col)
                ax_top = plt.subplot2grid((num_samples * 2, num_cols), (i * 2 + 1, col), rowspan=1, colspan=1)
                im = ax_top.contourf(X, Y, gt_slice, levels=20, cmap='viridis')
                ax_top.set_xlabel('Position x', fontsize=9)
                ax_top.set_ylabel('Position y', fontsize=9)
                ax_top.set_title(f'GT: t={t_val:.4f} (z={z[z_mid_idx]:.3f})', fontsize=10, fontweight='bold')
                plt.colorbar(im, ax=ax_top)
                
                # Bottom subplot: Prediction (row i*2, col)
                ax_bottom = plt.subplot2grid((num_samples * 2, num_cols), (i * 2, col), rowspan=1, colspan=1)
                im = ax_bottom.contourf(X, Y, pred_slice, levels=20, cmap='viridis')
                ax_bottom.set_xlabel('Position x', fontsize=9)
                ax_bottom.set_ylabel('Position y', fontsize=9)
                ax_bottom.set_title(f'Pred: t={t_val:.4f} (z={z[z_mid_idx]:.3f})', fontsize=10, fontweight='bold')
                plt.colorbar(im, ax=ax_bottom)
                
                # Compute error (full 3D)
                error = np.mean((pred_at_t - gt_at_t)**2)
                ax_bottom.text(0.05, 0.95, f'MSE: {error:.6f}', transform=ax_bottom.transAxes,
                        fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def animate_trajectory(
    dataset: HeatEquationDataset,
    trajectory: list | np.ndarray,
    time_values: np.ndarray,
    dimension: int,
    title: str = "Evolution Over Time",
    sample_idx: int = 0,
    interval: int = 100,
    save_path: str | Path | None = None,
) -> animation.FuncAnimation:
    """Animate the evolution of a trajectory over time.
    
    Args:
        dataset: Dataset object (contains spatial coordinates)
        trajectory: List or array of time steps, each with shape (*spatial_dims)
        time_values: Time values for each frame
        dimension: Spatial dimension (1, 2, or 3)
        title: Title for the animation
        sample_idx: Sample index (for display)
        interval: Animation frame interval in milliseconds
        save_path: Optional path to save animation (GIF or MP4)
        
    Returns:
        Animation object
    """
    if dimension == 1:
        x = dataset.x.numpy()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get data range for y-axis
        all_data = np.concatenate([t.flatten() for t in trajectory])
        y_min, y_max = all_data.min(), all_data.max()
        
        line, = ax.plot([], [], linewidth=2, color="blue")
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Position x", fontsize=12, fontweight="bold")
        ax.set_ylabel("Temperature u(x,t)", fontsize=12, fontweight="bold")
        ax.set_title(f"{title} (Sample {sample_idx + 1})", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        
        time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=12,
                            verticalalignment="top", 
                            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        
        def animate(frame):
            data = trajectory[frame].flatten()
            line.set_data(x, data)
            time_text.set_text(f"Time: t = {time_values[frame]:.4f}")
            return line, time_text
        
    elif dimension == 2:
        x = dataset.x.numpy()
        y = dataset.y.numpy()
        X, Y = np.meshgrid(x, y)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get data range for colorbar
        all_data = np.concatenate([t.flatten() for t in trajectory])
        vmin, vmax = all_data.min(), all_data.max()
        
        # Initial contour plot
        data_0 = trajectory[0].squeeze()
        if data_0.ndim > 2:
            data_0 = data_0.squeeze()
        im = ax.contourf(X, Y, data_0, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label='Temperature')
        ax.set_xlabel("Position x", fontsize=12, fontweight="bold")
        ax.set_ylabel("Position y", fontsize=12, fontweight="bold")
        ax.set_title(f"{title} (Sample {sample_idx + 1})", fontsize=14, fontweight="bold")
        
        time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=12,
                            verticalalignment="top", 
                            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        
        def animate(frame):
            ax.clear()
            data = trajectory[frame].squeeze()
            if data.ndim > 2:
                data = data.squeeze()
            im = ax.contourf(X, Y, data, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_xlabel("Position x", fontsize=12, fontweight="bold")
            ax.set_ylabel("Position y", fontsize=12, fontweight="bold")
            ax.set_title(f"{title} (Sample {sample_idx + 1})", fontsize=14, fontweight="bold")
            time_text = ax.text(0.02, 0.98, f"Time: t = {time_values[frame]:.4f}", 
                               transform=ax.transAxes, fontsize=12,
                               verticalalignment="top", 
                               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            return [im, time_text]
        
    else:  # dimension == 3
        x = dataset.x.numpy()
        y = dataset.y.numpy()
        z = dataset.z.numpy()
        X, Y = np.meshgrid(x, y)
        z_mid_idx = len(z) // 2
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get data range for colorbar (from middle slice)
        all_data = np.concatenate([t[z_mid_idx, :, :].flatten() for t in trajectory])
        vmin, vmax = all_data.min(), all_data.max()
        
        # Initial contour plot
        data_0 = trajectory[0][z_mid_idx, :, :]
        im = ax.contourf(X, Y, data_0, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label='Temperature')
        ax.set_xlabel("Position x", fontsize=12, fontweight="bold")
        ax.set_ylabel("Position y", fontsize=12, fontweight="bold")
        ax.set_title(f"{title} (Sample {sample_idx + 1}, z={z[z_mid_idx]:.3f})", fontsize=14, fontweight="bold")
        
        time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=12,
                            verticalalignment="top", 
                            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        
        def animate(frame):
            ax.clear()
            data = trajectory[frame][z_mid_idx, :, :]
            im = ax.contourf(X, Y, data, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_xlabel("Position x", fontsize=12, fontweight="bold")
            ax.set_ylabel("Position y", fontsize=12, fontweight="bold")
            ax.set_title(f"{title} (Sample {sample_idx + 1}, z={z[z_mid_idx]:.3f})", fontsize=14, fontweight="bold")
            time_text = ax.text(0.02, 0.98, f"Time: t = {time_values[frame]:.4f}", 
                               transform=ax.transAxes, fontsize=12,
                               verticalalignment="top", 
                               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            return [im, time_text]
    
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(trajectory),
        interval=interval,
        blit=(dimension == 1),  # blit only works well for 1D
        repeat=True,
    )
    
    if save_path:
        save_path = Path(save_path)
        print(f"    Saving animation to {save_path}...", end="", flush=True)
        try:
            if save_path.suffix.lower() == ".gif":
                anim.save(save_path, writer="pillow", fps=1000 // interval)
            elif save_path.suffix.lower() in [".mp4", ".avi"]:
                anim.save(save_path, writer="ffmpeg", fps=1000 // interval)
            else:
                raise ValueError(f"Unsupported file format: {save_path.suffix}. Use .gif or .mp4")
            print(f" Done")
        except Exception as e:
            print(f" Failed: {e}")
            print("      (You may need to install 'pillow' for GIF or 'ffmpeg' for MP4)")
    
    plt.tight_layout()
    return anim


def animate_comparison(
    dataset: HeatEquationDataset,
    gt_trajectory: list | np.ndarray,
    pred_trajectory: list | np.ndarray,
    time_values: np.ndarray,
    dimension: int,
    sample_idx: int = 0,
    interval: int = 100,
    save_path: str | Path | None = None,
) -> animation.FuncAnimation:
    """Animate ground truth and prediction side by side.
    
    Args:
        dataset: Dataset object (contains spatial coordinates)
        gt_trajectory: Ground truth trajectory, list or array of time steps
        pred_trajectory: Prediction trajectory, list or array of time steps
        time_values: Time values for each frame
        dimension: Spatial dimension (1, 2, or 3)
        sample_idx: Sample index (for display)
        interval: Animation frame interval in milliseconds
        save_path: Optional path to save animation (GIF or MP4)
        
    Returns:
        Animation object
    """
    if dimension == 1:
        x = dataset.x.numpy()
        fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Get data range for y-axis (from both trajectories)
        all_data = np.concatenate([t.flatten() for t in gt_trajectory] + [t.flatten() for t in pred_trajectory])
        y_min, y_max = all_data.min(), all_data.max()
        
        # Ground truth plot
        line_gt, = ax_gt.plot([], [], linewidth=2, color="blue")
        ax_gt.set_xlim(x.min(), x.max())
        ax_gt.set_ylim(y_min, y_max)
        ax_gt.set_xlabel("Position x", fontsize=12, fontweight="bold")
        ax_gt.set_ylabel("Temperature u(x,t)", fontsize=12, fontweight="bold")
        ax_gt.set_title("Ground Truth", fontsize=14, fontweight="bold")
        ax_gt.grid(True, alpha=0.3)
        
        # Prediction plot
        line_pred, = ax_pred.plot([], [], linewidth=2, color="red")
        ax_pred.set_xlim(x.min(), x.max())
        ax_pred.set_ylim(y_min, y_max)
        ax_pred.set_xlabel("Position x", fontsize=12, fontweight="bold")
        ax_pred.set_ylabel("Temperature u(x,t)", fontsize=12, fontweight="bold")
        ax_pred.set_title("Prediction", fontsize=14, fontweight="bold")
        ax_pred.grid(True, alpha=0.3)
        
        # Time text (shared)
        time_text = fig.suptitle("", fontsize=14, fontweight="bold")
        
        def animate(frame):
            gt_data = gt_trajectory[frame].flatten()
            pred_data = pred_trajectory[frame].flatten()
            line_gt.set_data(x, gt_data)
            line_pred.set_data(x, pred_data)
            time_text.set_text(f"Sample {sample_idx + 1} - Time: t = {time_values[frame]:.4f}")
            return line_gt, line_pred, time_text
        
    elif dimension == 2:
        x = dataset.x.numpy()
        y = dataset.y.numpy()
        X, Y = np.meshgrid(x, y)
        
        fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Get data range for colorbar (from both trajectories)
        all_data = np.concatenate([t.flatten() for t in gt_trajectory] + [t.flatten() for t in pred_trajectory])
        vmin, vmax = all_data.min(), all_data.max()
        
        # Initial contour plots
        gt_data_0 = gt_trajectory[0].squeeze()
        pred_data_0 = pred_trajectory[0].squeeze()
        if gt_data_0.ndim > 2:
            gt_data_0 = gt_data_0.squeeze()
        if pred_data_0.ndim > 2:
            pred_data_0 = pred_data_0.squeeze()
        
        im_gt = ax_gt.contourf(X, Y, gt_data_0, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(im_gt, ax=ax_gt, label='Temperature')
        ax_gt.set_xlabel("Position x", fontsize=12, fontweight="bold")
        ax_gt.set_ylabel("Position y", fontsize=12, fontweight="bold")
        ax_gt.set_title("Ground Truth", fontsize=14, fontweight="bold")
        
        im_pred = ax_pred.contourf(X, Y, pred_data_0, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(im_pred, ax=ax_pred, label='Temperature')
        ax_pred.set_xlabel("Position x", fontsize=12, fontweight="bold")
        ax_pred.set_ylabel("Position y", fontsize=12, fontweight="bold")
        ax_pred.set_title("Prediction", fontsize=14, fontweight="bold")
        
        # Time text (shared)
        time_text = fig.suptitle("", fontsize=14, fontweight="bold")
        
        # Store colorbar references
        cbar_gt = None
        cbar_pred = None
        
        def animate(frame):
            nonlocal cbar_gt, cbar_pred
            
            # Remove old colorbars if they exist
            if cbar_gt is not None:
                cbar_gt.remove()
            if cbar_pred is not None:
                cbar_pred.remove()
            
            ax_gt.clear()
            ax_pred.clear()
            
            gt_data = gt_trajectory[frame].squeeze()
            pred_data = pred_trajectory[frame].squeeze()
            if gt_data.ndim > 2:
                gt_data = gt_data.squeeze()
            if pred_data.ndim > 2:
                pred_data = pred_data.squeeze()
            
            im_gt = ax_gt.contourf(X, Y, gt_data, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
            ax_gt.set_xlabel("Position x", fontsize=12, fontweight="bold")
            ax_gt.set_ylabel("Position y", fontsize=12, fontweight="bold")
            ax_gt.set_title("Ground Truth", fontsize=14, fontweight="bold")
            cbar_gt = plt.colorbar(im_gt, ax=ax_gt, label='Temperature')
            
            im_pred = ax_pred.contourf(X, Y, pred_data, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
            ax_pred.set_xlabel("Position x", fontsize=12, fontweight="bold")
            ax_pred.set_ylabel("Position y", fontsize=12, fontweight="bold")
            ax_pred.set_title("Prediction", fontsize=14, fontweight="bold")
            cbar_pred = plt.colorbar(im_pred, ax=ax_pred, label='Temperature')
            
            time_text.set_text(f"Sample {sample_idx + 1} - Time: t = {time_values[frame]:.4f}")
            return [im_gt, im_pred, time_text]
        
    else:  # dimension == 3
        x = dataset.x.numpy()
        y = dataset.y.numpy()
        z = dataset.z.numpy()
        X, Y = np.meshgrid(x, y)
        z_mid_idx = len(z) // 2
        
        fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Get data range for colorbar (from middle slice of both trajectories)
        all_data = np.concatenate([t[z_mid_idx, :, :].flatten() for t in gt_trajectory] + 
                                  [t[z_mid_idx, :, :].flatten() for t in pred_trajectory])
        vmin, vmax = all_data.min(), all_data.max()
        
        # Initial contour plots
        gt_data_0 = gt_trajectory[0][z_mid_idx, :, :]
        pred_data_0 = pred_trajectory[0][z_mid_idx, :, :]
        
        im_gt = ax_gt.contourf(X, Y, gt_data_0, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(im_gt, ax=ax_gt, label='Temperature')
        ax_gt.set_xlabel("Position x", fontsize=12, fontweight="bold")
        ax_gt.set_ylabel("Position y", fontsize=12, fontweight="bold")
        ax_gt.set_title(f"Ground Truth (z={z[z_mid_idx]:.3f})", fontsize=14, fontweight="bold")
        
        im_pred = ax_pred.contourf(X, Y, pred_data_0, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(im_pred, ax=ax_pred, label='Temperature')
        ax_pred.set_xlabel("Position x", fontsize=12, fontweight="bold")
        ax_pred.set_ylabel("Position y", fontsize=12, fontweight="bold")
        ax_pred.set_title(f"Prediction (z={z[z_mid_idx]:.3f})", fontsize=14, fontweight="bold")
        
        # Time text (shared)
        time_text = fig.suptitle("", fontsize=14, fontweight="bold")
        
        # Store colorbar references
        cbar_gt = None
        cbar_pred = None
        
        def animate(frame):
            nonlocal cbar_gt, cbar_pred
            
            # Remove old colorbars if they exist
            if cbar_gt is not None:
                cbar_gt.remove()
            if cbar_pred is not None:
                cbar_pred.remove()
            
            ax_gt.clear()
            ax_pred.clear()
            
            gt_data = gt_trajectory[frame][z_mid_idx, :, :]
            pred_data = pred_trajectory[frame][z_mid_idx, :, :]
            
            im_gt = ax_gt.contourf(X, Y, gt_data, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
            ax_gt.set_xlabel("Position x", fontsize=12, fontweight="bold")
            ax_gt.set_ylabel("Position y", fontsize=12, fontweight="bold")
            ax_gt.set_title(f"Ground Truth (z={z[z_mid_idx]:.3f})", fontsize=14, fontweight="bold")
            cbar_gt = plt.colorbar(im_gt, ax=ax_gt, label='Temperature')
            
            im_pred = ax_pred.contourf(X, Y, pred_data, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
            ax_pred.set_xlabel("Position x", fontsize=12, fontweight="bold")
            ax_pred.set_ylabel("Position y", fontsize=12, fontweight="bold")
            ax_pred.set_title(f"Prediction (z={z[z_mid_idx]:.3f})", fontsize=14, fontweight="bold")
            cbar_pred = plt.colorbar(im_pred, ax=ax_pred, label='Temperature')
            
            time_text.set_text(f"Sample {sample_idx + 1} - Time: t = {time_values[frame]:.4f}")
            return [im_gt, im_pred, time_text]
    
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(gt_trajectory),
        interval=interval,
        blit=(dimension == 1),  # blit only works well for 1D
        repeat=True,
    )
    
    if save_path:
        save_path = Path(save_path)
        print(f"    Saving animation to {save_path}...", end="", flush=True)
        try:
            if save_path.suffix.lower() == ".gif":
                anim.save(save_path, writer="pillow", fps=1000 // interval)
            elif save_path.suffix.lower() in [".mp4", ".avi"]:
                anim.save(save_path, writer="ffmpeg", fps=1000 // interval)
            else:
                raise ValueError(f"Unsupported file format: {save_path.suffix}. Use .gif or .mp4")
            print(f" Done")
        except Exception as e:
            print(f" Failed: {e}")
            print("      (You may need to install 'pillow' for GIF or 'ffmpeg' for MP4)")
    
    plt.tight_layout()
    return anim


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="Run inference with trained FNO model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing saved model checkpoints",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="best_model",
        help="Name of checkpoint to load ('model', 'best_model', etc.)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("heat_eq_data/data"),
        help="Directory containing train.npz, test.npz, validate.npz",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "validate"],
        default="test",
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=None,
        choices=[1, 2, 3],
        help="Spatial dimension (1, 2, or 3). If None, auto-detect from data.",
    )
    parser.add_argument(
        "--n-modes",
        type=int,
        default=16,
        help="Number of Fourier modes per dimension (must match training config)",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=64,
        help="Hidden channels (must match training config)",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=4,
        help="Number of layers (must match training config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cpu', 'cuda', or None for auto-detect)",
    )
    parser.add_argument(
        "--save-fig",
        type=Path,
        default=None,
        help="Path to save visualization figure (optional)",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Create animations of ground truth and predictions",
    )
    parser.add_argument(
        "--animation-dir",
        type=Path,
        default=None,
        help="Directory to save animations (default: same as --save-fig directory or current dir)",
    )
    parser.add_argument(
        "--animation-interval",
        type=int,
        default=200,
        help="Animation frame interval in milliseconds (default: 200, higher = slower)",
    )
    parser.add_argument(
        "--animation-format",
        type=str,
        default="gif",
        choices=["gif", "mp4"],
        help="Animation file format (gif or mp4)",
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Loading model from: {args.checkpoint_dir}")
    print(f"Evaluating on: {args.split} split")
    print(f"Dimension: {args.dimension if args.dimension else 'auto-detect'}")
    print()
    
    # Load dataset first to detect dimension (if not provided)
    print(f"Loading {args.split} dataset...")
    dataset = HeatEquationDataset(
        args.data_dir, split=args.split, use_final_state=True, dimension=args.dimension
    )
    
    # Use detected dimension
    dimension = dataset.dimension
    print(f"Detected dimension: {dimension}D")
    print()
    
    # Load model with correct dimension
    model, epoch = load_model(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        dimension=dimension,
        n_modes=args.n_modes,
        hidden_channels=args.hidden_channels,
        n_layers=args.n_layers,
        device=device,
    )
    
    # Load data processor (will auto-detect dimension from data)
    print("Loading data processor...")
    data_processor, _ = load_data_processor(args.data_dir, dimension=dimension, device=device)
    
    # Evaluate model
    print(f"Evaluating on {args.num_samples} samples...")
    # Use fewer time steps with larger gaps between them for more visible differences
    # With total_time=0.008 and nt=64, we want timestamps that are far apart
    num_time_steps = min(5, len(dataset.t))  # Extract 5 time steps (or fewer if not enough)
    
    # Print time information for debugging
    if len(dataset.t) > 0:
        total_time = dataset.t[-1].item() if hasattr(dataset.t[-1], 'item') else float(dataset.t[-1])
        dt = total_time / (len(dataset.t) - 1) if len(dataset.t) > 1 else 0
        print(f"Time information: total_time={total_time:.6f}, nt={len(dataset.t)}, dt={dt:.6f}")
    results = evaluate_model(
        model=model,
        data_processor=data_processor,
        dataset=dataset,
        num_samples=args.num_samples,
        num_time_steps=num_time_steps,
        device=device,
    )
    
    # Visualize
    print("Visualizing predictions...")
    visualize_predictions(
        dataset=dataset,
        inputs=results["inputs"],
        predictions=results["predictions"],
        ground_truth_trajectories=results["ground_truth_trajectories"],
        time_values=results["time_values"],
        dimension=dimension,
        save_path=args.save_fig,
    )
    
    # Create animations if requested
    if args.animate:
        print("\nCreating animations...")
        
        # Determine animation save directory
        if args.animation_dir:
            anim_dir = Path(args.animation_dir)
        elif args.save_fig:
            anim_dir = Path(args.save_fig).parent
        else:
            anim_dir = Path(".")
        anim_dir.mkdir(parents=True, exist_ok=True)
        
        # Create side-by-side animations for each sample
        for sample_idx in range(min(args.num_samples, len(results["predictions"]))):
            print(f"  Sample {sample_idx + 1}:")
            
            gt_traj = results["ground_truth_trajectories"][sample_idx]
            pred_traj = results["predictions"][sample_idx]
            anim_path = anim_dir / f"comparison_sample_{sample_idx + 1}.{args.animation_format}"
            animate_comparison(
                dataset=dataset,
                gt_trajectory=gt_traj,
                pred_trajectory=pred_traj,
                time_values=results["time_values"],
                dimension=dimension,
                sample_idx=sample_idx,
                interval=args.animation_interval,
                save_path=anim_path,
            )
            plt.close()  # Close figure to free memory
        
        print(f"\nAnimations saved to {anim_dir}")
    
    # Compute overall statistics (using final time step predictions)
    # Handle both array and list inputs
    preds = np.array(results["predictions"]) if isinstance(results["predictions"], list) else results["predictions"]
    # Get final time step ground truth from trajectories
    gt_final = np.array([traj[-1] for traj in results["ground_truth_trajectories"]])
    
    # Flatten for error computation
    preds_flat = preds.flatten() if isinstance(preds, np.ndarray) else np.concatenate([p.flatten() for p in preds])
    truths_flat = gt_final.flatten() if isinstance(gt_final, np.ndarray) else np.concatenate([t.flatten() for t in gt_final])
    
    mse = np.mean((preds_flat - truths_flat)**2)
    mae = np.mean(np.abs(preds_flat - truths_flat))
    print(f"\nEvaluation Statistics:")
    print(f"  Mean Squared Error (MSE): {mse:.6f}")
    print(f"  Mean Absolute Error (MAE): {mae:.6f}")
    print(f"  Number of samples evaluated: {args.num_samples}")
    print(f"  Dimension: {dimension}D")


if __name__ == "__main__":
    main()

