"""
Train FNO (Fourier Neural Operator) on heat equation dataset.

This script trains an FNO to learn the mapping from initial conditions
to the final state of the heat equation solution.

Usage:
    python fnnoperator/training/train_heat_fno.py --data-dir neuraloperator/heat_eq_data/data --dimension 2
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent  # Go up to neuraloperator/
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from neuralop.models import FNO
from fnnoperator.training.trainers.FNNTrainer import FNNTrainer
from fnnoperator.Losses import MSELoss, SpectralLoss, CombinedLoss, PhysicsInformedLoss
from fnnoperator.Error_Analysis import compute_spectral_error


def load_heat_data(data_dir: Path, split: str = "train"):
    """Load heat equation data from NPZ file.
    
    Args:
        data_dir: Directory containing train.npz, test.npz, validate.npz
        split: Which split to load ('train', 'test', 'validate')
    
    Returns:
        tuple: (initial_conditions, final_states, dimension, grid_shape, all_states, n_time_steps)
            - initial_conditions: (n_samples, *spatial_dims) tensor
            - final_states: (n_samples, *spatial_dims) tensor
            - dimension: int (1, 2, or 3)
            - grid_shape: tuple of spatial dimensions
            - all_states: (n_samples, nt, *spatial_dims) tensor
            - n_time_steps: int
    """
    file_path = data_dir / f"{split}.npz"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load data
    data = np.load(file_path, allow_pickle=True)
    initial_raw = data["initial"]  # (n_samples, *spatial_dims)
    solution_raw = data["solution"]  # (n_samples, nt, *spatial_dims)
    
    # Detect dimension from data shape
    dimension = None
    if "metadata" in data:
        metadata = data["metadata"]
        for item in metadata:
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "dimension":
                dimension = int(item[1])
                break
    
    # If not in metadata, infer from initial condition shape
    if dimension is None:
        spatial_dims = len(initial_raw.shape) - 1  # Subtract batch dimension
        dimension = spatial_dims
    
    # Get all time steps
    n_time_steps = solution_raw.shape[1]
    all_states = solution_raw  # (n_samples, nt, *spatial_dims)
    final_states = solution_raw[:, -1, ...]  # (n_samples, *spatial_dims)
    
    # Determine grid shape based on dimension
    if dimension == 1:
        grid_shape = (initial_raw.shape[1],)  # (nx,)
    elif dimension == 2:
        grid_shape = (initial_raw.shape[1], initial_raw.shape[2])  # (ny, nx)
    elif dimension == 3:
        grid_shape = (initial_raw.shape[1], initial_raw.shape[2], initial_raw.shape[3])  # (nz, ny, nx)
    else:
        raise ValueError(f"Unsupported dimension: {dimension}")
    
    # Convert to tensors
    initial_conditions = torch.tensor(initial_raw, dtype=torch.float32)
    final_states = torch.tensor(final_states, dtype=torch.float32)
    all_states = torch.tensor(all_states, dtype=torch.float32)
    
    return initial_conditions, final_states, dimension, grid_shape, all_states, n_time_steps


def prepare_data_for_fno(initial_conditions, states, dimension, use_all_time_steps=False):
    """Prepare data for FNO.
    
    FNO expects input shape: (batch, channels, *spatial_dims)
    We need to add a channel dimension.
    
    Args:
        initial_conditions: (n_samples, *spatial_dims)
        states: If use_all_time_steps=False: (n_samples, *spatial_dims) - final state
                If use_all_time_steps=True: (n_samples, nt, *spatial_dims) - all time steps
        dimension: int (1, 2, or 3)
        use_all_time_steps: If True, prepare data for all time steps prediction
    
    Returns:
        If use_all_time_steps=False: (X, Y) where both have shape (n_samples, 1, *spatial_dims)
        If use_all_time_steps=True: (X, Y) where X is (n_samples, 1, *spatial_dims) and Y is (n_samples, nt, 1, *spatial_dims)
    """
    # Add channel dimension to input: (n_samples, *spatial_dims) -> (n_samples, 1, *spatial_dims)
    X = initial_conditions.unsqueeze(1)
    
    if use_all_time_steps:
        # states is (n_samples, nt, *spatial_dims)
        # Add channel dimension: (n_samples, nt, *spatial_dims) -> (n_samples, nt, 1, *spatial_dims)
        Y = states.unsqueeze(2)  # Insert channel dimension after time dimension
    else:
        # states is (n_samples, *spatial_dims) - final state only
        # Add channel dimension: (n_samples, *spatial_dims) -> (n_samples, 1, *spatial_dims)
        Y = states.unsqueeze(1)
    
    return X, Y


def main():
    parser = argparse.ArgumentParser(
        description="Train FNO on heat equation dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("neuraloperator/heat_eq_data/data"),
        help="Directory containing train.npz, test.npz, validate.npz",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=None,
        choices=[1, 2, 3],
        help="Spatial dimension (1, 2, or 3). If None, auto-detect from data.",
    )
    parser.add_argument(
        "--in-channels",
        type=int,
        default=1,
        help="Number of input channels (default: 1)",
    )
    parser.add_argument(
        "--out-channels",
        type=int,
        default=1,
        help="Number of output channels (default: 1)",
    )
    parser.add_argument(
        "--n-modes",
        type=int,
        nargs="+",
        default=None,
        help="Number of Fourier modes along each dimension (default: auto-based on grid size). "
             "For 2D, can specify as --n-modes 16 16",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=128,
        help="Number of hidden channels in FNO (default: 128, increased for better expressivity)",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=4,
        help="Number of FNO layers (default: 4)",
    )
    parser.add_argument(
        "--predict-all-time-steps",
        action="store_true",
        help="If set, predict all time steps instead of just final state",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=200,
        help="Number of training epochs (default: 200, increased for better convergence)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32). Reduce to 1-4 for large models to avoid OOM.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps (default: 1). Use >1 to simulate larger batch size.",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Use Automatic Mixed Precision (AMP) for faster training and lower memory usage",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--loss-function",
        type=str,
        default="mse",
        choices=["mse", "spectral", "combined"],
        help="Loss function to use: 'mse', 'spectral', or 'combined' (spectral + mse)",
    )
    parser.add_argument(
        "--spectral-normalize",
        action="store_true",
        default=True,
        help="Normalize spectral loss by target magnitude (default: True)",
    )
    parser.add_argument(
        "--no-spectral-normalize",
        dest="spectral_normalize",
        action="store_false",
        help="Disable normalization in spectral loss",
    )
    parser.add_argument(
        "--spectral-weight-low",
        type=float,
        default=1.0,
        help="Weight for low-frequency components in spectral loss (default: 1.0)",
    )
    parser.add_argument(
        "--spectral-weight-high",
        type=float,
        default=1.0,
        help="Weight for high-frequency components in spectral loss (default: 1.0)",
    )
    parser.add_argument(
        "--spectral-freq-threshold",
        type=float,
        default=0.25,
        help="Frequency threshold to separate low/high frequencies in spectral loss (default: 0.25)",
    )
    parser.add_argument(
        "--combined-spectral-weight",
        type=float,
        default=0.5,
        help="Weight for spectral component in combined loss (default: 0.5). MSE weight = 1 - this value.",
    )
    parser.add_argument(
        "--normalize-combined-losses",
        action="store_true",
        default=True,
        help="Normalize spectral and MSE losses to similar scales before combining (default: True).",
    )
    parser.add_argument(
        "--no-normalize-combined-losses",
        dest="normalize_combined_losses",
        action="store_false",
        help="Disable normalization of losses before combining in combined loss",
    )
    parser.add_argument(
        "--debug-combined-loss",
        action="store_true",
        help="Print debug information for combined loss (shows raw and normalized values at each epoch)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fnnoperator/outputs"),
        help="Base directory for all outputs (checkpoints, logs, etc.)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this training run (default: auto-generated with timestamp)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use (cuda or cpu). If not specified, auto-detect (prefers GPU if available).",
    )
    
    args = parser.parse_args()
    
    # Resolve all paths relative to project root
    if not args.data_dir.is_absolute():
        args.data_dir = (project_root / args.data_dir).resolve()
    else:
        args.data_dir = Path(args.data_dir).resolve()
    
    if not args.output_dir.is_absolute():
        args.output_dir = (project_root / args.output_dir).resolve()
    else:
        args.output_dir = Path(args.output_dir).resolve()
    
    # Load data first to determine dimension
    print("Loading data...")
    X_train, Y_train_final, dim, grid_shape, all_states_train, n_time_steps = load_heat_data(args.data_dir, split="train")
    X_val, Y_val_final, _, _, all_states_val, _ = load_heat_data(args.data_dir, split="validate")
    X_test, Y_test_final, _, _, all_states_test, _ = load_heat_data(args.data_dir, split="test")
    
    # Override dimension if specified
    if args.dimension is not None:
        dim = args.dimension
    
    # Determine if we're predicting all time steps
    use_all_time_steps = args.predict_all_time_steps
    if use_all_time_steps:
        print(f"\nMode: Predicting all {n_time_steps} time steps")
        Y_train = all_states_train
        Y_val = all_states_val
        Y_test = all_states_test
    else:
        print(f"\nMode: Predicting final state only")
        Y_train = Y_train_final
        Y_val = Y_val_final
        Y_test = Y_test_final
    
    # Create organized output directory structure
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"fno_heat_eq_{dim}d_{timestamp}"
    
    # Create subdirectories
    base_output_dir = args.output_dir / args.run_name
    checkpoints_dir = base_output_dir / "checkpoints"
    logs_dir = base_output_dir / "logs"
    predictions_dir = base_output_dir / "predictions"
    
    # Create all directories
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directories:")
    print(f"  Base: {base_output_dir}")
    print(f"  Checkpoints: {checkpoints_dir}")
    print(f"  Logs: {logs_dir}")
    print(f"  Predictions: {predictions_dir}")
    
    print(f"Data loaded:")
    print(f"  Dimension: {dim}D")
    print(f"  Grid shape: {grid_shape}")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Input shape: {X_train.shape}")
    print(f"  Output shape: {Y_train.shape}")
    
    # Prepare data for FNO (add channel dimension)
    X_train, Y_train = prepare_data_for_fno(X_train, Y_train, dim, use_all_time_steps=use_all_time_steps)
    X_val, Y_val = prepare_data_for_fno(X_val, Y_val, dim, use_all_time_steps=use_all_time_steps)
    X_test, Y_test = prepare_data_for_fno(X_test, Y_test, dim, use_all_time_steps=use_all_time_steps)
    
    print(f"\nAfter adding channel dimension:")
    print(f"  Input shape: {X_train.shape}")
    print(f"  Output shape: {Y_train.shape}")
    
    # Determine n_modes for FNO
    if args.n_modes is None:
        # Auto-determine: use more modes to capture high-frequency details
        # For better accuracy, use about 1/2 of grid size (up to Nyquist limit)
        # This helps prevent over-smoothing/diffusion
        if dim == 1:
            n_modes = (min(16, grid_shape[0] // 2),)
        elif dim == 2:
            # For 32x32 grid, this gives (16, 16) which is much better than (8, 8)
            n_modes = (min(16, grid_shape[0] // 2), min(16, grid_shape[1] // 2))
        else:  # 3D
            n_modes = (min(8, grid_shape[0] // 2), min(8, grid_shape[1] // 2), min(8, grid_shape[2] // 2))
    else:
        n_modes = tuple(args.n_modes)
        if len(n_modes) != dim:
            raise ValueError(f"Number of n_modes ({len(n_modes)}) must match dimension ({dim})")
    
    # Create FNO model
    print(f"\nCreating FNO model...")
    print(f"  Grid shape: {grid_shape}")
    print(f"  In channels: {args.in_channels}")
    print(f"  Out channels: {args.out_channels}")
    print(f"  Hidden channels: {args.hidden_channels}")
    print(f"  N layers: {args.n_layers}")
    print(f"  N modes: {n_modes}")
    print(f"  Time steps: {n_time_steps if use_all_time_steps else 1}")
    
    # Note: FNO doesn't natively support multi-time-step prediction in the same way
    # For now, we'll train it to predict final state only, or we can modify the output
    if use_all_time_steps:
        print("  WARNING: FNO will predict final state only (multi-time-step not yet implemented for FNO)")
        use_all_time_steps = False
        Y_train = Y_train_final.unsqueeze(1)
        Y_val = Y_val_final.unsqueeze(1)
        Y_test = Y_test_final.unsqueeze(1)
    
    # Create FNO with domain padding for better boundary handling
    # Domain padding helps with periodic boundary conditions
    model = FNO(
        n_modes=n_modes,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        hidden_channels=args.hidden_channels,
        n_layers=args.n_layers,
        domain_padding=0.1,  # 10% padding to help with boundaries
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Determine device
    if args.device:
        device = torch.device(args.device)
        if args.device == "cuda" and not torch.cuda.is_available():
            print(f"⚠ Warning: CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create trainer
    print(f"\nCreating trainer...")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  [GPU] {torch.cuda.get_device_name(0)}")
        print(f"  [CUDA] Version: {torch.version.cuda}")
        print(f"  [Memory] {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"  [WARNING] Using CPU")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    if args.use_amp:
        print(f"  [AMP] Mixed Precision: Enabled")
    
    # Create loss function based on argument
    print(f"\nLoss function: {args.loss_function}")
    if args.loss_function == "mse":
        loss_fn = MSELoss()
        print(f"  Using: Mean Squared Error (MSE)")
    elif args.loss_function == "spectral":
        loss_fn = SpectralLoss(
            reduction='mean',
            normalize=args.spectral_normalize,
            weight_low_freq=args.spectral_weight_low,
            weight_high_freq=args.spectral_weight_high,
            frequency_threshold=args.spectral_freq_threshold,
        )
        print(f"  Using: Spectral Loss")
        print(f"    Normalize: {args.spectral_normalize}")
        print(f"    Low freq weight: {args.spectral_weight_low}")
        print(f"    High freq weight: {args.spectral_weight_high}")
        print(f"    Frequency threshold: {args.spectral_freq_threshold}")
    elif args.loss_function == "combined":
        # Create individual loss functions
        mse_loss = MSELoss()
        spectral_loss = SpectralLoss(
            reduction='mean',
            normalize=args.spectral_normalize,
            weight_low_freq=args.spectral_weight_low,
            weight_high_freq=args.spectral_weight_high,
            frequency_threshold=args.spectral_freq_threshold,
        )
        
        # Calculate weights
        mse_weight = 1.0 - args.combined_spectral_weight
        
        # Combine using the general CombinedLoss class
        loss_fn = CombinedLoss(
            losses=[mse_loss, spectral_loss],
            weights=[mse_weight, args.combined_spectral_weight],
            normalize_losses=args.normalize_combined_losses,
            debug=args.debug_combined_loss,
            loss_names=['MSE', 'Spectral'],
        )
        print(f"  Using: Combined Loss (MSE + Spectral)")
        print(f"    MSE weight: {mse_weight}")
        print(f"    Spectral weight: {args.combined_spectral_weight}")
        print(f"    Spectral normalize (internal): {args.spectral_normalize}")
        print(f"    Normalize losses before combining: {args.normalize_combined_losses}")
        if args.debug_combined_loss:
            print(f"    Debug mode: Enabled (will print loss components at each epoch)")
    
    trainer = FNNTrainer(
        model=model,
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        X_test=X_test,
        Y_test=Y_test,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_amp=args.use_amp,
        loss_fn=loss_fn,
    )
    
    # Training with fit() method
    print(f"\nStarting training for {args.n_epochs} epochs...")
    print("-" * 60)
    
    history = trainer.fit()
    
    print("-" * 60)
    
    # Find best validation loss from history
    val_losses = [v for v in history['val_loss'] if v is not None]
    if val_losses:
        best_val_loss = min(val_losses)
        best_epoch = val_losses.index(best_val_loss) + 1
        print(f"Training complete! Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
    else:
        best_val_loss = None
        print("Training complete! (No validation data provided)")
    
    # Compute RMSE and RSE from validation metrics
    epochs = list(range(1, len(history['train_loss']) + 1))
    val_mse_list = history.get('val_mse', [])
    val_spectral_list = history.get('val_spectral_error', [])
    
    # Compute RMSE from MSE
    val_rmse_list = []
    for mse in val_mse_list:
        if mse is not None:
            val_rmse_list.append(np.sqrt(mse))
        else:
            val_rmse_list.append(None)
    
    # Compute training metrics
    # Note: train_loss is the training loss (MSE, spectral, or combined depending on loss function)
    # We compute approximate training MSE and RMSE when using MSE loss
    train_mse_list = []
    train_rmse_list = []
    train_spectral_loss_list = []
    
    for train_loss_val in history['train_loss']:
        if args.loss_function == 'mse':
            # Training loss is MSE
            train_mse_list.append(train_loss_val)
            train_rmse_list.append(np.sqrt(train_loss_val))
            train_spectral_loss_list.append(None)
        elif args.loss_function == 'spectral':
            # Training loss is spectral loss
            train_mse_list.append(None)
            train_rmse_list.append(None)
            train_spectral_loss_list.append(train_loss_val)
        else:  # combined
            # Training loss is combined - we can't separate without tracking during training
            train_mse_list.append(None)
            train_rmse_list.append(None)
            train_spectral_loss_list.append(None)
    
    # Prepare training metrics for JSON
    training_metrics = {
        'epochs': epochs,
        'train_loss': history['train_loss'],
        'train_mse': train_mse_list,
        'train_rmse': train_rmse_list,
        'train_spectral_loss': train_spectral_loss_list,
        'val_loss': [v if v is not None else None for v in history['val_loss']],
        'val_mse': [v if v is not None else None for v in val_mse_list],
        'val_rmse': [v if v is not None else None for v in val_rmse_list],
        'val_spectral_error': [v if v is not None else None for v in val_spectral_list],
    }
    
    # Save training history to logs directory
    history_file = logs_dir / "training_history.json"
    with open(history_file, 'w') as f:
        json.dump(training_metrics, f, indent=2)
    print(f"\nTraining history saved to {history_file}")
    
    # Plot training curves
    print("\nGenerating training curves...")
    
    # Filter out None values for plotting
    epochs_mse = []
    val_mse_filtered = []
    epochs_rmse = []
    val_rmse_filtered = []
    epochs_rse = []
    val_spectral_filtered = []
    
    for i, mse in enumerate(val_mse_list):
        if mse is not None:
            epochs_mse.append(epochs[i])
            val_mse_filtered.append(mse)
    
    for i, rmse in enumerate(val_rmse_list):
        if rmse is not None:
            epochs_rmse.append(epochs[i])
            val_rmse_filtered.append(rmse)
    
    for i, rse in enumerate(val_spectral_list):
        if rse is not None:
            epochs_rse.append(epochs[i])
            val_spectral_filtered.append(rse)
    
    # Create figure with subplots for MSE, RMSE, Spectral Loss, and RSE
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot MSE
    ax1 = axes[0, 0]
    if epochs_mse:
        ax1.plot(epochs_mse, val_mse_filtered, 'b-', linewidth=2, label='Validation MSE', marker='o', markersize=4)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('MSE', fontsize=12)
        ax1.set_title('Mean Squared Error (MSE) vs Epoch', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        if epochs_mse:
            ax1.set_xlim(min(epochs_mse), max(epochs_mse))
    else:
        ax1.text(0.5, 0.5, 'No validation MSE data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Mean Squared Error (MSE) vs Epoch', fontsize=13, fontweight='bold')
    
    # Plot RMSE
    ax2 = axes[0, 1]
    if epochs_rmse:
        ax2.plot(epochs_rmse, val_rmse_filtered, 'g-', linewidth=2, label='Validation RMSE', marker='s', markersize=4)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('RMSE', fontsize=12)
        ax2.set_title('Root Mean Squared Error (RMSE) vs Epoch', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        if epochs_rmse:
            ax2.set_xlim(min(epochs_rmse), max(epochs_rmse))
    else:
        ax2.text(0.5, 0.5, 'No validation RMSE data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Root Mean Squared Error (RMSE) vs Epoch', fontsize=13, fontweight='bold')
    
    # Plot Spectral Loss
    ax3 = axes[1, 0]
    # Plot training spectral loss if available
    train_spectral_filtered = [v for v in train_spectral_loss_list if v is not None]
    epochs_train_spectral = [epochs[i] for i, v in enumerate(train_spectral_loss_list) if v is not None]
    
    has_data = False
    if epochs_train_spectral:
        ax3.plot(epochs_train_spectral, train_spectral_filtered, 'm-', linewidth=2, label='Training Spectral Loss', marker='^', markersize=4)
        has_data = True
    
    # Also plot validation RSE if available
    if epochs_rse:
        ax3.plot(epochs_rse, val_spectral_filtered, 'c--', linewidth=2, label='Validation RSE', marker='s', markersize=4, alpha=0.8)
        has_data = True
    
    if has_data:
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Spectral Loss / Error', fontsize=12)
        ax3.set_title('Spectral Loss vs Epoch', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=11)
        all_epochs_spectral = epochs_train_spectral + epochs_rse
        if all_epochs_spectral:
            ax3.set_xlim(min(all_epochs_spectral), max(all_epochs_spectral))
    else:
        ax3.text(0.5, 0.5, 'Spectral loss not tracked', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Spectral Loss vs Epoch', fontsize=13, fontweight='bold')
    
    # Plot RSE (Relative Spectral Error)
    ax4 = axes[1, 1]
    if epochs_rse:
        ax4.plot(epochs_rse, val_spectral_filtered, 'r-', linewidth=2, label='Validation RSE', marker='d', markersize=4)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Relative Spectral Error (RSE)', fontsize=12)
        ax4.set_title('Relative Spectral Error (RSE) vs Epoch', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=11)
        if epochs_rse:
            ax4.set_xlim(min(epochs_rse), max(epochs_rse))
    else:
        ax4.text(0.5, 0.5, 'No validation RSE data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Relative Spectral Error (RSE) vs Epoch', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = logs_dir / "training_curves.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {plot_file}")
    plt.close(fig)
    
    # Also create a combined plot with RMSE and RSE on the same axes
    if epochs_rmse or epochs_rse:
        fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        if epochs_rmse:
            ax.plot(epochs_rmse, val_rmse_filtered, 'g-', linewidth=2, label='Validation RMSE', marker='s', markersize=4)
        
        ax_twin = ax.twinx()
        if epochs_rse:
            ax_twin.plot(epochs_rse, val_spectral_filtered, 'r-', linewidth=2, label='Validation RSE', marker='d', markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12, color='g')
        ax_twin.set_ylabel('Relative Spectral Error (RSE)', fontsize=12, color='r')
        ax.set_title('Training Metrics: RMSE and RSE vs Epoch', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='y', labelcolor='g')
        ax_twin.tick_params(axis='y', labelcolor='r')
        
        all_epochs = epochs_rmse + epochs_rse
        if all_epochs:
            ax.set_xlim(min(all_epochs), max(all_epochs))
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        if lines1 or lines2:
            ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=11)
        
        plt.tight_layout()
        
        plot_file_combined = logs_dir / "training_curves_rmse_rse.png"
        plt.savefig(plot_file_combined, dpi=300, bbox_inches='tight')
        print(f"Combined RMSE/RSE curves saved to {plot_file_combined}")
        plt.close(fig2)
    
    # Save final model
    final_checkpoint_path = checkpoints_dir / "final_model.pt"
    checkpoint_data = {
        'epoch': args.n_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'train_loss': history['train_loss'][-1],
        'grid_shape': grid_shape,
        'in_channels': args.in_channels,
        'out_channels': args.out_channels,
        'hidden_channels': args.hidden_channels,
        'n_layers': args.n_layers,
        'n_modes': n_modes,
        'dimension': dim,
    }
    
    if best_val_loss is not None:
        checkpoint_data['val_loss'] = best_val_loss
    
    torch.save(checkpoint_data, final_checkpoint_path)
    print(f"\nFinal model saved to {final_checkpoint_path}")
    
    # Verify checkpoint was saved
    if final_checkpoint_path.exists():
        file_size_mb = final_checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"  Checkpoint verified: {file_size_mb:.2f} MB")
    else:
        print(f"  WARNING: Checkpoint file not found at {final_checkpoint_path}")
    
    # Test evaluation
    print("\nEvaluating on test set...")
    test_result = trainer.test()
    if test_result[0] is not None:
        test_loss, test_mse, test_spectral, test_preds = test_result
        print(f"Test Loss (training loss): {test_loss:.6f}")
        print(f"Test MSE: {test_mse:.6f}")
        test_rmse = np.sqrt(test_mse)
        print(f"Test RMSE: {test_rmse:.6f}")
        if test_spectral is not None:
            print(f"Test RSE: {test_spectral:.6f}")
        
        # Update checkpoint with test metrics
        checkpoint_data['test_loss'] = test_loss
        checkpoint_data['test_mse'] = test_mse
        checkpoint_data['test_rmse'] = float(test_rmse)
        if test_spectral is not None:
            checkpoint_data['test_rse'] = float(test_spectral)
        torch.save(checkpoint_data, final_checkpoint_path)
        
        # Save test predictions
        if test_preds is not None:
            test_predictions = np.concatenate(test_preds, axis=0)
            predictions_file = predictions_dir / "test_predictions.npy"
            np.save(predictions_file, test_predictions)
            print(f"Test predictions saved to {predictions_file}")
            
            # Save test metrics to logs
            test_metrics = {
                'test_loss': float(test_loss),
                'test_mse': float(test_mse),
                'test_rmse': float(test_rmse),
                'test_rse': float(test_spectral) if test_spectral is not None else None,
                'n_samples': len(test_predictions),
                'prediction_shape': list(test_predictions.shape)
            }
            test_metrics_file = logs_dir / "test_metrics.json"
            with open(test_metrics_file, 'w') as f:
                json.dump(test_metrics, f, indent=2)
            print(f"Test metrics saved to {test_metrics_file}")
    else:
        test_loss = None
        test_mse = None
        test_rmse = None
        test_spectral = None
        print("No test data provided.")
    
    # Save run configuration
    config_file = logs_dir / "run_config.json"
    run_config = {
        'run_name': args.run_name,
        'model_type': 'FNO',
        'dimension': dim,
        'grid_shape': list(grid_shape),
        'in_channels': args.in_channels,
        'out_channels': args.out_channels,
        'hidden_channels': args.hidden_channels,
        'n_layers': args.n_layers,
        'n_modes': list(n_modes),
        'n_epochs': args.n_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'loss_function': args.loss_function,
        'spectral_normalize': args.spectral_normalize if args.loss_function in ['spectral', 'combined'] else None,
        'spectral_weight_low': args.spectral_weight_low if args.loss_function in ['spectral', 'combined'] else None,
        'spectral_weight_high': args.spectral_weight_high if args.loss_function in ['spectral', 'combined'] else None,
        'combined_spectral_weight': args.combined_spectral_weight if args.loss_function == 'combined' else None,
        'best_val_loss': float(best_val_loss) if best_val_loss is not None else None,
        'test_loss': float(test_loss) if test_loss is not None else None,
        'test_mse': float(test_mse) if test_mse is not None else None,
        'test_rmse': float(test_rmse) if test_rmse is not None else None,
        'test_rse': float(test_spectral) if test_spectral is not None else None,
    }
    with open(config_file, 'w') as f:
        json.dump(run_config, f, indent=2)
    print(f"\nRun configuration saved to {config_file}")
    print(f"\nAll outputs saved to: {base_output_dir}")


if __name__ == "__main__":
    main()

