"""
Train FNNOperator on heat equation dataset.

This script trains an FNNOperator to learn the mapping from initial conditions
to the final state of the heat equation solution.

Usage:
    python fnnoperator/train_heat_operator.py --data-dir neuraloperator/heat_eq_data/data --dimension 1
    python fnnoperator/train_heat_operator.py --data-dir neuraloperator/heat_eq_data/data --dimension 2
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn

from fnnoperator.FNNOperator import FNNOperator
from fnnoperator.FNNTrainer import FNNTrainer


def load_heat_data(data_dir: Path, split: str = "train"):
    """Load heat equation data from NPZ file.
    
    Args:
        data_dir: Directory containing train.npz, test.npz, validate.npz
        split: Which split to load ('train', 'test', 'validate')
    
    Returns:
        tuple: (initial_conditions, final_states, dimension, grid_shape)
            - initial_conditions: (n_samples, *spatial_dims) tensor
            - final_states: (n_samples, *spatial_dims) tensor
            - dimension: int (1, 2, or 3)
            - grid_shape: tuple of spatial dimensions
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
    
    # Get all time steps (or just final state if n_time_steps=1)
    # solution_raw shape: (n_samples, nt, *spatial_dims)
    n_time_steps = solution_raw.shape[1]
    all_states = solution_raw  # (n_samples, nt, *spatial_dims)
    final_states = solution_raw[:, -1, ...]  # (n_samples, *spatial_dims) - keep for backward compatibility
    
    # Determine grid shape based on dimension
    # Note: grid_shape must be a tuple (or list) for proper unpacking in FNNOperator
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


def prepare_data_for_operator(initial_conditions, states, dimension, use_all_time_steps=False):
    """Prepare data for FNNOperator.
    
    FNNOperator expects input shape: (batch, in_channels, *grid)
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
        description="Train FNNOperator on heat equation dataset"
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
        "--width",
        type=int,
        default=256,
        help="Width of hidden layers (default: 256)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Depth of the network (default: 3)",
    )
    parser.add_argument(
        "--predict-all-time-steps",
        action="store_true",
        help="If set, predict all time steps instead of just final state",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
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
    # Convert relative paths to absolute paths relative to project root
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
    
    # Create organized output directory structure (after dimension is known)
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"heat_eq_{dim}d_{timestamp}"
    
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
    
    # Prepare data for operator (add channel dimension)
    X_train, Y_train = prepare_data_for_operator(X_train, Y_train, dim, use_all_time_steps=use_all_time_steps)
    X_val, Y_val = prepare_data_for_operator(X_val, Y_val, dim, use_all_time_steps=use_all_time_steps)
    X_test, Y_test = prepare_data_for_operator(X_test, Y_test, dim, use_all_time_steps=use_all_time_steps)
    
    print(f"\nAfter adding channel dimension:")
    print(f"  Input shape: {X_train.shape}")
    print(f"  Output shape: {Y_train.shape}")
    
    # Create model
    print(f"\nCreating FNNOperator model...")
    print(f"  Grid shape: {grid_shape}")
    print(f"  In channels: {args.in_channels}")
    print(f"  Out channels: {args.out_channels}")
    print(f"  Width: {args.width}")
    print(f"  Depth: {args.depth}")
    print(f"  Time steps: {n_time_steps if use_all_time_steps else 1}")
    
    model = FNNOperator(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        grid_shape=grid_shape,
        width=args.width,
        depth=args.depth,
        n_time_steps=n_time_steps if use_all_time_steps else 1,
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
    
    # Save training history to logs directory
    history_file = logs_dir / "training_history.json"
    # Convert None values to strings for JSON serialization
    history_json = {
        'train_loss': history['train_loss'],
        'val_loss': [v if v is not None else None for v in history['val_loss']]
    }
    with open(history_file, 'w') as f:
        json.dump(history_json, f, indent=2)
    print(f"\nTraining history saved to {history_file}")
    
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
        'width': args.width,
        'depth': args.depth,
        'n_time_steps': n_time_steps if use_all_time_steps else 1,
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
    
    # Test evaluation (separate from training, as per ML best practices)
    print("\nEvaluating on test set...")
    test_loss, test_preds = trainer.test()
    if test_loss is not None:
        print(f"Test loss: {test_loss:.6f}")
        
        # Update checkpoint with test loss
        checkpoint_data['test_loss'] = test_loss
        torch.save(checkpoint_data, final_checkpoint_path)
        
        # Save test predictions
        if test_preds is not None:
            # Concatenate all batch predictions
            test_predictions = np.concatenate(test_preds, axis=0)
            predictions_file = predictions_dir / "test_predictions.npy"
            np.save(predictions_file, test_predictions)
            print(f"Test predictions saved to {predictions_file}")
            
            # Save test metrics to logs
            test_metrics = {
                'test_loss': float(test_loss),
                'n_samples': len(test_predictions),
                'prediction_shape': list(test_predictions.shape)
            }
            test_metrics_file = logs_dir / "test_metrics.json"
            with open(test_metrics_file, 'w') as f:
                json.dump(test_metrics, f, indent=2)
            print(f"Test metrics saved to {test_metrics_file}")
    else:
        print("No test data provided.")
    
    # Save run configuration
    config_file = logs_dir / "run_config.json"
    run_config = {
        'run_name': args.run_name,
        'dimension': dim,
        'grid_shape': list(grid_shape),
        'in_channels': args.in_channels,
        'out_channels': args.out_channels,
        'width': args.width,
        'depth': args.depth,
        'n_time_steps': n_time_steps if use_all_time_steps else 1,
        'predict_all_time_steps': use_all_time_steps,
        'n_epochs': args.n_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'best_val_loss': float(best_val_loss) if best_val_loss is not None else None,
        'test_loss': float(test_loss) if test_loss is not None else None,
    }
    with open(config_file, 'w') as f:
        json.dump(run_config, f, indent=2)
    print(f"\nRun configuration saved to {config_file}")
    print(f"\nAll outputs saved to: {base_output_dir}")


if __name__ == "__main__":
    main()

