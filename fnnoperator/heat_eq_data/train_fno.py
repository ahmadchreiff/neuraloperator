"""
Train FNO on heat equation dataset.

This script trains a Fourier Neural Operator (FNO) on the 1D heat equation dataset.
The model learns to map initial conditions to the final state of the solution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from neuralop.models import FNO
from neuralop import Trainer, LpLoss, H1Loss
from neuralop.training import AdamW
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.utils import count_model_params


class HeatEquationDataset(Dataset):
    """Dataset for heat equation data loaded from NPZ file.
    
    Supports 1D, 2D, and 3D spatial domains.
    
    Input: initial condition u(x, t=0) -> shape depends on dimension
    Output: final state u(x, t=T) -> shape depends on dimension
    """
    
    def __init__(
        self,
        npz_path: str | Path,
        split: str = "train",
        use_final_state: bool = True,
        dimension: int | None = None,
    ):
        """Initialize dataset.
        
        Args:
            npz_path: Path to directory containing train.npz, test.npz, validate.npz
            split: Which split to load ('train', 'test', 'validate')
            use_final_state: If True, predict final state. If False, predict full trajectory.
            dimension: Spatial dimension (1, 2, or 3). If None, auto-detect from data shape.
        """
        npz_path = Path(npz_path)
        file_path = npz_path / f"{split}.npz"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load data
        data = np.load(file_path, allow_pickle=True)
        self.initial_raw = torch.tensor(data["initial"], dtype=torch.float32)
        self.solution_raw = torch.tensor(data["solution"], dtype=torch.float32)
        self.t = torch.tensor(data["t"], dtype=torch.float32)  # (nt,)
        self.diffusivity = torch.tensor(data["diffusivity"], dtype=torch.float32)  # (nsamples,)
        
        # Detect dimension from data shape if not provided
        if dimension is None:
            # Check metadata first
            if "metadata" in data:
                metadata = data["metadata"]
                for item in metadata:
                    if isinstance(item, tuple) and len(item) == 2 and item[0] == "dimension":
                        dimension = int(item[1])
                        break
            
            # If not in metadata, infer from initial condition shape
            if dimension is None:
                # initial shape: (nsamples, ...spatial_dims)
                spatial_dims = len(self.initial_raw.shape) - 1  # Subtract batch dimension
                dimension = spatial_dims
        
        self.dimension = dimension
        self.use_final_state = use_final_state
        
        # Load spatial coordinates based on dimension
        if self.dimension == 1:
            self.x = torch.tensor(data["x"], dtype=torch.float32)  # (nx,)
            self.y = None
            self.z = None
        elif self.dimension == 2:
            self.x = torch.tensor(data["x"], dtype=torch.float32)  # (nx,)
            self.y = torch.tensor(data["y"], dtype=torch.float32)  # (ny,)
            self.z = None
        elif self.dimension == 3:
            self.x = torch.tensor(data["x"], dtype=torch.float32)  # (nx,)
            self.y = torch.tensor(data["y"], dtype=torch.float32)  # (ny,)
            self.z = torch.tensor(data["z"], dtype=torch.float32)  # (nz,)
        else:
            raise ValueError(f"Unsupported dimension: {self.dimension}")
        
        # Process initial conditions: add channel dimension for FNO
        # FNO expects: (batch, channels, *spatial_dims)
        if self.dimension == 1:
            # initial: (nsamples, nx) -> (nsamples, 1, nx)
            if self.initial_raw.ndim == 2:
                self.initial = self.initial_raw.unsqueeze(1)  # (nsamples, 1, nx)
            else:
                self.initial = self.initial_raw
        elif self.dimension == 2:
            # initial: (nsamples, ny, nx) -> (nsamples, 1, ny, nx)
            if self.initial_raw.ndim == 3:
                self.initial = self.initial_raw.unsqueeze(1)  # (nsamples, 1, ny, nx)
            else:
                self.initial = self.initial_raw
        else:  # 3D
            # initial: (nsamples, nz, ny, nx) -> (nsamples, 1, nz, ny, nx)
            if self.initial_raw.ndim == 4:
                self.initial = self.initial_raw.unsqueeze(1)  # (nsamples, 1, nz, ny, nx)
            else:
                self.initial = self.initial_raw
        
        # Process solutions
        if use_final_state:
            # Extract final time step
            if self.dimension == 1:
                # solution: (nsamples, nt, nx) -> target: (nsamples, 1, nx)
                self.target = self.solution_raw[:, -1, :].unsqueeze(1)
            elif self.dimension == 2:
                # solution: (nsamples, nt, ny, nx) -> target: (nsamples, 1, ny, nx)
                self.target = self.solution_raw[:, -1, :, :].unsqueeze(1)
            else:  # 3D
                # solution: (nsamples, nt, nz, ny, nx) -> target: (nsamples, 1, nz, ny, nx)
                self.target = self.solution_raw[:, -1, :, :, :].unsqueeze(1)
        else:
            # Use full trajectory (not implemented for multi-dim, use final state)
            raise NotImplementedError("Full trajectory prediction not yet implemented for multi-dimensional data")
        
        print(f"Loaded {split} dataset ({self.dimension}D): {len(self.initial)} samples")
        print(f"  Initial shape: {self.initial.shape}")
        print(f"  Target shape: {self.target.shape}")
        if self.dimension == 1:
            print(f"  Spatial resolution: {self.x.shape[0]}")
        elif self.dimension == 2:
            print(f"  Spatial resolution: {self.y.shape[0]} x {self.x.shape[0]}")
        else:
            print(f"  Spatial resolution: {self.z.shape[0]} x {self.y.shape[0]} x {self.x.shape[0]}")
        print(f"  Time steps: {self.t.shape[0]}")
    
    def __len__(self):
        return len(self.initial)
    
    def __getitem__(self, idx):
        return {
            "x": self.initial[idx],  # (1, *spatial_dims)
            "y": self.target[idx],   # (1, *spatial_dims)
        }


def create_data_loaders(
    data_dir: str | Path,
    batch_size: int = 32,
    test_batch_size: int = 32,
    use_final_state: bool = True,
    dimension: int | None = None,
    num_workers: int = 0,
):
    """Create data loaders for train, test, and validation sets.
    
    Args:
        data_dir: Directory containing train.npz, test.npz, validate.npz
        batch_size: Batch size for training
        test_batch_size: Batch size for testing/validation
        use_final_state: Whether to predict final state (True) or full trajectory (False)
        dimension: Spatial dimension (1, 2, or 3). If None, auto-detect from data.
        num_workers: Number of data loading workers
        
    Returns:
        train_loader, test_loader, val_loader, data_processor, dimension
    """
    data_dir = Path(data_dir)
    
    # Create datasets (dimension will be auto-detected if not provided)
    train_dataset = HeatEquationDataset(
        data_dir, split="train", use_final_state=use_final_state, dimension=dimension
    )
    test_dataset = HeatEquationDataset(
        data_dir, split="test", use_final_state=use_final_state, dimension=train_dataset.dimension
    )
    val_dataset = HeatEquationDataset(
        data_dir, split="validate", use_final_state=use_final_state, dimension=train_dataset.dimension
    )
    
    dimension = train_dataset.dimension
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    # Create data processor with normalization
    # Normalize over batch and all spatial dimensions, preserving channel dimension
    # For d-dimensional data: reduce over batch (0) and all spatial dims (2, 3, ...)
    # Example shapes:
    #   1D: (batch, channels, nx) -> reduce over [0, 2]
    #   2D: (batch, channels, ny, nx) -> reduce over [0, 2, 3]
    #   3D: (batch, channels, nz, ny, nx) -> reduce over [0, 2, 3, 4]
    
    ndim = train_dataset.initial.ndim
    # Spatial dimensions are all dims except batch (0) and channel (1)
    spatial_dims = list(range(2, ndim))
    normalize_dims = [0] + spatial_dims  # Normalize over batch + all spatial dims
    
    normalizer_x = UnitGaussianNormalizer(dim=normalize_dims)
    normalizer_y = UnitGaussianNormalizer(dim=normalize_dims)
    
    # Fit normalizers on training data
    x_train_batch = train_dataset.initial
    y_train_batch = train_dataset.target
    normalizer_x.fit(x_train_batch)
    normalizer_y.fit(y_train_batch)
    
    data_processor = DefaultDataProcessor(
        in_normalizer=normalizer_x,
        out_normalizer=normalizer_y,
    )
    
    return train_loader, test_loader, val_loader, data_processor, dimension


def create_model(
    dimension: int = 1,
    n_modes: int = 16,
    hidden_channels: int = 64,
    n_layers: int = 4,
    device: str = "cpu",
):
    """Create FNO model for d-dimensional heat equation.
    
    Args:
        dimension: Spatial dimension (1, 2, or 3)
        n_modes: Number of Fourier modes to keep per dimension (should be < spatial_resolution/2)
        hidden_channels: Width of the FNO (number of channels)
        n_layers: Number of FNO layers
        device: Device to place model on
        
    Returns:
        FNO model
    """
    if dimension == 1:
        n_modes_tuple = (n_modes,)
    elif dimension == 2:
        n_modes_tuple = (n_modes, n_modes)  # (modes_y, modes_x)
    elif dimension == 3:
        n_modes_tuple = (n_modes, n_modes, n_modes)  # (modes_z, modes_y, modes_x)
    else:
        raise ValueError(f"Unsupported dimension: {dimension}. Must be 1, 2, or 3.")
    
    model = FNO(
        n_modes=n_modes_tuple,
        in_channels=1,        # Input: initial condition (1 channel)
        out_channels=1,       # Output: solution (1 channel)
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        projection_channel_ratio=2,
        lifting_channel_ratio=2,
    )
    
    model = model.to(device)
    return model


def train(
    data_dir: str | Path = "heat_eq_data/data",
    batch_size: int = 32,
    test_batch_size: int = 32,
    n_epochs: int = 50,
    dimension: int | None = None,
    n_modes: int = 16,
    hidden_channels: int = 64,
    n_layers: int = 4,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | None = None,
    use_final_state: bool = True,
    eval_interval: int = 5,
    save_dir: str | Path | None = None,
):
    """Train FNO on heat equation dataset.
    
    Args:
        data_dir: Directory containing train.npz, test.npz, validate.npz
        batch_size: Training batch size
        test_batch_size: Test/validation batch size
        n_epochs: Number of training epochs
        dimension: Spatial dimension (1, 2, or 3). If None, auto-detect from data.
        n_modes: Number of Fourier modes per dimension
        hidden_channels: Width of FNO
        n_layers: Number of FNO layers
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        device: Device to train on ('cpu', 'cuda', or None for auto-detect)
        use_final_state: Whether to predict final state (True) or full trajectory (False)
        eval_interval: Evaluate every N epochs
        save_dir: Directory to save checkpoints (None to skip saving)
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    print(f"Training configuration:")
    print(f"  Data directory: {data_dir}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Dimension: {dimension if dimension else 'auto-detect'}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Predicting: {'Final state' if use_final_state else 'Full trajectory'}")
    print()
    
    # Create data loaders (dimension will be auto-detected if not provided)
    print("Loading data...")
    train_loader, test_loader, val_loader, data_processor, detected_dimension = create_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        use_final_state=use_final_state,
        dimension=dimension,
    )
    
    # Use detected dimension if not explicitly provided
    if dimension is None:
        dimension = detected_dimension
    
    data_processor = data_processor.to(device)
    
    print(f"Detected dimension: {dimension}D")
    print()
    
    # Create model
    print("Creating model...")
    model = create_model(
        dimension=dimension,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        device=device,
    )
    
    n_params = count_model_params(model)
    print(f"Model: FNO-{dimension}D with n_modes={n_modes}, hidden_channels={hidden_channels}, n_layers={n_layers}")
    print(f"Model has {n_params:,} parameters")
    print()
    
    # Create optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # Create loss functions
    # d parameter in loss functions refers to spatial dimension (not including time or batch)
    l2_loss = LpLoss(d=dimension, p=2)
    h1_loss = H1Loss(d=dimension)
    
    train_loss = h1_loss
    eval_losses = {"h1": h1_loss, "l2": l2_loss}
    
    print("Loss functions:")
    print(f"  Training: {train_loss}")
    print(f"  Evaluation: {eval_losses}")
    print()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        n_epochs=n_epochs,
        device=device,
        data_processor=data_processor,
        wandb_log=False,  # Disable wandb for simplicity
        eval_interval=eval_interval,
        use_distributed=False,
        verbose=True,
    )
    
    # Create test loaders dictionary (format expected by Trainer)
    test_loaders = {"test": test_loader}
    
    # Prepare save directory if provided
    save_dir = Path(save_dir) if save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    print("Starting training...")
    print("=" * 80)
    train_kwargs = {
        "train_loader": train_loader,
        "test_loaders": test_loaders,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "regularizer": False,
        "training_loss": train_loss,
        "eval_losses": eval_losses,
    }
    
    # Add checkpointing if save_dir is provided
    if save_dir:
        train_kwargs["save_every"] = eval_interval  # Save every eval_interval epochs
        train_kwargs["save_dir"] = save_dir
    
    trainer.train(**train_kwargs)
    
    print("=" * 80)
    print("Training completed!")
    
    return model, trainer, data_processor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train FNO on heat equation dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("heat_eq_data/data"),
        help="Directory containing train.npz, test.npz, validate.npz",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=32,
        help="Test/validation batch size",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=50,
        help="Number of training epochs",
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
        help="Number of Fourier modes per dimension (should be < spatial_resolution/2)",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=64,
        help="Width of FNO (number of channels)",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=4,
        help="Number of FNO layers",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cpu', 'cuda', or None for auto-detect)",
    )
    parser.add_argument(
        "--use-full-trajectory",
        action="store_true",
        help="Predict full trajectory instead of final state",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5,
        help="Evaluate every N epochs",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory to save checkpoints (None to skip)",
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    model, trainer, data_processor = train(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        n_epochs=args.n_epochs,
        dimension=args.dimension,
        n_modes=args.n_modes,
        hidden_channels=args.hidden_channels,
        n_layers=args.n_layers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        use_final_state=not args.use_full_trajectory,
        eval_interval=args.eval_interval,
        save_dir=args.save_dir,
    )
    
    print("\nTraining finished!")
    print(f"Model saved in trainer state.")


if __name__ == "__main__":
    main()

