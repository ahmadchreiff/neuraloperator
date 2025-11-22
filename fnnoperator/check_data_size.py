"""Quick script to check heat equation data size."""
import numpy as np
from pathlib import Path

data_dir = Path("neuraloperator/heat_eq_data/data")
splits = ['train', 'validate', 'test']

print("=== HEAT EQUATION DATA SUMMARY ===\n")

total_samples = 0
for split in splits:
    data = np.load(data_dir / f"{split}.npz", allow_pickle=True)
    initial = data['initial']
    solution = data['solution']
    
    print(f"{split.upper()}:")
    print(f"  Samples: {initial.shape[0]:,}")
    print(f"  Initial shape: {initial.shape}")
    print(f"  Solution shape: {solution.shape}")
    print(f"  Grid size: {initial.shape[1:]}")
    print(f"  Time steps: {solution.shape[1]}")
    print()
    
    total_samples += initial.shape[0]

print(f"TOTAL SAMPLES: {total_samples:,}")
print(f"\nData Details:")
train_data = np.load(data_dir / "train.npz", allow_pickle=True)
print(f"  Dimension: 2D")
print(f"  Grid: {train_data['initial'].shape[1]} x {train_data['initial'].shape[2]} (ny x nx)")
print(f"  Time steps: {train_data['solution'].shape[1]}")
print(f"  Total grid points per sample: {train_data['initial'].shape[1] * train_data['initial'].shape[2]:,}")

# Calculate approximate file sizes
import os
print(f"\nFile Sizes:")
for split in splits:
    file_path = data_dir / f"{split}.npz"
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"  {split}.npz: {size_mb:.2f} MB")

