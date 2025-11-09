# Navier-Stokes Pipeline: Complete Guide

This document provides a comprehensive overview of everything related to Navier-Stokes in the NeuralOperator repository, including the complete pipeline from data loading to training and evaluation.

---

## Table of Contents

1. [Overview](#overview)
2. [Data Source and Format](#data-source-and-format)
3. [Data Loading Pipeline](#data-loading-pipeline)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Architecture](#model-architecture)
6. [Training Pipeline](#training-pipeline)
7. [Evaluation Pipeline](#evaluation-pipeline)
8. [Results Display](#results-display)
9. [Configuration](#configuration)
10. [Key Files Reference](#key-files-reference)

---

## Overview

### What is Navier-Stokes?

The Navier-Stokes equations describe fluid flow (liquids and gases). In this repository:

- **Problem Type**: 2D incompressible Navier-Stokes equations
- **Data Format**: Vorticity fields (single channel)
- **Task**: Learn the operator that maps initial vorticity conditions to future vorticity states
- **Model**: Fourier Neural Operator (FNO) - specifically `FNO_Medium2d`

### Current State

- ✅ Dataset class exists (`NavierStokesDataset`)
- ✅ Training script exists (`train_navier_stokes.py`)
- ✅ Configuration exists (`navier_stokes_config.py`)
- ✅ Data available on Zenodo
- ❌ **No physics-informed loss** (only data losses: L2, H1)

---

## Data Source and Format

### Data Location

- **Source**: Zenodo archive
- **Record ID**: `12825163`
- **URL**: https://zenodo.org/records/12825163
- **Download**: Automatic via `download_from_zenodo_record()`

### Available Resolutions

- **128×128**: Standard resolution for training
- **1024×1024**: High-resolution data (for testing or fine-tuning)

### Data Format

The data is stored as PyTorch `.pt` files:

**File Structure:**
```
nsforcing_train_128.pt
nsforcing_test_128.pt
nsforcing_train_1024.pt
nsforcing_test_1024.pt
```

**Data Dictionary Structure:**
```python
{
    "x": torch.Tensor,  # Input vorticity fields
    "y": torch.Tensor   # Output vorticity fields
}
```

**Tensor Shapes:**
- `x`: `(n_samples, height, width)` - Initial vorticity
- `y`: `(n_samples, height, width)` - Future vorticity
- After loading: `(n_samples, 1, height, width)` - Channel dimension added

**Physical Meaning:**
- **Input (`x`)**: Vorticity field at initial time
- **Output (`y`)**: Vorticity field at a future time
- **Single Channel**: Only vorticity (not velocity components or pressure)

---

## Data Loading Pipeline

### Step 1: Dataset Initialization

**File**: `neuralop/data/datasets/navier_stokes.py`

```python
from neuralop.data.datasets.navier_stokes import load_navier_stokes_pt

train_loader, test_loaders, data_processor = load_navier_stokes_pt(
    data_root="~/data/navier_stokes/",
    train_resolution=128,
    n_train=10000,
    batch_size=8,
    test_resolutions=[128],
    n_tests=[2000],
    test_batch_sizes=[8],
    encode_input=True,
    encode_output=True,
)
```

### Step 2: What Happens During Loading

1. **Check for Data Files**
   - Looks for `.pt` files in `data_root`
   - If missing, downloads from Zenodo automatically

2. **Load Training Data**
   ```python
   data = torch.load("nsforcing_train_128.pt")
   x_train = data["x"]  # Shape: (n_total, H, W)
   y_train = data["y"]  # Shape: (n_total, H, W)
   ```

3. **Add Channel Dimension**
   ```python
   x_train = x_train.unsqueeze(1)  # (n_total, 1, H, W)
   y_train = y_train.unsqueeze(1)  # (n_total, 1, H, W)
   ```

4. **Subsample to Requested Size**
   - Takes first `n_train` samples from loaded data
   - Supports spatial subsampling if specified

5. **Create TensorDataset**
   ```python
   train_db = TensorDataset(x_train, y_train)
   # Returns dict: {"x": tensor, "y": tensor}
   ```

6. **Create Test Datasets**
   - One dataset per resolution in `test_resolutions`
   - Each with `n_tests[i]` samples

7. **Create DataLoaders**
   ```python
   train_loader = DataLoader(train_db, batch_size=8, ...)
   test_loaders = {
      128: DataLoader(test_db_128, batch_size=8, ...)
   }
   ```

### Step 3: Data Structure After Loading

**Training Batch:**
```python
sample = {
    "x": torch.Tensor,  # Shape: (batch_size, 1, 128, 128)
    "y": torch.Tensor  # Shape: (batch_size, 1, 128, 128)
}
```

---

## Data Preprocessing

### DataProcessor Overview

**File**: `neuralop/data/transforms/data_processors.py`

The `DataProcessor` handles normalization and preprocessing:

### Step 1: Normalizer Fitting (During Dataset Init)

**File**: `neuralop/data/datasets/pt_dataset.py`

```python
# Fit normalizers on training data
if encode_input:
    input_encoder = UnitGaussianNormalizer(dim=[0, 2, 3])  # Channel-wise
    input_encoder.fit(x_train)  # Computes mean/std per channel

if encode_output:
    output_encoder = UnitGaussianNormalizer(dim=[0, 2, 3])  # Channel-wise
    output_encoder.fit(y_train)  # Computes mean/std per channel
```

**Normalization Type**: `channel-wise`
- Computes mean and std across batch and spatial dimensions
- Preserves statistics per channel
- Formula: `(x - mean) / std`

### Step 2: Preprocessing (During Training)

**Method**: `data_processor.preprocess(data_dict)`

```python
# In training loop
sample = data_processor.preprocess(sample)

# What happens:
# 1. Move to device
x = sample["x"].to(device)
y = sample["y"].to(device)

# 2. Normalize inputs (if encode_input=True)
if in_normalizer:
    x = in_normalizer.transform(x)  # (x - mean) / std

# 3. Normalize outputs (if encode_output=True AND training mode)
if out_normalizer and training:
    y = out_normalizer.transform(y)  # (y - mean) / std
```

**Key Point**: Output normalization only happens during training. During evaluation, outputs are kept unnormalized for proper loss computation.

### Step 3: Postprocessing (After Model Prediction)

**Method**: `data_processor.postprocess(output, data_dict)`

```python
# After model forward pass
pred = model(x)  # Model outputs normalized predictions
pred, sample = data_processor.postprocess(pred, sample)

# What happens:
# 1. Denormalize predictions (if in eval mode)
if out_normalizer and not training:
    pred = out_normalizer.inverse_transform(pred)
    y = out_normalizer.inverse_transform(sample["y"])
```

**Why**: 
- Training loss computed on normalized data (better numerical stability)
- Evaluation loss computed on unnormalized data (physical units)

### Optional: Multi-Grid Patching

If `config.patching.levels > 0`, the `DataProcessor` is replaced with `MGPatchingDataProcessor`:

- Splits large inputs into patches
- Processes patches separately
- Stitches results back together
- Enables training on high-resolution data with limited memory

---

## Model Architecture

### Model Selection

**File**: `config/navier_stokes_config.py`

```python
model: ModelConfig = FNO_Medium2d()
```

### FNO_Medium2d Configuration

**File**: `config/models.py`

```python
class FNO_Medium2d(SimpleFNOConfig):
    data_channels: int = 1      # Input: 1 channel (vorticity)
    out_channels: int = 1        # Output: 1 channel (vorticity)
    n_modes: List[int] = [64, 64]  # Fourier modes per dimension
    hidden_channels: int = 64    # Hidden dimension
    projection_channel_ratio: int = 4
```

### FNO Architecture Overview

**File**: `neuralop/models/fno.py`

1. **Lifting Layer**
   - Projects input to hidden dimension
   - `(batch, 1, H, W) -> (batch, 64, H, W)`

2. **FNO Blocks** (4 layers by default)
   - Each block:
     - Spectral Convolution (Fourier domain)
     - Channel MLP (pointwise)
     - Skip connection
     - Activation (GELU)

3. **Projection Layer**
   - Projects back to output channels
   - `(batch, 64, H, W) -> (batch, 1, H, W)`

### Key Features

- **Resolution Invariant**: Can handle different input resolutions
- **Fourier Modes**: `n_modes=[64, 64]` means 64 modes in each spatial dimension
- **Periodic Boundary**: Assumes periodic boundary conditions

---

## Training Pipeline

### Entry Point

**File**: `scripts/train_navier_stokes.py`

### Step 1: Configuration Loading

```python
from config.navier_stokes_config import Default
config = make_config_from_cli(Default)
config = config.to_dict()
```

**Default Configuration:**
```python
# Optimization
n_epochs: 600
learning_rate: 3e-4
training_loss: "h1"  # or "l2"
weight_decay: 1e-4
scheduler: "StepLR"
step_size: 100
gamma: 0.5

# Data
n_train: 10000
train_resolution: 128
batch_size: 8
encode_input: True
encode_output: True
```

### Step 2: Data Loading

```python
train_loader, test_loaders, data_processor = load_navier_stokes_pt(
    data_root=config.data.folder,
    train_resolution=config.data.train_resolution,
    n_train=config.data.n_train,
    batch_size=config.data.batch_size,
    test_resolutions=config.data.test_resolutions,
    n_tests=config.data.n_tests,
    test_batch_sizes=config.data.test_batch_sizes,
    encode_input=config.data.encode_input,
    encode_output=config.data.encode_output,
)
```

### Step 3: Model Initialization

```python
from neuralop import get_model

model = get_model(config)  # Creates FNO_Medium2d
model = model.to(device)
```

### Step 4: Optimizer and Scheduler

```python
from neuralop.training import AdamW

optimizer = AdamW(
    model.parameters(),
    lr=config.opt.learning_rate,  # 3e-4
    weight_decay=config.opt.weight_decay,  # 1e-4
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=config.opt.step_size,  # 100
    gamma=config.opt.gamma,  # 0.5
)
```

### Step 5: Loss Function Setup

```python
from neuralop import H1Loss, LpLoss

l2loss = LpLoss(d=2, p=2)  # L2 loss in 2D
h1loss = H1Loss(d=2)        # H1 Sobolev loss in 2D

if config.opt.training_loss == "l2":
    train_loss = l2loss
elif config.opt.training_loss == "h1":
    train_loss = h1loss

eval_losses = {"h1": h1loss, "l2": l2loss}
```

**Current Limitation**: Only data losses are used. No physics-informed loss exists.

### Step 6: Trainer Initialization

```python
from neuralop import Trainer

trainer = Trainer(
    model=model,
    n_epochs=config.opt.n_epochs,
    data_processor=data_processor,
    device=device,
    mixed_precision=config.opt.mixed_precision,
    eval_interval=config.opt.eval_interval,
    log_output=config.wandb.log_output,
    use_distributed=config.distributed.use_distributed,
    verbose=config.verbose,
    wandb_log=config.wandb.log,
)
```

### Step 7: Training Loop

**File**: `neuralop/training/trainer.py`

#### Training One Epoch

```python
def train_one_epoch(epoch, train_loader, training_loss):
    model.train()
    data_processor.train()
    
    for sample in train_loader:
        # 1. Preprocess
        sample = data_processor.preprocess(sample)
        
        # 2. Forward pass
        pred = model(sample["x"])
        
        # 3. Postprocess (for eval, not training)
        # pred, sample = data_processor.postprocess(pred, sample)
        
        # 4. Compute loss
        loss = training_loss(pred, sample["y"])
        
        # 5. Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # 6. Update scheduler
    scheduler.step()
```

**Key Points**:
- Loss computed on **normalized** data during training
- Model sees normalized inputs
- Model outputs normalized predictions

---

## Evaluation Pipeline

### When Evaluation Happens

- Every `eval_interval` epochs (default: 1)
- After training completes

### Evaluation Process

**File**: `neuralop/training/trainer.py`

```python
def evaluate_all(epoch, eval_losses, test_loaders):
    all_metrics = {}
    
    for loader_name, loader in test_loaders.items():
        loader_metrics = self.evaluate(
            eval_losses,
            loader,
            log_prefix=loader_name,
            mode="single_step",
        )
        all_metrics.update(**loader_metrics)
    
    return all_metrics
```

### Evaluation Steps

```python
def evaluate(loss_dict, data_loader):
    model.eval()
    data_processor.eval()
    
    errors = {f"{loss_name}": 0 for loss_name in loss_dict.keys()}
    
    with torch.no_grad():
        for sample in data_loader:
            # 1. Preprocess (no output normalization in eval mode)
            sample = data_processor.preprocess(sample)
            
            # 2. Forward pass
            pred = model(sample["x"])
            
            # 3. Postprocess (denormalize for proper loss computation)
            pred, sample = data_processor.postprocess(pred, sample)
            
            # 4. Compute all evaluation losses
            for loss_name, loss_fn in loss_dict.items():
                error = loss_fn(pred, sample["y"])
                errors[loss_name] += error.item()
    
    # Average over batches
    for key in errors:
        errors[key] /= len(data_loader)
    
    return errors
```

### Evaluation Metrics

**Default Metrics:**
- `h1`: H1 Sobolev norm (includes derivatives)
- `l2`: L2 norm (pointwise error)

**Metric Names:**
- Format: `{loader_name}_{metric_name}`
- Example: `128_h1`, `128_l2` (for resolution 128)

---

## Results Display

### Console Output

**During Training:**
```
##### CONFIG #####
{config details}

### MODEL ###
FNO(...)

### OPTIMIZER ###
AdamW(...)

### SCHEDULER ###
StepLR(...)

### LOSSES ###
 * Train: H1Loss(...)
 * Test: {'h1': H1Loss(...), 'l2': LpLoss(...)}

### Beginning Training...

Epoch 0: Train Loss: 0.1234, Time: 45.6s
Epoch 0: Test 128_h1: 0.0567, Test 128_l2: 0.0234
...
```

**Format:**
- Training loss printed every epoch
- Evaluation metrics printed every `eval_interval` epochs

### Weights & Biases (W&B) Logging

**If `config.wandb.log = True`:**

**Logged Metrics:**
```python
{
    "train_err": float,           # Training loss
    "avg_loss": float,            # Average training loss
    "epoch_train_time": float,    # Time per epoch
    "128_h1": float,              # H1 loss on resolution 128
    "128_l2": float,              # L2 loss on resolution 128
    "n_params": int,              # Model parameter count
    "learning_rate": float,       # Current learning rate
}
```

**Visualizations:**
- Loss curves (training vs evaluation)
- Learning rate schedule
- Model parameters (if `wandb.watch(model)` enabled)
- Output images (if `log_output=True`)

### Logging Functions

**File**: `neuralop/training/trainer.py`

```python
def log_training(epoch, time, avg_loss, train_err, lr):
    # Prints to console
    print(f"Epoch {epoch}: Train Loss: {train_err:.4f}, Time: {time:.2f}s")
    
    # Logs to W&B
    if wandb_log:
        wandb.log({
            "train_err": train_err,
            "avg_loss": avg_loss,
            "epoch_train_time": time,
            "learning_rate": lr,
        }, step=epoch)

def log_eval(epoch, eval_metrics):
    # Prints to console
    for metric_name, value in eval_metrics.items():
        print(f"Epoch {epoch}: Test {metric_name}: {value:.4f}")
    
    # Logs to W&B
    if wandb_log:
        wandb.log(eval_metrics, step=epoch)
```

---

## Configuration

### Configuration File Structure

**File**: `config/navier_stokes_config.py`

```python
class NavierStokesOptConfig(OptimizationConfig):
    n_epochs: int = 600
    learning_rate: float = 3e-4
    training_loss: str = "h1"  # Options: "l2", "h1"
    weight_decay: float = 1e-4
    scheduler: str = "StepLR"  # Options: "StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"
    step_size: int = 100
    gamma: float = 0.5

class NavierStokesDatasetConfig(ConfigBase):
    folder: str = "~/data/navier_stokes/"
    batch_size: int = 8
    n_train: int = 10000
    train_resolution: int = 128
    n_tests: List[int] = [2000]
    test_resolutions: List[int] = [128]
    test_batch_sizes: List[int] = [8]
    encode_input: bool = True
    encode_output: bool = True

class Default(ConfigBase):
    model: ModelConfig = FNO_Medium2d()
    opt: OptimizationConfig = NavierStokesOptConfig()
    data: NavierStokesDatasetConfig = NavierStokesDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig()
```

### Command-Line Overrides

```bash
python scripts/train_navier_stokes.py \
    --opt.n_epochs 1000 \
    --opt.learning_rate 1e-3 \
    --data.n_train 5000 \
    --wandb.log True \
    --wandb.project "my_project"
```

---

## Key Files Reference

### Core Files

| File | Purpose |
|------|---------|
| `neuralop/data/datasets/navier_stokes.py` | Dataset class and loading function |
| `neuralop/data/datasets/pt_dataset.py` | Base dataset class (handles loading, normalization) |
| `neuralop/data/transforms/data_processors.py` | Preprocessing and normalization |
| `neuralop/models/fno.py` | FNO model implementation |
| `neuralop/training/trainer.py` | Training and evaluation loops |
| `scripts/train_navier_stokes.py` | Main training script |
| `config/navier_stokes_config.py` | Configuration settings |

### Loss Files

| File | Purpose |
|------|---------|
| `neuralop/losses/data_losses.py` | L2Loss, H1Loss (currently used) |
| `neuralop/losses/equation_losses.py` | **Missing: NavierStokesEqnLoss** |

### Model Config Files

| File | Purpose |
|------|---------|
| `config/models.py` | FNO_Medium2d definition |

---

## Summary: Complete Flow

```
1. DATA SOURCE
   └─> Zenodo (record 12825163)
       └─> Downloads: nsforcing_train_128.pt, nsforcing_test_128.pt

2. DATA LOADING
   └─> NavierStokesDataset.__init__()
       └─> Loads .pt files
       └─> Creates TensorDataset (x, y pairs)
       └─> Fits UnitGaussianNormalizer on training data
       └─> Creates DataProcessor with normalizers
       └─> Returns: train_loader, test_loaders, data_processor

3. MODEL SETUP
   └─> get_model(config) → FNO_Medium2d
       └─> n_modes=[64, 64], hidden_channels=64
       └─> 1 input channel, 1 output channel

4. TRAINING LOOP (per epoch)
   └─> For each batch:
       ├─> data_processor.preprocess() → Normalize x, y
       ├─> model(x) → Predict normalized output
       ├─> loss(pred, y) → Compute loss on normalized data
       ├─> loss.backward() → Backprop
       └─> optimizer.step() → Update weights
   └─> scheduler.step() → Update learning rate

5. EVALUATION (every eval_interval epochs)
   └─> For each test loader:
       ├─> data_processor.preprocess() → Normalize x (not y)
       ├─> model(x) → Predict normalized output
       ├─> data_processor.postprocess() → Denormalize pred, y
       └─> Compute eval_losses → H1, L2 on unnormalized data

6. RESULTS
   └─> Console: Printed metrics
   └─> W&B: Logged metrics and visualizations
```

---

## Current Limitations & Opportunities

### What's Missing

1. **No Physics-Informed Loss**
   - Currently only uses data losses (L2, H1)
   - No enforcement of Navier-Stokes equations
   - This is the gap you're filling!

### What Exists

- ✅ Complete data pipeline
- ✅ Model architecture
- ✅ Training infrastructure
- ✅ Evaluation framework
- ✅ Logging and visualization

### Where Your Contribution Fits

**Target File**: `neuralop/losses/equation_losses.py`

**Goal**: Add `NavierStokesEqnLoss` class that:
- Computes Navier-Stokes equation residuals
- Can be combined with data losses
- Works with existing training pipeline

**Integration Point**: `scripts/train_navier_stokes.py` (line 155-167)

---

## Next Steps

1. Understand the Navier-Stokes equations (momentum + continuity)
2. Study existing `BurgersEqnLoss` as a template
3. Implement `NavierStokesEqnLoss` in `equation_losses.py`
4. Integrate into training script
5. Test and validate

---

*This document was created to provide a complete understanding of the Navier-Stokes pipeline before implementing physics-informed losses.*

