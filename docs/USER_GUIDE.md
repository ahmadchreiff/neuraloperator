# NeuralOperator User Guide

A comprehensive guide to using the NeuralOperator repository for training, testing, plotting, and working with neural operators.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Available Models](#available-models)
4. [Available Datasets](#available-datasets)
5. [Training Models](#training-models)
   - [Using Pre-configured Training Scripts](#using-pre-configured-training-scripts)
   - [Using Configuration Files](#using-configuration-files)
   - [Custom Training Scripts](#custom-training-scripts)
6. [Testing Models](#testing-models)
7. [Visualization and Plotting](#visualization-and-plotting)
8. [Loading and Using Pre-trained Models](#loading-and-using-pre-trained-models)
9. [Configuration System](#configuration-system)
10. [Advanced Features](#advanced-features)
   - [Distributed Training](#distributed-training)
   - [Multi-Grid Patching](#multi-grid-patching)
   - [Incremental Training](#incremental-training)
   - [Weights & Biases Integration](#weights--biases-integration)
11. [Examples and Tutorials](#examples-and-tutorials)

---

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/NeuralOperator/neuraloperator
cd neuraloperator

# Install in editable mode (recommended)
pip install -e .

# Install dependencies
pip install -r requirements.txt
```

### Optional Dependencies

Some features require additional packages:

- **torch_harmonics**: Required for SFNO (Spherical FNO) and LocalNO models
- **the_well**: Required for The Well datasets (MHD64, Active Matter)

---

## Quick Start

### Simple Model Creation

```python
from neuralop.models import FNO

# Create a 2D Fourier Neural Operator
model = FNO(
    n_modes=(32, 32),      # Number of Fourier modes per dimension
    hidden_channels=64,    # Hidden channel dimension
    in_channels=2,         # Input channels
    out_channels=1         # Output channels
)
```

### Training a Model (Minimal Example)

```python
from neuralop.models import FNO
from neuralop import Trainer, LpLoss, H1Loss
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.training import AdamW
import torch

# Load data
train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=1000,
    batch_size=64,
    test_resolutions=[16, 32],
    n_tests=[100, 50],
    test_batch_sizes=[32, 32],
)

# Create model
model = FNO(n_modes=(8, 8), in_channels=1, out_channels=1, hidden_channels=24)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Setup training
optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
train_loss = H1Loss(d=2)
eval_losses = {"h1": H1Loss(d=2), "l2": LpLoss(d=2, p=2)}

# Train
trainer = Trainer(
    model=model,
    n_epochs=15,
    device=device,
    data_processor=data_processor,
    wandb_log=False,
    eval_interval=5,
    verbose=True,
)

trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    training_loss=train_loss,
    eval_losses=eval_losses,
)
```

---

## Available Models

The repository provides several neural operator architectures:

### Core Models

- **FNO** (Fourier Neural Operator): The original Fourier-based neural operator
  - Supports 1D, 2D, and 3D domains
  - Resolution invariant
  - Fast training and inference

- **TFNO** (Tensorized FNO): Memory-efficient version with tensor factorization
  - Tucker or CP factorization
  - Significantly fewer parameters
  - Maintains performance with reduced memory

### Specialized Models

- **SFNO** (Spherical FNO): For spherical coordinate systems (requires `torch_harmonics`)
- **UNO** (U-shaped Neural Operator): U-Net inspired architecture
- **UQNO** (Uncertainty Quantification NO): Includes uncertainty estimation
- **FNOGNO** (FNO + Graph Neural Operator): Hybrid architecture
- **GINO** (Graph Neural Operator): For irregular/mesh-based domains
- **LocalNO**: Efficient local neural operator (requires `torch_harmonics`)
- **CODANO**: Continuous-discrete neural operator

### Model Usage

```python
from neuralop.models import FNO, TFNO, UNO, GINO

# Standard FNO
fno = FNO(n_modes=(32, 32), hidden_channels=64, in_channels=2, out_channels=1)

# Tensorized FNO with Tucker factorization
tfno = TFNO(
    n_modes=(32, 32),
    hidden_channels=64,
    in_channels=2,
    out_channels=1,
    factorization='tucker',
    rank=0.1  # 10% of original parameters
)

# U-shaped Neural Operator
uno = UNO(n_modes=(32, 32), hidden_channels=64, in_channels=2, out_channels=1)
```

---

## Available Datasets

### Built-in Datasets

1. **Darcy Flow** (`load_darcy_flow_small`)
   - 2D elliptic PDE benchmark
   - Small dataset for quick testing
   - Multiple resolutions available

2. **Burgers Equation** (`load_mini_burgers_1dtime`)
   - 1D time-dependent nonlinear PDE
   - Good for testing time-dependent operators

3. **Navier-Stokes** (`load_navier_stokes_pt`)
   - Fluid dynamics equations
   - Requires downloading from Zenodo

4. **Car CFD** (`load_mini_car`)
   - Computational fluid dynamics on car geometry
   - Mesh-based data

5. **Spherical SWE** (`load_spherical_swe`)
   - Shallow water equations on spherical domains
   - Requires `torch_harmonics`

### Dataset Usage

```python
from neuralop.data.datasets import (
    load_darcy_flow_small,
    load_mini_burgers_1dtime,
    load_navier_stokes_pt
)

# Darcy Flow
train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=1000,
    batch_size=64,
    test_resolutions=[16, 32, 64],  # Multiple resolutions for zero-shot super-resolution
    n_tests=[100, 50, 25],
    test_batch_sizes=[32, 16, 8],
)

# Burgers Equation
train_loader, test_loaders, data_processor = load_mini_burgers_1dtime(
    data_path="./data",
    n_train=800,
    batch_size=16,
    n_test=400,
    test_batch_size=16,
)
```

---

## Training Models

### Using Pre-configured Training Scripts

The repository includes ready-to-use training scripts in the `scripts/` directory:

```bash
# Train on Darcy Flow
python scripts/train_darcy.py

# Train on Burgers Equation
python scripts/train_burgers.py

# Train on Navier-Stokes
python scripts/train_navier_stokes.py

# Train with Physics-Informed Loss (PINO)
python scripts/train_burgers_pino.py

# Train with Uncertainty Quantification
python scripts/train_uqno_darcy.py
```

### Customizing Training via Command Line

All training scripts accept configuration overrides via command line:

```bash
# Override specific config parameters
python scripts/train_darcy.py \
    --opt.n_epochs 500 \
    --opt.learning_rate 1e-3 \
    --data.n_train 2000 \
    --data.batch_size 32 \
    --model.hidden_channels 128 \
    --wandb.log True \
    --wandb.project my_project
```

### Using Configuration Files

Configuration files are located in `config/` directory. Each problem has its own config:

- `darcy_config.py` - Darcy Flow configuration
- `burgers_config.py` - Burgers equation configuration
- `navier_stokes_config.py` - Navier-Stokes configuration
- `default_config.py` - Default/base configuration

#### Configuration Structure

Configurations are organized into sections:

```python
# Example: config/darcy_config.py structure
class Default(ConfigBase):
    arch: str = "fno"                    # Model architecture
    model: ModelConfig = FNO_Small2d()   # Model parameters
    opt: OptimizationConfig = ...         # Training parameters
    data: DatasetConfig = ...            # Dataset parameters
    patching: PatchingConfig = ...      # Multi-grid patching
    wandb: WandbConfig = ...            # Weights & Biases logging
    distributed: DistributedConfig = ... # Distributed training
```

#### Creating Custom Configurations

1. Copy an existing config file (e.g., `darcy_config.py`)
2. Modify the parameters you need
3. Use it in your training script:

```python
from config.my_custom_config import Default
from zencfg import make_config_from_cli

config = make_config_from_cli(Default)
config = config.to_dict()
```

### Custom Training Scripts

For custom training, follow this template:

```python
import torch
from neuralop import Trainer, get_model, LpLoss, H1Loss
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.training import setup, AdamW
from zencfg import make_config_from_cli
from config.darcy_config import Default

# Load configuration
config = make_config_from_cli(Default)
config = config.to_dict()

# Setup device and distributed training
device, is_logger = setup(config)

# Load data
train_loader, test_loaders, data_processor = load_darcy_flow_small(
    data_root=config.data.folder,
    n_train=config.data.n_train,
    batch_size=config.data.batch_size,
    test_resolutions=config.data.test_resolutions,
    n_tests=config.data.n_tests,
    test_batch_sizes=config.data.test_batch_sizes,
)

# Create model
model = get_model(config)
model = model.to(device)

# Setup optimizer and scheduler
optimizer = AdamW(
    model.parameters(),
    lr=config.opt.learning_rate,
    weight_decay=config.opt.weight_decay,
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
)

# Setup losses
train_loss = H1Loss(d=2) if config.opt.training_loss == "h1" else LpLoss(d=2, p=2)
eval_losses = {"h1": H1Loss(d=2), "l2": LpLoss(d=2, p=2)}

# Create trainer
trainer = Trainer(
    model=model,
    n_epochs=config.opt.n_epochs,
    device=device,
    data_processor=data_processor,
    wandb_log=config.wandb.log,
    eval_interval=config.opt.eval_interval,
    verbose=config.verbose,
)

# Train
trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    training_loss=train_loss,
    eval_losses=eval_losses,
)
```

---

## Testing Models

### Testing from Configuration

Use the test script to verify model functionality:

```bash
python scripts/test_from_config.py --config config/test_config.py
```

### Manual Testing

```python
import torch
from neuralop.models import FNO

# Create model
model = FNO(n_modes=(32, 32), in_channels=2, out_channels=1, hidden_channels=64)
model.eval()

# Create dummy input
batch_size = 4
input_data = torch.randn(batch_size, 2, 64, 64)

# Forward pass
with torch.no_grad():
    output = model(input_data)
    print(f"Output shape: {output.shape}")  # Should be [4, 1, 64, 64]
```

### Evaluating on Test Data

```python
from neuralop import LpLoss, H1Loss

# Load test data
_, test_loaders, data_processor = load_darcy_flow_small(...)

# Setup losses
l2_loss = LpLoss(d=2, p=2)
h1_loss = H1Loss(d=2)

# Evaluate
model.eval()
total_l2 = 0
total_h1 = 0
n_samples = 0

with torch.no_grad():
    for batch in test_loaders[16]:  # Test at resolution 16
        data = data_processor.preprocess(batch, batched=True)
        x, y = data["x"], data["y"]
        
        pred = model(x)
        total_l2 += l2_loss(pred, y).item()
        total_h1 += h1_loss(pred, y).item()
        n_samples += x.shape[0]

print(f"L2 Loss: {total_l2 / n_samples}")
print(f"H1 Loss: {total_h1 / n_samples}")
```

---

## Visualization and Plotting

### Example Plotting Scripts

The `examples/` directory contains numerous plotting examples:

#### Data Visualization

```bash
# Visualize Darcy Flow dataset
python examples/data/plot_darcy_flow.py

# Visualize data spectrum
python examples/data/plot_darcy_flow_spectrum.py

# Visualize Car CFD data
python examples/data/plot_mini_car_cfd.py
```

#### Model Predictions

```bash
# Visualize FNO predictions on Darcy Flow
python examples/models/plot_FNO_darcy.py

# Visualize UNO predictions
python examples/models/plot_UNO_darcy.py

# Visualize SFNO on shallow water equations
python examples/models/plot_SFNO_swe.py
```

#### Training Visualizations

```bash
# Checkpoint and resume training
python examples/training/checkpoint_FNO_darcy.py

# Incremental training visualization
python examples/training/plot_incremental_FNO_darcy.py

# Count FLOPs
python examples/training/plot_count_flops.py
```

### Custom Visualization

```python
import matplotlib.pyplot as plt
import torch
from neuralop.models import FNO
from neuralop.data.datasets import load_darcy_flow_small

# Load model and data
model = FNO(...)
model.load_state_dict(torch.load("checkpoint.pt"))
model.eval()

train_loader, test_loaders, data_processor = load_darcy_flow_small(...)

# Get a test sample
test_samples = test_loaders[16].dataset
data = test_samples[0]
data = data_processor.preprocess(data, batched=False)

x = data["x"]
y = data["y"]

# Make prediction
with torch.no_grad():
    pred = model(x.unsqueeze(0))

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(x[0].cpu().numpy(), cmap="gray")
axes[0].set_title("Input")
axes[0].axis("off")

axes[1].imshow(y.squeeze().cpu().numpy(), cmap="viridis")
axes[1].set_title("Ground Truth")
axes[1].axis("off")

axes[2].imshow(pred.squeeze().cpu().numpy(), cmap="viridis")
axes[2].set_title("Prediction")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("prediction.png")
plt.show()
```

### Zero-Shot Super-Resolution Visualization

One key feature of neural operators is resolution invariance. You can visualize this:

```python
# Train on low resolution (16x16)
train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=1000,
    batch_size=64,
    test_resolutions=[16, 32, 64],  # Test at multiple resolutions
    n_tests=[100, 50, 25],
    test_batch_sizes=[32, 16, 8],
)

# Model trained on 16x16 can predict at 32x32 and 64x64 without retraining!
for resolution in [16, 32, 64]:
    test_samples = test_loaders[resolution].dataset
    data = test_samples[0]
    data = data_processor.preprocess(data, batched=False)
    
    with torch.no_grad():
        pred = model(data["x"].unsqueeze(0))
    
    # Visualize predictions at different resolutions
    # ... plotting code ...
```

---

## Loading and Using Pre-trained Models

### Loading from Checkpoint

#### Method 1: Using `from_checkpoint` (Recommended)

```python
from neuralop.models import FNO

# Load model with saved initialization parameters
model = FNO.from_checkpoint(
    save_folder="./checkpoints",
    save_name="best_model",
    map_location="cpu"  # or "cuda:0" for GPU
)

model.eval()
```

#### Method 2: Manual Loading

```python
import torch
from neuralop.models import FNO

# Create model with same architecture
model = FNO(n_modes=(32, 32), in_channels=2, out_channels=1, hidden_channels=64)

# Load state dict
checkpoint = torch.load("checkpoint.pt", map_location="cpu")
model.load_state_dict(checkpoint)

model.eval()
```

#### Method 3: Resuming Training

```python
from neuralop import Trainer
from neuralop.training import load_training_state

# Create model, optimizer, scheduler
model = FNO(...)
optimizer = AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100)

# Load full training state
model, optimizer, scheduler, _, epoch = load_training_state(
    save_dir="./checkpoints",
    save_name="model",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
)

# Resume training
trainer = Trainer(model=model, n_epochs=100, ...)
trainer.train(
    ...,
    resume_from_dir="./checkpoints",  # Or use trainer.resume_state_from_dir()
)
```

### Saving Models

#### During Training

```python
trainer = Trainer(...)

# Save checkpoints periodically
trainer.train(
    ...,
    save_every=10,        # Save every 10 epochs
    save_dir="./checkpoints",  # Save directory
)

# Or save best model based on evaluation metric
trainer.train(
    ...,
    save_best="test_l2",  # Save model with best test_l2 metric
    save_dir="./checkpoints",
    eval_interval=5,      # Evaluate every 5 epochs
)
```

#### Manual Saving

```python
# Save model state only
model.save_checkpoint(save_folder="./checkpoints", save_name="my_model")

# Save full training state
from neuralop.training import save_training_state

save_training_state(
    save_dir="./checkpoints",
    save_name="model",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=current_epoch,
)
```

### Using Saved Models for Inference

```python
import torch
from neuralop.models import FNO
from neuralop.data.datasets import load_darcy_flow_small

# Load model
model = FNO.from_checkpoint("./checkpoints", "best_model")
model.eval()

# Load data
_, test_loaders, data_processor = load_darcy_flow_small(...)

# Run inference
with torch.no_grad():
    for batch in test_loaders[32]:
        data = data_processor.preprocess(batch, batched=True)
        x = data["x"]
        predictions = model(x)
        # Process predictions...
```

---

## Configuration System

### Configuration File Structure

Configuration files use the `zencfg` library and are organized hierarchically:

```python
from zencfg import ConfigBase
from .models import ModelConfig
from .opt import OptimizationConfig

class MyDatasetConfig(ConfigBase):
    folder: str = "~/data/my_dataset/"
    batch_size: int = 32
    n_train: int = 1000
    test_resolutions: List[int] = [64, 128]

class MyOptConfig(OptimizationConfig):
    n_epochs: int = 500
    learning_rate: float = 1e-3
    training_loss: str = "h1"

class Default(ConfigBase):
    arch: str = "fno"
    model: ModelConfig = FNO_Small2d()
    opt: MyOptConfig = MyOptConfig()
    data: MyDatasetConfig = MyDatasetConfig()
```

### Command-Line Overrides

Any configuration parameter can be overridden from the command line:

```bash
python train_script.py \
    --opt.n_epochs 1000 \
    --opt.learning_rate 5e-4 \
    --data.batch_size 64 \
    --model.hidden_channels 128 \
    --model.n_modes "[64, 64]"
```

### Common Configuration Parameters

#### Model Configuration (`model`)
- `model_arch`: Architecture type ("fno", "tfno", "uno", etc.)
- `n_modes`: Number of Fourier modes per dimension `[height, width]` or `[depth, height, width]`
- `hidden_channels`: Hidden channel dimension
- `n_layers`: Number of layers
- `factorization`: For TFNO ("tucker", "cp", or None)
- `rank`: Factorization rank (0.0 to 1.0)

#### Optimization Configuration (`opt`)
- `n_epochs`: Number of training epochs
- `learning_rate`: Learning rate
- `weight_decay`: Weight decay for regularization
- `training_loss`: Loss function ("l2", "h1", "equation", "ic", or combinations)
- `scheduler`: Learning rate scheduler ("StepLR", "CosineAnnealingLR", "ReduceLROnPlateau")
- `step_size`: For StepLR scheduler
- `gamma`: Learning rate decay factor
- `mixed_precision`: Use mixed precision training (bool)
- `eval_interval`: Epochs between evaluations

#### Data Configuration (`data`)
- `folder`: Path to data directory
- `batch_size`: Training batch size
- `n_train`: Number of training samples
- `train_resolution`: Training data resolution
- `test_resolutions`: List of test resolutions
- `n_tests`: Number of test samples per resolution
- `test_batch_sizes`: Batch sizes for each test resolution
- `encode_input`: Normalize inputs (bool)
- `encode_output`: Normalize outputs (bool)

#### Patching Configuration (`patching`)
- `levels`: Number of multi-grid patching levels (0 = disabled)
- `padding`: Padding fraction for patches
- `stitching`: Use patch stitching (bool)

#### WandB Configuration (`wandb`)
- `log`: Enable WandB logging (bool)
- `project`: WandB project name
- `entity`: WandB entity/username
- `group`: WandB run group
- `name`: Custom run name (optional)

---

## Advanced Features

### Distributed Training

Enable distributed training for multi-GPU setups:

```python
# In your training script
from neuralop.training import setup

config.distributed.use_distributed = True
config.distributed.model_parallel_size = 4  # Number of GPUs

device, is_logger = setup(config)

# Model will be automatically wrapped in DDP
model = get_model(config)
if config.distributed.use_distributed:
    from torch.nn.parallel import DistributedDataParallel as DDP
    model = DDP(model, device_ids=[device.index])
```

Run with `torchrun`:

```bash
torchrun --nproc_per_node=4 scripts/train_darcy.py
```

### Multi-Grid Patching

Multi-grid patching allows training on high-resolution data by splitting it into patches:

```python
from neuralop.data.transforms.data_processors import MGPatchingDataProcessor

# Enable patching in config
config.patching.levels = 2  # 2 levels of patching
config.patching.padding = 16  # Padding fraction
config.patching.stitching = True  # Stitch patches back together

# Data processor will be automatically converted to MGPatchingDataProcessor
# in the training script if patching.levels > 0
```

### Incremental Training

Incremental training gradually increases model capacity during training:

```python
from neuralop.training.incremental import IncrementalFNOTrainer
from neuralop.data.transforms.data_processors import IncrementalDataProcessor

# Create model with max modes
model = FNO(
    max_n_modes=(32, 32),  # Maximum capacity
    n_modes=(8, 8),        # Starting capacity
    ...
)

# Incremental data processor
data_processor = IncrementalDataProcessor(
    in_normalizer=...,
    out_normalizer=...,
    subsampling_rates=[2, 1],
    dataset_resolution=64,
    epoch_gap=10,
)

# Incremental trainer
trainer = IncrementalFNOTrainer(
    model=model,
    n_epochs=100,
    incremental_grad=True,  # Use gradient-based mode updates
    incremental_grad_eps=0.9999,
    ...
)
```

### Weights & Biases Integration

#### Setup

1. Create `config/wandb_api_key.txt` with your WandB API key
2. Enable logging in config:

```python
config.wandb.log = True
config.wandb.project = "my_project"
config.wandb.entity = "my_username"
config.wandb.group = "experiment_group"
```

#### Using WandB

```bash
# Training will automatically log to WandB
python scripts/train_darcy.py --wandb.log True --wandb.project my_project
```

The trainer automatically logs:
- Training and validation losses
- Learning rate
- Model parameters
- Evaluation metrics
- Output visualizations (if `log_output=True`)

### Physics-Informed Training (PINO)

Train with physics-informed losses:

```python
from neuralop import BurgersEqnLoss, ICLoss, WeightedSumLoss

# Physics-informed losses
equation_loss = BurgersEqnLoss(visc=0.01, method="fdm")
ic_loss = ICLoss()  # Initial condition loss
data_loss = LpLoss(d=2, p=2)

# Combine losses
training_loss = WeightedSumLoss(
    losses=[data_loss, equation_loss, ic_loss],
    weights=[1.0, 0.1, 0.1]  # Weight each loss
)

# Use in training
trainer.train(..., training_loss=training_loss)
```

---

## Examples and Tutorials

### Example Scripts Location

All examples are in the `examples/` directory:

- **`examples/data/`**: Data loading and visualization
- **`examples/models/`**: Model usage examples
- **`examples/training/`**: Training techniques
- **`examples/layers/`**: Layer and component examples

### Running Examples

```bash
# Data examples
cd examples/data
python plot_darcy_flow.py

# Model examples
cd examples/models
python plot_FNO_darcy.py

# Training examples
cd examples/training
python checkpoint_FNO_darcy.py
```

### Key Examples to Explore

1. **`examples/models/plot_FNO_darcy.py`**: Complete training workflow
2. **`examples/training/checkpoint_FNO_darcy.py`**: Checkpointing and resuming
3. **`examples/training/plot_incremental_FNO_darcy.py`**: Incremental training
4. **`examples/data/plot_darcy_flow.py`**: Understanding data structure

---

## Checkpoint Management

The Trainer supports several checkpointing options:

- **`save_every`**: Save checkpoint every N epochs
- **`save_best`**: Save model with best evaluation metric (e.g., `"test_l2"`, `"test_h1"`)
- **`save_dir`**: Directory to save checkpoints
- **`resume_from_dir`**: Resume training from saved checkpoint

Example:

```python
trainer.train(
    ...,
    save_every=10,              # Periodic saves every 10 epochs
    save_best="test_h1",        # Save best model based on test_h1 metric
    save_dir="./checkpoints",
    resume_from_dir=None,       # Or provide path to resume
)
```

When using `save_best`, the trainer will:
- Monitor the specified metric (e.g., `"test_h1"`)
- Save the model whenever it achieves the best value
- Save as `best_model_state_dict.pt` in the save directory
- Also save optimizer, scheduler, and epoch information

You can use both `save_every` and `save_best` together for comprehensive checkpointing.

---

## Tips and Best Practices

### Model Selection

- **FNO**: Good starting point, fast and effective for most problems
- **TFNO**: Use when memory is constrained or for very large models
- **UNO**: Good for problems requiring multi-scale features
- **GINO**: Use for irregular domains or mesh-based data

### Training Tips

1. **Start Small**: Begin with small models and datasets to verify your setup
2. **Resolution Invariance**: Test zero-shot super-resolution to verify model quality
3. **Loss Functions**: H1 loss often works better than L2 for PDE problems
4. **Learning Rate**: Start with 1e-3 and adjust based on convergence
5. **Batch Size**: Larger batches generally help, but adjust based on memory

### Debugging

1. **Check Data Shapes**: Ensure input/output shapes match model expectations
2. **Verify Normalization**: Check that data processor is working correctly
3. **Monitor Losses**: Watch for NaN values or exploding gradients
4. **Test on Small Data**: Use `n_train=10` to quickly test your pipeline

### Performance Optimization

1. **Mixed Precision**: Enable `mixed_precision=True` for faster training
2. **Multi-Grid Patching**: Use for high-resolution data that doesn't fit in memory
3. **Tensorization**: Use TFNO to reduce model size and memory usage
4. **Distributed Training**: Use multiple GPUs for faster training

---

## Getting Help

- **Documentation**: https://neuraloperator.github.io/dev/index.html
- **GitHub Issues**: https://github.com/NeuralOperator/neuraloperator/issues
- **Examples**: Check the `examples/` directory for working code
- **Config Files**: Study existing configs in `config/` for reference

---

## Summary

This guide covers the essential aspects of using NeuralOperator:

1. **Installation**: Set up the environment and dependencies
2. **Models**: Choose the right architecture for your problem
3. **Data**: Load and prepare datasets
4. **Training**: Use scripts, configs, or custom code
5. **Testing**: Evaluate model performance
6. **Visualization**: Plot results and predictions
7. **Checkpointing**: Save and load models
8. **Advanced Features**: Distributed training, patching, incremental learning

Start with the quick start examples, then explore the pre-configured training scripts, and finally customize for your specific needs. The repository is designed to be flexible while providing sensible defaults for common use cases.

