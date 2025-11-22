# NeuralOperator Repository Exploration Summary

## Overview
**NeuralOperator** is a comprehensive PyTorch library for learning neural operators - models that learn mappings between function spaces. Unlike regular neural networks that learn between finite-dimensional spaces, neural operators enable resolution-invariant learning for PDEs and other infinite-dimensional problems.

**Repository**: https://github.com/ahmadchreiff/neuraloperator
**License**: MIT
**Python Version**: >=3.9

---

## Core Concepts

### What are Neural Operators?
Neural operators are designed to learn mappings between **function spaces** rather than between finite-dimensional vectors. This makes them particularly powerful for:
- Solving Partial Differential Equations (PDEs)
- Physics-informed machine learning
- Scientific computing
- Resolution-invariant predictions

### Key Advantage: Resolution Invariance
A trained neural operator can make predictions at **any resolution** without retraining! This is called "zero-shot super-resolution" and is a fundamental property of neural operators.

---

## Repository Structure

```
neuraloperator/
├── neuralop/                  # Main library code
│   ├── models/               # Neural operator architectures
│   ├── layers/               # Building blocks
│   ├── data/                 # Datasets & transforms
│   ├── training/             # Training framework
│   ├── losses/               # Loss functions
│   └── mpu/                  # Multi-processing utilities
├── config/                   # Configuration files
├── scripts/                  # Training scripts
├── examples/                 # Tutorials & examples
└── doc/                      # Documentation
```

---

## Architecture Overview

### 1. Models (`neuralop/models/`)

#### Base Model Class
- **`BaseModel`**: Foundation for all models
  - Auto-registration system for models
  - Parameter saving/loading with version checking
  - Checkpoint management

#### Core Models

**FNO (Fourier Neural Operator)**
- **File**: `fno.py`
- **Key Idea**: Uses spectral (Fourier) convolutions to learn in frequency domain
- **Parameters**:
  - `n_modes`: Number of Fourier modes to keep (e.g., [32, 32] for 2D)
  - `hidden_channels`: Width of the network
  - `in_channels/out_channels`: Input/output dimensions
  - `n_layers`: Number of Fourier layers
- **Forward Pass**:
  1. **Lifting**: Project input to higher dimensional space
  2. **FNO Blocks**: Sequence of Fourier layers with spectral convolutions
  3. **Projection**: Map back to output space

**TFNO (Tensorized FNO)**
- **File**: `fno.py` (extends FNO)
- **Key Feature**: Tucker factorization for parameter efficiency
- **Default**: `factorization="Tucker"`, `rank=0.1` (10% of parameters)
- Can achieve 90% reduction in parameters!

**Other Models**:
- **UNO**: U-shaped architecture with multiple resolutions
- **SFNO**: Spherical FNO for spherical geometry
- **GINO**: Geometry-Informed NO for arbitrary meshes
- **FNOGNO**: Hybrid FNO + Graph Neural Operator
- **UQNO**: Uncertainty quantification variants
- **CODANO**: Computational graph-aware NO

### 2. Layers (`neuralop/layers/`)

#### Key Building Blocks

**`fno_block.py`**: FNOBlocks
- Implements sequence of Fourier layers
- Options: normalizations, skip connections, channel MLPs

**`spectral_convolution.py`**: SpectralConv
- **Core Innovation**: Frequency-domain convolutions
- Operations: FFT → Multiply by weights → IFFT
- Supports: Dense, Tucker, CP, TT factorizations
- Efficient einsum-based tensor contractions

**`channel_mlp.py`**: ChannelMLP
- Channel-wise mixing layer
- Configurable expansion, dropout, skip connections

**`padding.py`**: DomainPadding
- Periodic-like padding for better boundary handling

**`embeddings.py`**: GridEmbeddingND, GridEmbedding2D
- Positional encodings for spatial data

**`skip_connections.py`**
- Linear, identity, soft-gating options

**`normalization_layers.py`**
- AdaIN, InstanceNorm, BatchNorm, GroupNorm

**`gno_block.py`**: GNOBlock
- Graph Neural Operator for unstructured data

**`integral_transform.py`**
- Integral operators for function-to-function mappings

### 3. Data Loading (`neuralop/data/`)

#### Datasets (`datasets/`)
- **`darcy.py`**: 2D Darcy flow (porous media)
- **`burgers.py`**: 1D/2D Burgers equation
- **`navier_stokes.py`**: Navier-Stokes equations
- **`car_cfd_dataset.py`**: Car CFD simulations
- **`pt_dataset.py`**: PyTorch tensor datasets
- **`hdf5_dataset.py`**: HDF5 data loading
- **`zarr_dataset.py`**: Zarr array data

#### Transforms (`transforms/`)
- **`data_processors.py`**: Normalization, encoding
- **`patching_transforms.py`**: Multi-grid patching
- **`normalizers.py`**: Unit Gaussian, standard normalizers
- **`base_transforms.py`**: Base transform classes

### 4. Training Framework (`neuralop/training/`)

**`trainer.py`**: Trainer Class
- **Key Features**:
  - Distributed training support (DDP)
  - Mixed precision training (AMP)
  - Weights & Biases logging
  - Checkpointing & resuming
  - Autoregressive evaluation
  - Data processor integration
- **Training Methods**:
  - `train()`: Main training loop
  - `eval()`: Evaluation on test sets
  - `autoregressive_eval()`: Multi-step prediction

**`training_state.py`**: State Management
- Save/load training state (epoch, optimizer, scheduler)
- Checkpoint management

**`adamw.py`**: Custom AdamW optimizer
- Weight decay and momentum options

**`incremental.py`**: IncrementalFNOTrainer
- Progressive training strategies

**`torch_setup.py`**: Distributed Setup
- Multi-GPU initialization

### 5. Losses (`neuralop/losses/`)

- **`LpLoss`**: Lp norm losses (L1, L2, etc.)
- **`H1Loss`**: Sobolev H1 norm (includes gradients)
- **`BurgersEqnLoss`**: Physics-informed loss for Burgers
- **`ICLoss`**: Initial condition loss
- **`FourierDiff`**: Frequency-domain differentiation
- **`FiniteDiff`**: Finite difference approximations

---

## Configuration System

### Config Files (`config/`)

The library uses `zencfg` for configuration management.

**Example**: `darcy_config.py`, `burgers_config.py`

Each config defines:
- **Model**: Architecture choice and parameters
- **Data**: Dataset paths, batch sizes, resolutions
- **Optimization**: Learning rates, schedulers, epochs
- **Patching**: Multi-grid settings
- **Wandb**: Logging configuration
- **Distributed**: Multi-GPU settings

**Architecture Presets** in `config/models.py`:
- `FNO_Small2d`, `FNO_Medium2d`, `FNO_Large2d`, `FNO_Huge2d`
- `GINO_Small3d`, `FNOGNO_Small3d`
- Pre-configured for different scales

---

## Key Technologies

### Dependencies
- **PyTorch**: Deep learning framework
- **TensorLy**: Tensor decompositions (Tucker, CP, TT)
- **opt-einsum**: Optimized tensor contractions
- **h5py/zarr**: Large data handling
- **wandb**: Experiment tracking
- **ruamel-yaml**: YAML parsing

### Tensor Factorization
- **Tucker Decomposition**: Most common in TFNO
- **CP (Canonical Polyadic)**: Very low-rank option
- **TT (Tensor Train)**: Alternative low-rank format
- **Factorized vs Reconstructed**:
  - Factorized: Direct contraction (faster, memory-efficient)
  - Reconstructed: Build full tensor (slower, more memory)

---

## Workflow Example

### 1. Load Data
```python
from neuralop.data.datasets import load_darcy_flow_small

train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=1000,
    batch_size=64,
    test_resolutions=[16, 32],  # Multi-resolution testing
)
```

### 2. Create Model
```python
from neuralop.models import FNO

model = FNO(
    n_modes=(16, 16),      # Keep 16 modes in each dim
    hidden_channels=64,    # Width
    in_channels=1,         # Input channels
    out_channels=1,        # Output channels
)
```

### 3. Setup Training
```python
from neuralop import Trainer, LpLoss, H1Loss
from neuralop.training import AdamW

optimizer = AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

trainer = Trainer(
    model=model,
    n_epochs=100,
    device="cuda",
    data_processor=data_processor,
)
```

### 4. Train
```python
trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    training_loss=H1Loss(d=2),
    eval_losses={"l2": LpLoss(d=2, p=2)},
)
```

---

## Advanced Features

### Multi-Grid Patching
- **Purpose**: Handle high-resolution data efficiently
- **Method**: Divide domain into overlapping patches
- **Implementation**: `MGPatchingDataProcessor`
- **Config**: Set `patching.levels > 0`

### Distributed Training
- **DDP**: Distributed Data Parallel
- **Setup**: `neuralop.training.setup(config)`
- **Features**: Multi-node, multi-GPU

### Zero-Shot Super-Resolution
- Train at low resolution, test at high resolution
- Automatic with neural operators!
- Demonstrated in `examples/models/plot_FNO_darcy.py`

### Physics-Informed Training
- Combine data loss + PDE residual loss
- Example: `BurgersEqnLoss`, `ICLoss`

---

## Example Scripts

**Training Scripts** (`scripts/`):
- `train_darcy.py`: Darcy flow equation
- `train_burgers.py`: Burgers equation
- `train_navier_stokes.py`: Navier-Stokes
- `train_gino_carcfd.py`: GINO on CFD
- `train_poisson.py`: Poisson equation

**Examples** (`examples/`):
- `plot_FNO_darcy.py`: FNO tutorial
- `plot_SFNO_swe.py`: Spherical FNO
- `plot_UNO_darcy.py`: UNO architecture

---

## Key Research Papers

1. **FNO**: "Fourier Neural Operator for Parametric Partial Differential Equations" (Li et al., ICLR 2021)
2. **TFNO**: "Multi-Grid Tensorized Fourier Neural Operator" (Kossaifi et al., TMLR 2024)
3. **General**: "Neural Operator: Learning Maps Between Function Spaces" (Kovachki et al., 2021)
4. **Architecture**: "Principled Approaches for Extending Neural Architectures to Function Spaces" (Berner et al., 2025)

---

## Design Patterns

### 1. BaseModel Auto-Registration
Models register themselves via `__init_subclass__`

### 2. Config-Driven Training
YAML/pydantic configs → instantiate models, datasets, optimizers

### 3. Data Processor Pipeline
Preprocess → Model → Postprocess pattern

### 4. Factorized vs Dense
Switch between implementations via `implementation` parameter

### 5. Multi-Resolution Testing
Built-in support for testing at multiple resolutions simultaneously

---

## Next Steps for Deep Dive

1. **Start with FNO**: Read `neuralop/models/fno.py` thoroughly
2. **Understand SpectralConv**: Core frequency-domain operation in `neuralop/layers/spectral_convolution.py`
3. **Run Examples**: Execute `examples/models/plot_FNO_darcy.py`
4. **Custom Dataset**: Create your own using `neuralop/data/datasets/`
5. **Experiment**: Try different factorizations in TFNO

---

## Common Use Cases

1. **PDE Solving**: Learn solution operators for PDEs
2. **CFD**: Fluid dynamics simulations
3. **Material Science**: Property prediction from microstructure
4. **Climate**: Weather pattern modeling
5. **Medical Imaging**: Image-to-image translation

---

## Performance Considerations

- **Memory**: Use TFNO with low rank for large models
- **Batch Size**: Smaller batches needed for high resolutions
- **Mixed Precision**: Enable for faster training
- **Distributed**: Essential for large-scale experiments

---

## Conclusion

NeuralOperator is a mature, well-structured library for learning neural operators. Its modular design allows:
- Easy experimentation with different architectures
- Built-in support for multi-resolution, distributed training
- Efficient implementations via tensor factorization
- Resolution-invariant models for scientific computing

The codebase is clean, well-documented, and follows PyTorch best practices. It's production-ready for research and applications in scientific machine learning.

---

*Last Updated: Repository exploration completed*
*Explore more at: https://neuraloperator.github.io/dev/index.html*






