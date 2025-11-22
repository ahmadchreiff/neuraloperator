# FNO Lifting Code: Complete Deep Dive

## Overview

This document provides a **thorough, line-by-line explanation** of the code responsible for **Lifting** in FNO. We'll examine:
1. How lifting is initialized in the FNO constructor
2. The ChannelMLP class implementation
3. How the forward pass works through lifting
4. Why each design choice was made
5. Step-by-step examples with actual tensor shapes

---

## Table of Contents

1. [Lifting Initialization in FNO](#lifting-initialization-in-fno)
2. [ChannelMLP Class: Complete Breakdown](#channelmlp-class-complete-breakdown)
3. [Forward Pass Through Lifting](#forward-pass-through-lifting)
4. [Detailed Examples with Tensor Shapes](#detailed-examples-with-tensor-shapes)
5. [Why These Design Choices?](#why-these-design-choices)
6. [Complete Code Flow Diagram](#complete-code-flow-diagram)

---

## Lifting Initialization in FNO

### Location: `neuralop/models/fno.py`, Lines 301-329

Let's break down exactly how the lifting layer is created in the FNO constructor.

### Step 1: Calculate Input Channels

```python
lifting_in_channels = self.in_channels
if self.positional_embedding is not None:
    lifting_in_channels += self.n_dim
```

**What this does:**
- Starts with `self.in_channels` (the number of input feature channels)
- If positional embedding is enabled, adds `self.n_dim` (spatial dimensions) to account for coordinate channels

**Example:**
```python
# For a 2D problem:
self.in_channels = 1        # Temperature field
self.n_dim = 2              # 2D (x and y dimensions)
self.positional_embedding = GridEmbeddingND(...)  # Enabled

lifting_in_channels = 1 + 2 = 3
# Input will be: (temperature, x_coord, y_coord)
```

**Why?** The positional embedding adds coordinate channels. After positional embedding:
- Original: `(batch, 1, H, W)` - just temperature
- After embedding: `(batch, 3, H, W)` - temperature + x + y coordinates

### Step 2: Decide on Lifting Architecture

The code has **two paths** depending on whether `lifting_channels` is set:

#### Path A: Two-Layer MLP (Default)

```python
if self.lifting_channels:
    self.lifting = ChannelMLP(
        in_channels=lifting_in_channels,
        out_channels=self.hidden_channels,
        hidden_channels=self.lifting_channels,
        n_layers=2,
        n_dim=self.n_dim,
        non_linearity=non_linearity,
    )
```

**When this runs:** When `lifting_channel_ratio > 0` (default is 2)

**Architecture:**
```
Input (3 channels) 
  ↓
Layer 1: Conv1d(3 → 128) + GELU
  ↓
Layer 2: Conv1d(128 → 64)
  ↓
Output (64 channels)
```

**Parameters:**
- `in_channels`: 3 (temperature + x + y)
- `out_channels`: 64 (hidden_channels)
- `hidden_channels`: 128 (lifting_channels = 2 * 64)
- `n_layers`: 2

#### Path B: Single Linear Layer

```python
else:
    self.lifting = ChannelMLP(
        in_channels=lifting_in_channels,
        hidden_channels=self.hidden_channels,
        out_channels=self.hidden_channels,
        n_layers=1,
        n_dim=self.n_dim,
        non_linearity=non_linearity,
    )
```

**When this runs:** When `lifting_channel_ratio = 0` (rarely used)

**Architecture:**
```
Input (3 channels)
  ↓
Layer 1: Conv1d(3 → 64)
  ↓
Output (64 channels)
```

**No hidden layer, no nonlinearity!** Just a linear projection.

### Step 3: Handle Complex-Valued Data

```python
if self.complex_data:
    self.lifting = ComplexValued(self.lifting)
```

**What this does:** Wraps the ChannelMLP to handle complex numbers (for quantum mechanics, etc.)

**Implementation:** Creates separate real and imaginary parts, processes both, combines back.

---

## ChannelMLP Class: Complete Breakdown

### Location: `neuralop/layers/channel_mlp.py`

ChannelMLP is the **core building block** of lifting. Let's examine it line by line.

### Constructor Parameters

```python
def __init__(
    self,
    in_channels,              # Number of input channels
    out_channels=None,        # Number of output channels
    hidden_channels=None,     # Number of hidden channels
    n_layers=2,              # Number of layers
    n_dim=2,                 # Spatial dimension (for compatibility)
    non_linearity=F.gelu,    # Activation function
    dropout=0.0,             # Dropout probability
):
```

### Constructor Logic

#### Step 1: Store Parameters

```python
super().__init__()
self.n_layers = n_layers
self.in_channels = in_channels
self.out_channels = in_channels if out_channels is None else out_channels
self.hidden_channels = (
    in_channels if hidden_channels is None else hidden_channels
)
self.non_linearity = non_linearity
```

**Default behavior:**
- If `out_channels=None`: Output has same number of channels as input
- If `hidden_channels=None`: Hidden layer has same size as input

**For FNO lifting:**
```python
in_channels = 3
out_channels = 64
hidden_channels = 128
n_layers = 2
```

#### Step 2: Initialize Dropout (if needed)

```python
self.dropout = (
    nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
    if dropout > 0.0
    else None
)
```

**What this does:** Creates a dropout layer for each MLP layer (except when dropout=0)

**For FNO lifting:** Usually `dropout=0.0`, so `self.dropout = None`

#### Step 3: Build the Convolutional Layers

This is the **key insight**: Using `Conv1d(kernel_size=1)` instead of fully connected layers!

```python
self.fcs = nn.ModuleList()
for i in range(n_layers):
    if i == 0 and i == (n_layers - 1):
        # Single layer: input -> output
        self.fcs.append(nn.Conv1d(self.in_channels, self.out_channels, 1))
    elif i == 0:
        # First layer: input -> hidden
        self.fcs.append(nn.Conv1d(self.in_channels, self.hidden_channels, 1))
    elif i == (n_layers - 1):
        # Last layer: hidden -> output
        self.fcs.append(nn.Conv1d(self.hidden_channels, self.out_channels, 1))
    else:
        # Internal layers: hidden -> hidden
        self.fcs.append(
            nn.Conv1d(self.hidden_channels, self.hidden_channels, 1)
        )
```

**Why Conv1d with kernel_size=1?**
- **Resolution invariant**: Same operation at every spatial location
- **More efficient**: Better memory access patterns than FC layers
- **Easier to handle**: Works naturally with multi-dimensional tensors

**For FNO lifting (n_layers=2):**
- `i=0`: `Conv1d(3, 128, 1)` - First layer: 3 → 128
- `i=1`: `Conv1d(128, 64, 1)` - Last layer: 128 → 64

**Mathematical equivalence:**
```
Conv1d(in, out, kernel_size=1) applied to (B, in, H*W)
≡
Linear(in, out) applied at each spatial location
```

But Conv1d is more efficient and easier to work with!

---

## Forward Pass Through Lifting

### The Complete Forward Function

Let's trace through `ChannelMLP.forward()` step by step.

### Input Shape

For a 2D problem with 128×128 grid:
```python
x.shape = (batch_size, 3, 128, 128)
#          batch=32, channels=3 (temp+x+y), height=128, width=128
```

### Step 1: Check if Reshaping is Needed

```python
reshaped = False
size = list(x.shape)

if x.ndim > 3:
    x = x.reshape((*size[:2], -1))
    reshaped = True
```

**What happens:**
- `size = [32, 3, 128, 128]`
- `x.ndim = 4` (batch, channels, height, width)
- `size[:2] = [32, 3]` - keep batch and channels
- `-1` means "flatten all remaining dimensions"

**After reshape:**
```python
x.shape = (32, 3, 16384)  # 16384 = 128 * 128
```

**Visual representation:**
```
Before:
┌─────────────────────────────┐
│ (32, 3, 128, 128)          │
│ ┌──────────┬──────────┐    │
│ │ 128×128  │ 128×128  │    │ ← Each channel
│ │ 128×128  │          │    │
│ └──────────┴──────────┘    │
└─────────────────────────────┘

After reshape:
┌─────────────────────────────┐
│ (32, 3, 16384)             │
│ ┌──────────┬──────────┐    │
│ │ 16384    │ 16384    │    │ ← Flattened spatial dims
│ │ 16384    │          │    │
│ └──────────┴──────────┘    │
└─────────────────────────────┘
```

**Why reshape?** Conv1d expects 3D tensors: `(batch, channels, length)`. We flatten spatial dimensions into `length`.

### Step 2: Apply MLP Layers

```python
for i, fc in enumerate(self.fcs):
    x = fc(x)  # Linear transformation (1D conv with kernel size 1)
    if i < self.n_layers - 1:  # Apply nonlinearity to all layers except the last
        x = self.non_linearity(x)
    if self.dropout is not None:
        x = self.dropout[i](x)
```

Let's trace through **each iteration** for FNO lifting (n_layers=2):

#### Iteration 0: First Layer

**Before:**
```python
x.shape = (32, 3, 16384)
```

**Apply Conv1d:**
```python
fc = Conv1d(3, 128, kernel_size=1)
x = fc(x)
```

**Operation:**
```
For each of 16384 spatial positions:
  Take 3 input values (temp, x_coord, y_coord)
  Multiply by learned weight matrix W₁ (3×128)
  Output 128 values
```

**After Conv1d:**
```python
x.shape = (32, 128, 16384)
#          batch, channels, spatial_positions
```

**Visual:**
```
Before:                    After:
┌─────────┐               ┌──────────┐
│ (3, 1)  │  Conv1d(3→128)│ (128, 1) │
│ temp    │  ────────────→│ feature₁ │
│ x       │               │ feature₂ │
│ y       │               │ ...      │
└─────────┘               │ feature₁₂│
                          └──────────┘
(At each of 16384 positions)
```

**Apply Nonlinearity:**
```python
x = F.gelu(x)  # GELU activation
x.shape = (32, 128, 16384)  # Same shape
```

**Apply Dropout (if any):**
```python
if self.dropout:
    x = self.dropout[0](x)
# Usually skipped (dropout=0)
```

#### Iteration 1: Last Layer

**Before:**
```python
x.shape = (32, 128, 16384)
```

**Apply Conv1d:**
```python
fc = Conv1d(128, 64, kernel_size=1)
x = fc(x)
```

**Operation:**
```
For each of 16384 spatial positions:
  Take 128 input values (from previous layer)
  Multiply by learned weight matrix W₂ (128×64)
  Output 64 values
```

**After Conv1d:**
```python
x.shape = (32, 64, 16384)
```

**No nonlinearity:** `i == n_layers - 1`, so we skip the nonlinearity on the last layer.

**No dropout:** Usually `dropout=0`, but even if present, last layer often doesn't need it.

### Step 3: Reshape Back to Original Spatial Dimensions

```python
if reshaped:
    x = x.reshape((size[0], self.out_channels, *size[2:]))
```

**What happens:**
- `size[0] = 32` (batch size)
- `self.out_channels = 64`
- `size[2:] = [128, 128]` (original spatial dimensions)

**After reshape:**
```python
x.shape = (32, 64, 128, 128)
#          batch, channels, height, width
```

**Result:** We've lifted from 3 channels to 64 channels, preserving spatial structure!

---

## Detailed Examples with Tensor Shapes

Let's walk through a complete example with actual numbers.

### Example 1: Standard 2D FNO Lifting

**Configuration:**
```python
batch_size = 16
in_channels = 1
hidden_channels = 64
lifting_channel_ratio = 2
n_dim = 2
positional_embedding = "grid"
grid_size = (64, 64)  # 64×64 spatial grid
```

#### Step 1: Initial Input

```python
x = torch.randn(16, 1, 64, 64)
# Temperature field: 16 samples, 1 channel, 64×64 grid
```

#### Step 2: Positional Embedding (Before Lifting)

```python
# In FNO.forward(), before lifting:
if self.positional_embedding is not None:
    x = self.positional_embedding(x)

x.shape = (16, 3, 64, 64)
# Now has: temperature, x_coord, y_coord
```

#### Step 3: Enter ChannelMLP.forward()

**Input to ChannelMLP:**
```python
x.shape = (16, 3, 64, 64)
```

#### Step 4: Reshape for Conv1d

```python
x = x.reshape(16, 3, -1)  # Flatten spatial dimensions
x.shape = (16, 3, 4096)  # 4096 = 64 * 64
```

#### Step 5: First Conv1d Layer

```python
# Layer: Conv1d(3, 128, kernel_size=1)
x = self.fcs[0](x)
x.shape = (16, 128, 4096)
# At each of 4096 positions, we now have 128 features

x = F.gelu(x)  # Apply nonlinearity
x.shape = (16, 128, 4096)  # Same shape
```

#### Step 6: Second Conv1d Layer

```python
# Layer: Conv1d(128, 64, kernel_size=1)
x = self.fcs[1](x)
x.shape = (16, 64, 4096)
# Compressed from 128 to 64 channels
```

#### Step 7: Reshape Back

```python
x = x.reshape(16, 64, 64, 64)
x.shape = (16, 64, 64, 64)
# Back to spatial format, but now with 64 channels!
```

**Final Result:**
- Started with: `(16, 1, 64, 64)` - 1 channel (temperature)
- Ended with: `(16, 64, 64, 64)` - 64 channels (rich latent representation)

**Transformation:** 1 channel → 64 channels, resolution preserved!

---

### Example 2: 3D Problem

**Configuration:**
```python
batch_size = 8
in_channels = 2  # Pressure + velocity
hidden_channels = 96
lifting_channel_ratio = 2
n_dim = 3  # 3D problem
grid_size = (32, 32, 32)  # 32×32×32 grid
```

**Shapes:**
```python
# Input
x.shape = (8, 2, 32, 32, 32)

# After positional embedding (adds 3 coordinate channels)
x.shape = (8, 5, 32, 32, 32)  # 2 features + 3 coords

# Reshape for Conv1d
x.shape = (8, 5, 32768)  # 32768 = 32 * 32 * 32

# First layer: Conv1d(5, 192, 1)
x.shape = (8, 192, 32768)

# GELU
x.shape = (8, 192, 32768)

# Second layer: Conv1d(192, 96, 1)
x.shape = (8, 96, 32768)

# Reshape back
x.shape = (8, 96, 32, 32, 32)

# Result: 2 channels → 96 channels!
```

---

## Why These Design Choices?

### 1. Why Conv1d(kernel_size=1) Instead of FC Layers?

**FC Layer approach (alternative):**
```python
# Would need to:
x = x.reshape(batch, -1)  # Flatten everything
x = Linear(in_channels * H * W, out_channels * H * W)(x)
x = x.reshape(batch, out_channels, H, W)
```

**Problems:**
- **Not resolution invariant**: Weight matrix size depends on H×W
- **Memory inefficient**: Huge weight matrices (e.g., 4096×4096 for 64×64)
- **Can't generalize**: Can't handle different input sizes

**Conv1d(kernel_size=1) approach:**
```python
# Process each spatial location independently
# Weight matrix: (in_channels, out_channels) - same regardless of H×W
# Resolution invariant!
```

**Benefits:**
- ✅ Resolution invariant: Works at any spatial size
- ✅ Memory efficient: Small weight matrices
- ✅ Fast: Optimized convolution operations
- ✅ Clean code: No complex reshaping

### 2. Why Two Layers Instead of One?

**Single layer (linear):**
```python
Conv1d(3, 64, 1)
# Direct mapping: 3 → 64
# Limited expressiveness
```

**Two layers (with hidden):**
```python
Conv1d(3, 128, 1) → GELU → Conv1d(128, 64, 1)
# Expansive then compressive
# More expressive: can learn complex combinations
```

**Why expand first (3 → 128)?**
- More "room" to learn combinations
- Hidden layer can capture intermediate features
- Then compress to desired size (128 → 64)

**Analogy:** Like a funnel—wide at top (128), narrow at bottom (64)

### 3. Why GELU Instead of ReLU?

**GELU (Gaussian Error Linear Unit):**
```python
GELU(x) = x * Φ(x)  # where Φ is CDF of standard normal
```

**Benefits:**
- **Smoother**: Continuous derivative (unlike ReLU)
- **Better gradients**: No dead neurons
- **Proven performance**: Works well in transformers, neural operators

**ReLU alternative:**
```python
ReLU(x) = max(0, x)  # Hard cutoff at 0
```

GELU is generally preferred in modern architectures.

### 4. Why No Nonlinearity on Last Layer?

```python
if i < self.n_layers - 1:  # Skip on last layer
    x = self.non_linearity(x)
```

**Reason:** The last layer is often followed by normalization or other operations in the FNO block that benefit from unbounded outputs.

**However**, in practice, many implementations DO apply nonlinearity on the last layer. The choice depends on the specific architecture.

### 5. Why Flatten Spatial Dimensions?

**Original:**
```python
x.shape = (batch, channels, H, W, ...)  # Multi-dimensional
```

**After flatten:**
```python
x.shape = (batch, channels, H*W*...)  # 1D spatial dimension
```

**Why?** Conv1d expects 3D input: `(batch, channels, length)`. Flattening makes it work with any number of spatial dimensions (2D, 3D, etc.).

**Alternative approaches:**
- Use Conv2d for 2D, Conv3d for 3D, etc. (messy, not general)
- Keep flattened (clean, general)

---

## Complete Code Flow Diagram

Here's the complete flow from input to lifted output:

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT: Temperature Field                                    │
│ Shape: (batch, 1, H, W)                                     │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ POSITIONAL EMBEDDING (if enabled)                           │
│ Adds (x, y) coordinates as extra channels                   │
│ Shape: (batch, 3, H, W)  ← 1 feature + 2 coords            │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ ENTER ChannelMLP.forward()                                  │
│ Input: (batch, 3, H, W)                                     │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ RESHAPE for Conv1d                                          │
│ Flatten spatial dimensions                                  │
│ (batch, 3, H, W) → (batch, 3, H*W)                         │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 1: Conv1d(3 → 128, kernel=1)                         │
│ Apply learned weight matrix W₁ (3×128) at each position    │
│ (batch, 3, H*W) → (batch, 128, H*W)                        │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ ACTIVATION: GELU                                            │
│ Apply nonlinearity element-wise                             │
│ (batch, 128, H*W) → (batch, 128, H*W)                      │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 2: Conv1d(128 → 64, kernel=1)                        │
│ Apply learned weight matrix W₂ (128×64) at each position   │
│ (batch, 128, H*W) → (batch, 64, H*W)                       │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ RESHAPE back to spatial                                     │
│ Restore original spatial dimensions                         │
│ (batch, 64, H*W) → (batch, 64, H, W)                       │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ OUTPUT: Lifted Representation                               │
│ Shape: (batch, 64, H, W)                                    │
│ Rich 64-channel latent space ready for FNO blocks           │
└─────────────────────────────────────────────────────────────┘
```

---

## Mathematical Formulation

### What Lifting Actually Does

**At each spatial position (i, j):**

```
Input: [temp(i,j), x(i,j), y(i,j)]  ∈ ℝ³
                    ↓
        Linear transformation W₁ (3×128)
                    ↓
Intermediate: h(i,j) = GELU(W₁ · [temp, x, y])  ∈ ℝ¹²⁸
                    ↓
        Linear transformation W₂ (128×64)
                    ↓
Output: φ(i,j) = W₂ · h(i,j)  ∈ ℝ⁶⁴
```

**Matrix form (for all positions simultaneously):**

```
Input: X ∈ ℝ^(batch × 3 × H × W)
                    ↓
        Reshape: X' ∈ ℝ^(batch × 3 × HW)
                    ↓
Layer 1: H = GELU(X' @ W₁)  where W₁ ∈ ℝ^(3 × 128)
                    ↓
Layer 2: Y' = H @ W₂  where W₂ ∈ ℝ^(128 × 64)
                    ↓
        Reshape: Y ∈ ℝ^(batch × 64 × H × W)
                    ↓
Output: φ ∈ ℝ^(batch × 64 × H × W)
```

**Key insight:** The same transformation W₁ and W₂ is applied at **every** spatial position, making it resolution-invariant!

---

## Key Takeaways

1. **Lifting expands channels**: 1-3 input channels → 64 (or more) latent channels

2. **Resolution invariant**: Uses Conv1d(kernel=1) which processes each spatial location identically

3. **Two-layer architecture**: Expands (3→128) then compresses (128→64) for expressiveness

4. **Spatial structure preserved**: Flattens only for computation, reshapes back to original dimensions

5. **Learnable transformation**: Weights W₁ and W₂ are learned during training to extract relevant features

6. **Efficient implementation**: Conv1d is faster and more memory-efficient than FC layers

---

## Debugging Tips

If you want to inspect the lifting layer:

```python
from neuralop.models import FNO

model = FNO(
    n_modes=(16, 16),
    in_channels=1,
    out_channels=1,
    hidden_channels=64,
    lifting_channel_ratio=2,
)

# Inspect lifting layer
print(model.lifting)
# Output:
# ChannelMLP(
#   (fcs): ModuleList(
#     (0): Conv1d(3, 128, kernel_size=(1,), stride=(1,))
#     (1): Conv1d(128, 64, kernel_size=(1,), stride=(1,))
#   )
# )

# Check weight shapes
print(model.lifting.fcs[0].weight.shape)  # (128, 3, 1)
print(model.lifting.fcs[1].weight.shape)  # (64, 128, 1)

# Test forward pass
x = torch.randn(4, 1, 64, 64)
print(f"Input: {x.shape}")  # (4, 1, 64, 64)

# After positional embedding (if enabled)
if model.positional_embedding:
    x = model.positional_embedding(x)
    print(f"After embedding: {x.shape}")  # (4, 3, 64, 64)

# After lifting
x = model.lifting(x)
print(f"After lifting: {x.shape}")  # (4, 64, 64, 64)
```

---

## Summary

The lifting code is elegant in its simplicity:

1. **Initialization**: Creates 2 Conv1d layers with appropriate channel sizes
2. **Forward pass**: Reshapes → Conv1d → GELU → Conv1d → Reshape
3. **Result**: Low-dimensional input → High-dimensional latent space

The key innovation is using **Conv1d(kernel_size=1)** instead of FC layers, which makes it:
- ✅ Resolution invariant
- ✅ Memory efficient
- ✅ Fast
- ✅ Easy to work with

This design allows FNO to work at **any input resolution**, which is crucial for operator learning! 🎯





