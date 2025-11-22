# FNO Code Walkthrough: Understanding the Implementation

## Overview
This document walks through the FNO (Fourier Neural Operator) implementation in `neuralop/models/fno.py` line by line.

---

## The Big Picture

FNO learns mappings between **function spaces**. Think of it as:
- **Input**: A function sampled on a grid (e.g., temperature field on a 2D domain)
- **Output**: Another function sampled on the same grid (e.g., pressure field)

**Key Innovation**: Works in **frequency domain** via FFT, not spatial convolutions!

---

## Architecture Breakdown

### 3 Main Stages:

```
INPUT → [LIFTING] → [FNO BLOCKS] → [PROJECTION] → OUTPUT
  ↓         ↓              ↓             ↓
 f(x,y)   φ(x,y)      FFT→Mult→IFFT   g(x,y)
         ↑                               
    Higher dim latent space
```

1. **Lifting**: Project input to higher dimensional space
2. **FNO Blocks**: Sequence of spectral convolutions (the magic!)
3. **Projection**: Map back to output dimensions

---

## Class Definition: Line 25

```python
class FNO(BaseModel, name="FNO"):
```

- Inherits from `BaseModel` for checkpointing, registration, etc.
- Auto-registers itself with name "FNO"

---

## Constructor: Lines 164-479

### Key Parameters

#### Essential (You MUST provide):
```python
n_modes: (16, 16)          # Number of Fourier modes (determines input dimensionality)
in_channels: 1             # Input feature size
out_channels: 1            # Output feature size  
hidden_channels: 64        # Width of the network
```

#### Common Optional:
```python
n_layers: 4                # Number of FNO blocks (depth)
lifting_channel_ratio: 2   # Lifting width = 2 * hidden_channels
projection_channel_ratio: 2 # Projection width = 2 * hidden_channels
```

#### Advanced Optional:
```python
factorization: "Tucker"    # Use tensorized FNO? (None, "Tucker", "CP", "TT")
rank: 0.1                  # Compression factor (0.1 = 10% parameters)
use_channel_mlp: True      # Add MLP after each FNO block?
domain_padding: 0.078125   # Padding for periodic-like boundaries
positional_embedding: "grid" # Add coordinates as input?
```

---

## Constructor Logic: Step-by-Step

### Step 1: Store Dimensions (Lines 199-208)

```python
self.n_dim = len(n_modes)  # 2 for 2D, 3 for 3D, etc.
self._n_modes = n_modes    # Store modes
self.hidden_channels = hidden_channels
self.in_channels = in_channels
self.out_channels = out_channels
self.n_layers = n_layers
```

**Why store these?** Accessible throughout the class.

---

### Step 2: Set Channel Sizes (Lines 211-215)

```python
self.lifting_channel_ratio = lifting_channel_ratio
self.lifting_channels = int(lifting_channel_ratio * self.hidden_channels)

self.projection_channel_ratio = projection_channel_ratio
self.projection_channels = int(projection_channel_ratio * self.hidden_channels)
```

**Example**: If `hidden_channels=64` and `lifting_channel_ratio=2`:
- `lifting_channels = 128`
- Input (1 channel) → lifted to 128 → compressed to 64 for FNO blocks

**Why ratios?** Scales architecture width proportionally.

---

### Step 3: Positional Embedding (Lines 230-253)

```python
if positional_embedding == "grid":
    spatial_grid_boundaries = [[0.0, 1.0]] * self.n_dim
    self.positional_embedding = GridEmbeddingND(
        in_channels=self.in_channels,
        dim=self.n_dim,
        grid_boundaries=spatial_grid_boundaries,
    )
```

**What does it do?** Adds (x, y) coordinates as extra channels:
- Input: 1 channel (e.g., permeability field)
- After embedding: 3 channels (permeability, x, y)

**Why?** Helps model learn spatial patterns.

---

### Step 4: Domain Padding (Lines 255-265)

```python
if domain_padding is not None and (...):
    self.domain_padding = DomainPadding(
        domain_padding=domain_padding,
        resolution_scaling_factor=resolution_scaling_factor,
    )
```

**What does it do?** Adds padding around boundaries (periodic-like).

**Why?** Improves boundary handling for non-periodic domains.

---

### Step 5: Create FNO Blocks (Lines 274-299)

```python
self.fno_blocks = FNOBlocks(
    in_channels=hidden_channels,
    out_channels=hidden_channels,
    n_modes=self.n_modes,
    # ... many parameters ...
)
```

**This is the core!** Contains `n_layers` Fourier convolution layers.

**Key parameters passed**:
- `use_channel_mlp`: Add MLP after each spectral conv?
- `channel_mlp_expansion`: Width multiplier for MLP
- `fno_skip`: Skip connection type ("linear", "identity", "soft-gating")
- `factorization`: Tensor decomposition type
- `rank`: Compression factor
- `norm`: Normalization ("group_norm", "instance_norm", etc.)

---

### Step 6: Create Lifting Layer (Lines 301-329)

```python
lifting_in_channels = self.in_channels
if self.positional_embedding is not None:
    lifting_in_channels += self.n_dim  # Add coordinate channels
```

**Example**: 1D input + 2D coords → 3 channels input to lifting

```python
if self.lifting_channels:
    self.lifting = ChannelMLP(
        in_channels=lifting_in_channels,  # 3
        out_channels=self.hidden_channels,  # 64
        hidden_channels=self.lifting_channels,  # 128
        n_layers=2,
        n_dim=self.n_dim,
        non_linearity=non_linearity,
    )
```

**What does ChannelMLP do?**
- Applies 1D convolutions (kernel=1) to mix channels
- Uses `Conv1d(kernel_size=1)` to process channels independently at each spatial location

**Architecture**: Input(3) → Hidden(128) → Output(64)

If `lifting_channels` wasn't set → single linear layer.

---

### Step 7: Create Projection Layer (Lines 331-341)

```python
self.projection = ChannelMLP(
    in_channels=self.hidden_channels,  # 64
    out_channels=out_channels,  # 1
    hidden_channels=self.projection_channels,  # 128
    n_layers=2,
    n_dim=self.n_dim,
    non_linearity=non_linearity,
)
```

**Similar to lifting**, but:
- Input: 64 channels (from FNO blocks)
- Hidden: 128 channels
- Output: 1 channel

---

### Step 8: Handle Complex Data (Lines 328-329, 340-341)

```python
if self.complex_data:
    self.lifting = ComplexValued(self.lifting)
```

Wraps layers to handle complex-valued data (e.g., quantum mechanics).

---

## Forward Pass: Lines 343-402

This is where the magic happens!

### The Complete Flow:

```python
def forward(self, x, output_shape=None, **kwargs):
    # 1. Add positional embedding
    if self.positional_embedding is not None:
        x = self.positional_embedding(x)
    
    # 2. LIFT: Project to high dim
    x = self.lifting(x)
    
    # 3. Apply domain padding
    if self.domain_padding is not None:
        x = self.domain_padding.pad(x)
    
    # 4. Apply n_layers FNO blocks
    for layer_idx in range(self.n_layers):
        x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])
    
    # 5. Remove domain padding
    if self.domain_padding is not None:
        x = self.domain_padding.unpad(x)
    
    # 6. PROJECT: Map back to output
    x = self.projection(x)
    
    return x
```

---

## Detailed Forward Pass Explanation

### Stage 1: Positional Embedding (Line 386-387)

```python
if self.positional_embedding is not None:
    x = self.positional_embedding(x)
```

**Input**: `(batch, 1, H, W)`  
**Output**: `(batch, 3, H, W)` ← adds x, y coordinates

---

### Stage 2: Lifting (Line 389)

```python
x = self.lifting(x)
```

**What happens inside ChannelMLP**:
```
Input: (batch, 3, H, W)
↓
Reshape to: (batch, 3, H*W)
↓
Conv1d(3 → 128) → GELU → Conv1d(128 → 64)  # Channel mixing
↓
Reshape back: (batch, 64, H, W)
↓
Output: (batch, 64, H, W)
```

**Why 1D convs?** Each spatial location processed identically, resolution invariant!

---

### Stage 3: Domain Padding (Line 391-392)

```python
if self.domain_padding is not None:
    x = self.domain_padding.pad(x)
```

Pads boundaries for better periodic handling.

---

### Stage 4: FNO Blocks (Line 394-395)

```python
for layer_idx in range(self.n_layers):
    x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])
```

**This is where the frequency domain magic happens!**

Each FNO block does:
```
Input: (batch, hidden_channels, H, W)
↓
1. FFT → Frequency domain
↓
2. Keep only n_modes frequencies (e.g., 16x16 modes)
↓
3. Multiply by learned weights (complex!)
↓
4. IFFT → Spatial domain
↓
5. Add skip connection
↓
6. Apply normalization + nonlinearity
↓
7. (Optional) ChannelMLP
↓
Output: (batch, hidden_channels, H, W)
```

**Why FFT?** Convolutions become simple multiplications in frequency space!

**Why keep few modes?** Most information is in low frequencies.

---

### Stage 5: Unpad (Line 397-398)

```python
if self.domain_padding is not None:
    x = self.domain_padding.unpad(x)
```

Removes the padding added earlier.

---

### Stage 6: Projection (Line 400)

```python
x = self.projection(x)
```

Maps from 64 → 1 channels, similar to lifting but in reverse.

---

## Properties: Lines 404-411

```python
@property
def n_modes(self):
    return self._n_modes

@n_modes.setter
def n_modes(self, n_modes):
    self.fno_blocks.n_modes = n_modes  # Update FNO blocks too!
    self._n_modes = n_modes
```

Allows dynamic mode adjustment during training (e.g., curriculum learning).

---

## TFNO: Lines 444-479

```python
class TFNO(FNO):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("factorization", "Tucker")
        kwargs.setdefault("rank", 0.1)
        super().__init__(*args, **kwargs)
```

**TFNO = FNO with tensor factorization**

Just sets default factorization to Tucker with rank=0.1 (90% compression!)

---

## Understanding FNOBlocks

Let's peek inside what FNOBlocks does (refer to `fno_block.py`):

### Each Block Contains:

1. **SpectralConv**: The Fourier convolution layer
   - Does FFT → Multiply → IFFT
   
2. **Skip connection**: Identity, linear, or soft-gating
   - Helps with gradient flow
   
3. **Normalization**: Group norm, instance norm, etc.
   - Stabilizes training
   
4. **Nonlinearity**: GELU, ReLU, etc.
   - Adds expressiveness
   
5. **ChannelMLP** (optional): Additional channel mixing
   - Increases capacity

### Skip Connections:
- **Linear**: Learnable linear transform
- **Identity**: Direct addition
- **Soft-gating**: Learned weighted combination

---

## Understanding SpectralConv

This is the CORE of FNO (see `spectral_convolution.py`):

```python
# Simplified pseudo-code:
def spectral_conv(x):
    # 1. FFT to frequency domain
    x_freq = FFT(x)  # Shape: (batch, channels, H, W, 2) - complex!
    
    # 2. Keep only low frequencies
    x_freq_low = x_freq[:, :, :n_modes, :n_modes, :]
    
    # 3. Multiply by learned weights
    output_freq = x_freq_low * weights  # weights is learnable!
    
    # 4. IFFT back to spatial
    output = IFFT(output_freq)
    
    return output
```

**Key insight**: Only keeping `n_modes` frequencies makes it:
1. **Resolution invariant**: Works at any input size
2. **Parameter efficient**: Fewer weights
3. **Computationally efficient**: Fewer FFT operations

---

## Detailed SpectralConv Forward Pass (Lines 402-528)

Let's understand the actual implementation:

### Step 1: Get Input Dimensions (Line 414)
```python
batchsize, channels, *mode_sizes = x.shape
# For (32, 64, 128, 128): batchsize=32, channels=64, mode_sizes=[128, 128]
```

### Step 2: Prepare FFT Size (Lines 416-418)
```python
fft_size = list(mode_sizes)
if not self.complex_data:
    fft_size[-1] = fft_size[-1] // 2 + 1  # Real FFT is redundant
# For real data at 128×128: fft_size = [128, 65] (not 128×128!)
```

**Why half?** Real-valued spatial data → redundant FFT coefficients.

### Step 3: Transform to Frequency Domain (Lines 424-431)
```python
if self.complex_data:
    x = torch.fft.fftn(x, norm=self.fft_norm, dim=fft_dims)
else:
    x = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)
```

**rfftn** = Real-valued FFT (computes only non-redundant frequencies).

After this, `x` shape changes:
- `(batch, 64, 128, 128)` → `(batch, 64, 128, 65)` (real FFT)

### Step 4: FFT Shift (Line 434)
```python
if self.order > 1:  # Only for 2D+
    x = torch.fft.fftshift(x, dim=dims_to_fft_shift)
```

**What?** Moves zero-frequency to center of spectrum.

**Why?** Makes it easier to select low frequencies around DC (zero-freq).

### Step 5: Select Low Frequencies (Lines 449-497)

This is where we **keep only n_modes**:

```python
# Center indices for each dimension
center = all_modes // 2  # 128 // 2 = 64

# Calculate how many negative/positive frequencies to keep
negative_freqs = kept_modes // 2  # 16 // 2 = 8
positive_freqs = kept_modes // 2 + kept_modes % 2  # 8 or 9

# Slice: keep frequencies around center
slices_x += [slice(center - negative_freqs, center + positive_freqs)]
# For 128×128 with n_modes=(16,16): slice(56, 72) along each dim
```

**Example**: 128×128 input, n_modes=(16,16)
- Keep frequencies 56-72 along each dimension (around center 64)
- Discard high frequencies (0-56, 72-128)

### Step 6: Multiply by Learned Weights (Lines 505-507)
```python
out_fft[slices_x] = self._contract(x[slices_x], weight, separable=self.separable)
```

**Contract** = einsum or matrix multiplication
- **Dense**: `einsum("bcxy,ocxy->bocxy", x, weight)`
- **Tucker**: Direct contraction with factors (more efficient!)

### Step 7: Inverse FFT (Lines 518-523)
```python
if self.complex_data:
    x = torch.fft.ifftn(out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)
else:
    x = torch.fft.irfftn(out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)
```

Transform back to spatial domain! Result: `(batch, 64, 128, 128)`.

---

### Visual Example: 2D Spectral Convolution

```
INPUT (spatial domain):
┌─────────────────┐
│  (32, 64, 128, 128) │  ← Temperature field
└─────────────────┘
         ↓
   FFT (rfftn)
         ↓
FREQUENCY DOMAIN:
┌─────────────────┐
│  (32, 64, 128, 65) │  ← Complex frequencies
└─────────────────┘
         ↓
   FFT Shift
         ↓
   KEEP LOW FREQ:
┌─────────────────┐
│  (32, 64, 16, 16) │  ← Only n_modes frequencies
└─────────────────┘
         ↓
   × LEARNED WEIGHTS
         ↓
┌─────────────────┐
│  (32, 64, 16, 16) │
└─────────────────┘
         ↓
   ZERO-PAD TO ORIGINAL SIZE
         ↓
IFFT (irfftn)
         ↓
OUTPUT (spatial domain):
┌─────────────────┐
│  (32, 64, 128, 128) │  ← Convolved result
└─────────────────┘
```

**Key**: We only process 16×16 frequencies but output 128×128 spatial!

---

## Contract Functions (Lines 21-133)

Different contraction methods for efficiency:

### Dense Contraction (Lines 21-47)
```python
def _contract_dense(x, weight, separable=False):
    # x: (batch, in_channels, *spatial_dims)
    # weight: (in_channels, out_channels, *spatial_dims)
    # Output: (batch, out_channels, *spatial_dims)
    
    # Use einsum for flexibility
    eq = "batch,in,x,y,in,out,x,y->batch,out,x,y"
    return tl.einsum(eq, x, weight)
```

### Tucker Contraction (Lines 76-103)
```python
def _contract_tucker(x, tucker_weight, separable=False):
    # tucker_weight has: core + factors
    # Instead of full (in, out, modes, modes) tensor,
    # store as: core(mode_rank, mode_rank) + factors
    # Much smaller!
    
    # Contract directly with factors (more efficient)
    return tl.einsum(eq, x, tucker_weight.core, *tucker_weight.factors)
```

**Example**: 64×64×16×16 weight = 1M params
Tucker (rank=0.1): 100K params (10× smaller!)

---

## Understanding ChannelMLP

See `channel_mlp.py` for details:

```python
# Simplified:
class ChannelMLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, n_layers):
        self.fcs = nn.ModuleList()
        # Each fc is a Conv1d(kernel_size=1)
        # Layer 1: in_channels → hidden_channels
        # Layer 2 to n-1: hidden_channels → hidden_channels
        # Layer n: hidden_channels → out_channels
    
    def forward(self, x):
        # Reshape: (batch, C, H, W) → (batch, C, H*W)
        x = x.view(batch, C, -1)
        
        # Apply convolutions
        for fc in self.fcs:
            x = fc(x)  # Mixes channels
            x = nonlinearity(x)
            x = dropout(x)
        
        # Reshape back: (batch, C, H*W) → (batch, C, H, W)
        return x.view(batch, C, H, W)
```

**Why Conv1d(kernel=1)?**
- Processes each spatial location the same way
- More efficient than fully-connected layers
- Resolution invariant!

---

## Key Concepts Summary

### 1. Resolution Invariance

FNO is **resolution invariant** because:
- Lifting/Projection use 1D convs (same operation at each spatial location)
- FFT naturally handles any grid size
- Only keeping low frequencies means same number of parameters regardless of input size

**Result**: Train on 64×64, test on 512×512! (zero-shot super-resolution)

---

### 2. Frequency Domain Processing

Working in frequency domain:
- **Convolutions become multiplications** (much faster!)
- **Low frequencies contain most information**
- **Translation invariant** by design

---

### 3. Channel Mixing

ChannelMLPs mix information across feature dimensions:
- Input/output channels are independent per spatial location
- Allows expressive transformations
- 1D convs make this efficient

---

### 4. Skip Connections

Help with:
- Gradient flow in deep networks
- Learning residual patterns
- Stability during training

---

## Example: Building an FNO

```python
from neuralop.models import FNO

# For 2D Darcy flow problem:
model = FNO(
    n_modes=(16, 16),      # Keep 16x16 frequencies in each direction
    in_channels=1,         # Input: permeability field
    out_channels=1,        # Output: pressure field
    hidden_channels=64,    # Width of network
    n_layers=4,            # Depth
    lifting_channel_ratio=2,     # Lifting: 1→128→64
    projection_channel_ratio=2,  # Projection: 64→128→1
    use_channel_mlp=True,  # Add MLPs after each block
    fno_skip="linear",     # Linear skip connections
)

# Forward pass:
input = torch.randn(32, 1, 64, 64)  # 32 samples, 1 channel, 64x64 grid
output = model(input)                # Shape: (32, 1, 64, 64)
```

---

## Comparison with Standard CNNs

| Aspect | CNN | FNO |
|--------|-----|-----|
| **Convolution** | Spatial (sparse) | Spectral (global) |
| **Domain** | Spatial | Frequency |
| **Modes** | Kernel size × kernel size | n_modes × n_modes |
| **Resolution** | Fixed | Invariant |
| **Parameters** | Scales with kernel² | Scales with modes² |

**FNO advantages**:
- ✅ Fewer parameters
- ✅ Resolution invariant
- ✅ Captures long-range dependencies globally

---

## Advanced: Tensor Factorization (TFNO)

Instead of dense weights in frequency domain, use low-rank decomposition:

```python
# Dense FNO weight: (in_channels, out_channels, n_modes, n_modes)
# Tucker decomposition: Factorizes into smaller tensors
# Result: 10× fewer parameters!

model = TFNO(
    n_modes=(16, 16),
    hidden_channels=64,
    rank=0.1  # Use 10% parameters
)
```

This uses Tucker decomposition to compress weights while maintaining performance.

---

## Debugging Tips

1. **Check dimensions**:
   ```python
   print(model)  # See architecture
   ```

2. **Count parameters**:
   ```python
   from neuralop.utils import count_model_params
   n_params = count_model_params(model)
   ```

3. **Verify forward pass**:
   ```python
   x = torch.randn(1, in_channels, 64, 64)
   out = model(x)
   print(out.shape)  # Should be (1, out_channels, 64, 64)
   ```

4. **Test different resolutions**:
   ```python
   x1 = torch.randn(1, in_channels, 64, 64)
   x2 = torch.randn(1, in_channels, 128, 128)
   out1 = model(x1)  # Works!
   out2 = model(x2)  # Also works (resolution invariant)!
   ```

---

## Common Pitfalls

1. **n_modes too large**: `n_modes >= resolution // 2` violates Nyquist
2. **Not enough hidden_channels**: Model too narrow to learn
3. **Too many layers**: Vanishing gradients, overfitting
4. **Missing positional embedding**: Model doesn't know spatial locations
5. **Wrong factorization rank**: Too low → poor performance, too high → no compression

---

## Next Steps

1. **Read**: `fno_block.py` - Understand inner workings
2. **Read**: `spectral_convolution.py` - Core frequency operations
3. **Experiment**: Try different `n_modes`, `hidden_channels`
4. **Run**: `examples/models/plot_FNO_darcy.py` - See it in action!

---

## Key Takeaways

1. **FNO = Lifting + Spectral Convolutions + Projection**
2. **SpectralConv = FFT → Multiply → IFFT** (in frequency domain)
3. **ChannelMLP = Resolution-invariant channel mixing**
4. **Resolution invariance** comes from global operations + low frequencies
5. **TFNO** adds tensor decomposition for parameter efficiency

The FNO is elegant: by working in frequency domain, it captures global patterns efficiently while remaining resolution invariant! 🎯

