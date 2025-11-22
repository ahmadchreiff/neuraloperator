# FNO Quick Reference Guide

## One-Page Summary

### What is FNO?

**Fourier Neural Operator** = Learn mapping between function spaces using frequency domain.

```
Input Function → [LIFT] → [FNO BLOCKS] → [PROJECT] → Output Function
                                    ↓
                        FFT → Mult → IFFT (in frequency!)
```

---

## Key Files

| File | Purpose |
|------|---------|
| `neuralop/models/fno.py` | Main FNO class (480 lines) |
| `neuralop/layers/fno_block.py` | Stack of FNO layers |
| `neuralop/layers/spectral_convolution.py` | FFT→Mult→IFFT |
| `neuralop/layers/channel_mlp.py` | Channel mixing |

---

## Architecture at a Glance

### Constructor (Lines 164-341)
```python
FNO(
    n_modes=(16, 16),           # Keep 16×16 frequencies
    in_channels=1,              # Input features
    out_channels=1,             # Output features  
    hidden_channels=64,         # Width
    n_layers=4,                 # Depth
    lifting_channel_ratio=2,    # Lifting width = 128
    projection_channel_ratio=2, # Projection width = 128
)
```

### Forward Pass (Lines 343-402)
```python
1. Add coordinates → (1 channel → 3 channels)
2. Lift → (3 → 128 → 64 channels)
3. Apply domain padding
4. For each layer:
   - FNO Block (FFT → Mult → IFFT)
5. Remove padding
6. Project → (64 → 128 → 1 channel)
```

---

## SpectralConv: The Heart of FNO

### Algorithm (Lines 402-528)
```
Input: (batch, channels, H, W)
  ↓
FFT: (batch, channels, H, 65)  # Real FFT
  ↓
Shift: Move zero-freq to center
  ↓
Select: Keep n_modes around center (e.g., 16×16)
  ↓
Multiply: × learned weights (only for kept frequencies!)
  ↓
IFFT: (batch, channels, H, W)
  ↓
Output: (batch, channels, H, W)
```

### Key Insight
- **Process**: Only 16×16 frequencies (320× less than 128×128!)
- **Output**: Full 128×128 spatial resolution
- **Why**: Resolution-invariant!

---

## Three Key Components

### 1. Lifting (Lines 301-329)
```python
ChannelMLP(in_channels=3, out_channels=64, hidden_channels=128)
```
**What**: Projects input to higher dimensions  
**Why**: More expressive representations  
**How**: 1D convs (kernel=1) for resolution invariance

### 2. FNO Blocks (Lines 274-299, 394-395)
```python
FNOBlocks(n_layers=4, hidden_channels=64, n_modes=(16,16))
```
**What**: Stack of spectral convolutions  
**Why**: Learns frequency-domain patterns  
**How**: FFT → Mult → IFFT in each layer

### 3. Projection (Lines 331-341)
```python
ChannelMLP(in_channels=64, out_channels=1, hidden_channels=128)
```
**What**: Maps back to output space  
**Why**: Produces final output  
**How**: 1D convs (kernel=1)

---

## Frequency Domain Magic

### Why Frequency Domain?

1. **Global Operations**: Captures long-range dependencies (not just local like CNN)
2. **Efficiency**: Fewer parameters (only n_modes × n_modes weights vs full spatial)
3. **Resolution Invariant**: Input size doesn't matter!

### Real FFT Details

Real-valued spatial input → Redundant FFT coefficients

```
128×128 spatial → 128×65 frequency (not 128×128!)
Why? Real data → complex conjugate symmetry
```

### Keeping Low Frequencies

```
For 128×128 input, n_modes=(16,16):
- Keep frequencies: 56-72 (16 freqs around center 64)
- Discard: 0-56, 72-128 (high frequencies)
```

Most information is in low frequencies anyway!

---

## ChannelMLP: Resolution-Invariant Mixing

### What is it?
```python
ChannelMLP(in_channels, out_channels, hidden_channels, n_layers)
```

### Implementation
```python
# Reshape: (batch, C, H, W) → (batch, C, H*W)
# Apply: Conv1d(kernel=1) at each spatial location
# Reshape: (batch, C, H*W) → (batch, C, H, W)
```

### Why Conv1d(kernel=1)?
- Same operation at every spatial location
- Resolution invariant!
- More efficient than FC layers

---

## Parameters Deep Dive

### n_modes
```python
n_modes = (16, 16)  # For 2D
```
**Meaning**: Number of frequencies to keep per dimension  
**Constraint**: Must be < resolution / 2 (Nyquist)  
**Trade-off**: More modes = more params, more computation  
**Tip**: Start with 16-32

### hidden_channels
```python
hidden_channels = 64
```
**Meaning**: Width of the network  
**Trade-off**: Wider = more params, more memory  
**Tip**: Start with 64, increase if needed

### n_layers
```python
n_layers = 4
```
**Meaning**: Number of FNO blocks  
**Trade-off**: Deeper = more params, better representations  
**Tip**: 4-8 layers is typical

### factorization & rank
```python
factorization = "Tucker"  # or None, "CP", "TT"
rank = 0.1                # 10% of parameters
```
**When**: Use TFNO for large models  
**Benefit**: 10× parameter reduction!  
**Trade-off**: Slight performance drop

---

## Common Configurations

### Small Model (CPU-friendly)
```python
FNO(n_modes=(8, 8), hidden_channels=32, n_layers=2)
# ~100K parameters
```

### Medium Model (Single GPU)
```python
FNO(n_modes=(16, 16), hidden_channels=64, n_layers=4)
# ~500K parameters
```

### Large Model (Multi-GPU)
```python
FNO(n_modes=(32, 32), hidden_channels=128, n_layers=6)
# ~2M parameters
```

### Tensorized (Memory Efficient)
```python
TFNO(n_modes=(32, 32), hidden_channels=128, n_layers=6, rank=0.1)
# ~200K parameters (10× smaller!)
```

---

## Forward Pass Dimensions

### Example: 2D Input, 64×64 Grid

```
Input: (32, 1, 64, 64)
  ↓ positional embedding
(32, 3, 64, 64)  # + (x,y) coords
  ↓ lifting (3→128→64)
(32, 64, 64, 64)
  ↓ FNO Block 1
  ↓   - FFT: (32, 64, 64, 33)
  ↓   - Keep: (32, 64, 16, 16)  # only n_modes!
  ↓   - IFFT: (32, 64, 64, 64)
(32, 64, 64, 64)
  ↓ ... 3 more FNO blocks
(32, 64, 64, 64)
  ↓ projection (64→128→1)
(32, 1, 64, 64)  # Final output
```

---

## Key Concepts Checklist

- [x] **Resolution Invariance**: Works at any input size
- [x] **Frequency Domain**: FFT operations, not spatial convs
- [x] **Low Frequencies**: Only keep n_modes × n_modes
- [x] **Global Operations**: Captures long-range dependencies
- [x] **Channel Mixing**: 1D convs for efficiency
- [x] **Tensor Factorization**: Optional compression

---

## Common Mistakes

❌ **Too many modes**: `n_modes >= resolution // 2`  
✅ **Solution**: Use `n_modes < resolution // 2`

❌ **Too few hidden channels**: Can't learn complex patterns  
✅ **Solution**: Increase `hidden_channels` to 64-128

❌ **Forgetting positional embedding**: Model has no spatial sense  
✅ **Solution**: Keep default `positional_embedding="grid"`

❌ **Not enough layers**: Shallow representations  
✅ **Solution**: Use `n_layers >= 4`

---

## Performance Tips

1. **Use TFNO** for large models: 10× fewer parameters
2. **Enable mixed precision**: Faster training, less memory
3. **Start small**: n_modes=16, hidden=64, layers=4
4. **Scale up** if underfitting: Increase channels/layers first
5. **Test resolution invariance**: Try different input sizes!

---

## Next Steps

1. Read: `FNO_CODE_WALKTHROUGH.md` - Full explanation
2. Try: `examples/models/plot_FNO_darcy.py` - Run it!
3. Modify: Change parameters, see what happens
4. Debug: Print intermediate shapes in forward()

---

## Quick Formula

**Number of FNO parameters ≈**

```
Lifting: (in_chans + coords) × lifting_chans + lifting_chans × hidden_chans
+ 
n_layers × [ (in_chans × out_chans × n_modes × n_modes) × 2 ]  # Each block
+ 
Projection: hidden_chans × projection_chans + projection_chans × out_chans
```

**For TFNO**: Multiply FNO weight part by `rank` (e.g., 0.1 for 10× reduction)

---

**Remember**: FNO is just lifting + (FFT→Mult→IFFT) + projection! 🎯






