# Understanding FNO: A Complete Guide

## What You Have Now

I've created **three comprehensive documents** to help you understand the FNO implementation:

### 1. **REPOSITORY_EXPLORATION_SUMMARY.md**
- Overview of the entire NeuralOperator library
- All models, layers, and features
- Good starting point for library-wide understanding

### 2. **FNO_CODE_WALKTHROUGH.md** ⭐ **START HERE**
- **Line-by-line** explanation of `fno.py`
- Detailed breakdown of constructor and forward pass
- Explains every major component
- Visual diagrams and examples
- **~800 lines of detailed explanation**

### 3. **FNO_QUICK_REFERENCE.md**
- One-page cheat sheet
- Quick formulas and configurations
- Common mistakes and tips
- Parameter reference

---

## How to Read These Documents

### For Deep Understanding:
1. Start with **FNO_CODE_WALKTHROUGH.md**
2. Read it alongside the actual code in `neuralop/models/fno.py`
3. Run the examples in `examples/models/plot_FNO_darcy.py`
4. Experiment with different parameters

### For Quick Reference:
- Use **FNO_QUICK_REFERENCE.md** when you need parameter info or formulas

### For Context:
- Read **REPOSITORY_EXPLORATION_SUMMARY.md** to understand how FNO fits into the library

---

## The FNO in 30 Seconds

**Fourier Neural Operator** learns to map one function to another:

```
Temperature Field → FNO → Pressure Field
```

**How?** Work in **frequency domain** instead of spatial:

```
Spatial Domain          Frequency Domain
─────────────────       ─────────────────
Input: f(x,y)    → FFT →  F(ω₁,ω₂)
                          ↓ × Learned Weights
                     ← IFFT ←  G(ω₁,ω₂)
Output: g(x,y)
```

**Why frequency domain?**
- ✅ Global operations (not just local like CNN)
- ✅ Fewer parameters (only keep low frequencies)
- ✅ Resolution invariant (works at any size!)

---

## Key Implementation Details

### Architecture (from fno.py)

```
class FNO:
    def __init__:
        # 1. Lifting layer
        self.lifting = ChannelMLP(1→128→64)
        
        # 2. FNO blocks (the magic!)
        self.fno_blocks = FNOBlocks(
            n_layers=4,
            n_modes=(16,16),
            hidden_channels=64
        )
        
        # 3. Projection layer
        self.projection = ChannelMLP(64→128→1)
    
    def forward(x):
        x = positional_embedding(x)  # Add (x,y) coords
        x = self.lifting(x)          # 1→64 channels
        x = self.fno_blocks(x)       # 4× spectral convs
        x = self.projection(x)       # 64→1 channel
        return x
```

### SpectralConv (the heart)

Each FNO block does:

```python
def spectral_conv(x):
    # Transform to frequency domain
    x_freq = FFT(x)                    # (batch, 64, 128, 128)
    
    # Keep only low frequencies
    x_low = x_freq[:, :, 56:72, 56:72] # (batch, 64, 16, 16)
    
    # Multiply by learned weights
    out_low = x_low * weight           # (batch, 64, 16, 16)
    
    # Transform back
    out = IFFT(out_low)                # (batch, 64, 128, 128)
    
    return out
```

**Key insight**: Only process 16×16 frequencies but output 128×128 spatial!

---

## Main File Locations

| What | Where | Lines |
|------|-------|-------|
| **FNO Class** | `neuralop/models/fno.py` | 480 |
| **FNO Blocks** | `neuralop/layers/fno_block.py` | 423 |
| **SpectralConv** | `neuralop/layers/spectral_convolution.py` | 529 |
| **ChannelMLP** | `neuralop/layers/channel_mlp.py` | 188 |
| **Base Model** | `neuralop/models/base_model.py` | 193 |

---

## Recommended Reading Order

### 1. Understand the Concept
- Read the "Big Picture" section in **FNO_CODE_WALKTHROUGH.md**
- Understand why we work in frequency domain

### 2. See the Code
- Open `neuralop/models/fno.py`
- Read it alongside **FNO_CODE_WALKTHROUGH.md**

### 3. Understand Each Component
- **Lifting**: Lines 301-329 in walkthrough
- **FNO Blocks**: Lines 274-299, 394-395
- **Projection**: Lines 331-341
- **Forward Pass**: Lines 343-402

### 4. Deep Dive
- Read SpectralConv section (Lines 402-528 explanation)
- Understand how FFT works
- See the contract functions

### 5. Experiment
- Run `examples/models/plot_FNO_darcy.py`
- Change parameters
- Print intermediate shapes

---

## Quick Test

Try this to verify understanding:

```python
from neuralop.models import FNO

# Create a simple FNO
model = FNO(
    n_modes=(8, 8),
    in_channels=1,
    out_channels=1,
    hidden_channels=64
)

# Test forward pass
x = torch.randn(4, 1, 64, 64)  # 4 samples, 1 channel, 64×64 grid
out = model(x)
print(out.shape)  # Should be (4, 1, 64, 64)

# Test resolution invariance
x2 = torch.randn(4, 1, 128, 128)  # Different resolution!
out2 = model(x2)
print(out2.shape)  # Should be (4, 1, 128, 128) - resolution invariant!
```

---

## Common Questions Answered

### Q: Why frequency domain?
**A**: Convolutions become simple multiplications. Plus global operations capture long-range dependencies.

### Q: Why only keep low frequencies?
**A**: Most signal energy is in low frequencies. Fewer parameters, resolution invariant, faster.

### Q: How is it resolution invariant?
**A**: FFT works at any size, and we only keep n_modes × n_modes frequencies regardless of input size.

### Q: What's the difference between FNO and CNN?
**A**: CNN = local convolutions in spatial domain. FNO = global operations in frequency domain.

### Q: What's TFNO vs FNO?
**A**: TFNO uses tensor factorization (Tucker decomposition) to compress weights. 10× fewer parameters!

### Q: When to use TFNO?
**A**: When model is too large for memory or you want fewer parameters.

---

## Key Takeaways

1. **FNO = Lifting + SpectralConv + Projection**
2. **SpectralConv = FFT → Multiply → IFFT** (in frequency)
3. **Only keep n_modes frequencies** (e.g., 16×16)
4. **Resolution invariant** by design
5. **ChannelMLP** uses 1D convs for efficiency
6. **TFNO** compresses via tensor decomposition

---

## Next Steps

1. ✅ **Read** FNO_CODE_WALKTHROUGH.md thoroughly
2. ✅ **Read** the actual fno.py code
3. ✅ **Run** examples/models/plot_FNO_darcy.py
4. ✅ **Experiment** with parameters
5. ✅ **Read** spectral_convolution.py if you want deeper understanding
6. ✅ **Try** building your own model!

---

## Resources

- **Original Paper**: Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations" (ICLR 2021)
- **Code**: `neuralop/models/fno.py`
- **Examples**: `examples/models/plot_FNO_darcy.py`
- **Documentation**: https://neuraloperator.github.io/dev/

---

**You now have everything you need to understand FNO!** 🎉

Start with **FNO_CODE_WALKTHROUGH.md** and read it alongside the actual code. The explanations are designed to match line-by-line with the implementation.

Good luck! 🚀






