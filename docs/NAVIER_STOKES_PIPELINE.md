# Navier-Stokes Guide (A–Z)

Everything in this repository that touches Navier–Stokes: data, loaders, preprocessing, model, training script, configuration, evaluation, and the hook points you need to add a physics-informed loss.

---

## Quick map (files)
- `neuralop/data/datasets/navier_stokes.py` — dataset wrapper + loader helper
- `neuralop/data/datasets/pt_dataset.py` — base loader that handles normalization, subsampling, and DataProcessor creation
- `neuralop/data/transforms/data_processors.py` — preprocessing / postprocessing (normalizers, device move, optional multi-grid patching)
- `config/navier_stokes_config.py` — defaults for data/optim/model
- `config/models.py` — `FNO_Medium2d` config used by the training script
- `scripts/train_navier_stokes.py` — end-to-end training entrypoint
- `neuralop/models/fno.py` — Fourier Neural Operator implementation
- `neuralop/losses/data_losses.py` — H1/L2 losses used today
- `neuralop/losses/equation_losses.py` — place to add a physics-informed Navier–Stokes loss

---

## PDE and data representation
- 2D incompressible Navier–Stokes in **vorticity form**; only a single vorticity channel is stored.
- Task: learn the operator mapping an initial vorticity snapshot to a later vorticity field.
- Boundary handling in the model: FNO assumes periodic padding (matches how the data were generated).

### Dataset source
- Downloaded automatically from Zenodo record `12825163`.
- Files stored per resolution (compressed as `nsforcing_<res>.tgz` on Zenodo, unpack to `.pt`):
  - `nsforcing_train_128.pt`, `nsforcing_test_128.pt`
  - `nsforcing_train_1024.pt`, `nsforcing_test_1024.pt`
- `.pt` structure: `{"x": tensor, "y": tensor}` with shapes `(N, H, W)` before channel is added.

### Shapes after loading
- Inputs/targets are unsqueezed to `(N, 1, H, W)` (channel dim = 1 by default).
- Data are cast to `float32`.
- Training and test splits are taken from the first `n_train` / `n_tests[i]` entries; optional spatial subsampling is available.

---

## Data loading and preprocessing

### Load helper
`load_navier_stokes_pt` in `neuralop/data/datasets/navier_stokes.py` builds:
- `train_loader`: PyTorch DataLoader on a TensorDataset of `(x, y)`.
- `test_loaders`: dict keyed by resolution (e.g., `test_loaders[128]`).
- `data_processor`: `DefaultDataProcessor` with fitted normalizers.

Default arguments in the helper: `encode_input=False`, `encode_output=True`, `train_resolution=128`, `num_workers=1`, but the training script overrides via config (see below).

### Dataset internals (via `PTDataset`)
- Performs the download check (only if `download=True`).
- Unsqueezes the channel dimension when there is only one channel.
- Optional spatial subsampling per dimension for inputs/outputs.
- Fits `UnitGaussianNormalizer` on the **training** split:
  - `encoding="channel-wise"` → reduce dims are all axes except channel.
  - If `encode_input/output` is False, the corresponding normalizer is `None`.
- Wraps data in `TensorDataset` and creates a `DefaultDataProcessor` with those normalizers.

### DataProcessor behavior (`DefaultDataProcessor`)
- `preprocess`: moves `{"x","y"}` to device; applies input normalization always, output normalization **only in training mode**.
- `postprocess`: during eval mode, denormalizes model output (and leaves `y` untouched unless you do it explicitly).
- `MGPatchingDataProcessor` swap: if `config.patching.levels > 0`, training uses multi-grid patching (tiling inputs, forwarding per patch, stitching outputs).

---

## Model

### Config used
`config/navier_stokes_config.py` sets `model: FNO_Medium2d()` (from `config/models.py`):
- `data_channels=1`, `out_channels=1`
- `n_modes=[64, 64]`
- `hidden_channels=64`
- `projection_channel_ratio=4`
- `n_layers=4`, `lifting_channel_ratio=2` (from `FNOConfig` defaults)

### Implementation
`neuralop/models/fno.py`
- Lifting MLP → stack of FNO blocks (spectral conv + channel MLP + skip) → projection.
- Periodic domain padding is supported (default disabled via config; `domain_padding` in config would enable).
- Works on arbitrary resolutions as long as `n_modes` fit within Nyquist.

---

## Training script flow
File: `scripts/train_navier_stokes.py`

1) **Config load**
```python
from config.navier_stokes_config import Default
config = make_config_from_cli(Default).to_dict()
```
Defaults (small for quick runs):
- `opt`: `n_epochs=4`, `learning_rate=1e-3`, `training_loss="l2"`, `weight_decay=1e-4`, `scheduler="StepLR"`, `step_size=5`, `gamma=0.7`, `mixed_precision=False`, `eval_interval=1`
- `data`: `folder="~/data/navier_stokes/"`, `batch_size=8`, `n_train=100`, `train_resolution=128`, `n_tests=[50]`, `test_resolutions=[128]`, `test_batch_sizes=[8]`, `encode_input=True`, `encode_output=True`
- `model`: `FNO_Medium2d`
- `patching`: disabled by default
- `wandb`: logging off by default

2) **Device / distributed setup**
`neuralop.training.setup` handles rank, device, and logging guard (`is_logger`).

3) **Data**
```python
train_loader, test_loaders, data_processor = load_navier_stokes_pt(
    data_root=Path(config.data.folder).expanduser(),
    train_resolution=config.data.train_resolution,
    n_train=config.data.n_train,
    batch_size=config.data.batch_size,
    test_resolutions=config.data.test_resolutions,
    n_tests=config.data.n_tests,
    test_batch_sizes=config.data.test_batch_sizes,
    encode_input=config.data.encode_input,
    encode_output=config.data.encode_output,
    num_workers=0,
)
```
If `config.patching.levels > 0`, the processor is replaced with `MGPatchingDataProcessor(model=...)`.

4) **Distributed DataLoaders**
If `config.distributed.use_distributed`, wrap loaders with `DistributedSampler`.

5) **Model / optimizer / scheduler**
```python
model = get_model(config).to(device)
optimizer = AdamW(model.parameters(), lr=config.opt.learning_rate, weight_decay=config.opt.weight_decay)
# Scheduler selected by string (StepLR / CosineAnnealingLR / ReduceLROnPlateau)
```

6) **Losses**
```python
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
train_loss = l2loss if config.opt.training_loss == "l2" else h1loss
eval_losses = {"h1": h1loss, "l2": l2loss}
```

7) **Training**
```python
trainer = Trainer(..., n_epochs=config.opt.n_epochs, data_processor=data_processor, device=device, eval_interval=config.opt.eval_interval, ...)
trainer.train(train_loader, test_loaders, optimizer, scheduler, regularizer=False, training_loss=train_loss, eval_losses=eval_losses)
```
`Trainer` calls `data_processor.preprocess` before forward; during eval it also calls `postprocess` to denormalize outputs for metrics.

8) **Logging**
If W&B is enabled, logs losses/metrics and model parameters. Console prints losses each epoch when `config.verbose` is True.

---

## Evaluation & metrics
- Metrics computed on **denormalized** outputs (because `postprocess` runs in eval mode).
- Default metrics: `h1` (Sobolev norm, periodic finite differences by default) and `l2`.
- Metrics are keyed by resolution: e.g., `128_h1`, `128_l2`.
- Eval runs every `eval_interval` epochs and at the end of training.

---

## Configuration tips
- Override any field via CLI: `python scripts/train_navier_stokes.py --opt.n_epochs 200 --opt.training_loss h1 --data.n_train 10000 --opt.scheduler CosineAnnealingLR`
- For full-scale training (closer to the comments in `navier_stokes_config.py`), bump `n_epochs`, `n_train`, and consider `learning_rate=3e-4`, `step_size=100`, `gamma=0.5`.
- Enable patching for high-res data (e.g., training on 1024²): `--patching.levels 2 --patching.padding 0.25`.
- Turn on W&B: `--wandb.log True --wandb.project navier --wandb.name fno_navier`.

---

## Where to add a physics-informed Navier–Stokes loss
- Implement in `neuralop/losses/equation_losses.py` (see `BurgersEqnLoss` for a template). The data pipeline already provides normalized `x`/`y`; decide whether to work in normalized or physical units.
- Wire it in `scripts/train_navier_stokes.py`:
  1. Instantiate your `NavierStokesEqnLoss` alongside `LpLoss`/`H1Loss`.
  2. Combine losses (e.g., weighted sum) before backprop.
  3. Optionally log the physics residual separately.
- Required ingredients for residuals:
  - The dataset only stores vorticity; velocity/pressure are not available. To enforce NSE you can:
    * Work in vorticity form (need spatial derivatives of vorticity; grid spacing is implicitly 1.0 on a unit square with periodic BCs).
    * Or reconstruct velocity from vorticity via stream-function Poisson solve (not present in repo, would need to add).
  - Use periodic finite differences consistent with `H1Loss` (periodic in x/y by default).
- If you need normalization-aware derivatives, either denormalize inside the loss or bake mean/std into the differential operator.

---

## Quick start command
```bash
python scripts/train_navier_stokes.py \
  --data.folder ~/data/navier_stokes \
  --data.n_train 1000 \
  --opt.n_epochs 50 \
  --opt.training_loss h1
```
This downloads data (if missing), trains an FNO at 128², and prints H1/L2 metrics each epoch.

---

## Summary (pipeline in one glance)
1. **Data**: Download from Zenodo (record 12825163) → `.pt` files with vorticity fields at 128² / 1024².
2. **Load**: `NavierStokesDataset` → `TensorDataset` → `DataLoader` + `DefaultDataProcessor` (normalizers fitted on train split).
3. **Model**: `FNO_Medium2d` (1→1 channels, 64 modes, 64 hidden).
4. **Train**: `Trainer` loop with AdamW + chosen scheduler; losses: L2 or H1 on normalized data.
5. **Eval**: Denormalize outputs, report `h1` and `l2` per resolution; optional W&B logging.
6. **Extend**: Add physics-informed residual loss in `neuralop/losses/equation_losses.py` and plug it into `scripts/train_navier_stokes.py`.
