# Navier–Stokes: Baseline vs Physics (Fixed/Adaptive) – How to run and compare

This is a runnable cheat-sheet for the Navier–Stokes pipeline with the physics loss, covering run commands, naming, and the metrics to compare.

## Experiments to run (128² train/eval)
- **Baseline (data-only)**  
  ```bash
  python scripts/train_navier_stokes.py \
    --physics_loss.enabled False \
    --wandb.log True --wandb.name ns_baseline_128
  ```
- **Physics / fixed weight**  
  ```bash
  python scripts/train_navier_stokes.py \
    --physics_loss.enabled True \
    --physics_loss.weight_schedule none \
    --physics_loss.initial_weight 1.0 --physics_loss.max_weight 1.0 \
    --wandb.log True --wandb.name ns_phy_fixed_128
  ```
- **Physics / adaptive (linear warmup)**  
  ```bash
  python scripts/train_navier_stokes.py \
    --physics_loss.enabled True \
    --physics_loss.weight_schedule linear_warmup \
    --physics_loss.initial_weight 0.1 --physics_loss.max_weight 1.0 \
    --physics_loss.warmup_epochs 50 \
    --wandb.log True --wandb.name ns_phy_adaptive_128
  ```

Adjust `--data.folder` to your dataset location. Increase `--opt.n_epochs` / `--data.n_train` for full runs.

## Cross-resolution / data-scarce settings
- Cross-res eval (adds 1024² test):  
  `--data.test_resolutions "[128,1024]" --data.test_batch_sizes "[8,1]"`
- Data-scarce example:  
  `--data.n_train 200 --opt.n_epochs 50`

## Key outputs and naming
- Run names include `baseline`, `phy_fixed`, or `phy_adaptive` (set automatically if `wandb.name` not provided).
- Trainer logs per-epoch: `avg_data_loss`, `avg_physics_loss`, `physics_weight`, plus component means (prefixed `physics_`).
- Eval metrics per resolution (examples): `128_l2`, `128_h1`, `128_physics`, `128_physics_loss_residual`, `128_physics_loss_adv`.

## Flags quick reference (physics block in config)
- Enable/disable: `--physics_loss.enabled True|False`
- Derivatives: `--physics_loss.derivative_mode spectral|finite_diff`
- Weights: `--physics_loss.initial_weight`, `--physics_loss.max_weight`, `--physics_loss.weight_schedule none|linear_warmup|plateau`, `--physics_loss.warmup_epochs`
- PDE params: `--physics_loss.viscosity`, `--physics_loss.dx`, `--physics_loss.dy`, `--physics_loss.dt`
- Denormalization: `--physics_loss.denormalize True|False` (keeps derivatives in physical units when normalizer is provided)

## Compare checklist
- Plot/lift from logs: `physics_weight`, `avg_physics_loss`, `128_physics_loss_residual`, `128_l2`, `128_h1` (and `1024_*` if cross-res).
- Ensure consistent seed and splits across runs.
- Use identical model/opt settings; only change physics flags/weights to isolate effects.
