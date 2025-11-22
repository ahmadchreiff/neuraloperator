# Navier–Stokes Physics-Informed Extension – Detailed Report

This document summarizes the features added to the Navier–Stokes pipeline: physics-informed loss (NavierStokesEqnLoss), adaptive physics weighting, trainer and config wiring, metrics/logging, tests, and how to run comparisons.

---

## 1) Physics-Informed Loss (NavierStokesEqnLoss)
**Path:** `neuralop/losses/equation_losses.py`  
**Purpose:** Enforce 2D incompressible Navier–Stokes in vorticity form during training/eval.

- **Inputs:** prediction shaped `(B, C, T, H, W)` or `(B, T, C, H, W)`. Layout handled via heuristic or `inputs["channel_first"]/layout`.
- **Denormalization:** If `denormalize=True` and a normalizer is passed, `inverse_transform` is applied before derivatives.
- **Derivatives:** Spectral (`torch.fft.rfftn`) or periodic finite-difference (`FiniteDiff`) for ∂x, ∂y, Laplacian. Time derivative uses central difference over T.
- **Residual (vorticity form):**  
  ω = ∂x u_y − ∂y u_x  
  R = ω_t + (u · ∇)ω − ν Δω − fω
- **Loss composition:** MSE of residual plus component penalties  
  `loss = w_adv * ||adv||^2 + w_diff * ||νΔω||^2 + w_force * ||f||^2 + ||R||^2`
- **Outputs:** Returns scalar loss; optionally `(loss, components)` with `loss_adv`, `loss_diff`, `loss_forcing`, `loss_residual`, `residual_mse`.
- **Assumptions:** Periodic BCs; vorticity form only (pressure not enforced).

---

## 2) Adaptive Physics Weighting
**Path:** `neuralop/training/schedulers.py` → `PhysicsWeightScheduler`  
**Modes:** `none` (fixed), `linear_warmup`, `plateau` (simple plateau trigger).  
**Fields:** `initial_weight`, `max_weight`, `warmup_epochs`, `plateau_patience`, `plateau_threshold`.

---

## 3) Trainer Integration
**Path:** `neuralop/training/trainer.py`

- Trainer now accepts `physics_loss_fn`, `physics_weight`, `physics_weight_scheduler`, `physics_normalizer_train/eval`, `physics_return_components`, `eval_physics_loss`.
- Training loop mixes `data_loss + physics_weight * physics_loss`; logs `avg_data_loss`, `avg_physics_loss`, `physics_weight`, and per-component means.
- Eval loop can compute physics metrics: keys like `<loader>_physics`, `<loader>_physics_loss_residual`, `<loader>_physics_loss_adv`, etc.

---

## 4) Config & CLI Wiring
**Path:** `config/navier_stokes_config.py`  
Added `physics_loss` block:
- `enabled`
- PDE params: `viscosity`, `dx`, `dy`, `dt`
- Component weights: `advection_weight`, `diffusion_weight`, `forcing_weight`
- Options: `use_vorticity_form`, `derivative_mode` (`spectral|finite_diff`), `denormalize`
- Weight schedule: `initial_weight`, `max_weight`, `warmup_epochs`, `weight_schedule` (`none|linear_warmup|plateau`)

**Training script:** `scripts/train_navier_stokes.py`
- Instantiates `NavierStokesEqnLoss` when enabled; sets up scheduler; passes normalizer (if denormalizing).
- W&B run name prefix auto-encodes mode: `baseline` (off) vs `phy_fixed` vs `phy_adaptive`.
- Trainer fed physics knobs for loss mixing and logging.

---

## 5) Metrics & Logging
- Per-epoch (train): `avg_data_loss`, `avg_physics_loss`, `physics_weight`, component averages (prefixed `physics_`).
- Eval: per loader/resolution metrics `*_l2`, `*_h1`, plus `*_physics`, `*_physics_loss_residual`, `*_physics_loss_adv`, etc.
- Residual heatmaps or outputs can be logged via existing `log_output` + W&B support.

---

## 6) Tests
- `neuralop/losses/tests/test_navier_stokes_eqn_loss.py`: zero-field residual, Laplacian sanity (spectral/FD), layout invariance.
- `neuralop/training/tests/test_schedulers.py`: scheduler behavior for `none`, `linear_warmup`, `plateau`.

---

## 7) How to Run Comparisons (quick commands)
Use `--data.folder` pointing to your dataset; add `--opt.n_epochs` / `--data.n_train` overrides as needed.

- **Baseline (data-only):**
  ```bash
  python scripts/train_navier_stokes.py \
    --physics_loss.enabled False \
    --wandb.log True --wandb.name ns_baseline_128
  ```
- **Physics / fixed weight:**
  ```bash
  python scripts/train_navier_stokes.py \
    --physics_loss.enabled True \
    --physics_loss.weight_schedule none \
    --physics_loss.initial_weight 1.0 --physics_loss.max_weight 1.0 \
    --wandb.log True --wandb.name ns_phy_fixed_128
  ```
- **Physics / adaptive (linear warmup):**
  ```bash
  python scripts/train_navier_stokes.py \
    --physics_loss.enabled True \
    --physics_loss.weight_schedule linear_warmup \
    --physics_loss.initial_weight 0.1 --physics_loss.max_weight 1.0 \
    --physics_loss.warmup_epochs 50 \
    --wandb.log True --wandb.name ns_phy_adaptive_128
  ```

Cross-resolution eval: add `--data.test_resolutions "[128,1024]" --data.test_batch_sizes "[8,1]"`.  
Data-scarce example: `--data.n_train 200 --opt.n_epochs 50`.

Comparison focus: `physics_weight`, `avg_physics_loss`, `*_physics_loss_residual`, `*_l2`, `*_h1` (per resolution). Keep seeds/splits/hparams constant across runs.

---

## 8) File Map (new/modified)
- `neuralop/losses/equation_losses.py` (NavierStokesEqnLoss)
- `neuralop/training/schedulers.py` (PhysicsWeightScheduler)
- `neuralop/training/trainer.py` (physics integration)
- `config/navier_stokes_config.py` (physics config block)
- `scripts/train_navier_stokes.py` (wiring + naming)
- `docs/navier_stokes_comparison.md` (how-to run)
- `docs/navier_stokes_report.md` (this report)
- Tests: `neuralop/losses/tests/test_navier_stokes_eqn_loss.py`, `neuralop/training/tests/test_schedulers.py`
