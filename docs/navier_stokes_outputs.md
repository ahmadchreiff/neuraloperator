# Navier–Stokes batch run outputs (what each file means)

All artifacts are saved under `outputs/navier_stokes_batch/` by `scripts/run_navier_stokes_batch.py`. Each file helps you compare baseline vs physics runs and pick figures for a presentation.

- `baseline_metrics.json`, `phy_fixed_metrics.json`, `phy_adaptive_metrics.json`  
  Full JSON dump per run: config used, per-epoch train/eval histories, and final metrics. Good for post-hoc analysis or custom plotting.

- `train_data_loss.png`  
  Training curve of `avg_data_loss` (data-only loss) for all runs. Shows how quickly each run fits the data.

- `train_physics_loss.png`  
  Training curve of `avg_physics_loss` (physics residual) for physics runs. Baseline is absent. Use to see how the physics constraint tightens.

- `train_physics_weight.png`  
  Curve of the physics loss weight over epochs (e.g., linear warmup for the adaptive run). Baseline stays at zero.

- `eval_128_l2.png`, `eval_128_h1.png`  
  Eval curves (per epoch) of L2 and H1 errors at 128², plotted for each run. Use to compare accuracy across runs.

- `eval_128_physics.png`, `eval_128_residual.png`  
  Eval curves of the physics loss and residual component at 128² for the physics runs. Baseline absent. Indicates how well the model satisfies the PDE at eval time.

- `eval_last_snapshot.png`  
  Bar chart of the last-eval metrics (`128_l2`, `128_h1`, `128_physics`, `128_physics_loss_residual`) across the three runs. Quick side-by-side comparison for slides.
