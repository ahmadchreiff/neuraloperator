# Navier–Stokes Code Map

Brief pointers to the Navier–Stokes pieces in this repo.

- `scripts/train_navier_stokes.py` — End-to-end training entrypoint (data load, model, optim, loop, logging).
- `config/navier_stokes_config.py` — Defaults for data, optimization, patching, and model choice (`FNO_Medium2d`).
- `config/models.py` (`FNO_Medium2d`) — Model hyperparameters: channels, modes, hidden size.
- `neuralop/data/datasets/navier_stokes.py` — Navier–Stokes dataset wrapper and loader helper (`load_navier_stokes_pt`).
- `neuralop/data/datasets/pt_dataset.py` — Base PT dataset: file I/O, subsampling, normalizer fitting, DataProcessor creation.
- `neuralop/data/transforms/data_processors.py` — `DefaultDataProcessor` (normalization/device move) and `MGPatchingDataProcessor` (multigrid patching).
- `neuralop/models/fno.py` — FNO implementation used for Navier–Stokes runs.
- `neuralop/losses/data_losses.py` — Data losses (L2, H1) currently used.
- `neuralop/losses/equation_losses.py` — Place to add a physics-informed Navier–Stokes residual loss.
- `neuralop/training/trainer.py` — Training/eval loops and metric computation.
- `docs/navier_stokes_pipeline.md` — Full pipeline walkthrough with shapes, config defaults, and extension points.
