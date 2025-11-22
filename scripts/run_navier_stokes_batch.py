"""
Run baseline, physics/fixed, and physics/adaptive Navier-Stokes experiments sequentially
and save metrics/plots for presentation (Colab-friendly).

Example (Colab):
    !git clone <your repo url>
    %cd neuraloperator
    %pip install -q -e .
    !python scripts/run_navier_stokes_batch.py --data-root /root/data/navier_stokes --demo
Switch to --full-run for longer training once the setup is verified.
"""

import argparse
import json
import pathlib
import random
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from config.navier_stokes_config import Default
from neuralop import H1Loss, LpLoss, get_model
from neuralop.data.datasets.navier_stokes import load_navier_stokes_pt
from neuralop.data.transforms.data_processors import MGPatchingDataProcessor
from neuralop.losses.equation_losses import NavierStokesEqnLoss
from neuralop.training import AdamW, PhysicsWeightScheduler
from neuralop.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Navier-Stokes runs (baseline vs physics)")
    parser.add_argument(
        "--data-root",
        type=pathlib.Path,
        default=pathlib.Path("~/data/navier_stokes").expanduser(),
        help="Path to Navier-Stokes dataset folder",
    )
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=pathlib.Path("outputs/navier_stokes_batch"),
        help="Directory to store metrics and plots",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use fast demo settings (few epochs, small train set). Default if neither flag is set.",
    )
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="Use longer training (overrides --demo).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs (otherwise picked from demo/full defaults).",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=None,
        help="Override number of training samples.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override train batch size.",
    )
    parser.add_argument(
        "--test-res",
        type=int,
        default=128,
        help="Evaluation resolution (default 128).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of training data to hold out for validation (0 disables).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def build_scheduler(cfg, optimizer):
    if cfg.opt.scheduler == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=cfg.opt.gamma, patience=cfg.opt.scheduler_patience, mode="min"
        )
    if cfg.opt.scheduler == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.opt.scheduler_T_max)
    if cfg.opt.scheduler == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.opt.step_size, gamma=cfg.opt.gamma)
    raise ValueError(f"Unknown scheduler {cfg.opt.scheduler}")


def _to_float(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().item()
    if isinstance(x, (int, float)):
        return float(x)
    return x


class LoggingTrainer(Trainer):
    """Trainer that records per-epoch train/eval metrics in memory."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_history: List[Dict[str, Any]] = []
        self.eval_history: List[Dict[str, Any]] = []

    def train_one_epoch(self, epoch, train_loader, training_loss):
        train_err, avg_loss, avg_lasso_loss, epoch_train_time = super().train_one_epoch(
            epoch, train_loader, training_loss
        )
        record = {
            "epoch": epoch,
            "train_err": _to_float(train_err),
            "avg_loss": _to_float(avg_loss),
            "avg_lasso_loss": _to_float(avg_lasso_loss) if avg_lasso_loss is not None else None,
            "time": _to_float(epoch_train_time),
        }
        for k, v in self.latest_train_metrics.items():
            if isinstance(v, dict):
                record[k] = {kk: _to_float(vv) for kk, vv in v.items()}
            else:
                record[k] = _to_float(v)
        self.train_history.append(record)
        return train_err, avg_loss, avg_lasso_loss, epoch_train_time

    def evaluate_all(self, epoch, eval_losses, test_loaders, eval_modes, max_autoregressive_steps=None):
        metrics = super().evaluate_all(epoch, eval_losses, test_loaders, eval_modes, max_autoregressive_steps)
        if epoch is not None:
            record = {"epoch": epoch}
            for k, v in metrics.items():
                record[k] = _to_float(v)
            self.eval_history.append(record)
        return metrics


def run_experiment(run_name: str, physics_cfg: Dict[str, Any], cfg: Default, device, data_root: pathlib.Path):
    if not data_root.exists():
        raise FileNotFoundError(
            f"DATA_ROOT {data_root} not found. Copy/mount your Navier-Stokes dataset there or pass --data-root."
        )

    cfg = cfg.copy() if hasattr(cfg, "copy") else cfg  # defensive
    cfg.wandb.log = False
    cfg.verbose = True
    cfg.distributed.use_distributed = False
    cfg.patching.levels = 0

    cfg.data.folder = str(data_root)
    cfg.data.train_resolution = cfg.data.test_resolutions[0]

    cfg.physics_loss.enabled = physics_cfg.get("enabled", False)
    cfg.physics_loss.weight_schedule = physics_cfg.get("weight_schedule", "none")
    cfg.physics_loss.initial_weight = physics_cfg.get("initial_weight", 0.0)
    cfg.physics_loss.max_weight = physics_cfg.get("max_weight", 1.0)
    cfg.physics_loss.warmup_epochs = min(cfg.opt.n_epochs, physics_cfg.get("warmup_epochs", cfg.physics_loss.warmup_epochs))
    cfg.physics_loss.denormalize = physics_cfg.get("denormalize", cfg.physics_loss.denormalize)
    cfg.physics_loss.derivative_mode = physics_cfg.get("derivative_mode", cfg.physics_loss.derivative_mode)

    train_loader, test_loaders, data_processor = load_navier_stokes_pt(
        data_root=data_root,
        train_resolution=cfg.data.train_resolution,
        n_train=cfg.data.n_train,
        batch_size=cfg.data.batch_size,
        test_resolutions=cfg.data.test_resolutions,
        n_tests=cfg.data.n_tests,
        test_batch_sizes=cfg.data.test_batch_sizes,
        encode_input=cfg.data.encode_input,
        encode_output=cfg.data.encode_output,
        num_workers=2,
    )

    # Optional train/validation split
    val_fraction = getattr(cfg, "val_fraction", 0.0)
    if val_fraction and val_fraction > 0 and val_fraction < 1:
        total = len(train_loader.dataset)
        if total > 1:
            val_size = max(1, int(val_fraction * total))
            val_size = min(val_size, total - 1)
            train_size = total - val_size
            train_subset, val_subset = random_split(
                train_loader.dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(getattr(cfg, "seed", 0)),
            )
            train_loader = DataLoader(
                train_subset,
                batch_size=cfg.data.batch_size,
                shuffle=True,
                num_workers=train_loader.num_workers,
                pin_memory=False,
                persistent_workers=False,
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=cfg.data.batch_size,
                shuffle=False,
                num_workers=train_loader.num_workers,
                pin_memory=False,
                persistent_workers=False,
            )
            # Add validation loader under a dedicated key
            test_loaders = dict(test_loaders)
            test_loaders["val"] = val_loader

    # get_model expects mapping-style config; convert if available
    model = get_model(cfg.to_dict() if hasattr(cfg, "to_dict") else cfg).to(device)
    if cfg.patching.levels > 0:
        data_processor = MGPatchingDataProcessor(
            model=model,
            in_normalizer=data_processor.in_normalizer,
            out_normalizer=data_processor.out_normalizer,
            padding_fraction=cfg.patching.padding,
            stitching=cfg.patching.stitching,
            levels=cfg.patching.levels,
            use_distributed=False,
        )
    data_processor = data_processor.to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = build_scheduler(cfg, optimizer)

    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)
    train_loss = h1loss if cfg.opt.training_loss == "h1" else l2loss
    eval_losses = {"h1": h1loss, "l2": l2loss}

    phy_loss_fn = None
    phy_scheduler = None
    physics_weight = cfg.physics_loss.initial_weight
    phy_normalizer_train = None
    phy_normalizer_eval = None
    if cfg.physics_loss.enabled:
        phy_loss_fn = NavierStokesEqnLoss(
            viscosity=cfg.physics_loss.viscosity,
            dx=cfg.physics_loss.dx,
            dy=cfg.physics_loss.dy,
            dt=cfg.physics_loss.dt,
            advection_weight=cfg.physics_loss.advection_weight,
            diffusion_weight=cfg.physics_loss.diffusion_weight,
            forcing_weight=cfg.physics_loss.forcing_weight,
            use_vorticity_form=cfg.physics_loss.use_vorticity_form,
            derivative_mode=cfg.physics_loss.derivative_mode,
            denormalize=cfg.physics_loss.denormalize,
        )
        if cfg.physics_loss.weight_schedule in ["linear_warmup", "plateau"]:
            phy_scheduler = PhysicsWeightScheduler(
                initial_weight=cfg.physics_loss.initial_weight,
                max_weight=cfg.physics_loss.max_weight,
                warmup_epochs=cfg.physics_loss.warmup_epochs,
                mode=cfg.physics_loss.weight_schedule,
            )
        if cfg.physics_loss.denormalize and not isinstance(data_processor, MGPatchingDataProcessor):
            phy_normalizer_train = getattr(data_processor, "out_normalizer", None)
        phy_normalizer_eval = None if not isinstance(data_processor, MGPatchingDataProcessor) else phy_normalizer_train

    trainer = LoggingTrainer(
        model=model,
        n_epochs=cfg.opt.n_epochs,
        data_processor=data_processor,
        device=device,
        mixed_precision=cfg.opt.mixed_precision,
        eval_interval=cfg.opt.eval_interval,
        log_output=False,
        use_distributed=False,
        verbose=True,
        wandb_log=False,
    )

    final_metrics = trainer.train(
        train_loader,
        test_loaders,
        optimizer,
        scheduler,
        regularizer=False,
        training_loss=train_loss,
        eval_losses=eval_losses,
        physics_loss_fn=phy_loss_fn,
        physics_weight=physics_weight,
        physics_weight_scheduler=phy_scheduler,
        physics_normalizer_train=phy_normalizer_train,
        physics_normalizer_eval=phy_normalizer_eval,
        physics_return_components=True,
        eval_physics_loss=bool(phy_loss_fn),
    )

    result = {
        "name": run_name,
        "config": cfg.to_dict() if hasattr(cfg, "to_dict") else cfg.__dict__,
        "train_history": trainer.train_history,
        "eval_history": trainer.eval_history,
        "final_metrics": {k: _to_float(v) for k, v in final_metrics.items()},
    }
    torch.cuda.empty_cache()
    return result


def plot_train_metric(results, key, ylabel, out_path: pathlib.Path):
    plt.figure(figsize=(7, 4))
    plotted = False
    for r in results:
        df = pd.DataFrame(r["train_history"])
        if key in df:
            plt.plot(df["epoch"], df[key], label=r["name"])
            plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_eval_metric(results, key, ylabel, out_path: pathlib.Path):
    plt.figure(figsize=(7, 4))
    plotted = False
    for r in results:
        df = pd.DataFrame(r["eval_history"])
        if key in df:
            plt.plot(df["epoch"], df[key], label=r["name"])
            plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_last_eval_bar(results, keys, out_path: pathlib.Path):
    rows = []
    for r in results:
        if not r["eval_history"]:
            continue
        last = r["eval_history"][-1]
        rows.append({"run": r["name"], **{k: last.get(k, None) for k in keys}})
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    idx = range(len(df))
    bar_width = 0.8 / max(1, len(keys))
    for i, k in enumerate(keys):
        ax.bar([j + i * bar_width for j in idx], df[k], width=bar_width, label=k)
    ax.set_xticks([j + bar_width * (len(keys) - 1) / 2 for j in idx])
    ax.set_xticklabels(df["run"].tolist())
    ax.set_ylabel("metric value")
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Defaults for demo/full
    if args.full_run:
        n_epochs_default = 50  # raise to 600 if you have time/GPU
        n_train_default = 2000
        batch_default = 8
        test_batch_default = 8
    else:
        n_epochs_default = 3
        n_train_default = 256
        batch_default = 4
        test_batch_default = 4

    n_epochs = args.epochs if args.epochs is not None else n_epochs_default
    n_train = args.n_train if args.n_train is not None else n_train_default
    batch_size = args.batch_size if args.batch_size is not None else batch_default
    test_batch_size = test_batch_default if args.batch_size is None else max(1, args.batch_size // 2)

    output_root: pathlib.Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    cfg = Default()
    cfg.data.n_train = n_train
    cfg.data.batch_size = batch_size
    cfg.data.test_resolutions = [args.test_res]
    cfg.data.test_batch_sizes = [test_batch_size]
    cfg.data.n_tests = [min(128, cfg.data.n_tests[0])]
    cfg.val_fraction = max(0.0, min(0.9, args.val_fraction))
    cfg.seed = args.seed
    cfg.opt.n_epochs = n_epochs
    cfg.opt.eval_interval = 1
    cfg.opt.mixed_precision = False
    cfg.verbose = True

    experiments = [
        {
            "name": "baseline",
            "physics": {"enabled": False, "weight_schedule": "none", "initial_weight": 0.0, "max_weight": 0.0},
        },
        {
            "name": "phy_fixed",
            "physics": {"enabled": True, "weight_schedule": "none", "initial_weight": 1.0, "max_weight": 1.0},
        },
        {
            "name": "phy_adaptive",
            "physics": {
                "enabled": True,
                "weight_schedule": "linear_warmup",
                "initial_weight": 0.1,
                "max_weight": 1.0,
                "warmup_epochs": max(1, n_epochs // 2),
            },
        },
    ]

    all_results = []
    for exp in experiments:
        print(f"\n=== Running {exp['name']} ({n_epochs} epochs, n_train={n_train}, batch={batch_size}) ===")
        res = run_experiment(exp["name"], exp["physics"], cfg, device, args.data_root)
        all_results.append(res)
        print(f"Finished {exp['name']}, last metrics: {res['final_metrics']}")

    # Save metrics
    for r in all_results:
        fname = output_root / f"{r['name']}_metrics.json"
        with open(fname, "w") as f:
            json.dump(r, f, indent=2)
        print(f"Saved metrics: {fname}")

    # Plots
    plot_train_metric(all_results, "avg_data_loss", "avg_data_loss (train)", output_root / "train_data_loss.png")
    plot_train_metric(all_results, "avg_physics_loss", "avg_physics_loss (train)", output_root / "train_physics_loss.png")
    plot_train_metric(all_results, "physics_weight", "physics weight", output_root / "train_physics_weight.png")

    plot_eval_metric(all_results, f"{args.test_res}_l2", f"{args.test_res}_l2 (eval)", output_root / f"eval_{args.test_res}_l2.png")
    plot_eval_metric(all_results, f"{args.test_res}_h1", f"{args.test_res}_h1 (eval)", output_root / f"eval_{args.test_res}_h1.png")
    plot_eval_metric(
        all_results,
        f"{args.test_res}_physics",
        f"{args.test_res}_physics loss",
        output_root / f"eval_{args.test_res}_physics.png",
    )
    plot_eval_metric(
        all_results,
        f"{args.test_res}_physics_loss_residual",
        f"{args.test_res} residual loss",
        output_root / f"eval_{args.test_res}_residual.png",
    )
    # Validation curves (if present)
    plot_eval_metric(all_results, "val_l2", "val_l2 (eval)", output_root / "eval_val_l2.png")
    plot_eval_metric(all_results, "val_h1", "val_h1 (eval)", output_root / "eval_val_h1.png")
    plot_eval_metric(all_results, "val_physics", "val physics loss", output_root / "eval_val_physics.png")
    plot_eval_metric(all_results, "val_physics_loss_residual", "val residual loss", output_root / "eval_val_residual.png")

    plot_last_eval_bar(
        all_results,
        [f"{args.test_res}_l2", f"{args.test_res}_h1", f"{args.test_res}_physics", f"{args.test_res}_physics_loss_residual"],
        output_root / f"eval_last_snapshot.png",
    )

    print("\nDone. Artifacts saved under", output_root.resolve())
    for p in sorted(output_root.glob("*")):
        print("-", p)


if __name__ == "__main__":
    main()
