from __future__ import annotations

from typing import Dict, Optional


class PhysicsWeightScheduler:
    """
    Lightweight scheduler for physics loss weighting.
    Supports fixed weight, linear warmup, and a simple plateau trigger.
    """

    def __init__(
        self,
        initial_weight: float = 0.0,
        max_weight: float = 1.0,
        warmup_epochs: int = 0,
        mode: str = "none",
        plateau_patience: int = 5,
        plateau_threshold: float = 1e-4,
    ):
        self.initial_weight = initial_weight
        self.max_weight = max_weight
        self.warmup_epochs = max(0, warmup_epochs)
        self.mode = mode
        self.plateau_patience = plateau_patience
        self.plateau_threshold = plateau_threshold

    def step(self, epoch: int, history: Optional[Dict[str, list]] = None) -> float:
        """
        Update physics weight for the given epoch.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.
        history : dict, optional
            Dictionary of past losses, expects key 'data_loss' for plateau mode.

        Returns
        -------
        float
            The physics weight to use for this epoch.
        """
        if self.mode == "none":
            return self.initial_weight

        if self.mode == "linear_warmup":
            if self.warmup_epochs <= 0:
                return self.max_weight
            frac = min(max(epoch, 0) / float(self.warmup_epochs), 1.0)
            return self.initial_weight + (self.max_weight - self.initial_weight) * frac

        if self.mode == "plateau":
            weight = self.max_weight
            if self.warmup_epochs > 0:
                frac = min(max(epoch, 0) / float(self.warmup_epochs), 1.0)
                weight = self.initial_weight + (self.max_weight - self.initial_weight) * frac
            if history is not None and "data_loss" in history:
                recent = history["data_loss"][-self.plateau_patience :]
                if len(recent) == self.plateau_patience:
                    span = max(recent) - min(recent)
                    if span > self.plateau_threshold * (abs(recent[0]) + 1e-8):
                        return weight
            return self.max_weight

        raise ValueError(f"Unknown physics weighting mode '{self.mode}'")
