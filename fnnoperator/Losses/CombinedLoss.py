"""
Combined Loss Function

A general class for combining multiple loss functions with specified weights.
Each loss function handles its own normalization internally.
"""

import torch
import torch.nn as nn
import sys
from typing import List, Union, Dict, Optional


class CombinedLoss(nn.Module):
    """
    General combined loss: Weighted combination of multiple loss functions.
    
    This class allows you to combine any number of loss functions with specified
    weights. Each loss function is responsible for its own normalization and
    calculations. The combined loss can optionally normalize the individual losses
    to similar scales before combining them.
    
    Args:
        losses: List of loss function modules (nn.Module instances that take (pred, target))
        weights: List of weights for each loss (must match length of losses).
                 If None, all losses are weighted equally (1.0 each).
        normalize_losses: If True, normalize all losses to similar scales before combining
                         (default: True). This prevents one loss from dominating due to
                         scale differences. Uses exponential moving average to track scales.
        momentum: Momentum for exponential moving average of loss scales (default: 0.1)
        debug: If True, print debug information at each epoch (default: False)
        loss_names: Optional list of names for each loss (for debug output).
                    If None, uses 'loss_0', 'loss_1', etc.
    
    Example:
        >>> from fnnoperator.Losses import MSELoss, SpectralLoss
        >>> 
        >>> mse_loss = MSELoss()
        >>> spectral_loss = SpectralLoss(normalize=True)
        >>> 
        >>> combined = CombinedLoss(
        ...     losses=[mse_loss, spectral_loss],
        ...     weights=[0.7, 0.3],
        ...     normalize_losses=True
        ... )
        >>> 
        >>> pred = torch.randn(4, 1, 64, 64)
        >>> target = torch.randn(4, 1, 64, 64)
        >>> loss = combined(pred, target)
    """
    
    def __init__(
        self,
        losses: List[nn.Module],
        weights: Optional[List[float]] = None,
        normalize_losses: bool = True,
        momentum: float = 0.1,
        debug: bool = False,
        loss_names: Optional[List[str]] = None,
    ):
        super().__init__()
        
        if not losses:
            raise ValueError("losses list cannot be empty")
        
        if not isinstance(losses, list):
            raise ValueError("losses must be a list of nn.Module instances")
        
        self.num_losses = len(losses)
        self.normalize_losses = normalize_losses
        self.momentum = momentum
        self.debug = debug
        self.eps = 1e-8
        
        # Store losses as nn.ModuleList so they're registered as submodules
        self.losses = nn.ModuleList(losses)
        
        # Set weights (default to equal weights if not provided)
        if weights is None:
            weights = [1.0] * self.num_losses
        elif len(weights) != self.num_losses:
            raise ValueError(
                f"Number of weights ({len(weights)}) must match number of losses ({self.num_losses})"
            )
        
        # Normalize weights to sum to 1.0 (optional, but good practice)
        weight_sum = sum(weights)
        if weight_sum > 0:
            self.weights = [w / weight_sum for w in weights]
        else:
            self.weights = weights
        
        # Set loss names for debug output
        if loss_names is None:
            self.loss_names = [f"loss_{i}" for i in range(self.num_losses)]
        else:
            if len(loss_names) != self.num_losses:
                raise ValueError(
                    f"Number of loss names ({len(loss_names)}) must match number of losses ({self.num_losses})"
                )
            self.loss_names = loss_names
        
        # Track running statistics for normalization (if enabled)
        if self.normalize_losses:
            self.register_buffer('running_scales', torch.ones(self.num_losses))
            self.register_buffer('update_count', torch.tensor(0))
        
        # Track epoch-level statistics for debug printing
        # Initialize as empty lists (will be reset at start of each epoch)
        self._epoch_raw_losses = []
        self._epoch_normalized_losses = []
        self._epoch_combined_losses = []
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, u_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute combined loss from all individual losses.
        
        Args:
            pred: Predicted solution
            target: Ground truth solution
            u_input: Input solution (needed for physics-informed losses). Default: None
        
        Returns:
            Combined loss value
        """
        # Compute all individual losses
        individual_losses = []
        for loss_fn in self.losses:
            # Check if this is a PhysicsInformedLoss that needs u_input
            if 'PhysicsInformedLoss' in str(type(loss_fn)) and u_input is not None:
                loss_val = loss_fn(pred, target, u_input=u_input)
            else:
                loss_val = loss_fn(pred, target)
            individual_losses.append(loss_val)
        
        # Normalize losses to similar scales if enabled
        if self.normalize_losses:
            # Update running scale estimates using exponential moving average
            if self.training:
                with torch.no_grad():
                    self.update_count += 1
                    
                    # Update running scales for each loss
                    for i, loss_val in enumerate(individual_losses):
                        loss_scale = torch.abs(loss_val).item()
                        
                        if self.update_count == 1:
                            # Initialize with first values
                            self.running_scales[i] = loss_scale + self.eps
                        else:
                            # Exponential moving average
                            self.running_scales[i] = (
                                (1 - self.momentum) * self.running_scales[i] +
                                self.momentum * (loss_scale + self.eps)
                            )
            
            # Normalize by running scales
            normalized_losses = [
                loss_val / self.running_scales[i]
                for i, loss_val in enumerate(individual_losses)
            ]
            
            # Combine normalized losses
            combined_loss = sum(
                self.weights[i] * normalized_losses[i]
                for i in range(self.num_losses)
            )
            
            # Track for debug printing (accumulate over epoch)
            if self.debug and self.training:
                with torch.no_grad():
                    self._epoch_raw_losses.append([loss.item() for loss in individual_losses])
                    self._epoch_normalized_losses.append([loss.item() for loss in normalized_losses])
                    self._epoch_combined_losses.append(combined_loss.item())
        else:
            # Combine raw losses directly
            combined_loss = sum(
                self.weights[i] * individual_losses[i]
                for i in range(self.num_losses)
            )
            
            # Track for debug printing (accumulate over epoch)
            if self.debug and self.training:
                with torch.no_grad():
                    self._epoch_raw_losses.append([loss.item() for loss in individual_losses])
                    self._epoch_combined_losses.append(combined_loss.item())
        
        return combined_loss
    
    def reset_epoch_stats(self):
        """
        Reset epoch statistics for debug printing.
        Call this at the start of each epoch.
        """
        if self.debug:
            self._epoch_raw_losses = []
            self._epoch_normalized_losses = []
            self._epoch_combined_losses = []
    
    def print_epoch_stats(self, epoch: int):
        """
        Print epoch statistics for debug.
        Call this at the end of each epoch.
        """
        if not self.debug:
            return
        
        if not self._epoch_raw_losses:
            return
        
        # Compute averages over epoch
        import numpy as np
        
        avg_raw = np.mean(self._epoch_raw_losses, axis=0)
        avg_combined = np.mean(self._epoch_combined_losses)
        
        sys.stdout.write(f"\n[Combined Loss Debug - Epoch {epoch}]\n")
        
        if self.normalize_losses:
            avg_normalized = np.mean(self._epoch_normalized_losses, axis=0)
            
            for i in range(self.num_losses):
                sys.stdout.write(f"  {self.loss_names[i]}:\n")
                sys.stdout.write(f"    Raw Loss: {avg_raw[i]:.6f}\n")
                sys.stdout.write(f"    Running Scale: {self.running_scales[i].item():.6f}\n")
                sys.stdout.write(f"    Normalized Loss: {avg_normalized[i]:.6f}\n")
                sys.stdout.write(f"    Weight: {self.weights[i]:.4f}\n")
                sys.stdout.write(f"    Weighted Contribution: {self.weights[i] * avg_normalized[i]:.6f}\n")
        else:
            for i in range(self.num_losses):
                sys.stdout.write(f"  {self.loss_names[i]}:\n")
                sys.stdout.write(f"    Raw Loss: {avg_raw[i]:.6f}\n")
                sys.stdout.write(f"    Weight: {self.weights[i]:.4f}\n")
                sys.stdout.write(f"    Weighted Contribution: {self.weights[i] * avg_raw[i]:.6f}\n")
        
        sys.stdout.write(f"  Combined Loss: {avg_combined:.6f}\n")
        sys.stdout.flush()

