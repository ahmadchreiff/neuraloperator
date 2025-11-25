"""
Mean Squared Error (MSE) Loss

A wrapper around PyTorch's nn.MSELoss for consistency with other loss classes.
"""

import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """
    Mean Squared Error Loss.
    
    This is a wrapper around PyTorch's nn.MSELoss for consistency with other
    loss classes in the package. Each loss handles its own normalization internally.
    
    Args:
        reduction: Specifies the reduction to apply to the output.
                  Options: 'mean', 'sum', 'none'. Default: 'mean'
    
    Example:
        >>> loss_fn = MSELoss(reduction='mean')
        >>> pred = torch.randn(4, 1, 64, 64)
        >>> target = torch.randn(4, 1, 64, 64)
        >>> loss = loss_fn(pred, target)
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")
        
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=reduction)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss between predictions and targets.
        
        Args:
            pred: Predicted solution. Any shape compatible with target.
            target: Ground truth solution. Same shape as pred.
        
        Returns:
            MSE loss value (scalar if reduction='mean' or 'sum', tensor if 'none')
        """
        return self.mse_loss(pred, target)


