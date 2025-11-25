"""
Physics-Informed Loss for Heat Equation

This module implements a physics-informed loss that enforces the heat equation
residual: ∂u/∂t = α∇²u, where α is the thermal diffusivity.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple


class PhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Loss: Enforces the heat equation residual.
    
    The heat equation in 2D is:
        ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
    
    The residual is:
        R = ∂u/∂t - α∇²u
    
    This loss penalizes the squared residual, encouraging the model to satisfy
    the physics of the heat equation.
    
    Args:
        diffusivity: Thermal diffusivity α (can be a scalar or tensor per sample)
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction
        dt: Time step
        reduction: Specifies the reduction to apply to the output.
                  Options: 'mean', 'sum', 'none'. Default: 'mean'
        use_input_for_laplacian: If True, compute Laplacian on input u(t).
                                If False, compute on predicted u(t+Δt). Default: True
        boundary_mask: Optional mask to exclude boundary points from loss computation.
                      Shape should match spatial dimensions. Default: None (includes all points)
    
    Example:
        >>> loss_fn = PhysicsInformedLoss(
        ...     diffusivity=0.1,
        ...     dx=0.01,
        ...     dy=0.01,
        ...     dt=0.001
        ... )
        >>> u_input = torch.randn(4, 1, 64, 64)  # u(t)
        >>> u_pred = torch.randn(4, 1, 64, 64)   # u(t+Δt)
        >>> loss = loss_fn(u_pred, u_input)
    """
    
    def __init__(
        self,
        diffusivity: Union[float, torch.Tensor],
        dx: float,
        dy: float,
        dt: float,
        reduction: str = 'mean',
        use_input_for_laplacian: bool = True,
        boundary_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")
        
        self.reduction = reduction
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.use_input_for_laplacian = use_input_for_laplacian
        self.eps = 1e-10
        
        # Handle diffusivity (can be scalar or per-sample tensor)
        if isinstance(diffusivity, (int, float)):
            self.register_buffer('diffusivity', torch.tensor(float(diffusivity)))
            self.diffusivity_is_tensor = False
        else:
            self.register_buffer('diffusivity', diffusivity)
            self.diffusivity_is_tensor = True
        
        # Boundary mask (if provided)
        if boundary_mask is not None:
            self.register_buffer('boundary_mask', boundary_mask)
        else:
            self.boundary_mask = None
    
    def compute_laplacian_2d(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D Laplacian using finite differences.
        
        ∇²u = ∂²u/∂x² + ∂²u/∂y²
        
        Uses central differences:
        - ∂²u/∂x² ≈ (u[i, j+1] - 2u[i, j] + u[i, j-1]) / dx²
        - ∂²u/∂y² ≈ (u[i+1, j] - 2u[i, j] + u[i-1, j]) / dy²
        
        Args:
            u: Input tensor of shape (batch, channels, H, W) or (batch, H, W)
        
        Returns:
            Laplacian of shape (batch, channels, H, W) or (batch, H, W)
        """
        # Handle different input shapes
        if len(u.shape) == 4:  # (batch, channels, H, W)
            u_2d = u[:, 0, :, :]  # Extract first channel
            has_channels = True
        elif len(u.shape) == 3:  # (batch, H, W)
            u_2d = u
            has_channels = False
        else:
            raise ValueError(f"Unsupported input shape: {u.shape}. Expected (batch, H, W) or (batch, channels, H, W)")
        
        batch_size, H, W = u_2d.shape
        
        # Initialize Laplacian
        laplacian = torch.zeros_like(u_2d)
        
        # Compute ∂²u/∂x² using central differences
        # u[:, :, j+1] - 2u[:, :, j] + u[:, :, j-1]
        d2u_dx2 = torch.zeros_like(u_2d)
        d2u_dx2[:, :, 1:-1] = (
            u_2d[:, :, 2:] - 2 * u_2d[:, :, 1:-1] + u_2d[:, :, :-2]
        ) / (self.dx ** 2)
        
        # Compute ∂²u/∂y² using central differences
        # u[:, i+1, :] - 2u[:, i, :] + u[:, i-1, :]
        d2u_dy2 = torch.zeros_like(u_2d)
        d2u_dy2[:, 1:-1, :] = (
            u_2d[:, 2:, :] - 2 * u_2d[:, 1:-1, :] + u_2d[:, :-2, :]
        ) / (self.dy ** 2)
        
        # Laplacian = ∂²u/∂x² + ∂²u/∂y²
        laplacian = d2u_dx2 + d2u_dy2
        
        # Add channel dimension back if needed
        if has_channels:
            laplacian = laplacian.unsqueeze(1)  # (batch, 1, H, W)
        
        return laplacian
    
    def compute_temporal_derivative(
        self,
        u_pred: torch.Tensor,
        u_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute temporal derivative using finite differences.
        
        ∂u/∂t ≈ (u(t+Δt) - u(t)) / Δt
        
        Args:
            u_pred: Predicted solution at t+Δt
            u_input: Input solution at t
        
        Returns:
            Temporal derivative
        """
        return (u_pred - u_input) / (self.dt + self.eps)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        u_input: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute physics-informed loss.
        
        Args:
            pred: Predicted solution. Shape: 
                  - (batch, channels, H, W) for single time step
                  - (batch, n_time_steps, channels, H, W) for multiple time steps
            target: Ground truth solution. Shape: same as pred.
                   Note: This is used for compatibility with other losses, but not directly
                   used in the residual computation.
            u_input: Input solution u(t). Shape: (batch, channels, H, W) or (batch, H, W).
                    If None, assumes pred and target represent consecutive time steps.
                    Default: None
        
        Returns:
            Physics-informed loss value (scalar if reduction='mean' or 'sum', tensor if 'none')
        """
        # If u_input is not provided, assume we're computing residual from consecutive predictions
        # This is a fallback - ideally u_input should be provided
        if u_input is None:
            # For operator learning, we typically have u(t) as input and predict u(t+Δt)
            # If u_input is not provided, we can't compute the temporal derivative properly
            # In this case, we'll use a simplified version that assumes we have the full trajectory
            # For now, raise an error to make it explicit
            raise ValueError(
                "u_input must be provided for physics-informed loss. "
                "This should be the input solution u(t) at the previous time step."
            )
        
        # Handle multi-time-step predictions: (batch, n_time_steps, channels, H, W)
        if len(pred.shape) == 5:
            # Multiple time steps: compute physics loss for each time step and average
            n_time_steps = pred.shape[1]
            residuals = []
            
            # For first time step: use input u(t) and first prediction u(t+Δt)
            pred_first = pred[:, 0, ...]  # (batch, channels, H, W)
            du_dt_first = self.compute_temporal_derivative(pred_first, u_input)
            
            # Compute Laplacian
            if self.use_input_for_laplacian:
                laplacian_u = self.compute_laplacian_2d(u_input)
            else:
                laplacian_u = self.compute_laplacian_2d(pred_first)
            
            # Get diffusivity
            if self.diffusivity_is_tensor:
                alpha = self.diffusivity.view(-1, 1, 1, 1)
                if len(laplacian_u.shape) == 3:
                    alpha = alpha.squeeze(1)
            else:
                alpha = self.diffusivity
            
            # Compute residual for first time step
            residual_first = du_dt_first - alpha * laplacian_u
            residuals.append(residual_first)
            
            # For subsequent time steps: use previous prediction and current prediction
            for t in range(1, n_time_steps):
                pred_prev = pred[:, t-1, ...]  # (batch, channels, H, W)
                pred_curr = pred[:, t, ...]     # (batch, channels, H, W)
                du_dt_curr = self.compute_temporal_derivative(pred_curr, pred_prev)
                
                # Compute Laplacian
                if self.use_input_for_laplacian:
                    laplacian_u = self.compute_laplacian_2d(pred_prev)
                else:
                    laplacian_u = self.compute_laplacian_2d(pred_curr)
                
                # Compute residual
                residual_curr = du_dt_curr - alpha * laplacian_u
                residuals.append(residual_curr)
            
            # Stack residuals: (batch, n_time_steps, channels, H, W) or (batch, n_time_steps, H, W)
            residual = torch.stack(residuals, dim=1)
        else:
            # Single time step: standard computation
            # Compute temporal derivative: ∂u/∂t ≈ (u_pred - u_input) / dt
            du_dt = self.compute_temporal_derivative(pred, u_input)
            
            # Compute Laplacian on either input or prediction
            if self.use_input_for_laplacian:
                laplacian_u = self.compute_laplacian_2d(u_input)
            else:
                laplacian_u = self.compute_laplacian_2d(pred)
            
            # Get diffusivity (handle both scalar and per-sample cases)
            if self.diffusivity_is_tensor:
                # If diffusivity is a tensor, it should match batch size
                # Expand to match spatial dimensions
                alpha = self.diffusivity.view(-1, 1, 1, 1)  # (batch, 1, 1, 1)
                if len(pred.shape) == 3:  # (batch, H, W)
                    alpha = alpha.squeeze(1)  # (batch, 1, 1)
            else:
                alpha = self.diffusivity
            
            # Compute residual: R = ∂u/∂t - α∇²u
            residual = du_dt - alpha * laplacian_u
        
        # Apply boundary mask if provided
        if self.boundary_mask is not None:
            residual = residual * self.boundary_mask
        
        # Compute squared residual
        residual_squared = residual ** 2
        
        # Apply reduction
        if self.reduction == 'mean':
            loss = torch.mean(residual_squared)
        elif self.reduction == 'sum':
            loss = torch.sum(residual_squared)
        else:  # 'none'
            loss = residual_squared
        
        return loss


# Note: For operator learning, we need to modify the trainer to pass u_input
# to the loss function. This will be handled in the training script.

