import torch
import torch.nn as nn
import numpy as np
from fnnoperator.models.FNN import FNN


class FNNOperator(FNN):
    def __init__(self, in_channels, out_channels, grid_shape, width = 256, depth = 3, n_time_steps = 1):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            grid_shape: Shape of the spatial grid
            width: Width of hidden layers
            depth: Depth of the network
            n_time_steps: Number of time steps to predict (default: 1 for final state only)
        """
        # If n_time_steps > 1, output all time steps: (batch, nt, out_channels, *grid)
        output_dim = out_channels * np.prod(grid_shape) * n_time_steps
        super().__init__(in_channels * np.prod(grid_shape), output_dim, width, depth)

        # Store the input and output channels, width, and depth.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.depth = depth
        self.grid_shape = grid_shape
        self.n_time_steps = n_time_steps
        
    def forward(self, x):
        """
        Forward pass of the FNNOperator.

        Args:
            x: (batch, in_channels, *grid)
        
        Returns:
            If n_time_steps == 1: (batch, out_channels, *grid) - final state only
            If n_time_steps > 1: (batch, n_time_steps, out_channels, *grid) - all time steps
        """

        # Flatten the input tensor
        batch = x.shape[0]
        x = x.reshape(batch, -1)

        # Apply the FNN
        out = super().forward(x)

        # Unflatten the output tensor
        if self.n_time_steps == 1:
            # Single time step: (batch, out_channels, *grid)
            return out.reshape(batch, self.out_channels, *self.grid_shape)
        else:
            # Multiple time steps: (batch, n_time_steps, out_channels, *grid)
            return out.reshape(batch, self.n_time_steps, self.out_channels, *self.grid_shape)

