"""
Loss Functions Package

This package contains loss functions for neural operator training, including
spectral loss functions that operate in the frequency domain, MSE loss, and
a general combined loss class for combining multiple losses.
"""

from .MSELoss import MSELoss
from .SpectralLoss import SpectralLoss, CombinedSpectralMSELoss
from .CombinedLoss import CombinedLoss
from .PhysicsInformedLoss import PhysicsInformedLoss

__all__ = [
    'MSELoss',
    'SpectralLoss',
    'CombinedSpectralMSELoss',  # Kept for backward compatibility
    'CombinedLoss',  # General combined loss class
    'PhysicsInformedLoss',
]

