"""
Error Analysis Module

This module provides tools for analyzing prediction errors in the frequency domain
using Fast Fourier Transform (FFT) analysis.
"""

from .spectral_error import (
    compute_spectral_error,
    compute_frequency_band_errors,
    visualize_spectral_error
)

__all__ = [
    'compute_spectral_error',
    'compute_frequency_band_errors',
    'visualize_spectral_error'
]


