"""
Spectral Error Analysis using FFT

This module provides functions to compute spectral errors between predicted
and ground truth solutions using Fast Fourier Transform (FFT) analysis.
"""

import numpy as np
import torch


def compute_spectral_error(u_pred, u_true, normalize=True, return_fft=False):
    """
    Compute spectral error between predicted and ground truth solutions using FFT.
    
    This function performs FFT on both predicted and ground truth solutions,
    then computes error metrics in the frequency domain. This is particularly
    useful for analyzing how well the model captures different frequency components
    of the solution, which is important for PDE problems where different scales
    of features are present.
    
    Args:
        u_pred: Predicted solution. Can be numpy array or torch.Tensor.
                Shape: (batch, *spatial_dims) or (batch, time_steps, *spatial_dims)
                For multi-time-step predictions, errors are computed per time step.
        u_true: Ground truth solution. Same type and shape as u_pred.
        normalize: If True, normalize errors by the total energy of true solution's FFT
                   (sum of all frequency magnitudes). This uses global normalization,
                   matching the SpectralLoss training method.
        return_fft: If True, also return the FFT coefficients for further analysis.
    
    Returns:
        dict: Dictionary containing:
            - 'magnitude_diff': Fourier magnitude difference per frequency
              Shape: Same as input spatial dimensions
            - 'normalized_error': Normalized per-frequency error (if normalize=True)
              Shape: Same as input spatial dimensions
            - 'fft_pred': FFT coefficients of prediction (if return_fft=True)
            - 'fft_true': FFT coefficients of ground truth (if return_fft=True)
            - 'mean_magnitude_error': Mean magnitude error across all frequencies
            - 'mean_normalized_error': Mean normalized error (if normalize=True)
            - 'relative_spectral_error': Global relative spectral error (total_diff / total_energy)
              This matches the SpectralLoss normalization method (global normalization)
    
    Example:
        >>> u_pred = np.random.randn(10, 64, 64)  # 10 samples, 64x64 grid
        >>> u_true = np.random.randn(10, 64, 64)
        >>> results = compute_spectral_error(u_pred, u_true)
        >>> print(f"Mean magnitude error: {results['mean_magnitude_error']:.6f}")
    """
    # Convert to numpy if torch tensors
    if isinstance(u_pred, torch.Tensor):
        u_pred = u_pred.detach().cpu().numpy()
    if isinstance(u_true, torch.Tensor):
        u_true = u_true.detach().cpu().numpy()
    
    # Ensure float64 for FFT precision
    u_pred = np.asarray(u_pred, dtype=np.float64)
    u_true = np.asarray(u_true, dtype=np.float64)
    
    # Handle batch dimension
    original_shape = u_pred.shape
    has_batch = len(original_shape) > 2  # Assuming at least 2D spatial data
    
    if has_batch:
        # Reshape to process all samples
        if len(original_shape) == 3:  # (batch, H, W) - 2D spatial
            u_pred_flat = u_pred.reshape(-1, original_shape[1], original_shape[2])
            u_true_flat = u_true.reshape(-1, original_shape[1], original_shape[2])
            spatial_dims = (original_shape[1], original_shape[2])
        elif len(original_shape) == 4:  # (batch, time, H, W) - 2D with time
            n_batch, n_time = original_shape[0], original_shape[1]
            u_pred_flat = u_pred.reshape(-1, original_shape[2], original_shape[3])
            u_true_flat = u_true.reshape(-1, original_shape[2], original_shape[3])
            spatial_dims = (original_shape[2], original_shape[3])
        elif len(original_shape) == 2:  # (H, W) - single 2D sample
            u_pred_flat = u_pred[np.newaxis, ...]
            u_true_flat = u_true[np.newaxis, ...]
            spatial_dims = original_shape
        else:
            raise ValueError(f"Unsupported input shape: {original_shape}")
    else:
        u_pred_flat = u_pred[np.newaxis, ...]
        u_true_flat = u_true[np.newaxis, ...]
        spatial_dims = original_shape
    
    # Compute FFT for all samples
    fft_pred = np.fft.fft2(u_pred_flat, axes=(-2, -1))
    fft_true = np.fft.fft2(u_true_flat, axes=(-2, -1))
    
    # Compute magnitude (power spectrum)
    magnitude_pred = np.abs(fft_pred)
    magnitude_true = np.abs(fft_true)
    
    # Fourier magnitude difference
    magnitude_diff = np.abs(magnitude_pred - magnitude_true)
    
    # Average over batch/time dimension
    magnitude_diff_avg = np.mean(magnitude_diff, axis=0)
    magnitude_true_avg = np.mean(magnitude_true, axis=0)
    
    # Compute total energy for global normalization (matching SpectralLoss)
    total_energy_true = np.sum(magnitude_true_avg)  # Total energy (global normalization)
    total_energy_diff = np.sum(magnitude_diff_avg)  # Total magnitude difference
    
    # Compute mean errors
    mean_magnitude_error = np.mean(magnitude_diff_avg)
    
    # Global normalization: divide by total energy (matches SpectralLoss)
    # This is consistent with how SpectralLoss normalizes during training
    eps = 1e-10
    if normalize:
        # Global normalization: normalize magnitude_diff by total energy
        # This gives a normalized error map (each frequency normalized by total energy)
        normalized_error_avg = magnitude_diff_avg / (total_energy_true + eps)
        mean_normalized_error = np.mean(normalized_error_avg)
    else:
        normalized_error_avg = None
        mean_normalized_error = None
    
    # Compute relative spectral error using GLOBAL normalization (matching SpectralLoss)
    # This matches the training loss: divide by total energy (sum of all magnitudes)
    relative_spectral_error = total_energy_diff / (total_energy_true + eps)
    
    # Prepare results
    results = {
        'magnitude_diff': magnitude_diff_avg,
        'mean_magnitude_error': mean_magnitude_error,
        'relative_spectral_error': relative_spectral_error,
    }
    
    if normalize:
        results['normalized_error'] = normalized_error_avg
        results['mean_normalized_error'] = mean_normalized_error
    
    if return_fft:
        results['fft_pred'] = fft_pred
        results['fft_true'] = fft_true
    
    return results


def compute_frequency_band_errors(u_pred, u_true, frequency_bands=None):
    """
    Compute spectral errors within specific frequency bands.
    
    This is useful for analyzing how errors vary across different scales
    (low frequencies = large scale features, high frequencies = fine details).
    
    Args:
        u_pred: Predicted solution (numpy array or torch.Tensor)
        u_true: Ground truth solution (same type and shape as u_pred)
        frequency_bands: List of tuples defining frequency bands as (low, high) in
                         normalized frequency units [0, 1]. If None, uses default bands:
                         - Low: [0, 0.25]
                         - Mid: [0.25, 0.5]
                         - High: [0.5, 1.0]
    
    Returns:
        dict: Dictionary with error statistics for each frequency band
    """
    if frequency_bands is None:
        frequency_bands = {
            'low': (0.0, 0.25),
            'mid': (0.25, 0.5),
            'high': (0.5, 1.0)
        }
    
    # Get spectral error results
    results = compute_spectral_error(u_pred, u_true, normalize=True, return_fft=True)
    
    # Get spatial dimensions
    H, W = results['magnitude_diff'].shape
    normalized_error = results['normalized_error']
    
    # Create frequency grids
    freq_y = np.fft.fftfreq(H)
    freq_x = np.fft.fftfreq(W)
    freq_y_2d, freq_x_2d = np.meshgrid(freq_x, freq_y)
    
    # Compute radial frequency (distance from DC component)
    radial_freq = np.sqrt(freq_x_2d**2 + freq_y_2d**2)
    radial_freq_normalized = radial_freq / np.max(radial_freq)
    
    # Compute errors per frequency band
    band_errors = {}
    for band_name, (low, high) in frequency_bands.items():
        mask = (radial_freq_normalized >= low) & (radial_freq_normalized < high)
        band_error = normalized_error[mask]
        band_errors[band_name] = {
            'mean_error': np.mean(band_error),
            'std_error': np.std(band_error),
            'max_error': np.max(band_error),
            'fraction_of_spectrum': np.sum(mask) / mask.size
        }
    
    results['frequency_band_errors'] = band_errors
    return results


def visualize_spectral_error(magnitude_diff, normalized_error=None, save_path=None):
    """
    Visualize spectral errors as 2D heatmaps.
    
    Args:
        magnitude_diff: Magnitude difference array (2D)
        normalized_error: Normalized error array (2D, optional)
        save_path: Path to save the figure (optional)
    
    Returns:
        matplotlib figure object
    """
    import matplotlib.pyplot as plt
    
    n_plots = 2 if normalized_error is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    
    if n_plots == 1:
        axes = [axes]
    
    # Plot magnitude difference
    im1 = axes[0].imshow(magnitude_diff, cmap='hot', origin='lower')
    axes[0].set_title('Fourier Magnitude Difference')
    axes[0].set_xlabel('Frequency (x)')
    axes[0].set_ylabel('Frequency (y)')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot normalized error if provided
    if normalized_error is not None:
        im2 = axes[1].imshow(normalized_error, cmap='hot', origin='lower')
        axes[1].set_title('Normalized Per-Frequency Error')
        axes[1].set_xlabel('Frequency (x)')
        axes[1].set_ylabel('Frequency (y)')
        plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    # Example usage
    print("Spectral Error Analysis Module")
    print("=" * 50)
    
    # Create example data
    H, W = 64, 64
    u_true = np.random.randn(10, H, W)
    u_pred = u_true + 0.1 * np.random.randn(10, H, W)  # Add some noise
    
    # Compute spectral errors
    results = compute_spectral_error(u_pred, u_true, normalize=True)
    
    print(f"\nSpectral Error Results:")
    print(f"  Mean magnitude error: {results['mean_magnitude_error']:.6f}")
    print(f"  Mean normalized error: {results['mean_normalized_error']:.6f}")
    print(f"  Magnitude diff shape: {results['magnitude_diff'].shape}")
    print(f"  Normalized error shape: {results['normalized_error'].shape}")
    
    # Frequency band analysis
    band_results = compute_frequency_band_errors(u_pred, u_true)
    print(f"\nFrequency Band Errors:")
    for band, stats in band_results['frequency_band_errors'].items():
        print(f"  {band.upper()} frequencies:")
        print(f"    Mean error: {stats['mean_error']:.6f}")
        print(f"    Std error: {stats['std_error']:.6f}")
        print(f"    Fraction of spectrum: {stats['fraction_of_spectrum']:.3f}")

