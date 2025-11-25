"""
Spectral Loss for Neural Operator Training

This module implements a spectral loss function that penalizes errors in the frequency domain
using Fast Fourier Transform (FFT) and Fourier Magnitude Difference.
"""

import torch
import torch.nn as nn
import numpy as np


class SpectralLoss(nn.Module):
    """
    Spectral Loss: Computes loss in the frequency domain using FFT.
    
    This loss function transforms predictions and targets to the frequency domain,
    computes the magnitude difference, and returns a loss value. This is particularly
    useful for PDE problems where different frequency components have different importance.
    
    Args:
        reduction: Specifies the reduction to apply to the output.
                  Options: 'mean', 'sum', 'none'. Default: 'mean'
        normalize: If True, normalizes the magnitude difference by the total energy of the target
                   spectrum (sum of all frequency magnitudes). This uses global normalization rather
                   than per-frequency normalization, making the loss scale-invariant. Default: True
        weight_low_freq: Optional weight for low-frequency components (default: 1.0)
        weight_high_freq: Optional weight for high-frequency components (default: 1.0)
        frequency_threshold: Threshold to separate low and high frequencies (normalized, 0-1).
                            Frequencies below this are considered low. Default: 0.25
    
    Example:
        >>> loss_fn = SpectralLoss(reduction='mean', normalize=True)
        >>> pred = torch.randn(4, 1, 64, 64)  # (batch, channels, H, W)
        >>> target = torch.randn(4, 1, 64, 64)
        >>> loss = loss_fn(pred, target)
    """
    
    def __init__(
        self,
        reduction: str = 'mean',
        normalize: bool = True,
        weight_low_freq: float = 1.0,
        weight_high_freq: float = 1.0,
        frequency_threshold: float = 0.25,
    ):
        super().__init__()
        
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")
        
        self.reduction = reduction
        self.normalize = normalize
        self.weight_low_freq = weight_low_freq
        self.weight_high_freq = weight_high_freq
        self.frequency_threshold = frequency_threshold
        self.eps = 1e-10  # Small epsilon to avoid division by zero
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral loss between predictions and targets.
        
        Args:
            pred: Predicted solution. Shape: (batch, channels, *spatial_dims) or 
                  (batch, time, channels, *spatial_dims)
            target: Ground truth solution. Same shape as pred.
        
        Returns:
            Spectral loss value (scalar if reduction='mean' or 'sum', tensor if 'none')
        """
        # Handle different input shapes
        original_shape = pred.shape
        
        # Remove channel dimension if present (we'll work with 2D spatial data)
        if len(original_shape) == 4:  # (batch, channels, H, W)
            pred_2d = pred[:, 0, :, :]  # (batch, H, W)
            target_2d = target[:, 0, :, :]
        elif len(original_shape) == 5:  # (batch, time, channels, H, W)
            # Use final time step for loss computation
            pred_2d = pred[:, -1, 0, :, :]  # (batch, H, W)
            target_2d = target[:, -1, 0, :, :]
        elif len(original_shape) == 3:  # (batch, H, W) - already 2D
            pred_2d = pred
            target_2d = target
        else:
            raise ValueError(f"Unsupported input shape: {original_shape}. "
                           f"Expected (batch, channels, H, W) or (batch, time, channels, H, W)")
        
        batch_size, H, W = pred_2d.shape
        
        # Compute 2D FFT
        # fft2 returns complex tensor with shape (batch, H, W)
        fft_pred = torch.fft.fft2(pred_2d, dim=(-2, -1))
        fft_target = torch.fft.fft2(target_2d, dim=(-2, -1))
        
        # Compute magnitude (power spectrum)
        magnitude_pred = torch.abs(fft_pred)  # (batch, H, W)
        magnitude_target = torch.abs(fft_target)  # (batch, H, W)
        
        # Fourier magnitude difference
        magnitude_diff = torch.abs(magnitude_pred - magnitude_target)  # (batch, H, W)
        
        # Optionally normalize by total energy of the target image
        if self.normalize:
            # GLOBAL NORMALIZATION (not per-frequency):
            # Compute total energy: sum of ALL frequency magnitudes across the entire spectrum
            # This gives a single scalar value per batch sample: E_total = sum_{i,j} |FFT_target[i,j]|
            total_energy_target = torch.sum(magnitude_target, dim=(-2, -1), keepdim=True)  # (batch, 1, 1)
            total_energy_target_safe = total_energy_target + self.eps
            
            # Normalize magnitude difference by total energy (GLOBAL normalization):
            # Each frequency component's error is divided by the SAME total energy value
            # This is different from per-frequency normalization where each frequency would be
            # normalized by its own magnitude: |FFT_pred[i,j] - FFT_target[i,j]| / |FFT_target[i,j]|
            spectral_error = magnitude_diff / total_energy_target_safe  # (batch, H, W)
        else:
            spectral_error = magnitude_diff  # (batch, H, W)
        
        # Apply frequency-dependent weighting if specified
        if self.weight_low_freq != 1.0 or self.weight_high_freq != 1.0:
            spectral_error = self._apply_frequency_weights(spectral_error, H, W)
        
        # Apply reduction
        if self.reduction == 'mean':
            loss = torch.mean(spectral_error)
        elif self.reduction == 'sum':
            loss = torch.sum(spectral_error)
        else:  # 'none'
            loss = spectral_error
        
        return loss
    
    def _apply_frequency_weights(
        self, 
        spectral_error: torch.Tensor, 
        H: int, 
        W: int
    ) -> torch.Tensor:
        """
        Apply frequency-dependent weights to spectral error.
        
        Args:
            spectral_error: Spectral error tensor (batch, H, W)
            H: Height of spatial grid
            W: Width of spatial grid
        
        Returns:
            Weighted spectral error
        """
        # Create frequency grids
        freq_y = torch.fft.fftfreq(H, device=spectral_error.device)  # (H,)
        freq_x = torch.fft.fftfreq(W, device=spectral_error.device)  # (W,)
        
        # Create 2D frequency grid
        freq_y_2d, freq_x_2d = torch.meshgrid(freq_y, freq_x, indexing='ij')
        
        # Compute radial frequency (distance from DC component)
        radial_freq = torch.sqrt(freq_x_2d**2 + freq_y_2d**2)
        radial_freq_normalized = radial_freq / (torch.max(radial_freq) + self.eps)
        
        # Create weight mask: low frequencies get weight_low_freq, high get weight_high_freq
        weight_mask = torch.where(
            radial_freq_normalized < self.frequency_threshold,
            self.weight_low_freq,
            self.weight_high_freq
        )
        
        # Expand to match batch dimension
        weight_mask = weight_mask.unsqueeze(0)  # (1, H, W)
        
        # Apply weights
        weighted_error = spectral_error * weight_mask
        
        return weighted_error


class CombinedSpectralMSELoss(nn.Module):
    """
    Combined loss: Weighted combination of Spectral Loss and MSE Loss.
    
    This allows balancing between spatial domain accuracy (MSE) and frequency domain
    accuracy (Spectral Loss).
    
    Args:
        spectral_weight: Weight for spectral loss component (default: 0.5)
        mse_weight: Weight for MSE loss component (default: 0.5)
        normalize_losses: If True, normalize both losses to similar scales before combining (default: True)
                         This prevents one loss from dominating due to scale differences.
        spectral_loss_kwargs: Additional arguments to pass to SpectralLoss
    
    Example:
        >>> loss_fn = CombinedSpectralMSELoss(spectral_weight=0.3, mse_weight=0.7)
        >>> pred = torch.randn(4, 1, 64, 64)
        >>> target = torch.randn(4, 1, 64, 64)
        >>> loss = loss_fn(pred, target)
    """
    
    def __init__(
        self,
        spectral_weight: float = 0.5,
        mse_weight: float = 0.5,
        normalize_losses: bool = True,
        debug: bool = False,
        **spectral_loss_kwargs
    ):
        super().__init__()
        
        self.spectral_weight = spectral_weight
        self.mse_weight = mse_weight
        self.normalize_losses = normalize_losses
        self.debug = debug
        self.spectral_loss = SpectralLoss(**spectral_loss_kwargs)
        self.mse_loss = nn.MSELoss()
        
        # Track running statistics for normalization (if enabled)
        if self.normalize_losses:
            self.register_buffer('spectral_running_scale', torch.tensor(1.0))
            self.register_buffer('mse_running_scale', torch.tensor(1.0))
            self.register_buffer('update_count', torch.tensor(0))
            self.momentum = 0.1  # Exponential moving average momentum
            self.eps = 1e-8
        
        # Track epoch-level statistics for debug printing
        if self.debug:
            self.register_buffer('epoch_spectral_raw', torch.tensor(0.0))
            self.register_buffer('epoch_mse_raw', torch.tensor(0.0))
            self.register_buffer('epoch_spectral_norm', torch.tensor(0.0))
            self.register_buffer('epoch_mse_norm', torch.tensor(0.0))
            self.register_buffer('epoch_combined', torch.tensor(0.0))
            self.register_buffer('epoch_batch_count', torch.tensor(0))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined spectral and MSE loss.
        
        Args:
            pred: Predicted solution
            target: Ground truth solution
        
        Returns:
            Combined loss value
        """
        spectral = self.spectral_loss(pred, target)
        mse = self.mse_loss(pred, target)
        
        # Normalize losses to similar scales if enabled
        if self.normalize_losses:
            # Update running scale estimates using exponential moving average
            if self.training:
                with torch.no_grad():
                    self.update_count += 1
                    
                    # Use absolute value as scale estimate
                    spectral_scale = torch.abs(spectral).item()
                    mse_scale = torch.abs(mse).item()
                    
                    # Update running scales
                    if self.update_count == 1:
                        # Initialize with first values
                        self.spectral_running_scale.fill_(spectral_scale + self.eps)
                        self.mse_running_scale.fill_(mse_scale + self.eps)
                    else:
                        # Exponential moving average
                        self.spectral_running_scale = (
                            (1 - self.momentum) * self.spectral_running_scale + 
                            self.momentum * (spectral_scale + self.eps)
                        )
                        self.mse_running_scale = (
                            (1 - self.momentum) * self.mse_running_scale + 
                            self.momentum * (mse_scale + self.eps)
                        )
            
            # Normalize by running scales
            spectral_normalized = spectral / self.spectral_running_scale
            mse_normalized = mse / self.mse_running_scale
            
            combined_loss = self.spectral_weight * spectral_normalized + self.mse_weight * mse_normalized
            
            # Track for debug printing (accumulate over epoch)
            if self.debug and self.training:
                with torch.no_grad():
                    self.epoch_batch_count += 1
                    # Accumulate averages (will be printed at epoch end)
                    self.epoch_spectral_raw = (
                        (self.epoch_batch_count - 1) * self.epoch_spectral_raw + spectral.item()
                    ) / self.epoch_batch_count
                    self.epoch_mse_raw = (
                        (self.epoch_batch_count - 1) * self.epoch_mse_raw + mse.item()
                    ) / self.epoch_batch_count
                    self.epoch_spectral_norm = (
                        (self.epoch_batch_count - 1) * self.epoch_spectral_norm + spectral_normalized.item()
                    ) / self.epoch_batch_count
                    self.epoch_mse_norm = (
                        (self.epoch_batch_count - 1) * self.epoch_mse_norm + mse_normalized.item()
                    ) / self.epoch_batch_count
                    self.epoch_combined = (
                        (self.epoch_batch_count - 1) * self.epoch_combined + combined_loss.item()
                    ) / self.epoch_batch_count
        else:
            combined_loss = self.spectral_weight * spectral + self.mse_weight * mse
            
            # Track for debug printing (accumulate over epoch)
            if self.debug and self.training:
                with torch.no_grad():
                    self.epoch_batch_count += 1
                    self.epoch_spectral_raw = (
                        (self.epoch_batch_count - 1) * self.epoch_spectral_raw + spectral.item()
                    ) / self.epoch_batch_count
                    self.epoch_mse_raw = (
                        (self.epoch_batch_count - 1) * self.epoch_mse_raw + mse.item()
                    ) / self.epoch_batch_count
                    self.epoch_combined = (
                        (self.epoch_batch_count - 1) * self.epoch_combined + combined_loss.item()
                    ) / self.epoch_batch_count
        
        return combined_loss
    
    def reset_epoch_stats(self):
        """Reset epoch statistics for debug printing. Call at the start of each epoch."""
        if self.debug:
            self.epoch_spectral_raw.zero_()
            self.epoch_mse_raw.zero_()
            self.epoch_spectral_norm.zero_()
            self.epoch_mse_norm.zero_()
            self.epoch_combined.zero_()
            self.epoch_batch_count.zero_()
    
    def print_epoch_stats(self, epoch: int):
        """Print epoch statistics for debug. Call at the end of each epoch."""
        if self.debug:
            if self.normalize_losses:
                print(f"\n[Combined Loss Debug - Epoch {epoch}]")
                print(f"  Raw Spectral Loss: {self.epoch_spectral_raw.item():.6f}")
                print(f"  Raw MSE Loss: {self.epoch_mse_raw.item():.6f}")
                print(f"  Running Spectral Scale: {self.spectral_running_scale.item():.6f}")
                print(f"  Running MSE Scale: {self.mse_running_scale.item():.6f}")
                print(f"  Normalized Spectral Loss: {self.epoch_spectral_norm.item():.6f}")
                print(f"  Normalized MSE Loss: {self.epoch_mse_norm.item():.6f}")
                print(f"  Weighted Spectral: {self.spectral_weight * self.epoch_spectral_norm.item():.6f}")
                print(f"  Weighted MSE: {self.mse_weight * self.epoch_mse_norm.item():.6f}")
                print(f"  Combined Loss: {self.epoch_combined.item():.6f}")
            else:
                print(f"\n[Combined Loss Debug - Epoch {epoch}]")
                print(f"  Raw Spectral Loss: {self.epoch_spectral_raw.item():.6f}")
                print(f"  Raw MSE Loss: {self.epoch_mse_raw.item():.6f}")
                print(f"  Weighted Spectral: {self.spectral_weight * self.epoch_spectral_raw.item():.6f}")
                print(f"  Weighted MSE: {self.mse_weight * self.epoch_mse_raw.item():.6f}")
                print(f"  Combined Loss: {self.epoch_combined.item():.6f}")


if __name__ == "__main__":
    # Example usage and testing
    print("Spectral Loss Module")
    print("=" * 50)
    
    # Create example data
    batch_size, channels, H, W = 4, 1, 64, 64
    pred = torch.randn(batch_size, channels, H, W)
    target = torch.randn(batch_size, channels, H, W)
    
    # Test SpectralLoss
    print("\n1. Testing SpectralLoss (normalized):")
    loss_fn = SpectralLoss(reduction='mean', normalize=True)
    loss = loss_fn(pred, target)
    print(f"   Loss value: {loss.item():.6f}")
    
    print("\n2. Testing SpectralLoss (not normalized):")
    loss_fn2 = SpectralLoss(reduction='mean', normalize=False)
    loss2 = loss_fn2(pred, target)
    print(f"   Loss value: {loss2.item():.6f}")
    
    print("\n3. Testing SpectralLoss with frequency weighting:")
    loss_fn3 = SpectralLoss(
        reduction='mean',
        normalize=True,
        weight_low_freq=2.0,  # Emphasize low frequencies
        weight_high_freq=0.5,  # De-emphasize high frequencies
        frequency_threshold=0.25
    )
    loss3 = loss_fn3(pred, target)
    print(f"   Loss value: {loss3.item():.6f}")
    
    print("\n4. Testing CombinedSpectralMSELoss:")
    combined_loss = CombinedSpectralMSELoss(spectral_weight=0.3, mse_weight=0.7)
    loss4 = combined_loss(pred, target)
    print(f"   Combined loss value: {loss4.item():.6f}")
    
    print("\n5. Testing with multi-time-step predictions:")
    pred_multi = torch.randn(batch_size, 10, channels, H, W)  # (batch, time, channels, H, W)
    target_multi = torch.randn(batch_size, 10, channels, H, W)
    loss5 = loss_fn(pred_multi, target_multi)
    print(f"   Loss value: {loss5.item():.6f}")
    
    print("\n[OK] All tests completed!")

