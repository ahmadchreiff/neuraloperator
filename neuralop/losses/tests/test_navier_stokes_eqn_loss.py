import math

import pytest
import torch
from torch.testing import assert_close

from neuralop.losses.equation_losses import NavierStokesEqnLoss


def test_zero_velocity_residual_is_zero():
    loss_fn = NavierStokesEqnLoss(viscosity=0.01, derivative_mode="finite_diff")
    preds = torch.zeros(2, 3, 2, 8, 8)  # (B, T, C, H, W)

    loss, components = loss_fn(preds, inputs={}, return_components=True)

    assert_close(loss, torch.tensor(0.0, dtype=preds.dtype))
    for value in components.values():
        assert_close(value, torch.tensor(0.0, dtype=preds.dtype))


@pytest.mark.parametrize("diff_mode", ["spectral", "finite_diff"])
@torch.no_grad()
def test_laplacian_matches_sine(diff_mode: str):
    H = W = 32
    domain = 1.0
    dx = domain / H
    loss_fn = NavierStokesEqnLoss(
        viscosity=0.1, dx=dx, dy=dx, derivative_mode=diff_mode
    )

    x = torch.arange(H, dtype=torch.float32) * dx
    y = torch.arange(W, dtype=torch.float32) * dx
    X, Y = torch.meshgrid(x, y, indexing="ij")
    omega = torch.sin(2 * math.pi * X / domain) * torch.sin(2 * math.pi * Y / domain)
    omega = omega.unsqueeze(0).unsqueeze(0)  # (B, T, H, W)

    lap_omega = loss_fn._laplacian(omega)
    expected = -2 * (2 * math.pi / domain) ** 2 * omega
    tol = 1e-3 if diff_mode == "spectral" else 5e-2
    assert_close(lap_omega, expected, rtol=tol, atol=tol)


def test_time_channel_permutation_invariance():
    loss_fn = NavierStokesEqnLoss(viscosity=0.01, derivative_mode="finite_diff")
    base = torch.randn(1, 2, 2, 6, 6)  # (B, T, C, H, W)
    channel_first = base.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

    loss_a, _ = loss_fn(base, inputs={}, return_components=True)
    loss_b, _ = loss_fn(
        channel_first, inputs={"channel_first": True}, return_components=True
    )

    assert_close(loss_a, loss_b, rtol=1e-5, atol=1e-5)
