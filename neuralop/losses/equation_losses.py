import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.autograd import grad

from .differentiation import FiniteDiff

# Set warning filter to show each warning only once
import warnings

warnings.filterwarnings("once", category=UserWarning)


class BurgersEqnLoss(object):
    """
    Computes loss for Burgers' equation.
    """

    def __init__(self, visc=0.01, method="fdm", loss=F.mse_loss, domain_length=1.0):
        super().__init__()
        self.visc = visc
        self.method = method
        self.loss = loss
        self.domain_length = domain_length
        if not isinstance(self.domain_length, (tuple, list)):
            self.domain_length = [self.domain_length] * 2

    def fdm(self, u):
        # remove extra channel dimensions
        u = u.squeeze(1)

        # shapes
        _, nt, nx = u.shape

        # we assume that the input is given on a regular grid
        dt = self.domain_length[0] / (nt - 1)
        dx = self.domain_length[1] / nx

        # Get derivatives, du/dt and du/dx and d^2u/dxx
        fd2d = FiniteDiff(dim=2, h=(dt, dx), periodic_in_x=False, periodic_in_y=False)
        dudt, dudx = fd2d.dx(u), fd2d.dy(u)
        dudxx = fd2d.dy(u, order=2)

        # right hand side
        right_hand_side = -dudx * u + self.visc * dudxx

        # compute the loss of the left and right hand sides of Burgers' equation
        return self.loss(dudt, right_hand_side)

    def __call__(self, y_pred, **kwargs):
        if kwargs:
            warnings.warn(
                f"BurgersLoss.__call__() received unexpected keyword arguments: {list(kwargs.keys())}. "
                "These arguments will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        if self.method == "fdm":
            return self.fdm(u=y_pred)
        raise NotImplementedError()


class ICLoss(object):
    """
    Computes loss for initial value problems.

    Extracts the initial condition and computes the loss between predicted 
    and true initial conditions for all channels.
    
    Expected input shape: (batch_size, channels, time_dim, *spatial_dims)
    """

    def __init__(self, loss=F.mse_loss):
        super().__init__()
        self.loss = loss
        
    def __call__(self, y_pred, y, **kwargs):
        """Expected input shape: (batch_size, channels, time_dim, *spatial_dims)"""
        if kwargs:
            warnings.warn(
                f"ICLoss.__call__() received unexpected keyword arguments: {list(kwargs.keys())}. "
                "These arguments will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        return self.loss(y_pred[:, :, 0], y[:, :, 0])


class PoissonInteriorLoss(object):
    """
    PoissonInteriorLoss computes the loss on the interior points of model outputs
    according to Poisson's equation in 2d: ∇·((1 + 0.1u^2)∇u(x)) = f(x)

    Parameters
    ----------
    method : Literal['autograd'] only (for now)
        How to compute derivatives for equation loss.

        * If 'autograd', differentiates using torch.autograd.grad. This can be used with outputs with any irregular
        point cloud structure.
    loss: Callable, optional
        Base loss class to compute distances between expected and true values,
        by default torch.nn.functional.mse_loss
    """

    def __init__(self, method="autograd", loss=F.mse_loss):
        super().__init__()
        self.method = method
        self.loss = loss
    
    def autograd(self, u, output_queries, output_source_terms_domain, num_boundary, **kwargs):
        """
        Compute loss between the left-hand side and right-hand side of
        nonlinear Poisson's equation: ∇·((1 + 0.1u^2)∇u(x)) = f(x)

        u: torch.Tensor | dict
            output of the model.

            * If output_queries is passed to the model as a dict, this will be a
            dict of outputs provided over the points at each value in output_queries.
            Each tensor will be shape (batch, n_points, 2).

            * If a tensor, u will be of shape (batch, num_boundary + num_interior, 2), where
            u[:, 0:num_boundary, :] are boundary points and u[:, num_boundary:, :] are interior points.
        output_queries: torch.Tensor | dict
            output queries provided to the model. If provided as a dict of tensors,
            u will also be returned as a dict keyed the same way. If provided as a tensor,
            u will be a tensor of the same shape except for number of channels. If a tensor,
            output_queries[:, 0:num_boundary, :] are boundary points and output_queries[:, num_boundary:, :]
            are interior points.
        output_source_terms_domain: torch.Tensor
            source terms f(x) defined for this specific instance of Poisson's equation.

        """

        if isinstance(output_queries, dict):
            output_queries_domain = output_queries["domain"]
            u_prime = grad(
                outputs=u.sum(),
                inputs=output_queries_domain,
                create_graph=True,
                retain_graph=True,
            )[0]
        else:
            # We only care about U defined over the interior. Grab it now if the entire U is passed.
            output_queries_domain = None
            u = u[:, num_boundary:, ...]
            u_prime = grad(
                outputs=u.sum(),
                inputs=output_queries,
                create_graph=True,
                retain_graph=True,
            )[0][:, num_boundary:, :]

        u_x = u_prime[:, :, 0]
        u_y = u_prime[:, :, 1]

        # compute second derivatives
        if output_queries_domain is not None:
            u_xx = grad(
                outputs=u_x.sum(),
                inputs=output_queries_domain,
                create_graph=True,
                retain_graph=True,
            )[0][:, :, 0]
            u_yy = grad(
                outputs=u_y.sum(),
                inputs=output_queries_domain,
                create_graph=True,
                retain_graph=True,
            )[0][:, :, 1]
        else:
            u_xx = grad(
                outputs=u_x.sum(),
                inputs=output_queries,
                create_graph=True,
                retain_graph=True,
            )[0][:, num_boundary:, 0]
            u_yy = grad(
                outputs=u_y.sum(),
                inputs=output_queries,
                create_graph=True,
                retain_graph=True,
            )[0][:, num_boundary:, 1]
        u_xx = u_xx.squeeze(0)
        u_yy = u_yy.squeeze(0)
        u_prime = u_prime.squeeze(0)
        u = u.squeeze([0, -1])

        # compute LHS of the Poisson equation
        u_sq = torch.pow(u, 2)
        laplacian = u_xx + u_yy
        norm_grad_u = torch.pow(u_prime, 2).sum(dim=-1)

        assert u_sq.shape == u_xx.shape == u_yy.shape == norm_grad_u.shape

        left_hand_side = laplacian + laplacian * 0.1 * u_sq + 0.2 * u * norm_grad_u
        output_source_terms_domain = output_source_terms_domain.squeeze(0)

        assert left_hand_side.shape == output_source_terms_domain.shape
        loss = self.loss(left_hand_side, output_source_terms_domain)

        assert not u_prime.isnan().any()
        assert not u_yy.isnan().any()
        assert not u_xx.isnan().any()
        del u_xx, u_yy, u_x, u_y, left_hand_side

        return loss

    def __call__(self, y_pred, **kwargs):
        if kwargs:
            warnings.warn(
                f"PoissonInteriorLoss.__call__() received unexpected keyword arguments: {list(kwargs.keys())}. "
                "These arguments will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        if self.method == "autograd":
            return self.autograd(u=y_pred, **kwargs)
        elif self.method == "finite_difference":
            raise NotImplementedError()
        else:
            raise NotImplementedError()


class PoissonBoundaryLoss(object):
    def __init__(self, loss=F.mse_loss):
        super().__init__()
        self.loss = loss
        self.counter = 0

    def __call__(self, y_pred, num_boundary, out_sub_level, y, output_queries, **kwargs):
        if kwargs:
            warnings.warn(
                f"PoissonBoundaryLoss.__call__() received unexpected keyword arguments: {list(kwargs.keys())}. "
                "These arguments will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        num_boundary = int(num_boundary.item() * out_sub_level)
        boundary_pred = y_pred.squeeze(0).squeeze(-1)[:num_boundary]
        y_bound = y.squeeze(0).squeeze(-1)[:num_boundary]

        assert boundary_pred.shape == y_bound.shape
        return self.loss(boundary_pred, y_bound)


class PoissonEqnLoss(object):
    """PoissonEqnLoss computes a weighted sum of equation loss computed on the interior points of a model's output
    and a boundary loss computed on the boundary points.

    Parameters
    ----------
    boundary_weight : float
        weight by which to multiply boundary loss
    interior_weight : float
        weight by which to multiply interior loss
    diff_method : Literal['autograd', 'finite_difference'], optional
        method to use to compute derivatives, by default 'autograd'
    base_loss : Callable, optional
        base loss class to use inside equation and boundary loss, by default F.mse_loss
    """
    def __init__(self, boundary_weight, interior_weight, diff_method: str="autograd", base_loss=F.mse_loss): 
        super().__init__()
        self.boundary_weight = boundary_weight
        self.boundary_loss = PoissonBoundaryLoss(loss=base_loss)

        self.interior_weight = interior_weight
        self.interior_loss = PoissonInteriorLoss(method=diff_method, loss=base_loss)

    def __call__(self, out, y, **kwargs):
        if kwargs:
            warnings.warn(
                f"PoissonEqnLoss.__call__() received unexpected keyword arguments: {list(kwargs.keys())}. "
                "These arguments will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        if isinstance(out, dict):
            interior_loss = self.interior_weight * self.interior_loss(out['domain'], **kwargs)
            bc_loss = self.boundary_weight * self.boundary_loss(out['boundary'], y=y['boundary'],  **kwargs)
        else:
            interior_loss = self.interior_weight * self.interior_loss(out, **kwargs)
            bc_loss = self.boundary_weight * self.boundary_loss(out, y=y, **kwargs)
        return interior_loss + bc_loss


class NavierStokesEqnLoss(nn.Module):
    """
    Physics-informed loss for 2D incompressible Navier-Stokes in vorticity form.

    Expects velocity channels (u_x, u_y) and optionally pressure/other channels.
    Supports spectral or periodic finite-difference derivatives and can denormalize
    predictions before computing residuals.
    """

    def __init__(
        self,
        viscosity: float,
        dx: float = 1.0,
        dy: Optional[float] = None,
        dt: float = 1.0,
        advection_weight: float = 1.0,
        diffusion_weight: float = 1.0,
        forcing_weight: float = 0.0,
        use_vorticity_form: bool = True,
        derivative_mode: str = "spectral",
        denormalize: bool = True,
    ):
        super().__init__()
        self.viscosity = viscosity
        self.dx = dx
        self.dy = dx if dy is None else dy
        self.dt = dt
        self.advection_weight = advection_weight
        self.diffusion_weight = diffusion_weight
        self.forcing_weight = forcing_weight
        self.use_vorticity_form = use_vorticity_form
        self.derivative_mode = derivative_mode
        self.denormalize = denormalize

        allowed_modes = {"spectral", "finite_diff"}
        if derivative_mode not in allowed_modes:
            raise ValueError(
                f"Expected derivative_mode in {allowed_modes}, got {derivative_mode}"
            )

        self._spectral_cache: Dict[
            Tuple[int, int, torch.device, torch.dtype], Tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._fd = (
            FiniteDiff(dim=2, h=(self.dx, self.dy), periodic_in_x=True, periodic_in_y=True)
            if derivative_mode == "finite_diff"
            else None
        )

    def forward(
        self,
        prediction: torch.Tensor,
        inputs: dict,
        normalizer: Optional[object] = None,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        prediction : torch.Tensor
            Shape (B, C, T, H, W) or (B, T, C, H, W). If T is absent, a dummy
            length-1 time dimension is added. The first two channels are treated
            as (u_x, u_y).
        inputs : dict
            Batch dictionary. Forcing can be provided under key 'forcing'.
        normalizer : optional
            If provided and self.denormalize is True, inverse_transform is applied
            before computing derivatives.
        return_components : bool, default False
            When True, returns a tuple of (loss, component_dict).
        """
        layout_hint = None
        if isinstance(inputs, dict):
            layout_hint = inputs.get("layout") or inputs.get("input_layout")
            if layout_hint is None and inputs.get("channel_first") is not None:
                layout_hint = "channel_first" if inputs.get("channel_first") else "time_first"
            if isinstance(layout_hint, str):
                layout_hint = layout_hint.lower()

        if prediction.ndim == 4:
            # (B, C, H, W) -> (B, 1, C, H, W)
            prediction = prediction.unsqueeze(1)
        elif prediction.ndim != 5:
            raise ValueError(
                f"Expected prediction with 4 or 5 dims, got shape {prediction.shape}"
            )

        # canonicalize to (B, T, C, H, W)
        if prediction.ndim == 5:
            if layout_hint == "channel_first":
                prediction = prediction.permute(0, 2, 1, 3, 4)
            elif layout_hint == "time_first":
                pass
            elif prediction.shape[1] in (1, 2, 3) and prediction.shape[2] not in (1, 2, 3):
                prediction = prediction.permute(0, 2, 1, 3, 4)
            elif prediction.shape[2] not in (1, 2, 3):
                prediction = prediction.permute(0, 2, 1, 3, 4)

        if self.denormalize and normalizer is not None:
            prediction = normalizer.inverse_transform(prediction)

        batch, time, channels, height, width = prediction.shape
        if channels < 2:
            raise ValueError(
                f"NavierStokesEqnLoss expects at least 2 channels for velocity, got {channels}"
            )
        if not self.use_vorticity_form:
            raise NotImplementedError("Only vorticity form is implemented.")

        u_x = prediction[:, :, 0]
        u_y = prediction[:, :, 1]
        omega = self._grad_x(u_y) - self._grad_y(u_x)

        omega_t = self._time_derivative(omega)
        omega_x = self._grad_x(omega)
        omega_y = self._grad_y(omega)
        lap_omega = self._laplacian(omega)

        adv_term = u_x * omega_x + u_y * omega_y
        diff_term = self.viscosity * lap_omega
        forcing_term = self._get_forcing(inputs.get("forcing") if isinstance(inputs, dict) else None, omega)

        residual = omega_t + adv_term - diff_term - forcing_term

        loss_adv = (adv_term**2).mean()
        loss_diff = (diff_term**2).mean()
        loss_forcing = (forcing_term**2).mean() if self.forcing_weight != 0.0 else torch.tensor(0.0, device=prediction.device, dtype=prediction.dtype)
        loss_res = (residual**2).mean()

        total_loss = (
            self.advection_weight * loss_adv
            + self.diffusion_weight * loss_diff
            + self.forcing_weight * loss_forcing
            + loss_res
        )

        if return_components:
            components = {
                "loss_adv": loss_adv,
                "loss_diff": loss_diff,
                "loss_forcing": loss_forcing,
                "loss_residual": loss_res,
                "residual_mse": loss_res,
            }
            return total_loss, components
        return total_loss

    def _get_forcing(self, forcing: Optional[torch.Tensor], omega: torch.Tensor) -> torch.Tensor:
        if forcing is None:
            return torch.zeros_like(omega)
        forcing = forcing.to(device=omega.device, dtype=omega.dtype)

        if forcing.ndim == omega.ndim + 1:
            # Drop singleton channel/time dims if present
            if forcing.shape[2] == 1:
                forcing = forcing.squeeze(2)
            elif forcing.shape[1] == 1:
                forcing = forcing.squeeze(1)

        if forcing.ndim == 4:
            forcing = forcing.unsqueeze(1)

        if forcing.shape != omega.shape:
            if forcing.numel() == 1:
                forcing = forcing.expand_as(omega)
            else:
                raise ValueError(
                    f"Forcing shape {forcing.shape} is not compatible with omega shape {omega.shape}"
                )
        return forcing

    def _grad_x(self, field: torch.Tensor) -> torch.Tensor:
        if self.derivative_mode == "spectral":
            kx, ky = self._get_wavenumbers(field)
            field_hat = torch.fft.rfftn(field, dim=(-2, -1))
            return torch.fft.irfftn(1j * kx * field_hat, s=field.shape[-2:], dim=(-2, -1)).real
        return self._fd.dx(field)

    def _grad_y(self, field: torch.Tensor) -> torch.Tensor:
        if self.derivative_mode == "spectral":
            kx, ky = self._get_wavenumbers(field)
            field_hat = torch.fft.rfftn(field, dim=(-2, -1))
            return torch.fft.irfftn(1j * ky * field_hat, s=field.shape[-2:], dim=(-2, -1)).real
        return self._fd.dy(field)

    def _laplacian(self, field: torch.Tensor) -> torch.Tensor:
        if self.derivative_mode == "spectral":
            kx, ky = self._get_wavenumbers(field)
            field_hat = torch.fft.rfftn(field, dim=(-2, -1))
            lap_hat = -(kx**2 + ky**2) * field_hat
            return torch.fft.irfftn(lap_hat, s=field.shape[-2:], dim=(-2, -1)).real
        return self._fd.laplacian(field)

    def _time_derivative(self, field: torch.Tensor) -> torch.Tensor:
        if field.shape[1] < 2:
            return torch.zeros_like(field)
        forward = torch.roll(field, shifts=-1, dims=1)
        backward = torch.roll(field, shifts=1, dims=1)
        dt_center = (forward - backward) / (2.0 * self.dt)
        dt_center[:, 0] = (field[:, 1] - field[:, 0]) / self.dt
        dt_center[:, -1] = (field[:, -1] - field[:, -2]) / self.dt
        return dt_center

    def _get_wavenumbers(
        self, field: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        height, width = field.shape[-2], field.shape[-1]
        key = (height, width, field.device, field.dtype)
        if key not in self._spectral_cache:
            kx_base = 2 * math.pi * torch.fft.fftfreq(
                height, d=self.dx, device=field.device, dtype=field.dtype
            ).view(height, 1)
            ky_base = 2 * math.pi * torch.fft.rfftfreq(
                width, d=self.dy, device=field.device, dtype=field.dtype
            ).view(1, -1)
            self._spectral_cache[key] = (kx_base, ky_base)

        kx_base, ky_base = self._spectral_cache[key]
        kx = kx_base
        ky = ky_base
        while kx.dim() < field.dim():
            kx = kx.unsqueeze(0)
        while ky.dim() < field.dim():
            ky = ky.unsqueeze(0)
        return kx, ky
