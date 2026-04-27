"""Minimal KAN-GP model extracted from `gp_kan_physics`.

This keeps the early KAN-as-deep-kernel design:

    k_KAN(x, x') = k_base(KAN(x), KAN(x'))

The default base kernel is an RBF kernel wrapped in a ScaleKernel.  Optional
physics/low-fidelity information enters as a GP mean function.
"""
from __future__ import annotations

from typing import Callable, Optional

import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import Mean, ZeroMean
from gpytorch.models import ExactGP


class KANLinear(nn.Module):
    """A compact B-spline KAN layer.

    Each input coordinate has a learnable univariate spline contribution, plus
    a SiLU base path.  This is the early `gp_kan_physics.models.kan_network`
    layer with only the forward-pass dependency retained.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_noise = scale_noise
        self.scale_base = scale_base

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1) * h
        grid = grid + grid_range[0]
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            noise = (
                torch.rand(self.grid_size + self.spline_order, self.in_features, self.out_features)
                - 0.5
            ) * self.scale_noise / self.grid_size
            self.spline_weight.copy_(noise.permute(2, 1, 0))
            self.base_weight.copy_(
                self.scale_base * torch.randn(self.out_features, self.in_features)
            )

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Compute B-spline bases for a 2D input tensor `(batch, features)`."""
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(f"expected x with shape (batch, {self.in_features})")

        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:-1]) & (x < grid[1:])).to(x.dtype)

        for k in range(1, self.spline_order + 1):
            left_den = grid[k:-1] - grid[: -k - 1]
            right_den = grid[k + 1 :] - grid[1:-k]
            bases = (
                (x - grid[: -k - 1]) / left_den * bases[..., :-1]
                + (grid[k + 1 :] - x) / right_den * bases[..., 1:]
            )
        return bases.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.silu(x) @ self.base_weight.T
        spline_basis = self.b_splines(x)
        spline_output = torch.einsum("bij,oij->bo", spline_basis, self.spline_weight)

        output = base_output + spline_output
        return output.reshape(*original_shape[:-1], self.out_features)


class KANFeatureExtractor(nn.Module):
    """Stack of KAN layers used as the deep-kernel feature map."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        output_dim: int = 16,
        num_layers: int = 2,
        grid_size: int = 5,
        spline_order: int = 3,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim

        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(
                KANLinear(
                    in_features=in_dim,
                    out_features=out_dim,
                    grid_size=grid_size,
                    spline_order=spline_order,
                )
            )
            if use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def get_output_dim(self) -> int:
        return self.output_dim


class PhysicsMean(nn.Module):
    """Base class for optional prior mean functions."""

    def __init__(self, trainable: bool = False) -> None:
        super().__init__()
        self.trainable = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CallableMean(PhysicsMean):
    """Wrap any callable `fn(x) -> y` as a GP mean."""

    def __init__(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        trainable: bool = False,
    ) -> None:
        super().__init__(trainable=trainable)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.trainable):
            return self.fn(x).squeeze(-1)


class LowFidelityMean(PhysicsMean):
    """Low-fidelity model with optional learnable affine calibration."""

    def __init__(
        self,
        low_fidelity_fn: Callable[[torch.Tensor], torch.Tensor],
        trainable: bool = False,
        scale: float = 1.0,
        bias: float = 0.0,
    ) -> None:
        super().__init__(trainable=trainable)
        self.low_fidelity_fn = low_fidelity_fn

        if trainable:
            self.scale = nn.Parameter(torch.tensor(float(scale)))
            self.bias = nn.Parameter(torch.tensor(float(bias)))
        else:
            self.register_buffer("scale", torch.tensor(float(scale)))
            self.register_buffer("bias", torch.tensor(float(bias)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.low_fidelity_fn(x).squeeze(-1)
        return self.scale * y + self.bias


class PhysicsMeanWrapper(Mean):
    """Bridge a `PhysicsMean` module to GPyTorch's mean-function interface."""

    def __init__(self, physics_mean: PhysicsMean) -> None:
        super().__init__()
        self.physics_mean = physics_mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.physics_mean(x)


class KANDeepKernel(Kernel):
    """Deep kernel `k_base(KAN(x), KAN(x'))`."""

    is_stationary = True

    def __init__(
        self,
        kan_extractor: KANFeatureExtractor,
        base_kernel: Optional[Kernel] = None,
    ) -> None:
        super().__init__()
        self.kan_extractor = kan_extractor
        self.base_kernel = base_kernel or ScaleKernel(RBFKernel())

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ):
        if last_dim_is_batch:
            raise NotImplementedError("last_dim_is_batch is not supported")

        features1 = self.kan_extractor(x1)
        features2 = self.kan_extractor(x2)
        return self.base_kernel(features1, features2, diag=diag, **params)

    def get_feature_dim(self) -> int:
        return self.kan_extractor.get_output_dim()


class PIMFGPKAN(ExactGP):
    """Exact GP with optional physics mean and KAN deep kernel."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: GaussianLikelihood,
        physics_mean: Optional[PhysicsMean] = None,
        kan_extractor: Optional[KANFeatureExtractor] = None,
        input_dim: int = 1,
        hidden_dim: int = 32,
        feature_dim: int = 16,
        num_kan_layers: int = 2,
        use_physics_mean: bool = True,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.physics_mean = physics_mean
        self.use_physics_mean = use_physics_mean and physics_mean is not None

        self.mean_module = (
            PhysicsMeanWrapper(physics_mean)
            if self.use_physics_mean and physics_mean is not None
            else ZeroMean()
        )

        self.kan_extractor = kan_extractor or KANFeatureExtractor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=feature_dim,
            num_layers=num_kan_layers,
        )
        self.covar_module = KANDeepKernel(self.kan_extractor)

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def get_residual(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.physics_mean is None:
            return y
        return y - self.physics_mean(x)


def create_pimfgpkan(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    physics_mean_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    trainable_physics_mean: bool = False,
    physics_mean_scale: float = 1.0,
    physics_mean_bias: float = 0.0,
    input_dim: int = 1,
    hidden_dim: int = 32,
    feature_dim: int = 16,
    num_kan_layers: int = 2,
    noise_floor: float = 1e-6,
) -> tuple[PIMFGPKAN, GaussianLikelihood]:
    """Factory matching the early `gp_kan_physics` exact-GP path."""
    likelihood = GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(noise_floor)
    )

    physics_mean: Optional[PhysicsMean]
    if physics_mean_fn is None:
        physics_mean = None
    elif trainable_physics_mean:
        physics_mean = LowFidelityMean(
            low_fidelity_fn=physics_mean_fn,
            trainable=True,
            scale=physics_mean_scale,
            bias=physics_mean_bias,
        )
    else:
        physics_mean = CallableMean(physics_mean_fn)

    model = PIMFGPKAN(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        physics_mean=physics_mean,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        feature_dim=feature_dim,
        num_kan_layers=num_kan_layers,
    )
    return model, likelihood
