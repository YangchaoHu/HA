"""Smoke example for the minimal KAN-GP package.

Run:
    python example_train.py
"""
from __future__ import annotations

import math

import gpytorch
import torch

from kan_gp import create_pimfgpkan


def target_fn(x: torch.Tensor) -> torch.Tensor:
    x0 = x[:, 0]
    return torch.sin(2 * math.pi * x0) + 0.25 * torch.cos(6 * math.pi * x0)


def low_fidelity_mean(x: torch.Tensor) -> torch.Tensor:
    return 0.8 * torch.sin(2 * math.pi * x[:, 0])


def train(model, likelihood, train_x, train_y, iters: int = 60, lr: float = 0.03) -> None:
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for step in range(1, iters + 1):
        optimizer.zero_grad()
        loss = -mll(model(train_x), train_y)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 20 == 0:
            print(f"step={step:03d} loss={loss.item():.4f}")


def main() -> None:
    torch.manual_seed(0)

    n_train = 40
    n_test = 120
    train_x = torch.linspace(0.0, 1.0, n_train).unsqueeze(-1)
    test_x = torch.linspace(0.0, 1.0, n_test).unsqueeze(-1)

    train_y = target_fn(train_x) + 0.05 * torch.randn(n_train)
    test_y = target_fn(test_x)

    model, likelihood = create_pimfgpkan(
        train_x=train_x,
        train_y=train_y,
        physics_mean_fn=low_fidelity_mean,
        input_dim=1,
        hidden_dim=12,
        feature_dim=4,
        num_kan_layers=2,
    )

    train(model, likelihood, train_x, train_y)

    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(test_x))

    rmse = torch.sqrt(torch.mean((pred.mean - test_y) ** 2)).item()
    print(f"test_rmse={rmse:.4f}")
    print("first_five_predictions=", pred.mean[:5].tolist())


if __name__ == "__main__":
    main()
