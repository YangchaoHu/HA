# Minimal KAN-GP

This folder extracts the early `gp_kan_physics` KAN-GP model into a small,
runnable dependency set.

Core model:

```text
f(x) ~ GP(mu_phys(x), k_KAN(x, x'))
k_KAN(x, x') = k_base(KAN(x), KAN(x'))
```

Included pieces:

- `KANLinear`: B-spline KAN layer from the early feature extractor.
- `KANFeatureExtractor`: stacked KAN feature map.
- `KANDeepKernel`: GPyTorch kernel wrapping `k_base(KAN(x), KAN(x'))`.
- `PIMFGPKAN`: exact GP with optional physics/low-fidelity mean.
- `create_pimfgpkan`: convenience factory.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python example_train.py
```

Expected output includes training losses, a test RMSE, and several predicted
means. The example uses a simple sinusoidal low-fidelity mean and lets the
KAN-GP model the residual.
