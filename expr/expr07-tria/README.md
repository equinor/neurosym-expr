# Adaptive P-spline triangular transport

This experiment contains a PyTorch implementation of the adaptive triangular
transport from *Adaptive Nonlinear Data Assimilation through P-Spline
Triangular Measure Transport*.

The implementation uses:

- local Cox-de Boor evaluation for differentiable B-spline bases;
- a PyTorch damped-Newton inner solver;
- direct implicit differentiation using the fitted spline Hessian;
- `torch.optim.LBFGS` for AICc-based smoothing selection;
- PyTorch linear algebra and bisection for forward and inverse maps.

It does not use NumPy or SciPy.

```python
import torch

from expr07_tria import AdaptiveSplineTransport

samples = torch.randn(100, 3, dtype=torch.float64)

transport = AdaptiveSplineTransport()
transport.fit(samples)

reference = transport(samples)
reconstructed = transport.inverse(reference)
```

`AdaptiveSplineTransport` is a `torch.nn.Module`. Its fitted tensors are
registered buffers, so `state_dict()` includes the learned map and `.to(...)`
moves or converts the map consistently:

```python
transport = transport.to(device="cuda", dtype=torch.float32)
reference = transport(samples.to(device="cuda", dtype=torch.float32))
```

For conditional sampling, place conditioned variables first and set
`skip_dimensions` to their count:

```python
transport = AdaptiveSplineTransport(skip_dimensions=1)
transport.fit(samples)

fixed_values = torch.full((100, 1), 0.5, dtype=torch.float64)
reference = torch.randn(100, 2, dtype=torch.float64)
conditional_samples = transport.conditional_inverse(fixed_values, reference)
```

`lambda_initial` and the values exposed through `transport.log_smoothing_` are
logarithms of smoothing penalties. One smoothing value is fitted for every
active additive spline block. Fitted coefficients and diagnostics are
available through `coefficients_`, `effective_dof_`, `aicc_`, and `nll_`.