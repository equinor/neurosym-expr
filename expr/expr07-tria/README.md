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

For the target workload of at least 64 samples and roughly 12,000 independent
parameters, use the focused batched implementation. It uses fixed cubic
P-splines, fits all dimensions together on the input tensor's device, and
caches interval polynomials for low-overhead GPU evaluation:

```python
from expr07_tria import BatchedDiagonalSplineTransport

samples = torch.randn(64, 12_000, device="cuda")
transport = BatchedDiagonalSplineTransport().fit(samples)
```

The batched transport defaults to 10 safeguarded Newton iterations, which is
optimized for fitting speed at this workload. Increase `max_fit_iterations`
when fit quality is more important than throughput:

```python
transport = BatchedDiagonalSplineTransport(max_fit_iterations=20).fit(samples)
```

For sparse multiscale conditional dependence, the boosted wavelet transport
keeps physical variables on the monotone diagonal and adds causal Haar spline
features from preceding variables:

```python
from expr07_tria import BoostedWaveletSplineTransport

transport = BoostedWaveletSplineTransport(
    max_wavelet_level=6,
    max_parent_distance=256,
    max_learners_per_component=4,
    block_size=256,
).fit(samples)

conditioned = transport.conditional_inverse(samples[:, :1], reference[:, 1:])
exceedances = transport.sample_exceedance(
    component=0,
    threshold=1.0,
    sample_count=100,
)
```

For large GPU workloads, `block_size` groups mapped components that share the
same preceding parent variables. Fitting and inversion are vectorized within
each block, so 12,000 components with `block_size=256` require only about 47
sequential inverse stages. During fitting, screening signals and spline
designs are built one parent window at a time and released after each block;
keep `max_parent_distance` finite to make the peak working memory independent
of the total component count. The default `block_size=1` preserves scalar
triangular behavior. Wavelet supports are always restricted to preceding
blocks, which preserves exact inversion. Exact exceedance sampling currently
supports the leading component; later-component inequalities require
sequential importance sampling.

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

## SmolLM3 hidden-state steering

`smollm3_hidden_steering.py` captures the last-token hidden state immediately
before `lm_head` computes the next-token logits. It samples a local cloud,
fits a boosted causal Haar-wavelet spline transport, and repeatedly refits it
to states that combine low next-token entropy with high fitted density.
Candidate norms are matched to the original hidden state before evaluation.

```bash
uv run python smollm3_hidden_steering.py \
  "Explain why the sky is blue." \
  --samples 256 \
  --iterations 2 \
  --elite-fraction 0.25 \
  --noise-scale 0.05 \
  --density-weight 1.0 \
  --max-wavelet-level 6 \
  --max-parent-distance 256 \
  --block-size 256 \
  --seed 7
```

These wavelet and block settings are the defaults and keep transport fitting
bounded to a local parent window. Increase `--block-size` for faster inversion
at the cost of excluding dependencies between coordinates in the same block.

`Transport.last_hidden_state` contains the unmodified state, while
`last_steered_hidden_state` contains the selected vector passed to `lm_head`.
For each generation step, the script writes the original greedy token and the
token selected after transport, including their IDs and decoded text, to
standard error.

## SmolLM3 particle sampling

`smollm3_particle_sampling.py` keeps a population of model-generated token
trajectories instead of perturbing hidden states. With the default entropy
weight, temperature, and top-p settings, it reduces to ancestral sampling from
SmolLM3. Temperature and top-p can define a different proposal; sequential
importance weights correct for that proposal, and low-effective-sample-size
populations are resampled.

```bash
uv run python smollm3_particle_sampling.py \
  "Explain why the sky is blue." \
  --particles 32 \
  --entropy-weight 0.05 \
  --resample-threshold 0.5 \
  --seed 7
```

A positive entropy weight targets trajectories proportional to the model
sequence probability times `exp(-weight * cumulative_entropy)`. This favors
prefixes where the model remains confident while ensuring that every particle
is reached through ordinary token generation. `--best` returns the
highest-scoring surviving trajectory instead of sampling from the final
particle population.