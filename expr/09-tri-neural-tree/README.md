# Boosted hard- and soft-tree triangular transports

This experiment implements a self-contained PyTorch triangular transport whose
hard-tree leaves are monotone cubic P-splines. It copies and adapts the spline
and causal Haar numerical machinery from experiment 07; it does not import
code from another experiment.

For component \(i\), a tree partitions causal Haar features of the preceding
coordinates and selects one monotone spline:

\[
T_i(x_{1:i}) = s_{i,\ell(W_i x_{<i})}(x_i).
\]

Trees are grown greedily. Candidate child splines choose their complexity using
effective degrees of freedom, while an MDL/BIC-style criterion charges for
both spline complexity and split search. A split is accepted only when this
criterion improves. Additional triangular stages are composed until the next
stage is the identity, so invertibility does not depend on a boosting learning
rate.

```python
import torch

from nine import BoostedHardTreeTransport

samples = torch.randn(128, 4, dtype=torch.float64)
transport = BoostedHardTreeTransport().fit(samples)

reference = transport(samples)
reconstructed = transport.inverse(reference)
log_density = transport.log_prob(samples)
```

The fitted map supports exact triangular log-determinants through
`forward_with_logdet`, sequential inverse evaluation, prefix-conditional
inversion, `state_dict()` persistence, and device/dtype conversion.

## Soft-tree neural spline transport

`BoostedSoftTreeTransport` replaces hard routing and P-spline leaves with
differentiable sigmoid gates and rational-quadratic neural splines:

\[
T_i(x_{1:i}) =
\operatorname{RQS}\left(
x_i;
\sum_\ell \pi_{i,\ell}(x_{<i})\theta_{i,\ell}
\right).
\]

The tree blends leaf parameter logits before constructing one monotone spline,
rather than averaging spline outputs. This preserves closed-form inversion and
log derivatives. Gates only inspect preceding coordinates, so every stage
remains lower triangular.

```python
from nine import BoostedSoftTreeTransport

transport = BoostedSoftTreeTransport(
    max_depth=2,
    num_bins=8,
).fit(samples)
```

Candidate components share one batched optimizer loop, with selected Haar
projections cached throughout fitting. Components are retained only when they
improve their validation objective. Candidate stages are then retained only
when they improve both the held-out objective and the full training objective
by the configured minimum improvement per sample. After stage construction,
accepted stages are jointly fine-tuned against the composed likelihood;
`fine_tune_epochs=0` disables this step. Soft-tree parameters, topology, and
feature metadata participate in `state_dict()` persistence and device/dtype
conversion.

## Paper benchmark reproduction

The repository's overfitting-mixture and wavy-distribution experiments are
available as a self-contained comparison between the copied adaptive P-spline
baseline and the boosted hard-tree model:

```bash
uv run nine-benchmarks --repetitions 5
```

This preserves the paper repository's training sizes of 25 and 30,
respectively. It reports held-out NLL relative to the known data density,
reference mean/covariance error, fitting time, and selected tree complexity.
The generated `benchmark_results.json` records every seeded run and an
aggregate summary.

The upstream Lorenz-63 study is a 1,000-step filtering benchmark over eight
ensemble sizes and ten seeds, rather than a standalone density-estimation
example. A resumable self-contained runner is provided:

```bash
uv run nine-lorenz63
```

Its defaults reproduce the upstream ensemble sizes, seeds, 1,000-step EnKF
spin-up, 1,000 filtering steps, sequential scalar observations, observation
error, model error, and RMSE calculation. Results are checkpointed after every
model/seed/ensemble-size combination in `lorenz63_results.json`.

The groundwater study additionally requires MODFLOW 6 and generated simulation
assets. It is not silently approximated by these runners.

### Current comparison

Five PyTorch-seeded repetitions with 1,000 independent validation samples gave:

| Experiment | Model | Validation NLL | Excess over oracle | Reference covariance RMSE | Fit seconds | Selected stages/leaves |
|---|---|---:|---:|---:|---:|---:|
| 25-sample mixture | Adaptive P-spline | 1.410 | 0.165 | 0.314 | 0.030 | - |
| 25-sample mixture | Boosted hard tree | 1.447 | 0.203 | 0.370 | 0.010 | 0 / 0 |
| 30-sample wavy | Adaptive P-spline | 2.694 | 1.405 | 1.194 | 0.071 | - |
| 30-sample wavy | Boosted hard tree | **2.501** | **1.212** | **0.514** | 0.241 | 1.0 / 2.2 |

The one-dimensional mixture has no preceding coordinate on which a triangular
tree can split, so its result compares the two marginal spline estimators. The
adaptive P-spline is modestly better there. On the conditional wavy problem,
the hard-tree model selects about two leaves and improves held-out likelihood
and Gaussian-reference covariance, at roughly three times the fitting cost.

These are distribution-matched reproductions, not bitwise reruns of the
NumPy/SciPy samples in the upstream scripts. The complete per-seed values and
oracle likelihoods are stored in `benchmark_results.json`.

`max_stages` and `max_leaves` are search safety budgets. If either is reached,
`stopping_reason_` is `"search_budget_exhausted"`; this distinguishes a
truncated search from automatic statistical stopping. Both implementations use scalar triangularity. Block-triangular
approximations remain out of scope.

For ordered high-dimensional fields, `max_parent_distance` restricts each tree
component to a causal window of preceding coordinates. For example,
`BoostedHardTreeTransport(max_parent_distance=256)` avoids a dense global
parent screen while preserving triangularity. Leave it as `None` when the
ordering does not imply local dependence.