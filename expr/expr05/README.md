# Predictive maintenance logistic regression

Fit a logistic regression model to predict the `Machine failure` target and
print its classification report:

```console
uv run expr05
```

Pass another CSV path as the first argument to use a different dataset.

## Research agent

Run the researcher with its model and the separately trained probabilitier:

```console
uv run researcher --prob-model path/to/probabilitier
```

`src/expr05/researcher.py` contains the complete pipeline and has no imports
from other local modules or projects. Its remaining dependencies are
installable Python packages.

The researcher loads `res/ai4i2020.csv` into a pandas DataFrame named `df`. It
can run stateful Python/pandas analysis and call the probabilitier as
`model_builder(description)`. Each generated probabilistic model is parsed,
sampled once per data row, and its modeled failure rate is compared with the
observed failure rate. The researcher receives the measured error and revises
the model until it is within the default absolute tolerance of `0.01`. Every
system, user, assistant, and tool message is printed as it is produced, so each
turn's analysis and latest tool results remain visible. Pass a CSV path as the
positional argument, use `--target` when its target is not `Machine failure`,
`--failure-rate-tolerance` to change the acceptance threshold,
`--research-time-limit` to change the default five-minute limit per request,
and `--prompt` for a single non-interactive request:

```console
uv run researcher data.csv --prob-model path/to/probabilitier \
  --prompt "Investigate the main drivers of machine failure"
```