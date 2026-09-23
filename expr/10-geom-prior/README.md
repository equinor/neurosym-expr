# Geometric-prior activation steering

The script can derive its steering subspace either dynamically from a prior
prompt or directly from paired desired and opposite texts.

Use contrastive text pairs by passing equally sized lists. Entries are paired
by position:

```bash
uv run 10-geom-prior \
  --steering-text \
    "Give a precise, factual, logically structured answer." \
    "State assumptions and distinguish evidence from speculation." \
  --steering-opposite \
    "Give a vague, emotional, disorganized answer." \
    "Present guesses as facts without qualification." \
  --prompt "Explain why authenticated encryption matters."
```

The selected layer's final-token activations are captured for every text. SVD
is applied to each `steering text - opposite text` activation difference, and
the desired-side activations calibrate the transport controller. The number of
pairs must be at least `--latent-dim` (two by default).

Without `--steering-text` and `--steering-opposite`, the existing dynamic
low-entropy search based on `--prior-prompt` is used.