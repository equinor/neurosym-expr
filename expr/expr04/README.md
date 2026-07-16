# Probabilistic expression language

This language describes a probabilistic model as a short sequence of sampled
and computed variables. Distribution names correspond to distributions in
`scipy.stats`, and function names correspond to NumPy functions. A model ends
by returning the value or condition of interest.

## A complete model

```text
height ~ norm(loc=176, scale=7.1)
measurement_error ~ norm(loc=0, scale=1.5)
observed_height = height + measurement_error
is_tall = observed_height > 180
return is_tall
```

Each statement must be on its own line. Blank lines and `#` comments are
allowed.

## Sample random variables

Use `~` to sample a variable from a distribution:

```text
temperature ~ norm(loc=20, scale=2)
arrivals ~ poisson(mu=4)
success ~ bernoulli(p=0.75)
category ~ randint(low=0, high=3)
return temperature
```

The distribution name is the unqualified name from `scipy.stats`: use `norm`,
not `scipy.stats.norm`. Distribution arguments follow the corresponding SciPy
API and may be positional, keyword, or a combination of positional arguments
followed by keyword arguments:

```text
x ~ norm(0, scale=1)
return x
```

Parentheses are required even for a distribution with no arguments:

```text
x ~ norm()
return x
```

The grammar accepts any identifier as a distribution name; the component that
evaluates the model is responsible for resolving it against `scipy.stats` and
reporting unknown distributions or invalid parameters.

### Empirical distributions

Use `empirical` to sample from observed values. It maps to
`probabilit.EmpiricalDistribution` rather than a SciPy distribution:

```text
die ~ empirical([1, 2, 3, 4, 5, 6], method="closest_observation")
return die
```

The data must be a non-empty, one-dimensional list and may also be passed as
the `data` keyword argument. Other arguments are forwarded to `numpy.quantile`.

### Discrete and cumulative distributions

Use `discrete` for categorical values with optional probabilities:

```text
risk ~ discrete(["low", "medium", "high"], probabilities=[0.2, 0.3, 0.5])
return risk
```

Use `cumulative` to define a distribution by quantile positions and values:

```text
cost ~ cumulative([0, 0.2, 0.8, 1], [10, 15, 20, 25])
return cost
```

These map to `probabilit.DiscreteDistribution` and
`probabilit.CumulativeDistribution`, respectively.

### Multivariate distributions

Assign the marginals of a SciPy multivariate distribution to multiple variables:

```text
x, y ~ multivariate_normal(
    mean=[1, 2],
    cov=[[1, 0.5], [0.5, 1]],
)
return x + y
```

The number of assigned variables must match the number of values produced by
the distribution. Multivariate distributions inherit Probabilit's limitations:
their parameters cannot contain other distributions and they use pseudorandom
sampling.

## Compute derived values

Use `=` for deterministic expressions:

```text
radius ~ uniform(loc=0, scale=10)
area = 3.14159 * (radius ** 2)
rounded_area = round(area)
return rounded_area
```

Function calls use unqualified NumPy names, such as `sqrt(x)`, `exp(x)`,
`floor(x)`, `maximum(x, y)`, or `mean(values)`. Positional arguments must come
before keyword arguments, and a trailing comma is allowed:

```text
samples = clip(values, a_min=0, a_max=1)
return mean(samples,)
```

As with distributions, the grammar only validates the call syntax. An
evaluator must resolve function names against NumPy.

## Literals and lists

The language supports:

- integers and decimal numbers, such as `3` and `2.5`;
- escaped double-quoted strings, such as `"accepted"`;
- `true`, `false`, and `null`;
- lists, including multiline and nested lists.

```text
label_index ~ randint(low=0, high=3)
labels = ["red", "green", "blue"]
weights = [
    0.2,
    0.3,
    0.5,
]
return label_index
```

A trailing comma is optional in lists and function or distribution calls.

## Operators and grouping

Supported binary operators are:

| Kind | Operators |
| --- | --- |
| Arithmetic | `+`, `-`, `*`, `/`, `//`, `%`, `**` |
| Comparison | `==`, `!=`, `<`, `<=`, `>`, `>=` |
| Logic | `and`, `or` |
| Membership | `in` |

Unary negation uses `-`; logical negation uses `not`:

```text
return not (result in ["failed", "cancelled"])
```

There is deliberately no implicit operator precedence. An expression may
contain only one ungrouped binary operator, so use parentheses whenever
combining operations:

```text
# Valid
total = base + (rate * duration)
inside = (lower <= value) and (value <= upper)
return total
```

The following forms are rejected because their grouping is ambiguous:

```text
return base + rate * duration
```

```text
return lower < value < upper
```

Unary operators bind before the binary operator. For example, `-x ** 2` is
interpreted as `(-x) ** 2`; write `-(x ** 2)` for the other interpretation.

Parentheses also permit line breaks:

```text
distance = (
    dx ** 2
) + (
    dy ** 2
)
return sqrt(distance)
```

## Correlate sampled variables

Use a correlation statement after declaring the relevant variables:

```text
height ~ norm(loc=176, scale=7.1)
weight ~ norm(loc=75, scale=12)
correlate height with weight at 0.65
return (height > 180) and (weight > 90)
```

The value after `at` must be a signed numeric literal. The grammar does not
enforce the usual correlation range of `-1` to `1`; an evaluator should
validate that range and apply the requested correlation.

Correlate more than two variables by supplying their complete correlation
matrix. Variable order must match the matrix rows and columns:

```text
a ~ norm()
b ~ norm()
c ~ norm()
correlate [a, b, c] with [
    [1, 0.5, -0.2],
    [0.5, 1, 0.3],
    [-0.2, 0.3, 1],
]
return (a + b) + c
```

Correlation matrices must be square, symmetric, finite, contain ones on the
diagonal, and have all entries between `-1` and `1`.

## Return a result

Every model must end with exactly one `return` statement:

```text
die ~ randint(low=1, high=7)
return die == 6
```

The return value may be a sampled variable, a derived value, a literal, a list,
a function call, or a grouped expression. No model statements may follow it.

## Parse a model from Python

The package exports a Lark parser:

```python
from expr import parser

source = """\
x ~ norm(loc=0, scale=1)
return x > 1.96
"""

tree = parser.parse(source)
print(tree.pretty())
```

Parsing checks syntax only. Use `parse_model` to convert source directly into
a lazy `probabilit` computational graph:

```python
from expr import parse_model

model = parse_model(source)
samples = model.sample(10_000, random_state=42)
```

An existing parse tree can be converted with `model_from_tree(tree)`.
Distribution calls become `probabilit.Distribution` nodes, NumPy calls and
operators become transform nodes, and correlation statements are attached to
the returned graph. Conversion reports undefined variables, invalid
correlations, and unknown NumPy functions as `ModelConversionError`.