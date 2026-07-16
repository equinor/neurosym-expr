import numpy as np
import pytest

from expr import parse_model
from probabilit import Distribution


def sampled_distributions(model):
    return sorted(
        {node for node in model.nodes() if isinstance(node, Distribution)},
        key=lambda node: node._id,
    )


def test_die_problem():
    model = parse_model(
        """\
roll1 ~ uniform()
roll2 ~ uniform()
die1 = floor(1 + (roll1 * 6))
die2 = floor(1 + (roll2 * 6))
return die1 == die2
"""
    )

    samples = model.sample(999, random_state=42)

    np.testing.assert_allclose(samples.mean(), 1 / 6, atol=0.001)


def test_estimating_pi():
    model = parse_model(
        """\
x ~ uniform()
y ~ uniform()
distance_squared = (x ** 2) + (y ** 2)
inside = distance_squared < 1
return 4 * inside
"""
    )

    samples = model.sample(9_999, random_state=42)

    np.testing.assert_allclose(samples.mean(), np.pi, atol=0.01)


def test_broken_stick_problem():
    model = parse_model(
        """\
cut1 ~ uniform(loc=0, scale=1)
cut2 ~ uniform(loc=0, scale=1)
length1 = minimum(cut1, cut2)
length2 = maximum(cut1, cut2) - minimum(cut1, cut2)
length3 = 1 - maximum(cut1, cut2)
first_short = length1 < 0.5
second_short = length2 < 0.5
third_short = length3 < 0.5
return first_short and (second_short and third_short)
"""
    )

    samples = model.sample(9_999, random_state=42)

    np.testing.assert_allclose(samples.mean(), 1 / 4, atol=0.01)


def test_mutual_fund_problem():
    statements = ["returns_0 = 0"]
    for year in range(1, 21):
        statements.extend(
            [
                f"interest_{year} ~ norm(loc=1.11, scale=0.15)",
                f"returns_{year} = (returns_{year - 1} * interest_{year}) + 1200",
            ]
        )
    statements.append("return returns_20")
    model = parse_model("\n".join(statements))

    samples = model.sample(999, random_state=42)

    np.testing.assert_allclose(samples.mean(), 76_630.897017, rtol=1e-4)
    np.testing.assert_allclose(samples.std(), 34_507.634828, rtol=1e-4)


def test_total_person_hours():
    statements = ["total_0 = 0"]
    for plate in range(1, 563):
        statements.extend(
            [
                f"time_{plate} ~ triang(c=0.2857142857142857, "
                "loc=3.75, scale=1.75)",
                f"total_{plate} = total_{plate - 1} + time_{plate}",
            ]
        )
    statements.append("return total_562")
    model = parse_model("\n".join(statements))

    samples = model.sample(1_000, random_state=np.random.default_rng(42))

    np.testing.assert_allclose(samples.mean(), 4.5 * 562, rtol=0.02)
    np.testing.assert_allclose(
        samples.std(ddof=1), 0.368 * np.sqrt(562), rtol=0.02
    )


def test_conditional_twin_heights():
    model = parse_model(
        """\
height1 ~ norm(loc=176, scale=7.1)
independent_height2 ~ norm(loc=176, scale=7.1)
is_twin ~ bernoulli(p=0.1)
height2 = (is_twin * height1) + ((1 - is_twin) * independent_height2)
return absolute(height2 - height1)
"""
    )
    height1, _, _ = sampled_distributions(model)

    samples = model.sample(999, random_state=42)

    assert np.any(np.isclose(samples, 0))
    assert height1.samples_.shape == samples.shape


def test_fault_controlled_contact():
    model = parse_model(
        """\
contact1 ~ uniform(loc=1995, scale=10)
fault_is_open ~ bernoulli(p=0.3)
independent_contact2 ~ uniform(loc=1950, scale=50)
contact2 = (fault_is_open * contact1) + ((1 - fault_is_open) * independent_contact2)
return contact2
"""
    )
    contact1, fault_is_open, _ = sampled_distributions(model)

    samples = model.sample(100, random_state=42)
    open_fault = fault_is_open.samples_.astype(bool)

    assert open_fault.any()
    assert (~open_fault).any()
    np.testing.assert_allclose(samples[open_fault], contact1.samples_[open_fault])
    assert np.all((1950 <= samples[~open_fault]) & (samples[~open_fault] <= 2000))


def test_stopping_distance_problem():
    dry_model = parse_model(
        """\
velocity_error ~ norm(0, 0.03)
velocity_kmh = exp(velocity_error) * 100
velocity_ms = velocity_kmh / 3.6
gravity ~ norm(loc=9.8220, scale=0.0020)
friction ~ norm(loc=0.7, scale=0.02)
return (velocity_ms ** 2) / ((2 * friction) * gravity)
"""
    )
    wet_model = parse_model(
        """\
velocity_error ~ norm(0, 0.03)
velocity_kmh = exp(velocity_error) * 100
velocity_ms = velocity_kmh / 3.6
gravity ~ norm(loc=9.8220, scale=0.0020)
is_concrete ~ bernoulli(p=0.5)
base_friction ~ norm(loc=0.53, scale=0.02)
friction = base_friction + (0.05 * is_concrete)
return (velocity_ms ** 2) / ((2 * friction) * gravity)
"""
    )

    dry_samples = dry_model.sample(999, random_state=42, method="lhs")
    wet_samples = wet_model.sample(999, random_state=42, method="lhs")

    np.testing.assert_allclose(dry_samples.mean(), 56.261536, rtol=1e-4)
    np.testing.assert_allclose(dry_samples.std(), 3.753207, rtol=1e-4)
    np.testing.assert_allclose(wet_samples.mean(), 71.138801, rtol=1e-4)
    np.testing.assert_allclose(wet_samples.std(), 5.91761, rtol=1e-4)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 2", 4),
        ("5 - 2", 3),
        ("2 - 5", -3),
        ("5 / 2", 2.5),
        ("2 / 5", 0.4),
        ("-(-2)", 2),
        ("5 ** 2", 25),
        ("(2 + 2) - ((5 ** 2) - absolute(-5))", -16),
        ("(2 / 5) - ((2 ** 3) - exp(5))", 140.8131591025766),
        ("(1 / 5) - (log(5) + exp(log(10)))", -11.4094379124341),
    ],
)
def test_constant_expressions(expression, expected):
    model = parse_model(f"return {expression}")

    np.testing.assert_allclose(model.sample(), expected)


def test_single_constant_expression():
    model = parse_model("return 2")

    np.testing.assert_allclose(model.sample(), 2)


def test_empirical_distribution_with_categorical_observations():
    model = parse_model(
        """\
dice ~ empirical([1, 2, 3, 4, 5, 6], method="closest_observation")
return dice
"""
    )

    samples = model.sample(9, random_state=42)

    np.testing.assert_array_equal(samples, [2, 6, 4, 4, 1, 1, 1, 5, 4])


def test_empirical_distribution_with_numeric_observations():
    model = parse_model(
        """\
cost ~ empirical(data=[200, 200, 300, 250, 225])
return cost
"""
    )

    samples = model.sample(9, random_state=42)

    np.testing.assert_allclose(
        samples,
        [
            212.45401188,
            290.14286128,
            248.19939418,
            234.86584842,
            200,
            200,
            200,
            273.23522915,
            235.11150117,
        ],
    )


def test_discrete_distribution_with_probabilities():
    model = parse_model(
        """\
choice ~ discrete(["low", "medium", "high"], probabilities=[0.2, 0.3, 0.5])
return choice
"""
    )

    samples = model.sample(9, random_state=42)

    np.testing.assert_array_equal(
        samples,
        ["medium", "high", "high", "high", "low", "low", "low", "high", "high"],
    )


def test_cumulative_distribution():
    model = parse_model(
        """\
value ~ cumulative([0, 0.2, 0.8, 1], [10, 15, 20, 25])
return value
"""
    )

    samples = model.sample(9, random_state=42)

    np.testing.assert_allclose(
        samples,
        [
            16.45450099,
            23.76785766,
            19.43328285,
            18.32215403,
            13.90046601,
            13.89986301,
            11.4520903,
            21.65440364,
            18.3426251,
        ],
    )


def test_multivariate_distribution_assignment():
    model = parse_model(
        """\
x, y ~ multivariate_normal(mean=[1, 2], cov=[[1, 0.5], [0.5, 1]])
return x + y
"""
    )

    samples = model.sample(99, random_state=42)

    assert samples.shape == (99,)
    np.testing.assert_allclose(samples.mean(), 3, atol=0.5)


@pytest.mark.parametrize(
    "source",
    [
        "value ~ empirical()\nreturn value",
        "value ~ empirical([])\nreturn value",
        "value ~ empirical([[1, 2]])\nreturn value",
        "value ~ empirical([1], data=[2])\nreturn value",
    ],
)
def test_empirical_distribution_requires_one_dimensional_data(source):
    with pytest.raises(ValueError, match=r"empirical\(\)"):
        parse_model(source)


def test_distribution_parameter_can_be_a_transform():
    transformed_model = parse_model(
        """\
zero = 0
nine = 9
location = (zero + (nine ** 0.5)) - log(2.718281828459045)
result ~ norm(loc=location)
return result
"""
    )
    plain_model = parse_model("result ~ norm(loc=2)\nreturn result")

    transformed_samples = transformed_model.sample(99, random_state=0)
    plain_samples = plain_model.sample(99, random_state=0)

    np.testing.assert_allclose(transformed_samples, plain_samples)


def test_pairwise_correlation():
    model = parse_model(
        """\
a ~ norm(loc=0, scale=1)
b ~ norm(loc=0, scale=1)
correlate a with b at 0.8
return a + b
"""
    )
    a, b = sampled_distributions(model)

    model.sample(999, random_state=42)

    np.testing.assert_allclose(np.corrcoef(a.samples_, b.samples_)[0, 1], 0.8, atol=0.075)


def test_matrix_correlation():
    model = parse_model(
        """\
a ~ norm()
b ~ norm()
c ~ norm()
correlate [a, b, c] with [[1, 0.5, -0.2], [0.5, 1, 0.3], [-0.2, 0.3, 1]]
return (a + b) + c
"""
    )
    a, b, c = sampled_distributions(model)

    model.sample(2_999, random_state=42)

    np.testing.assert_allclose(
        np.corrcoef(np.vstack([a.samples_, b.samples_, c.samples_])),
        [[1, 0.5, -0.2], [0.5, 1, 0.3], [-0.2, 0.3, 1]],
        atol=0.075,
    )


def test_disjoint_correlations_can_exceed_sample_size():
    model = parse_model(
        """\
a ~ norm()
b ~ norm()
c ~ norm()
d ~ norm()
e ~ norm()
f ~ norm()
correlate a with b at 0.5
correlate c with d at -0.5
correlate e with f at 0.3
ab = a + b
cd = c + d
ef = e + f
return (ab + cd) + ef
"""
    )

    samples = model.sample(size=5, random_state=42, method="lhs")

    assert samples.shape == (5,)


def test_overlapping_correlations_fail_when_sampled():
    model = parse_model(
        """\
a ~ norm()
b ~ norm()
c ~ norm()
correlate a with b at 0.5
correlate a with c at -0.5
return (a + b) + c
"""
    )

    with pytest.raises(ValueError):
        model.sample(size=5, random_state=42, method="lhs")


def test_dependent_sampled_nodes_cannot_be_correlated():
    source = """\
a ~ norm()
b ~ norm(loc=a)
correlate a with b at 0.8
return b
"""

    with pytest.raises(ValueError, match="Cannot correlate"):
        parse_model(source).sample(999, random_state=42)
