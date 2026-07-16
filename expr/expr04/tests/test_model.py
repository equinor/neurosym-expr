import numpy as np
import pytest
from probabilit import Distribution

from expr import ModelConversionError, model_from_tree, parse_model, parser


def test_converts_sampled_and_computed_variables_to_model():
    model = parse_model(
        """\
roll ~ uniform(loc=1, scale=6)
die = floor(roll)
return die == 3
"""
    )

    samples = model.sample(500, random_state=42)

    assert samples.dtype == np.dtype(bool)
    assert 0 < samples.sum() < len(samples)


def test_converts_an_existing_parse_tree():
    model = model_from_tree(parser.parse("x ~ norm()\nreturn x"))

    assert isinstance(model, Distribution)
    assert model.distr == "norm"


def test_numpy_functions_accept_keyword_arguments():
    model = parse_model("x ~ norm()\nreturn clip(x, a_min=-1, a_max=1)")

    samples = model.sample(25, random_state=42)

    assert np.all((-1 <= samples) & (samples <= 1))


def test_string_comparison_uses_a_probabilit_constant():
    model = parse_model(
        'choice ~ randint(low=0, high=2)\nreturn choice == "accepted"'
    )

    assert "Constant(accepted)" in repr(model)


def test_applies_correlation_to_returned_model():
    model = parse_model(
        """\
x ~ norm()
y ~ norm()
correlate x with y at 0.75
return x + y
"""
    )

    assert len(model._correlations) == 1
    assert np.array_equal(
        model._correlations[0][1], np.array([[1.0, 0.75], [0.75, 1.0]])
    )


def test_applies_matrix_correlation_to_returned_model():
    matrix = np.array(
        [
            [1.0, 0.2, -0.3],
            [0.2, 1.0, 0.4],
            [-0.3, 0.4, 1.0],
        ]
    )
    model = parse_model(
        """\
x ~ norm()
y ~ norm()
z ~ norm()
correlate [x, y, z] with [[1, 0.2, -0.3], [0.2, 1, 0.4], [-0.3, 0.4, 1]]
return (x + y) + z
"""
    )

    assert len(model._correlations) == 1
    assert np.array_equal(model._correlations[0][1], matrix)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("return missing", "Unknown variable"),
        ("x ~ norm()\ncorrelate x with x at 0.5\nreturn x", "itself"),
        (
            "x ~ norm()\ny ~ norm()\ncorrelate x with y at 1.5\nreturn x + y",
            "between -1 and 1",
        ),
        (
            "x ~ norm()\ny ~ norm()\n"
            "correlate [x, y] with [[1, 0.5, 0], [0.5, 1, 0]]\n"
            "return x + y",
            "shape",
        ),
        ("return unknown_function(1)", "Unknown NumPy function"),
    ],
)
def test_reports_invalid_models(source, message):
    with pytest.raises(ModelConversionError, match=message):
        parse_model(source)
