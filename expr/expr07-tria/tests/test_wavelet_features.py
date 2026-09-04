import torch

from expr07_tria.wavelet_features import make_haar_features, project_haar


def test_haar_features_have_expected_supports_and_values() -> None:
    features = make_haar_features(8, max_level=3)
    values = torch.arange(8, dtype=torch.float64).unsqueeze(0)
    projected = project_haar(values, features.starts, features.widths)

    assert features.starts.tolist() == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        0,
        2,
        4,
        6,
        0,
        4,
        0,
    ]
    assert features.widths.tolist() == [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        4,
        4,
        8,
    ]
    assert torch.equal(projected[0, :8], values[0])
    assert torch.allclose(
        projected[0, 8:],
        torch.tensor(
            [
                -0.5**0.5,
                -0.5**0.5,
                -0.5**0.5,
                -0.5**0.5,
                -2.0,
                -2.0,
                -(32.0**0.5),
            ],
            dtype=torch.float64,
        ),
    )


def test_haar_features_can_be_filtered_for_causal_support() -> None:
    features = make_haar_features(8)
    component = 5
    admissible = features.ends <= component

    assert torch.all(features.ends[admissible] <= component)
    assert not torch.any(
        (features.starts[admissible] <= component)
        & (features.ends[admissible] > component)
    )
