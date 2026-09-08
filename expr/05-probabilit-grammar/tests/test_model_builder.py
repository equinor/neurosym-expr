from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from expr05.researcher import EvaluatedModelBuilder, ModelEvaluationError


class FixedModel:
    def __init__(self, predictions: list[int]) -> None:
        self.predictions = predictions

    def sample(self, size: int, random_state: int) -> list[int]:
        return self.predictions


class EvaluatedModelBuilderTests(unittest.TestCase):
    def test_accepts_model_with_matching_failure_rate(self) -> None:
        data = pd.DataFrame({"failed": [0] * 90 + [1] * 10})
        builder = EvaluatedModelBuilder(
            lambda _: "model",
            data,
            "failed",
            parser=lambda _: FixedModel([0] * 89 + [1] * 11),
            failure_rate_tolerance=0.01,
        )

        evaluation = builder("description")

        self.assertTrue(evaluation.accepted)
        self.assertAlmostEqual(evaluation.observed_failure_rate, 0.10)
        self.assertAlmostEqual(evaluation.modeled_failure_rate, 0.11)
        self.assertAlmostEqual(evaluation.absolute_error, 0.01)

    def test_rejects_model_outside_failure_rate_tolerance(self) -> None:
        data = pd.DataFrame({"failed": [0] * 90 + [1] * 10})
        builder = EvaluatedModelBuilder(
            lambda _: "model",
            data,
            "failed",
            parser=lambda _: FixedModel([0] * 75 + [1] * 25),
            failure_rate_tolerance=0.01,
        )

        evaluation = builder("description")

        self.assertFalse(evaluation.accepted)
        self.assertAlmostEqual(evaluation.absolute_error, 0.15)
        self.assertIn("models_failure_rate: no", str(evaluation))

    def test_rejects_non_binary_model_output(self) -> None:
        data = pd.DataFrame({"failed": [0, 0, 1]})
        builder = EvaluatedModelBuilder(
            lambda _: "model",
            data,
            "failed",
            parser=lambda _: FixedModel([0, 0.5, 1]),
        )

        with self.assertRaisesRegex(
            ModelEvaluationError, "Generated model values must all be binary"
        ):
            builder("description")

    def test_rejects_non_binary_target(self) -> None:
        data = pd.DataFrame({"failed": np.array([0, 0.5, 1])})

        with self.assertRaisesRegex(ValueError, "Target values must all be binary"):
            EvaluatedModelBuilder(lambda _: "model", data, "failed")


if __name__ == "__main__":
    unittest.main()
