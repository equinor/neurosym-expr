from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from expr.lang import parse_model


RANDOM_STATE = 15_205
DEFAULT_FAILURE_RATE_TOLERANCE = 0.01


class SampleableModel(Protocol):
    def sample(self, size: int, random_state: int) -> object: ...


class ModelEvaluationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelEvaluation:
    accepted: bool
    observed_failure_rate: float
    modeled_failure_rate: float
    absolute_error: float
    tolerance: float
    sample_size: int

    def __str__(self) -> str:
        return "\n".join(
            [
                f"models_failure_rate: {'yes' if self.accepted else 'no'}",
                f"observed_failure_rate: {self.observed_failure_rate:.6f}",
                f"modeled_failure_rate: {self.modeled_failure_rate:.6f}",
                f"absolute_error: {self.absolute_error:.6f}",
                f"required_tolerance: {self.tolerance:.6f}",
                f"sample_size: {self.sample_size}",
            ]
        )


class EvaluatedModelBuilder:
    def __init__(
        self,
        generator: Callable[[str], str],
        data: pd.DataFrame,
        target: str,
        parser: Callable[[str], SampleableModel] = parse_model,
        failure_rate_tolerance: float = DEFAULT_FAILURE_RATE_TOLERANCE,
    ) -> None:
        if target not in data:
            raise ValueError(f"Target column {target!r} is not present in the CSV")
        if not 0 <= failure_rate_tolerance <= 1:
            raise ValueError("Failure-rate tolerance must be between 0 and 1")

        self.generator = generator
        self.target = np.asarray(data[target])
        self.parser = parser
        self.failure_rate_tolerance = failure_rate_tolerance
        self._validate_binary_values(self.target, "Target")

    @staticmethod
    def _validate_binary_values(values: np.ndarray, name: str) -> None:
        try:
            valid = np.isin(values, [0, 1]).all()
        except TypeError:
            valid = False
        if not valid:
            raise ValueError(f"{name} values must all be binary (0 or 1)")

    def __call__(self, description: str) -> ModelEvaluation:
        try:
            source = self.generator(description)
            model = self.parser(source)
            predictions = np.asarray(
                model.sample(size=len(self.target), random_state=RANDOM_STATE)
            )
            if predictions.shape != self.target.shape:
                raise ValueError(
                    "The generated model must return one prediction per data row"
                )
            self._validate_binary_values(predictions, "Generated model")

            observed_failure_rate = float(np.mean(self.target))
            modeled_failure_rate = float(np.mean(predictions))
            absolute_error = abs(modeled_failure_rate - observed_failure_rate)
            return ModelEvaluation(
                accepted=absolute_error <= self.failure_rate_tolerance,
                observed_failure_rate=observed_failure_rate,
                modeled_failure_rate=modeled_failure_rate,
                absolute_error=absolute_error,
                tolerance=self.failure_rate_tolerance,
                sample_size=len(self.target),
            )
        except Exception as error:
            raise ModelEvaluationError(
                "Generated model failed evaluation: "
                f"{type(error).__name__}: {error}"
            ) from None
