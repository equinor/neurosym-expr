from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_DATASET = Path("res/ai4i2020.csv")
TARGET = "Machine failure"
EXCLUDED_COLUMNS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit logistic regression to predict machine failure."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"CSV dataset path (default: {DEFAULT_DATASET})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.dataset)

    features = data.drop(columns=[TARGET, *EXCLUDED_COLUMNS])
    target = data[TARGET]
    categorical_columns = features.select_dtypes(exclude="number").columns.tolist()
    numeric_columns = features.select_dtypes(include="number").columns.tolist()

    preprocessing = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
        ]
    )
    model = make_pipeline(
        preprocessing,
        LogisticRegression(class_weight="balanced", max_iter=1_000),
    )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=15205,
        stratify=target,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print(
        classification_report(
            y_test, predictions, target_names=["no failure", "failure"]
        )
    )
