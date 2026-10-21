"""Model training, evaluation, persistence, and filename inference."""

import json
import os
import re
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = ROOT / "data" / "file_dataset.csv"
DEFAULT_MODEL_PATH = ROOT / "models" / "best_model.joblib"
DEFAULT_METRICS_PATH = ROOT / "reports" / "model_metrics.json"
DEFAULT_CONFUSION_MATRIX_PATH = ROOT / "reports" / "confusion_matrix.png"
RANDOM_STATE = 42
TEST_SIZE = 0.2
APPROVED_CATEGORIES = {
    "academic",
    "resume",
    "receipt",
    "presentation",
    "image",
    "other",
}
VARIANT_SUFFIXES = (
    "_2025_01",
    "_2025_06",
    "_copy_02",
    "_draft",
    "_final",
    "_revised",
    "_signed",
    "_v2",
)


def normalize_filename(file_name: str) -> str:
    """Turn common filename separators into spaces for useful TF-IDF tokens."""
    return re.sub(r"[^a-zA-Z0-9]+", " ", Path(file_name).stem).lower().strip()


def load_dataset(dataset_path: str | Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Load and validate the project's three-column labeled dataset."""
    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    data = pd.read_csv(dataset_path)
    required_columns = ["file_name", "extension", "category"]
    if list(data.columns) != required_columns:
        raise ValueError(
            f"Dataset columns must be exactly {required_columns}; "
            f"found {list(data.columns)}"
        )
    if data.empty or data[required_columns].isna().any().any():
        raise ValueError("Dataset must contain rows without missing values")
    if data["file_name"].duplicated().any():
        raise ValueError("Dataset contains duplicate filenames")
    unknown_categories = set(data["category"]) - APPROVED_CATEGORIES
    missing_categories = APPROVED_CATEGORIES - set(data["category"])
    if unknown_categories or missing_categories:
        raise ValueError(
            f"Dataset categories must include all approved labels only; "
            f"unknown={sorted(unknown_categories)}, "
            f"missing={sorted(missing_categories)}"
        )

    suffixes = data["file_name"].map(lambda name: Path(name).suffix.lower())
    if not suffixes.equals(data["extension"].str.lower()):
        raise ValueError("Each extension must match its filename suffix")
    if not data["extension"].str.startswith(".").all():
        raise ValueError("Extensions must include a leading dot")
    return data


def _group_key(file_name: str) -> str:
    stem = Path(file_name).stem.lower()
    for suffix in VARIANT_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def grouped_train_test_split(
    data: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratify filename families so related variants cannot leak across splits."""
    group_labels = data.assign(group=data["file_name"].map(_group_key))
    labels_per_group = group_labels.groupby("group")["category"].nunique()
    if (labels_per_group != 1).any():
        raise ValueError("A filename family has conflicting category labels")
    groups = group_labels[["group", "category"]].drop_duplicates()
    train_groups, test_groups = train_test_split(
        groups["group"],
        test_size=test_size,
        random_state=random_state,
        stratify=groups["category"],
    )
    group_values = group_labels["group"].to_numpy()
    train_set = set(train_groups)
    test_set = set(test_groups)
    train_indices = np.flatnonzero(np.isin(group_values, list(train_set)))
    test_indices = np.flatnonzero(np.isin(group_values, list(test_set)))
    if set(group_values[train_indices]) & set(group_values[test_indices]):
        raise RuntimeError("Filename families leaked between train and test splits")
    return train_indices, test_indices


def _make_pipeline(classifier, svd_components: int | None = None) -> Pipeline:
    steps = [
        (
            "tfidf",
            TfidfVectorizer(
                preprocessor=normalize_filename,
                lowercase=False,
                ngram_range=(1, 2),
                sublinear_tf=True,
            ),
        )
    ]
    if svd_components is not None:
        steps.append(
            (
                "svd",
                TruncatedSVD(n_components=svd_components, random_state=RANDOM_STATE),
            )
        )
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def _evaluate(y_true, y_pred, labels, file_names) -> dict:
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in labels
        },
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist(),
        "misclassifications": [
            {
                "file_name": file_name,
                "true_category": true_label,
                "predicted_category": predicted_label,
            }
            for file_name, true_label, predicted_label in zip(
                file_names,
                y_true,
                y_pred,
            )
            if true_label != predicted_label
        ],
    }


def train_and_evaluate(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    confusion_matrix_path: str | Path = DEFAULT_CONFUSION_MATRIX_PATH,
    svd_components: int = 100,
    feedback_path: str | Path | None = None,
    feedback_event_count: int = 0,
) -> dict:
    """Compare candidates, then refit the selected model with optional corrections."""
    data = load_dataset(dataset_path)
    feedback = None
    if feedback_path is not None:
        from src.feedback import load_feedback

        feedback = load_feedback(feedback_path)

    train_indices, test_indices = grouped_train_test_split(data)
    x_train = data.iloc[train_indices]["file_name"].tolist()
    y_train = data.iloc[train_indices]["category"].to_numpy()
    x_test = data.iloc[test_indices]["file_name"].tolist()
    y_test = data.iloc[test_indices]["category"].to_numpy()
    labels = sorted(data["category"].unique().tolist())

    if len(x_train) <= svd_components:
        raise ValueError(
            f"SVD components ({svd_components}) must be fewer than "
            f"training rows ({len(x_train)})"
        )

    candidates = {
        "tfidf_multinomial_nb": _make_pipeline(MultinomialNB()),
        "tfidf_logistic_regression": _make_pipeline(
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        ),
        "tfidf_svd_logistic_regression": _make_pipeline(
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            svd_components=svd_components,
        ),
    }
    evaluations = {}
    for name, pipeline in candidates.items():
        started = time.perf_counter()
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        elapsed_seconds = time.perf_counter() - started
        result = _evaluate(y_test, y_pred, labels, x_test)
        result["runtime_seconds"] = round(elapsed_seconds, 6)
        result["train_rows"] = len(train_indices)
        result["test_rows"] = len(test_indices)
        result["svd_components"] = (
            svd_components if name == "tfidf_svd_logistic_regression" else None
        )
        evaluations[name] = result

    best_name = max(
        evaluations,
        key=lambda name: (
            evaluations[name]["macro_f1"],
            evaluations[name]["weighted_f1"],
            evaluations[name]["accuracy"],
        ),
    )
    final_training_data = data
    if feedback is not None:
        feedback_names = set(feedback["file_name"].str.casefold())
        final_training_data = pd.concat(
            [
                data.loc[
                    ~data["file_name"].str.casefold().isin(feedback_names)
                ],
                feedback,
            ],
            ignore_index=True,
        )
    final_model = clone(candidates[best_name])
    final_model.fit(
        final_training_data["file_name"].tolist(),
        final_training_data["category"].to_numpy(),
    )
    final_model.sortiq_feedback_event_count = feedback_event_count

    metrics = {
        "dataset": {
            "path": str(Path(dataset_path)),
            "rows": len(data),
            "class_counts": {
                label: int(count)
                for label, count in Counter(data["category"]).items()
            },
            "provenance": "Synthetic generated filenames; not real-world observations.",
        },
        "evaluation": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "split_method": (
                "Stratified split of filename families; all suffix variants "
                "of a filename stem stay in one split."
            ),
            "labels": labels,
        },
        "models": evaluations,
        "selected_model": best_name,
        "feedback": {
            "used": feedback is not None,
            "correction_count": len(feedback) if feedback is not None else 0,
            "event_count": feedback_event_count,
            "final_training_rows": len(final_training_data),
            "retraining": (
                "Automatic after the configured correction threshold when a "
                "feedback file is supplied."
            ),
        },
    }

    model_path = Path(model_path)
    metrics_path = Path(metrics_path)
    confusion_matrix_path = Path(confusion_matrix_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths = []
    for path in (model_path, metrics_path, confusion_matrix_path):
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=path.suffix,
            delete=False,
        ) as temporary_file:
            temporary_paths.append(Path(temporary_file.name))
    model_temp, metrics_temp, confusion_matrix_temp = temporary_paths
    try:
        joblib.dump(final_model, model_temp)
        metrics_temp.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        _save_confusion_matrix(
            evaluations[best_name]["confusion_matrix"],
            labels,
            confusion_matrix_temp,
        )
        os.replace(metrics_temp, metrics_path)
        os.replace(confusion_matrix_temp, confusion_matrix_path)
        os.replace(model_temp, model_path)
    finally:
        for temporary_path in temporary_paths:
            if temporary_path.exists():
                temporary_path.unlink()
    return metrics


def _save_confusion_matrix(matrix, labels, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted category",
        ylabel="True category",
        title="Confusion Matrix — Selected Model",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = max(max(row) for row in matrix) / 2
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    """Load the persisted complete preprocessing-and-classifier pipeline."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Run python -m src.run_experiments to train it."
        )
    return joblib.load(model_path)


def predict_file_names(model, file_names: Iterable[str]) -> list[dict[str, object]]:
    """Predict filename categories and return optional class probabilities."""
    names = list(file_names)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("file_names must contain non-empty strings")
    if not names:
        return []

    predictions = model.predict(names)
    probabilities = (
        model.predict_proba(names) if hasattr(model, "predict_proba") else None
    )
    results = []
    for index, (file_name, predicted) in enumerate(zip(names, predictions)):
        result = {
            "file_name": file_name,
            "predicted_category": str(predicted),
        }
        if probabilities is not None:
            result["confidence"] = float(np.max(probabilities[index]))
        results.append(result)
    return results
