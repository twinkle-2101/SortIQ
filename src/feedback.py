"""Local, user-reviewed corrections with batched automatic model retraining."""

import json
import uuid
from pathlib import Path

import pandas as pd

from src.ml_pipeline import (
    APPROVED_CATEGORIES,
    DEFAULT_CONFUSION_MATRIX_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    train_and_evaluate,
)


FEEDBACK_COLUMNS = ["file_name", "extension", "category"]
DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data"
DEFAULT_FEEDBACK_PATH = DATA_DIRECTORY / "user_feedback.csv"
DEFAULT_FEEDBACK_EVENTS_PATH = DATA_DIRECTORY / "user_feedback_events.jsonl"
RETRAIN_AFTER_CORRECTIONS = 5


def load_feedback(feedback_path: str | Path) -> pd.DataFrame:
    """Load correction records and validate the shared three-column contract."""
    feedback_path = Path(feedback_path)
    if not feedback_path.is_file():
        raise FileNotFoundError(f"Feedback file not found: {feedback_path}")
    feedback = pd.read_csv(feedback_path)
    if list(feedback.columns) != FEEDBACK_COLUMNS:
        raise ValueError(
            f"Feedback columns must be exactly {FEEDBACK_COLUMNS}; "
            f"found {list(feedback.columns)}"
        )
    if feedback.empty or feedback[FEEDBACK_COLUMNS].isna().any().any():
        raise ValueError("Feedback must contain rows without missing values")
    if feedback["file_name"].str.casefold().duplicated(keep=False).any():
        raise ValueError("Feedback contains duplicate filenames")
    if feedback["file_name"].map(lambda name: not _is_bare_filename(name)).any():
        raise ValueError("Feedback file_name values must be filenames, not paths")
    if feedback["extension"].str.lower().ne(feedback["extension"]).any():
        raise ValueError("Feedback extensions must be lowercase")
    suffixes = feedback["file_name"].map(lambda name: Path(name).suffix.lower())
    if not suffixes.equals(feedback["extension"]):
        raise ValueError("Each feedback extension must match its filename suffix")
    if not feedback["extension"].str.startswith(".").all():
        raise ValueError("Feedback extensions must include a leading dot")
    unknown_categories = set(feedback["category"]) - APPROVED_CATEGORIES
    if unknown_categories:
        raise ValueError(
            f"Feedback contains unapproved categories: {sorted(unknown_categories)}"
        )
    return feedback


def save_correction(
    file_name: str,
    category: str,
    feedback_path: str | Path = DEFAULT_FEEDBACK_PATH,
) -> int:
    """Add or replace one reviewed label in a local CSV; return its row count."""
    if not _is_bare_filename(file_name):
        raise ValueError("file_name must be a filename, not a path")
    if not Path(file_name).suffix:
        raise ValueError("file_name must include an extension")
    if category not in APPROVED_CATEGORIES:
        raise ValueError(f"category must be one of {sorted(APPROVED_CATEGORIES)}")

    feedback_path = Path(feedback_path)
    if feedback_path.is_file():
        feedback = load_feedback(feedback_path)
    else:
        feedback = pd.DataFrame(columns=FEEDBACK_COLUMNS)

    normalized_name = file_name.casefold()
    existing_names = feedback["file_name"].str.casefold()
    feedback = feedback.loc[existing_names != normalized_name]
    correction = pd.DataFrame(
        [
            {
                "file_name": file_name,
                "extension": Path(file_name).suffix.lower(),
                "category": category,
            }
        ]
    )
    updated = pd.concat([feedback, correction], ignore_index=True)
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(feedback_path, index=False)
    return len(updated)


def record_correction(
    file_name: str,
    category: str,
    feedback_path: str | Path = DEFAULT_FEEDBACK_PATH,
    events_path: str | Path = DEFAULT_FEEDBACK_EVENTS_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    confusion_matrix_path: str | Path = DEFAULT_CONFUSION_MATRIX_PATH,
    retrain_after: int = RETRAIN_AFTER_CORRECTIONS,
) -> dict[str, object]:
    """Record a confirmed correction and retrain after enough new corrections."""
    if retrain_after < 1:
        raise ValueError("retrain_after must be at least 1")

    feedback_path = Path(feedback_path)
    events_path = Path(events_path)
    model_path = Path(model_path)
    metrics_path = Path(metrics_path)
    confusion_matrix_path = Path(confusion_matrix_path)
    save_correction(file_name, category, feedback_path)

    events_path.parent.mkdir(parents=True, exist_ok=True)
    latest_category_by_filename = {}
    correction_count = 0
    if events_path.exists():
        with events_path.open(encoding="utf-8") as event_file:
            for line_number, line in enumerate(event_file, start=1):
                if not line.strip():
                    continue
                try:
                    previous_event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid feedback event at {events_path}:{line_number}"
                    ) from exc
                if (
                    not isinstance(previous_event, dict)
                    or not isinstance(previous_event.get("file_name"), str)
                    or not isinstance(previous_event.get("category"), str)
                ):
                    raise ValueError(
                        f"Invalid feedback event at {events_path}:{line_number}"
                    )
                latest_category_by_filename[
                    previous_event["file_name"].casefold()
                ] = previous_event["category"].casefold()
                correction_count += 1

    if latest_category_by_filename.get(file_name.casefold()) != category.casefold():
        event = {
            "correction_id": uuid.uuid4().hex,
            "file_name": file_name,
            "extension": Path(file_name).suffix.lower(),
            "category": category,
        }
        with events_path.open("a", encoding="utf-8", newline="\n") as event_file:
            event_file.write(json.dumps(event) + "\n")
            event_file.flush()
        correction_count += 1

    trained_count = 0
    if model_path.is_file():
        import joblib

        previous_model = joblib.load(model_path)
        trained_count = int(
            getattr(previous_model, "sortiq_feedback_event_count", 0)
        )
        if trained_count > correction_count:
            raise ValueError(
                "The feedback event log has fewer corrections than the saved model "
                "records as trained. Restore the local event log before recording "
                "more feedback."
            )

    pending_corrections = correction_count - trained_count
    if pending_corrections < retrain_after:
        return {
            "correction_count": correction_count,
            "pending_corrections": pending_corrections,
            "retrained": False,
        }

    train_and_evaluate(
        feedback_path=feedback_path,
        model_path=model_path,
        metrics_path=metrics_path,
        confusion_matrix_path=confusion_matrix_path,
        feedback_event_count=correction_count,
    )
    return {
        "correction_count": correction_count,
        "pending_corrections": 0,
        "retrained": True,
    }


def _is_bare_filename(file_name: object) -> bool:
    return (
        isinstance(file_name, str)
        and bool(file_name.strip())
        and file_name in {Path(file_name).name, Path(file_name).name.strip()}
        and "/" not in file_name
        and "\\" not in file_name
    )
