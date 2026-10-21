"""Local, user-reviewed category corrections for explicit model retraining."""

from pathlib import Path

import pandas as pd

from src.ml_pipeline import APPROVED_CATEGORIES


FEEDBACK_COLUMNS = ["file_name", "extension", "category"]
DEFAULT_FEEDBACK_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "user_feedback.csv"
)


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


def _is_bare_filename(file_name: object) -> bool:
    return (
        isinstance(file_name, str)
        and bool(file_name.strip())
        and file_name in {Path(file_name).name, Path(file_name).name.strip()}
        and "/" not in file_name
        and "\\" not in file_name
    )
