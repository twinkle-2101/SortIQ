"""Helpers for limiting review and feedback to ambiguous predictions."""

from collections.abc import Iterable, Mapping


def category_for_file(
    file_name: str,
    predicted_category: str,
    confidence: float,
    confidence_threshold: float,
    user_overrides: Mapping[str, str],
) -> str:
    """Use an override only when the prediction is flagged for review."""
    if confidence < confidence_threshold:
        return user_overrides.get(file_name, predicted_category)
    return predicted_category


def get_pending_corrections(
    file_names: Iterable[str],
    predictions: Mapping[str, Mapping[str, object]],
    user_overrides: Mapping[str, str],
    submitted_feedback: Mapping[str, str],
    confidence_threshold: float,
) -> dict[str, str]:
    """Return changed, unsubmitted overrides for ambiguous files only."""
    pending = {}
    for file_name in file_names:
        override = user_overrides.get(file_name)
        prediction = predictions.get(file_name, {})
        predicted_category = prediction.get("predicted_category")
        confidence = prediction.get("confidence", 0.5)
        if (
            override is not None
            and confidence < confidence_threshold
            and override != predicted_category
            and submitted_feedback.get(file_name) != override
        ):
            pending[file_name] = override
    return pending
