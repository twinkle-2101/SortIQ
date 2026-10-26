"""Category-to-folder mapping and destination path resolution."""

from pathlib import Path
from typing import Mapping

DEFAULT_CATEGORY_MAPPING: dict[str, str] = {
    "academic": "Academic",
    "resume": "Resumes",
    "receipt": "Receipts",
    "presentation": "Presentations",
    "image": "Images",
    "other": "Other",
}

CATEGORY_ICONS: dict[str, str] = {
    "academic": "🎓",
    "resume": "📄",
    "receipt": "🧾",
    "presentation": "📊",
    "image": "🖼️",
    "other": "📦",
}

CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "academic": "Course notes, lecture slides, textbooks, research papers, assignments",
    "resume": "CVs, resumes, cover letters, portfolios, job applications",
    "receipt": "Invoices, payment receipts, utility bills, tax documents, statements",
    "presentation": "Slide decks, pitch presentations, seminar talks, conference keynotes",
    "image": "Photos, wallpapers, screenshots, graphics, diagrams, illustrations",
    "other": "Software installers, logs, configuration files, misc downloads",
}


def get_destination_folder(
    category: str,
    custom_mapping: Mapping[str, str] | None = None,
) -> str:
    """Return the destination subfolder name for a predicted or overridden category.
    
    Falls back to 'Other' if the category is not recognized.
    """
    mapping = custom_mapping if custom_mapping is not None else DEFAULT_CATEGORY_MAPPING
    return mapping.get(category.lower(), mapping.get("other", "Other"))


def get_destination_path(
    target_base_dir: str | Path,
    category: str,
    file_name: str,
    custom_mapping: Mapping[str, str] | None = None,
) -> Path:
    """Resolve full intended destination path for a file inside its category folder."""
    subfolder = get_destination_folder(category, custom_mapping)
    return Path(target_base_dir) / subfolder / file_name
