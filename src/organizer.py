"""Safe file organizer with dry-run planning, collision resolution, and explicit confirmation."""

import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from src.category_mapping import get_destination_folder

COLLISION_RENAME = "rename"
COLLISION_SKIP = "skip"
ALLOWED_COLLISION_POLICIES = {COLLISION_RENAME, COLLISION_SKIP}


@dataclass
class MovePlanItem:
    """Represents a planned file organization step."""
    file_name: str
    source_path: Path
    category: str
    destination_dir: Path
    intended_destination_path: Path
    resolved_destination_path: Path
    collision_detected: bool
    status: str  # "ready", "rename", "skip", "already_in_place", "error"
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert item to a serializable dictionary."""
        data = asdict(self)
        data["source_path"] = str(self.source_path)
        data["destination_dir"] = str(self.destination_dir)
        data["intended_destination_path"] = str(self.intended_destination_path)
        data["resolved_destination_path"] = str(self.resolved_destination_path)
        return data


def resolve_collision_path(
    destination_path: Path,
    existing_reserved_paths: set[Path] | None = None,
) -> Path:
    """Generate a collision-free path by appending a numeric counter (e.g., 'file (1).ext').
    
    Checks both existing files on disk and paths already reserved in the current batch.
    """
    reserved = existing_reserved_paths or set()
    parent = destination_path.parent
    stem = destination_path.stem
    suffix = destination_path.suffix

    counter = 1
    candidate = destination_path
    while candidate.exists() or candidate in reserved:
        candidate = parent / f"{stem} ({counter}){suffix}"
        counter += 1

    return candidate


def plan_organization(
    items: list[dict[str, Any]],
    destination_root: str | Path,
    collision_policy: str = COLLISION_RENAME,
    custom_mapping: Mapping[str, str] | None = None,
) -> list[MovePlanItem]:
    """Generate a preview/dry-run plan for organizing files.
    
    This function is strictly read-only and will NEVER move or modify files.
    
    Args:
        items: List of dictionaries with 'file_name', 'source_path' (or full_path), and 'category'.
        destination_root: Root directory where category subfolders will be placed.
        collision_policy: 'rename' or 'skip'.
        custom_mapping: Optional category to folder mapping override.
        
    Returns:
        List of MovePlanItem objects detailing planned actions.
    """
    if collision_policy not in ALLOWED_COLLISION_POLICIES:
        raise ValueError(
            f"Invalid collision policy '{collision_policy}'. "
            f"Must be one of {sorted(ALLOWED_COLLISION_POLICIES)}"
        )

    root_dir = Path(destination_root).expanduser().resolve()
    plan: list[MovePlanItem] = []
    reserved_targets: set[Path] = set()

    for item in items:
        file_name = item["file_name"]
        source_path = Path(item.get("source_path", item.get("full_path", ""))).resolve()
        category = item.get("category", "other")
        folder_name = get_destination_folder(category, custom_mapping)
        dest_dir = (root_dir / folder_name).resolve()
        intended_target = (dest_dir / file_name).resolve()

        # Validation 1: Check source file existence
        if not source_path.is_file():
            plan.append(
                MovePlanItem(
                    file_name=file_name,
                    source_path=source_path,
                    category=category,
                    destination_dir=dest_dir,
                    intended_destination_path=intended_target,
                    resolved_destination_path=intended_target,
                    collision_detected=False,
                    status="error",
                    error_message=f"Source file does not exist: {source_path}",
                )
            )
            continue

        # Validation 2: Prevent moving file into itself
        if source_path == intended_target:
            plan.append(
                MovePlanItem(
                    file_name=file_name,
                    source_path=source_path,
                    category=category,
                    destination_dir=dest_dir,
                    intended_destination_path=intended_target,
                    resolved_destination_path=intended_target,
                    collision_detected=False,
                    status="already_in_place",
                    error_message="File is already in the target destination folder.",
                )
            )
            continue

        # Check for destination collision (on disk or already reserved in this batch)
        is_collision = intended_target.exists() or intended_target in reserved_targets

        if is_collision:
            if collision_policy == COLLISION_SKIP:
                plan.append(
                    MovePlanItem(
                        file_name=file_name,
                        source_path=source_path,
                        category=category,
                        destination_dir=dest_dir,
                        intended_destination_path=intended_target,
                        resolved_destination_path=intended_target,
                        collision_detected=True,
                        status="skip",
                        error_message="Destination file already exists (skipped per collision policy).",
                    )
                )
                continue
            elif collision_policy == COLLISION_RENAME:
                resolved_target = resolve_collision_path(intended_target, reserved_targets)
                reserved_targets.add(resolved_target)
                plan.append(
                    MovePlanItem(
                        file_name=file_name,
                        source_path=source_path,
                        category=category,
                        destination_dir=dest_dir,
                        intended_destination_path=intended_target,
                        resolved_destination_path=resolved_target,
                        collision_detected=True,
                        status="rename",
                        error_message="",
                    )
                )
                continue
        else:
            reserved_targets.add(intended_target)
            plan.append(
                MovePlanItem(
                    file_name=file_name,
                    source_path=source_path,
                    category=category,
                    destination_dir=dest_dir,
                    intended_destination_path=intended_target,
                    resolved_destination_path=intended_target,
                    collision_detected=False,
                    status="ready",
                    error_message="",
                )
            )

    return plan


def execute_move(
    plan: list[MovePlanItem],
    confirmed: bool = False,
) -> dict[str, Any]:
    """Execute the pre-computed organization plan safely.
    
    CRITICAL: Never moves any file unless confirmed=True.
    
    Args:
        plan: The MovePlanItem list generated from plan_organization.
        confirmed: Explicit boolean user confirmation.
        
    Returns:
        Execution summary dictionary with moved, skipped, and error lists.
        
    Raises:
        ValueError: If confirmed is not True.
    """
    if not confirmed:
        raise ValueError(
            "Move execution rejected: explicit confirmation (confirmed=True) is required."
        )

    results: dict[str, Any] = {
        "total_requested": len(plan),
        "moved_count": 0,
        "skipped_count": 0,
        "failed_count": 0,
        "moved": [],
        "skipped": [],
        "errors": [],
    }

    for item in plan:
        if item.status in ("skip", "already_in_place"):
            results["skipped_count"] += 1
            results["skipped"].append(
                {
                    "file_name": item.file_name,
                    "reason": item.error_message or item.status,
                    "source": str(item.source_path),
                }
            )
            continue

        if item.status == "error":
            results["failed_count"] += 1
            results["errors"].append(
                {
                    "file_name": item.file_name,
                    "error": item.error_message,
                    "source": str(item.source_path),
                }
            )
            continue

        # Execute the move
        try:
            target_path = item.resolved_destination_path
            # Ensure destination directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Extra safety check: destination must still not exist (avoid race condition)
            if target_path.exists():
                results["failed_count"] += 1
                results["errors"].append(
                    {
                        "file_name": item.file_name,
                        "error": f"Collision safeguard prevented overwrite: {target_path} already exists.",
                        "source": str(item.source_path),
                    }
                )
                continue

            shutil.move(str(item.source_path), str(target_path))
            results["moved_count"] += 1
            results["moved"].append(
                {
                    "file_name": item.file_name,
                    "source": str(item.source_path),
                    "destination": str(target_path),
                    "renamed": item.collision_detected,
                }
            )
        except (PermissionError, OSError) as exc:
            results["failed_count"] += 1
            results["errors"].append(
                {
                    "file_name": item.file_name,
                    "error": f"Failed to move: {exc}",
                    "source": str(item.source_path),
                }
            )

    return results
