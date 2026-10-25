"""SortIQ: Smart Downloads Folder Organizer - Streamlit Web Application."""

import json
import os
import shutil
from pathlib import Path
from typing import Any

import streamlit as st

from src.category_mapping import (
    CATEGORY_DESCRIPTIONS,
    CATEGORY_ICONS,
    DEFAULT_CATEGORY_MAPPING,
    get_destination_folder,
)
from src.feedback import record_correction
from src.file_scanner import format_file_size, scan_directory, validate_directory_path
from src.ml_pipeline import (
    APPROVED_CATEGORIES,
    DEFAULT_CONFUSION_MATRIX_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    load_model,
    predict_file_names,
)
from src.organizer import (
    COLLISION_RENAME,
    COLLISION_SKIP,
    execute_move,
    plan_organization,
)

# Page configuration
st.set_page_config(
    page_title="SortIQ — Smart Folder Organizer",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, modern aesthetics
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .badge-academic { background-color: #1E3A8A; color: #93C5FD; padding: 4px 8px; border-radius: 6px; font-weight: 600; }
    .badge-resume { background-color: #4C1D95; color: #C4B5FD; padding: 4px 8px; border-radius: 6px; font-weight: 600; }
    .badge-receipt { background-color: #064E3B; color: #6EE7B7; padding: 4px 8px; border-radius: 6px; font-weight: 600; }
    .badge-presentation { background-color: #78350F; color: #FCD34D; padding: 4px 8px; border-radius: 6px; font-weight: 600; }
    .badge-image { background-color: #164E63; color: #67E8F9; padding: 4px 8px; border-radius: 6px; font-weight: 600; }
    .badge-other { background-color: #334155; color: #CBD5E1; padding: 4px 8px; border-radius: 6px; font-weight: 600; }
    .badge-ambiguous { background-color: #991B1B; color: #FCA5A5; padding: 3px 6px; border-radius: 4px; font-size: 0.8rem; }
    .badge-confident { background-color: #065F46; color: #A7F3D0; padding: 3px 6px; border-radius: 4px; font-size: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_cached_model():
    """Load and cache the trained ML pipeline."""
    try:
        return load_model(), None
    except Exception as exc:
        return None, str(exc)


def load_model_metrics() -> dict[str, Any] | None:
    """Load model performance metrics if available."""
    if DEFAULT_METRICS_PATH.is_file():
        try:
            return json.loads(DEFAULT_METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def create_demo_folder() -> Path:
    """Generate a disposable demo folder with realistic unorganized files for safe testing."""
    demo_dir = Path(__file__).resolve().parent / "data" / "demo_downloads"
    demo_dir.mkdir(parents=True, exist_ok=True)

    sample_files = {
        "algorithms_lecture_unit3.pdf": "Academic algorithms notes",
        "cs204_dbms_assignment_final.pdf": "Database management assignment",
        "jordan_smith_data_scientist_resume.pdf": "Data scientist resume",
        "sarah_connor_cover_letter.docx": "Cover letter document",
        "target_store_order_receipt_2025.pdf": "Store purchase receipt",
        "aws_cloud_monthly_invoice.pdf": "Cloud hosting invoice",
        "q3_executive_pitch_deck.pptx": "Quarterly pitch presentation",
        "deep_learning_seminar_talk.pptx": "Seminar slide deck",
        "yosemite_national_park_sunset.jpg": "Scenic nature photo",
        "screenshot_error_debug.png": "Bug screenshot",
        "docker_desktop_installer.dmg": "Docker installer package",
        "nginx_configuration_backup.conf": "Server config file",
    }

    for filename, content in sample_files.items():
        file_path = demo_dir / filename
        if not file_path.exists():
            file_path.write_text(content, encoding="utf-8")

    return demo_dir


def rule_based_fallback_predict(file_names: list[str]) -> list[dict[str, Any]]:
    """Heuristic fallback prediction when the ML model artifact is missing."""
    results = []
    academic_keywords = ["notes", "lecture", "unit", "assignment", "thesis", "study", "syllabus"]
    resume_keywords = ["resume", "cv", "curriculum", "cover_letter", "profile"]
    receipt_keywords = ["receipt", "invoice", "bill", "tax", "order", "statement"]
    presentation_keywords = ["presentation", "slides", "deck", "talk", "keynote", "seminar"]
    image_exts = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]

    for name in file_names:
        lower = name.lower()
        suffix = Path(name).suffix.lower()

        if suffix in image_exts:
            cat = "image"
        elif any(k in lower for k in resume_keywords):
            cat = "resume"
        elif any(k in lower for k in receipt_keywords):
            cat = "receipt"
        elif any(k in lower for k in academic_keywords):
            cat = "academic"
        elif suffix in [".ppt", ".pptx"] or any(k in lower for k in presentation_keywords):
            cat = "presentation"
        else:
            cat = "other"

        results.append({
            "file_name": name,
            "predicted_category": cat,
            "confidence": 0.50,
        })
    return results


# Header
st.markdown('<div class="main-header">SortIQ 🧹</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Intelligent Downloads & Folder Cleaner with Active ML Learning and Safe Dry-Run</div>',
    unsafe_allow_html=True,
)

# Load model & metrics
model, model_error = get_cached_model()
metrics = load_model_metrics()

# Sidebar: Controls, Model Info, Metrics, Safety
with st.sidebar:
    st.header("⚙️ Configuration & Model")

    # Model status
    if model is not None:
        selected_model_name = metrics.get("selected_model", "Trained Pipeline") if metrics else "Active Pipeline"
        st.success(f"● Model Active: `{selected_model_name}`")
        if metrics and "models" in metrics and selected_model_name in metrics["models"]:
            selected_perf = metrics["models"][selected_model_name]
            acc = selected_perf.get("accuracy", 0.0)
            f1 = selected_perf.get("macro_f1", 0.0)
            st.metric("Holdout Accuracy", f"{acc * 100:.1f}%", help="Evaluated on held-out filename families.")
            st.metric("Macro F1-Score", f"{f1 * 100:.1f}%")
    else:
        st.warning("⚠️ ML Model Not Found")
        st.caption(
            f"Error: {model_error}\n\nRun `python -m src.run_experiments` to train and save the pipeline."
        )

    st.markdown("---")
    st.subheader("🛡️ Safety Settings")
    confidence_threshold = st.slider(
        "Ambiguity Threshold",
        min_value=0.40,
        max_value=0.95,
        value=0.65,
        step=0.05,
        help="Predictions below this confidence level are flagged for human review.",
    )

    collision_policy_label = st.selectbox(
        "Name Collision Policy",
        options=["Rename colliding files (file (1).ext)", "Skip colliding files"],
        index=0,
        help="How to handle existing files with the same name in the destination directory.",
    )
    collision_policy = (
        COLLISION_RENAME if "Rename" in collision_policy_label else COLLISION_SKIP
    )

    include_hidden = st.checkbox(
        "Include hidden files (.*)",
        value=False,
        help="Excludes hidden system files (.DS_Store, .git) by default for safety.",
    )

    st.markdown("---")
    st.subheader("🧪 Safe Demo Sandbox")
    st.caption("Generate disposable sample files to test SortIQ without touching your real Downloads.")
    if st.button("Generate Demo Folder", use_container_width=True):
        demo_path = create_demo_folder()
        st.session_state["scan_folder_input"] = str(demo_path)
        st.success(f"Created demo folder with 12 sample files at: `{demo_path.name}`")

    # Metrics & Confusion Matrix expander
    if metrics or DEFAULT_CONFUSION_MATRIX_PATH.is_file():
        with st.expander("📊 View ML Metrics & SVD Evaluation"):
            if metrics:
                st.write("**Model Comparison on Holdout Split:**")
                comp_data = []
                for m_name, m_data in metrics.get("models", {}).items():
                    comp_data.append({
                        "Model": m_name,
                        "Accuracy": f"{m_data.get('accuracy', 0)*100:.1f}%",
                        "Macro-F1": f"{m_data.get('macro_f1', 0)*100:.1f}%",
                        "SVD Components": m_data.get("svd_components") or "None",
                        "Runtime (s)": m_data.get("runtime_seconds", 0),
                    })
                st.dataframe(comp_data, use_container_width=True)

            if DEFAULT_CONFUSION_MATRIX_PATH.is_file():
                st.image(
                    str(DEFAULT_CONFUSION_MATRIX_PATH),
                    caption="Confusion Matrix — Best Model (TF-IDF + SVD + Logistic Regression)",
                    use_container_width=True,
                )


# Step 1: Directory Selection & Scan
st.subheader("1️⃣ Select Folder to Organize")
default_scan_dir = st.session_state.get(
    "scan_folder_input",
    str(Path.home() / "Downloads"),
)

folder_col, scan_col = st.columns([4, 1])
with folder_col:
    folder_input = st.text_input(
        "Folder Path:",
        value=default_scan_dir,
        placeholder="/path/to/folder or ~/Downloads",
        label_visibility="collapsed",
    )
with scan_col:
    scan_button = st.button("🔍 Scan Folder", use_container_width=True, type="primary")

# Maintain state
if scan_button or "scanned_files" not in st.session_state:
    is_valid, err_msg = validate_directory_path(folder_input)
    if not is_valid:
        st.error(f"❌ {err_msg}")
        st.session_state["scanned_files"] = []
    else:
        try:
            scanned = scan_directory(
                folder_input,
                recursive=False,
                include_hidden=include_hidden,
            )
            st.session_state["scanned_files"] = scanned
            st.session_state["current_scan_path"] = str(Path(folder_input).expanduser().resolve())
            # Reset overrides when a new scan occurs
            st.session_state["user_overrides"] = {}
            st.session_state["move_confirmed"] = False
        except Exception as exc:
            st.error(f"Scan failed: {exc}")
            st.session_state["scanned_files"] = []

scanned_files = st.session_state.get("scanned_files", [])
current_path = st.session_state.get("current_scan_path", folder_input)

if not scanned_files:
    st.info("💡 Enter a valid folder path or click 'Generate Demo Folder' in the sidebar to start scanning.")
else:
    total_size = sum(f.size_bytes for f in scanned_files)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Files Found", len(scanned_files))
    with col2:
        st.metric("Total Size", format_file_size(total_size))
    with col3:
        unique_exts = len({f.extension for f in scanned_files if f.extension})
        st.metric("File Extensions", unique_exts)
    with col4:
        st.metric("Scan Mode", "Read-Only (Direct Files)")

    # Step 2: Prediction & Interactive Review
    st.markdown("---")
    st.subheader("2️⃣ Review File Predictions & Destination Folders")
    st.caption(
        "SortIQ predicts categories based on filename tokens. Ambiguous predictions are highlighted for your review. "
        "You can override any recommendation and optionally save confirmed corrections to trigger automatic retraining."
    )

    # Compute predictions
    file_names = [f.file_name for f in scanned_files]
    if model is not None:
        predictions = predict_file_names(model, file_names)
    else:
        st.warning("⚡ ML model not loaded; using rule-based fallback predictions.")
        predictions = rule_based_fallback_predict(file_names)

    pred_lookup = {p["file_name"]: p for p in predictions}

    if "user_overrides" not in st.session_state:
        st.session_state["user_overrides"] = {}
    if "selected_for_move" not in st.session_state:
        st.session_state["selected_for_move"] = {f.file_name: True for f in scanned_files}

    # Selection helpers
    sel_col1, sel_col2, _ = st.columns([1, 1, 4])
    with sel_col1:
        if st.button("Select All", use_container_width=True):
            st.session_state["selected_for_move"] = {f.file_name: True for f in scanned_files}
            st.rerun()
    with sel_col2:
        if st.button("Deselect All", use_container_width=True):
            st.session_state["selected_for_move"] = {f.file_name: False for f in scanned_files}
            st.rerun()

    # Review table
    category_list = sorted(list(APPROVED_CATEGORIES))
    reviewed_items = []

    for index, file_info in enumerate(scanned_files):
        fname = file_info.file_name
        pred = pred_lookup.get(fname, {"predicted_category": "other", "confidence": 0.5})
        predicted_cat = pred["predicted_category"]
        conf = pred.get("confidence", 0.5)

        is_selected = st.session_state["selected_for_move"].get(fname, True)
        current_override = st.session_state["user_overrides"].get(fname, predicted_cat)

        card_col1, card_col2, card_col3, card_col4, card_col5 = st.columns([0.4, 3.5, 1.8, 2.2, 1.8])

        with card_col1:
            checked = st.checkbox("", value=is_selected, key=f"chk_{index}")
            st.session_state["selected_for_move"][fname] = checked

        with card_col2:
            st.markdown(f"**`{fname}`**")
            st.caption(f"Size: {file_info.size_formatted} | Modified: {file_info.modified_str}")

        with card_col3:
            icon = CATEGORY_ICONS.get(predicted_cat, "📁")
            conf_percent = f"{conf * 100:.0f}%"
            is_ambiguous = conf < confidence_threshold
            conf_badge = (
                f'<span class="badge-ambiguous">⚠️ {conf_percent} (Ambiguous)</span>'
                if is_ambiguous
                else f'<span class="badge-confident">✓ {conf_percent}</span>'
            )
            st.markdown(
                f'{icon} <span class="badge-{predicted_cat}">{predicted_cat.capitalize()}</span> {conf_badge}',
                unsafe_allow_html=True,
            )

        with card_col4:
            cat_index = category_list.index(current_override) if current_override in category_list else 0
            new_cat = st.selectbox(
                f"Override for {fname}",
                options=category_list,
                index=cat_index,
                key=f"cat_{index}",
                label_visibility="collapsed",
            )
            if new_cat != current_override:
                st.session_state["user_overrides"][fname] = new_cat
                st.rerun()

            current_dest = get_destination_folder(new_cat)
            st.caption(f"Target folder: `{current_dest}/`")

        with card_col5:
            # If user overrode category, offer to save ML feedback correction
            if new_cat != predicted_cat:
                if st.button("💾 Save ML Feedback", key=f"fb_{index}", help="Submit confirmed correction to local active-learning loop."):
                    try:
                        fb_result = record_correction(fname, new_cat)
                        st.success(f"Feedback recorded ({fb_result['correction_count']} total)")
                        if fb_result.get("retrained"):
                            st.balloons()
                            st.success("🎉 5 corrections reached! Model was automatically retrained and reloaded.")
                            st.cache_resource.clear()
                            st.rerun()
                    except Exception as fb_exc:
                        st.error(f"Feedback error: {fb_exc}")

        if checked:
            reviewed_items.append({
                "file_name": fname,
                "source_path": file_info.full_path,
                "category": new_cat,
            })

    # Step 3: Safe Dry-Run Planning & Collision Check
    st.markdown("---")
    st.subheader("3️⃣ Safe Dry-Run Preview")
    st.caption("SortIQ computes the exact planned actions without moving any file. Verify the destinations below.")

    if not reviewed_items:
        st.warning("No files are currently selected for organization.")
    else:
        plan = plan_organization(
            items=reviewed_items,
            destination_root=current_path,
            collision_policy=collision_policy,
        )

        ready_count = sum(1 for p in plan if p.status == "ready")
        rename_count = sum(1 for p in plan if p.status == "rename")
        skip_count = sum(1 for p in plan if p.status == "skip")
        error_count = sum(1 for p in plan if p.status == "error")

        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        with p_col1:
            st.metric("Ready to Move", ready_count)
        with p_col2:
            st.metric("Collisions (Auto-Renamed)", rename_count)
        with p_col3:
            st.metric("Skipped", skip_count)
        with p_col4:
            st.metric("Errors", error_count)

        with st.expander("📋 View Detailed Movement Plan (Dry-Run)", expanded=True):
            plan_table_data = []
            for item in plan:
                rel_dest = str(item.resolved_destination_path.relative_to(Path(current_path)))
                plan_table_data.append({
                    "File": item.file_name,
                    "Category": item.category.capitalize(),
                    "Destination Subfolder": item.destination_dir.name,
                    "Destination Filename": item.resolved_destination_path.name,
                    "Status": item.status.upper(),
                    "Notes": item.error_message or ("Safe move" if item.status == "ready" else "Collision resolved with unique name"),
                })
            st.dataframe(plan_table_data, use_container_width=True)

        # Step 4: Explicit Confirmation & Safe Execution
        st.markdown("---")
        st.subheader("4️⃣ Explicit Confirmation & Move Execution")
        st.warning(
            "⚠️ **Safety Check:** Files will be organized into category subfolders inside your chosen directory. "
            "No files will be deleted. Silently overwriting existing files is strictly prevented."
        )

        confirm_checkbox = st.checkbox(
            f"I have reviewed the preview plan and explicitly authorize moving {len(reviewed_items)} file(s).",
            value=False,
            key="explicit_confirm_checkbox",
        )

        move_btn_col, reset_btn_col = st.columns([2, 1])
        with move_btn_col:
            execute_btn = st.button(
                "🚀 Execute File Organization",
                type="primary",
                disabled=not confirm_checkbox or not reviewed_items,
                use_container_width=True,
            )
        with reset_btn_col:
            if st.button("Reset / Cancel", use_container_width=True):
                st.session_state["user_overrides"] = {}
                st.session_state["move_confirmed"] = False
                st.rerun()

        if execute_btn:
            with st.spinner("Moving files safely..."):
                try:
                    move_results = execute_move(plan, confirmed=True)
                    st.success(
                        f"✅ Organization complete! "
                        f"Moved: {move_results['moved_count']} | "
                        f"Skipped: {move_results['skipped_count']} | "
                        f"Errors: {move_results['failed_count']}"
                    )

                    if move_results["moved"]:
                        with st.expander("✨ Successfully Moved Files", expanded=True):
                            st.dataframe(move_results["moved"], use_container_width=True)

                    if move_results["errors"]:
                        with st.expander("⚠️ Failed Items", expanded=True):
                            st.dataframe(move_results["errors"], use_container_width=True)

                    # Refresh scan
                    st.session_state["scanned_files"] = scan_directory(
                        current_path,
                        recursive=False,
                        include_hidden=include_hidden,
                    )
                    st.session_state["user_overrides"] = {}
                except Exception as exec_error:
                    st.error(f"Failed to execute move: {exec_error}")
