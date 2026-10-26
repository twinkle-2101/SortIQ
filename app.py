"""SortIQ: Smart Downloads Folder Organizer — Editorial Web Application.

Design System Reference: design.md (ElevenLabs Editorial / Print Magazine aesthetic)
- Off-white canvas (#f5f5f5 / #fafafa) with warm near-black ink (#0c0a09 / #292524)
- Display typography: EB Garamond weight 300 (editorial serif, negative letter-spacing)
- Body typography: Inter weight 400/500 (+0.16px tracking)
- Atmospheric pastel gradient orbs (mint, peach, lavender, sky, rose)
- Pill geometry (9999px) for CTAs and tags, 16px rounded white cards with 1px hairlines
"""

import base64
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
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
    ROOT,
    load_model,
    predict_file_names,
)
from src.organizer import (
    COLLISION_RENAME,
    COLLISION_SKIP,
    execute_move,
    organization_plan_signature,
    plan_organization,
)
from src.review import category_for_file, get_pending_corrections

# Page configuration
st.set_page_config(
    page_title="SortIQ — Smart Folder Organizer",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)

SANDBOX_DIR = ROOT / "sandbox"


def get_image_base64(path: Path | str) -> str | None:
    """Read a local image file and return a base64 encoded data URI."""
    p = Path(path)
    if p.is_file():
        try:
            raw = p.read_bytes()
            b64 = base64.b64encode(raw).decode("utf-8")
            return f"data:image/png;base64,{b64}"
        except Exception:
            return None
    return None


# Custom CSS adhering strictly to design.md
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,300..700;1,300..700&family=Inter:wght@300;400;500;600&display=swap');

    /* Global canvas & typography */
    html, body, [data-testid="stAppViewContainer"], .main, .stApp {
        background-color: #f5f5f5 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #0c0a09 !important;
        letter-spacing: 0.16px !important;
    }

    [data-testid="stSidebar"] {
        background-color: #fafafa !important;
        border-right: 1px solid #e7e5e4 !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: #e7e5e4 !important;
        margin: 20px 0 !important;
    }

    /* Top navigation bar */
    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 0 20px 0;
        margin-bottom: 24px;
        border-bottom: 1px solid #e7e5e4;
    }
    .top-nav-logo {
        font-family: 'EB Garamond', Georgia, serif;
        font-weight: 400;
        font-size: 28px;
        letter-spacing: -0.6px;
        color: #0c0a09;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .top-nav-tag {
        font-family: 'Inter', sans-serif;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.96px;
        color: #777169;
        background: #f0efed;
        border: 1px solid #e7e5e4;
        padding: 3px 10px;
        border-radius: 9999px;
    }

    /* Atmospheric Hero Band with Signature Gradient Orbs */
    .hero-band {
        position: relative;
        background: #ffffff;
        border: 1px solid #e7e5e4;
        border-radius: 20px;
        padding: 44px 44px 40px 44px;
        margin-bottom: 32px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02);
    }

    /* 5 Atmospheric Gradient Orbs (Mint, Peach, Lavender, Sky, Rose) */
    .orb-lavender {
        position: absolute;
        top: -80px;
        left: -60px;
        width: 380px;
        height: 380px;
        background: radial-gradient(circle, rgba(200, 184, 224, 0.38) 0%, rgba(200, 184, 224, 0) 70%);
        pointer-events: none;
        z-index: 0;
    }
    .orb-peach {
        position: absolute;
        top: -60px;
        right: 6%;
        width: 320px;
        height: 320px;
        background: radial-gradient(circle, rgba(244, 197, 168, 0.38) 0%, rgba(244, 197, 168, 0) 70%);
        pointer-events: none;
        z-index: 0;
    }
    .orb-mint {
        position: absolute;
        bottom: -60px;
        right: -40px;
        width: 340px;
        height: 340px;
        background: radial-gradient(circle, rgba(167, 229, 211, 0.38) 0%, rgba(167, 229, 211, 0) 70%);
        pointer-events: none;
        z-index: 0;
    }
    .orb-sky {
        position: absolute;
        bottom: -80px;
        left: 28%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(168, 200, 232, 0.32) 0%, rgba(168, 200, 232, 0) 70%);
        pointer-events: none;
        z-index: 0;
    }

    .hero-inner {
        position: relative;
        z-index: 1;
        max-width: 860px;
    }

    .hero-h1 {
        font-family: 'EB Garamond', Georgia, serif !important;
        font-weight: 300 !important;
        font-size: 52px !important;
        line-height: 1.08 !important;
        letter-spacing: -1.5px !important;
        color: #0c0a09 !important;
        margin: 12px 0 14px 0 !important;
    }

    .hero-p {
        font-family: 'Inter', sans-serif !important;
        font-size: 15.5px !important;
        line-height: 1.6 !important;
        color: #4e4e4e !important;
        letter-spacing: 0.16px !important;
        margin-bottom: 0 !important;
    }

    /* Headings */
    .editorial-h2 {
        font-family: 'EB Garamond', Georgia, serif !important;
        font-weight: 300 !important;
        font-size: 32px !important;
        line-height: 1.15 !important;
        letter-spacing: -0.6px !important;
        color: #0c0a09 !important;
        margin-top: 36px !important;
        margin-bottom: 6px !important;
    }

    .editorial-sub {
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
        color: #777169 !important;
        margin-bottom: 20px !important;
    }

    /* Stat Tiles */
    .stat-tile {
        background: #ffffff;
        border: 1px solid #e7e5e4;
        border-radius: 16px;
        padding: 20px 22px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
    }
    .stat-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.96px;
        color: #777169;
        margin-bottom: 6px;
    }
    .stat-number {
        font-family: 'EB Garamond', Georgia, serif;
        font-weight: 300;
        font-size: 34px;
        line-height: 1.05;
        color: #0c0a09;
        letter-spacing: -0.5px;
    }
    .stat-desc {
        font-size: 12px;
        color: #777169;
        margin-top: 4px;
    }

    /* Voice-Row Editorial Card Deck */
    .voice-deck {
        background: #ffffff;
        border: 1px solid #e7e5e4;
        border-radius: 16px;
        overflow: hidden;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.02);
    }
    .voice-row {
        display: flex;
        align-items: center;
        padding: 14px 20px;
        border-bottom: 1px solid #f0efed;
        gap: 16px;
        transition: background 0.15s ease;
    }
    .voice-row:last-child {
        border-bottom: none;
    }
    .voice-row:hover {
        background-color: #fafafa;
    }

    .voice-icon {
        width: 34px;
        height: 34px;
        border-radius: 9999px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        flex-shrink: 0;
    }
    .voice-meta {
        flex: 1;
        min-width: 0;
    }
    .voice-name {
        font-size: 14px;
        font-weight: 500;
        color: #0c0a09;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .voice-sub {
        font-size: 12px;
        color: #777169;
        margin-top: 2px;
    }

    /* Badges & Pills */
    .pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 11.5px;
        font-weight: 500;
    }

    .pill-academic { background: rgba(168, 200, 232, 0.35); color: #1e3a8a; border: 1px solid rgba(168, 200, 232, 0.6); }
    .pill-resume { background: rgba(200, 184, 224, 0.35); color: #4c1d95; border: 1px solid rgba(200, 184, 224, 0.6); }
    .pill-receipt { background: rgba(167, 229, 211, 0.4); color: #064e3b; border: 1px solid rgba(167, 229, 211, 0.6); }
    .pill-presentation { background: rgba(244, 197, 168, 0.4); color: #78350f; border: 1px solid rgba(244, 197, 168, 0.6); }
    .pill-image { background: rgba(232, 184, 196, 0.35); color: #831843; border: 1px solid rgba(232, 184, 196, 0.6); }
    .pill-other { background: #f0efed; color: #292524; border: 1px solid #e7e5e4; }

    .status-confident {
        display: inline-block;
        font-size: 11px;
        font-weight: 500;
        padding: 2px 8px;
        border-radius: 9999px;
        background: #f0fdf4;
        color: #15803d;
        border: 1px solid #bbf7d0;
    }
    .status-ambiguous {
        display: inline-block;
        font-size: 11px;
        font-weight: 500;
        padding: 2px 8px;
        border-radius: 9999px;
        background: #fff1f2;
        color: #be123c;
        border: 1px solid #fecdd3;
    }

    /* Buttons — Near-Black Ink Pill {component.button-primary} */
    .stButton > button {
        border-radius: 9999px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        padding: 8px 22px !important;
        height: 42px !important;
        border: 1px solid #d6d3d1 !important;
        background-color: #ffffff !important;
        color: #0c0a09 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        background-color: #f5f5f5 !important;
        border-color: #0c0a09 !important;
        color: #0c0a09 !important;
    }

    /* Primary CTAs — Warm Ink Pill */
    .stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
        background-color: #292524 !important;
        color: #ffffff !important;
        border: 1px solid #292524 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
    }
    .stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background-color: #0c0a09 !important;
        border-color: #0c0a09 !important;
        color: #ffffff !important;
    }

    /* Inputs */
    .stTextInput input {
        border-radius: 8px !important;
        border: 1px solid #d6d3d1 !important;
        background-color: #ffffff !important;
        color: #0c0a09 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
        height: 44px !important;
        padding: 10px 14px !important;
    }
    .stTextInput input:focus {
        border-color: #0c0a09 !important;
        box-shadow: 0 0 0 1px #0c0a09 !important;
    }

    /* Checkbox label styling */
    [data-testid="stCheckbox"] label span {
        color: #0c0a09 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }

    /* Selectbox dropdowns */
    .stSelectbox div[data-baseweb="select"] {
        border-radius: 8px !important;
        background-color: #ffffff !important;
        border-color: #d6d3d1 !important;
        color: #0c0a09 !important;
    }

    /* Refined, thin slider track and neat circular thumb */
    [data-testid="stSlider"] div[role="slider"] {
        background-color: #292524 !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.25) !important;
        width: 16px !important;
        height: 16px !important;
    }

    /* Custom Editorial Dry-Run Table */
    .ledger-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13.5px;
        background: #ffffff;
        border: 1px solid #e7e5e4;
        border-radius: 14px;
        overflow: hidden;
    }
    .ledger-table th {
        background: #fafafa;
        color: #777169;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.96px;
        padding: 12px 18px;
        text-align: left;
        border-bottom: 1px solid #e7e5e4;
    }
    .ledger-table td {
        padding: 12px 18px;
        border-bottom: 1px solid #f0efed;
        color: #0c0a09;
    }
    .ledger-table tr:last-child td {
        border-bottom: none;
    }
    .ledger-table tr:hover td {
        background: #fafafa;
    }

    /* Dividers */
    hr {
        border-color: #e7e5e4 !important;
        margin: 32px 0 !important;
    }
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


def get_sandbox_path() -> Path:
    """Return the absolute path to the root sandbox directory."""
    return SANDBOX_DIR


def ensure_sandbox_files() -> int:
    """Ensure the root sandbox directory contains realistic dummy files for testing."""
    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    dummy_files = {
        # Academic
        "computer_networks_unit4_lecture_notes.pdf": "Course: Computer Networks\nUnit 4: Transport Layer Protocols (TCP/UDP)\nReliable data transfer, sliding window, congestion control algorithms.",
        "machine_learning_assignment_final.pdf": "CS401 Machine Learning Assignment 2\nSupport Vector Machines and Neural Networks implementation report.",
        "discrete_mathematics_syllabus_2025.docx": "Discrete Mathematics and Graph Theory Syllabus for Spring 2025 Semester.",
        "operating_systems_exam_prep.pdf": "Operating Systems Study Guide: Virtual memory, paging, process synchronization semaphores.",
        "research_paper_transformer_attention.pdf": "Attention Is All You Need - Empirical evaluation of Transformer models in NLP.",

        # Resume
        "alex_morgan_software_engineer_resume.pdf": "Alex Morgan\nSenior Software Engineer\nExperience: Python, PyTorch, Distributed Systems, Cloud Architecture.",
        "sarah_jenkins_cv_october_2025.pdf": "Sarah Jenkins - Curriculum Vitae\nProduct Designer & UX Researcher\nPortfolio and Case Studies.",
        "david_chen_frontend_developer_profile.docx": "David Chen - Lead React/TypeScript Frontend Developer Profile & Project Showcase.",
        "emily_clark_cover_letter_google.pdf": "Dear Hiring Manager,\nI am writing to express my enthusiastic interest in the ML Engineer role at Google.",

        # Receipt
        "amazon_order_tax_invoice_10928.pdf": "Amazon.com Tax Invoice\nOrder # 112-9847291-1029482\nItem: Wireless Ergonomic Keyboard\nTotal: $89.99\nPayment: Confirmed.",
        "starbucks_coffee_store_receipt.pdf": "Starbucks Store #4928\nGrande Caramel Macchiato - $5.75\nButter Croissant - $3.45\nTotal Paid: $9.20\nTax Invoice.",
        "aws_cloud_monthly_statement_oct2025.pdf": "Amazon Web Services Statement\nAccount: 9482-1029-4820\nUsage: EC2, S3, RDS compute charges\nTotal Amount Due: $142.50",
        "uber_trip_receipt_san_francisco.pdf": "Uber Trip Receipt\nTrip from Market St to SFO Airport\nFare: $44.20\nTip: $8.00\nTotal: $52.20\nPayment Method: Apple Pay",

        # Presentation
        "q3_product_growth_deck.pptx": "Q3 Product Growth Review Slides\nMonthly Active Users +34%\nRetention Cohort Analysis\nKey Roadmap Initiatives.",
        "ai_startup_pitch_presentation.pptx": "SortIQ Seed Round Pitch Deck\nProblem, Solution, Market Opportunity, Financial Projections, Founder Team.",
        "cybersecurity_seminar_keynote.pptx": "Keynote Presentation: Zero Trust Architecture in Modern Cloud Infrastructure.",
        "annual_shareholders_meeting_slides.pdf": "Annual General Meeting Slide Deck\nFinancial Performance, Governance, FY2026 Guidance.",

        # Image
        "golden_gate_bridge_sunset.jpg": "\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb" + "JPEG_IMAGE_DATA_SUNSET" * 50,
        "screenshot_2025_10_25_ui_bug.png": "\x89PNG\r\n\x1a\n" + "PNG_SCREENSHOT_DATA" * 50,
        "team_offsite_group_photo.jpeg": "\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb" + "JPEG_GROUP_PHOTO" * 50,
        "system_architecture_diagram.png": "\x89PNG\r\n\x1a\n" + "PNG_ARCHITECTURE_DIAGRAM" * 50,

        # Other
        "docker_desktop_installer_v4.dmg": "DMG_BINARY_INSTALLER_PAYLOAD_MOCK_DATA" * 50,
        "python_environment_config.yaml": "name: sortiq-env\nchannels:\n  - conda-forge\ndependencies:\n  - python=3.11\n  - scikit-learn\n  - streamlit\n",
        "postgresql_backup_dump.sql": "CREATE TABLE users (id SERIAL PRIMARY KEY, name VARCHAR(255));\nINSERT INTO users VALUES (1, 'Alice');",
        "node_modules_cache_archive.zip": "PK\x03\x04" + "ZIP_ARCHIVE_MOCK_PAYLOAD" * 50,
    }

    created_count = 0
    for filename, content in dummy_files.items():
        file_path = SANDBOX_DIR / filename
        if not file_path.exists():
            if isinstance(content, str):
                file_path.write_text(content, encoding="utf-8")
            else:
                file_path.write_bytes(content)
            created_count += 1
    return created_count


def create_demo_folder() -> Path:
    """Backwards-compatible helper for test suite."""
    ensure_sandbox_files()
    demo_dir = ROOT / "data" / "demo_downloads"
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
        fp = demo_dir / filename
        if not fp.exists():
            fp.write_text(content, encoding="utf-8")
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


# Ensure sandbox is ready on first launch
ensure_sandbox_files()

# Load model & metrics
model, model_error = get_cached_model()
metrics = load_model_metrics()

# Top Navigation Bar
st.markdown(
    """
    <div class="top-nav">
        <div class="top-nav-logo">
            <span>SortIQ</span>
            <span class="top-nav-tag">Editorial ML File Organizer</span>
        </div>
        <div style="font-size: 13px; color: #777169;">
            Active Model: <strong style="color: #0c0a09;">TF-IDF + TruncatedSVD (100) + Logistic Regression</strong>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Configuration
with st.sidebar:
    st.markdown('<div class="top-nav-tag">System Parameters</div>', unsafe_allow_html=True)
    st.markdown(
        "<h3 style='font-family: \"EB Garamond\", serif; font-weight: 300; font-size: 24px; margin-top: 6px; color: #0c0a09;'>Model Architecture</h3>",
        unsafe_allow_html=True,
    )

    if model is not None and metrics:
        sel_name = metrics.get("selected_model", "Selected Pipeline")
        perf = metrics.get("models", {}).get(sel_name, {})
        acc = perf.get("accuracy", 0.0)
        f1 = perf.get("macro_f1", 0.0)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Accuracy", f"{acc * 100:.1f}%")
        with c2:
            st.metric("Macro-F1", f"{f1 * 100:.1f}%")
        st.caption(f"Pipeline: `{sel_name}`")
    else:
        st.warning("Model Not Found")
        st.caption(f"{model_error}\nRun `python -m src.run_experiments` to train.")

    st.markdown("---")
    st.markdown(
        "<h4 style='font-family: \"EB Garamond\", serif; font-weight: 300; font-size: 20px; color: #0c0a09;'>Safety & Collision</h4>",
        unsafe_allow_html=True,
    )
    confidence_threshold = st.slider(
        "Ambiguity Review Threshold",
        min_value=0.40,
        max_value=0.95,
        value=0.65,
        step=0.05,
        help="Predictions below this confidence level are flagged for human review.",
    )

    collision_policy_label = st.selectbox(
        "Name Collision Policy",
        options=["Rename Colliding Files (file (1).ext)", "Skip Colliding Files"],
        index=0,
        help="Safely preserves existing files in target folders without overwriting.",
    )
    collision_policy = (
        COLLISION_RENAME if "Rename" in collision_policy_label else COLLISION_SKIP
    )

    include_hidden = st.checkbox(
        "Include hidden files (.*)",
        value=False,
        help="Excludes system dotfiles by default.",
    )

    st.markdown("---")
    st.markdown(
        "<h4 style='font-family: \"EB Garamond\", serif; font-weight: 300; font-size: 20px; color: #0c0a09;'>Root Sandbox</h4>",
        unsafe_allow_html=True,
    )
    st.caption("A disposable folder `sandbox/` with 25 test files for safe evaluation.")
    if st.button("Reset Sandbox (25 Files)", use_container_width=True):
        count = ensure_sandbox_files()
        st.session_state["scan_folder_input"] = str(SANDBOX_DIR.resolve())
        st.session_state["scanned_files"] = scan_directory(SANDBOX_DIR.resolve(), recursive=False)
        st.session_state["predictions"] = None
        st.success(f"Sandbox refreshed ({count} files).")
        st.rerun()

    # Model Metrics & Confusion Matrix viewer (using embedded Base64 for guaranteed rendering)
    if metrics or DEFAULT_CONFUSION_MATRIX_PATH.is_file():
        with st.expander("📊 SVD Benchmark & Confusion Matrix", expanded=False):
            if metrics:
                st.markdown("<div style='font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.96px; color: #777169; margin-bottom: 8px;'>Candidate Holdout Comparison</div>", unsafe_allow_html=True)
                comp_data = []
                for m_name, m_data in metrics.get("models", {}).items():
                    comp_data.append({
                        "Model": m_name.replace("tfidf_", "").replace("_", " ").title(),
                        "Accuracy": f"{m_data.get('accuracy', 0)*100:.1f}%",
                        "Macro-F1": f"{m_data.get('macro_f1', 0)*100:.1f}%",
                        "SVD Components": (
                            str(m_data["svd_components"])
                            if m_data.get("svd_components") is not None
                            else "None"
                        ),
                        "Runtime": f"{m_data.get('runtime_seconds', 0):.4f}s",
                    })
                st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

            if DEFAULT_CONFUSION_MATRIX_PATH.is_file():
                img_data_uri = get_image_base64(DEFAULT_CONFUSION_MATRIX_PATH)
                if img_data_uri:
                    st.markdown(
                        f"""
                        <div style="text-align: center; margin-top: 14px;">
                            <img src="{img_data_uri}" style="max-width: 100%; border-radius: 10px; border: 1px solid #e7e5e4;" />
                            <div style="font-size: 11px; color: #777169; margin-top: 6px;">
                                Confusion Matrix — Selected Model (TF-IDF + SVD + Logistic Regression)
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# Atmospheric Hero Band
st.markdown(
    """
    <div class="hero-band">
        <div class="orb-lavender"></div>
        <div class="orb-peach"></div>
        <div class="orb-mint"></div>
        <div class="orb-sky"></div>
        <div class="hero-inner">
            <span class="top-nav-tag">Machine Learning Organization</span>
            <h1 class="hero-h1">Quietly organized.<br>Precisely categorized.</h1>
            <p class="hero-p">
                SortIQ tokenizes filenames, applies TF-IDF with TruncatedSVD dimensionality reduction,
                and routes downloads into organized categories with guaranteed dry-run preview and explicit confirmation.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Step 1: Directory Selection & Scan
st.markdown('<div class="editorial-h2">1. Directory Selection & Inspection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="editorial-sub">Choose a target directory to scan. Scanning is strictly read-only and never modifies or deletes files.</div>',
    unsafe_allow_html=True,
)

default_scan_dir = st.session_state.get(
    "scan_folder_input",
    str(SANDBOX_DIR.resolve()) if SANDBOX_DIR.exists() else str(Path.home() / "Downloads"),
)

input_col, btn_sandbox_col, btn_scan_col = st.columns([3.5, 1.4, 1.2])

with input_col:
    folder_input = st.text_input(
        "Directory Path:",
        value=default_scan_dir,
        placeholder="Enter path (e.g., /path/to/sandbox or ~/Downloads)",
        label_visibility="collapsed",
    )

with btn_sandbox_col:
    if st.button("📂 Use Sandbox", use_container_width=True, help="Load root sandbox with 25 test files"):
        ensure_sandbox_files()
        st.session_state["scan_folder_input"] = str(SANDBOX_DIR.resolve())
        st.rerun()

with btn_scan_col:
    scan_button = st.button("Scan Folder", use_container_width=True, type="primary")

# Execute scan
if scan_button or "scanned_files" not in st.session_state:
    is_valid, err_msg = validate_directory_path(folder_input)
    if not is_valid:
        st.error(f"Directory Error: {err_msg}")
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
            st.session_state["predictions"] = None
        except Exception as exc:
            st.error(f"Scan operation failed: {exc}")
            st.session_state["scanned_files"] = []

scanned_files = st.session_state.get("scanned_files", [])
current_path = st.session_state.get("current_scan_path", folder_input)

if not scanned_files:
    st.info("No files found in this directory. Click 'Use Sandbox' above to inspect the 25 sample files.")
else:
    total_size = sum(f.size_bytes for f in scanned_files)
    unique_exts = len({f.extension for f in scanned_files if f.extension})

    # Metric Strip
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(
            f"""
            <div class="stat-tile">
                <div class="stat-label">Files Discovered</div>
                <div class="stat-number">{len(scanned_files)}</div>
                <div class="stat-desc">Direct files in root folder</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            f"""
            <div class="stat-tile">
                <div class="stat-label">Total Volume</div>
                <div class="stat-number">{format_file_size(total_size)}</div>
                <div class="stat-desc">Cumulative byte size</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            f"""
            <div class="stat-tile">
                <div class="stat-label">File Extensions</div>
                <div class="stat-number">{unique_exts}</div>
                <div class="stat-desc">Unique formats identified</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s4:
        st.markdown(
            """
            <div class="stat-tile">
                <div class="stat-label">Integrity Protocol</div>
                <div class="stat-number" style="font-size: 26px; padding-top: 4px;">Read-Only</div>
                <div class="stat-desc">Zero filesystem mutations</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Step 2: Prediction & Interactive Review Table
    st.markdown('<div class="editorial-h2">2. Category Predictions & Interactive Review</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="editorial-sub">SortIQ classifies each file into 6 categories. '
        f'Predictions below {int(confidence_threshold*100)}% are highlighted as ambiguous. '
        f'Review only low-confidence predictions, then confirm which files to move.</div>',
        unsafe_allow_html=True,
    )

    # Generate predictions if not cached in session
    file_names = [f.file_name for f in scanned_files]
    if st.session_state.get("predictions") is None:
        if model is not None:
            predictions = predict_file_names(model, file_names)
        else:
            predictions = rule_based_fallback_predict(file_names)
        st.session_state["predictions"] = {p["file_name"]: p for p in predictions}

    pred_map = st.session_state["predictions"]

    # Initialize review state
    if "user_overrides" not in st.session_state:
        st.session_state["user_overrides"] = {}
    if "submitted_feedback" not in st.session_state:
        st.session_state["submitted_feedback"] = {}
    if "selected_files" not in st.session_state:
        st.session_state["selected_files"] = {f.file_name: True for f in scanned_files}

    # Category filter pills
    category_list = sorted(list(APPROVED_CATEGORIES))
    ambiguous_count = sum(
        1 for f in scanned_files
        if pred_map.get(f.file_name, {}).get("confidence", 0.5) < confidence_threshold
    )
    pending_corrections = get_pending_corrections(
        [file_info.file_name for file_info in scanned_files],
        pred_map,
        st.session_state["user_overrides"],
        st.session_state["submitted_feedback"],
        confidence_threshold,
    )

    # Filter selector
    filter_options = ["All Files", f"⚠️ Ambiguous ({ambiguous_count})"] + [c.capitalize() for c in category_list]
    active_filter = st.selectbox(
        "Filter View:",
        options=filter_options,
        index=0,
        label_visibility="collapsed",
    )

    # Filtered subset of scanned files
    visible_files = []
    for f in scanned_files:
        pred_item = pred_map.get(f.file_name, {"predicted_category": "other", "confidence": 0.5})
        pred_cat = pred_item["predicted_category"]
        conf = float(pred_item.get("confidence", 0.5))

        if active_filter == "All Files":
            visible_files.append(f)
        elif active_filter.startswith("⚠️ Ambiguous"):
            if conf < confidence_threshold:
                visible_files.append(f)
        else:
            if pred_cat.lower() == active_filter.lower():
                visible_files.append(f)

    # Bulk actions
    b_col1, b_col2, b_col3 = st.columns([1.2, 1.2, 5])
    with b_col1:
        if st.button("Select All", use_container_width=True):
            for f in visible_files:
                st.session_state["selected_files"][f.file_name] = True
            st.rerun()
    with b_col2:
        if st.button("Deselect All", use_container_width=True):
            for f in visible_files:
                st.session_state["selected_files"][f.file_name] = False
            st.rerun()

    # Editorial Voice-Row Card Deck
    reviewed_items = []
    st.markdown('<div class="voice-deck">', unsafe_allow_html=True)

    for index, file_info in enumerate(visible_files):
        fname = file_info.file_name
        pred = pred_map.get(fname, {"predicted_category": "other", "confidence": 0.5})
        predicted_cat = pred["predicted_category"]
        conf = float(pred.get("confidence", 0.5))

        is_selected = st.session_state["selected_files"].get(fname, True)
        is_ambiguous = conf < confidence_threshold
        if not is_ambiguous:
            st.session_state["user_overrides"].pop(fname, None)
        current_cat = category_for_file(
            fname,
            predicted_cat,
            conf,
            confidence_threshold,
            st.session_state["user_overrides"],
        )

        row_col1, row_col2, row_col3, row_col4 = st.columns([0.4, 4.4, 2.4, 2.8])

        with row_col1:
            checked = st.checkbox(
                f"Select {fname}",
                value=is_selected,
                key=f"sel_{fname}",
                label_visibility="collapsed",
            )
            st.session_state["selected_files"][fname] = checked

        with row_col2:
            icon = CATEGORY_ICONS.get(current_cat, "📁")
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div class="voice-icon pill-{current_cat}">{icon}</div>
                    <div class="voice-meta">
                        <div class="voice-name" title="{fname}">{fname}</div>
                        <div class="voice-sub">{file_info.size_formatted} · {file_info.extension.upper() or 'FILE'} · Modified {file_info.modified_str}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with row_col3:
            is_ambiguous = conf < confidence_threshold
            conf_badge = (
                f"<span class='status-ambiguous'>⚠️ {conf*100:.0f}% Review</span>"
                if is_ambiguous
                else f"<span class='status-confident'>✓ {conf*100:.0f}%</span>"
            )
            cat_badge = f"<span class='pill-badge pill-{predicted_cat}'>{predicted_cat.capitalize()}</span>"
            st.markdown(f"{cat_badge} {conf_badge}", unsafe_allow_html=True)

        with row_col4:
            if is_ambiguous:
                cat_idx = category_list.index(current_cat) if current_cat in category_list else 0
                new_cat = st.selectbox(
                    f"Review category for {fname}",
                    options=category_list,
                    index=cat_idx,
                    key=f"cat_{fname}",
                    label_visibility="collapsed",
                )
                if new_cat != current_cat:
                    st.session_state["user_overrides"][fname] = new_cat
                    st.rerun()
            else:
                new_cat = predicted_cat
                st.caption("No review needed")

            dest_folder = get_destination_folder(new_cat)
            st.caption(f"Destination: `{dest_folder}/`")

        st.markdown("<div style='border-bottom: 1px solid #f0efed; margin: 2px 0 8px 0;'></div>", unsafe_allow_html=True)

        if checked:
            reviewed_items.append({
                "file_name": fname,
                "source_path": file_info.full_path,
                "category": new_cat,
            })

    st.markdown('</div>', unsafe_allow_html=True)

    # Active learning submission banner
    if pending_corrections:
        fb_col1, fb_col2 = st.columns([4, 1.5])
        with fb_col1:
            st.markdown(
                f"""
                <div style="background: #ffffff; border: 1px solid #e7e5e4; border-left: 3px solid #292524; border-radius: 12px; padding: 14px 18px;">
                    <div style="font-weight: 500; font-size: 13.5px; color: #0c0a09;">
                        Active Learning Feedback ({len(pending_corrections)} correction{'s' if len(pending_corrections) > 1 else ''} pending)
                    </div>
                    <div style="font-size: 12.5px; color: #777169; margin-top: 2px;">
                        Confirmed corrections are saved locally. The pipeline retrains automatically after every 5 confirmed corrections.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with fb_col2:
            if st.button("💾 Save Ambiguous Corrections", use_container_width=True, type="secondary"):
                retrained_any = False
                recorded_count = 0
                for fname, new_cat in pending_corrections.items():
                    try:
                        res = record_correction(fname, new_cat)
                    except Exception as feedback_error:
                        st.error(f"Could not save correction for {fname}: {feedback_error}")
                        continue
                    st.session_state["submitted_feedback"][fname] = new_cat
                    recorded_count += 1
                    if res.get("retrained"):
                        retrained_any = True
                if recorded_count:
                    st.success(
                        f"Saved {recorded_count} confirmed ambiguous correction(s)."
                    )
                if retrained_any:
                    st.balloons()
                    st.success("🎉 Threshold reached: Model automatically retrained and reloaded!")
                    st.cache_resource.clear()
                    st.rerun()

    # Step 3: Safe Dry-Run Simulation
    st.markdown('<div class="editorial-h2">3. Safe Dry-Run Simulation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="editorial-sub">The dry-run engine checks destination collisions and prepares the organization ledger. No files move yet.</div>',
        unsafe_allow_html=True,
    )

    if not reviewed_items:
        st.warning("No files are currently selected for organization. Check the selection box next to files above.")
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

        # Dry-run stats
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.markdown(
                f"""
                <div class="stat-tile">
                    <div class="stat-label">Ready to Move</div>
                    <div class="stat-number">{ready_count}</div>
                    <div class="stat-desc">Direct relocation</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with d2:
            st.markdown(
                f"""
                <div class="stat-tile">
                    <div class="stat-label">Auto-Renamed</div>
                    <div class="stat-number">{rename_count}</div>
                    <div class="stat-desc">Collision safeguard</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with d3:
            st.markdown(
                f"""
                <div class="stat-tile">
                    <div class="stat-label">Skipped</div>
                    <div class="stat-number">{skip_count}</div>
                    <div class="stat-desc">Per collision policy</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with d4:
            st.markdown(
                f"""
                <div class="stat-tile">
                    <div class="stat-label">Validation Errors</div>
                    <div class="stat-number">{error_count}</div>
                    <div class="stat-desc">Issues detected</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Dry-Run Movement Ledger
        ledger_rows = []
        for item in plan:
            try:
                rel_dest = str(item.resolved_destination_path.relative_to(Path(current_path)))
            except ValueError:
                rel_dest = str(item.resolved_destination_path)

            safeguard_note = (
                "Auto-renamed (collision resolved)"
                if item.status == "rename"
                else (item.error_message or "Verified safe")
            )

            ledger_rows.append({
                "Source Filename": item.file_name,
                "Category": item.category.capitalize(),
                "Target Subfolder": f"{item.destination_dir.name}/",
                "Destination Path": rel_dest,
                "Action Status": item.status.upper(),
                "Collision Safeguard": safeguard_note,
            })

        st.dataframe(
            pd.DataFrame(ledger_rows),
            use_container_width=True,
            hide_index=True,
        )

        # Step 4: Explicit Confirmation & Execution
        st.markdown('<div class="editorial-h2">4. Explicit Authorization & Execution</div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div style="background: #ffffff; border: 1px solid #e7e5e4; border-left: 3px solid #292524; border-radius: 12px; padding: 18px 22px; margin-bottom: 20px;">
                <div style="font-weight: 500; font-size: 14px; color: #0c0a09; margin-bottom: 4px;">
                    Safety Invariant & Confirmation Protocol
                </div>
                <div style="font-size: 13px; color: #4e4e4e;">
                    Selected files will be relocated into dedicated category folders within the specified root directory.
                    Silent overwriting is strictly prevented. Files will never be deleted.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        current_plan_signature = organization_plan_signature(
            plan,
            current_path,
            collision_policy,
        )
        if st.session_state.get("preview_plan_signature") != current_plan_signature:
            st.session_state["explicit_confirm_checkbox"] = False
            st.session_state["confirmed_plan_signature"] = None
            st.session_state["preview_plan_signature"] = current_plan_signature

        confirm_checkbox = st.checkbox(
            f"I have verified the dry-run ledger and explicitly authorize moving {len(reviewed_items)} file(s).",
            value=False,
            key="explicit_confirm_checkbox",
        )
        if confirm_checkbox:
            st.session_state["confirmed_plan_signature"] = current_plan_signature

        exec_col, reset_col = st.columns([2.5, 1])
        with exec_col:
            execute_btn = st.button(
                "Execute File Organization",
                type="primary",
                disabled=(
                    not confirm_checkbox
                    or st.session_state.get("confirmed_plan_signature")
                    != current_plan_signature
                    or not reviewed_items
                ),
                use_container_width=True,
            )
        with reset_col:
            if st.button("Reset / Clear Review", use_container_width=True):
                st.session_state["predictions"] = None
                st.session_state["user_overrides"] = {}
                st.rerun()

        if execute_btn:
            with st.spinner("Executing safe file relocation..."):
                try:
                    move_results = execute_move(plan, confirmed=True)
                    st.success(
                        f"Organization complete: {move_results['moved_count']} moved, "
                        f"{move_results['skipped_count']} skipped, {move_results['failed_count']} errors."
                    )

                    if move_results["moved"]:
                        with st.expander("Relocation Report", expanded=True):
                            st.dataframe(pd.DataFrame(move_results["moved"]), use_container_width=True, hide_index=True)

                    if move_results["errors"]:
                        with st.expander("Error Report", expanded=True):
                            st.dataframe(pd.DataFrame(move_results["errors"]), use_container_width=True, hide_index=True)

                    # Refresh directory state
                    st.session_state["scanned_files"] = scan_directory(
                        current_path,
                        recursive=False,
                        include_hidden=include_hidden,
                    )
                    st.session_state["predictions"] = None
                    st.session_state["user_overrides"] = {}
                except Exception as exec_error:
                    st.error(f"Execution failed: {exec_error}")
