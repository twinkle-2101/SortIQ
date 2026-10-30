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
import tempfile
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
    initial_sidebar_state="collapsed",
)

SANDBOX_DIR = ROOT / "sandbox"
SANDBOX_SAMPLE_FILENAMES: set[str] = set()


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

st.markdown(
    """
    <style>
    [data-testid="stMainBlockContainer"] {
        max-width: 1480px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    [data-testid="stAppViewContainer"], .stApp {
        background: #f7f8fa !important;
    }

    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e6e9ef !important;
    }

    h1, h2, h3, h4 {
        color: #17212f !important;
        letter-spacing: -0.035em !important;
    }

    h1 {
        font-size: clamp(2rem, 3vw, 2.8rem) !important;
        line-height: 1.08 !important;
        margin-bottom: 0.35rem !important;
    }

    [data-testid="stCaptionContainer"], .stMarkdown p {
        color: #687386;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border: 1px solid #e3e8ef !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 16px rgba(27, 39, 55, 0.035);
        padding: 1rem 1.15rem;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e3e8ef;
        border-radius: 14px;
        padding: 0.9rem 1rem;
    }

    [data-testid="stMetricLabel"] p {
        color: #687386 !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {
        color: #17212f !important;
        font-size: 1.65rem !important;
        font-weight: 650 !important;
    }

    .stButton > button {
        min-height: 2.65rem !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }

    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: #176b58 !important;
        border-color: #176b58 !important;
    }

    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background: #125746 !important;
        border-color: #125746 !important;
    }

    .stTextInput input,
    .stSelectbox div[data-baseweb="select"] {
        border-radius: 10px !important;
    }

    [data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #e3e8ef;
        border-radius: 12px;
    }

    .workflow-step {
        color: #176b58;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .workflow-title {
        color: #17212f;
        font-size: 1.35rem;
        font-weight: 650;
        letter-spacing: -0.025em;
        margin: 0 0 0.2rem;
    }

    .workflow-help {
        color: #687386;
        font-size: 0.92rem;
        margin: 0 0 0.8rem;
    }

    .voice-name {
        color: #17212f;
        font-size: 0.92rem;
        font-weight: 600;
        overflow-wrap: anywhere;
    }

    .voice-sub {
        color: #758094;
    }

    hr {
        border-color: #e3e8ef !important;
        margin: 1.25rem 0 !important;
    }

    .app-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        border: 1px solid #d5e8e1;
        border-radius: 999px;
        background: #edf7f3;
        color: #176b58;
        font-size: 0.8rem;
        font-weight: 650;
        padding: 0.45rem 0.8rem;
    }

    .review-summary {
        border-left: 4px solid #176b58;
        border-radius: 10px;
        background: #eef6f3;
        color: #315d52;
        padding: 0.75rem 1rem;
    }

    @media (max-width: 760px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1rem 1rem 3rem;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            padding: 0.8rem;
        }
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


def create_fresh_demo_folder(base_dir: Path | None = None) -> Path:
    """Create an isolated demo run so prior moves cannot affect its preview."""
    ensure_sandbox_files()
    demo_parent = base_dir or (Path(tempfile.gettempdir()) / "sortiq-demo-runs")
    demo_parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=demo_parent))
    for source_file in SANDBOX_DIR.iterdir():
        if source_file.is_file() and source_file.name in SANDBOX_SAMPLE_FILENAMES:
            shutil.copy2(source_file, run_dir / source_file.name)

    demo_collisions = {
        "Academic": (
            "computer_networks_unit4_lecture_notes.pdf",
            "Existing demo copy: archived Computer Networks lecture notes.",
        ),
        "Resumes": (
            "alex_morgan_software_engineer_resume.pdf",
            "Existing demo copy: previous resume version for collision testing.",
        ),
        "Receipts": (
            "amazon_order_tax_invoice_10928.pdf",
            "Existing demo copy: archived order invoice for collision testing.",
        ),
        "Presentations": (
            "quarterly_sales_enablement_training_deck.pptx",
            "Existing demo copy: prior sales enablement deck for collision testing.",
        ),
        "Images": (
            "family_hiking_trail_photo_2026.jpg",
            "Existing demo copy: alternate hiking photo for collision testing.",
        ),
        "Other": (
            "nginx_reverse_proxy_configuration.conf",
            "Existing demo copy: prior proxy configuration for collision testing.",
        ),
    }
    for folder_name, (file_name, content) in demo_collisions.items():
        destination_dir = run_dir / folder_name
        destination_dir.mkdir()
        (destination_dir / file_name).write_text(content, encoding="utf-8")

    return run_dir


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
        "database_systems_lab_report_normalization.pdf": "Database Systems Lab Report\nCourse: CS 312\nAssignment: Normalize the university enrollment schema to 3NF and document the functional dependencies.",
        "statistics_midterm_review_chapter_6.docx": "Statistics II Midterm Review\nChapter 6: Confidence intervals, hypothesis testing, and interpreting p-values.",

        # Resume
        "alex_morgan_software_engineer_resume.pdf": "Alex Morgan\nSenior Software Engineer\nExperience: Python, PyTorch, Distributed Systems, Cloud Architecture.",
        "sarah_jenkins_cv_october_2025.pdf": "Sarah Jenkins - Curriculum Vitae\nProduct Designer & UX Researcher\nPortfolio and Case Studies.",
        "david_chen_frontend_developer_profile.docx": "David Chen - Lead React/TypeScript Frontend Developer Profile & Project Showcase.",
        "emily_clark_cover_letter_google.pdf": "Dear Hiring Manager,\nI am writing to express my enthusiastic interest in the ML Engineer role at Google.",
        "maya_patel_product_manager_resume_2026.pdf": "Maya Patel\nProduct Manager\nExperience: roadmap planning, customer research, analytics, and cross-functional delivery.",
        "robert_kim_backend_engineer_cv.docx": "Robert Kim\nBackend Engineer\nSkills: Java, Python, PostgreSQL, REST APIs, cloud deployment.",
        "ux_researcher_portfolio_case_studies.pdf": "UX Research Portfolio\nSelected case studies covering usability testing, interview synthesis, and accessibility reviews.",

        # Receipt
        "amazon_order_tax_invoice_10928.pdf": "Amazon.com Tax Invoice\nOrder # 112-9847291-1029482\nItem: Wireless Ergonomic Keyboard\nTotal: $89.99\nPayment: Confirmed.",
        "starbucks_coffee_store_receipt.pdf": "Starbucks Store #4928\nGrande Caramel Macchiato - $5.75\nButter Croissant - $3.45\nTotal Paid: $9.20\nTax Invoice.",
        "aws_cloud_monthly_statement_oct2025.pdf": "Amazon Web Services Statement\nAccount: 9482-1029-4820\nUsage: EC2, S3, RDS compute charges\nTotal Amount Due: $142.50",
        "uber_trip_receipt_san_francisco.pdf": "Uber Trip Receipt\nTrip from Market St to SFO Airport\nFare: $44.20\nTip: $8.00\nTotal: $52.20\nPayment Method: Apple Pay",
        "city_power_utility_bill_september_2026.pdf": "City Power Utilities\nBilling period: September 1-30, 2026\nElectricity: 284 kWh\nAmount due: $74.18\nDue date: October 20, 2026.",
        "hotel_booking_confirmation_invoice.pdf": "Harborview Hotel\nStay: October 18-21, 2026\nRoom and taxes: $684.00\nReservation confirmed.",
        "whole_foods_market_receipt_2026-10-12.pdf": "Whole Foods Market\nPurchase date: October 12, 2026\nGroceries and household items\nSubtotal: $63.42\nTotal paid: $68.19.",

        # Presentation
        "q3_product_growth_deck.pptx": "Q3 Product Growth Review Slides\nMonthly Active Users +34%\nRetention Cohort Analysis\nKey Roadmap Initiatives.",
        "ai_startup_pitch_presentation.pptx": "SortIQ Seed Round Pitch Deck\nProblem, Solution, Market Opportunity, Financial Projections, Founder Team.",
        "cybersecurity_seminar_keynote.pptx": "Keynote Presentation: Zero Trust Architecture in Modern Cloud Infrastructure.",
        "annual_shareholders_meeting_slides.pdf": "Annual General Meeting Slide Deck\nFinancial Performance, Governance, FY2026 Guidance.",
        "customer_success_quarterly_business_review.pptx": "Customer Success Quarterly Business Review\nRenewal outlook, account health, support trends, and next-quarter priorities.",
        "renewable_energy_project_proposal_deck.pptx": "Renewable Energy Project Proposal\nSite assessment, installation timeline, budget, and projected energy output.",
        "engineering_team_sprint_review_slides.pptx": "Engineering Sprint Review\nCompleted stories, release notes, open risks, and the next sprint backlog.",

        # Image
        "golden_gate_bridge_sunset.jpg": "\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb" + "JPEG_IMAGE_DATA_SUNSET" * 50,
        "screenshot_2025_10_25_ui_bug.png": "\x89PNG\r\n\x1a\n" + "PNG_SCREENSHOT_DATA" * 50,
        "team_offsite_group_photo.jpeg": "\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb" + "JPEG_GROUP_PHOTO" * 50,
        "system_architecture_diagram.png": "\x89PNG\r\n\x1a\n" + "PNG_ARCHITECTURE_DIAGRAM" * 50,
        "kitchen_renovation_before_after_photo.jpg": "\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb" + "JPEG_RENOVATION_PHOTO" * 50,
        "product_dashboard_error_screenshot_2026-10-26.png": "\x89PNG\r\n\x1a\n" + "PNG_DASHBOARD_SCREENSHOT" * 50,

        # Other
        "docker_desktop_installer_v4.dmg": "DMG_BINARY_INSTALLER_PAYLOAD_MOCK_DATA" * 50,
        "python_environment_config.yaml": "name: sortiq-env\nchannels:\n  - conda-forge\ndependencies:\n  - python=3.11\n  - scikit-learn\n  - streamlit\n",
        "postgresql_backup_dump.sql": "CREATE TABLE users (id SERIAL PRIMARY KEY, name VARCHAR(255));\nINSERT INTO users VALUES (1, 'Alice');",
        "node_modules_cache_archive.zip": "PK\x03\x04" + "ZIP_ARCHIVE_MOCK_PAYLOAD" * 50,
        "terraform_staging_environment_config.tf": "terraform {\n  required_version = \">= 1.6.0\"\n}\n\nresource \"aws_s3_bucket\" \"staging_artifacts\" {\n  bucket = \"sortiq-staging-artifacts\"\n}",
        "application_server_error_log_2026-10-26.log": "2026-10-26T14:32:18Z INFO request_id=demo-1842 route=/api/files status=200 duration_ms=42\n2026-10-26T14:33:07Z WARN request_id=demo-1843 route=/api/predict status=503 retry=1",

        # Additional academic files
        "computer_science_data_structures_study_guide.pdf": "Study guide covering arrays, trees, graphs, and algorithmic complexity.",
        "chemistry_lab_report_titration.docx": "Chemistry laboratory report describing an acid-base titration experiment.",
        "history_research_paper_renaissance.pdf": "Research paper on trade networks and cultural change during the Renaissance.",
        "calculus_exam_review_derivatives.pdf": "Calculus exam review covering derivatives, limits, and optimization problems.",
        "biology_cell_structure_lecture_notes.pdf": "Lecture notes on cell organelles, membranes, and cellular transport.",
        "database_sql_practice_assignment.sql": "Course assignment with SQL queries for joins, grouping, and normalization.",
        "physics_quantum_mechanics_chapter_3.epub": "Textbook chapter introducing wave functions and quantum states.",
        "economics_microeconomics_course_syllabus.docx": "Course syllabus with weekly topics, readings, grading, and office hours.",
        "machine_learning_model_evaluation_lab.ipynb": "Notebook for a class lab comparing classification metrics and validation methods.",
        "research_methods_literature_review.pdf": "Academic literature review summarizing research methods and published findings.",

        # Additional resume and career files
        "olivia_bennett_data_analyst_resume_2026.pdf": "Data analyst resume highlighting SQL, dashboards, and business reporting.",
        "noah_williams_product_designer_cv.docx": "Curriculum vitae for a product designer with research and prototyping experience.",
        "priya_shah_software_engineer_cover_letter.pdf": "Cover letter applying for a software engineering position.",
        "ethan_brooks_cloud_architect_portfolio.pdf": "Portfolio of cloud architecture projects and technical case studies.",
        "ava_mitchell_marketing_manager_resume.docx": "Marketing manager resume with campaign, analytics, and team leadership experience.",
        "liam_carter_research_scientist_cv.pdf": "Research scientist CV with publications, laboratory work, and conference presentations.",
        "sophia_chen_ux_designer_job_application.pdf": "UX designer job application with experience summary and portfolio references.",
        "mateo_garcia_backend_developer_resume.pdf": "Backend developer resume featuring Python, APIs, and database systems.",
        "isabella_ross_professional_biography.docx": "Professional biography and career profile for a product consultant.",
        "jackson_lee_internship_cover_letter.docx": "Cover letter for a summer software engineering internship.",

        # Additional receipt and financial files
        "northwind_office_supplies_invoice_84021.pdf": "Office supply invoice with itemized purchases, tax, and total due.",
        "metro_transit_monthly_pass_receipt.pdf": "Transit pass receipt with purchase date and payment confirmation.",
        "electricity_account_statement_november_2026.pdf": "Monthly electricity statement showing usage and amount due.",
        "hotel_riverfront_booking_invoice_5538.pdf": "Hotel booking invoice listing room charges, taxes, and reservation dates.",
        "grocery_market_purchase_receipt_2026-10-18.pdf": "Grocery purchase receipt with subtotal, tax, and card payment.",
        "phone_plan_autopay_bill_october.pdf": "Mobile phone service bill with plan charges and autopay status.",
        "web_hosting_annual_renewal_invoice.pdf": "Annual web hosting renewal invoice with subscription details.",
        "parking_garage_payment_receipt_2714.pdf": "Parking garage receipt with entry time, duration, and payment amount.",
        "health_clinic_visit_statement_2026.pdf": "Clinic visit statement with service dates and patient balance.",
        "airline_booking_payment_confirmation.pdf": "Airline booking payment confirmation with fare and ticket details.",

        # Additional presentation files, including one intentional collision
        "quarterly_sales_enablement_training_deck.pptx": "Sales enablement training presentation with product updates and examples.",
        "renewable_energy_investor_pitch_slides.pptx": "Investor pitch deck for a renewable energy project.",
        "customer_research_findings_presentation.pptx": "Presentation summarizing customer interviews and product research.",
        "annual_engineering_strategy_keynote.pptx": "Engineering strategy keynote covering architecture and delivery goals.",
        "community_health_program_proposal_slides.pdf": "Proposal presentation for a community health program.",
        "mobile_app_launch_roadmap_deck.pptx": "Launch roadmap presentation with milestones, owners, and dates.",
        "climate_science_conference_talk.pptx": "Conference talk slides on climate science and regional projections.",
        "quarterly_finance_review_presentation.pptx": "Quarterly finance review with budget trends and forecasts.",
        "design_system_workshop_slides.pptx": "Workshop presentation covering interface patterns and design tokens.",
        "research_symposium_poster_presentation.pdf": "Research symposium presentation of methods and findings.",

        # Additional image files, including one intentional collision
        "family_hiking_trail_photo_2026.jpg": "Sample outdoor photograph for the image-classification demo.",
        "warehouse_inventory_barcode_scan.png": "Sample barcode scanner screenshot.",
        "coastal_village_travel_photograph.jpeg": "Sample coastal travel photograph.",
        "mobile_app_wireframe_mockup.png": "Sample mobile application wireframe.",
        "spring_garden_flower_closeup.jpg": "Sample garden flower photograph.",
        "network_topology_diagram_export.png": "Sample network topology diagram.",
        "product_packaging_front_view.webp": "Sample product packaging image.",
        "team_video_call_screenshot.png": "Sample video call screenshot.",
        "mountain_lake_camping_photo.jpeg": "Sample mountain lake photograph.",
        "quarterly_sales_chart_illustration.png": "Sample sales chart graphic.",

        # Additional miscellaneous files, including one intentional collision
        "nginx_reverse_proxy_configuration.conf": "Demo reverse proxy configuration file.",
        "python_linting_tool_configuration.toml": "Demo linting and formatting tool configuration.",
        "application_startup_diagnostic.log": "Demo application startup and diagnostic log.",
        "local_database_snapshot_archive.zip": "Demo archive placeholder for a local database snapshot.",
        "desktop_productivity_installer.pkg": "Demo software installer placeholder.",
        "staging_environment_variables.example": "Demo environment variable template with non-secret placeholder values.",
        "weekly_build_status_report.txt": "Demo build status summary with successful checks.",
        "browser_extension_package.crx": "Demo browser extension package placeholder.",
        "data_import_mapping_configuration.json": "Demo data import mapping configuration.",
        "system_monitoring_export.csv": "Demo system monitoring export with sample values.",
    }

    SANDBOX_SAMPLE_FILENAMES.update(dummy_files)
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


def request_demo_folder_load() -> None:
    """Prepare a fresh, isolated demo scan on the next Streamlit rerun."""
    for file_info in st.session_state.get("scanned_files", []):
        st.session_state.pop(f"destination_name_{file_info.file_name}", None)
    demo_folder = create_fresh_demo_folder()
    st.session_state["demo_folder_path"] = str(demo_folder.resolve())
    st.session_state["scan_folder_input"] = str(demo_folder.resolve())
    st.session_state["demo_folder_load_requested"] = True
    st.session_state["user_overrides"] = {}


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


# Keep the sample library ready, and give each app session a clean demo workspace.
ensure_sandbox_files()
if "demo_folder_path" not in st.session_state:
    initial_demo_folder = create_fresh_demo_folder()
    st.session_state["demo_folder_path"] = str(initial_demo_folder.resolve())

# Load model & metrics
model, model_error = get_cached_model()
metrics = load_model_metrics()

# App header
header_title, header_status = st.columns([3, 1], vertical_alignment="center")
with header_title:
    st.title("SortIQ")
    st.caption("Review recommendations, then organize your files safely.")
with header_status:
    st.markdown(
        '<div class="app-status-pill">● &nbsp;Private by design · Preview before moving</div>',
        unsafe_allow_html=True,
    )

# Sidebar Configuration
with st.sidebar:
    st.header("Settings")
    st.caption("Tune review and file-handling preferences.")

    with st.expander("Model status & benchmark", expanded=False):
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

        if metrics or DEFAULT_CONFUSION_MATRIX_PATH.is_file():
            if metrics:
                comp_data = [
                    {
                        "Model": m_name.replace("tfidf_", "").replace("_", " ").title(),
                        "Accuracy": f"{m_data.get('accuracy', 0) * 100:.1f}%",
                        "Macro-F1": f"{m_data.get('macro_f1', 0) * 100:.1f}%",
                        "SVD": (
                            str(m_data["svd_components"])
                            if m_data.get("svd_components") is not None
                            else "—"
                        ),
                    }
                    for m_name, m_data in metrics.get("models", {}).items()
                ]
                st.dataframe(
                    pd.DataFrame(comp_data),
                    use_container_width=True,
                    hide_index=True,
                )
            if DEFAULT_CONFUSION_MATRIX_PATH.is_file():
                st.image(
                    DEFAULT_CONFUSION_MATRIX_PATH,
                    caption="Confusion matrix · selected model",
                    use_container_width=True,
                )

    st.divider()
    st.subheader("Review & safety")
    confidence_threshold = st.slider(
        "Ambiguity Review Threshold",
        min_value=0.40,
        max_value=0.95,
        value=0.75,
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

    st.divider()
    st.subheader("Safe demo")
    st.caption("Each demo run starts with 100 sample files and six intentional destination-name conflicts across all categories.")
    if st.button("Start a fresh demo run", use_container_width=True):
        request_demo_folder_load()
        st.rerun()

# Step 1: Directory Selection & Scan
st.markdown('<div class="workflow-step">Step 1 · Choose a folder</div>', unsafe_allow_html=True)
st.markdown('<div class="workflow-title">Where should SortIQ look?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="workflow-help">Only direct files are scanned. Nothing is changed during scanning or prediction.</div>',
    unsafe_allow_html=True,
)

default_scan_dir = st.session_state.get(
    "scan_folder_input",
    st.session_state["demo_folder_path"],
)

with st.container(border=True):
    input_col, btn_sandbox_col, btn_scan_col = st.columns([5, 1.5, 1.5], vertical_alignment="bottom")

    with input_col:
        folder_input = st.text_input(
            "Folder path",
            value=default_scan_dir,
            placeholder="Paste or type a folder path",
            key="scan_folder_input",
        )

    with btn_sandbox_col:
        current_scan_path = st.session_state.get("current_scan_path")
        demo_folder_path = st.session_state.get("demo_folder_path")
        demo_folder_loaded = (
            "scanned_files" in st.session_state
            and current_scan_path is not None
            and demo_folder_path is not None
            and Path(current_scan_path).resolve() == Path(demo_folder_path).resolve()
        )
        st.button(
            "Demo folder loaded ✓" if demo_folder_loaded else "Use demo folder",
            use_container_width=True,
            help="Start a clean demo run with a fresh copy of the sample files.",
            disabled=demo_folder_loaded,
            on_click=request_demo_folder_load,
        )

    with btn_scan_col:
        scan_button = st.button(
            "Scan folder",
            use_container_width=True,
            type="primary",
        )

# Execute scan
demo_folder_load_requested = st.session_state.pop("demo_folder_load_requested", False)
initial_scan = "scanned_files" not in st.session_state
if scan_button or demo_folder_load_requested or initial_scan:
    is_valid, err_msg = validate_directory_path(folder_input)
    if not is_valid:
        st.error(f"Directory Error: {err_msg}")
        st.session_state["scanned_files"] = []
    else:
        scan_succeeded = False
        try:
            scanned = scan_directory(
                folder_input,
                recursive=False,
                include_hidden=include_hidden,
            )
            st.session_state["scanned_files"] = scanned
            st.session_state["current_scan_path"] = str(Path(folder_input).expanduser().resolve())
            st.session_state["predictions"] = None
            if demo_folder_load_requested:
                st.session_state["selected_files"] = {
                    file_info.file_name: True for file_info in scanned
                }
            scan_succeeded = True
        except Exception as exc:
            st.error(f"Scan operation failed: {exc}")
            st.session_state["scanned_files"] = []
        if scan_succeeded:
            st.rerun()

scanned_files = st.session_state.get("scanned_files", [])
current_path = st.session_state.get("current_scan_path", folder_input)

if not scanned_files:
    st.info("No files found here. Choose another folder or use the demo folder.")
else:
    total_size = sum(f.size_bytes for f in scanned_files)
    unique_exts = len({f.extension for f in scanned_files if f.extension})

    stat_files, stat_size, stat_extensions = st.columns(3)
    stat_files.metric("Files found", len(scanned_files))
    stat_size.metric("Total size", format_file_size(total_size))
    stat_extensions.metric("File types", unique_exts)

    # Step 2: Prediction & Interactive Review Table
    st.markdown('<div class="workflow-step">Step 2 · Review</div>', unsafe_allow_html=True)
    st.markdown('<div class="workflow-title">Check the recommendations</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="workflow-help">Only predictions below {int(confidence_threshold * 100)}% need a category decision. Confident predictions are ready as-is.</div>',
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
    name_conflict_files = set()
    for file_info in scanned_files:
        prediction = pred_map.get(
            file_info.file_name,
            {"predicted_category": "other", "confidence": 0.5},
        )
        current_category = category_for_file(
            file_info.file_name,
            prediction["predicted_category"],
            float(prediction.get("confidence", 0.5)),
            confidence_threshold,
            st.session_state["user_overrides"],
        )
        destination = (
            Path(current_path)
            / get_destination_folder(current_category)
            / file_info.file_name
        )
        if (
            destination.exists()
            and destination.resolve() != Path(file_info.full_path).resolve()
        ):
            name_conflict_files.add(file_info.file_name)

    pending_corrections = get_pending_corrections(
        [file_info.file_name for file_info in scanned_files],
        pred_map,
        st.session_state["user_overrides"],
        st.session_state["submitted_feedback"],
        confidence_threshold,
    )

    with st.container(border=True):
        st.markdown(
            '<div class="workflow-title" style="font-size: 1.05rem;">File list</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"{ambiguous_count} file(s) need review · "
            f"{len(scanned_files) - ambiguous_count} confident prediction(s)"
        )

        # Filter selector
        ambiguous_filter = f"Needs review ({ambiguous_count})"
        name_conflict_filter = f"Name conflicts ({len(name_conflict_files)})"
        filter_options = ["All files", ambiguous_filter, name_conflict_filter] + [
            category.capitalize() for category in category_list
        ]
        active_filter = st.selectbox(
            "Show",
            options=filter_options,
            index=0,
        )

    # Filtered subset of scanned files
    visible_files = []
    for f in scanned_files:
        pred_item = pred_map.get(f.file_name, {"predicted_category": "other", "confidence": 0.5})
        pred_cat = pred_item["predicted_category"]
        conf = float(pred_item.get("confidence", 0.5))

        if active_filter == "All files":
            visible_files.append(f)
        elif active_filter == ambiguous_filter:
            if conf < confidence_threshold:
                visible_files.append(f)
        elif active_filter == name_conflict_filter:
            if f.file_name in name_conflict_files:
                visible_files.append(f)
        else:
            if pred_cat.lower() == active_filter.lower():
                visible_files.append(f)

    # Bulk actions
    b_col1, b_col2, _ = st.columns([1.2, 1.2, 5])
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

    # File review cards
    reviewed_items = []

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
        destination_folder = get_destination_folder(current_cat)
        intended_destination = Path(current_path) / destination_folder / fname
        has_destination_collision = (
            intended_destination.exists()
            and intended_destination.resolve() != Path(file_info.full_path).resolve()
        )

        with st.container(border=True):
            row_col1, row_col2, row_col3 = st.columns(
                [0.35, 3.8, 3.2],
                vertical_alignment="center",
            )

            with row_col1:
                checked = st.checkbox(
                    f"Select {fname}",
                    value=is_selected,
                    key=f"sel_{fname}",
                    label_visibility="collapsed",
                )
                st.session_state["selected_files"][fname] = checked

            with row_col2:
                st.markdown(f"**{fname}**")
                st.caption(
                    f"{file_info.extension.upper() or 'FILE'} · "
                    f"{file_info.size_formatted} · Modified {file_info.modified_str}"
                )
                destination_name_key = f"destination_name_{fname}"
                if has_destination_collision:
                    if destination_name_key not in st.session_state:
                        st.session_state[destination_name_key] = fname
                    destination_name = st.text_input(
                        "Rename destination file",
                        key=destination_name_key,
                        help="This destination filename already exists. Choose another name or keep it to use automatic collision handling.",
                    )
                else:
                    st.session_state.pop(destination_name_key, None)
                    destination_name = fname

            with row_col3:
                is_ambiguous = conf < confidence_threshold
                conf_badge = (
                    f"<span class='status-ambiguous'>Needs review · {conf * 100:.0f}%</span>"
                    if is_ambiguous
                    else f"<span class='status-confident'>Confident · {conf * 100:.0f}%</span>"
                )
                cat_badge = (
                    f"<span class='pill-badge pill-{predicted_cat}'>"
                    f"{predicted_cat.capitalize()}</span>"
                )
                st.markdown(f"{cat_badge} {conf_badge}", unsafe_allow_html=True)

                if is_ambiguous:
                    cat_idx = (
                        category_list.index(current_cat)
                        if current_cat in category_list
                        else 0
                    )
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

                dest_folder = get_destination_folder(new_cat)
                st.caption(
                    f"Destination: {dest_folder}/{destination_name or '(enter a filename)'}"
                )

            if checked:
                reviewed_items.append(
                    {
                        "file_name": destination_name,
                        "source_path": file_info.full_path,
                        "category": new_cat,
                    }
                )

    # Active learning submission banner
    if pending_corrections:
        with st.container(border=True):
            feedback_col, save_col = st.columns([4, 1.5], vertical_alignment="center")
            with feedback_col:
                st.markdown(
                    f"**Active learning · {len(pending_corrections)} correction"
                    f"{'s' if len(pending_corrections) != 1 else ''} pending**"
                )
                st.caption(
                    "Confirmed corrections are saved locally. The pipeline retrains "
                    "automatically after every 5 confirmed corrections."
                )
            with save_col:
                if st.button(
                    "💾 Save Ambiguous Corrections",
                    use_container_width=True,
                    type="secondary",
                ):
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
    st.markdown('<div class="workflow-step">Step 3 · Preview</div>', unsafe_allow_html=True)
    st.markdown('<div class="workflow-title">Review the move plan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="workflow-help">Nothing moves in this step. Check each destination and any collision handling below.</div>',
        unsafe_allow_html=True,
    )

    if not reviewed_items:
        st.info("Select at least one file above to build a move preview.")
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

        ready_metric, rename_metric, skipped_metric, errors_metric = st.columns(4)
        ready_metric.metric("Ready to move", ready_count)
        rename_metric.metric("Will be renamed", rename_count)
        skipped_metric.metric("Will be skipped", skip_count)
        errors_metric.metric("Needs attention", error_count)

        st.markdown("#### Move preview")
        st.caption(
            f"Destination root: `{current_path}` · "
            f"Collision policy: {'rename' if collision_policy == COLLISION_RENAME else 'skip'}"
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
        st.markdown('<div class="workflow-step">Step 4 · Confirm</div>', unsafe_allow_html=True)
        st.markdown('<div class="workflow-title">Nothing moves until you approve</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="workflow-help">Files are never deleted or silently overwritten. Changing the preview resets your confirmation.</div>',
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
