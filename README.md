# SortIQ — Smart Downloads Folder Cleaner 🧹

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![Streamlit App](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](app.py)

SortIQ is a Python machine learning application that predicts file categories from filenames and extensions, recommends organized destination folders, and safely moves selected files only after explicit user confirmation.

---

## 🎯 Project Overview & Objective

User download folders quickly accumulate hundreds of unstructured files. Conventional rule-based extensions cannot distinguish between a resume PDF, a store receipt PDF, and an academic lecture PDF. SortIQ solves this using **Natural Language Processing (TF-IDF tokenization)** and **Machine Learning classification**, augmented with **Dimensionality Reduction (TruncatedSVD)** and an **Active Learning human-in-the-loop retraining mechanism**.

### Core Categories & Destination Mapping

| Category | Typical Files | Destination Folder |
|---|---|---|
| `academic` | Course notes, lecture slides, assignments, research papers | `Academic/` |
| `resume` | Resumes, CVs, cover letters, portfolios | `Resumes/` |
| `receipt` | Invoices, payment receipts, utility bills, bank statements | `Receipts/` |
| `presentation` | Keynote decks, seminar talks, slide presentations | `Presentations/` |
| `image` | Photos, screenshots, wallpapers, graphics, diagrams | `Images/` |
| `other` | Installers, scripts, archives, logs, misc files | `Other/` |

---

## 👥 Two-Member Division of Work

### Member A — Dataset, ML Pipeline & Experiments
- Dataset generation and balance validation (1,200 synthetic rows across 6 categories in `data/file_dataset.csv`).
- Data preprocessing with token normalization and TF-IDF feature extraction.
- Stratified holdout evaluation keeping filename families together to prevent data leakage.
- Fair model comparison: **Multinomial Naive Bayes** vs. **Logistic Regression** vs. **TruncatedSVD + Logistic Regression**.
- Required TruncatedSVD comparison (evaluating component dimensionality, runtime, and macro-F1).
- Complete pipeline persistence (`models/best_model.joblib`), metrics serialization (`reports/model_metrics.json`), and confusion matrix generation (`reports/confusion_matrix.png`).
- Continuous active-learning retraining engine (`src/feedback.py`) triggering automatic retrain after 5 confirmed corrections.

### Member B — Frontend, Safe File Operations & Testing
- Modern, interactive Streamlit web application (`app.py`).
- Read-only directory scanner with path and permission validation (`src/file_scanner.py`).
- Category-to-folder mapping and path resolution (`src/category_mapping.py`).
- Safe organizer engine with collision resolution (auto-rename / skip) and dry-run preview (`src/organizer.py`).
- Ambiguity detection: highlighting predictions below the user-defined confidence threshold.
- Interactive category override interface with one-click ML feedback submission.
- End-to-end integration tests and safety verification (`tests/test_file_scanner.py`, `tests/test_organizer.py`, `tests/test_integration.py`, `tests/test_app.py`).
- Built-in disposable demo folder generator for safe sandbox testing.

---

## 📊 ML Model Comparison & Results

Evaluated on the exact same stratified held-out test split (240 unseen filename families):

| Pipeline Candidate | Accuracy | Macro F1-Score | Weighted F1-Score | SVD Components | Fit+Eval Runtime |
|---|---|---|---|---|---|
| **TF-IDF + Multinomial Naive Bayes** | 74.2% | 0.716 | 0.716 | N/A | 0.023s |
| **TF-IDF + Logistic Regression** | 83.3% | 0.829 | 0.829 | N/A | 0.065s |
| **TF-IDF + TruncatedSVD + Logistic Regression (Selected)** | **84.6%** | **0.841** | **0.841** | **100** | **0.473s** |

> **Key Finding on SVD:** On this synthetic holdout, adding 100 TruncatedSVD components improved accuracy by about 1.3 percentage points and macro-F1 from 0.829 to 0.841 compared with TF-IDF + Logistic Regression. Runtime is environment-dependent.

The checked-in model was trained with scikit-learn 1.9.0. The exact version is
pinned in `requirements.txt` because scikit-learn model artifacts are not
guaranteed to be compatible across versions.

---

## 🛡️ Mandatory Safety Principles

1. **Read-Only Scanning**: Scanning and ML prediction never alter or move files.
2. **Preview / Dry-Run by Default**: Detailed movement plan is shown before any disk write.
3. **Explicit Confirmation**: File movements require an explicit confirmation checkbox.
4. **Zero Silent Overwrites**: Collisions are either auto-renamed (`file (1).ext`) or safely skipped.
5. **No File Deletion**: SortIQ never deletes any user files.
6. **Non-Recursive by Default**: Protects against accidentally traversing entire system volumes.

---

## 🚀 Quick Start Guide

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/twinkle-2101/SortIQ.git
cd SortIQ
pip install -r requirements.txt
```

### 2. Run Model Experiments & Retraining

To train the models and generate metrics locally:

```bash
python -m src.run_experiments
```

### 3. Launch the Streamlit Web Application

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

### 4. Run the Full Test Suite

Run all unit and integration tests:

```bash
python -m pytest
```

---

## 📁 Repository Structure

```text
SortIQ/
├── app.py                         # Streamlit Web Application (Member B)
├── app/
│   └── README.md                  # Frontend Architecture & Documentation
├── data/
│   ├── file_dataset.csv           # 1,200-row labeled starter dataset (Member A)
│   ├── generate_dataset.py        # Dataset generator script
│   └── README.md                  # Dataset schema documentation
├── docs/
│   └── plan.md                    # Project requirements and phase plan
├── ml/
│   └── README.md                  # ML methodology and experiments documentation
├── models/
│   └── best_model.joblib          # Persisted full scikit-learn pipeline
├── notebooks/
│   └── model_experiments.ipynb    # Exploration & Colab experiment notebook
├── reports/
│   ├── confusion_matrix.png       # Confusion matrix visualization
│   └── model_metrics.json         # Serialized holdout evaluation metrics
├── requirements.txt               # Project dependencies
├── src/
│   ├── __init__.py
│   ├── category_mapping.py        # Category to directory mapping (Member B)
│   ├── feedback.py                # Active learning feedback & retrain loop (Member A)
│   ├── file_scanner.py            # Read-only folder scanner & validator (Member B)
│   ├── ml_pipeline.py             # Preprocessing, training, and inference (Member A)
│   ├── organizer.py               # Dry-run planner & safe move engine (Member B)
│   └── run_experiments.py         # Baseline comparison execution script
└── tests/
    ├── test_app.py                # Streamlit helper & fallback tests (Member B)
    ├── test_category_mapping.py   # Destination resolution tests (Member B)
    ├── test_dataset.py            # Dataset contract tests (Member A)
    ├── test_file_scanner.py       # F1-F2 Scanner & permission tests (Member B)
    ├── test_integration.py        # I1-I2 End-to-end workflow tests (Member B)
    ├── test_ml_pipeline.py        # M1-M4 Model & feedback tests (Member A)
    └── test_organizer.py          # F3-F8 Dry-run, collision & move tests (Member B)
```
