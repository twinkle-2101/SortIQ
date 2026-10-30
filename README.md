# SortIQ

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Built with Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

SortIQ is a safety-first file organizer that uses machine learning to classify files from their names, recommends category folders, and lets you review a move plan before organizing anything.

> **Try it safely:** SortIQ includes a disposable demo folder. Start there before selecting a real directory.

## What it does

1. Scans a directory and predicts a category for each file based on its filename.
2. Flags low-confidence predictions for review and lets you correct their categories.
3. Previews the proposed organization, including how name conflicts will be handled.
4. Moves selected files only after you confirm the preview.

| Category | Destination folder | Examples |
| --- | --- | --- |
| `academic` | `Academic/` | Course notes, assignments, research papers |
| `resume` | `Resumes/` | Resumes, CVs, cover letters |
| `receipt` | `Receipts/` | Invoices, bills, receipts, statements |
| `presentation` | `Presentations/` | Slide decks, seminar talks, pitch decks |
| `image` | `Images/` | Photos, screenshots, graphics |
| `other` | `Other/` | Installers, logs, configuration files |

## Safety by design

- **Scanning is read-only.** Files are not changed during scanning or prediction.
- **Scans are non-recursive by default.** Hidden files are excluded unless you opt in.
- **Review before moving.** The app displays a dry-run plan before it can move files.
- **Confirmation is required.** You must confirm the current preview to execute it.
- **No silent overwrites.** Name conflicts can be renamed (for example, `report (1).pdf`) or skipped.
- **No file deletion.** SortIQ does not delete files; moving a selected file relocates it to its category folder.

## Quick start

### Requirements

- Python 3.11 or newer
- pip

### Install

Clone the repository and create an environment:

```shell
git clone https://github.com/twinkle-2101/SortIQ.git
cd SortIQ
python -m venv .venv
```

Activate the environment and install dependencies:

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**macOS / Linux**

```shell
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Launch the app

From the repository root, run:

```shell
streamlit run app.py
```

Open the local URL printed by Streamlit (typically <http://localhost:8501>).

### Try the disposable demo

Use **Start a fresh demo run** in the app to create a clean sample workspace with 100 varied files and six intentional destination-name conflicts across all categories. At the default 75% review threshold, the checked-in model currently flags 15 demo files for review. Try both rename and skip conflict policies without starting with your own Downloads folder. This demo corpus is for workflow demonstrations; it is not a labeled evaluation set and does not change the reported model accuracy.

When you are ready to organize your own files, enter the directory to scan in the app. Review the categories and destination plan, choose how to handle any conflicts, and confirm the preview before executing moves.

## Model and evaluation

SortIQ uses a TF-IDF text pipeline to classify filenames. The checked-in model is a TF-IDF + TruncatedSVD + Logistic Regression pipeline. Users can review low-confidence predictions and submit corrections; after each batch of five new confirmed corrections, the app retrains the model using the starter dataset and reviewed feedback.

The committed evaluation compares three pipelines on the same stratified 20% holdout, keeping variants of a filename family together:

| Pipeline | Accuracy | Macro F1 |
| --- | ---: | ---: |
| TF-IDF + Multinomial Naive Bayes | 74.2% | 0.716 |
| TF-IDF + Logistic Regression | 83.3% | 0.829 |
| TF-IDF + TruncatedSVD + Logistic Regression | **84.6%** | **0.841** |

**These are development results, not estimates of real-world performance.** The 1,200-row dataset is synthetic, and the holdout was used to select the best pipeline. See [data/README.md](data/README.md) and [ml/README.md](ml/README.md) for dataset provenance and methodology.

The model artifact was trained with scikit-learn 1.9.0. That version is pinned in `requirements.txt` because scikit-learn model artifacts are not guaranteed to work across versions.

### Run the model experiments

From the repository root:

```shell
python -m src.run_experiments
```

This retrains and evaluates the candidate pipelines, then writes the selected model to `models/best_model.joblib` and evaluation artifacts to `reports/`.

## Tests

Run the test suite from the repository root:

```shell
python -m pytest
```

## Local feedback data

Confirmed corrections are stored in `data/user_feedback.csv`; correction events are recorded in `data/user_feedback_events.jsonl`. These files are local and git-ignored. They may contain filenames from your files, so do not share or commit them.

## Repository layout

```text
SortIQ/
├── app.py                       # Streamlit interface and demo workflow
├── app/README.md                # App behavior and safety details
├── data/
│   ├── file_dataset.csv         # Synthetic starter dataset
│   ├── generate_dataset.py      # Reproducible dataset generator
│   └── README.md                # Dataset schema and limitations
├── ml/README.md                 # Experiment methodology and inference API
├── models/
│   └── best_model.joblib        # Trained classification pipeline
├── notebooks/
│   └── model_experiments.ipynb  # Exploratory experiments
├── reports/
│   ├── confusion_matrix.png     # Evaluation visualization
│   └── model_metrics.json       # Evaluation metrics and metadata
├── requirements.txt             # Python dependencies
├── src/                         # Scanner, classifier, feedback, organizer
└── tests/                       # Unit and integration tests
```
