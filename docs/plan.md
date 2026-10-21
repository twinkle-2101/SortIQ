# Smart Downloads Folder Cleaner — Project Plan

## Goal
Build a Python ML app that predicts a file category from its filename and extension, recommends a destination folder, and moves selected files only after explicit user confirmation.

**Core categories:** `academic`, `resume`, `receipt`, `presentation`, `image`, `other`.

**Recommended stack:** Python, Pandas, NumPy, scikit-learn, TF-IDF, Naive Bayes, Logistic Regression, TruncatedSVD, Matplotlib, Streamlit, joblib, pytest. Use Google Colab for experiments and run the Streamlit file-organizing app locally. This project requires an SVD comparison.

## 1. Two-member split

### Member A — Dataset and ML
Owns:
- Category definitions and dataset creation/validation.
- Data cleaning, class counts, TF-IDF.
- Naive Bayes and Logistic Regression training.
- Fair train/test evaluation: accuracy, per-class precision/recall/F1, macro-F1, weighted-F1, confusion matrix.
- TruncatedSVD comparison using the same evaluation split as the baseline models.
- Save best complete pipeline and metrics.
- Write ML methodology/results for the report.

Deliverables:
- `data/file_dataset.csv`
- `notebooks/model_experiments.ipynb`
- `src/ml_pipeline.py`
- `models/best_model.joblib`
- `reports/model_metrics.json`
- Confusion-matrix image.

### Member B — Frontend and file operations
Owns:
- Streamlit UI.
- Read-only folder scanner and path validation.
- Category-to-folder mapping.
- Prediction review table and manual category override.
- Preview/dry-run and explicit confirmation before moving.
- Duplicate filename, permission, invalid path, and move-error handling.
- App tests and demo screenshots.

Deliverables:
- `app.py`
- `src/file_scanner.py`
- `src/category_mapping.py`
- `src/organizer.py`
- `tests/test_file_scanner.py`
- `tests/test_organizer.py`

### Shared work
Both agree on category labels and interfaces first, review each other's code, test integration, write README/conclusion, rehearse the demo, and understand the full project.

## 2. Contracts for independent work

### Dataset schema
`data/file_dataset.csv`:

| Column | Meaning | Example |
|---|---|---|
| `file_name` | Filename only, not a personal full path | `DBMS_unit3_notes.pdf` |
| `extension` | Lowercase suffix | `.pdf` |
| `category` | Correct label | `academic` |

Use synthetic, public, or consented examples. Keep class counts reasonably balanced, remove duplicate/conflicting labels, and describe the dataset honestly. A `.pdf` extension alone cannot tell a resume from a receipt or academic document.

### Model API
Member A provides `src/ml_pipeline.py` with:
- `load_model(model_path="models/best_model.joblib")`
- `predict_file_names(model, file_names)`

Prediction output should be a list of dictionaries containing `file_name`, `predicted_category`, and optional `confidence`. Save the full scikit-learn Pipeline (TF-IDF + classifier), not just the classifier.

### Destination mapping
- `academic` → `Academic`
- `resume` → `Resumes`
- `receipt` → `Receipts`
- `presentation` → `Presentations`
- `image` → `Images`
- `other` → `Other`

The app recommends these folders; users can review or override them.

## 3. Stages and phases

### Stage 0 — Agree on requirements
**Member A:** Define category meanings and provide 5–10 example filenames per class. Decide dataset sourcing/labeling.
**Member B:** Sketch the UI workflow and list file-operation risks.
**Together:** Read the faculty rubric; confirm any additional requirements and the UI; agree on categories, CSV schema, and model API. The SVD comparison is required for this project.

**Done when:** categories and interfaces are agreed; no real files are moved.

### Stage 1 — Work independently
**Member A:** Create CSV schema and starter dataset; validate labels, duplicates, missing values, and class counts.
**Member B:** Build a Streamlit skeleton using mock predictions; create scan/preview table, category override, destination preview. Keep move action disabled.

**Done when:** dataset validates and UI runs with mock rows.

### Stage 2 — ML baseline and read-only scanner
**Member A:** Split train/test with a fixed random seed and stratification where possible. Train TF-IDF + Multinomial Naive Bayes and TF-IDF + Logistic Regression on the same split. Record metrics and misclassifications.
**Member B:** Implement read-only scanning of direct files in a chosen folder; show filename/extension/size; handle invalid paths and permissions; test with temporary dummy files. Do not recurse by default.

**Done when:** both models run and scanner does not alter files.

### Stage 3 — Evaluation and safe organizer
**Member A:** Compare accuracy, per-class precision/recall/F1, macro-F1, weighted-F1, and confusion matrix. Analyse class imbalance and errors. Do not repeatedly tune against the held-out test set.
**Member B:** Implement destination mapping, dry-run preview, manual override, collision handling, and a separate move function. Never overwrite silently; skip or create a unique name. Add tests proving dry-run changes nothing.

**Done when:** metrics are documented and file movement is isolated/tested.

### Stage 4 — Integrate model and frontend
**Member A:** Choose model based on results and project needs; save/reload full pipeline; provide `load_model` and `predict_file_names`; document dependencies.
**Member B:** Replace mock predictions with the model API; display predicted class/confidence/destination; allow category overrides and file selection; show source/destination/count and collision policy before confirmation.

**Together:** test the same filenames in notebook and app; verify label consistency and missing-model error handling.

**Done when:** app displays real model predictions and still requires confirmation to move files.

### Stage 5 — SVD comparison and polish
**Member A:** Compare `TruncatedSVD` after TF-IDF against the baseline using the same split; record component count, runtime, and metrics.
**Member B:** Improve messages/loading states, add low-confidence review if supported, prepare a safe demo folder.

**Done when:** the required SVD comparison is reported and does not break the baseline.

### Stage 6 — End-to-end testing
**Member A:** Check saved model, inference consistency, metrics, and limitations.
**Member B:** Test scan, preview, override, cancel, confirmation, moves, collisions, and permission errors.
**Together:** Run the full workflow using disposable test files. Confirm nothing changes before confirmation, no overwrites occur, and partial failures are reported accurately.

### Stage 7 — Report and demo
**Member A:** Dataset, preprocessing, TF-IDF, model methods, metrics, confusion matrix, comparison, limitations, SVD if used.
**Member B:** Architecture, UI, scanning, destination mapping, confirmation, error handling, screenshots, application tests.
**Together:** Abstract, problem statement, objectives, conclusion, future scope, references, contribution table, README, requirements, and demo rehearsal.

## 4. Suggested project structure

```text
smart-downloads-cleaner/
├── app.py
├── plan.md
├── README.md
├── requirements.txt
├── data/file_dataset.csv
├── notebooks/model_experiments.ipynb
├── src/
│   ├── __init__.py
│   ├── ml_pipeline.py
│   ├── file_scanner.py
│   ├── category_mapping.py
│   └── organizer.py
├── models/best_model.joblib
├── reports/model_metrics.json
└── tests/
    ├── test_file_scanner.py
    ├── test_organizer.py
    └── test_ml_pipeline.py
```

Adapt to the real repository; do not duplicate existing files unnecessarily.

## 5. Safety requirements (mandatory)
1. Scanning and prediction are read-only.
2. Preview/dry-run is the default.
3. Files move only after explicit confirmation.
4. Never delete files or silently overwrite existing files.
5. Handle collisions, invalid paths, permissions, and per-file failures.
6. Do not recursively scan subfolders by default.
7. Prevent source/destination path mistakes and recursive self-moves.
8. Test using temporary folders and dummy files.
9. Do not upload filenames or contents to external services.
10. Keep a manual override because filename-based predictions can be wrong.

## 6. Test checklist

| ID | Test | Expected result |
|---|---|---|
| D1 | Dataset validation | Required columns and approved labels only |
| M1 | Train both classifiers | Both complete successfully |
| M2 | Compare models | Same held-out test split and comparable features |
| M3 | Save and reload model | Predictions remain consistent |
| M4 | Metrics and confusion matrix | Correct labels and values |
| F1 | Scan valid folder | Files listed; no changes made |
| F2 | Invalid folder | Clear error message |
| F3 | Preview recommendations | Files remain unchanged |
| F4 | Manual category override | Destination recommendation updates |
| F5 | Cancel confirmation | No files move |
| F6 | Confirm test move | Only selected file moves |
| F7 | Destination name collision | Existing file is not overwritten |
| F8 | Permission/move error | Clear per-file failure report |
| I1 | Missing model file | Helpful error; app does not crash |
| I2 | Full workflow | Scan → predict → review → confirm → results |
| R1 | Follow README on a fresh setup | Project can be run reproducibly |

Record real test outcomes; do not claim tests passed unless run.

## 7. Definition of done
- [ ] Dataset and labels documented.
- [ ] TF-IDF implemented.
- [ ] Naive Bayes and Logistic Regression compared fairly.
- [ ] Required metrics and confusion matrix reported.
- [ ] Full model pipeline saved and reload tested.
- [ ] SVD comparison included using the same evaluation split as the baseline models.
- [ ] Streamlit app scans, predicts, previews, and allows overrides.
- [ ] Moves require explicit confirmation.
- [ ] Collisions and errors handled safely.
- [ ] Tests, README, requirements, report, and screenshots complete.
- [ ] Both members can explain the full project.

## 8. Instructions for another LLM

Use this file as project context. Before coding, inspect the actual repository, faculty rubric, dataset, and existing files. Identify the current stage and what is complete versus unverified. Work one stage at a time, preserve working code, keep Member A's ML module independent from Member B's app module, and respect the shared CSV/model contracts. Do not invent dataset sizes, accuracy scores, files, or test results. Provide Windows PowerShell-friendly commands when relevant. Test file operations only in disposable directories. Never implement automatic movement without preview and explicit confirmation. When working interactively, give one actionable step and wait for the user to report the result.

### Required response format from an LLM
- Current stage and goal
- Member A tasks / Member B tasks
- Files to create or change
- Step-by-step instructions
- Commands and expected output
- Tests and actual results
- Remaining issues and handoff

## 9. Immediate next actions
1. Read the faculty rubric and confirm SVD requirements.
2. Agree on category definitions and schema.
3. Add this file as `plan.md` in the repository.
4. Member A creates and validates a small labelled dataset.
5. Member B builds a Streamlit skeleton with mock rows only.
6. Review both deliverables, then begin Stage 2.

**Status:** Plan prepared. No model, dataset, or app behaviour should be described as complete until it is implemented and verified.
