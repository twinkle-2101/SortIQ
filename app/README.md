# SortIQ — Web Application (Member B Deliverable)

The SortIQ frontend is an interactive, safety-first Streamlit web application designed to categorize and organize messy directories (such as `~/Downloads`) using a machine learning inference pipeline.

---

## Architecture & Safety Guarantees

1. **Strictly Read-Only Scanning**:
   - The scanner lists regular files in the selected directory without altering or deleting any items.
   - Non-recursive by default to prevent scanning inside system directories or subfolders.
   - Ignores hidden files (e.g. `.DS_Store`, `.git`) by default.

2. **ML Classification & Ambiguity Detection**:
   - Filenames are tokenized and processed through the trained TF-IDF + TruncatedSVD + Logistic Regression pipeline (`models/best_model.joblib`).
   - Each prediction includes a probability confidence score.
   - Predictions below the configurable threshold (default `0.75`) are highlighted as **Ambiguous / Needs Review**. The 100-file demo corpus uses filename patterns recognized by the checked-in model and currently flags 15 files for review.
   - Category correction controls are shown only for ambiguous predictions; confident predictions keep their model category without requiring user input.

3. **Active Learning Retraining Loop**:
   - Users can correct a flagged ambiguous prediction using its category selector.
   - Users can click **Save Ambiguous Corrections** to register confirmed corrections.
   - When 5 reviewed corrections are accumulated, SortIQ automatically triggers retraining of the pipeline on starter data + feedback, re-evaluates the baseline models, updates `models/best_model.joblib`, and reloads the pipeline seamlessly without application restart.
   - Re-submitting the same filename/category correction is idempotent and does not increment the retraining count.

4. **Safe Dry-Run Preview**:
   - Before any file movement can occur, SortIQ computes an exact, read-only dry-run plan.
   - Detects collisions with pre-existing files in the target subfolder.
   - Resolves collisions using configurable policies:
     - `Rename`: Appends an incremental counter (`filename (1).ext`) without overwriting existing files.
     - `Skip`: Skips moving colliding files.

5. **Explicit User Confirmation**:
   - File movement requires explicit user confirmation via a checkbox tied to the current preview; changing the plan resets confirmation.
   - The execution button remains strictly disabled until confirmed.
   - Silent overwrites and file deletions are architecturally impossible.

---

## How to Run the Application

From the project root:

```bash
streamlit run app.py
```

The app will launch at `http://localhost:8501`.

### Disposable Demo Sandbox

SortIQ includes a built-in sandbox for demonstration purposes:
- Click **"Use demo folder"** or **"Start a fresh demo run"** to start with a clean copy of 100 varied sample files in a temporary workspace. Each run includes six intentional destination-name collisions, one in each category folder, to demonstrate both rename and skip policies; other demo destinations start empty.
- Use the **"Name conflicts"** file-list filter to focus on files whose recommended destination already contains the same name.
- A **"Rename destination file"** field appears only when a file's destination already exists. The source remains untouched until you confirm the move; leaving the name unchanged retains the automatic collision safeguard.
- You can test the entire scan, predict, override, dry-run, and organize workflow safely without touching your real Downloads folder.
- The demo files are for workflow demonstrations, not a labeled evaluation set; adding demo files does not change model accuracy or evaluation metrics.
