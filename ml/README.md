# ML experiments

The comparison uses the same fixed, stratified holdout for TF-IDF + Multinomial
Naive Bayes, TF-IDF + Logistic Regression, and TF-IDF + TruncatedSVD +
Logistic Regression. The split is stratified by category at the filename-family
level: suffix variants of one stem are kept together so the test set measures
generalization to unseen filename stems rather than memorization of nearly
identical variants.

TruncatedSVD is compared with Logistic Regression because SVD-transformed
features can contain negative values, which Multinomial Naive Bayes does not
support. The number of SVD components and the random seed are recorded in the
metrics.

The resulting metrics are based on a synthetic dataset. They are reproducible
development results, not estimates of performance on real user files. The
highest-scoring pipeline is selected using this comparison holdout, so these
scores should not be presented as an independent final estimate.

From the repository root, run:

```powershell
python -m src.run_experiments
```

Artifacts are written to `models/best_model.joblib`,
`reports/model_metrics.json`, and `reports/confusion_matrix.png`.
The committed pipeline was trained using scikit-learn 1.9.0, which is pinned in
`requirements.txt`; `load_model` reports a clear error when the installed
scikit-learn version does not match the model artifact.

## Inference API

`src.ml_pipeline` exposes the shared interface expected by the app:

```python
from src.ml_pipeline import load_model, predict_file_names

model = load_model()  # models/best_model.joblib
predictions = predict_file_names(
    model,
    ["machine_learning_course_notes.pdf", "orange_cat_photo.jpg"],
)
```

Each result contains `file_name`, `predicted_category`, and `confidence` (the
highest predicted class probability). The function preserves input order,
returns an empty list for an empty input, and rejects blank or non-string
filenames. `load_model` raises a helpful `FileNotFoundError` if training has
not yet produced the model file.

## Learning from reviewed ambiguous files

The model does not ask users to label every file. After a person confirms a
category correction for an ambiguous prediction, the app should call
`record_correction`:

```python
from src.feedback import record_correction

result = record_correction("quarterly_store_invoice.pdf", "receipt")
if result["retrained"]:
    model = load_model()  # Reload the replaced model in the app.
```

Corrections are stored locally in `data/user_feedback.csv`; a git-ignored
`data/user_feedback_events.jsonl` records confirmed correction events. Repeated
submissions of the same filename and category are idempotent and do not count
twice; a new category correction for that filename is a distinct event. Both
files may contain private filenames and should remain local.

After every five confirmed corrections since the previous successful training,
the pipeline reruns the fixed baseline comparison and fits the selected model
on the starter data plus the latest reviewed labels. Matching starter rows are
replaced by their reviewed labels. Before the threshold is reached,
`record_correction` returns `retrained=False` and the app can continue using the
current model.

If training fails, the exception should be shown to the user. The previous
model file is preserved; corrections remain recorded and will be retried after
the next correction. When `retrained=True`, reload the saved model (and clear
any application cache) to use the updated pipeline. The evaluation scores still
describe the fixed synthetic dataset, not the feedback-trained model's
real-world performance.