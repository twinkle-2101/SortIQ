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