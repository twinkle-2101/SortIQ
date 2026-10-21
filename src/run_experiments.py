"""Run the reproducible model comparison and save its artifacts."""

import argparse

from src.ml_pipeline import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--svd-components",
        type=int,
        default=100,
        help="Number of TruncatedSVD components for the SVD + Logistic Regression model.",
    )
    args = parser.parse_args()

    metrics = train_and_evaluate(svd_components=args.svd_components)
    print(f"Selected model: {metrics['selected_model']}")
    for name, result in metrics["models"].items():
        print(
            f"{name}: accuracy={result['accuracy']:.3f}, "
            f"macro_f1={result['macro_f1']:.3f}, "
            f"weighted_f1={result['weighted_f1']:.3f}, "
            f"runtime={result['runtime_seconds']:.3f}s"
        )
    print("Saved model, metrics, and confusion matrix.")


if __name__ == "__main__":
    main()
