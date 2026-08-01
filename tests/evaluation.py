"""
Evaluate the model on test data.

Usage:
    uv run --only-group app python -m tests.evaluation --model_name <model_name> --drawing_store_url <drawing_store_url>
"""

import argparse
import json
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from countryguess.data import ReferenceDataset
from countryguess.evaluation import EvaluationDataset
from countryguess.model import fetch_model
from countryguess.training import evaluate

REPO_ROOT = Path(__file__).resolve().parents[1]
ACCURACY_SHIELD_PATH = REPO_ROOT / "data" / "accuracy-shield.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the model on test data.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="triplet_model",
        help="Name of the model to evaluate",
    )
    parser.add_argument(
        "--drawing_store_url",
        type=str,
        default="http://127.0.0.1:8080",
        help="URL of the drawing store",
    )
    return parser.parse_args()


def print_report(report, metric_width=25, value_width=15):
    """Print a formatted table of evaluation results."""
    # Table header and separator
    header = f"| {'Metric'.ljust(metric_width)} | {'Value'.ljust(value_width)} |"
    separator = f"|{'-' * (metric_width + 2)}|{'-' * (value_width + 2)}|"

    # Generate rows dynamically
    rows = [
        f"| {metric.ljust(metric_width)} | {str(value).ljust(value_width)} |"
        for metric, value in report.items()
    ]

    # Combine header, separator, and rows into a single table
    table = "\n".join([header, separator] + rows)

    print(f"### Evaluation Results\n\n{table}\n")


def update_accuracy_shield(top_1_accuracy):
    """Write the current top-1 accuracy to the repository badge file."""
    shield = {
        "schemaVersion": 1,
        "label": "accuracy",
        "message": f"{top_1_accuracy:.1f}%",
        "color": "blue",
    }
    ACCURACY_SHIELD_PATH.write_text(
        json.dumps(shield, indent=2) + "\n", encoding="utf-8"
    )


def main():
    args = parse_args()
    model_name = args.model_name

    # Load model
    model, _ = fetch_model(model_name)

    # Initialize datasets and dataloader
    ref_data = ReferenceDataset(shape=model.shape)
    test_data = EvaluationDataset(args.drawing_store_url, shape=model.shape)
    test_dl = DataLoader(test_data, batch_size=32)  # type: ignore

    # Evaluate the model
    _, ranking, _ = evaluate(model, test_dl, ref_data)

    # Results
    nr_test_samples = len(test_data)
    if nr_test_samples == 0:
        raise RuntimeError("No validated evaluation drawings were returned")

    avg_rank = np.mean(ranking) + 1
    top_10_acc = 100 * np.mean(ranking < 10)
    top_1_acc = 100 * np.mean(ranking < 1)

    # Generate report
    report = {
        "Model Name": model_name,
        "Number of Test Samples": nr_test_samples,
        "Average Rank": f"{avg_rank:.2f}",
        "Top 10 Accuracy": f"{top_10_acc:.1f}%",
        "Top 1 Accuracy": f"{top_1_acc:.1f}%",
    }

    # Print report
    print_report(report)
    update_accuracy_shield(top_1_acc)


if __name__ == "__main__":
    main()
