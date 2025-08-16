"""
Evaluate the model on test data.

Usage:
    python -m tests.evaluation --model_name <model_name>
"""

import argparse
import json

import numpy as np
from torch.utils.data import DataLoader

from countryguess.data import Dataset, TestDataset
from countryguess.model import fetch_model
from countryguess.training import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the model on test data.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="triplet_model",
        help="Name of the model to evaluate",
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


def main():
    args = parse_args()
    model_name = args.model_name

    # Load model
    model, _ = fetch_model(model_name)

    # Initialize datasets and dataloader
    ref_data = Dataset(shape=model.shape)
    test_data = TestDataset(shape=model.shape)
    test_dl = DataLoader(test_data, batch_size=32)

    # Evaluate the model
    _, ranking, _ = evaluate(model, test_dl, ref_data)

    # Results
    nr_test_samples = len(test_data)
    avg_rank = np.mean(ranking) + 1
    top_10_acc = 100 * np.mean(ranking < 10)
    top_1_acc = 100 * np.mean(ranking < 1)

    # Generate report
    report = {
        "model_name": model_name,
        "nr_test_samples": nr_test_samples,
        "avg_rank": avg_rank,
        "top_10_acc": top_10_acc,
        "top_1_acc": top_1_acc,
    }

    # Print report
    print_report(report)

    # Save report
    with open("data/evaluation.json", "w") as file:
        json.dump(report, file)


if __name__ == "__main__":
    main()
