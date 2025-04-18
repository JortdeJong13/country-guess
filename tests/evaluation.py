"""
Evaluate the model on test data.

Usage:
    python -m tests.evaluation --model_name <model_name>
"""

import argparse
import logging

import numpy as np
from torch.utils.data import DataLoader

from countryguess.data import Dataset, TestDataset
from countryguess.model import fetch_model
from countryguess.training import evaluate

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the model on test data.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="triplet_model",
        help="Name of the model to evaluate",
    )
    return parser.parse_args()


def generate_report(model_name, nr_test_samples, avg_rank, top_10_acc, top_1_acc):
    """Generate a formatted table of evaluation results."""
    # Define column widths
    metric_width, value_width = 25, 15

    # Table header and separator
    header = f"| {'Metric'.ljust(metric_width)} | {'Value'.ljust(value_width)} |"
    separator = f"|{'-' * (metric_width + 2)}|{'-' * (value_width + 2)}|"

    # Metrics to display
    metrics = [
        ("Model Name", model_name),
        ("Number of Test Samples", nr_test_samples),
        ("Average Rank", f"{avg_rank:.2f}"),
        ("Top 10 Accuracy", f"{top_10_acc:.1f}%"),
        ("Top 1 Accuracy", f"{top_1_acc:.1f}%"),
    ]

    # Generate rows dynamically
    rows = [
        f"| {metric.ljust(metric_width)} | {str(value).ljust(value_width)} |"
        for metric, value in metrics
    ]

    # Combine header, separator, and rows into a single table
    table = "\n".join([header, separator] + rows)

    return f"### Evaluation Results\n\n{table}\n"


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
    country_names, ranking, conf_scores = evaluate(model, test_dl, ref_data)

    # Results
    nr_test_samples = len(test_data)
    avg_rank = np.mean(ranking) + 1
    top_10_acc = 100 * np.mean(ranking < 10)
    top_1_acc = 100 * np.mean(ranking < 1)

    # Generate report
    report = generate_report(
        model_name, nr_test_samples, avg_rank, top_10_acc, top_1_acc
    )

    # Log the report
    logger.info(report)

    # Return results for CI/CD
    return report, nr_test_samples, avg_rank, top_10_acc, top_1_acc


if __name__ == "__main__":
    main()
