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

    logger.info("-" * 40)
    logger.info("--- Evaluation Results ---")
    logger.info(f"Model: {model_name} evaluated on {nr_test_samples} samples:")
    logger.info(
        f"Average rank: {avg_rank:.2f}\t Top 10 accuracy: {top_10_acc:.1f}%\t Top 1 accuracy: {top_1_acc:.1f}%"
    )
    logger.info("-" * 40)

    # Return results for CI/CD
    return nr_test_samples, avg_rank, top_10_acc, top_1_acc


if __name__ == "__main__":
    main()
