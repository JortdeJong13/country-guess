# Imports
import numpy as np
import random
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import mlflow

from countryguess.data import Dataset, TestDataset
from countryguess.model import CustomEmbeddingModel, TripletModel, get_device
from countryguess.generate import TripletDataset, ValDataset
from countryguess.training import train, evaluate


device = get_device()
mlflow.set_experiment("parameter-search")


def train_session():
    params = {
        "channels": 4 * random.randint(1, 6),
        "nr_conv_blocks": random.randint(2, 6),
        "embedding_size": 256 * random.randint(1, 8),
        "shape": 48 * random.randint(1, 4),
        "learning_rate": 0.01,  # random.uniform(0.006, 0.012),
        "margin": 0.8,  # random.uniform(0.3, 1.2),
        "temperature": random.uniform(0.4, 1),
        "nr_epochs": random.randint(10, 25),
    }

    # Set up model
    model = TripletModel(CustomEmbeddingModel(**params)).to(device)
    params["embedding_model"] = model.embedding_model.__class__.__name__

    optimizer = torch.optim.SGD(model.parameters(), lr=params["learning_rate"])
    triplet_loss = nn.TripletMarginLoss(margin=params["margin"])

    print("\n".join(f"{key}: {value}" for key, value in params.items()))

    # Initialise datasets
    ref_data = Dataset(shape=(params["shape"], params["shape"]))
    train_data = TripletDataset(
        shape=(params["shape"], params["shape"]), temp=params["temperature"]
    )
    val_data = ValDataset(
        shape=(params["shape"], params["shape"]), temp=params["temperature"]
    )

    # Initialise dataloaders
    train_dl = DataLoader(train_data, batch_size=24, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_data, batch_size=24, num_workers=2)

    with mlflow.start_run() as run:
        mlflow.log_params(params)

        # Start training
        for epoch in range(params["nr_epochs"]):
            loss = train(model, train_dl, triplet_loss, optimizer)
            mlflow.log_metric("loss", loss, epoch)
            print(
                "Epoch: {}/{}\t loss: {:.3f}".format(
                    epoch + 1, params["nr_epochs"], loss
                )
            )

            if (epoch + 1) % 4 == 0:
                _, ranking, _ = evaluate(model, val_dl, ref_data)

                mlflow.log_metric("val_avg_rank", np.mean(ranking) + 1, epoch)
                mlflow.log_metric("val_top_10_acc", np.mean(ranking < 10), epoch)
                mlflow.log_metric("val_top_1_acc", np.mean(ranking < 1), epoch)

                print(
                    "Epoch: {}/{}\t Average rank: {:.2f}\t top 10 acc: {:.1f}%\t top 1 acc: {:.1f}%\t".format(
                        epoch + 1,
                        params["nr_epochs"],
                        np.mean(ranking) + 1,
                        100 * np.mean(ranking < 10),
                        100 * np.mean(ranking < 1),
                    )
                )

    # Evaluate model
    test_data = TestDataset(shape=(params["shape"], params["shape"]))
    test_dl = DataLoader(test_data, batch_size=32)

    country_names, ranking, conf_scores = evaluate(model, test_dl, ref_data)

    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.log_metric("nr_test_samples", len(test_data))
        mlflow.log_metric("test_avg_rank", np.mean(ranking) + 1)
        mlflow.log_metric("test_top_10_acc", np.mean(ranking < 10))
        mlflow.log_metric("test_top_1_acc", np.mean(ranking < 1))

    print(
        "Average rank: {:.2f}\t top 10 acc: {:.1f}%\t top 1 acc: {:.1f}%\t".format(
            np.mean(ranking) + 1,
            100 * np.mean(ranking < 10),
            100 * np.mean(ranking < 1),
        )
    )


if __name__ == "__main__":
    for _ in range(250):
        train_session()
