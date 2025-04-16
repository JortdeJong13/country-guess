"""Training and evaluation functions for the country guess model."""

import numpy as np
import torch
from torch.nn.functional import cross_entropy

from countryguess.generate import generate_scribbles


def triplet_mining(anc_emb, pos_emb, neg_emb, pos_idx, neg_idx):
    """Combine embeddings to form valid triplets."""
    # All combinations
    anc_emb = anc_emb.tile((anc_emb.shape[0], 1))
    pos_emb = pos_emb.tile((pos_emb.shape[0], 1))
    neg_emb = neg_emb.repeat(neg_emb.shape[0], 1)
    pos_idx = pos_idx.tile((pos_idx.shape[0]))
    neg_idx = neg_idx.repeat((neg_idx.shape[0]))

    # Mask valid triplets
    valid = pos_idx != neg_idx

    return anc_emb[valid], pos_emb[valid], neg_emb[valid]


def train(model, train_dl, triplet_loss, optimizer, lmbda=0.1, train_emb=True):
    """Train the model for a single epoch."""
    device = next(model.parameters()).device
    model.train()
    losses, scribble_losses = [], []

    for batch in train_dl:
        optimizer.zero_grad(set_to_none=True)

        # Forward pass: compute embeddings
        anc_emb = model(batch["drawing"][:, None, :, :].float().to(device))
        pos_emb = model(batch["pos_img"][:, None, :, :].float().to(device))
        neg_emb = model(batch["neg_img"][:, None, :, :].float().to(device))

        # Scribble detection
        scribbles = generate_scribbles(len(batch["drawing"]))
        scribble_emb = model(scribbles[:, None, :, :].float().to(device))
        pred = model.scribble_detection(torch.cat([anc_emb, scribble_emb], dim=0))
        scribble = torch.tensor([False] * len(anc_emb) + [True] * len(scribble_emb)).to(
            device
        )

        # Mine triplets
        anc_emb, pos_emb, neg_emb = triplet_mining(
            anc_emb, pos_emb, neg_emb, batch["pos_idx"], batch["neg_idx"]
        )

        # Compute loss
        loss = triplet_loss(anc_emb, pos_emb, neg_emb)
        scribble_loss = cross_entropy(pred, scribble)

        losses.append(loss.item())
        scribble_losses.append(scribble_loss.item())

        # Backpropagation
        if train_emb:
            loss += lmbda * scribble_loss
            loss.backward()
        else:
            loss.backward(retain_graph=True)
            model.embedding_model.zero_grad()
            scribble_loss.backward()

        # Optimize step
        optimizer.step()

    return np.mean(losses), np.mean(scribble_losses)


@torch.no_grad
def evaluate(model, dl, ref_data):
    """Evaluate the model and return the ranking and confidence scores."""
    device = next(model.parameters()).device

    model.eval()
    model.load_reference(ref_data)

    country_names = []
    conf_scores = []
    ranking = np.array([])

    for batch in dl:
        drawings = batch["drawing"][:, None, :, :].float().to(device)
        countries, scores = model.rank_countries(drawings)

        true_countries = np.array(batch["country_name"])
        ranks = np.where(countries == true_countries[:, None])[1].astype(int)
        batch_indices = np.arange(len(ranks))
        scores = scores[batch_indices, ranks]

        ranking = np.append(ranking, ranks)
        country_names.extend(batch["country_name"])
        conf_scores.extend(scores)

    return country_names, ranking, conf_scores
