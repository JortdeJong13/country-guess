import numpy as np


def triplet_mining(anc_emb, pos_emb, neg_emb, pos_idx, neg_idx):
    # All combinations
    anc_emb = anc_emb.tile((anc_emb.shape[0], 1))
    pos_emb = pos_emb.tile((pos_emb.shape[0], 1))
    neg_emb = neg_emb.repeat(neg_emb.shape[0], 1)
    pos_idx = pos_idx.tile((pos_idx.shape[0]))
    neg_idx = neg_idx.repeat((neg_idx.shape[0]))

    # Mask valid triplets
    valid = pos_idx != neg_idx

    return anc_emb[valid], pos_emb[valid], neg_emb[valid]


def train(model, train_dl, triplet_loss, optimizer):
    device = next(model.parameters()).device
    model.train()
    losses = []

    for batch in train_dl:
        optimizer.zero_grad(set_to_none=True)

        # Forward pass: compute embeddings
        anc_emb = model(batch["drawing"][:, None, :, :].float().to(device))
        pos_emb = model(batch["pos_img"][:, None, :, :].float().to(device))
        neg_emb = model(batch["neg_img"][:, None, :, :].float().to(device))

        # Mine triplets
        anc_emb, pos_emb, neg_emb = triplet_mining(
            anc_emb, pos_emb, neg_emb, batch["pos_idx"], batch["neg_idx"]
        )

        # Compute loss
        loss = triplet_loss(anc_emb, pos_emb, neg_emb)
        losses.append(loss.item())

        # Backpropagation
        loss.backward()
        optimizer.step()

    return np.mean(losses)


def evaluate(model, dl, ref_data):
    device = next(model.parameters()).device

    model.eval()
    model.load_reference(ref_data)

    country_names = []
    ranking = np.array([])

    for batch in dl:
        drawings = batch["drawing"][:, None, :, :].float().to(device)
        countries, distances = model.rank_countries(drawings)

        all_rank = np.argsort(distances, axis=0)
        index = [countries.index(country) for country in batch["country_name"]]
        rank = np.argmax(all_rank == index, axis=0)

        ranking = np.append(ranking, rank)
        country_names.extend(batch["country_name"])

    return country_names, ranking
