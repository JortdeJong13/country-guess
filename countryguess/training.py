import torch
import numpy as np

def triplet_mining(anc_emb, pos_emb, neg_emb, pos_idx, neg_idx):
    #All combinations
    anc_emb = anc_emb.tile((anc_emb.shape[0], 1))
    pos_emb = pos_emb.tile((pos_emb.shape[0], 1))
    neg_emb = neg_emb.repeat(neg_emb.shape[0], 1)
    pos_idx = pos_idx.tile((pos_idx.shape[0]))
    neg_idx = neg_idx.repeat((neg_idx.shape[0]))
    
    #Mask valid triplets
    valid = pos_idx!=neg_idx
    
    return anc_emb[valid], pos_emb[valid], neg_emb[valid]


def eval_fn(model, batch):
    drawings = batch['drawing'][:, None, :, :].type(torch.float32).to(model.linear.weight.device)
    countries, distances = model.rank_countries(drawings)
    all_rank = np.argsort(distances, axis=0)
    index = [countries.index(country) for country in batch['country_name']]
    rank = np.argmax(all_rank==index, axis=0)

    return rank