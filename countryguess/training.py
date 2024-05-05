import torch
from torch import nn
import numpy as np

from countryguess.utils import poly_to_img


class Model(nn.Module):
    def __init__(self, nr_conv_blocks=4, channels=16, embedding_size=128, dropout=0.2, shape=64, **kwargs):
        super().__init__()
        self.shape = (shape, shape)
        self._ref_countries = None
        
        conv_blocks = [self.conv_block(1, channels)]
        for idx in range(nr_conv_blocks-1):
            conv_blocks.append(self.conv_block(channels * 2**idx, 
                                               channels * 2**(idx+1)))
        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(int(channels * 2**(nr_conv_blocks-1) * (shape / (2**nr_conv_blocks))**2), 
                                embedding_size)

    
    def conv_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.MaxPool2d(2))


    def __call__(self, x):
        x = self.conv_blocks(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        x = self.linear(x)
        
        return x

    
    def load_reference(self, ref_data):
        assert ref_data.shape==self.shape
        self._ref_countries = {}
        for idx in range(len(ref_data)):
            img = poly_to_img(ref_data[idx], ref_data.shape)
            embedding = self(torch.tensor(img[None, None, :, :], dtype=torch.float32).to(self.linear.weight.device))
            self._ref_countries[ref_data.country_name[idx]] = embedding

    
    @torch.no_grad
    def rank_countries(self, drawings):
        embedding = self(drawings)
        countries = []
        distances = []

        if not self._ref_countries:
            raise Exception("First the reference dataset needs to be loaded!")            

        for country, ref_emb in self._ref_countries.items():
            countries.append(country)
            distance = torch.linalg.norm(embedding - ref_emb, axis=-1)
            distances.append(distance.cpu())

        return countries, np.array(distances)


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