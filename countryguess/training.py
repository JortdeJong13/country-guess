import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.nn.losses import triplet_loss


def train(model, train_dl, val_dl, ref_data, epochs=40, lr=0.004):
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.Adam(learning_rate=lr)    
    
    for epoch in range(epochs):
        losses = []        
        model.train()
        for batch in train_dl:
            loss, grads = loss_and_grad_fn(model, batch)
            losses.append(loss.item())
            optimizer.update(model, grads)
            grad_fn = mx.grad(loss_fn)
            mx.eval(model.parameters(), optimizer.state)
            
        print('Epoch: {}\t loss: {:.3f}'.format(epoch, np.mean(losses)))

        if epoch%5==0:
            ranking = []  
            model.eval()
            model.load_reference(ref_data)
            for batch in val_dl:
                rank = eval_fn(model, batch)
                ranking.extend(rank)
    
            print('Epoch: {}\t Average rank: {:.1f}\t top 10 acc: {:.1f}%\t top 1 acc: {:.1f}%\t'.format(epoch, np.mean(rank)+1, 100*np.mean(rank < 10), 100*np.mean(rank < 1)))


def loss_fn(model, batch):
    anc_emb = model(mx.array(batch['drawing'], dtype=mx.float32))
    pos_emb = model(mx.array(batch['pos_img'], dtype=mx.float32))
    neg_emb = model(mx.array(batch['neg_img'], dtype=mx.float32))

    #Mine triplets
    anc_emb, pos_emb, neg_emb = triplet_mining(anc_emb, pos_emb, neg_emb, 
                                               batch["pos_idx"], batch["neg_idx"])
    
    return triplet_loss(anc_emb, pos_emb, neg_emb, reduction='mean')


def triplet_mining(anc_emb, pos_emb, neg_emb, pos_idx, neg_idx):
    #All combinations
    anc_emb = mx.tile(anc_emb, (anc_emb.shape[0], 1))
    pos_emb = mx.tile(pos_emb, (pos_emb.shape[0], 1))
    neg_emb = mx.repeat(neg_emb, neg_emb.shape[0], 0)
    pos_idx = mx.tile(mx.array(pos_idx), (pos_idx.shape[0]))
    neg_idx = mx.repeat(mx.array(neg_idx), neg_idx.shape[0])
    
    #Mask valid triplets
    valid = mx.array(np.where(pos_idx!=neg_idx)[0][..., np.newaxis])
    anc_emb = mx.take_along_axis(anc_emb, valid, axis=0)
    pos_emb = mx.take_along_axis(pos_emb, valid, axis=0)
    neg_emb = mx.take_along_axis(neg_emb, valid, axis=0)
    
    return anc_emb, pos_emb, neg_emb


def eval_fn(model, batch):
    drawings = mx.array(batch['drawing'], dtype=mx.float32)
    countries, distances = model.rank_countries(drawings)
    all_rank = np.argsort(distances, axis=0)
    index = [countries.index(country) for country in batch['country_name']]
    rank = np.argmax(all_rank==index, axis=0)

    return rank
