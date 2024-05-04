import torch


def train(model, optimizer, train_dl, val_dl, ref_data, epochs=40, lr=0.004, device=torch.device('cpu')): 
    for epoch in range(epochs):
        losses = []        
        model.train()
        for batch in train_dl:
            optimizer.zero_grad()
            anc_emb = model(batch['drawing'][:, None, :, :].type(torch.float32).to(device))
            pos_emb = model(batch['pos_img'][:, None, :, :].type(torch.float32).to(device))
            neg_emb = model(batch['neg_img'][:, None, :, :].type(torch.float32).to(device))

            #Mine triplets
            anc_emb, pos_emb, neg_emb = triplet_mining(anc_emb, pos_emb, neg_emb, 
                                                       batch["pos_idx"], batch["neg_idx"])

            loss = triplet_loss(anc_emb, pos_emb, neg_emb)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            
        print('Epoch: {}\t loss: {:.3f}'.format(epoch, np.mean(losses)))

        if epoch%5==0:
            ranking = []  
            model.eval()
            model.load_reference(ref_data, device)
            for batch in val_dl:
                rank = eval_fn(model, batch, device)
                ranking.extend(rank)
    
            print('Epoch: {}\t Average rank: {:.1f}\t top 10 acc: {:.1f}%\t top 1 acc: {:.1f}%\t'.format(epoch, np.mean(rank)+1, 100*np.mean(rank < 10), 100*np.mean(rank < 1)))



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


def eval_fn(model, batch, device=torch.device('cpu')):
    drawings = batch['drawing'][:, None, :, :].type(torch.float32).to(device)
    countries, distances = model.rank_countries(drawings)
    all_rank = torch.argsort(distances, axis=0)
    index = [countries.index(country) for country in batch['country_name']]
    rank = np.argmax(all_rank==index, axis=0)

    return rank
