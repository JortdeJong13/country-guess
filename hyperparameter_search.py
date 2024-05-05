import random
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import mlflow
import intel_extension_for_pytorch as ipex

from countryguess.data import Dataset, TestDataset, ValDataset, TripletDataset
from countryguess.training import Model, triplet_mining, eval_fn


device = torch.device("xpu")

mlflow.set_experiment("hyperparameter_search")

idx=0
while True:
    params = {"channels": 8*random.randint(1, 4),
              "nr_conv_blocks": random.randint(4, 6),
              "embedding_size": 32*random.randint(2, 7),
              "shape": 32*random.randint(2, 6),
              "learning_rate": random.uniform(0.005, 0.012), 
              "margin": random.uniform(0.5, 1.5),
              "temperature": random.uniform(0.5, 1.1),
              "nr_epochs": random.randint(48, 128)}

    model = Model(**params).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'])
    model, optimizer = ipex.optimize(model, optimizer, dtype=torch.float32)
    triplet_loss = nn.TripletMarginLoss(margin=params['margin'])

    #Initialise datasets
    ref_data = Dataset(shape=(params["shape"], params["shape"]))
    train_data = TripletDataset(shape=(params["shape"], params["shape"]), temp=params["temperature"])
    val_data = ValDataset(shape=(params["shape"], params["shape"]), temp=params["temperature"])
    test_data = TestDataset(shape=(params["shape"], params["shape"]))

    train_dl = DataLoader(train_data, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=32)
    test_dl = DataLoader(test_data, batch_size=32)

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        
        #Start training
        for epoch in range(params['nr_epochs']):
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
        
            mlflow.log_metric("loss", np.mean(losses), epoch)
        
            if (epoch+1)%4==0:
                ranking = np.array([])
                model.eval()
                model.load_reference(ref_data)
                for batch in val_dl:
                    rank = eval_fn(model, batch)
                    ranking = np.append(ranking, rank)
        
                mlflow.log_metric("val_avg_rank", np.mean(ranking) + 1, epoch)
                mlflow.log_metric("val_top_10_acc", np.mean(ranking < 10), epoch)
                mlflow.log_metric("val_top_1_acc", np.mean(ranking < 1), epoch)

        #Evaluate model
        ranking = np.array([])
        model.eval()
        country_names = []
        for batch in test_dl:
            rank = eval_fn(model, batch)
            ranking = np.append(ranking, rank)
            country_names.extend(batch['country_name'])
            
        mlflow.log_metric("test_avg_rank", np.mean(ranking) + 1)
        mlflow.log_metric("test_top_10_acc", np.mean(ranking < 10))
        mlflow.log_metric("test_top_1_acc", np.mean(ranking < 1))

        print('{}\t Average rank: {:.2f}\t top 10 acc: {:.1f}%\t top 1 acc: {:.1f}%\t'
            .format(idx, np.mean(ranking)+1, 100*np.mean(ranking < 10), 100*np.mean(ranking < 1))) 
        idx+=1

