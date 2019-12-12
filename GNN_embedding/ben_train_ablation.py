from typing import Any
"""

@author: Ben, Yusuf, Kendrick
"""
import torch
import torch.nn.functional as F
import numpy as np
from load_assoc_ben_with_features import ProcessData
import copy
from neural_net import GNN
import utils
import matplotlib.pyplot as plt


def train(loader, epochs=100):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = GNN(11, 32, 2, 'GCNConv')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = F.nll_loss
    val_acc = []
    losses = []
    model_save = copy.deepcopy(model.cpu())

    for epoch in range(epochs):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            weight = utils.get_weight(batch.y, device=device)
            loss = criterion(out[batch.train_mask],
                             batch.y[batch.train_mask],weight=weight)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            print('loss on epoch', epoch, 'is', loss.item())

            if epoch % 1 == 0:
                val_acc.append(utils.get_acc(model, loader, is_val=True)['f1'])
                print('Validation:', val_acc[-1])
                if (val_acc[-1] == np.max(val_acc)):
                    model_save = copy.deepcopy(model.cpu())
                    best_acc = val_acc[-1]
    return val_acc, model_save, best_acc, losses


def trainer(num_folds=5):
    # X_file = 'https://github.com/yhr91/CS224W_project/blob/master/Data/ForAnalysis/X/TCGA_GTEX_GeneExpression.csv?raw=true'
    # y_file = 'https://github.com/yhr91/CS224W_project/raw/master/Data/ForAnalysis/Y/NCG_cancergenes_list.txt'
    #edgelist_file = 'https://github.com/yhr91/CS224W_project/blob/master/Data/PP-Decagon_ppi.csv?raw=true'

    edgelist_file = '../dataset_collection/PP-Decagon_ppi.csv'
    #edgelist_file = '../dataset_collection/Decagon_GNBR.csv'
    # y_file = '../dataset_collection/DG-AssocMiner_miner-disease-gene.tsv'

    processed_data = ProcessData(edgelist_file)
    X = processed_data.X
    X = torch.tensor(X.values, dtype=torch.float)
    plt.figure()

    for column in ['Parkinson Disease']:
        # print(column)
        y = processed_data.Y[column].tolist()
        # y = processed_data.Y
        edges = processed_data.get_edges()

        y = torch.tensor(y, dtype=torch.long)
        edges = torch.tensor(edges.values, dtype=torch.long)

        # Set up train and test sets:
        data_generator = utils.load_pyg(X, edges, y, folds=num_folds, test_size=0.10)

        # 5-fold cross validation
        val_accs, models, accs = [], [], []

        for idx, loader in enumerate(data_generator):
            print('fold number:', idx)
            val_acc, model, best_acc, losses = train(loader)
            plt.plot(val_acc)
            #f.write(str([losses, val_acc]))
            val_accs.append(val_acc)
            models.append(model)
            accs.append(best_acc)

        best_model = models[np.argmax(accs)]
        print('Best model accuracy:')
        acc = utils.get_acc(best_model, loader, is_val=False)
        print(acc)


if __name__ == '__main__':
    trainer()
