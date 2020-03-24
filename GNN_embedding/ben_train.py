from typing import Any # what is this lol
"""

@author: Ben, Yusuf, Kendrick
"""
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from datetime import datetime
import numpy as np
# import load_entrez
from load_assoc_ben import ProcessData
# from load_assoc_ben_with_features import ProcessData
import copy
from neural_net import GNN
import utils
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score


def train(loader, epochs=100):
    writer = SummaryWriter('tensorboard_runs/gcn')
    writer.add_scalar('he', 123)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = GNN(1, 32, 2, 'SAGEConv')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = F.nll_loss
    val_f1 = []
    losses = []
    model_save = copy.deepcopy(model.cpu())

    for epoch in range(epochs):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            writer.add_graph(model, batch)
            weight = utils.get_weight(batch.y, device=device)
            loss = criterion(out[batch.train_mask],
                             batch.y[batch.train_mask],weight=weight)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            print('loss on epoch', epoch, 'is', loss.item())

            if epoch % 1 == 0:
                val_f1.append(utils.get_acc(model, loader, is_val=True)['f1'])
                print('Validation:', val_f1[-1])
                if (val_f1[-1] == np.max(val_f1)):
                    model_save = copy.deepcopy(model.cpu())
                    best_f1 = val_f1[-1]
    writer.flush()
    writer.close()
    return val_f1, model_save, best_f1, losses


def trainer(num_folds=5):
    # X_file = 'https://github.com/yhr91/CS224W_project/blob/master/Data/ForAnalysis/X/TCGA_GTEX_GeneExpression.csv?raw=true'
    # y_file = 'https://github.com/yhr91/CS224W_project/raw/master/Data/ForAnalysis/Y/NCG_cancergenes_list.txt'
    #edgelist_file = 'https://github.com/yhr91/CS224W_project/blob/master/Data/PP-Decagon_ppi.csv?raw=true'
    # y_file = '../dataset_collection/DG-AssocMiner_miner-disease-gene.tsv'

    # Decagon alone
    # edgelist_file = '../dataset_collection/PP-Decagon_ppi.csv'

    # GNBR alone
    edgelist_file = '../dataset_collection/GNBR-edgelist.csv'

    # Decagon+GNBR
    #edgelist_file = '../dataset_collection/Decagon_GNBR.csv'

    # edgelist_file = '../dataset_collection/PP-Decagon_ppi.csv'
    processed_data = ProcessData(edgelist_file)
    X = processed_data.X
    X = torch.tensor(X.values, dtype=torch.float)
    curr_results = {}
    for ind, column in enumerate(processed_data.Y):
        if (ind > 0 and ind % 100 == 0): # write 100
        # columns to each file,
            # so if it
            # fails then
        # it's ok
            dt = str(datetime.now())[8:19].replace(' ', '-')
            curr_file = open(f'bensmodels/{dt}-{ind}-thru-{ind+100}.txt', 'w')
            curr_file.write(str(curr_results))
            curr_file.close()
            curr_results = {}

        # print(column)
        y = processed_data.Y[column].tolist()

        # y = processed_data.Y
        edges = processed_data.get_edges()

        y = torch.tensor(y, dtype=torch.long)
        edges = torch.tensor(edges.values, dtype=torch.long)

        # Set up train and test sets:
        test_size = .1
        data_generator = utils.load_pyg(X, edges, y, folds=num_folds, test_size=test_size)

        # 5-fold cross validation
        val = [] # val f1 scores
        models = [] # save models for now
        model_f1s = [] # save model recalls

        for loader in data_generator:
            val_f1, model, best_f1, _ = train(loader)
            val.append(val_f1)
            models.append(model)
            model_f1s.append(best_f1)

        best_model = models[np.argmax(model_f1s)]
        print('Best model accuracy:')
        test_recall = utils.get_acc(best_model, loader, is_val=False)
        print(test_recall)
        curr_results[ind] = [val, test_recall]

    dt = str(datetime.now())[8:19].replace(' ', '-')
    curr_file = open(f'bensmodels/{dt}-{ind}-thru-{ind+100}.txt', 'w')
    curr_file.write(str(curr_results))
    curr_file.close()

if __name__ == '__main__':
    trainer()
