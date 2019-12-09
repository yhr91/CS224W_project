from typing import Any
"""

@author: Ben, Yusuf, Kendrick
"""
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from datetime import datetime
import numpy as np
# import load_entrez
#import load_assoc
from load_assoc_ben import ProcessData
import copy
from neural_net import GNN
import utils
from sklearn.metrics import f1_score


def get_acc(model, loader, is_val=False, k=100):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    probs = []
    model.eval()
    preds, trues = [], []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
            prob = output.cpu().numpy()[:,1]
            pred = output.max(dim=1)[1]
            label = data.y

            # Prints predicted class distribution
            print(np.unique(pred.cpu(), return_counts=True)[1])

    if (is_val):
        probs.extend(prob[data.val_mask.cpu()])
        preds.extend(pred[data.val_mask.cpu()].cpu().numpy())
        trues.extend(label[data.val_mask.cpu()].cpu().numpy())
    else:
        probs.extend(prob[data.test_mask.cpu()])
        preds.extend(pred[data.test_mask].cpu().numpy())
        trues.extend(label[data.test_mask].cpu().numpy())

    res = {}
    res['f1'] = f1_score(trues,preds)
    correct = np.sum(np.array(trues)[np.argsort(probs)[-k:]])
    res['recall'] = correct/np.sum(trues)
    return res


def get_weight(x_, device):
    a, b = np.unique(x_.cpu().numpy(), return_counts=True)[1]
    return torch.tensor([(1 - a / (a + b)), (1 - b / (a + b))],
                        device = device)

def train(loader, epochs=100):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    model = GNN(1, 32, 2, 'GCNConv')
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
            weight = get_weight(batch.y, device=device)
            loss = criterion(out[batch.train_mask],
                             batch.y[batch.train_mask],weight=weight)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            print('loss on epoch', epoch, 'is', loss.item())

            if epoch % 1 == 0:
                val_acc.append(get_acc(model, loader, is_val=True)['recall'])
                print('Validation:', val_acc[-1])
                if (val_acc[-1] == np.max(val_acc)):
                    model_save = copy.deepcopy(model.cpu())
                    best_acc = val_acc[-1]
    return val_acc, model_save, best_acc, losses


def trainer(num_folds=5):
    # X_file = 'https://github.com/yhr91/CS224W_project/blob/master/Data/ForAnalysis/X/TCGA_GTEX_GeneExpression.csv?raw=true'
    # y_file = 'https://github.com/yhr91/CS224W_project/raw/master/Data/ForAnalysis/Y/NCG_cancergenes_list.txt'
    edgelist_file = 'https://github.com/yhr91/CS224W_project/blob/master/Data/PP-Decagon_ppi.csv?raw=true'
    # y_file = '../dataset_collection/DG-AssocMiner_miner-disease-gene.tsv'
    # edgelist_file = '../dataset_collection/PP-Decagon_ppi.csv'
    processed_data = ProcessData()
    X = processed_data.X
    X = torch.tensor(X.values, dtype=torch.float)

    for column in ['Parkinson Disease']:
        # print(column)
        y = processed_data.Y[column].tolist()
        test_size = int(0.2*np.sum(y))
        # y = processed_data.Y
        edges = processed_data.get_edges(edgelist_file)

        y = torch.tensor(y, dtype=torch.long)
        edges = torch.tensor(edges.values, dtype=torch.long)

        # Set up train and test sets:
        test_size = 0.2
        data_generator = utils.load_pyg_recall(X, edges, y, folds=num_folds, test_size=test_size)

        # 5-fold cross validation
        val_accs, models, accs = [], [], []
        #with open(column.replace(' ','-') + "_" + str(datetime.now())[:19].replace(' ',
        #                                                                       '-') + '.txt',
        #          'w') as f:
        for idx, loader in enumerate(data_generator):
            print('fold number:', idx)
            val_acc, model, best_acc, losses = train(loader)
            #f.write(str([losses, val_acc]))
            val_accs.append(val_acc)
            models.append(model)
            accs.append(best_acc)

        best_model = models[np.argmax(accs)]
        print('Best model accuracy:')
        acc = get_acc(model, loader, is_val=False)
        print(acc)
        #best_file.write(column+"\t"+str(acc))



if __name__ == '__main__':
    trainer()
