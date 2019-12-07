import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader

import numpy as np
import load_entrez
import load_assoc
import copy
from neural_net import GNN
import utils

def get_acc(model, loader, is_val = False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    correct = 0
    total = 0
    model.eval()
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(dim=1)[1]
            label = data.y
            print(np.unique(pred.cpu(), return_counts=True)[1])
        correct += pred[data.test_mask].eq(label[data.test_mask]).sum().item()
        total += torch.sum(data.test_mask).item()
    return correct / total

def get_weight(x_):
    a, b = np.unique(x_, return_counts=True)[1]
    return torch.tensor([(1 - a / (a + b)), (1 - b / (a + b))])

# Identifies k nodes form each class within a given mask and removes the rest
def sample_from_mask(mask, data, k):
    counts = {}
    counts[0] = 0
    counts[1] = 0
    for i, val in enumerate(mask):
        if val == True:
            if data.y[i] == 0:
                counts[0] += 1
            else:
                counts[1] += 1
            if counts[data.y[i].item()] > k:
                mask[i] = False
    return mask


def train(loader, weight=None, epochs=50):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = GNN(3, 32, 2, 'GCNConv')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = F.nll_loss
    val_acc = []
    model_save = copy.deepcopy(model.cpu())

    for epoch in range(epochs):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()
            print('loss on epoch', epoch, 'is', loss.item())

            if epoch % 5 == 0:
                val_acc.append(get_acc(model, loader, is_val=True))
                print('Validation:', val_acc[-1])
                if (val_acc[-1] == np.max(val_acc)):
                    model_save = copy.deepcopy(model.cpu())
                    best_acc = val_acc[-1]

    return val_acc, model_save, best_acc

def trainer(num_folds = 5):
    X_file = 'https://github.com/yhr91/CS224W_project/blob/master/Data/ForAnalysis/X/TCGA_GTEX_GeneExpression.csv?raw=true'
    y_file = 'https://github.com/yhr91/CS224W_project/raw/master/Data/ForAnalysis/Y/NCG_cancergenes_list.txt'
    edgelist_file = 'https://github.com/yhr91/CS224W_project/blob/master/Data/PP-Decagon_ppi.csv?raw=true'
    # y_file = '../dataset_collection/DG-AssocMiner_miner-disease-gene.tsv'
    # edgelist_file = '../dataset_collection/PP-Decagon_ppi.csv'
    X = load_entrez.get_X(X_file)
    y = load_entrez.get_y(X, y_file)
    edges = load_entrez.get_edges(X, edgelist_file)

    X = torch.tensor(X.iloc[:, 1:4].values, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    edges = torch.tensor(edges.values, dtype=torch.long)

    # Set up train and test sets:
    data_generator = utils.load_pyg(X, edges, y, folds=num_folds)

    # 5-fold cross validation
    val_accs, models, accs = [], [], []
    for idx, loader in enumerate(data_generator):
        print('fold number:', idx)
        val_acc, model, best_acc = train(loader)
        val_accs.append(val_acc)
        models.append(model)
        accs.append(best_acc)

    best_model = models[np.argmax(accs)]
    #print('Best model accuracy:')
    #acc = get_acc(model, test_set, is_val = False)
    #print(acc)

    return val_accs

if __name__ == '__main__':
    trainer()
