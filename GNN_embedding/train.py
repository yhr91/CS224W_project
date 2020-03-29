"""
@author: Ben, Yusuf, Kendrick

Trains models to predict disease gene associations.
"""
import torch
import torch.nn.functional as F
from datetime import datetime
import numpy as np
from load_assoc import ProcessData
from neural_net import GNN
import utils
from torch.utils.tensorboard import SummaryWriter
import copy
import random
import os

def train(loader, args, epochs=100):
    writer = SummaryWriter('tensorboard_runs/gcn/'+args.network_type+'_'+args.dataset)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = GNN(args.in_dim, args.hidden_dim, args.out_dim, args.network_type)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = F.nll_loss
    best_f1 = 0
    model_save = copy.deepcopy(model.cpu())

    for epoch in range(epochs):
        batches = list(loader)
        random.shuffle(batches)
        model.train()

        for batch in batches:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            weight = utils.get_weight(batch.y, device=device)
            loss = criterion(out[batch.train_mask],
                             batch.y[batch.train_mask],weight=weight)
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train', loss.item(), epoch)
            print('loss on epoch', epoch, 'is', loss.item())

            if epoch % 1 == 0:
                val_f1 = utils.get_acc(model, loader, is_val=True)['f1']
                writer.add_scalar('F1/validation', val_f1, epoch)
                print('Validation:', val_f1)

                if val_f1 > best_f1:
                    model_save = copy.deepcopy(model.cpu())
                    best_f1 = val_f1

    writer.flush()
    writer.close()
    return model_save, best_f1


def trainer(args, num_folds=5):
    edgelist_file = {
        'Decagon': '../dataset_collection/PP-Decagon_ppi.csv',
        'GNBR': '../dataset_collection/GNBR-edgelist.csv',
        'Decagon_GNBR': '../dataset_collection/Decagon_GNBR.csv'
    }[args.dataset]
    
    processed_data = ProcessData(edgelist_file)
    X = processed_data.X
    X = torch.tensor(X.values, dtype=torch.float)
    # TODO switch to using tensorboard for tracing results
    curr_results = {}

    for ind, column in enumerate(processed_data.Y):
        # TODO what does the code below do???
        if (ind > 0 and ind % 100 == 0): # write 100
        # columns to each file,
            # so if it
            # fails then
        # it's ok
            # Todo change to using tensorboard for this
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
        models = [] # save models for now
        model_f1s = [] # save model recalls

        for loader in data_generator:
            model, best_f1 = train(loader, args)
            models.append(model)
            model_f1s.append(best_f1)

        best_model = models[np.argmax(model_f1s)]
        print('Best model f1:')
        test_recall = utils.get_acc(best_model, loader, is_val=False)
        print(test_recall)
        curr_results[ind] = [test_recall]

    dt = str(datetime.now())[8:19].replace(' ', '-')
    curr_file = open(f'bensmodels/{dt}-{ind}-thru-{ind+100}.txt', 'w')
    curr_file.write(str(curr_results))
    curr_file.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Define network type and dataset.')
    parser.add_argument('--network_type', type=str, choices=['GCNConv', 'SAGEConv', 'GATConv'], default='GCNConv')
    parser.add_argument('--dataset', type=str, choices=['Decagon', 'GNBR', 'Decagon_GNBR'], default='GNBR')
    parser.add_argument('--in_dim', type=int, default=11)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--out_dim', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=3)
    args = parser.parse_args()


    def seed_torch(seed=1029):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    seed_torch()
    trainer(args)