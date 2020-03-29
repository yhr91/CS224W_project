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

def train(loader, args, ind, it, epochs=500):
    if args.use_features:
        feat_str = 'feats'
    else:
        feat_str = 'no_feats'

    writer = SummaryWriter('./tensorboard_runs/gcn/'+args.expt_name+'/'
                           +args.network_type+'_'+args.dataset+'_'+feat_str)
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = GNN(args.in_dim, args.hidden_dim, args.out_dim, args.network_type)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = F.nll_loss
    best_f1 = 0
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
        writer.add_scalar('TrainLoss/disease_'+str(ind), loss.item(), it*epochs+epoch)
        print('loss on epoch', epoch, 'is', loss.item())

        if epoch % 1 == 0:
            val_f1 = utils.get_acc(model, loader, is_val=True)['f1']
            writer.add_scalar('ValF1/disease_'+str(ind), val_f1, it*epochs+epoch)

            val_recall = utils.get_acc(model, loader, is_val=True)['recall']
            writer.add_scalar('ValRecall/disease_' + str(ind), val_f1, it*epochs+epoch)
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

    processed_data = ProcessData(edgelist_file, use_features=args.use_features)
    X = processed_data.X
    X = torch.tensor(X.values, dtype=torch.float)
    disease_test_scores = {}

    # Iterate over diseases
    dir_ = './tensorboard_runs/gcn/'+args.expt_name
    
    for ind, column in enumerate(processed_data.Y):
        if ind > 100: break # TODO: Remove this later on. For testing purposes only
        
        y = processed_data.Y[column].tolist()
        edges = processed_data.get_edges()
        
        y = torch.tensor(y, dtype=torch.long)
        edges = torch.tensor(edges.values, dtype=torch.long)
        
        # Set up train and test sets:
        test_size = .1
        data_generator = utils.load_pyg(X, edges, y,
                                        folds=num_folds, test_size=test_size)
            
        # 5-fold cross validation
        models = [] # save models for now
        model_scores = [] # save model recalls

        for it,loader in enumerate(data_generator):
            model, score = train(loader, args, ind, it)
            models.append(model)
            model_scores.append(score)

        best_model = models[np.argmax(model_scores)]
        best_test_score = utils.get_acc(best_model, loader, is_val=False)
        print('Best model f1:')
        print(best_test_score)
        disease_test_scores[ind] = [best_test_score]

        # Save results
        np.save(dir_+'/results', disease_test_scores)

if __name__ == '__main__':
    import argparse
    dt = str(datetime.now())[8:19].replace(' ', '_').replace(':', '-')
    
    parser = argparse.ArgumentParser(description='Define network type and dataset.')
    parser.add_argument('--network-type', type=str, choices=['GCNConv', 'SAGEConv', 'GATConv'], default='GCNConv')
    parser.add_argument('--dataset', type=str, choices=['Decagon', 'GNBR', 'Decagon_GNBR'], default='GNBR')
    parser.add_argument('--expt_name', type=str, default=dt)
    parser.add_argument('--use-features', type=bool, nargs='?', const=True, default=False)
    parser.add_argument('--in-dim', type=int, default=11)
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--out-dim', type=int, default=2)
    parser.add_argument('--num-heads', type=int, default=3)
    args = parser.parse_args()
    
    if not args.use_features and args.in_dim > 1:
        print('Cannot have in dim of', args.in_dim, 'changing to 1.')
        args.in_dim = 1

    def seed_torch(seed=1029):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    seed_torch()
    trainer(args)
