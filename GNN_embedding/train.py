"""
    @author: Ben, Yusuf, Kendrick
    Trains models to predict disease gene associations.
    """
import torch
import torch.nn.functional as F
from datetime import datetime
import numpy as np
from load_assoc import ProcessData
import neural_net
import pandas as pd
import utils
from torch.utils.tensorboard import SummaryWriter
import copy
import random
from collections import defaultdict
import time

def train(data, tasks, args, ind, fold_num, step=50):
    '''
    data is a Data object
    tasks is a list of ((train, val, test), label) tuples
    '''
    # Set up tensorboard writer
    writer = SummaryWriter(args.dir_)

    # retrieve the requested NN model + push to device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = neural_net.get_neural_network(args).to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = F.nll_loss
    model_save = copy.deepcopy(model.state_dict())
    best_score = -np.inf

    epochs = args.epochs
    tasks_ = list(enumerate(tasks))
    for epoch in range(epochs):
        model.train()
        loss_sum = 0

        # Model selection for multi task
        if args.score.split('_')[1] == 'sum':
            epoch_score = 0
        elif args.score.split('_')[1] == 'max':
            epoch_score = np.inf

        if args.shuffle:
            tasks_ = tasks_.copy()
            np.random.shuffle(tasks_)

        # If MTL, iterate over all diseases, if not then just single disease
        for idx, ((train_mask, val_mask, _), y) in tasks_:
            train_mask, val_mask, y = train_mask.to(device), val_mask.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(data)
            weight = utils.get_weight(y, device=device)

            # if multi-task learning, apply the corresponding final linear layer
            if args.MTL:
                out = model.tasks[idx](out) # Last layer of NN that is specific to each task
            else:
                out = model.final(out)

            out = F.log_softmax(out, dim=1) # Softmax
            loss = criterion(out[train_mask], y[train_mask], weight=weight)

            if args.MTL:
                res, _ = utils.get_acc(model, data, val_mask, y, task=idx)
            else:
                res, _ = utils.get_acc(model, data, val_mask, y, task=None)

            # Model selection metric
            if args.score.split('_')[0] == 'f1':
                task_score = res['f1']
            elif args.score.split('_')[0] == 'loss':
                task_score = -loss.item()

            # Model selection for multi task
            if args.score.split('_')[1] == 'sum':
                epoch_score += task_score
            elif args.score.split('_')[1] == 'max':
                epoch_score = min(epoch_score, task_score)

            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

            # once per 'step' epochs tensorboard writing, at the 'disease' level
            if epoch % step == 0:
                writer.add_scalar('TrainLoss/disease_'+str(ind), loss.item(), fold_num * epochs + epoch)
                writer.add_scalar('ValF1/disease_'+str(ind), res['f1'], fold_num * epochs + epoch)
                writer.add_scalar('ValRecall/disease_' + str(ind), res['recall'], fold_num * epochs + epoch)
                if args.MTL:
                    print('disease ', idx,' loss on epoch', epoch, 'is', loss.item())
                else:
                    print('disease ', ind,' loss on epoch', epoch, 'is', loss.item())

        # Every epoch, test if best model, then save
        if epoch_score > best_score:
            model_save = copy.deepcopy(model.state_dict())
            best_score = epoch_score

        # Once per 'step' epochs tensorboard writing, at the epoch level
        if epoch % step == 0:
            if args.MTL:
                print('Overall MTL loss on epoch', epoch, 'is', loss_sum)
                writer.add_scalar('MTL/TrainLoss' + str(ind), loss_sum, fold_num * epochs + epoch)
            else:
                pass

    writer.flush()
    writer.close()
    model.load_state_dict(model_save)
    return model, best_score

def trainer(args, num_folds=10):
    start = time.time()

    edgelist_file = {
        'Decagon': '../dataset_collection/PP-Decagon_ppi.csv',
        'GNBR': '../dataset_collection/GNBR-edgelist.csv',
        'Decagon_GNBR': '../dataset_collection/Decagon_GNBR_2.csv',
        'Pathways': '../dataset_collection/bio-pathways-network.csv'
    }[args.dataset]

    if args.heterogeneous:
        edgelist_file = '../dataset_collection/PP-Decagon_ppi.csv'
        edgelist_file2 = '../dataset_collection/GNBR-edgelist.csv'
    het_str = 'heterogeneous' * args.heterogeneous

    if args.use_features:
        feat_str = 'feats'
    else:
        feat_str = 'no_feats'

    args.dir_ = './tensorboard_runs/'+args.expt_name+'/'+\
                args.network_type+'_'+args.dataset+'_'+feat_str+'_'+het_str

    # If creating a multi graph. TO-DO: move this to load_assoc
    if args.edge_attr>1:
        edge_attr = pd.read_csv('../dataset_collection/Decagon_GNBR_MultiEdges.csv')
        edge_attr = torch.tensor(edge_attr.T.values, dtype=torch.float)
    else:
        edge_attr = None

    # load graph
    processed_data = ProcessData(edgelist_file, use_features=args.use_features)
    X = processed_data.X
    X = torch.tensor(X.values, dtype=torch.float)
    edges = processed_data.get_edges()
    edges = torch.tensor(edges.values, dtype=torch.long)
    data = utils.load_graph(X, edges, edge_attr=edge_attr)

    if args.heterogeneous:
        edges2 = processed_data.get_edges(edgelist_file=edgelist_file2)
        edges2 = torch.tensor(edges2.values, dtype=torch.long)
        edges2 = utils.insert_edges(edges2)
        data.edge_index = (data.edge_index, edges2.t().contiguous())

    # load labels: returns all disease indices corresponding to given disease classes
    if args.sample_diseases: # for hyperparameter tuning
        sel_diseases = [469, 317, 473, 6, 426]
    elif args.disease_class:
        if args.disease_class is not list:
            args.disease_class = [args.disease_class]
        sel_diseases = processed_data.get_disease_class_idx(args.disease_class)
        
        if args.holdout:
            sel_diseases.append(args.holdout)
    
    else:
        sel_diseases = range(len(processed_data.Y.columns))
    
    processed_data.Y = processed_data.Y.iloc[:,sel_diseases]
    disease_test_scores = defaultdict(list)
    
    # If multi task learning
    if args.MTL:
        args.tasks = len(sel_diseases)
        # Load masks and labels for each task
        masks_and_labels = []
        for ind, col in enumerate(processed_data.Y):
            y = torch.tensor(processed_data.Y[col].values.astype('int'), dtype=torch.long)
            masks_and_labels.append((utils.load_masks(y), y))

        # Iterate over folds
        for f in range(num_folds):
            print('fold', f, 'out of', num_folds)
            tasks = []
            # collect the set of masks and labels associated with this fold
            for masks, label in masks_and_labels:
                tasks.append((masks[f], label))

            # Use the list of training datasets for all diseases at a specifc fold to train
            model, score = train(data, tasks, args, len(tasks), f)

            # compute accuracy
            for ind, (masks, label) in enumerate(masks_and_labels):
                test_score, output = utils.get_acc(model, data, masks[f][2], label, task=ind)
                print('On fold', f, 'and disease', ind, 'score is', test_score)
                disease_test_scores[ind].append(test_score)

    # If single task learning
    else:
        args.tasks = 1
        for ind, column in enumerate(processed_data.Y):
            print(ind,column,'out of',len(processed_data.Y))

            y = torch.tensor(processed_data.Y[column].values.astype('int'), dtype=torch.long)
            masks = utils.load_masks(y)

            for f in range(num_folds):
                task = [(masks[f], y)]
                model, score = train(data, task, args, ind, f)

                test_score, output = utils.get_acc(model, data, masks[f][2], y, task=None)
                print('On fold', f, 'and disease', ind, 'score is', test_score)
                disease_test_scores[ind].append(test_score)

    # Save results
    end = time.time()
    time_taken = end - start

    # Save model state and node embeddings
    torch.save(model.state_dict(), args.dir_+'/model')

    # Save results and time
    np.save(args.dir_+'/time',time_taken)

if __name__ == '__main__':
    import argparse
    dt = str(datetime.now())[5:19].replace(' ', '_').replace(':', '-')
    
    parser = argparse.ArgumentParser(description='Define network type and dataset.')
    parser.add_argument('--network-type', type=str, choices=['GEO_GCN', 'SAGE', 'SAGE_GCN', 'GCN', 'GEO_GAT', 'ADA_GCN','NO_GNN', ], default='GEO_GCN')
    parser.add_argument('--dataset', type=str, choices=['Decagon', 'GNBR', 'Decagon_GNBR', 'Pathways'], default='GNBR')
    parser.add_argument('--expt_name', type=str, default=dt)
    parser.add_argument('--use-features', type=bool, nargs='?', const=True, default=False)
    parser.add_argument('--MTL', type=bool, nargs='?', const=True, default=False)
    parser.add_argument('--in-dim', type=int, default=13)
    parser.add_argument('--hidden-dim', type=int, default=24)
    parser.add_argument('--out-dim', type=int, default=2)
    parser.add_argument('--edge-attr', type=int, default=1)
    parser.add_argument('--num-heads', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--shuffle', type=bool, nargs ='?', const=True, default=False)
    parser.add_argument('--score', type=str, default='loss_sum')
    parser.add_argument('--holdout', type=int, default=False)
    parser.add_argument('--sample-diseases', type=bool, nargs='?', const=True, default=False)
    parser.add_argument('--disease_class', type=str, default='nervous system disease')
    #parser.add_argument('--heterogeneous', type=bool, nargs='?', const=True, default=False)
    args = parser.parse_args()

    if not args.use_features and args.in_dim > 1:
        print('Cannot have in dim of', args.in_dim, 'changing to 1.')
        args.in_dim = 1
    
    args.heterogeneous = args.network_type == 'ADA_GCN'

    def seed_torch(seed=1029):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    seed_torch()
    print('Experiment name is', args.expt_name)
    trainer(args)
