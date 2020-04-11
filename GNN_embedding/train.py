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
import utils
from torch.utils.tensorboard import SummaryWriter
import copy
import random
from collections import defaultdict

def train(data, tasks, args, ind, fold_num, step=50):
    '''
    data is a Data object
    tasks is a list of ((train, val, test), label) tuples
    '''
    if args.use_features:
        feat_str = 'feats'
    else:
        feat_str = 'no_feats'

    writer = SummaryWriter('./tensorboard_runs/'+args.expt_name+'/'
                           +args.network_type+'_'+args.dataset+'_'+feat_str)

    # retrieve the requested NN model + push to device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = neural_net.get_neural_network(args).to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = F.nll_loss
    best_f1 = 0
    model_save = copy.deepcopy(model.state_dict())

    epochs = args.epochs
    for epoch in range(epochs):
        model.train()
        loss_sum = 0
        f1_sum = 0

        # If MTL, iterate over all diseases, if not then just single disease
        for idx, ((train_mask, val_mask, _), y) in enumerate(tasks):
            train_mask, val_mask, y = train_mask.to(device), val_mask.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(data)
            weight = utils.get_weight(y, device=device)

            # if multi-task learning, apply the corresponding final linear layer
            if args.MTL:
                out = model.tasks[idx](out) # Last layer of NN that is specific to each task

            out = F.log_softmax(out, dim=1) # Softmax
            loss = criterion(out[train_mask], y[train_mask], weight=weight)
            res = utils.get_acc(model, data, val_mask, y, task=idx)

            if args.MTL:
                loss_sum += loss.item()
                f1_sum += res['f1']

            loss.backward()
            optimizer.step()

            # once per 'step' epoch tensorboard writing, at the 'disease' level
            if epoch % step == 0:
                writer.add_scalar('TrainLoss/disease_'+str(ind), loss.item(), fold_num * epochs + epoch)
                writer.add_scalar('ValF1/disease_'+str(ind), res['f1'], fold_num * epochs + epoch)
                writer.add_scalar('ValRecall/disease_' + str(ind), res['recall'], fold_num * epochs + epoch)
                print('disease ', idx,' loss on epoch', epoch, 'is', loss.item())

        # Every epoch, test if best model, then save
        if args.MTL:
            # Use the sum of F1's over all the diseases to select a paricular multi task model
            if f1_sum > best_f1:
                model_save = copy.deepcopy(model.state_dict())
                best_f1 = f1_sum

        else:
            # Model selection
            if res['f1'] > best_f1:
                model_save = copy.deepcopy(model.state_dict())
                best_f1 = res['f1']

        # Once per 'step' epoch tensorboard writing, at the epoch level
        if epoch % step == 0:
            if args.MTL:
                print('Overall MTL loss on epoch', epoch, 'is', loss_sum)
                writer.add_scalar('MTL/TrainLoss' + str(ind), loss_sum, fold_num * epochs + epoch)
            else:
                res = utils.get_acc(model, data, val_mask, y, task=None)

    writer.flush()
    writer.close()
    model.load_state_dict(model_save)
    return model, best_f1

def trainer(args, num_folds=10):
    edgelist_file = {
        'Decagon': '../dataset_collection/PP-Decagon_ppi.csv',
        'GNBR': '../dataset_collection/GNBR-edgelist.csv',
        'Decagon_GNBR': '../dataset_collection/Decagon_GNBR.csv'
    }[args.dataset]

    # load graph
    processed_data = ProcessData(edgelist_file, use_features=args.use_features)
    X = processed_data.X
    X = torch.tensor(X.values, dtype=torch.float)
    edges = processed_data.get_edges()
    edges = torch.tensor(edges.values, dtype=torch.long)
    data = utils.load_graph(X, edges)

    # load labels: returns all disease indices corresponding to given disease classes
    if args.sample_diseases: # for hyperparameter tuning
        sel_diseases = [469, 317, 473, 6, 426]
    else:
        sel_diseases = processed_data.get_disease_class_idx(['cancer'])
    processed_data.Y = processed_data.Y.iloc[:,sel_diseases]

    disease_test_scores = defaultdict(list)
    dir_ = './tensorboard_runs/'+args.expt_name
    
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
                disease_test_scores[ind].append(utils.get_acc(model, data, masks[f][2], label))

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

            test_score = utils.get_acc(model, data, masks[f], y, is_val=False)
            print('Best model f1:', test_score)
            disease_test_scores[ind] = [test_score]

    # Save results
    np.save(dir_+'/results', disease_test_scores)

if __name__ == '__main__':
    import argparse
    dt = str(datetime.now())[5:19].replace(' ', '_').replace(':', '-')
    
    parser = argparse.ArgumentParser(description='Define network type and dataset.')
    parser.add_argument('--network-type', type=str, choices=['GEO_GCN', 'SAGE', 'SAGE_GCN', 'GCN', 'GEO_GAT'], default='GEO_GCN')
    parser.add_argument('--dataset', type=str, choices=['Decagon', 'GNBR', 'Decagon_GNBR'], default='GNBR')
    parser.add_argument('--expt_name', type=str, default=dt)
    parser.add_argument('--use-features', type=bool, nargs='?', const=True, default=True)
    parser.add_argument('--MTL', type=bool, default=True)
    parser.add_argument('--in-dim', type=int, default=13)
    parser.add_argument('--hidden-dim', type=int, default=24)
    parser.add_argument('--out-dim', type=int, default=2)
    parser.add_argument('--num-heads', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--sample-diseases', type=bool, default=False)
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
    print('Experiment name is', args.expt_name)
    trainer(args)
