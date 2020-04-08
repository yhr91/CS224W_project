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
import pandas as pd
import conv_layers
import optimizers

def train(loaders, args, ind, it):
    if args.MTL==False:
        loaders = [loaders]

    if args.use_features:
        feat_str = 'feats'
    else:
        feat_str = 'no_feats'

    writer = SummaryWriter('./tensorboard_runs/'+args.expt_name+'/'
                           +args.network_type+'_'+args.dataset+'_'+feat_str)

    # retrieve the requested NN model + push to device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = neural_net.get_neural_network(args)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = F.nll_loss
    best_f1 = 0
    model_save = copy.deepcopy(model.state_dict())

    epochs = args.epochs
    for epoch in range(epochs):
        model.train()
        it = 0
        loss_sum = 0
        f1_sum = 0

        # If MLT, iterate over all diseases, if not then just single disease
        for task_i,loader in enumerate(loaders):

            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                weight = utils.get_weight(batch.y, device=device)

                # Multi-task learning
                if args.MTL:
                    out = model.tasks[it](out)
                    loss = criterion(out[batch.train_mask], batch.y[batch.train_mask], weight=weight)
                    loss_sum += loss.item()

                # Single task learning
                else:
                    loss = criterion(out[batch.train_mask], batch.y[batch.train_mask], weight=weight)
                loss.backward()
                optimizer.step()
                print('disease ', task_i,' loss on epoch', epoch, 'is', loss.item())

                # Tensorboard writing and evaluate validation f1
                if epoch % 50 == 0:
                    if args.MTL:
                        res = utils.get_acc(model, loader, is_val=True, task=task_i)
                        f1_sum += res['f1']
                    else:
                        res = utils.get_acc(model, loader, is_val=True, task=None)
                    writer.add_scalar('TrainLoss/disease_'+str(ind), loss.item(), it*epochs+epoch)
                    writer.add_scalar('ValF1/disease_'+str(ind), res['f1'], it*epochs+epoch)
                    writer.add_scalar('ValRecall/disease_' + str(ind), res['recall'], it*epochs+epoch)

                if args.MTL==False:
                    # Model selection
                    if res['f1'] > best_f1:
                        model_save = copy.deepcopy(model.state_dict())
                        best_f1 = res['f1']

                    if not torch.cuda.is_available():
                        # if training locally, print out progress, otherwise remove all I/O
                        # to speed up training
                        if epoch % 100 == 0:
                            print('loss on epoch', epoch, 'is', loss.item())
                            writer.flush()

        # Write the MTL loss to tensorboard
        if args.MTL:
            print('Overall MTL loss on epoch', epoch, 'is', loss_sum)
            writer.add_scalar('MTL/TrainLoss' + str(ind), loss_sum, it * epochs + epoch)

            # Use the sum of F1's over all the diseases to select a paricular multi task model
            if f1_sum > best_f1:
                model_save = copy.deepcopy(model.state_dict())
                best_f1 = f1_sum

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
    dir_ = './tensorboard_runs/'+args.expt_name

    # This returns all disease indices corresponding to given disease classes
    sel_diseases = processed_data.get_disease_class_idx(['cancer'])[:2]
    args.tasks = len(sel_diseases)
    processed_data.Y = processed_data.Y.iloc[:,sel_diseases]

    if args.MTL:

        edges = processed_data.get_edges()
        edges = torch.tensor(edges.values, dtype=torch.long)
        data_generators = []

        # Create a separate data generator for each task
        for ind, column in enumerate(processed_data.Y):
            print(ind, column, 'out of', len(processed_data.Y))

            y = processed_data.Y[column].tolist()
            y = torch.tensor(np.array(y).astype('int'), dtype=torch.long)

            # Set up train and test sets:
            test_size = .1
            data_generators.append(utils.load_pyg(X, edges, y,
                                            folds=num_folds, test_size=test_size))

        # Iterate over folds
        models = []  # save models for now
        model_scores = []  # save model recalls
        for f in range(num_folds):
            loaders = []
            # Iterate over diseases
            for it, loader_all_folds in enumerate(data_generators):
                loaders.append(loader_all_folds[f])  #Pick the same fold for each disease

            # Use the list of training datasets for all diseases at a specifc fold to train
            model, score = train(loaders, args, ind, it)
            models.append(model)
            model_scores.append(score)

        best_model = models[np.argmax(model_scores)]

        # retrieve the state dict of the best-performing model, load into new model
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = neural_net.get_neural_network(args)
        model = model.to(device)
        model.load_state_dict(best_model)

        best_test_scores = []
        for it, loader_all_folds in enumerate(data_generators):
            disease_test_scores[it] = utils.get_acc(model, loader_all_folds[0], is_val=False, task=it)

        # Save results
        np.save(dir_ + '/results', disease_test_scores)


    else:
        for ind, column in enumerate(processed_data.Y):
            print(ind,column,'out of',len(processed_data.Y))

            y = processed_data.Y[column].tolist()
            edges = processed_data.get_edges()

            y = torch.tensor(np.array(y).astype('int'), dtype=torch.long)
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

            # retrieve the state dict of the best-performing model, load into new model
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            model = neural_net.get_neural_network(args)
            model = model.to(device)
            model.load_state_dict(best_model)

            best_test_score = utils.get_acc(model, loader, is_val=False)
            print('Best model f1:')
            print(best_test_score)
            disease_test_scores[ind] = [best_test_score]

            # Save results
            np.save(dir_+'/results', disease_test_scores)

if __name__ == '__main__':
    import argparse
    dt = str(datetime.now())[5:19].replace(' ', '_').replace(':', '-')
    
    parser = argparse.ArgumentParser(description='Define network type and dataset.')
    parser.add_argument('--network-type', type=str, choices=['GCNConv', 'SAGEConvMean', 'SAGEConvMin', 'SAGEConvMax', 'HGCNConv', 'GATConv'], default='GCNConv')
    parser.add_argument('--dataset', type=str, choices=['Decagon', 'GNBR', 'Decagon_GNBR'], default='GNBR')
    parser.add_argument('--expt_name', type=str, default=dt)
    parser.add_argument('--use-features', type=bool, nargs='?', const=True, default=False)
    parser.add_argument('--MTL', type=bool, default=True)
    parser.add_argument('--in-dim', type=int, default=11)
    parser.add_argument('--hidden-dim', type=int, default=24)
    parser.add_argument('--out-dim', type=int, default=2)
    parser.add_argument('--num-heads', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0001)
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
