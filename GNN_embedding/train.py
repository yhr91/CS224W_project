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
from torch_geometric.data import DataLoader

def train(loaders, args, ind, edges=None):
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
            if edges is not None:
                loader.edge_index = edges.t().contiguous()
                loader = DataLoader([loader], batch_size=32)

            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                weight = utils.get_weight(batch.y, device=device)

                # Multi-task learning
                if args.MTL:
                    out = model.tasks[it](out)   # Last layer of NN that is specific to each task
                    out = F.log_softmax(out, dim=1)  # Softmax
                    loss = criterion(out[batch.train_mask], batch.y[batch.train_mask], weight=weight)
                    loss_sum += loss.item()

                # Single task learning
                else:
                    out = F.log_softmax(out, dim=1)
                    loss = criterion(out[batch.train_mask], batch.y[batch.train_mask], weight=weight)
                loss.backward()
                optimizer.step()
                if epoch % 50 == 0:
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
            if epoch % 50 == 0:
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
    if args.sample_diseases:
        sel_diseases = [469, 317, 473, 6, 426]
    else:
        sel_diseases = processed_data.get_disease_class_idx(['cancer'])
    args.tasks = len(sel_diseases)
    processed_data.Y = processed_data.Y.iloc[:,sel_diseases]

    if args.MTL:

        # Preparing edges separately from the rest of the data
        edges = processed_data.get_edges()
        edges = torch.tensor(edges.values, dtype=torch.long)
        edges = utils.load_edges(edges)

        # Create a separate data generator for each task
        data_generators = []
        for ind, column in enumerate(processed_data.Y):
            print(ind, column, 'out of', len(processed_data.Y))

            y = processed_data.Y[column].tolist()
            y = torch.tensor(np.array(y).astype('int'), dtype=torch.long)

            # Set up train and test sets:
            test_size = .1
            data_generators.append(utils.load_pyg(X=X, edges=None, y=y,
                                            folds=num_folds, test_size=test_size))
            # Note: edges are being loaded separately to save memory

        # Iterate over folds
        models = []  # save models for now
        model_scores = []  # save model recalls
        for f in range(num_folds):
            loaders = []
            # Iterate over diseases
            for it, loader_all_folds in enumerate(data_generators):
                loaders.append(next(loader_all_folds))       #Pick the same fold for each disease

            # Use the list of training datasets for all diseases at a specifc fold to train
            model, score = train(loaders, args, ind, edges=edges)
            models.append(model)
            model_scores.append(score)

        best_model = models[np.argmax(model_scores)]

        # retrieve the state dict of the best-performing model, load into new model
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = neural_net.get_neural_network(args)
        model = model.to(device)
        model.load_state_dict(best_model)

        for it, loader_all_folds in enumerate(data_generators):
            disease_test_scores[it] = utils.get_acc(model, loaders[it],
                                                    is_val=False, task=it)

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
    parser.add_argument('--use-features', type=bool, nargs='?', const=True, default=True)
    parser.add_argument('--MTL', type=bool, default=True)
    parser.add_argument('--in-dim', type=int, default=11)
    parser.add_argument('--hidden-dim', type=int, default=24)
    parser.add_argument('--out-dim', type=int, default=2)
    parser.add_argument('--num-heads', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--sample-diseases', type=bool, default=True)
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
