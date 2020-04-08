#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Nov 7 21:27:10 2019

@author: Kendrick, Yusuf

util functions
"""


import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import f1_score

def load_pyg_new(X, edges, y, folds=10, val_size=0.1):
    '''
    First, remove self-edges and ensure graph is undirected.
    Then, load data according to k-fold cross validation. First do a
    stratified split on y, creating the k hold-out test sets. Then,
    do a stratified split on the remaining 90% of the data, creating
    train and validation sets.
    Yields:
        a DataLoader for each fold
    '''
    edges = [e for e in edges if e[0] != e[1]] # Remove self edges
    reverse_edges = torch.flip(edges, 1)
    edges = torch.cat([edges, reverse_edges])
    edges = torch.unique(edges, dim=0) # Remove repeats
    edges = edges.type(torch.long)

    # each fold is simply going to change its train/val/test idx
    data = Data(x=X, edge_index=edges.t().contiguous(), y=y)

    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=2)
    for train_val_idx, test_idx in kf.split(X, y):
        # further split train_val_idx, stratify according to its labels
        train_idx, val_idx = train_test_split(train_val_idx, test_size=val_size,
            stratify=y[train_val_idx], random_state=3)

        train_mask = torch.zeros(len(X), dtype=torch.bool)
        val_mask = torch.zeros(len(X), dtype=torch.bool)
        test_mask = torch.zeros(len(X), dtype=torch.bool)

        train_mask[train_idx] = val_mask[val_idx] = test_mask[test_idx] = 1
        data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask
        loader = DataLoader([data], batch_size=32) # shuffling done at train time (is this true? were not batching)
        yield loader

# This version of load pyg is designed to work with recall@100 metric
# It does not require an even class split in the test set
def load_pyg(X, edges, y, folds=5, test_size=0.1):

    # First identify the test set
    indices = np.arange(len(X))
    _,test_idx = train_test_split(indices, test_size=test_size,
                                             stratify=y, random_state=40)

    # Now use the remainder of the data to create train and val sets
    kf = StratifiedKFold(n_splits=folds, random_state=2)

    # Consider all edges and their reverse
    edges = [e for e in edges.numpy() if e[0] != e[1]] # Remove self edges
    reverse_edges = np.flip(np.array(edges),1)
    edges = np.concatenate([edges,reverse_edges])
    edges = np.unique(edges, axis=0) # Remove repeats
    edges = torch.tensor(edges, dtype=torch.long)

    for train_idx, val_idx in kf.split(X, y):
        data = Data(x=X, edge_index=edges.t().contiguous(), y=y)

        train_mask = [int(i in train_idx and i not in test_idx) for i in range(len(X))]
        val_mask = [int(i in val_idx and i not in test_idx) for i in range(len(y))]
        test_mask = [int(i in test_idx) for i in range(len(X))]

        data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
        data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

        loader = DataLoader([data], batch_size=32) # shuffling done at train time
        yield loader


# Evaluates the validation, test accuracy
def get_acc(model, loader, is_val=False, k=100):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    probs = []
    model.eval()
    model = model.to(device)
    preds, trues = [], []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
            prob = output.cpu().numpy()[:,1]
            pred = output.max(dim=1)[1]
            label = data.y

            # Prints predicted class distribution
            #print(np.unique(pred.cpu(), return_counts=True)[1])

    if is_val:
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


# Weighs the loss function
def get_weight(x_, device):
    a, b = np.unique(x_.cpu().numpy(), return_counts=True)[1]
    return torch.tensor([(1 - a / (a + b)), (1 - b / (a + b))],
                        device = device)
