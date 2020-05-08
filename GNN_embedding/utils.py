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
from torch_geometric.data import Data
from sklearn.metrics import f1_score

def load_masks(y, folds=10, val_size=0.1):
    '''
    Split a list of labels in train/val/test masks via k-fold cross validation
    '''
    if torch.is_tensor(y): # for usage with sklearn
        y = y.cpu().numpy()
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=2)
    num_nodes = len(y)
    masks = []
    for train_val_idx, test_idx in kf.split(np.zeros(num_nodes), y): # dummy x since only y is necessary
        # further split train_val_idx, stratify according to its labels
        train_idx, val_idx = train_test_split(train_val_idx, test_size=val_size,
            stratify=y[train_val_idx], random_state=3)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_idx] = val_mask[val_idx] = test_mask[test_idx] = 1
        masks.append((train_mask, val_mask, test_mask))
    return masks

def insert_edges(edges):
    '''
    Remove self edges and repeats, and ensure graph is undirected
    Expects edges to be a tensor of shape [E, 2]
    '''
    edges = edges[edges[:, 0] != edges[:, 1]] # remove self edges
    reverse_edges = torch.flip(edges, [1])
    edges = torch.cat([edges, reverse_edges])
    edges = torch.unique(edges, dim=0) # Remove repeats
    edges = edges.type(torch.long)
    return edges

def load_graph(X, edges, y=None, edge_attr=None):
    '''
    Wrap data in a Data object for pyg
    Expects X to be a tensor of shape [N, feats]
    Expects edges to be a tensor of shape [E, 2]
    Expects y to be a tensor of shape [N], or None
    '''
    edges = insert_edges(edges)
    return Data(x=X, edge_index=edges.t().contiguous(), y=y,
                edge_attr=edge_attr)

# Evaluates the validation, test accuracy
def get_acc(model, data, mask, label, is_val=False, k=100, task=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model = model.to(device)
    data = data.to(device)
    mask,label = mask.to(device),label.to(device)
    with torch.no_grad():
        output = model(data)
        embedding = output.cpu().numpy()
        if task is not None:
            output = model.tasks[task](output)
        else:
            output = model.final(output)
        prob = output[:,1]
        pred = output.max(dim=1)[1]

        # Prints predicted class distribution
        #print(np.unique(pred.cpu(), return_counts=True)[1])
    
    probs = prob[mask].cpu().numpy()
    preds = pred[mask].cpu().numpy()
    label = label[mask].cpu().numpy()

    res = {}
    res['f1'] = f1_score(label,preds)
    correct = np.sum(label[np.argsort(probs)[-k:]])
    res['recall'] = correct/np.sum(label)
    return res, embedding

# Weighs the loss function
def get_weight(x_, device):
    a, b = np.unique(x_.cpu().numpy(), return_counts=True)[1]
    return torch.tensor([(1 - a / (a + b)), (1 - b / (a + b))],
                        device = device, dtype=torch.float32)
