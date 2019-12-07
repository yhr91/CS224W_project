#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Nov 7 21:27:10 2019

@author: Kendrick, Yusuf

util functions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

import torch
from torch_geometric.data import Data, DataLoader

def load_pyg(X, edges, y, folds=5):
    kf = KFold(n_splits=folds, random_state=2)
    for train_idx, test_idx in kf.split(X):
        data = Data(x=X, edge_index=edges.t().contiguous(), y=y)
        train_mask = [int(i in train_idx) for i in range(len(X))]
        test_mask = [int(i in test_idx) for i in range(len(y))]
        data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
        loader = DataLoader([data], batch_size=32, shuffle=True)
        yield loader

    # # Create masks
    # np.random.seed(20) # to make sure the test set is always the same
    # ones = np.random.choice(np.where(y.cpu() == 1)[0], size=size, replace=False)
    # np.random.seed(30)
    # zeros = np.random.choice(np.where(y.cpu() == 0)[0], size=size, replace=False)

    # test_mask = [True if i in ones or i in zeros 
    #             else False for i in range(len(y))]
    # train_mask = [True if i not in ones and i not in zeros 
    #             else False for i in range(len(y))]

    # # Return data loader
    # data = Data(x=X, edge_index=edges.t().contiguous(), y=y)
    # data.train_mask = torch.tensor(train_mask, dtype=torch.bool, device = device)
    # data.test_mask = torch.tensor(test_mask, dtype=torch.bool, device = device)

    # return data

## Given a graph and training mask, obtain 5 folds for train/val splits
def make_cross_val_sets(data, n=5):
    
    idx_0 = np.where(data.y.cpu()[data.train_mask]==0)[0]
    np.random.seed(40)
    np.random.shuffle(idx_0)
    intervals_0 = np.linspace(0,len(idx_0),n+1).astype(int)
    
    idx_1 = np.where(data.y.cpu()[data.train_mask]==1)[0]
    np.random.seed(50)
    np.random.shuffle(idx_1)
    intervals_1 = np.linspace(0,len(idx_1),n+1).astype(int)
    
    masks = []
    prev = 0
    for i in range(1,n+1):
        s = np.array([False]*len(data.y))
        s[idx_0[intervals_0[i-1]:intervals_0[i]]] = True
        s[idx_1[intervals_1[i-1]:intervals_1[i]]] = True
        masks.append(s)
    
    return masks
