#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Nov 7 21:27:10 2019

@author: Kendrick, Yusuf

util functions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

import torch
from torch_geometric.data import Data, DataLoader

def load_pyg(X, edges, y, folds=5, test_size=50):

    # First identify the test set
    np.random.seed(20) # to make sure the test set is always the same
    ones = np.random.choice(np.where(y.cpu().numpy() == 1)[0], size=test_size, replace=False)
    np.random.seed(30)
    zeros = np.random.choice(np.where(y.cpu().numpy() == 0)[0], size=test_size, replace=False)
    test_idx = np.concatenate([ones, zeros])

    # Now use the remainder of the data to create train and val sets
    kf = StratifiedKFold(n_splits=folds, random_state=2)
    for train_idx, val_idx in kf.split(X, y):
        data = Data(x=X, edge_index=edges.t().contiguous(), y=y)

        train_mask = [int(i in train_idx and i not in test_idx) for i in range(len(X))]
        val_mask = [int(i in val_idx and i not in test_idx) for i in range(len(y))]
        test_mask = [int(i in test_idx) for i in range(len(X))]

        data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
        data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

        loader = DataLoader([data], batch_size=32, shuffle=True)
        yield loader

