#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Nov 7 21:27:10 2019

@author: Kendrick

util functions
"""
import pandas as pd
import numpy as np

import torch
from torch_geometric.data import Data, DataLoader

def load_pyg(x, edge_index, y):
    X = pd.read_csv(x)
    X.drop(columns=X.columns[0], inplace=True)
    X = torch.tensor(X.values, dtype=torch.float)
    
    edges = pd.read_csv(edge_index)
    edges.drop(columns=edges.columns[0], inplace=True)
    edges = torch.tensor(edges.values, dtype=torch.long)
    
    Y = np.loadtxt(y)
    ones = np.where(Y == 1)[0]
    zeros = np.where(Y == 0)[0]
    indices_ones = np.random.choice(ones, size=50)
    indices_zeros = np.random.choice(zeros, size=50)
    test_mask = [1 if i in indices_ones or i in indices_zeros else 0 for i in range(len(Y))]
    train_mask = [1 if i not in indices_ones and i not in indices_zeros else 0 for i in range(len(Y))]
    
    data = Data(x=X, edge_index=edges.t().contiguous(), y=Y)
    data.train_mask = torch.tensor(train_mask, dtype=torch.long)
    data.test_mask = torch.tensor(test_mask, dtype=torch.long)
    return data