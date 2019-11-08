#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Nov 7 11:24:10 2019

@author: Kendrick

"""

import torch
from neural_net import GNN
# import graph_embedding
import torch.nn as nn
import pandas as pd
import numpy as np
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

def load_pyg(x, edge_index, y):
    X = pd.read_csv(x)
    X.drop(columns=X.columns[0], inplace=True)
    X = torch.tensor(X.values, dtype=torch.float)
    
    edges = pd.read_csv(edge_index)
    edges.drop(columns=edges.columns[0], inplace=True)
    edges = torch.tensor(edges.values, dtype=torch.long)
    
    Y = np.loadtxt(y)
    Y = torch.tensor(Y, dtype=torch.long)
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

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# def prepare(data, params):
#     X, y, graph = graph_embedding.load_data()
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=True)
#     model = GNN()
#     # model = model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
#     criterion = nn.CrossEntropyLoss()

def train():
    data = load_pyg('feature_matrix.csv', 'edgelist.csv', 'y.txt')
    test_loader = loader = DataLoader([data], batch_size=32, shuffle=True)
    model = GNN(3, 32, 1, 'GCNConv')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(20):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()
            print(loss.item())

    model.eval()
    correct = 0
    for d in test_loader:
        with torch.no_grad():
            pred = model(d).max(dim=1)[1][d.test_mask]
            label = d.y[d.test_mask]
        correct += pred.eq(label).sum().item()
    total = 0
    for d in test_loader.dataset:
        total += torch.sum(d.test_mask).item()
    print('Accuracy is', correct / total)

if __name__ == '__main__':
    train()