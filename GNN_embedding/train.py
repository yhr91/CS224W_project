#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 23:25:36 2019

@author: Kendrick, Yusuf

"""

import torch
import sys
import numpy as np
sys.path.append('../')
from neural_net import GNN
from dataloader import load_pyg
import torch.nn.functional as F
from torch_geometric.data import DataLoader

def get_acc(model, loader):
  correct = 0
  total = 0
  for d in loader:
          with torch.no_grad():
              pred = model(d).max(dim=1)[1][d.test_mask]
              label = d.y[d.test_mask]

              print(np.unique(label,return_counts=True))
              print(np.unique(pred,return_counts=True))
          correct += pred.eq(label).sum().item()
          total += torch.sum(d.test_mask).item()
  return correct/total

def get_weight(x_):
  a,b = np.unique(x_, return_counts=True)[1]
  return torch.tensor([(1-a/(a+b)), (1-b/(a+b))])

def train():
    x = 'https://github.com/yhr91/CS224W_project/blob/master/Data/ForAnalysis/X/TCGA_GTEX_GeneExpression.csv?raw=true'
    y = 'https://github.com/yhr91/CS224W_project/raw/master/Data/ForAnalysis/Y/NCG_cancergenes_list.txt'
    edgelist_file = 'https://github.com/yhr91/CS224W_project/blob/master/Data/PP-Decagon_ppi.csv?raw=true'
    data = load_pyg(x, edgelist_file, y)

    test_loader = loader = DataLoader([data], batch_size=32, shuffle=True)
    model = GNN(3, 32, 2, 'GCNConv')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    weight = get_weight(data.y)
    test_acc = []

    print('Before training')
    print(get_acc(model, test_loader))
    for epoch in range(15):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = F.nll_loss(out[batch.train_mask], 
                              batch.y[batch.train_mask], weight=weight)
            loss.backward()
            optimizer.step()
            print(loss.item())
            
            if epoch % 5 == 0:
                test_acc.append(get_acc(model, test_loader))
                print(test_acc[-1])
          

    model.eval()
    acc = get_acc(model, test_loader)
    print('Final Accuracy is', acc)

    
if __name__ == '__main__':
    train()