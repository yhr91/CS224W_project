#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Nov 7 11:24:10 2019

@author: Kendrick

"""

from neural_net import GNN
import graph_embedding
import torch.optim as optim
import torch.nn as nn

class Hyperparams:
    self.num_layers = 3
    self.batch_size = 32
    self.hidden_dim = 32
    self.dropout = .2
    self.epochs = 100
    self.opt = 'adam'
    self.lr = .01
    self.model_type = 'GCNConv'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def prepare(data, params):
    X, y, graph = graph_embedding.load_data()
    model = GNN()
    model = model.to(device)
    if params.opt == 'adam':
        optimizer = optim.Adam(lr=params.lr)
    criterion = nn.CrossEntropyLoss()

def train(X, Y, model, criterion, opt, params):
    for epoch in range(params.epochs):
        for sample in X:
            sample = sample.to(device)
            logits = model(sample)
            loss = criterion(logits, Y)
            opt.zero_grad()
            loss.backward()
            opt.step()
