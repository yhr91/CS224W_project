#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 6 21:31:42 2019

@author: Kendrick

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

class GNN(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=1, out_dim=1, model_type='GCNConv'):
        super(GNN, self).__init__()

        self.model_type = model_type
        self.num_layers = 2
        self.dropout = .2

        self.layers = nn.ModuleList()
        self.layers.append(self.build_model(in_dim, hidden_dim))
        for l in range(self.num_layers - 1):
            self.layers.append(self.build_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(self.dropout), 
            nn.Linear(hidden_dim, out_dim))
    
    def build_model(self, in_dim, out_dim):
        if self.model_type == 'SAGEConv':
            return pyg_nn.SAGEConv(in_dim, out_dim)
        elif self.model_type == 'GCNConv':
            return pyg_nn.GCNConv(in_dim, out_dim)
        elif self.model_type == 'GATConv':
            return pyg_nn.GATConv(in_dim, out_dim, heads=3)
        else:
            raise Exception('Unsupported model type')

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if data.num_node_features == 0:
            x = torch.ones(data.num_nodes, 1)

        # for some reason, the layers were in CPU, so have to convert every time
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        for i in range(self.num_layers):
            self.layers[i] = self.layers[i].to(device)
            x = self.layers[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        self.post_mp = self.post_mp.to(device)
        x = self.post_mp(x)
        return F.log_softmax(x, dim=1)


class NN(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=1, out_dim=1):
        super(Perceptron, self).__init__()
        self.num_layers = 2
        self.dropout = .2

        self.layers = nn.ModuleList()
        for l in range(self.num_layers - 1):
            self.layers.append(self.build_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, out_dim))

    def forward(self, data):
        x, batch = data.x, data.batch

        # for some reason, the layers were in CPU, so have to convert every time
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        for i in range(self.num_layers):
            self.layers[i] = self.layers[i].to(device)
            x = self.layers[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        self.post_mp = self.post_mp.to(device)
        x = self.post_mp(x)
        return F.log_softmax(x, dim=1)