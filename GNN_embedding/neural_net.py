import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import conv_layers

def get_neural_network(args):
    model = {
        'GEO_GCN': geo_gcn,
        'SAGE': sage,
        'SAGE_GCN': sage_gcn,
        'GCN': gcn,
        'GEO_GAT': geo_gat,
    }[args.network_type]
    use_adj = 'GEO' not in args.network_type
    model = model(args.in_dim, args.hidden_dim, args.out_dim, use_adj=use_adj,
        tasks=args.tasks)
    return model

class Neural_Base(nn.Module):
    '''Abstract class for general neural networking architecture'''
    def __init__(self, in_dim=1, hidden_dim=1, out_dim=1,
            num_layers=2, dropout=0.2, use_adj=True,
            tasks=1, num_heads=1):
        super(Neural_Base, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.use_adj = use_adj
        self.num_heads = num_heads

        self.layers = nn.ModuleList()
        self.layers.append(self.get_conv(in_dim, hidden_dim))
        for l in range(self.num_layers - 1):
            self.layers.append(self.get_conv(hidden_dim * self.num_heads, hidden_dim))

        # post-message-passing
        if tasks>1: # to handle Multi task learning
            self.post_mp = nn.Sequential(
                nn.Linear(hidden_dim * self.num_heads, hidden_dim), nn.Dropout(self.dropout))
            self.tasks = nn.ModuleList()
            for i in range(tasks):
                self.tasks.append(nn.Linear(hidden_dim, out_dim))
        else:
            self.post_mp = nn.Sequential(
                nn.Linear(hidden_dim * self.num_heads, hidden_dim), nn.Dropout(self.dropout),
                nn.Linear(hidden_dim, out_dim))

    def get_conv(self, in_dim, hidden_dim):
        raise NotImplementedError

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if self.use_adj and not hasattr(self, 'adj_mat'):
            self.convert_to_adj(edge_index, len(x))

        arg = self.adj_mat if self.use_adj else edge_index
        for i in range(self.num_layers):
            x = self.layers[i](x, arg)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.post_mp(x)
        return x

class geo_gcn(Neural_Base):
    def get_conv(self, in_dim, out_dim):
        return pyg_nn.GCNConv(in_dim, out_dim)

class geo_gat(Neural_Base):
    def get_conv(self, in_dim, out_dim):
        return pyg_nn.GATConv(in_dim, out_dim, heads=self.num_heads)

class sage(Neural_Base):
    def get_conv(self, in_dim, out_dim):
        return conv_layers.sage_layer(in_dim, out_dim)

    def convert_to_adj(self, edge_index, num_nodes):
        '''we want [2, E] -> [N, N]'''
        # push to device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        edge_index = edge_index.to(device) # coordinates to put values
    
        # divide adjacency matrix by node degree to obtain the mean
        # according to the GraphSAGE paper: https://arxiv.org/abs/1706.02216
        vals = torch.ones(edge_index.shape[1], device=device)
        degs = scatter(vals, edge_index[0], dim=0, dim_size=num_nodes, reduce='sum')
        vals = vals / degs[edge_index[0]] # divide by node degree

        adj_mat = torch.sparse.FloatTensor(edge_index, vals, (num_nodes, num_nodes)).to(device)
        self.adj_mat = adj_mat

class sage_gcn(Neural_Base):
    def get_conv(self, in_dim, out_dim):
        return conv_layers.gcn_layer(in_dim, out_dim)

    def convert_to_adj(self, edge_index, num_nodes):
        '''we want [2, E] -> [N, N]'''
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # add self loops to edge_index
        self_loops = torch.stack((torch.arange(num_nodes), torch.arange(num_nodes)))
        edge_index = torch.cat((edge_index, self_loops), dim=1)
        edge_index = edge_index.to(device)

        # divide adjacency matrix by node degree to obtain the mean
        # according to the GraphSAGE paper: https://arxiv.org/abs/1706.02216
        vals = torch.ones(edge_index.shape[1], device=device)
        degs = scatter(vals, edge_index[0], dim=0, dim_size=num_nodes, reduce='sum')
        vals = vals / degs[edge_index[0]] # divide by node degree

        adj_mat = torch.sparse.FloatTensor(edge_index, vals, (num_nodes, num_nodes)).to(device)
        self.adj_mat = adj_mat

class gcn(Neural_Base):
    def get_conv(self, in_dim, out_dim):
        return conv_layers.gcn_layer(in_dim, out_dim)

    def convert_to_adj(self, edge_index, num_nodes):
        '''we want [2, E] -> [N, N]'''
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # add self loops to edge_index
        self_loops = torch.stack((torch.arange(num_nodes), torch.arange(num_nodes)))
        edge_index = torch.cat((edge_index, self_loops), dim=1)
        edge_index = edge_index.to(device)

        # scaling according to original GCN paper: arxiv.org/abs/1609.02907
        vals = torch.ones(edge_index.shape[1], device=device)
        degs = scatter(vals, edge_index[0], dim=0, dim_size=num_nodes, reduce='sum')
        degs_inv = degs.pow(-0.5)
        degs_inv[degs_inv == float('inf')] = 0
        vals = degs_inv[edge_index[0]] * vals * degs_inv[edge_index[1]]

        adj_mat = torch.sparse.FloatTensor(edge_index, vals, (num_nodes, num_nodes)).to(device)
        self.adj_mat = adj_mat