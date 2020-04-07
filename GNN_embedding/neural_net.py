import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import conv_layers
from layers import layers

def get_neural_network(args):
    if args.network_type == 'GCNConv': # slowly add more of these
        model = RexGCNConv(args.in_dim, args.hidden_dim, args.out_dim)
    else:
        model = GNN(args.in_dim, args.hidden_dim, args.out_dim, args.network_type)
    return model

class GNN(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=1, out_dim=1, num_layers=2,
            dropout=0.2, model_type='GCNConv'):
        super(GNN, self).__init__()

        self.model_type = model_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = 3
        mult = self.num_heads if self.model_type == 'GATConv' else 1

        self.layers = nn.ModuleList()
        self.layers.append(self.build_model(in_dim, hidden_dim))
        for l in range(self.num_layers - 1):
            self.layers.append(self.build_model(hidden_dim * mult, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim * mult, hidden_dim), nn.Dropout(self.dropout), 
            nn.Linear(hidden_dim, out_dim))
    
    def build_model(self, in_dim, out_dim):
        if self.model_type == 'SAGEConvMean':
            return conv_layers.SAGEConv(in_dim, out_dim, aggr='mean')
        elif self.model_type == 'SAGEConvMin':
            return conv_layers.SAGEConv(in_dim, out_dim, aggr='min')
        elif self.model_type == 'SAGEConvMax':
            return conv_layers.SAGEConv(in_dim, out_dim, aggr='max')
        elif self.model_type == 'GCNConv':
            return pyg_nn.GCNConv(in_dim, out_dim)
        elif self.model_type == 'GATConv':
            return pyg_nn.GATConv(in_dim, out_dim, heads=self.num_heads)
        else:
            raise Exception('Unsupported model type')

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

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


class RexGCNConv(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.2):
        super(RexGCNConv, self).__init__()

        self.act = F.relu
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.adj_mat = None
        '''All the convolutional layers!'''
        modules = []
        modules.append(layers.GraphConvolution(in_dim, hidden_dim, self.dropout, self.act, True))
        for l in range(self.num_layers - 1):
            modules.append(layers.GraphConvolution(hidden_dim, hidden_dim, self.dropout, self.act, True))
        self.conv = nn.Sequential(*modules)
        '''Post mp layers'''
        self.post_mp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.Dropout(self.dropout), 
            nn.Linear(self.hidden_dim, self.out_dim))
    
    def convert_to_adj(self, edge_index, num_nodes):
        '''we want [2, E] -> [N, N]'''
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        edge_index = edge_index.to(device) # coordinates to put values
        vals = torch.ones(edge_index.shape[1], device=device) # values to put
        adj_mat = torch.sparse.FloatTensor(edge_index, vals, (num_nodes, num_nodes)).to(device)
        self.adj_mat = adj_mat
    
    def forward(self, data):
        '''
        data.x shape: [N, feats]
        data.edge_index shape: [2, E]
        '''
        x = data.x
        if self.adj_mat is None:
            self.convert_to_adj(data.edge_index, len(x))
        
        x, _ = self.conv((x, self.adj_mat))
        x = F.normalize(x, p=2, dim=1) # should we do this?
        x = self.post_mp(x)
        return F.log_softmax(x, dim=1)