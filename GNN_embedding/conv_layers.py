import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from torch_scatter import scatter
from models import encoders, decoders
from layers import layers

class gcn_layer(nn.Module):
    def __init__(self, in_features, out_features):
        super(gcn_layer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        hidden = self.linear(x)
        if adj.is_sparse:
            hidden = torch.sparse.mm(adj, hidden)
        else:
            hidden = torch.mm(adj, hidden)
        return hidden

class sage_layer(nn.Module):
    def __init__(self, in_features, out_features):
        super(sage_layer, self).__init__()
        self.linear = nn.Linear(in_features * 2, out_features)

    def forward(self, x, adj):
        '''sum the neighbors and then concat'''
        if adj.is_sparse:
            support = torch.sparse.mm(adj, x)
        else:
            support = torch.mm(adj, x)
        x = torch.cat((x, support), dim=1)
        hidden = self.linear(x)
        return hidden

class ada_a_conv_layer(nn.Module):
    def __init__(self, in_features, out_features):
        super(ada_a_conv_layer, self).__init__()
        self.lin1 = nn.Linear(in_features, out_features)
        self.lin2 = nn.Linear(in_features, out_features)

    def forward(self, x, adjs):
        adj1, adj2 = adjs
        hidden1 = self.lin1(x)
        hidden2 = self.lin2(x)
        if adj1.is_sparse:
            hidden1 = torch.sparse.mm(adj1, hidden1)
        else:
            hidden1 = torch.mm(adj1, hidden1)
        if adj2.is_sparse:
            hidden2 = torch.sparse.mm(adj2, hidden2)
        else:
            hidden2 = torch.mm(adj2, hidden2)
        return hidden1 + hidden2

class ada_b_conv_layer(nn.Module):
    def __init__(self, in_features, out_features):
        super(ada_b_conv_layer, self).__init__()
        self.lin1 = nn.Linear(in_features, out_features)
        self.lin2 = nn.Linear(in_features, out_features)
        self.c = nn.Parameter(torch.Tensor([.5]))

    def forward(self, x, adjs):
        adj1, adj2 = adjs
        hidden1 = self.lin1(x)
        hidden2 = self.lin2(x)
        if adj1.is_sparse:
            hidden1 = torch.sparse.mm(adj1, hidden1)
        else:
            hidden1 = torch.mm(adj1, hidden1)
        if adj2.is_sparse:
            hidden2 = torch.sparse.mm(adj2, hidden2)
        else:
            hidden2 = torch.mm(adj2, hidden2)
        return 2 * (self.c * hidden1 + (1 - self.c) * hidden2)

class ada_c_conv_layer(nn.Module):
    def __init__(self, in_features, out_features):
        super(ada_c_conv_layer, self).__init__()
        self.lin1 = nn.Linear(in_features, out_features)
        self.lin2 = nn.Linear(in_features, out_features)
        self.lin3 = nn.Linear(in_features, out_features)

    def forward(self, x, adjs):
        adj1, adj2, adj3 = adjs
        hidden1 = self.lin1(x)
        hidden2 = self.lin2(x)
        hidden3 = self.lin3(x)
        if adj1.is_sparse:
            hidden1 = torch.sparse.mm(adj1, hidden1)
        else:
            hidden1 = torch.mm(adj1, hidden1)
        if adj2.is_sparse:
            hidden2 = torch.sparse.mm(adj2, hidden2)
        else:
            hidden2 = torch.mm(adj2, hidden2)
        if adj3.is_sparse:
            hidden3 = torch.sparse.mm(adj3, hidden3)
        else:
            hidden3 = torch.mm(adj3, hidden3)
        return hidden1 + hidden2 + hidden3

class ada_d_conv_layer(nn.Module):
    def __init__(self, in_features, out_features):
        super(ada_d_conv_layer, self).__init__()
        self.lin1 = nn.Linear(in_features, 2 * out_features)
        self.lin2 = nn.Linear(in_features, 2 * out_features)

    def forward(self, x, adjs):
        adj1, adj2 = adjs
        dim = x.shape[1] // 2
        x1, x2 = x[:, :dim], x[:, dim:]
        hidden1 = self.lin1(x1)
        hidden2 = self.lin2(x2)
        if adj1.is_sparse:
            hidden1 = torch.sparse.mm(adj1, hidden1)
        else:
            hidden1 = torch.mm(adj1, hidden1)
        if adj2.is_sparse:
            hidden2 = torch.sparse.mm(adj2, hidden2)
        else:
            hidden2 = torch.mm(adj2, hidden2)
        return hidden1 + hidden2

class ada_e_conv_layer(nn.Module):
    def __init__(self, in_features, out_features):
        super(ada_e_conv_layer, self).__init__()
        self.lin = nn.Linear(2 * in_features, 2 * out_features)

    def forward(self, x, adjs):
        adj1, adj2 = adjs
        dim = x.shape[1] // 2
        x1, x2 = x[:, :dim], x[:, dim:]
        if adj1.is_sparse:
            hidden1 = torch.sparse.mm(adj1, x1)
        else:
            hidden1 = torch.mm(adj1, x1)
        if adj2.is_sparse:
            hidden2 = torch.sparse.mm(adj2, x2)
        else:
            hidden2 = torch.mm(adj2, x2)
        hidden = torch.cat((hidden1, hidden2), dim=1)
        hidden = self.lin(hidden)
        return hidden

class SAGEConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, aggr='mean',
            concat=True, transform_mess=False, nonlin=F.relu):
        '''
        Args:
            normalize: if true, perform L2 normalization on output
            aggr: method of aggregating neighborhood (not including self)
            concat
            transform_mess: if true, pass messages through linear function before propagating
        '''
        super(SAGEConv, self).__init__()
        self.aggr = aggr

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.normalize = normalize
        self.concat = concat
        self.transform_mess = transform_mess
        self.nonlin = nonlin

        if self.transform_mess:
            self.lin = nn.Linear(in_channels, in_channels) # actually decreases performance here (at least < 100 epochs)
        self.agg_lin = nn.Linear(in_channels * (2 if self.concat else 1), out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        '''
        x is of shape [N, in_channels], edge_index of shape [2, E]
        if not concat, incorporate self features by adding self loops
        '''
        if not self.concat:
            edge_index, edge_weight = pyg_utils.add_remaining_self_loops(
                edge_index, edge_weight=edge_weight, fill_value=1, num_nodes=x.size(0))

        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        '''
        x_j is of shape [E, in_channels], denoting the features of the source node
        of each edge
        '''
        msg = x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
        if self.transform_mess:
            msg = self.nonlin(self.lin(msg))
        return msg
    
    def aggregate(self, inputs, index, dim_size=None):
        '''
        Inputs is of shape [N, in_channels], index is of shape [N], where the elements of index
        denote which index of the output vector the corresponding element of inputs shall be aggregated in
        '''
        if self.aggr in ['sum', 'mean', 'min', 'max']:
            return scatter(inputs, index, dim=0, dim_size=dim_size, reduce=self.aggr)
        raise NotImplementedError('No such aggr for ' + self.aggr + '.')

    def update(self, aggr_out, x):
        '''
        aggr_out, x initially both have shape [N, in_channels]
        then aggr_out has shape [N, out_channels]
        '''
        if self.concat:
            aggr_out = torch.cat([x, aggr_out], dim=1)
        aggr_out = self.nonlin(self.agg_lin(aggr_out))
        if self.normalize:
            aggr_out = F.normalize(aggr_out)
        return aggr_out

class HGCNConv(nn.Module):
    def __init__(self, args):
        super(HGCNConv, self).__init__()

        args.act='relu'
        args.dim = args.hidden_dim
        args.manifold='Hyperboloid'
        args.model='HGCN'
        args.num_layers=2
        args.n_nodes=19678 # for GNBR
        args.n_classes= args.out_dim
        args.c=None
        args.feat_dim = args.in_dim
        args.cuda=0
        args.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        args.task='nc'
        args.dropout=0.2
        args.bias=1
        
        self.device = args.device
        self.manifold_name = args.manifold
        self.manifold = getattr(manifolds, args.manifold)()
        self.c = nn.Parameter(torch.Tensor([1.])) # curvature
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)
        self.decoder = decoders.model2decoder[args.model](self.c, args)
        self.adj_mat = None
    
    def convert_to_adj(self, edge_index, num_nodes):
        '''we want [2, E] -> [N, N]'''
        edge_index = edge_index.to(self.device)
        vals = torch.ones(edge_index.shape[1], device=self.device)
        adj_mat = torch.sparse.FloatTensor(edge_index, vals, (num_nodes, num_nodes)).to(self.device)
        # adj_mat = torch.zeros(num_nodes, num_nodes)
        # for edge in range(edge_index.shape[1]):
            # assert adj_mat[edge_index[0, edge], edge_index[1, edge]] == 0 # no repeated edges
            # adj_mat[edge_index[0, edge], edge_index[1, edge]] = 1
        self.adj_mat = adj_mat
        # assert torch.all(adj_mat.eq(adj_mat.t())) # symmetric because undirected
    
    def forward(self, data):
        '''
        data.x shape: [N, feats]
        data.edge_index shape: [2, E]
        '''
        x = data.x
        if data.num_node_features == 0:
            x = torch.ones(data.num_nodes, 1)
        x = x.to(self.device)

        if self.adj_mat is None:
            self.convert_to_adj(data.edge_index, len(x)) # must transform this
        
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x) # north pole!
            x = torch.cat([o[:, 0:1], x], dim=1)

        h = self.encoder.encode(x, self.adj_mat)
        output = self.decoder.decode(h, self.adj_mat)
        return F.log_softmax(output, dim=1)
