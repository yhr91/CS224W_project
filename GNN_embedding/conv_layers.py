import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from torch_scatter import scatter

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
        super(SAGEConv, self).__init__(aggr=aggr)

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
    
    def aggregate(self, inputs, index):
        '''
        Inputs is of shape [N, in_channels], index is of shape [N], where the elements of index
        denote which index of the output vector the corresponding element of inputs shall be aggregated in
        '''
        if self.aggr in ['sum', 'mean', 'min', 'max']:
            return scatter(inputs, index, dim=0, reduce=self.aggr)
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