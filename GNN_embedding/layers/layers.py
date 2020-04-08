"""Euclidean layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


def get_dim_act(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
    return dims, acts


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(GraphConvolution, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act

    def forward(self, input):
        x, adj = input
        hidden = self.linear.forward(x)
        if adj.is_sparse:
            hidden = torch.sparse.mm(adj, hidden)
        else:
            hidden = torch.mm(adj, hidden)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        output = self.act(hidden), adj
        return output

    def extra_repr(self):
        return 'input_dim={}, output_dim={}'.format(
                self.in_features, self.out_features
        )

class GraphSAGEConvolution(Module):
    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(GraphSAGEConvolution, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features * 2, out_features, use_bias)
        self.act = act

    def forward(self, input):
        '''sum the neighbors and then concat'''
        x, adj = input
        if adj.is_sparse:
            support = torch.sparse.mm(adj, x)
        else:
            support = torch.mm(adj, x)
        x = torch.cat((x, support), dim=1)
        x = self.linear.forward(x)
        x = F.dropout(x, self.dropout, training=self.training)
        output = self.act(x), adj
        return output

    def extra_repr(self):
        return 'input_dim={}, output_dim={}'.format(
                self.in_features, self.out_features
        )


class Linear(Module):
    """
    Simple Linear layer with dropout.
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act

    def forward(self, x):
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        out = self.act(hidden)
        return out


class FermiDiracDecoder(Module):
    """Fermi Dirac to compute edge probabilities based on distances."""

    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.0)
        return probs

