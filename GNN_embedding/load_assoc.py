#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Kendrick
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

def get_X(edgelist_file):
    edges = get_edges(edgelist_file)
    nodes = set(np.unique(edges.values.flatten()))
    X = np.ones((len(nodes), 1))
    return X

def get_y(y_file, edgelist_file):
    edges = get_edges(edgelist_file)
    nodes = np.unique(edges.values.flatten())
    X = pd.read_csv(y_file, delimiter='\t')
    diseases = pd.unique(X.loc[X['Gene ID'].isin(nodes)]['Disease Name'])
    mappings = {d: i for i, d in enumerate(diseases)}
    classification = [mappings[X.loc[X['Gene ID'] == n]['Disease Name'].iloc[0]] for n in sorted(nodes)]
    return classification

def get_edges(edgelist_file):
    edges = pd.read_csv(edgelist_file, header=None)
    return edges