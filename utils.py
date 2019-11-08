#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Nov 7 21:27:10 2019

@author: Kendrick

util functions`
"""
import pandas as pd
import numpy as np

from torch_geometric.data import Data, DataLoader

def load_pyg(x, edge_index, y):
    X = pd.read_csv(x)
    edges = pd.read_csv(edge_index)
    Y = np.loadtxt(y)
    graph = Data(x=X, edge_index=edges, y=Y)