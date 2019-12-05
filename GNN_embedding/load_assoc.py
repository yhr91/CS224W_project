#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Kendrick
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

def get_X(X_file):
    pass

    # X = pd.read_csv(x)
    # X = X.drop(columns=X.columns[0])
    # X = X.drop_duplicates(subset='Entrez')

    # ## Some Entrez ID's are negative, not sure why...
    # if not keep_all_entrez:
    #     pass
    # #X = X[X.Entrez>0] # Removing some junk values
    # return X

def get_y(X_file, y_file):
    pass

def get_edges(edgelist_file):
    

## Get Y values
def get_y(X, y):
    Y = pd.read_csv(y, delimiter='\t')
    Y = set(Y['711_Known_Cancer_Genes'])
    #   Y = [x for x in list(Y) if x == x] # not sure what this is supposed to do
    Y = list(Y)

    ## Map Y to Entrez IDs
    Y = pd.DataFrame(Y).merge(X.loc[:,['Entrez','Hugo_Symbol']],
                            right_on='Hugo_Symbol',left_on = 0)
    Y = Y.Entrez.values
    Y = [int(i in Y) for i in X.Entrez.values]
    return Y

## Get edge list
def get_edges(X, edgelist_file):
    edgelist = pd.read_csv(edgelist_file, header=None)

    # Remove edges for which we don't have entrez
    idx = np.logical_and(edgelist[0].isin(X.Entrez.values),
                    edgelist[1].isin(X.Entrez.values))
    edgelist = edgelist[idx]
    # Get the ordering of samples in X,Y...
    mappings = {X.Entrez.values[i]: i for i in range(len(X))}

    # .. and use it to number nodes in edge list
    edgelist[0] = edgelist[0].apply(lambda cell: mappings[cell])
    edgelist[1] = edgelist[1].apply(lambda cell: mappings[cell])
    return edgelist