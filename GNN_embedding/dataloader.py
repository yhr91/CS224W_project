#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 23:25:36 2019

@author: Kendrick, Yusuf


"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

## Get X values
def get_X(x, keep_all_entrez = True):
  X = pd.read_csv(x)
  X = X.drop(columns=X.columns[0])
  X = X.drop_duplicates(subset = 'Entrez')

  ## Some Entrez ID's are negative, not sure why...
  if (keep_all_entrez != True):
    pass
    #X = X[X.Entrez>0] # Removing some junk values
  return X

## Get Y values
def get_Y(X,y):
  Y = pd.read_csv(y,delimiter='\t')
  Y = set(Y['711_Known_Cancer_Genes'])
  Y = [x for x in list(Y) if x == x]

  ## Map Y to Entrez IDs
  Y = pd.DataFrame(Y).merge(X.loc[:,['Entrez','Hugo_Symbol']],
                            right_on='Hugo_Symbol',left_on = 0)
  Y = Y.Entrez.values
  Y = [1 if i in Y else 0 for i in X.Entrez.values]

  return Y

## Get edge list
def get_edges(X, edgelist_file):
  edgelist = pd.read_csv(edgelist_file, header=None)

  # Remove edges for which we don't have entrez
  idx = np.logical_and(edgelist[0].isin(X.Entrez.values),
                edgelist[1].isin(X.Entrez.values))
  edgelist = edgelist[idx]
  return edgelist

def load_pyg(x, edgelist_file, y): 

  # Get X,Y,edges
  X = get_X(x)
  Y = get_Y(X, y)
  edgelist = get_edges(X, edgelist_file)

  # Get the ordering of samples in X,Y...
  mappings = {X.Entrez.values[i]: i for i in range(len(X))}
  X = torch.tensor(X.iloc[:,1:4].values, dtype=torch.float)
  Y = torch.tensor(Y,dtype=torch.long)

  # .. and use it to number nodes in edge list
  edgelist[0] = edgelist[0].apply(lambda cell: mappings[cell])
  edgelist[1] = edgelist[1].apply(lambda cell: mappings[cell])
  edges = torch.tensor(edgelist.values, dtype=torch.long)

  # Create masks
  np.random.seed(20) # to make sure the test set is always the same
  ones = np.random.choice(np.where(Y == 1)[0], size = 100, replace = False)
  np.random.seed(30)
  zeros = np.random.choice(np.where(Y == 0)[0], size = 100, replace = False)

  test_mask = [True if i in ones or i in zeros 
              else False for i in range(len(Y))]
  train_mask = [True if i not in ones and i not in zeros 
                else False for i in range(len(Y))]

  # Return data loader
  data = Data(x=X, edge_index=edges.t().contiguous(), y=Y)
  data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
  data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

  return data