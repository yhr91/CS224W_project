#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 23:25:36 2019

@author: Kendrick, Yusuf


"""

import numpy as np
import pandas as pd
import torch

import torch
from torch_geometric.data import Data, DataLoader

## Get X values
def get_X():
  X = 'https://github.com/yhr91/CS224W_project/blob/master/Data/ForAnalysis/X/TCGA_GTEX_GeneExpression.csv?raw=true'
  X = pd.read_csv(X)
  X.drop(columns=X.columns[0], inplace=True)
  X = X[X.Entrez>0] # Removing some junk values
  return X

## Get Y values
def get_Y(X):
  Y = 'https://github.com/yhr91/CS224W_project/raw/master/Data/ForAnalysis/Y/NCG_cancergenes_list.txt'
  Y = pd.read_csv(Y,delimiter='\t')
  Y = set(Y['711_Known_Cancer_Genes'])
  Y = [x for x in list(Y) if x != 'nan']

  ## Map Y to Entrez IDs
  Y = pd.DataFrame(Y).merge(X.loc[:,['Entrez','Hugo_Symbol']],right_on='Hugo_Symbol',left_on = 0)
  Y = Y.Entrez.values
  
  return Y

## Get edge list
def get_edges():
  edgelist = 'https://github.com/yhr91/CS224W_project/blob/master/Data/PP-Decagon_ppi.csv?raw=true'
  edgelist = pd.read_csv(edgelist, header=None)
  return edgelist

# Create data loader
def load_pyg(x, edge_index, y): 
  # Get X,Y,edges
  X = get_X()
  Y = get_Y(X)
  edgelist = get_edges()

  X = torch.tensor(X.iloc[:,1:4].values)
  Y = torch.tensor(Y)
  edges = torch.tensor(edgelist, dtype=torch.long)

  # Create masks
  ones = np.random.choice(np.where(Y == 1)[0], size = 50)
  zeros = np.random.choice(np.where(Y == 0)[0], size = 50)
  test_mask = [1 if i in ones or i in zeros 
              else 0 for i in range(len(Y))]
  train_mask = [1 if i not in ones and i not in zeros 
                else 0 for i in range(len(Y))]

  # Return data loader
  data = Data(x=X, edge_index=edges.t().contiguous(), y=Y)
  data.train_mask = torch.tensor(train_mask, dtype=torch.long)
  data.test_mask = torch.tensor(test_mask, dtype=torch.long)

  return data