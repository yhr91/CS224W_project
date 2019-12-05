#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Nov 7 21:27:10 2019

@author: Kendrick, Yusuf

util functions
"""

import pandas as pd
import numpy as np

import torch
from torch_geometric.data import Data

def load_pyg(X, edges, y, size=100): 
  # Create masks
  np.random.seed(20) # to make sure the test set is always the same
  ones = np.random.choice(np.where(y == 1)[0], size=size, replace=False)
  np.random.seed(30)
  zeros = np.random.choice(np.where(y == 0)[0], size=size, replace=False)

  test_mask = [True if i in ones or i in zeros 
              else False for i in range(len(y))]
  train_mask = [True if i not in ones and i not in zeros 
                else False for i in range(len(y))]

  # Return data loader
  data = Data(x=X, edge_index=edges.t().contiguous(), y=y)
  data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
  data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

  return data