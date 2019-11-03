#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 22:37:57 2019

@author: Yusuf
"""

import snap
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def convert_category_to_num(edges):
    cat = np.concatenate([edges[:,0],edges[:,1]])
    cat = pd.Categorical(cat)
    return np.array([cat.codes[:int(len(cat.codes)/2)],
               cat.codes[int(len(cat.codes)/2):]])


## Read in consensusPathDB dataset
df = pd.read_csv('/Users/Yusuf/Google Drive/CS224_Project/Data/ConsensusPathDB_human_PPI.tsv',skiprows=[0], 
                 delimiter='\t')

## Pull out only those interactions which have confidence score > 0.5
print('Original size of dataset', str(len(df)))
df = df[df.iloc[:,3]>0.5]
print('Size of dataset with only high confidence elements',
      str(len(df)))

## Converting edge IDs from categorical to numerical
edges = np.array([d.split(',') for d in df.iloc[:,2].values])
edges = convert_category_to_num(edges)
edges = pd.DataFrame(edges.T)
edges.to_csv('ConsensusPathDB_human_PPI_HiConfidence'
            ,header=False,index=False)

## Read in snap.py version of graph
s = snap.LoadEdgeList(snap.PUNGraph,'ConsensusPathDB_human_PPI_HiConfidence',
                      0,1,',')