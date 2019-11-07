#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:17:42 2019

@author: Yusuf

Script for testing a graph embedding using a PPI graph and feature vectors for
each node. The classification task is to predict gene nodes
"""

import pandas as pd
import snap

home = '/Users/Yusuf/Google Drive/CS224_Project/Data/ForAnalysis/'

def load_data():
    ## Load snap graph
    graph = snap.LoadEdgeList(snap.PUNGraph,home+'ConsensusPathDB_human_PPI_HiConfidence_snap.csv',
                          0,1,',') 
    
    ## Load feature matrix
    X = pd.read_csv(home+'ConsensusPathDB_human_PPI_HiConfidence_snap_FeatsMat.csv')
    X = X.iloc[:,1:]
    
    ## Load y labels
    Y = pd.read_csv(home+'Y/NCG_cancergenes_list.txt',skiprows=[0],delimiter='\t',
                    header=None)
    
    # Note that only the first column are known cancer genes, the second columns
    # are highly likely cancer genes, but we are not using currently
    Y = Y[0].unique()
    
    return X,Y,graph

if __name__ == '__main__':
    X,Y,graph = load_data()