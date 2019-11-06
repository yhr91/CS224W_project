#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 22:37:57 2019

Reading in ConsensusDB, cleaning up and converting into a Snap readable
format

@author: Yusuf
"""

import snap
import pandas as pd
import glob
import numpy as np
from matplotlib import pyplot as plt

def convert_category_to_num(edges):
    cat = np.concatenate([edges[:,0],edges[:,1]])
    cat = pd.Categorical(cat)
    return [np.array([cat.codes[:int(len(cat.codes)/2)],
                      cat.codes[int(len(cat.codes)/2):]]),
            cat.categories.values]
    
def find_samples(search_for, search_in):
    sel_samples= []
    for regex in search_for:  
        sel_samples.extend([s for s in search_in if regex in s])
    
    return sel_samples

def read_tar(d):
    return pd.read_csv(d, compression='gzip', header=0, sep='\t', 
                     quotechar='"', error_bad_lines=False)

def get_ConsensusPathDB_data(keys = ['*']):
    ## Read in all ConsensusPathDB sample_ids
    dirs = []
    for k in keys:
        dirs.extend(glob.glob('/Users/Yusuf/Google Drive/CS224_Project/Data/5330593/*'+k+'*.gz'))
    
    dfs = []
    for d in dirs:
        dfs.append(read_tar(d))   
        
    return dfs

def get_ConsensusPathDB_sample_IDs(dfs):
    samples = []
    for df_ in dfs:
        samples.extend([l for l in df_.columns])
    return samples
        
def get_TCGA_sample_IDs(path):
    df = read_tar(path)   
    return df.sample_submitter_id.values

def get_GTex_sample_IDs(path, organs):
    df = pd.read_csv(path,delimiter='\t')
    for o in organs:
        ret = df[df['Characteristics[organism part]']==o]['Source Name'].unique()
    return ret

def merge_list_dfs(dfs,on='Hugo_Symbol'):
    merged_df = dfs[0]
    for f in dfs[1:]:
        merged_df = merged_df.merge(f,how='inner')
    return merged_df

def clean_gene_labels(list_):
    new_list_ = []
    for l in list_:
        new_list_.append([m.split('_')[0] for m in l])
    return new_list_
    

def get_feature_vec(organ):
    ## Read in ConsensusPathDB GeneExpression data for that organ
    dfs = get_ConsensusPathDB_data(organ)
    
    ## And get sample ids
    all_samples = get_ConsensusPathDB_sample_IDs(dfs)

    ## Refer to TCGA to identify sample_IDs that have a specific cancer
    samples_tcga = get_TCGA_sample_IDs('/Users/Yusuf/Google Drive/CS224_Project/Data/biospecimen.cases_selection.2019-11-04.tar.gz')
    
    ## Refer to GTEx to identify sample_IDs that have a specific cancer
    samples_gtex = get_GTex_sample_IDs('/Users/Yusuf/Google Drive/CS224_Project/Data/GTex.txt',organ)

    ## Use those samples that are linked to cancer of interest to calculate means
    samples = find_samples(samples_tcga,all_samples)
    samples.extend(find_samples(samples_gtex,all_samples))
    
    mean_dfs = []
    for i,df_ in enumerate(dfs):
        mean_df = df_.loc[:,samples+['Hugo_Symbol']]
        mean_df[organ[i]] = mean_df.iloc[:,:-1].mean(1)
        mean_dfs.append(mean_df.loc[:,['Hugo_Symbol',organ[i]]])
    return mean_dfs
    

def main():
    ## Read in consensusPathDB dataset
    df = pd.read_csv('/Users/Yusuf/Google Drive/CS224_Project/Data/ConsensusPathDB_human_PPI.tsv',skiprows=[0], 
                     delimiter='\t')

    ## Pull out only those interactions which have confidence score > 0.5
    df = df[df.iloc[:,3]>0.5]

    ## Create a snap graph: convert edge IDs from categorical to numerical
    edges = np.array([d.split(',') for d in df.iloc[:,2].values])
    edges = np.array(clean_gene_labels(edges))
    edges,categories = convert_category_to_num(edges)
    edges = pd.DataFrame(edges.T)
    edges.to_csv('ConsensusPathDB_human_PPI_HiConfidence_snap.csv'
                ,header=False,index=False)
    
    ## Read in snap.py version of graph
    s = snap.LoadEdgeList(snap.PUNGraph,'ConsensusPathDB_human_PPI_HiConfidence_snap.csv',
                          0,1,',')

    #%% Adding feature vector
    organs = [['breast'],['bladder'],['kidney']]
    feats_arr = []
    
    for organ in organs:
        for f in get_feature_vec(organ):
            feats_arr.append(f)
      
    # Create a single feature matrix and replace genes with node IDs
    feats_df = merge_list_dfs(feats_arr,on='Hugo_Symbol')
    
    # Convert categories into a df for merging
    categories = pd.DataFrame(categories).reset_index()
    categories = categories.rename(columns={0:'Hugo_Symbol'})
    feats_df = feats_df.merge(categories, on='Hugo_Symbol')
    
    return feats_df
    
df = main()