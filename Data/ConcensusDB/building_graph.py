#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 22:37:57 2019

Reading in ConsensusDB, cleaning up and converting into a Snap readable
format, Adding TCGA/GTEX data

@author: Yusuf
"""

import pandas as pd
import glob
import numpy as np

home = '/Users/Yusuf/Google Drive/CS224_Project/Data/'

# --------------------------
## --   General functions
# --------------------------

def setup_entrez_dict():
    dict_df = pd.read_csv('./dataset_collection/HGNC_to_NCBI.txt',delimiter='\t')
    dict_ = {}
    for k,d in dict_df.iterrows():
        dict_[d[0]] = d[1]
    return dict_

#entrez_dict = setup_entrez_dict()

def name_to_entrez(entities, entrez_dict):
    list_ = []
    for f in entities:
        try:
            list_.append(entrez_dict[f])
        except:
            list_.append(-1)
    return np.array(list_)
    
def find_samples(search_for, search_in):
    sel_samples= []
    for regex in search_for:  
        sel_samples.extend([s for s in search_in if regex in s])
    
    return sel_samples

def read_tar(d):
    return pd.read_csv(d, compression='gzip', header=0, sep='\t', 
                     quotechar='"', error_bad_lines=False)
    
def clean_gene_labels(list_):
    new_list_ = []
    for l in list_:
        new_list_.append([m.split('_')[0] for m in l])
    return new_list_

def merge_list_dfs(dfs,on='Hugo_Symbol'):
    merged_df = dfs[0]
    for f in dfs[1:]:
        merged_df = merged_df.merge(f,how='inner')
    return merged_df 
# --------------------------
## --   Handling ConsensusPathDB
# --------------------------
class ConsensusPathDB(object):  
    def __init__(self, path=home+'5330593/', keys = ['*']):
        ## Read in all ConsensusPathDB sample_ids
        dirs = []
        for k in keys:
            dirs.extend(glob.glob(path+'*'+k+'*.gz'))
        
        dfs = []
        for d in dirs:
            dfs.append(read_tar(d))   
            
        self.dfs = dfs

    def get_sample_IDs(self):
        samples = []
        for df_ in self.dfs:
            samples.extend([l for l in df_.columns])
        return samples

# --------------------------
# -- Handling TCGA GTex data 
# --------------------------
class TCGA_GTex(object):
    @staticmethod
    def get_TCGA_sample_IDs(organ):
        df = read_tar(home+'TCGA_sampleID/'+organ+'.gz')   
        return df.sample_submitter_id.values
    
    @staticmethod
    def get_GTex_sample_IDs(path, organs):
        df = pd.read_csv(path,delimiter='\t')
        for o in organs:
            ret = df[df['Characteristics[organism part]']==o]['Source Name'].unique()
        return ret
    
    @staticmethod
    def get_feature_vec(organ, tcga, gtex):
        ## Read in ConsensusPathDB and get samples ids for that organ
        CPDB = ConsensusPathDB(keys=organ)
        all_samples = CPDB.get_sample_IDs()
    
        ## Refer to TCGA/GTEX to identify sample_IDs that have a specific cancer
        if (tcga): samples_tcga = TCGA_GTex.get_TCGA_sample_IDs(organ[0]);
        if (gtex): samples_gtex = TCGA_GTex.get_GTex_sample_IDs(home+'GTex.txt',organ);
    
        ## Use those samples that are linked to cancer of interest to calculate means
        samples=[]
        if (tcga): samples.extend(find_samples(samples_tcga,all_samples));
        if (gtex): samples.extend(find_samples(samples_gtex,all_samples));
        
        mean_dfs = []
        for i,df_ in enumerate(CPDB.dfs):
            mean_df = df_.loc[:,samples+['Hugo_Symbol']]
            mean_df[organ[i]] = mean_df.iloc[:,:-1].mean(1)
            mean_dfs.append(mean_df.loc[:,['Hugo_Symbol',organ[i]]])
        return mean_dfs
    
# --------------------------
# -- Main
# --------------------------
def main(tcga=True,gtex=True):
    ## Read in consensusPathDB dataset
    df = pd.read_csv(home+'ConsensusPathDB_human_PPI.tsv',skiprows=[0], 
                     delimiter='\t')

    ## Pull out only those interactions which have confidence score > 0.5
    df = df[df.iloc[:,3]>0.5]

    ## Create a snap graph: convert edge IDs from categorical to numerical
    edges = np.array([d.split(',') for d in df.iloc[:,2].values])
    edges = np.array(clean_gene_labels(edges))
    
    ## Get an EntrezID dictionary
    edges = [name_to_entrez(edges[:,0],entrez_dict),
             name_to_entrez(edges[:,1],entrez_dict)]
    edges = np.array(edges).astype('int').T
    edges = pd.DataFrame(edges)
    
    edges = edges[edges[0] != -1.0]
    edges = edges[edges[1] != -1.0]
    
    edges.to_csv('ConsensusPathDB_human_PPI_HiConfidence_snap.csv'
                ,header=False,index=False)
    
    #%% Adding feature vector
    organs = [['breast'],['bladder','urinary bladder'],['kidney','cortex of kidney']]
    feats_arr = []
    
    for organ in organs:
        for f in TCGA_GTex.get_feature_vec(organ, tcga, gtex):
            feats_arr.append(f)
      
    # Create a single feature matrix and replace genes with node IDs
    feats_df = merge_list_dfs(feats_arr,on='Hugo_Symbol')
    
    # Convert categories into a df for merging
    feats_df['Entrez'] = name_to_entrez(feats_df['Hugo_Symbol'],entrez_dict).astype('int')
    
    feats_df.to_csv(home+'TCGA_GTEX_GeneExpression.csv')
    return edges, feats_df
    
df = main(tcga=False, gtex=True)

## To check overlap with PPI
x = pd.read_csv('./dataset_collection/PP-Decagon_ppi.csv')
y = x.as_matrix().flatten()
len([d for d in df[1]['Entrez'] if d in y])