#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Kendrick, Ben
"""

from collections import defaultdict
import itertools
import networkx as nx
import numpy as np
import pandas as pd
import pathlib
from sklearn.preprocessing import StandardScaler # for normalizing uniprot features

class ProcessData:
    def __init__(self, edgelist_file, use_features=True):
        self.edgelist_file = edgelist_file
        self.use_features = use_features
        self.disease = set()
        if self.use_features:
            self.conversions_dict = dict()
            self.read_ID_file()
        self.gene_to_disease_dict = defaultdict(set)
        self.disease_to_genes_dict = defaultdict(set)
        self.protein_df = self.create_protein_df()
        if self.use_features:
            self.protein_df = self.load_uniprot_features()
        self.load_diseases()
        self.X, self.Y, self.combined = self.create_x_and_y()


    def load_diseases(self):
        with open("../Data/Y/DG-AssocMiner_miner-disease-gene.tsv", 'r') as f:
            for line in f.readlines()[1:]:
                line = line.strip().split('\t')
                if len(line) < 3:
                    continue
                disease = line[1].strip('"')
                gene = line[2]
                if gene not in self.protein_df.index:
                    continue
                self.gene_to_disease_dict[gene].add(disease)
                self.disease_to_genes_dict[disease].add(gene)
                self.disease.add(disease)
        self.disease = sorted(list(self.disease))

    def create_disease_df(self):
        start = np.empty((len(self.gene_to_disease_dict), len(self.disease)))
        df = pd.DataFrame(start, index=self.gene_to_disease_dict.keys(), columns=self.disease)
        for key in self.gene_to_disease_dict:
            diseases = self.gene_to_disease_dict[key]
            for disease in diseases:
                df.at[key, disease] = 1
        return df

    def create_protein_df(self):
        network = nx.read_edgelist(self.edgelist_file, delimiter=',')
        nodes = network.nodes()
        X = np.ones((len(nodes), 1))
        df = pd.DataFrame(X, index=nodes, columns=['Features'])
        return df

    def create_x_and_y(self):
        protein_df = self.protein_df
        disease_df = self.create_disease_df()
        if self.use_features:
            combined = pd.concat([protein_df, disease_df], axis=1, sort=False)
            combined.fillna(0.0, inplace=True)
            x = combined.copy()
            y = combined
            drop_set = set()
            for column in x.columns:
                if column in self.disease:
                    drop_set.add(column)
            for drop in drop_set:
                x = x.drop(drop, axis=1)
            for column in x.columns:
                y = y.drop(column, axis=1)
            return x,y, combined
        else:
            combined = protein_df.merge(disease_df, left_index=True,
                                        right_index=True, how='outer')
            combined.Features.fillna(1.0, inplace=True)
            combined.fillna(0.0, inplace=True)
            x = combined[['Features']]
            y = combined.drop(columns='Features')
            return x,y, combined

    def check_overlap(self):
        for pair in itertools.combinations(self.disease_to_genes_dict, 2):
            # print("what")
            if len(self.disease_to_genes_dict[pair[0]].intersection(self.disease_to_genes_dict[pair[
                1]])) > 0:
                print(pair[0]+" overlaps with "+pair[1])

    def get_edges(self, edgelist_file=None):
        if edgelist_file is None:
            edgelist_file = self.edgelist_file
        edgelist = pd.read_csv(edgelist_file, header=None)

        # Remove edges for which we don't have entrez
        idx = np.logical_and(edgelist[0].isin(self.X.index.values),
                             edgelist[1].isin(self.X.index.values))
        edgelist = edgelist[idx]
        # Get the ordering of samples in X,Y...
        mappings = {float(self.X.index.values[i]): i for i in range(len(self.X))}
        # print(mappings)

        # .. and use it to number nodes in edge list
        edgelist[0] = edgelist[0].apply(lambda cell: mappings[cell])
        edgelist[1] = edgelist[1].apply(lambda cell: mappings[cell])
        return edgelist
    
    '''NOTE: used for uniprot features'''
    def read_ID_file(self, filename="../Data/NCBI_to_UNIPROT.txt"):
        with open(filename, 'r') as f:
            for line in f.readlines()[1:]:
                line = line.strip()
                line = line.split()
                if len(line) > 1:
                    self.conversions_dict[line[0]] = line[1]

    # TODO normalize features
    def load_uniprot_features(self):
        df = pd.read_pickle("./Protein_Features/uniprot_ptm_features.pkl")
        df = df.fillna(0)
        # Convert to binary features (rather than some features being counts)
        # print(df.columns)
        # print(df)
        df = df.where(df == 0.0, 1.0)
        # print(df)
        # Converts names from uniprot to NCBI---note: some nodes don't have mapping for some
        # dumb reason
        df = df.rename(index=self.conversions_dict)
        drop_set = set()
        for column in df.columns:
            if column not in set(['Acetylation', 'Glycoprotein', 'Hydroxylation',
                                  'INTRAMEM', 'Methylation','N6-succinyllysine',
                                  'N6-Lysine Modification, alternate','Neddylation',
                                  'Phosphoprotein', 'Sumoylation', 'TRANSMEM', 'Ubiquitination']):
                drop_set.add(column)
        for dropped in drop_set:
            df = df.drop(dropped, axis=1)
        drop_set = set()
        for row in df.index:
            if row not in self.protein_df.index:
                drop_set.add(row)
        for dropped in drop_set:
            df = df.drop(dropped, axis=0)
        combined = pd.concat([df, self.protein_df], axis=1, sort=False)
        combined.fillna(0, inplace=True)
        # print(combined)
        scaler = StandardScaler(copy=False)
        scaler.fit_transform(combined)
        # print(combined)
        return combined

    # Returns the indices of all diseases that fall within specific classes
    def get_disease_class_idx(self, disease_list):
        disease_classes = pd.read_csv('../Data/bio-pathways-diseaseclasses.csv')
        sel_disease_classes = []
        for d in disease_list:
            sel_disease_classes.extend(
                disease_classes[disease_classes['Disease Class'] == d]['Disease Name'].values)
        return [it for it, i in enumerate(self.Y.columns.values) if i in sel_disease_classes]

# ProcessData()
