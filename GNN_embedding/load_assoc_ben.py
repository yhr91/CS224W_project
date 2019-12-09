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

class ProcessData:
    def __init__(self):
        self.disease = set()
        self.gene_to_disease_dict = defaultdict(set)
        self.disease_to_genes_dict = defaultdict(set)
        self.protein_df = self.create_protein_df()
        self.load_diseases()
        self.X, self.Y, self.combined = self.create_x_and_y()


    def load_diseases(self):
        with open("DG-AssocMiner_miner-disease-gene.tsv", 'r') as f:
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

    def create_disease_df(self):
        start = np.empty((len(self.gene_to_disease_dict), len(self.disease)))
        df = pd.DataFrame(start, index=self.gene_to_disease_dict.keys(), columns=self.disease)
        for key in self.gene_to_disease_dict:
            diseases = self.gene_to_disease_dict[key]
            for disease in diseases:
                df.at[key, disease] = 1
        return df

    def create_protein_df(self):
        network = nx.read_edgelist("PP-Decagon_ppi.csv", delimiter=',')
        nodes = network.nodes()
        X = np.ones((len(nodes), 1))
        df = pd.DataFrame(X, index=nodes, columns=['Features'])
        return df

    def create_x_and_y(self):
        protein_df = self.protein_df
        disease_df = self.create_disease_df()
        combined = pd.concat([protein_df, disease_df], axis=1, sort=False)
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

    def get_edges(self, edgelist_file):
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

# ProcessData()