#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Kendrick
"""

from collections import defaultdict
import networkx as nx
import numpy as np
import pandas as pd
import pathlib

class ProcessData:
    def __init__(self):
        # self.disease_gene_path = pathlib.Path(__file__).parents[0]
        # print(self.disease_gene_path)
        self.disease = set()
        self.disease_gene_dict = defaultdict(set)
        self.load_diseases()
        self.X, self.Y, self.combined = self.create_x_and_y()
        # print(self.x_values)
        # print(self.y_values)
        # self.get_X()

    def load_diseases(self):
        with open("DG-AssocMiner_miner-disease-gene.tsv", 'r') as f:
            for line in f.readlines()[1:]:
                line = line.strip().split('\t')
                if len(line) < 3:
                    continue
                disease = line[1].strip('"')
                gene = line[2]
                self.disease_gene_dict[gene].add(disease)
                self.disease.add(disease)

    def create_disease_df(self):
        start = np.zeros((len(self.disease_gene_dict), len(self.disease)))
        df = pd.DataFrame(start, index=self.disease_gene_dict.keys(), columns=self.disease)
        for key in self.disease_gene_dict:
            diseases = self.disease_gene_dict[key]
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
        protein_df = self.create_protein_df()
        disease_df = self.create_disease_df()
        combined = protein_df.append(disease_df)
        combined.Features.fillna(1.0, inplace=True)
        combined.fillna(0.0, inplace=True)
        x = combined[['Features']]
        y = combined.drop(columns='Features')
        return x,y, combined

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
