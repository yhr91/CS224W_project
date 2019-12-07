from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import networkx as nx


class VisualizeUniProtFeatures:
    def __init__(self):
        self.features = self.read_uniprot_features()
        self.G = self.load_graph()
        self.conversion_dict = dict()
        self.read_ID_file()
        self.disease_gene_sets = defaultdict(list)
        self.load_diseases()
        nx.relabel_nodes(self.G, self.conversion_dict, copy=False)
        self.graph_network()


    def read_uniprot_features(self):
        df = pd.read_pickle("uniprot_protein_features.pkl")
        print(df)
        return df

    def load_graph(self):
        G = nx.read_edgelist("PP-Decagon_ppi.csv",delimiter=',')
        return G

    def load_diseases(self):
        with open("DG-AssocMiner_miner-disease-gene.tsv", 'r') as f:
            for line in f.readlines()[1:]:
                line = line.strip().split('\t')
                if len(line) < 3:
                    continue
                disease = line[1].strip('"')
                gene = line[2]
                self.disease_gene_sets[disease].append(gene)

    def read_ID_file(self):
        with open("NCBI_to_UNIPROT.txt", 'r') as f:
            for line in f.readlines()[1:]:
                line = line.strip()
                line = line.split()
                if len(line) > 1:
                    if self.G.has_node(line[1]):
                        self.conversion_dict[line[1]] = line[0]

    def graph_network(self):
        fig = plt.figure(dpi=150, figsize=[15, 15])
        plt.axis('off')
        # nodes = []
        # pd_genes = self.disease_gene_sets["Parkinsonian Disorders"]
        # for gene in pd_genes:
        #     nodes.append(gene)
        # for edge in self.G.edges():
        #     print(edge)
        nodes = [n for n in self.G.neighbors('120892')]
        nodes = list(set(nodes))
        G2 = self.G.subgraph(nodes)
        print("Drawing")
        # colors = []
        # for node in nodes:
        #     colors.append(self.protein_color[node])
        nx.draw_circular(G2, nodes=G2.nodes(), node_size=80, font_size=8)
        fig.savefig("PPI_Graph.png", format='png')

if __name__ == '__main__':
    vizualization = VisualizeUniProtFeatures()



