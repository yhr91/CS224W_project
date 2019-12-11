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
        self.conversion_dict_reverse = dict()
        self.read_ID_file()
        self.disease_gene_sets = defaultdict(list)
        self.load_diseases()
        nx.relabel_nodes(self.G, self.conversion_dict, copy=False)
        print(self.features.columns.values)
        # print(self.disease_gene_sets["Parkinson Disease"])
        for gene in self.disease_gene_sets["Parkinson Disease"]:
            if gene == '1620':
                print("Yep")
            if self.conversion_dict.get(gene, None) is not None:
                self.graph_network("Parkinson_Disease", self.conversion_dict[gene], 'Lipoprotein')



    def read_uniprot_features(self):
        df = pd.read_pickle("uniprot_protein_features.pkl")
        return df

    def load_graph(self):
        G = nx.read_edgelist("PP-Decagon_ppi.csv",delimiter=',')
        # print(G.nodes())
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
                        self.conversion_dict_reverse[line[0]] = line[1]

    def graph_network(self, disease, gene, feature):
        fig = plt.figure(dpi=150, figsize=[8, 8])
        plt.axis('off')
        # nodes = []
        # pd_genes = self.disease_gene_sets["Parkinsonian Disorders"]
        # for gene in pd_genes:
        #     nodes.append(gene)
        # for edge in self.G.edges():
        #     print(edge)
        # print(self.G.nodes())
        wcc = []
        #LRRK2 removed nodes
        # wcc = ['Q9UFV1', 'P10635', '641373', 'Q2M2D7', 'Q86UD7', 'Q8N3J3', 'O95563', 'P0C7X1', '654341']
        nodes = [n for n in self.G.neighbors(gene) if n not in wcc]
        nodes = list(set(nodes))
        G2 = self.G.subgraph(nodes)
        print("Drawing")
        # colors = []
        # for node in nodes:
        #     colors.append(self.protein_color[node])
        # nx.spectral_layout(G2)
        self.features = self.features.fillna(0)
        colors = []
        for node in G2.nodes():
            if node in self.features.index:
                if self.features.loc[node][feature] > 0:
                    colors.append('m')
                else:
                    colors.append('b')
            else:
                colors.append('b')
        nx.draw_networkx(G2, pos=nx.spring_layout(G2, k=0.75, scale=8),
                         node_color=colors, with_labels=True, node_size=80, font_size=8)
        # gene = self.conversion_dict_reverse[gene]
        disease = disease.replace(" ","_")
        plt.title("{}_{}_subgraph.png".format(gene, feature))
        # plt.legend("Blue i")
        fig.savefig("{}/{}/{}_subgraph.png".format(disease, feature, gene), format='png')
        plt.clf()

if __name__ == '__main__':
    vizualization = VisualizeUniProtFeatures()



