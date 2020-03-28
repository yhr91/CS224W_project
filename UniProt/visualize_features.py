from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import networkx as nx
from sklearn.preprocessing import StandardScaler


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
        # for feature in self.features.columns:
        #             self.graph_disease("MITOCHONDRIAL COMPLEX I DEFICIENCY", feature)
        self.calculate_feature_enrichment('S-cysteinyl cysteine')
        # print(self.disease_gene_sets["Parkinson Disease"])
        # for gene in self.disease_gene_sets["Osteogenesis imperfecta type IV (disorder)"]:
        #     # if gene == '1620':
        #     #     print("Yep")
        #     if self.conversion_dict.get(gene, None) is not None:
        #         self.graph_network("Osteogenesis_imperfecta_type_IV_(disorder)",
        #                            self.conversion_dict[gene], 'Hydroxylation')
        # self.graph_disease("Lewy Body Disease", "S-nitrosylation")
        # self.graph_network("Parkinson_Disease", 'P04179', 'Phosphoprotein')



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

    def graph_disease(self, disease, feature):
        fig = plt.figure(dpi=150, figsize=[60, 60])
        plt.axis('off')
        node_set = set()
        disease_genes = [self.conversion_dict[n] for n in self.disease_gene_sets[disease] if n in
                         self.conversion_dict]
        # print(disease_genes)
        for gene in disease_genes:
            nodes = [n for n in self.G.neighbors(gene)]
            # print(nodes)
            node_set = node_set.union(set(nodes))
            # print(node_set)
        G2 = self.G.subgraph(list(node_set))
        print("Drawing")
        self.features = self.features.fillna(0)
        colors = []
        for node in G2.nodes():
            if node in self.features.index:
                if self.features.loc[node][feature] > 0:
                    colors.append('m')
                elif self.conversion_dict_reverse[node] in disease_genes:
                    colors.append('r')
                else:
                    colors.append('b')
            else:
                colors.append('b')
        nx.draw_networkx(G2, pos=nx.spring_layout(G2, k=0.75, scale=8),
                         node_color=colors, edge_color='0.9', with_labels=True, node_size=400, \
                                                                                   font_size=16)
        disease = disease.replace(" ", "_")
        plt.title("{} Subgraph with {} in Magenta".format(disease, feature))
        fig.savefig("{}/{}_subgraph_with_{}_features.png".format(disease, disease, feature),
        format='png')
        plt.clf()

    def graph_network(self, disease, gene, feature):
        fig = plt.figure(dpi=150, figsize=[8, 8])
        plt.axis('off')
        nodes = [n for n in self.G.neighbors(gene)]
        nodes = list(set(nodes))
        G2 = self.G.subgraph(nodes)
        print("Drawing")
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
        disease = disease.replace(" ","_")
        plt.title("{}_{}_subgraph.png".format(gene, feature))
        fig.savefig("{}/{}/{}_subgraph.png".format(disease, feature, gene), format='png')
        plt.clf()

    def calculate_feature_enrichment(self, feature):
        disease_feature_dict = dict()
        disease_counter = 1
        for disease in self.disease_gene_sets:
            node_set = set()
            disease_genes = [self.conversion_dict[n] for n in self.disease_gene_sets[disease] if n in
                             self.conversion_dict]
            # print(disease_genes)
            for gene in disease_genes:
                nodes = [n for n in self.G.neighbors(gene)]
                # print(nodes)
                node_set = node_set.union(set(nodes))
                # print(node_set)
            G2 = self.G.subgraph(list(node_set))
            self.features = self.features.fillna(0)
            colors = []
            count = 0
            feature_count = 0
            for node in G2.nodes():
                if node in self.features.index:
                    if self.features.loc[node][feature] > 0:
                        feature_count += 1
                        count += 1
                    else:
                        count += 1
                else:
                    count += 1
            disease_feature_dict[disease] = feature_count/count
            # print(disease_counter)
            disease_counter += 1
        # print(disease_feature_dict)
        df = pd.DataFrame.from_dict(disease_feature_dict, orient='index', columns=[feature])
        scaler = StandardScaler(copy=False)
        scaler.fit_transform(df)
        df = df.sort_values([feature])
        print(df)
        df.to_csv('Enrichment_{}.csv'.format(feature))


if __name__ == '__main__':
    vizualization = VisualizeUniProtFeatures()



