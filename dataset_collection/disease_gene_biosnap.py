import matplotlib.pyplot as plt
import numpy as np
import snap

from scipy import stats

class DGMiner(object):
    def __init__(self):
        self.disease_ID_conversions_dict = self.load_disease_conversion_dict(
            'MedGen_UID_CUI_history.txt')
        self.gene_ID_conversions_dict = self.load_disease_conversion_dict('non_alt_loci_set.txt',
                                                                  current=25, converted=18)
        self.graph = self.load_graph('DG-Miner_miner-disease-gene.tsv')


    def load_graph(self, path):
        graph = snap.TUNGraph.New()
        with open(path, 'r') as f:
            for unprocessed_line in f.readlines()[1:]:
                line = unprocessed_line.split('\t')
                head = line[0].lstrip("MESH:")
                tail = line[1]
                if self.disease_ID_conversions_dict.get(head, None) is not None:
                    head = self.disease_ID_conversions_dict[head]
                else:
                    print("ID Not Found: {}".format(head))
                    continue
                if self.gene_ID_conversions_dict.get(tail, None) is not None:
                    tail = self.gene_ID_conversions_dict[tail]
                else:
                    print("ID Not Found: {}".format(tail))
                    continue
                if graph.IsNode(head) is False:
                    graph.AddNode(head)
                if graph.IsNode(tail) is False:
                    graph.AddNode(tail)
                graph.AddEdge(head, tail)
        return graph


    def load_conversion_dict(self, path, current=1, converted=0):
        conversion_dict = dict()
        with open(path, 'r') as f:
            first_line = True
            for line in f.readlines():
                if first_line is True:
                    first_line = False
                    continue
                else:
                    lines = line.split()
                    if len(lines) > current and len(lines) > converted:
                        conversion_dict[lines[current]] = lines[converted]
        return conversion_dict


    def plot_degree_distribution(self):
        degree_counts = snap.TIntPrV()
        snap.GetOutDegCnt(self.graph, degree_counts)
        out_degree_list = []
        num_nodes_with_degree = []
        max = 0
        max_node = ''
        for degree_count in degree_counts:
            out_degree_list.append(np.log10((degree_count.GetVal1())))
            num_nodes_with_degree.append(np.log(degree_count.GetVal2()))
        slope, intercept, r_value, p_value, std_err = stats.linregress(out_degree_list,
                                                                       num_nodes_with_degree)
        # slope_line = int(slope) * out_degree_list + intercept
        y_values_for_slope_line = np.array(out_degree_list)
        y_values_for_slope_line = np.multiply(y_values_for_slope_line, slope) + intercept
        plt.scatter(out_degree_list, num_nodes_with_degree, alpha=0.5)
        plt.plot(out_degree_list, y_values_for_slope_line)
        plt.xlabel("Out Degree (log-scale)")
        plt.ylabel("Number of Nodes with Degree (log-scale)")
        plt.title("Degree Distribution of PP-Decagon Dataset")
        plt.savefig("degree_distribution_pp_decagon.png")
        print("print this???")

class DGAssocMiner(object):
    def __init__(self):
        self.disease_ID_conversions_dict = self.load_conversion_dict('MedGen_UID_CUI_history.txt')
        # self.gene_ID_conversions_dict = self.load_conversion_dict('non_alt_loci_set.txt',
        #                                                           current=25, converted=18)
        self.graph = self.load_graph('DG-AssocMiner_miner-disease-gene.tsv')


    def load_graph(self, path):
        graph = snap.TUNGraph.New()
        with open(path, 'r') as f:
            for unprocessed_line in f.readlines()[1:]:
                line = unprocessed_line.split('\t')
                disease = line[0]
                gene = int(line[2])
                if self.disease_ID_conversions_dict.get(disease, None) is not None:
                    disease = int(self.disease_ID_conversions_dict[disease])
                else:
                    print("ID Not Found: {}".format(disease))
                    continue
                if graph.IsNode(disease) is False:
                    graph.AddNode(disease)
                if graph.IsNode(gene) is False:
                    graph.AddNode(gene)
                graph.AddEdge(disease, gene)
        return graph


    def load_conversion_dict(self, path, current=1, converted=0):
        conversion_dict = dict()
        with open(path, 'r') as f:
            first_line = True
            for line in f.readlines():
                if first_line is True:
                    first_line = False
                    continue
                else:
                    lines = line.split()
                    if len(lines) > current and len(lines) > converted:
                        conversion_dict[lines[current]] = lines[converted]
        return conversion_dict

    def load_disease_conversion_dict(self, path, current=1, converted=0):
        conversion_dict = dict()
        with open(path, 'r') as f:
            first_line = True
            for line in f.readlines():
                if first_line is True:
                    first_line = False
                    continue
                else:
                    lines = line.split()
                    if len(lines) > current and len(lines) > converted:
                        if lines[-1][-1] != '0':
                            conversion_dict[lines[current]] = lines[converted]
        return conversion_dict


    def plot_degree_distribution(self):
        degree_counts = snap.TIntPrV()
        snap.GetOutDegCnt(self.graph, degree_counts)
        out_degree_list = []
        num_nodes_with_degree = []
        max = 0
        max_node = ''
        for degree_count in degree_counts:
            out_degree_list.append(np.log10((degree_count.GetVal1())))
            num_nodes_with_degree.append(np.log(degree_count.GetVal2()))
        slope, intercept, r_value, p_value, std_err = stats.linregress(out_degree_list,
                                                                       num_nodes_with_degree)
        # slope_line = int(slope) * out_degree_list + intercept
        y_values_for_slope_line = np.array(out_degree_list)
        y_values_for_slope_line = np.multiply(y_values_for_slope_line, slope) + intercept
        plt.scatter(out_degree_list, num_nodes_with_degree, alpha=0.5)
        plt.plot(out_degree_list, y_values_for_slope_line, 'r-')
        plt.xlim(0, max(out_degree_list))
        plt.ylim(0, max(num_nodes_with_degree))
        plt.xlabel("Out Degree (log-scale)")
        plt.ylabel("Number of Nodes with Degree (log-scale)")
        plt.title("Degree Distribution of DG_AssocMiner Dataset")
        plt.savefig("degree_distribution_DG_AssocMiner.png")
        print("THIS IS HAPPENING")

if __name__ == "__main__":
    graph = DGAssocMiner()
    graph.plot_degree_distribution()
    nodeID = snap.GetMxOutDegNId(graph.graph)
    print(nodeID)