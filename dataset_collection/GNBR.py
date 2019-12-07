import matplotlib.pyplot as plt
import numpy as np
import re
import snap

from scipy import stats

class GNBR(object):
    def __init__(self):
        # self.disease_ID_conversions_dict = self.load_conversion_dict('MedGen_UID_CUI_history.txt')
        # self.gene_ID_conversions_dict = self.load_conversion_dict('non_alt_loci_set.txt',
        #                                                           current=25, converted=18)
        self.graph = self.load_graph('part-ii-dependency-paths-gene-gene-sorted-with-themes.txt')

    def load_graph(self, path):
        graph = snap.TUNGraph.New()
        with open(path, 'r') as f:
            for unprocessed_line in f.readlines():
                categories = unprocessed_line.split('\t')
                gene1 = categories[8]
                # print(gene1)
                gene2 = categories[9]
                # print(gene2)
                if len(re.split("\W", gene1)) > 1 or len(re.split("\W", gene2)) > 1:
                    continue
                gene1 = int(gene1)
                gene2 = int(gene2)
                # if self.ID_conversions_dict.get(head, None) is not None:
                #     head = self.ID_conversions_dict[head]
                # else:
                #     print("ID Not Found: {}".format(head))
                # if self.ID_conversions_dict.get(tail, None) is not None:
                #     tail = self.ID_conversions_dict[tail]
                # else:
                #     print("ID Not Found: {}".format(tail))
                if graph.IsNode(gene1) is False:
                    graph.AddNode(gene1)
                if graph.IsNode(gene2) is False:
                    graph.AddNode(gene2)
                graph.AddEdge(gene1, gene2)
        return graph

    def load_conversion_dict(self, path, current=1, converted=0):
        conversion_dict = dict()
        with open(path, 'r') as f:
            first_line = True
            for line in f.readlines():
                if first_line is True:
                    # print(line.split('\t'))
                    # print(line.split('\t')[18])
                    # print(line.split('\t')[25])
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
        # max = 0
        # max_node = ''
        for degree_count in degree_counts:
            out_degree_list.append(np.log10((degree_count.GetVal1())))
            num_nodes_with_degree.append(np.log10(degree_count.GetVal2()))
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
        plt.title("Degree Distribution of GNBR Gene-Gene Dataset")
        plt.savefig("degree_distribution_GNBR_Gene_Gene.png")

if __name__ == "__main__":
    graph = GNBR()
    graph.plot_degree_distribution()
    nodeID = snap.GetMxOutDegNId(graph.graph)
    print(nodeID)
    Count = snap.CntUniqUndirEdges(graph.graph)
    print(Count)