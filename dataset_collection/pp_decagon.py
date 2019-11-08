import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import snap

class PPDecagon(object):
    def __init__(self):
        self.graph = self.load_graph('PP-Decagon_ppi.csv')

    def load_graph(self, path):
        graph = snap.TUNGraph.New()
        with open(path, 'r') as f:
            for unprocessed_line in f.readlines():
                line = unprocessed_line.split(',')
                head = int(line[0])
                tail = int(line[1])
                if graph.IsNode(head) is False:
                    graph.AddNode(head)
                if graph.IsNode(tail) is False:
                    graph.AddNode(tail)
                graph.AddEdge(head, tail)
        return graph

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
        plt.plot(out_degree_list, y_values_for_slope_line)
        plt.xlabel("Out Degree (log-scale)")
        plt.ylabel("Number of Nodes with Degree (log-scale)")
        plt.title("Degree Distribution of PP-Decagon Dataset")
        plt.savefig("degree_distribution_pp_decagon.png")




if __name__ == "__main__":
    graph = PPDecagon()
    graph.plot_degree_distribution()
    Count = snap.CntUniqUndirEdges(graph.graph)
    print("Number of edges: " +str(Count))
    # nodeID = snap.GetMxOutDegNId(graph.graph)
    # print(nodeID)


