import ast
import os
import numpy as np
from load_assoc import ProcessData

def main():
    directory = 'bensmodels/GConv_GNBR_nofeat2'

    # join all the file contents into a single dictionary
    results = {}
    for fname in os.listdir(directory):
        if fname[-4:] != '.txt': continue
        f = open(os.path.join(directory, fname), 'r')
        d = ast.literal_eval(f.read())
        results.update(d)
    
    # print out the mean recall
    recalls = []
    max_recall = float('-inf')
    max_recall_disease = None
    for key in sorted(results.keys()):
        # should be 0 to num_diseases - 1
        val = results[key]
        recalls.append(val[1]['recall'])
        if max_recall < recalls[-1]:
            max_recall = recalls[-1]
            max_recall_disease = key
    print(np.mean(recalls))
    print(max_recall, max_recall_disease)

    # print out weighted recall
    # edgelist_file = '../dataset_collection/PP-Decagon_ppi.csv'
    # edgelist_file = '../dataset_collection/Decagon_GNBR.csv'
    edgelist_file = '../dataset_collection/GNBR-edgelist.csv'
    data = ProcessData(edgelist_file, features=False)
    Y = data.Y
    # check total number of disease nodes
    total = 0
    for row in range(len(Y)):
        if np.any(Y.iloc[row] > 0):
            total += 1
    # check each column index
    weighted_recalls = []
    max_weighted_recall = float('-inf')
    max_weighted_recall_disease = None
    for ind, col in enumerate(Y.columns[:-2]):
        non_weighted_recall = recalls[ind]
        weight = np.sum(Y[col]) / total
        weighted_recalls.append(non_weighted_recall * weight)
        if max_weighted_recall < weighted_recalls[-1]:
            max_weighted_recall = weighted_recalls[-1]
            max_weighted_recall_disease = ind
    print(np.mean(weighted_recalls))
    print(max_weighted_recall, max_weighted_recall_disease)

    # print vals
    # import matplotlib.pyplot as plt
    # for i in range(10):
    #     curve = results[i][0][0]
    #     plt.plot(curve)
    # plt.show()

if __name__ == '__main__':
    main()