"""
Author @Ben
"""
import ast
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

#Code for assessing GCONV with Features
with open("bensmodels/GConv_PPI_feat/10-10:15:59-100-thru-200.txt", "r") as data:
    dictionary = ast.literal_eval(data.read())
with open("bensmodels/GConv_PPI_feat/10-10:46:55-200-thru-300.txt", "r") as data:
    dictionary1 = ast.literal_eval(data.read())
with open("bensmodels/GConv_PPI_feat/10-11:17:52-300-thru-400.txt", "r") as data:
    dictionary2 = ast.literal_eval(data.read())
with open("bensmodels/GConv_PPI_feat/10-11:48:49-400-thru-500.txt", "r") as data:
    dictionary3 = ast.literal_eval(data.read())
with open("bensmodels/GConv_PPI_feat/10-12:19:46-500-thru-600.txt", "r") as data:
    dictionary4 = ast.literal_eval(data.read())
with open("bensmodels/GConv_PPI_feat/10-16:00:47-518-thru-618.txt", "r") as data:
    dictionary5 = ast.literal_eval(data.read())
dictionary.update(dictionary1)
dictionary.update(dictionary2)
dictionary.update(dictionary3)
dictionary.update(dictionary4)
dictionary.update(dictionary5)
f1 = 0
recall = 0
PPI_G_conv_with_features_recalls = []
for key in dictionary:
    recall += dictionary[key][1]['recall']
    f1 += dictionary[key][1]['f1']
    PPI_G_conv_with_features_recalls.append(dictionary[key][1]['recall'])

f1 = f1/len(dictionary)
recall = recall/len(dictionary)
print("Results for GCONV with Features")
print("Average f1 is: "+str(f1))
print("Average recall is: "+str(recall))

#Code for assessing GCONV without Features
with open("bensmodels/GConv_PPI_nofeat/10-21:44:11-100-thru-200.txt", "r") as data:
    dictionary = ast.literal_eval(data.read())
with open("bensmodels/GConv_PPI_nofeat/10-22:16:47-200-thru-300.txt", "r") as data:
    dictionary1 = ast.literal_eval(data.read())
with open("bensmodels/GConv_PPI_nofeat/10-22:49:33-300-thru-400.txt", "r") as data:
    dictionary2 = ast.literal_eval(data.read())
with open("bensmodels/GConv_PPI_nofeat/10-23:21:32-400-thru-500.txt", "r") as data:
    dictionary3 = ast.literal_eval(data.read())
with open("bensmodels/GConv_PPI_nofeat/10-23:53:11-500-thru-600.txt", "r") as data:
    dictionary4 = ast.literal_eval(data.read())
with open("bensmodels/GConv_PPI_nofeat/10-23:58:44-518-thru-618.txt", "r") as data:
    dictionary5 = ast.literal_eval(data.read())
dictionary.update(dictionary1)
dictionary.update(dictionary2)
dictionary.update(dictionary3)
dictionary.update(dictionary4)
dictionary.update(dictionary5)
f1 = 0
recall = 0
PPI_G_conv_without_features_recalls = []
for key in dictionary:
    recall += dictionary[key][1]['recall']
    f1 += dictionary[key][1]['f1']
    PPI_G_conv_without_features_recalls.append(dictionary[key][1]['recall'])

f1 = f1/len(dictionary)
recall = recall/len(dictionary)
print("Results for GCONV without Features")
print("Average f1 is: "+str(f1))
print("Average recall is: "+str(recall))

#Code for assessing PPI SAGEConv with Features
with open("bensmodels/SAGEConv_PPI_feat/10-16:36:33-100-thru-200.txt", "r") as data:
    dictionary = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_PPI_feat/10-17:05:39-200-thru-300.txt", "r") as data:
    dictionary1 = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_PPI_feat/10-17:34:38-300-thru-400.txt", "r") as data:
    dictionary2 = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_PPI_feat/10-18:03:37-400-thru-500.txt", "r") as data:
    dictionary3 = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_PPI_feat/10-18:32:29-500-thru-600.txt", "r") as data:
    dictionary4 = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_PPI_feat/10-18:37:41-518-thru-618.txt", "r") as data:
    dictionary5 = ast.literal_eval(data.read())
dictionary.update(dictionary1)
dictionary.update(dictionary2)
dictionary.update(dictionary3)
dictionary.update(dictionary4)
dictionary.update(dictionary5)
f1 = 0
recall = 0
PPI_SAGEConv_with_features_recalls = []
for key in dictionary:
    recall += dictionary[key][1]['recall']
    f1 += dictionary[key][1]['f1']
    PPI_SAGEConv_with_features_recalls.append(dictionary[key][1]['recall'])
f1 = f1/len(dictionary)
recall = recall/len(dictionary)
print("Results for PPI SAGEConv with Features")
print("Average f1 is: "+str(f1))
print("Average recall is: "+str(recall))

#Code for assessing PPI SAGEConv without Features
with open("bensmodels/SAGEConv_PPI_nofeat/11-00:50:16-100-thru-200.txt", "r") as data:
    dictionary = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_PPI_nofeat/11-01:28:28-200-thru-300.txt", "r") as data:
    dictionary1 = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_PPI_nofeat/11-02:08:03-300-thru-400.txt", "r") as data:
    dictionary2 = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_PPI_nofeat/11-02:47:26-400-thru-500.txt", "r") as data:
    dictionary3 = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_PPI_nofeat/11-03:26:14-500-thru-600.txt", "r") as data:
    dictionary4 = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_PPI_nofeat/11-03:33:35-518-thru-618.txt", "r") as data:
    dictionary5 = ast.literal_eval(data.read())
dictionary.update(dictionary1)
dictionary.update(dictionary2)
dictionary.update(dictionary3)
dictionary.update(dictionary4)
dictionary.update(dictionary5)
f1 = 0
recall = 0
PPI_SAGEConv_without_features_recalls = []
for key in dictionary:
    recall += dictionary[key][1]['recall']
    f1 += dictionary[key][1]['f1']
    PPI_SAGEConv_without_features_recalls.append(dictionary[key][1]['recall'])
f1 = f1/len(dictionary)
recall = recall/len(dictionary)
print("Results for SAGEConv without Features")
print("Average f1 is: "+str(f1))
print("Average recall is: "+str(recall))

#Code for assessing GNBR SAGEConv with Features
with open("bensmodels/SAGEConv_GNBR_feat/11-03:38:48-100-thru-200.txt", "r") as data:
    dictionary = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_GNBR_feat/11-03:58:10-200-thru-300.txt", "r") as data:
    dictionary1 = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_GNBR_feat/11-04:17:28-300-thru-400.txt", "r") as data:
    dictionary2 = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_GNBR_feat/11-04:36:44-400-thru-500.txt", "r") as data:
    dictionary3 = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_GNBR_feat/11-04:55:56-500-thru-600.txt", "r") as data:
    dictionary4 = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_GNBR_feat/11-04:59:34-518-thru-618.txt", "r") as data:
    dictionary5 = ast.literal_eval(data.read())
dictionary.update(dictionary1)
dictionary.update(dictionary2)
dictionary.update(dictionary3)
dictionary.update(dictionary4)
dictionary.update(dictionary5)
f1 = 0
recall = 0
GNBR_SAGEConv_with_features_recalls = []
for key in dictionary:
    recall += dictionary[key][1]['recall']
    f1 += dictionary[key][1]['f1']
    GNBR_SAGEConv_with_features_recalls.append(dictionary[key][1]['recall'])
f1 = f1/len(dictionary)
recall = recall/len(dictionary)
print("Results for GNBR SAGEConv with Features")
print("Average f1 is: "+str(f1))
print("Average recall is: "+str(recall))

#Code for assessing GNBR SAGEConv without Features
with open("bensmodels/SAGEConv_GNBR_nofeat/11-01:54:11-100-thru-200.txt", "r") as data:
    dictionary = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_GNBR_nofeat/11-02:15:53-200-thru-300.txt", "r") as data:
    dictionary1 = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_GNBR_nofeat/11-02:38:23-300-thru-400.txt", "r") as data:
    dictionary2 = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_GNBR_nofeat/11-03:00:41-400-thru-500.txt", "r") as data:
    dictionary3 = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_GNBR_nofeat/11-03:22:01-500-thru-600.txt", "r") as data:
    dictionary4 = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_GNBR_nofeat/11-03:26:06-518-thru-618.txt", "r") as data:
    dictionary5 = ast.literal_eval(data.read())
dictionary.update(dictionary1)
dictionary.update(dictionary2)
dictionary.update(dictionary3)
dictionary.update(dictionary4)
dictionary.update(dictionary5)
f1 = 0
recall = 0
GNBR_SAGEConv_without_features_recalls = []
for key in dictionary:
    recall += dictionary[key][1]['recall']
    f1 += dictionary[key][1]['f1']
    GNBR_SAGEConv_without_features_recalls.append(dictionary[key][1]['recall'])
f1 = f1/len(dictionary)
recall = recall/len(dictionary)
print("Results for GNBR SAGEConv without Features")
print("Average f1 is: "+str(f1))
print("Average recall is: "+str(recall))

print(len(GNBR_SAGEConv_with_features_recalls))
print(len(GNBR_SAGEConv_without_features_recalls))

#Code for assessing PP-Decagonn + GNBR SAGEConv with Features
with open("bensmodels/SAGEConv_PPI_GNBR_nofeat/11-01:26:36-100-thru-200.txt", "r") as data:
    dictionary = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_PPI_GNBR_nofeat/11-02:05:04-200-thru-300.txt", "r") as data:
    dictionary1 = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_PPI_GNBR_nofeat/11-02:42:55-300-thru-400.txt", "r") as data:
    dictionary2 = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_PPI_GNBR_nofeat/11-03:20:38-400-thru-500.txt", "r") as data:
    dictionary3 = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_GNBR_feat/11-04:55:56-500-thru-600.txt", "r") as data:
    dictionary4 = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_GNBR_feat/11-04:59:34-518-thru-618.txt", "r") as data:
    dictionary5 = ast.literal_eval(data.read())
dictionary.update(dictionary1)
dictionary.update(dictionary2)
dictionary.update(dictionary3)
dictionary.update(dictionary4)
dictionary.update(dictionary5)
f1 = 0
recall = 0
GNBR_SAGEConv_with_features_recalls = []
for key in dictionary:
    recall += dictionary[key][1]['recall']
    f1 += dictionary[key][1]['f1']
    GNBR_SAGEConv_with_features_recalls.append(dictionary[key][1]['recall'])
f1 = f1/len(dictionary)
recall = recall/len(dictionary)
print("Results for GNBR SAGEConv with Features")
print("Average f1 is: "+str(f1))
print("Average recall is: "+str(recall))

#Code for assessing PP-Decagonn + GNBR GCN without Features
with open("bensmodels/GConv_PPI_GNBR_nofeat/11-01:07:32-100-thru-200.txt", "r") as data:
    dictionary = ast.literal_eval(data.read())
with open("bensmodels/GConv_PPI_GNBR_nofeat/11-02:30:46-200-thru-300.txt", "r") as data:
    dictionary1 = ast.literal_eval(data.read())
with open("bensmodels/GConv_PPI_GNBR_nofeat/11-03:54:18-300-thru-400.txt", "r") as data:
    dictionary2 = ast.literal_eval(data.read())
with open("bensmodels/GConv_PPI_GNBR_nofeat/11-05:17:38-400-thru-500.txt", "r") as data:
    dictionary3 = ast.literal_eval(data.read())
with open("bensmodels/GConv_PPI_GNBR_nofeat/11-06:41:07-500-thru-600.txt", "r") as data:
    dictionary4 = ast.literal_eval(data.read())
with open("bensmodels/GConv_PPI_GNBR_nofeat/11-06:56:10-518-thru-618.txt", "r") as data:
    dictionary5 = ast.literal_eval(data.read())
dictionary.update(dictionary1)
dictionary.update(dictionary2)
dictionary.update(dictionary3)
dictionary.update(dictionary4)
dictionary.update(dictionary5)
f1 = 0
recall = 0
PPI_GNBR_GConv_without_features_recalls = []
for key in dictionary:
    recall += dictionary[key][1]['recall']
    f1 += dictionary[key][1]['f1']
    PPI_GNBR_GConv_without_features_recalls.append(dictionary[key][1]['recall'])
f1 = f1/len(dictionary)
recall = recall/len(dictionary)
print("Results for PPI+GNBR GCN without Features")
print("Average f1 is: "+str(f1))
print("Average recall is: "+str(recall))

#Comparison of GNBR GConv with and without Uniprot Features
GNBR_SAGEConv_without_features_recalls = GNBR_SAGEConv_without_features_recalls
GNBR_SAGEConv_with_features_recalls_array = np.array(GNBR_SAGEConv_with_features_recalls)
GNBR_SAGEConv_without_features_recalls_array = np.array(GNBR_SAGEConv_without_features_recalls)
difference = (GNBR_SAGEConv_with_features_recalls_array - GNBR_SAGEConv_without_features_recalls_array)
count = 0
indices_with_1_score = []
for item in difference:
    if item == 1.0:
        indices_with_1_score.append(count)
    count += 1
print(indices_with_1_score)
difference = -1*np.sort(-1*difference)
print(max(difference))
fig = plt.figure(figsize=[16, 5])
fig.tight_layout()
plt.bar(np.arange(len(difference)), difference, edgecolor=None)
plt.xlim(0, 520)
plt.ylim(-1.1, 1.1)
plt.title("Difference in Recall for GNBR GraphSage with and without UniProt Features across "
          "Diseases")
# plt.show()
fig.savefig("GNBR_SAGEConv_with_vs_without_features.png")
plt.clf()

#Comparison of PPI GConv with and without Uniprot Features
PPI_G_conv_without_features_recalls = PPI_G_conv_without_features_recalls[:-1]
PPI_G_conv_with_features_recalls_array = np.array(PPI_G_conv_with_features_recalls)
PPI_G_conv_without_features_recalls_array = np.array(PPI_G_conv_without_features_recalls)
difference = (PPI_G_conv_with_features_recalls_array - PPI_G_conv_without_features_recalls_array)
count = 0
indices_with_1_score = []
for item in difference:
    if item == 1.0:
        indices_with_1_score.append(count)
    count += 1
print(indices_with_1_score)
difference = -1*np.sort(-1*difference)
print(max(difference))
fig = plt.figure(figsize=[16, 5])
fig.tight_layout()
plt.bar(np.arange(len(difference)), difference, edgecolor=None)
plt.xlim(0,520)
plt.ylim(-1.1, 1.1)
plt.title("Difference in Recall for PP-Decagon GCN with and without UniProt Features across "
          "Diseases")
# plt.show()
fig.savefig("PPI_GCN_with_vs_without_features.png")
plt.clf()

#Comparison of PPI SAGEConv with and without Uniprot Features
PPI_SAGEConv_without_features_recalls = PPI_SAGEConv_without_features_recalls[:-1]
PPI_SAGEConv_with_features_recalls_array = np.array(PPI_SAGEConv_with_features_recalls)
PPI_SAGEConv_without_features_recalls_array = np.array(PPI_SAGEConv_without_features_recalls)
difference = (PPI_SAGEConv_with_features_recalls_array -
              PPI_SAGEConv_without_features_recalls_array)
count = 0
indices_with_1_score = []
for item in difference:
    if item == 1.0:
        indices_with_1_score.append(count)
    count += 1
print(indices_with_1_score)
difference = -1*np.sort(-1*difference)
print(max(difference))
fig = plt.figure(figsize=[16, 5])
fig.tight_layout()
plt.bar(np.arange(len(difference)), difference, color='g', edgecolor=None)
plt.xlim(0,520)
plt.ylim(-1.1, 1.1)
plt.title("Difference in Recall for PP-Decagon GraphSage with and without UniProt Features across "
          "Diseases")
# plt.show()
fig.savefig("PPI_SAGEConv_with_vs_without_features.png")
plt.clf()

#Comparison of using GNBR vs PPI (in combination with SAGEConv and  Uniprot Features)
difference = (PPI_SAGEConv_with_features_recalls_array -
              GNBR_SAGEConv_with_features_recalls_array[:-1])
count = 0
indices_with_1_score = []
for item in difference:
    if item == 1.0:
        indices_with_1_score.append(count)
    count += 1
print(indices_with_1_score)
difference = -1*np.sort(-1*difference)
print(max(difference))
fig = plt.figure(figsize=[16, 5])
fig.tight_layout()
plt.bar(np.arange(len(difference)), difference, color='m', edgecolor=None)
plt.xlim(0,520)
plt.ylim(-1.1, 1.1)
plt.title("Difference in Recall for PP-Decagon vs GNBR (in combination with SAGEConv and UniProt "
          "Features) across Diseases")
# plt.show()
fig.savefig("PPI_vs_GNBR_SAGEConv_with_features.png")
plt.clf()

#Comparison of using GConv vs. SAGEConv PPI without features
difference = (PPI_SAGEConv_without_features_recalls_array[:-1] -
              PPI_G_conv_without_features_recalls)
count = 0
indices_with_1_score = []
for item in difference:
    if item == 1.0:
        indices_with_1_score.append(count)
    count += 1
print(indices_with_1_score)
difference = -1*np.sort(-1*difference)
print(max(difference))
fig = plt.figure(figsize=[16, 5])
fig.tight_layout()
plt.bar(np.arange(len(difference)), difference, edgecolor=None)
plt.xlim(0,520)
plt.ylim(-1.1, 1.1)
plt.title("Difference in Recall for GraphSage vs GCN using PP-Decagon without UniProt Features "
          "across Diseases")
# plt.show()
fig.savefig("PPI_GCN_vs_SAGEConv_without_features.png")
plt.clf()

#Comparison of PPI+GNBR vs PPI alone GConv without Uniprot Features
PPI_GNBR_GConv_without_features_recalls = PPI_GNBR_GConv_without_features_recalls[:-1]
PPI_GNBR_GConv_without_features_recalls_array = np.array(PPI_GNBR_GConv_without_features_recalls)
difference = (PPI_GNBR_GConv_without_features_recalls_array -
              PPI_G_conv_without_features_recalls_array)
count = 0
indices_with_1_score = []
for item in difference:
    if item == 1.0:
        indices_with_1_score.append(count)
    count += 1
print(indices_with_1_score)
difference = -1*np.sort(-1*difference)
print(max(difference))
fig = plt.figure(figsize=[16, 5])
fig.tight_layout()
plt.bar(np.arange(len(difference)), difference, color='m', edgecolor=None)
plt.xlim(0, 520)
plt.ylim(-1.1, 1.1)
plt.title("Difference in Recall for PPI+GNBR vs PPI alone using GCN without features across "
          "Diseases")
# plt.show()
fig.savefig("PPI-GNBR_vs_PPI_GCN_without_features.png")
plt.clf()
