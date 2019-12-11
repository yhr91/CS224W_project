import ast
import pandas as pd

#Code for assessing GCONV with Features
with open("bensmodels/GConv_with_features/10-10:15:59-100-thru-200.txt", "r") as data:
    dictionary = ast.literal_eval(data.read())
with open("bensmodels/GConv_with_features/10-10:46:55-200-thru-300.txt", "r") as data:
    dictionary1 = ast.literal_eval(data.read())
with open("bensmodels/GConv_with_features/10-11:17:52-300-thru-400.txt", "r") as data:
    dictionary2 = ast.literal_eval(data.read())
with open("bensmodels/GConv_with_features/10-11:48:49-400-thru-500.txt", "r") as data:
    dictionary3 = ast.literal_eval(data.read())
with open("bensmodels/GConv_with_features/10-12:19:46-500-thru-600.txt", "r") as data:
    dictionary4 = ast.literal_eval(data.read())
with open("bensmodels/GConv_with_features/10-16:00:47-518-thru-618.txt", "r") as data:
    dictionary5 = ast.literal_eval(data.read())
# for
# print(dictionary[0][0][4])
dictionary.update(dictionary1)
dictionary.update(dictionary2)
dictionary.update(dictionary3)
dictionary.update(dictionary4)
dictionary.update(dictionary5)
# df = pd.DataFrame.from_dict(dictionary)
# print(df)
# print(len(dictionary))
f1 = 0
recall = 0
for key in dictionary:
    recall += dictionary[key][1]['recall']
    f1 += dictionary[key][1]['f1']

f1 = f1/len(dictionary)
recall = recall/len(dictionary)
print("Results for GCONV with Features")
print("Average f1 is: "+str(f1))
print("Average recall is: "+str(recall))

#Code for assessing GCONV without Features
with open("bensmodels/GConv_without_features/10-21:44:11-100-thru-200.txt", "r") as data:
    dictionary = ast.literal_eval(data.read())
with open("bensmodels/GConv_without_features/10-22:16:47-200-thru-300.txt", "r") as data:
    dictionary1 = ast.literal_eval(data.read())
with open("bensmodels/GConv_without_features/10-22:49:33-300-thru-400.txt", "r") as data:
    dictionary2 = ast.literal_eval(data.read())
with open("bensmodels/GConv_without_features/10-23:21:32-400-thru-500.txt", "r") as data:
    dictionary3 = ast.literal_eval(data.read())
with open("bensmodels/GConv_without_features/10-23:53:11-500-thru-600.txt", "r") as data:
    dictionary4 = ast.literal_eval(data.read())
with open("bensmodels/GConv_without_features/10-23:58:44-518-thru-618.txt", "r") as data:
    dictionary5 = ast.literal_eval(data.read())
# for
# print(dictionary[0][0][4])
dictionary.update(dictionary1)
dictionary.update(dictionary2)
dictionary.update(dictionary3)
dictionary.update(dictionary4)
dictionary.update(dictionary5)
# df = pd.DataFrame.from_dict(dictionary)
# print(df)
# print(len(dictionary))
f1 = 0
recall = 0
for key in dictionary:
    recall += dictionary[key][1]['recall']
    f1 += dictionary[key][1]['f1']

f1 = f1/len(dictionary)
recall = recall/len(dictionary)
print("Results for GCONV without Features")
print("Average f1 is: "+str(f1))
print("Average recall is: "+str(recall))

#Code for assessing SAGEConv with Features
with open("bensmodels/SAGEConv_with_Features/10-16:36:33-100-thru-200.txt", "r") as data:
    dictionary = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_with_Features/10-17:05:39-200-thru-300.txt", "r") as data:
    dictionary1 = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_with_Features/10-17:34:38-300-thru-400.txt", "r") as data:
    dictionary2 = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_with_Features/10-18:03:37-400-thru-500.txt", "r") as data:
    dictionary3 = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_with_Features/10-18:32:29-500-thru-600.txt", "r") as data:
    dictionary4 = ast.literal_eval(data.read())
with open("bensmodels/SAGEConv_with_Features/10-18:37:41-518-thru-618.txt", "r") as data:
    dictionary5 = ast.literal_eval(data.read())
# for
# print(dictionary[0][0][4])
dictionary.update(dictionary1)
dictionary.update(dictionary2)
dictionary.update(dictionary3)
dictionary.update(dictionary4)
dictionary.update(dictionary5)
# df = pd.DataFrame.from_dict(dictionary)
# print(df)
# print(len(dictionary))
f1 = 0
recall = 0
for key in dictionary:
    recall += dictionary[key][1]['recall']
    f1 += dictionary[key][1]['f1']

f1 = f1/len(dictionary)
recall = recall/len(dictionary)
print("Results for SAGEConv with Features")
print("Average f1 is: "+str(f1))
print("Average recall is: "+str(recall))