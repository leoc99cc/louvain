import pandas as pd
import numpy as np
import csv
import warnings
import math
warnings.filterwarnings('ignore')

graph = pd.read_csv('2022-ntust-practice-of-social-media-analytics-hw2/train.csv')
predict = pd.read_csv('2022-ntust-practice-of-social-media-analytics-hw2/test.csv')
graph = graph.to_numpy()
predict = predict.to_numpy()
adj_matrix = np.zeros((1048575, 1048575), int)
for x in graph:
    adj_matrix[x[0]][x[1]] = 1
    adj_matrix[x[1]][x[0]] = 1

with open("hw2ans.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Id", "Category"])
    i = 0
    for y in predict:
        interection = adj_matrix[y[1]] & adj_matrix[y[2]]
        union = adj_matrix[y[1]] | adj_matrix[y[2]]
        jaccard = np.sum(interection) / np.sum(union)
        adar = 0
        j = 0
        for z in interection:
            if z == 1:
                adar += 1 / (np.log(np.sum(adj_matrix[j]))+1E-16)
                j += 1
        if adar > 0.9 or adj_matrix[y[0]][y[1]] == 1:
            writer.writerow([i, 1])
            i += 1
        else:
            writer.writerow([i, 0])
            i += 1


