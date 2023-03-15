from community import community_louvain
import pandas as pd
import warnings
import networkx as nx
import csv
import pickle
warnings.filterwarnings('ignore')

def louvain():
    edges = list(zip(data_info['Node1'].to_list(), data_info['Node2'].to_list()))
    G = nx.Graph()
    G.add_edges_from(edges)
    partition = community_louvain.best_partition(G)
    with open('partition.pickle', 'wb') as f:
        pickle.dump(partition, f)


data_info = pd.read_csv('train.csv')
predict_info = pd.read_csv('test.csv')

#louvain()

nodes1 = predict_info['Node1'].to_list()
nodes2 = predict_info['Node2'].to_list()
with open('partition.pickle', 'rb') as f:
    partition = pickle.load(f)
with open("hw2ans.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Id", "Category"])
    i = 0
    for x, y in zip(nodes1, nodes2):
        if partition.get(x) == partition.get(y):
            writer.writerow([i, 1])
            i += 1
        else:
            writer.writerow([i, 0])
            i += 1
