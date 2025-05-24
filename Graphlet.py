import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import numpy as np
from itertools import combinations


def get_feature_GCGK(dataset):
    # generate graphlet of size 3
    G0 = nx.DiGraph()
    G0.add_nodes_from([1, 2, 3])
    G1 = nx.DiGraph()
    G1.add_nodes_from([1, 2, 3])
    G1.add_edges_from([(1, 2), (2, 1)])
    G2 = nx.DiGraph()
    G2.add_edges_from([(1, 2), (2, 3), (2, 1), (3, 2)])
    G3 = nx.DiGraph()
    G3.add_edges_from([(1, 2), (2, 3), (1, 3), (2, 1), (3, 2), (3, 1)])
    Gs = [G0, G1, G2, G3]

    graph_num = len(dataset)

    phi = []
    for i in range(graph_num):
        phi_i = [0, 0, 0, 0]
        G = to_networkx(dataset[i])
        nodes = list(G.nodes()) 
        node_triplets = combinations(nodes, 3)

        for triplet in node_triplets:
            subgraph = G.subgraph(triplet)
            #print(subgraph.edges)
            for i in range(4):
                if nx.is_isomorphic(subgraph, Gs[i]):
                    phi_i[i] += 1

        phi.append(phi_i)
  
    zero_cols = [all(row[i] == 0 for row in phi) for i in range(len(phi[0]))]


    phi = [[row[i] for i in range(len(phi[0])) if not zero_cols[i]] for row in phi]

    return phi
if __name__ == '__main__':
    dataset = TUDataset(root='./data/IMDB-BINARY', name='IMDB-BINARY')
    phi = get_feature_GCGK(dataset)
    print(phi)