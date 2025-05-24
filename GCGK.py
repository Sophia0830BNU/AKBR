import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import numpy as np
from itertools import combinations
import os

def read_to_list(path, l=True):
    lists = []
    with open(path, 'r') as file:
        for line in file:
            data = line.split()
            
            if l == True:
                data = [int(x) for x in data]
                lists.append(list(data))
            else:
                lists.append(int(data[0]))
            
           
    return lists
def get_feature_GCGK(dataset_name):
    # generate graphlet of size 3
    folder_path = './data'
    filepath = os.path.join(folder_path, dataset_name)

    adj_path = os.path.join(filepath, 'adj_list.txt')
    adj_list = read_to_list(adj_path)
    
    # adj_list = []
    # with open(adjpath, 'r') as file:
    #     for line in file:
    #         data = line.split()
    #         data = [int(x) for x in data]
    #         adj_list.append(list(data))

    graph_path = os.path.join(filepath, 'graph_indicator.txt')
    graph_inx = read_to_list(graph_path, l=False)
    
    node_path = os.path.join(filepath, 'node_label.txt')
    node_label = read_to_list(node_path, l=False)

    label = []
    label_tmp = []
    graph_ind = 0

    adj_lists =[]
    adj_list_tmp = []
    for i in range(len(node_label)):
        if graph_inx[i] == graph_ind:
            label_tmp.append(node_label[i])
            adj_list_tmp.append(adj_list[i])
            if i == len(node_label)-1:
                label.append(label_tmp)
                adj_lists.append(adj_list_tmp)
        else:
            label.append(label_tmp)
            label_tmp = []
            graph_ind += 1
            label_tmp.append(node_label[i])

            adj_lists.append(adj_list_tmp)
            adj_list_tmp = []
            adj_list_tmp.append(adj_list[i])
    graph_num = graph_inx[-1] +1
    
    Go = []
    for i in range(graph_num):
        G = nx.DiGraph()
        adj = adj_lists[i]
        for j in range(len(adj)):
            for k in adj[j]:
                #print(j,k)
                G.add_edge(j, k)
        Go.append(G)

    '''
    Define the graphlet of size 3
    '''
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

    graph_num = len(Go)

    phi = []
    for i in range(graph_num):
        phi_i = [0, 0, 0, 0]
        G = Go[i]
        nodes = list(G.nodes()) 
        node_triplets = combinations(nodes, 3)

        for triplet in node_triplets:
            #print(triplet)
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
    dataset_name = 'MUTAG1' 

    folder_path = './data'
    filepath = os.path.join(folder_path, dataset_name)

    y_path = os.path.join(filepath, 'graph_label.txt')   
    y = read_to_list(y_path, l=False)
    
    print(get_feature_GCGK(dataset_name))
    # print(y)