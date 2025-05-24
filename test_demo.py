import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import copy
from collections import defaultdict
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import numpy as np


def get_WL_feature(dataset, h, node_label =True):
    graph_num = len(dataset)
    label_num = dataset.num_node_features

    label = []

    n_nodes = 0
    for i in range(graph_num):
        n_nodes = n_nodes + dataset[i].num_nodes

    '''
    Initialize the labels 
    '''

    if node_label == True:
        for i in range(graph_num):
            mat = dataset[i].x
            label_i = []
            for j in range(dataset[i].num_nodes):
                 label_i.append(torch.nonzero(mat[j]).item())
            
            label.append(label_i)

    else: # for unlabeled graph
        for i in range(graph_num):
            edge_index = dataset[i].edge_index
            node_degrees = degree(edge_index[0]).tolist()
            for j in range(len(node_degrees)):
                node_degrees[j] = int(node_degrees[j])
           
            label.append(node_degrees)

    '''
    kernel 0-th iteration
    '''
    phi_init = np.zeros((graph_num, n_nodes))
    for i in range(graph_num):
        aux = np.bincount(label[i])
        phi_init[i, label[i]] = aux[label[i]]

    '''
    remove zero columns/features
    '''
    non_zero_columns = np.any(phi_init, axis=0)
    phi_init = phi_init[:, non_zero_columns]
    print(phi_init.shape)

    new_labels = copy.deepcopy(label)


    '''
    adjacency list
    '''
    lists = []
    for i in range(graph_num):
        adj_list = defaultdict(list)
        edge_index = dataset[i].edge_index
        adj = edge_index.t().tolist()
        for src, dst in edge_index.t().tolist():
            if dst not in adj_list[src]:
                adj_list[src].append(dst)
            if src not in adj_list[dst]:
                adj_list[dst].append(src)
        lists.append(adj_list)


    phi = np.zeros((graph_num, n_nodes))
    it = 0
    label_lookup = {}
    
    while it < h:
       
        label_counter = 0
        for i in range(graph_num):
            for j in range(dataset[i].num_nodes):
                neighbor_label = [label[i][k] for k in lists[i][j]]

                long_label = np.concatenate((np.atleast_1d(label[i][j]), np.sort(neighbor_label)))
                long_label_string = str(long_label)
                if long_label_string not in label_lookup:
                    label_lookup[long_label_string] = label_counter
                    new_labels[i][j] = label_counter
                    label_counter += 1
                else:
                    new_labels[i][j] = label_lookup[long_label_string]

            aux = np.bincount(new_labels[i])
           
            phi[i, new_labels[i]] += aux[new_labels[i]]
        label = copy.deepcopy(new_labels)
        #print(label)
        it = it + 1

    non_zero_columns = np.any(phi, axis=0)
    phi = phi[:, non_zero_columns]

    phi = np.concatenate((phi_init, phi), axis=1)
    return phi

if __name__ == '__main__':
    dataset = TUDataset(root='./data/MUTAG', name='MUTAG')
    print(get_WL_feature(dataset, h=10, node_label= True).shape)
    print(dataset.y)



















