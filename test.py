import torch
from torch_geometric.utils import degree
import copy
from collections import defaultdict
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import numpy as np
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
        

def get_WL_feature(dataset_name, h, node_label =True):




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
    # print(adj_lists)
    # print(label)




    n_nodes = 0
    graph_num = graph_inx[-1] +1
   
    for i in range(graph_num):
        n_nodes = n_nodes + len(label[i])
    print(n_nodes)

    '''
    kernel 0-th iteration
    '''
    phi = np.zeros((graph_num, n_nodes))
    for i in range(graph_num):
        aux = np.bincount(label[i])
        phi[i, label[i]] = aux[label[i]]

    '''
    remove zero columns/features
    '''
    non_zero_columns = np.any(phi, axis=0)
    feature = []
    phi = phi[:, non_zero_columns]
    feature.append(phi)

    print(phi.shape)

    new_labels = copy.deepcopy(label)


    # '''
    # adjacency list for TUDataset
    # '''
    # lists = []
    # for i in range(graph_num):
    #     adj_list = defaultdict(list)
    #     edge_index = dataset[i].edge_index
    #     adj = edge_index.t().tolist()
    #     for src, dst in edge_index.t().tolist():
    #         if dst not in adj_list[src]:
    #             adj_list[src].append(dst)
    #         if src not in adj_list[dst]:
    #             adj_list[dst].append(src)
    #     lists.append(adj_list)
 


    it = 0
    while it < h:
        label_lookup = {}
        label_counter = 0
        phi_tmp = np.zeros((graph_num, n_nodes))
        for i in range(graph_num):
            for j in range(len(label[i])):
                neighbor_label = [label[i][k] for k in adj_lists[i][j]]

                long_label = np.concatenate((np.atleast_1d(label[i][j]), np.sort(neighbor_label)))
                long_label_string = str(long_label)
                if long_label_string not in label_lookup:
                    label_lookup[long_label_string] = label_counter
                    new_labels[i][j] = label_counter
                    label_counter += 1
                else:
                    new_labels[i][j] = label_lookup[long_label_string]

            aux = np.bincount(new_labels[i])
            phi_tmp[i, new_labels[i]] = aux[new_labels[i]]
        label = copy.deepcopy(new_labels)
        #print(label)
        it = it + 1

        non_zero_columns = np.any(phi_tmp, axis=0)

        phi_t = phi_tmp[:, non_zero_columns]
        #print(phi_t.shape)

        feature.append(phi_t)

        phi = np.concatenate((phi, phi_t), axis=1)

    return feature, phi

if __name__ == '__main__':

    dataset = 'PTC'
    res = get_WL_feature(dataset, h=10, node_label= True)
    print(res.shape)
    np.set_printoptions(threshold=np.inf)
    




















