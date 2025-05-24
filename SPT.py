import torch
from torch_geometric.datasets import TUDataset
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

def get_feature(dataset_name):    
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
    

    labels = []
    label_tmp = []
    graph_ind = 0

    adj_lists =[]
    adj_list_tmp = []
    for i in range(len(node_label)):
        if graph_inx[i] == graph_ind:
            label_tmp.append(node_label[i])
            adj_list_tmp.append(adj_list[i])
            if i == len(node_label)-1:
                labels.append(label_tmp)
                adj_lists.append(adj_list_tmp)
        else:
            labels.append(label_tmp)
            label_tmp = []
            graph_ind += 1
            label_tmp.append(node_label[i])

            adj_lists.append(adj_list_tmp)
            adj_list_tmp = []
            adj_list_tmp.append(adj_list[i])
    graph_num = graph_inx[-1] +1
    
    Go = []
    for i in range(graph_num):
        G = nx.Graph()
        adj = adj_lists[i]
        for j in range(len(adj)):
            for k in adj[j]:
                #print(j,k)
                G.add_edge(j, k)
        Go.append(G)

    Gs = []
    min_sp = 100000
    max_sp = 0

  
    
    for i in range(graph_num):
        G = nx.floyd_warshall_numpy(Go[i])
        G = np.where(G == np.inf, 0, G)
        G = np.where(G == np.nan, 0, G)
        if np.max(G) > max_sp:
            max_sp = np.max(G)
        elif np.min(G) < min_sp:
            min_sp = np.min(G)
        np.fill_diagonal(G, -1)
        Gs.append(G)
    #print(Gs[0])
    feature_len = int(max_sp - min_sp) + 1

    unique_labels = np.unique([label for i in range(len(labels)) for label in labels[i]])
    
    label_pairs = [(l1, l2) for l1 in unique_labels for l2 in unique_labels]
    label_pair_to_index = {pair: idx for idx, pair in enumerate(label_pairs)}
    
    feature = np.zeros((graph_num, feature_len, len(label_pairs)))
    
    for i in range(graph_num):


        label_i = labels[i]
            
        for j in range(len(Gs[i])):
            for k in range(len(Gs[i])):
                path_len = int(Gs[i][j][k])
                if path_len > 0:
                    start_label = label_i[j]  # Convert to scalar
                    end_label = label_i[k]    # Convert to scalar
                    label_index = label_pair_to_index[(start_label, end_label)]
                    feature[i][path_len][label_index] += 1
    
    #feature = feature / 2
    
    # Flatten the feature matrix to a 2D matrix (graph_num, feature_len * len(label_pairs))
    feature = feature.reshape(graph_num, -1)
    feature = feature[:, [not np.all(feature[:, i] == 0) for i in range(feature.shape[1])]]
    return feature

if __name__ == '__main__':
    dataset_name = 'MUTAG1' 

    folder_path = './data'
    filepath = os.path.join(folder_path, dataset_name)

    y_path = os.path.join(filepath, 'graph_label.txt')   
    y = read_to_list(y_path, l=False)
    feature_vector = get_feature(dataset_name)
    # phi = get_feature(dataset_name)
    # print(np.dot(phi, phi.T))
    print(get_feature(dataset_name).shape)
    # print(y)






