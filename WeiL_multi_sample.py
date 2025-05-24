import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import copy
from collections import defaultdict
from torch_geometric.utils.convert import to_networkx
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset
import networkx as nx
import numpy as np


def get_labels(dataset, node_label=True):
    graph_num = len(dataset)
    #label_num = dataset.num_node_features

    label = []

    '''
    Initialize the labels 
    '''

    if node_label == True:
        for i in range(graph_num):
            mat = dataset[i].x
            #print(mat)          
            label_i = []
            for j in range(dataset[i].num_nodes):
                 label_i.append(torch.nonzero(mat[j]).item())
            
            label.append(label_i)

    else: # for unlabeled graph
        for i in range(graph_num):
            edge_index = dataset[i].edge_index
            node_degrees = degree(edge_index[0]).tolist()
            for j in range(dataset[i].num_nodes):
                if len(node_degrees)<=j:
                    node_degrees.append(0)
                else:
                    node_degrees[j] = int(node_degrees[j])

            label.append(node_degrees)
    return label

def get_WL_feature(dataset, h, node_label =True):
    graph_num = len(dataset)
    #label_num = dataset.num_node_features



    n_nodes = 0
    for i in range(graph_num):
        n_nodes = n_nodes + dataset[i].num_nodes

    '''
    Initialize the labels 
    '''
    label = get_labels(dataset, node_label)
    unique_labels = set()
    for row in label:
        for value in row:
            unique_labels.add(value)

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

    new_labels = copy.deepcopy(label)


    '''
    adjacency list for TUDataset
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

    
 

    label_dic = []
    it = 0
    while it < h:
        label_lookup = {}
        label_counter = 0
        phi_tmp = np.zeros((graph_num, n_nodes))
        for i in range(graph_num):
            #print(dataset[i].num_nodes)
            for j in range(dataset[i].num_nodes):
                neighbor_label = [label[i][j] for j in lists[i][j]]
                if neighbor_label:
                    long_label = np.concatenate((np.atleast_1d(label[i][j]), np.sort(neighbor_label)))
                else:
                    long_label = label[i][j]
                    
                long_label_string = str(long_label)
                if long_label_string not in label_lookup:
                    label_lookup[long_label_string] = label_counter # string => int
                    new_labels[i][j] = label_counter
                    label_counter += 1
                else:
                    new_labels[i][j] = label_lookup[long_label_string]

            aux = np.bincount(new_labels[i])
            phi_tmp[i, new_labels[i]] = aux[new_labels[i]]
        label = copy.deepcopy(new_labels)
        #print(label)
        
        label_dic.append(label_lookup)
        it = it + 1

        non_zero_columns = np.any(phi_tmp, axis=0)

        phi_t = phi_tmp[:, non_zero_columns]
        #print(phi_t.shape)

        feature.append(phi_t)

        phi = np.concatenate((phi, phi_t), axis=1)

    return label_dic, unique_labels, phi

def compute_graph_entropy(data):

    G = nx.Graph()
    G.add_edges_from(data.edge_index.t().numpy())
    degree_sequence = [d for _, d in G.degree()]
    unique, counts = np.unique(degree_sequence, return_counts=True)
    probabilities = counts / counts.sum()

    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy



def get_sample_graphs(dataset, sample_number):
    entropy_list = [(i, compute_graph_entropy(data)) for i, data in enumerate(dataset)]
    top_graphs = sorted(entropy_list, key=lambda x: x[1], reverse=True)[:sample_number]
    top_indices = [graph[0] for graph in top_graphs]
    top_graphs_data = [dataset[i] for i in top_indices]
    
    return top_graphs_data


def get_sample_WL_feature(dataset, sample_number, h, node_label):
    
    sample_graphs = get_sample_graphs(dataset, sample_number)  # sample algorithms
    sample_structures, labels, phi = get_WL_feature(sample_graphs, h, node_label) # excluding 0-th iteration
    #print(sample_structures)
    return sample_structures, labels, phi

def get_count_feature(batch_graphs, sample_structures, labels, h, node_label):
    len_sample = sum(len(sample) for sample in sample_structures )
    num_features = len_sample + len(labels)
    #print(num_features)
    phi = np.zeros((len(batch_graphs), num_features))
    label = []
    if node_label == True:
        for i in range(len(batch_graphs)):
            graph = batch_graphs[i]
            mat = graph.x
            label_i = []
            for j in range(graph.num_nodes):
                idx = torch.nonzero(mat[j]).item()
                label_i.append(idx)
                if idx < len(labels):
                    phi[i, idx] += 1
            label.append(label_i)
    else:
        for i in range(len(batch_graphs)):
            edge_index = batch_graphs[i].edge_index
            degree = torch.zeros(batch_graphs[i].num_nodes, dtype = torch.long)
            row, col = edge_index
            for node in row:
                degree[node] +=1
            for j in degree:
                phi[i, j] += 1
            label.append(degree)
                
        '''
    adjacency list for TUDataset
    '''
    
    lists = []
    for i in range(len(batch_graphs)):
        adj_list = defaultdict(list)
        edge_index = batch_graphs[i].edge_index
        adj = edge_index.t().tolist()
        for src, dst in adj:
            if dst not in adj_list[src]:
                adj_list[src].append(dst)
            if src not in adj_list[dst]:
                adj_list[dst].append(src)
        lists.append(adj_list)  
               
    it = 0 
    
    while it < h:
        for i in range(len(batch_graphs)):
            graph =batch_graphs[i]
            
            for j in range(graph.num_nodes):
                neighbor_label = [label[i][j] for j in lists[i][j]]
                
                if neighbor_label:
                    #print(label[i][j])
                    long_label = np.concatenate((np.atleast_1d(label[i][j]), np.sort(neighbor_label)))
                else:
                    long_label = label[i][j]
                long_label_string = str(long_label)
                
                if long_label_string in sample_structures[it]:
                    phi[i, sample_structures[it][long_label_string]+len(labels)] +=1
                    
        it = it +1
                    
                    
    return phi


if __name__ == '__main__':
    #dataset = TUDataset(root='./data/PROTEINS', name='PROTEINS')
    dataset = PygGraphPropPredDataset(name = 'ogbg-molhiv', root = 'dataset/')
    
    sample_structures, labels, phi_sample = get_sample_WL_feature(dataset, sample_number=1113, h=1, node_label= False)
    
    print(phi_sample.shape)
    batch_size = 32
    batch_graphs = [dataset[i:i+batch_size] for i in range(0, len(dataset), batch_size)]
    i = 0
    for batch in batch_graphs:
        phi = get_count_feature(batch, sample_structures, labels, h=1, node_label=False)
        print(phi.shape)
        i = i+len(batch)
    '''
    todo: training graphs features
    
    '''
    print(dataset.y)



















