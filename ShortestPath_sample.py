import numpy as np
import networkx as nx
from torch_geometric.datasets import TUDataset
import torch
from torch_geometric.utils import degree

def to_networkx(data):
    G = nx.Graph()
    edge_index = data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0, i], edge_index[1, i])
    return G

def get_feature(dataset, node_label):
    Gs = []
    min_sp = float('inf')
    max_sp = 0
    graph_num = len(dataset)
    
    for i in range(graph_num):
        G = nx.floyd_warshall_numpy(to_networkx(dataset[i]))
        G = np.where(np.isinf(G), 0, G)  # Correct replacement of inf values
        G = np.where(np.isnan(G), 0, G)  # Correct replacement of nan values
        max_sp = max(max_sp, np.max(G[G > 0]))
        min_sp = min(min_sp, np.min(G[G > 0]))
        np.fill_diagonal(G, -1)
        Gs.append(G)
    
    feature_len = int(max_sp) + 1
    if node_label == True:
        labels = []
        for i in range(graph_num):
            mat = dataset[i].x
                
            label_i = []
            for j in range(dataset[i].num_nodes):
                label_i.append(torch.nonzero(mat[j]).item())
                
            labels.append(label_i)
    else: # for unlabeled graph
        labels = []
        for i in range(graph_num):
            edge_index = dataset[i].edge_index
            node_degrees = degree(edge_index[0]).tolist()
            for j in range(len(node_degrees)):
                node_degrees[j] = int(node_degrees[j])
           
            labels.append(node_degrees)

    
    unique_labels = np.unique([label for i in range(len(labels)) for label in labels[i]])
    
    label_pairs = [(l1, l2) for l1 in unique_labels for l2 in unique_labels]
    label_pair_to_index = {pair: idx for idx, pair in enumerate(label_pairs)}
    
    feature = np.zeros((graph_num, feature_len, len(label_pairs)))
    #feature = np.zeros((graph_num, feature_len))
    
    for i in range(graph_num):
        graph = to_networkx(dataset[i])
        # mat = dataset[i].x  # Assuming the node labels are stored in 'x'
        # label_i = []
        # for j in range(dataset[i].num_nodes):
        #     label_i.append(torch.nonzero(mat[j]).item())
        label_i = labels[i]
        for j in range(len(Gs[i])):
            for k in range(len(Gs[i])):
                path_len = int(Gs[i][j][k])
                if path_len > 0:
                    start_label = label_i[j]  # Convert to scalar
                    end_label = label_i[k]    # Convert to scalar
                    label_index = label_pair_to_index[(start_label, end_label)]
                    feature[i][path_len][label_index] += 1
                    feature[i][path_len] += 1
    
    #feature = feature / 2
    
    #Flatten the feature matrix to a 2D matrix (graph_num, feature_len * len(label_pairs))
    feature = feature.reshape(graph_num, -1)
    zero_columns = np.where(np.all(feature == 0, axis=0))[0]
    feature = feature[:, ~np.all(feature == 0, axis=0)]
    
    return feature_len, label_pair_to_index, zero_columns, feature

def get_count_feature(batch, feature_len, label_pair,  zero_columns, node_label):
    graph_num = len(batch)
    if node_label == True:
        labels = []
        for i in range(graph_num):
            mat = batch[i].x
                
            label_i = []
            for j in range(batch[i].num_nodes):
                label_i.append(torch.nonzero(mat[j]).item())
                
            labels.append(label_i)
    else: # for unlabeled graph
        labels = []
        for i in range(graph_num):
            edge_index = batch[i].edge_index
            node_degrees = degree(edge_index[0]).tolist()
            for j in range(len(node_degrees)):
                node_degrees[j] = int(node_degrees[j])
           
            labels.append(node_degrees)
   
    phi = np.zeros((len(batch), feature_len, len(label_pair)))
    
    for k in range(len(batch)):
        graph = batch[k]
        G = nx.floyd_warshall_numpy(to_networkx(graph))
        G = np.where(np.isinf(G), 0, G)  # Correct replacement of inf values
        G = np.where(np.isnan(G), 0, G)  # Correct replacement of nan values
        np.fill_diagonal(G, -1)
        label_k = labels[k]
        for i in range(len(G)):
            for j in range(len(G)):
                if G[i][j] >0 and G[i][j] < feature_len:
                    start_label = label_k[i]
                    end_label = label_k[j]
                    pair = (start_label, end_label)
                    if pair in label_pair:
                        phi[k][int(G[i][j])][label_pair[pair]] += 1
    feature = phi.reshape(len(batch), -1)
    feature_new = np.delete(feature, zero_columns, axis=1)

    return feature_new        

def get_sample_SP_feature(dataset, sample_number, node_label=True):
    sample_graphs = dataset[:sample_number]
    feature_len, label_pair, zero_columns, phi = get_feature(sample_graphs, node_label)
    
    return feature_len, label_pair, zero_columns, phi
    

if __name__ == '__main__':
    dataset = TUDataset(root='./data/MUTAG', name='MUTAG')
    feature_len, label_pair, zero_columns, phi_sample = get_sample_SP_feature(dataset, sample_number=100, node_label= True)
    print(phi_sample.shape)
    batch_size = 32
    batch_graphs = [dataset[i:i+batch_size] for i in range(0, len(dataset), batch_size)]
    i = 0
    for batch in batch_graphs:
        phi = get_count_feature(batch, feature_len, label_pair, zero_columns, node_label=True)
        print(phi.shape)






