import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from collections import defaultdict
from torch_geometric.utils import degree

def get_path_types(graph, node_label):
    edge_index = graph.edge_index
    #print(edge_index.shape)
    if node_label == True:
        mat = graph.x 
        node_labels = []
        for j in range(graph.num_nodes):
            node_labels.append(torch.nonzero(mat[j]).item())
    else:
        
        node_labels = degree(edge_index[0]).tolist()
        #print(node_labels)
        for j in range(len(node_labels)):
            node_labels[j] = int(node_labels[j])
    
  
    path_types = defaultdict(int)


    for i in range(edge_index.shape[1]):
        node_i = edge_index[0, i].item()
        node_j = edge_index[1, i].item()
        
        label_i = node_labels[node_i]  # Label of node i
        label_j = node_labels[node_j]  # Label of node j
        
        # Create path type, e.g., '1-2' or '2-1'
        path_type = f"{label_i}-{label_j}"
        
       
        path_types[path_type] += 1
    
    return path_types


# Function to count the path types for each graph in the dataset and convert to feature vector
def get_RW_feature(dataset, node_label=True):
    path_type_counts = []
    all_path_types = defaultdict(int)
    
    
    for graph in dataset:
        
        path_types = get_path_types(graph, node_label)
        path_type_counts.append(path_types)
        
       
        for path_type in path_types:
            all_path_types[path_type] += 1
    
   
    all_path_types_list = sorted(all_path_types.keys())  # Sort for consistency
    
   
    feature_vectors = []
    for path_types in path_type_counts:
        
        feature_vector = torch.zeros(len(all_path_types_list))  # Initialize a zero vector
        
        for i, path_type in enumerate(all_path_types_list):
            feature_vector[i] = path_types.get(path_type, 0)  # Set the count or 0 if path type is absent
        
        feature_vectors.append(feature_vector)
    feature_vectors = torch.cat(feature_vectors, dim=0).view(len(feature_vectors), -1)
    print(feature_vectors.shape)
        
    return feature_vectors
    
# Function to count the path types for each graph in the dataset
def count_path_types(dataset):
    path_type_counts = []

    for graph in dataset:
        # Get path type statistics for this graph
        path_types = get_path_types(graph, node_label=False)
        path_type_counts.append(path_types)
    
    return path_type_counts


    
    
if __name__ == '__main__':    
    
    dataset = TUDataset(root='/tmp/IMDB-BINARY', name='IMDB-BINARY')
    feature_vectors = get_RW_feature(dataset, node_label=False)
    # Print the feature vector of the first graph
    print(f"Feature vector of the first graph: {feature_vectors[0]}")
    # Count the path types for the entire dataset
    
    path_type_counts = count_path_types(dataset)

    # Print path types count for the first graph in the dataset
    print(f"Path types in the first graph: {path_type_counts[0]}")

    # If you want to see the distribution of all path types across the entire dataset
    all_path_types = defaultdict(int)
    for path_types in path_type_counts:
        for path_type, count in path_types.items():
            all_path_types[path_type] += count

    # Print total path types across the dataset
    print(f"Total path types in the dataset: {all_path_types}")





