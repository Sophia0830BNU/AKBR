import numpy as np
import torch
import os
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from ModelAtt_loss import Model
#from train import train
from trainwhole_loss import train
from ShortestPath import get_feature
from WeiL_multi import get_WL_feature

import argparse

# Argument parser for command-line arguments
parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
parser.add_argument('--dataset', type=str, default="MUTAG", help='Name of dataset (default: MUTAG)')
parser.add_argument('--device', type=int, default=0, help='Which GPU to use if any (default: 0)')
parser.add_argument('--epochs', type=int, default=350, help='Number of epochs to train (default: 350)')
parser.add_argument('--lr', type=float, default=0.004, help='Learning rate (default: 0.004)')
parser.add_argument('--feature_hid', type=int, default=50, help='Number of hidden units (default: 50)')
parser.add_argument('--layers', type=int, default=1, help='Number of cluster layers')
parser.add_argument('--num_mlp_layers', type=int, default=1, help='Number of MLP layers')
parser.add_argument('--method', type=str, default="WL", help='Kernel-based method')
parser.add_argument('--iteration_num', type=int, default=1, help='WL iteration number (1-10)')
parser.add_argument('--loss_alpha', type=float, default=0.001, help='Loss coefficient (default: 0.001)')

args = parser.parse_args()

# Hyperparameters
weight_decay = 5e-8
hidden_num1 = 50
hidden_num2 = 300

# Device configuration
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")


def pre_process(labels, dim):
    """Pre-process labels: adjust -1 to 0 and normalize to 0-indexed."""
    y = [0 if label == -1 else label for label in labels]
    if dim == max(labels):
        y = [label - 1 for label in y]
    return y


def set_random_seed(seed, deterministic=False):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Set seeds for reproducibility
seeds = [0, 2, 16, 42, 123, 456, 789, 101112, 222324, 252627]

# Load dataset
dataset = TUDataset(root=f'./data/{args.dataset}', name=args.dataset)

# Feature extraction
if args.method == "WL":
    feature_vector = torch.tensor(get_WL_feature(dataset, h=args.iteration_num, node_label=True)[1]).float().to(device)
elif args.method == "SP":
    feature_vector = torch.tensor(get_feature(dataset, node_label=True)).float().to(device)
feature_dim = feature_vector.shape[1]

# Pre-process labels
y = dataset.y
label_dim = len(set(y))
y = pre_process(y, label_dim)

# Number of graphs
graph_num = feature_vector.shape[0]

# Initialize random seed
set_random_seed(0, deterministic=True)

# Variables to track accuracy
test_mean_acc = [0] * 10

# Cross-validation
for i in range(10):
    kfold = KFold(n_splits=10, shuffle=True, random_state=seeds[i])
    splits = kfold.split(np.zeros(len(y)), y)

    test_acc = 0
    for train_index, test_index in splits:
        # Initialize and train the model
        net = Model(graph_num=graph_num, feature_hid=args.feature_hid, feature_dim1=feature_dim,
                    layers=args.layers, num_mlp_layers=args.num_mlp_layers,
                    hidden_dim1=hidden_num1, hidden_dim2=hidden_num2, n_labels=label_dim)
        test_re = train(net, feature_vector, y, args.epochs, train_index, test_index, device,
                        args.lr, weight_decay, alpha=args.loss_alpha)
        test_acc += test_re

    mean_acc = test_acc / 10
    test_mean_acc[i] = mean_acc
    print(f'Running times: {i}')
    print(f'Current time mean accuracy: {mean_acc:.6f}')

print(f'Test mean accuracy: {np.mean(test_mean_acc):.6f}')
test_mean_acc = [x * 100 for x in test_mean_acc]
print(f'Test accuracy standard deviation: {np.std(test_mean_acc):.6f}')
