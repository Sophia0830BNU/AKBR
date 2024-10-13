import numpy as np
import os
import random
import time
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score

from WeiL_multi import get_WL_feature
from ShortestPath import get_feature
from Graphlet import get_feature_GCGK
from ModelAtt import Model
from trainwhole_copy import train

# Initialize TensorBoard writer
writer = SummaryWriter('./runs')

# Define hyperparameters and configuration
parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
parser.add_argument('--dataset', type=str, default="MUTAG", help='name of dataset (default: MUTAG)')
parser.add_argument('--device', type=int, default=0, help='which GPU to use if any (default: 0)')
parser.add_argument('--epochs', type=int, default=350, help='number of epochs to train (default: 350)')
parser.add_argument('--lr', type=float, default=0.004, help='learning rate (default: 0.004)')
parser.add_argument('--feature_hid', type=int, default=50, help='number of hidden units (default: 50)')
parser.add_argument('--method', type=str, default="WL", help='The kernel based method')
parser.add_argument('--iteration_num', type=int, default=1, help='The WL iteration_num 1-10')

args = parser.parse_args()

# Set constants
weight_decay = 5e-6
hidden_num1 = 50
hidden_num2 = 300

# Set device for PyTorch
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

def pre_process(labels, dim):
    """Preprocess the labels to adjust for 0 indexing."""
    y = labels.clone()  # Use clone to avoid modifying original tensor
    y[y == -1] = 0  # Set -1 labels to 0
    if dim == max(labels):
        y -= 1  # Adjust labels to be 0-indexed
    return y

def set_random_seed(seed, deterministic=False):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Load dataset
dataset = TUDataset(root=f'./data/{args.dataset}', name=args.dataset)

# Feature extraction
start_time = time.time()
if args.method == "WL":
    feature_vector = torch.tensor(get_WL_feature(dataset, h=args.iteration_num, node_label=False)[1]).float().to(device)
elif args.method == "SP":
    feature_vector = torch.tensor(get_feature(dataset, node_label=True)).float().to(device)
elif args.method == "GC":
    feature_vector = torch.tensor(get_feature_GCGK(dataset)).float().to(device)

# Preprocess labels
y = pre_process(dataset.y, torch.unique(dataset.y).size(0)).squeeze()

graph_num = feature_vector.shape[0]

# Set random seed for reproducibility
set_random_seed(0, deterministic=True)

final_results = []
test_ave_acc = []
test_ave_auc = []
seeds = list(range(10))

# Perform k-fold cross-validation
for seed in seeds:
    test_fold_acc = []
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    splits = kfold.split(np.zeros(len(y)), y)

    test_acc, test_auc = 0, 0
    
    for fold, (train_index, test_index) in enumerate(splits):
        net = Model(graph_num=graph_num, feature_hid=args.feature_hid,
                    feature_dim1=feature_vector.shape[1], hidden_dim1=hidden_num1,
                    hidden_dim2=hidden_num2, n_labels=torch.unique(y).size(0))

        ave_acc, acc_all = train(net, feature_vector, y, args.epochs, train_index, test_index,
                                 device, args.lr, weight_decay, writer)

        test_fold_acc.append(acc_all)
        print(f"fold: {fold:03d}\t ave_auc: {ave_acc:.4f}")
        test_acc += ave_acc

    # Record results
    ave_test = np.mean(test_fold_acc, axis=0)
    test_ave_acc.append(test_acc / 10)
    test_ave_auc.append(test_auc / 10)

    end_time = time.time()
    print(f"A single run time: {end_time - start_time:.4f}")

# Compute final metrics
final_ave_acc = [x * 100 for x in test_ave_acc]
final_ave_auc = [x * 100 for x in test_ave_auc]

print(f"Final 10 times 10 fold average acc: {np.mean(final_ave_acc):.4f} \t std: {np.std(final_ave_acc):.4f}")
print(f"Final 10 times 10 fold average auc: {np.mean(final_ave_auc):.4f} \t std: {np.std(final_ave_auc):.4f}")
