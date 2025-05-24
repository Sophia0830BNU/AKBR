import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import KFold
import argparse
import time

from WeiL_multi_sample import *
from ShortestPath import get_feature
from Graphlet import get_feature_GCGK
from ModelAtt_sample import *
from train import *


parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
parser.add_argument('--dataset', type=str, default="MUTAG", help='name of dataset (default: MUTAG)')
parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
parser.add_argument('--epochs', type=int, default=350, help='number of epochs to train (default: 350)')
parser.add_argument('--lr', type=float, default=0.004, help='learning rate (default: 0.004)')
parser.add_argument('--feature_hid', type=int, default=50, help='number of hidden units (default: 50)')
parser.add_argument('--method', type=str, default="WL", help='The kernel based method')
parser.add_argument('--iteration_num', type=int, default=1, help='The WL iteration_num 1-10')
parser.add_argument('--degree_as_tag', action="store_true", help='degree_as_tag ')
parser.add_argument('--sample_number', type=int, default=100, help='sampling number of graphs')
parser.add_argument('--attn_type', type=str, default="channel", choices=["channel", "sa", "none"],
                    help='Attention type: channel (default), sa (self-attention), none (no attention)')
args = parser.parse_args()

# ===================== hyper parameters =====================
CSE = nn.CrossEntropyLoss()
weight_decay = 5e-6
hidden_num1 = 50
hidden_num2 = 100
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
batch_size = 32

# ===================== process functions =====================
def MAE(scores, targets):
    MAE = F.l1_loss(scores, targets)
    return MAE.detach().item()

def pre_process(labels, dim):
    y = labels.clone() if torch.is_tensor(labels) else np.copy(labels)
    y[y == -1] = 0
    if dim == max(labels):
        y -= 1
    return y

def set_random_seed(seed, deterministic=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def calculate_max_moving_average(data, window_size=10):
    n = len(data)
    max_average = float('-inf')
    max_index = -1
    for i in range(n):
        window = data[i:i + window_size] if i + window_size <= n else data[i:n]
        average = sum(window) / len(window)
        if average > max_average:
            max_average = average
            max_index = i
    return max_average, max_index

# ===================== data loading =====================
set_random_seed(0, deterministic=True)
dataset = TUDataset(root=f'./data/{args.dataset}', name=args.dataset)
start = time.time()

sample_structures, sample_init_labels, phi_sample = get_sample_WL_feature(
    dataset, sample_number=args.sample_number, h=args.iteration_num, node_label=not args.degree_as_tag)
phi_sample = torch.tensor(phi_sample).float().to(device)
print("phi_sample shape:", phi_sample.shape)

len_sample = sum(len(sample) for sample in sample_structures)
num_features = len_sample + len(sample_init_labels)

y = dataset.y
label_dim = torch.unique(y).size(0)
y = pre_process(y, label_dim)


final_results = []
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

for i, seed in enumerate(seeds):
    test_fold_acc = []
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    splits = kfold.split(np.zeros(len(y)), y)
    fold = 0

    for train_index, test_index in splits:
        model = Model(
            args.sample_number, num_features, args.feature_hid,
            hidden_dim1=hidden_num1, hidden_dim2=hidden_num2,
            n_labels=label_dim, attn_type=args.attn_type
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
        train_labels = y[train_index]
        test_labels = y[test_index]
        train_graphs, test_graphs = dataset[train_index], dataset[test_index]
        batch_graphs = [train_graphs[i:i + batch_size] for i in range(0, len(train_graphs), batch_size)]
        batch_labels = [train_labels[i:i + batch_size] for i in range(0, len(train_labels), batch_size)]
        batch_test_graphs = [test_graphs[i:i + batch_size] for i in range(0, len(test_graphs), batch_size)]
        batch_test_labels = [test_labels[i:i + batch_size] for i in range(0, len(test_labels), batch_size)]

        phi_train_ave, phi_test_ave = [], []
        phi_train, phi_test = [], []
        for train_graphs in batch_graphs:
            phi = get_count_feature(train_graphs, sample_structures, sample_init_labels, args.iteration_num, node_label=not args.degree_as_tag)
            phi = torch.tensor(phi).float().to(device)
            phi_train.append(phi)
            phi_train_ave.append(torch.sum(phi, dim=0, keepdim=True) / phi.shape[0])
        for test_graphs in batch_test_graphs:
            phi = get_count_feature(test_graphs, sample_structures, sample_init_labels, args.iteration_num, node_label=not args.degree_as_tag)
            phi = torch.tensor(phi).float().to(device)
            phi_test.append(phi)
            phi_test_ave.append(torch.sum(phi, dim=0, keepdim=True) / phi.shape[0])

        feature_train = sum(phi_train_ave) / len(phi_train_ave)
        feature_test = sum(phi_test_ave) / len(phi_test_ave)
        acc_all = []

        for epoch in range(args.epochs):
            model.train()
            for phi, batch_label in zip(phi_train, batch_labels):
                batch_label = torch.tensor(batch_label).to(device)
                optimizer.zero_grad()
                logits = model(phi, phi_sample, feature_train)
                loss = CSE(logits, batch_label)
                acc = accuracy(logits, batch_label)
                loss.backward()
                optimizer.step()

            acc_test = 0
            n_batch = 0
            model.eval()
            for phi, batch_label in zip(phi_test, batch_test_labels):
                phi = torch.tensor(phi).float().to(device)
                batch_label = torch.tensor(batch_label).to(device)
                with torch.no_grad():
                    logits = model(phi, phi_sample, feature_test)
                acc = accuracy(logits, batch_label)
                acc_test += acc
                n_batch += 1
            acc_test = acc_test / n_batch
            acc_all.append(acc_test)
            print(f"Epoch {epoch+1:03d} | Test Acc: {acc_test:.4f}")
        test_fold_acc.append(acc_all)
        acc_fold, _ = calculate_max_moving_average(acc_all)
        print(f"fold: {fold:03d}\t ave_acc:{acc_fold:.4f}")
        fold += 1

    end = time.time()
    print(f"A single run time:{(end - start):.4f}")

    ave_test = np.mean(test_fold_acc, axis=0)
    max_mean = np.max(ave_test)
    max_index = np.argmax(ave_test)
    print(f"Best Acc {max_mean:.4f} \t Best Epoch {max_index}")
    final_results.append(max_mean)

final_acc = [x * 100 for x in final_results]
print("Final 10 times average acc: {:.4f} \t std:{:.4f}".format(np.mean(final_acc), np.std(final_acc)))