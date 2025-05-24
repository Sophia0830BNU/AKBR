import numpy as np
import time
import argparse
import torch
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import KFold

from WeiL_multi import get_WL_feature
from ShortestPath import get_feature
from Graphlet import get_feature_GCGK
from RandomWalk import get_RW_feature
from ModelAtt import Model
from train import train

def pre_process(labels, dim):
    y = labels.clone()
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

def extract_feature(dataset, method, args, device):
    if method == "WL":
        return [torch.tensor(get_WL_feature(dataset, h=args.iteration_num, node_label=False)[1]).float().to(device)]
    elif method == "SP":
        return [torch.tensor(get_feature(dataset, node_label=True)).float().to(device)]
    elif method == "GC":
        return [torch.tensor(get_feature_GCGK(dataset)).float().to(device)]
    elif method == "RW":
        return [get_RW_feature(dataset, node_label=True).float().to(device)]
    elif method == "hybrid":
        feature_WL = torch.tensor(get_WL_feature(dataset, h=args.iteration_num, node_label=False)[1]).float().to(device)
        feature_SP = torch.tensor(get_feature(dataset, node_label=True)).float().to(device)
        feature_RW = get_RW_feature(dataset, node_label=True).float().to(device)
        return [feature_WL, feature_SP, feature_RW]
    else:
        raise ValueError(f"Unknown feature extraction method: {method}")

def main():
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="MUTAG")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--lr', type=float, default=0.004)
    parser.add_argument('--feature_hid', type=int, default=50)
    parser.add_argument('--method', type=str, default="WL", help='WL, SP, GC, RW, hybrid')
    parser.add_argument('--iteration_num', type=int, default=1)
    parser.add_argument('--attn_type', type=str, default="channel", choices=["channel", "sa", "none"],
                        help='Attention type: channel (default), sa (self-attention), none (no attention)')
    args = parser.parse_args()

    weight_decay = 5e-6
    hidden_num1 = 50
    hidden_num2 = 300
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    set_random_seed(0, deterministic=True)

    dataset = TUDataset(root=f'./data/{args.dataset}', name=args.dataset)
    start_time = time.time()
    feature_list = extract_feature(dataset, args.method, args, device)
    print(f"Feature(s) dim: {[f.shape[1] for f in feature_list]}")

    y = pre_process(dataset.y, torch.unique(dataset.y).size(0)).squeeze()
    graph_num = feature_list[0].shape[0]
    feature_dims = [f.shape[1] for f in feature_list]
    feature_list = [f.float().to(device) for f in feature_list]

    seeds = [0, 2, 16, 42, 123, 456, 789, 101112, 222324, 252627]
    test_mean_acc = []

    for i, seed in enumerate(seeds):
        kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
        splits = kfold.split(np.zeros(len(y)), y)
        test_acc = 0
        for fold, (train_index, test_index) in enumerate(splits):
            net = Model(
                graph_num=graph_num,
                feature_dims=feature_dims,
                feature_hid=args.feature_hid,
                hidden_dim1=hidden_num1,
                hidden_dim2=hidden_num2,
                n_labels=torch.unique(y).size(0),
                attn_type=args.attn_type   
            )
            ave_acc = train(net, feature_list, y, args.epochs, train_index, test_index, device, args.lr, weight_decay)
            print(f"Seed {seed} Fold {fold:02d} acc: {ave_acc:.4f}")
            test_acc += ave_acc
        mean_acc = test_acc / 10
        print(f"Seed {seed} mean acc: {mean_acc:.4f}")
        test_mean_acc.append(mean_acc)

    final_ave_acc = [x * 100 for x in test_mean_acc]
    print(f"Final 10x10 fold average acc: {np.mean(final_ave_acc):.4f} \t std: {np.std(final_ave_acc):.4f}")

if __name__ == '__main__':
    main()