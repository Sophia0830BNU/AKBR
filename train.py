import torch
import time
import numpy as np
from torch import nn
from typing import List, Tuple, Union

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() / len(labels)

def evaluate(model: nn.Module, feature_list: Union[List[torch.Tensor], torch.Tensor], labels: torch.Tensor, loss_f, idx: torch.Tensor) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(feature_list)
    return loss_f(logits[idx], labels[idx]).item(), accuracy(logits[idx], labels[idx])

def calculate_max_moving_average(data: List[float], window_size: int = 1) -> Tuple[float, int]:
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

def train(
    model: nn.Module,
    feature_list: Union[List[torch.Tensor], torch.Tensor],
    labels: np.ndarray,
    epoches: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    device: torch.device,
    lr: float,
    weight_decay: float,
    loss_f=nn.CrossEntropyLoss(),
    verbose: bool = True
) -> Tuple[float, List[float]]:
    labels = torch.LongTensor(labels).to(device)
    train_idx = torch.LongTensor(train_idx).to(device)
    test_idx = torch.LongTensor(test_idx).to(device)
    model = model.to(device)
    train_label = labels[train_idx]
    test_label = labels[test_idx]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    acc_all = []
    for epoch in range(epoches):
        model.train()
        optimizer.zero_grad()
        logits = model(feature_list)
        loss = loss_f(logits[train_idx], train_label)
        acc = accuracy(logits[train_idx], train_label)
        loss_test, test_acc = evaluate(model, feature_list, labels, loss_f, test_idx)
        acc_all.append(test_acc)
        if verbose and (epoch % 10 == 0 or epoch == epoches - 1):
            print(f"Epoch {epoch:03d} | Train Acc: {acc:.4f} | Test Acc: {test_acc:.4f}")
        loss.backward()
        optimizer.step()
    best_ave_acc, _ = calculate_max_moving_average(acc_all)
    return best_ave_acc