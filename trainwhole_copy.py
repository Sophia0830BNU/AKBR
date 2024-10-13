import torch
import time
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
#from ModelAtt import Model
import math
import random
from torch import nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score



def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, feature_list, labels, loss_f, idx):
    model.eval()

    with torch.no_grad():
        logits= model(feature_list)

    return loss_f(logits[idx], labels[idx]).item(), accuracy(logits[idx], labels[idx])


# def load_array(data_arrays, batch_size, is_train=True):
#     dataset = data.TensorDataset(*data_arrays)
#     return data.DataLoader(dataset, batch_size, shuffle=is_train)

# def loss_K(K, labels, train_index):
#     #K = K[train_index, :][:, train_index]
#     loss = 0
#     for i in train_index:
#         for j in train_index:
#             if labels[i] == labels[j]:
#                 loss -= torch.log(K[i, j])
#             else:
#                 loss -= 1 - torch.log(K[i, j])
#     return loss

def calculate_max_moving_average(data, window_size=10):
    n = len(data)
    max_average = float('-inf')  
    max_index = -1  

    for i in range(n):
        if i + window_size <= n:
            window = data[i:i + window_size]
        else:
            window = data[i:n]
        
        average = sum(window) / len(window)

        if average > max_average:
            max_average = average
            max_index = i
    
    return max_average, max_index


def train(model, feature_list, labels, epoches, train_idx, test_idx, device, lr,
          weight_decay, writer, loss_f=nn.CrossEntropyLoss()):
    labels = torch.LongTensor(labels)
    labels = labels.to(device)
    train_idx = torch.LongTensor(train_idx).to(device)
    test_idx = torch.LongTensor(test_idx).to(device)

    #train_label = labels[train_idx].to(device)
    #model = nn.DataParallel(model)
    model = model.to(device)
    #model = nn.DataParallel(model, device_ids=[0,1])
    
    #features = features.to(device)

    # val_num = int(len(train_idx)/9)
    # valid_index = train_idx[-val_num:]
    # train_index =train_idx[:len(train_idx)-val_num]
    

    train_label = labels[train_idx].to(device)

    # valid_label = labels[valid_index].to(device)

    test_label = labels[test_idx].to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    #optimizer = torch.optim.SGD(model.parameters(), lr, weight_decay)

    #early_stopping = EarlyStopping(verbose=False)


    dur = []
    acc_10_epoch = []
    
    acc_all = []
    auc_all = []
    

    for epoch in range(epoches):
        
        #scheduler.step()
        model.train()

        t0 = time.time()

        optimizer.zero_grad()
        logits = model(feature_list)
        # for param in model.named_parameters():
        #     print(param)
        loss_0 = loss_f(logits[train_idx], train_label)
        #loss_1 = loss_K(K, labels, train_index)

        loss = loss_0 
        #writer.add_scalar('Loss/loss_1', loss, epoch)

        acc = accuracy(logits[train_idx], train_label)
        loss_test, test_acc = evaluate(model, feature_list, labels, loss_f, test_idx)
        y_scores = torch.softmax(logits[test_idx], dim=1)[:, 1].cpu().detach().numpy()
        y_test = test_label.cpu().numpy()
        #auc = roc_auc_score(y_test, y_scores)
        writer.add_scalar('Loss/ACC', test_acc, epoch)
        
        acc_all.append(test_acc)
        acc_10_epoch.append(test_acc)
        
        # if val_acc > best_acc:
        #     best_model = model
        #     best_acc = val_acc



        loss.backward()
        optimizer.step()
        #scheduler.step()

        dur.append(time.time() - t0)

        #print("Epoch {:05d} | Time(s) {:.4f} | Train_Acc {:.4f} | Test_Acc{:.4f} ".format(epoch, np.mean(dur), acc, test_acc))
    #model.load_state_dict(early_stopping.load_checkpoint())
    #model = best_model
    best_ave_acc, epoch = calculate_max_moving_average(acc_10_epoch)
    #best_ave_auc, epoch = calculate_max_moving_average(auc_all)
 
    #final_train, final_test = evaluate(model, feature_list, labels, loss_f, train_idx), evaluate(model, feature_list,  labels, loss_f, test_idx)
    # print(f'Train     : accuracy={acc:.4f}')
    # print(f'Test      : accuracy={best_ave_acc:.4f}')

    return best_ave_acc, acc_all




















