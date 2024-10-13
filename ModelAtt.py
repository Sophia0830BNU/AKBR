import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, graph_num, feature_hid, feature_dim1, hidden_dim1,hidden_dim2, n_labels):
        super(Model, self).__init__()

        self.alpha = 0.2
        self.Linear00_WL = nn.Linear(feature_dim1, feature_hid)
        self.Linear01_WL = nn.Linear(feature_hid, feature_dim1)

        # self.Linear00_SP = nn.Linear(feature_dim2, feature_hid)
        # self.Linear01_SP = nn.Linear(feature_hid, feature_dim2)
        # self.W = nn.Parameter(torch.empty(size=(graph_num, 1)))
        # nn.init.xavier_uniform_(self.W.data, gain=1.44)
        # self.Linear01 = nn.Linear(feature_dim, feature_hid)
        # self.Linear02 = nn.Linear(graph_num, 1)
        self.Linear1 = nn.Linear(graph_num, hidden_dim1)
        self.Linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.Linear3 = nn.Linear(hidden_dim2, n_labels)
        # self.Linear4 = nn.Linear(100, n_labels)

        #self.W = nn.Parameter(torch.rand(graph_num, graph_num))

        self.leakrelu = nn.LeakyReLU(self.alpha)
        self.softmax = nn.Softmax(dim=1)
        # self.BN = nn.BatchNorm1d(graph_num)
        #self.dropout = nn.Dropout(0.5)


    # def normalize_kernel_matrix(self, K):
    #     diag = torch.diag(K)
    #     diag_sqrt = torch.sqrt(diag)
        
    #     # 外积计算归一化因子
    #     norm_factor = torch.outer(diag_sqrt, diag_sqrt)
        
    #     # 防止除以零
    #     norm_factor[norm_factor == 0] = 1e-8
        
    #     # 对 Kernel 矩阵进行归一化
    #     K_normalized = K / norm_factor
        
    #     return K_normalized

    def forward(self, x):
        # x:X_WL


        # x1 = x.unsqueeze(1)
        # print(x1.shape)
        # x0 = self.Linear00(x)
        # x1 = torch.relu(x0)
        # x1 = self.Linear01(x1.T)
        #print(x.shape)
        
        x1 = torch.sum(x, dim=0, keepdim=True)/x.shape[0]
        x2 = self.Linear00_WL(x1)
        x3 = torch.relu(x2)
        x4_WL = self.Linear01_WL(x3)


        # x1 = torch.sum(x_SP, dim=0, keepdim=True)/x_SP.shape[0]
        # x2 = self.Linear00_SP(x1)
        # x3 = torch.relu(x2)
        # x4_SP = self.Linear01_SP(x3)
        #print(x4.shape)
        # k = self.Linear01(x)
        # w = torch.matmul(q, k.T)
        # x1 = self.Linear02(w)

        #weight = torch.relu(x4)
        attention= self.softmax(self.leakrelu(x4_WL))
        #attention =self.dropout(attention)
        #print(attention)
        #attention =self.dropout(attention)
        
        x5 = attention * x

        x5_WL = torch.relu(x5)

        # attention= self.softmax(self.leakrelu(x4_SP))
        # attention =self.dropout(attention)
        # x5 = attention * x_SP
        # x5_SP = torch.relu(x5)

        #x5 = F.normalize(x5, p=2, dim=1)
        #x5 = torch.relu(x5)


        #x4 =x3.view(x.size(0), -1)
        #x5 = torch.sum(x5 ** 2, dim=1)
        #x5 = x5[:, None] + x5[None, :]
        K_WL = torch.matmul(x5_WL, x5_WL.T)
        #K_WL = self.normalize_kernel_matrix(K_WL)
        #K_WL = F.normalize(K_WL, p=2, dim=1)
        # K_SP = torch.matmul(x5_SP, x5_SP.T)
        # K_SP = F.normalize(K_SP, p=2, dim=1)
        # K = self.W*K_WL + (1-self.W)*K_SP
        

        #X = torch.matmul(K, x5)
        #K = 1/(1+K)
        #K = -2*gamma*(1-K)
        #X = torch.exp(-K)
        #print(K.shape)
        # K = self.BN(K)
        #K =  torch.exp(-K)
        x55 = self.Linear1(K_WL)
        x5 = torch.relu(x55)
        #x5 = self.dropout(x5)
        x6 = self.Linear2(x5)
        x6 = torch.relu(x6)
        x7 = self.Linear3(x6)
        # x7 = torch.relu(x6)
        # x8 = self.Linear4(x7)
        return x7


if __name__ == '__main__':
    input = torch.randn(2, 20).float()
    net = Model(2, 4, 20, 4, 4, 4)
    print(input)
    output= net(input)
    print(output)
    print(output.shape)

