import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, graph_num, feature_hid, feature_dim1, hidden_dim1, hidden_dim2, n_labels):
        super(Model, self).__init__()
        
        # Define hyperparameters and layers
        self.alpha = 0.2
        self.Linear00_WL = nn.Linear(feature_dim1, feature_hid)
        self.Linear01_WL = nn.Linear(feature_hid, feature_dim1)
        
        # Main layers for graph embedding
        self.Linear1 = nn.Linear(graph_num, hidden_dim1)
        self.Linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.Linear3 = nn.Linear(hidden_dim2, n_labels)
        
        # Activation functions
        self.leakrelu = nn.LeakyReLU(self.alpha)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Feature processing for WL kernel
        x_mean = torch.mean(x, dim=0, keepdim=True)
        x_hidden = torch.relu(self.Linear00_WL(x_mean))
        x_transformed_WL = self.Linear01_WL(x_hidden)

        # Attention mechanism
        attention_weights = self.softmax(self.leakrelu(x_transformed_WL))
        x_weighted = attention_weights * x
        x_weighted_WL = torch.relu(x_weighted)

        # Compute kernel matrix for WL kernel
        K_WL = torch.matmul(x_weighted_WL, x_weighted_WL.T)

        # Graph embedding layers
        x_embedded = torch.relu(self.Linear1(K_WL))
        x_embedded = torch.relu(self.Linear2(x_embedded))
        output = self.Linear3(x_embedded)

        return output

if __name__ == '__main__':
    input = torch.randn(2, 20).float()
    net = Model(2, 4, 20, 4, 4, 4)
    print("Input:", input)
    output = net(input)
    print("Output:", output)
    print("Output Shape:", output.shape)
