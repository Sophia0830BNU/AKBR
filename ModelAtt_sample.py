import torch
import torch.nn as nn
import torch.nn.init as init

class Model(nn.Module):
    def __init__(self, graph_num, feature_dim1, feature_hid, hidden_dim1, hidden_dim2, n_labels, attn_type="channel"):
        super(Model, self).__init__()
        self.attn_type = attn_type
        self.alpha = 0.2

        # Feature linear transformation
        # feature_dim1: [N, F]
        self.feature_linear = nn.Sequential(
            nn.Linear(feature_dim1, feature_hid),
            nn.ReLU(),
            nn.Linear(feature_hid, feature_dim1)
        )

        if attn_type == "channel":
            self.attn_linear = nn.Sequential(
                nn.LeakyReLU(self.alpha),
                nn.Softmax(dim=1)
            )
        elif attn_type == "sa":
            self.sa_linear = nn.MultiheadAttention(embed_dim=feature_dim1, num_heads=4, dropout=0.1, batch_first=True)
        # "none" 

        self.Linear1 = nn.Linear(graph_num, hidden_dim1)
        self.Linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.Linear3 = nn.Linear(hidden_dim2, n_labels)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input, phi_sample, features):
        # features: [N, F]
        x = input
        
        x_hidden = self.feature_linear(features)
        if self.attn_type == "channel":
            attn_weights = self.attn_linear(x_hidden)
            x_weighted = attn_weights * x
        elif self.attn_type == "sa":
            x_input = x.unsqueeze(0)  # [1, N, F]
            x_sa, _ = self.sa_linear(x_input, x_input, x_input)
            x_sa = torch.relu(x_sa)
            x_weighted = x_sa.squeeze(0) * x
        else:  # "none"
            x_weighted = x
        x_weighted = torch.relu(x_weighted)
        K = torch.matmul(x_weighted, phi_sample.T)

        x1 = torch.relu(self.Linear1(K))
        x2 = torch.relu(self.Linear2(x1))
        output = self.Linear3(x2)
        return output

if __name__ == '__main__':
    input = torch.randn(2, 20).float()
    phi_sample = torch.randn(2, 20).float()
    features = torch.randn(2, 20).float()
    for attn_type in ["channel", "sa", "none"]:
        print(f"Testing attn_type={attn_type}")
        net = Model(2, 20, 4, 4, 4, 4, attn_type=attn_type)
        output = net(input, phi_sample, features)
        print("Output shape:", output.shape)