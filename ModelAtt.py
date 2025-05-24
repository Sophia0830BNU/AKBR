import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(
        self,
        graph_num,
        feature_dims,
        feature_hid,
        hidden_dim1,
        hidden_dim2,
        n_labels,
        attn_type="channel"  # "channel", "sa", "none"
    ):
        super(Model, self).__init__()
        self.attn_type = attn_type
        self.alpha = 0.2

        self.feature_linears = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, feature_hid),
                nn.ReLU(),
                nn.Linear(feature_hid, dim)
            ) for dim in feature_dims
        ])

        if attn_type == "channel":
            self.attn_linears = nn.ModuleList([
                nn.Sequential(
                    nn.LeakyReLU(self.alpha),
                    nn.Softmax(dim=1)
                ) for _ in feature_dims
            ])
        elif attn_type == "sa":
            self.sa_linears = nn.ModuleList([
                nn.MultiheadAttention(embed_dim=dim, num_heads=2, dropout=0.1, batch_first=True)
                for dim in feature_dims
            ])

        self.Linear1 = nn.Linear(graph_num, hidden_dim1)
        self.Linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.Linear3 = nn.Linear(hidden_dim2, n_labels)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features):
        if isinstance(features, torch.Tensor):
            features = [features]

        kernel_matrices = []
        for i, x in enumerate(features):
            x_mean = torch.mean(x, dim=0, keepdim=True)
            x_hidden = self.feature_linears[i](x_mean)
            if self.attn_type == "channel":
                attn_weights = self.attn_linears[i](x_hidden)
                x_weighted = attn_weights * x
            elif self.attn_type == "sa":
                # Self-Attention: x shape [N, F] -> [1, N, F]
                x_input = x.unsqueeze(0)
                x_sa, _ = self.sa_linears[i](x_input, x_input, x_input)
                x_sa = torch.relu(x_sa)
                x_weighted = x_sa.squeeze(0) * x
            else:  # "none"
                x_weighted = x
            x_weighted = torch.relu(x_weighted)
            K = torch.matmul(x_weighted, x_weighted.T)
            kernel_matrices.append(K)

        K_fused = sum(kernel_matrices)
        x_embedded = torch.relu(self.Linear1(K_fused))
        x_embedded = torch.relu(self.Linear2(x_embedded))
        output = self.Linear3(x_embedded)
        return output

if __name__ == '__main__':
    input1 = torch.randn(2, 20).float()
    input2 = torch.randn(2, 10).float()
    for attn_type in ["channel", "sa", "none"]:
        print(f"Testing attn_type={attn_type}")
        net = Model(graph_num=2, feature_dims=[20, 10], feature_hid=4, hidden_dim1=4, hidden_dim2=4, n_labels=4, attn_type=attn_type)
        output = net([input1, input2])
        print("Output shape:", output.shape)
