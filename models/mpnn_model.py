# models/mpnn_model.py
import torch
from torch import nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.nn.inits import glorot, zeros


class MPNNConv(MessagePassing):
    def __init__(self, node_dim, edge_dim):
        super().__init__(aggr='add')
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        input = torch.cat([x_j, edge_attr], dim=1)
        return self.edge_mlp(input)

    def update(self, aggr_out, x):
        input = torch.cat([x, aggr_out], dim=1)
        return self.update_mlp(input)


class MPNNModel(nn.Module):
    def __init__(self, in_channels, edge_channels, descriptor_dim, hidden_channels=64, num_layers=3, dropout=0.3):
        super().__init__()
        self.node_encoder = nn.Linear(in_channels, hidden_channels)
        self.edge_encoder = nn.Linear(edge_channels, hidden_channels)

        self.layers = nn.ModuleList([
            MPNNConv(hidden_channels, hidden_channels) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Linear(hidden_channels + descriptor_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)

        for conv in self.layers:
            x = conv(x, data.edge_index, edge_attr)

        x = global_mean_pool(x, data.batch)
        out = torch.cat([x, data.descriptors], dim=1)
        return self.readout(self.dropout(out)).view(-1)
