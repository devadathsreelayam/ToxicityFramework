import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCNModel(nn.Module):
    def __init__(self, in_channels, descriptor_dim, hidden_channels, num_layers, dropout):
        super(GCNModel, self).__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.descriptor_net = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lin = nn.Linear(hidden_channels * 2, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

        graph_emb = global_mean_pool(x, batch)
        desc_emb = self.descriptor_net(data.descriptors)
        out = torch.cat([graph_emb, desc_emb], dim=1)

        return torch.sigmoid(self.lin(out)).view(-1)