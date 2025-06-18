import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool


class GINEModel(nn.Module):
    def __init__(self, in_channels=8, edge_channels=4, descriptor_dim=12,
                 hidden_channels=64, num_layers=3, dropout=0.3, out_channels=1):
        super(GINEModel, self).__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU()
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_channels, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU()
        )

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU()
            )
            self.convs.append(GINEConv(mlp))

        self.descriptor_mlp = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU()
        )

        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        descriptors = data.descriptors.squeeze(1)

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)

        x = global_add_pool(x, batch)
        desc_emb = self.descriptor_mlp(descriptors)

        combined = torch.cat([x, desc_emb], dim=1)
        out = self.final_mlp(combined)
        return torch.sigmoid(out).view(-1)
