import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GATOptunaModel(nn.Module):
    def __init__(self, in_channels=8, edge_channels=4, descriptor_dim=48, hidden_channels=32, num_layers=4, heads=4, dropout=0.2):
        super(GATOptunaModel, self).__init__()
        torch.manual_seed(42)

        self.edge_encoder = nn.Linear(edge_channels, hidden_channels)
        self.dropout = dropout

        self.gat_layers = nn.ModuleList()
        input_dim = in_channels

        for i in range(num_layers):
            concat = True if i < num_layers - 1 else False
            out_dim = hidden_channels * heads if concat else hidden_channels
            self.gat_layers.append(GATConv(input_dim, hidden_channels, heads=heads, concat=concat, edge_dim=hidden_channels))
            input_dim = out_dim

        self.descriptor_net = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lin = nn.Linear(hidden_channels * 2, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_emb = self.edge_encoder(edge_attr)

        for gat in self.gat_layers:
            x = F.relu(gat(x, edge_index, edge_emb))
            x = F.dropout(x, p=self.dropout, training=self.training)

        graph_embedding = global_mean_pool(x, batch)
        descriptor_embedding = self.descriptor_net(data.descriptors)
        combined = torch.cat([graph_embedding, descriptor_embedding], dim=1)

        return torch.sigmoid(self.lin(combined)).view(-1)