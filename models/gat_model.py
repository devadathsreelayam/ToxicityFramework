import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class EnhancedGAT(torch.nn.Module):
    def __init__(self, in_channels=8, edge_channels=4, descriptor_dim=48, hidden_channels=32, out_channels=1):
        super(EnhancedGAT, self).__init__()
        torch.manual_seed(42)

        # Edge feature transformation
        self.edge_encoder = nn.Linear(edge_channels, hidden_channels)

        # Graph attention layers
        self.gat1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
        self.gat2 = GATConv(hidden_channels * 4, hidden_channels, heads=4, concat=True, edge_dim=hidden_channels)
        self.gat3 = GATConv(hidden_channels * 4, hidden_channels, heads=4, concat=True, edge_dim=hidden_channels)
        self.gat4 = GATConv(hidden_channels * 4, hidden_channels, heads=1, concat=False, edge_dim=hidden_channels)

        # Descriptor network
        self.descriptor_net = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Combined prediction
        self.lin = nn.Linear(hidden_channels * 2, out_channels)  # 2x for graph + descriptor features

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Encode edge features
        edge_emb = self.edge_encoder(edge_attr)

        # Graph processing
        x = F.relu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)

        x = F.relu(self.gat2(x, edge_index, edge_emb))
        x = F.dropout(x, p=0.2, training=self.training)

        x = F.relu(self.gat3(x, edge_index, edge_emb))
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.gat4(x, edge_index, edge_emb)
        graph_embedding = global_mean_pool(x, batch)

        # Descriptor processing
        # descriptor_embedding = self.descriptor_net(data.descriptors.view(-1, data.descriptors.size(1)))
        descriptor_embedding = self.descriptor_net(data.descriptors)

        # Combine features
        combined = torch.cat([graph_embedding, descriptor_embedding], dim=1)

        return torch.sigmoid(self.lin(combined)).view(-1)