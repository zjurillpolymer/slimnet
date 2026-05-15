"""仅模型结构，供 SLIMNet 导入使用"""
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F

class QM9_GNN(nn.Module):
    def __init__(self, in_channels=11, edge_dim=4, hidden=64, heads=4, num_tasks=4):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden, heads=heads, edge_dim=edge_dim, concat=True)
        self.gat2 = GATConv(hidden * heads, hidden, heads=heads, edge_dim=edge_dim, concat=True)
        self.lin = nn.Linear(hidden * heads, num_tasks)

    def forward(self, x, edge_index, edge_attr, batch, return_v=False):
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = global_mean_pool(x, batch)          # V_monomer [batch, 256]
        out = self.lin(x)
        if return_v:
            return out, x
        return out
