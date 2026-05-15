from torch import nn
from torch_geometric.loader import DataLoader
import torch
from torch_geometric.datasets import QM9
import numpy as np
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F


'''划分train/valid/test三个集合'''
dataset = QM9(root='../data/QM9')
# print(dataset[0])
np.random.seed(42)
n=len(dataset)
idx = np.random.permutation(n)
n_train = int(0.8 * n)
n_valid = int(0.1 * n)
train_idx = idx[:n_train]
valid_idx = idx[n_train:n_train + n_valid]
test_idx = idx[n_train + n_valid:]

train_dataset = dataset[train_idx]
valid_dataset = dataset[valid_idx]
test_dataset = dataset[test_idx]

# 计算 4 个 target 的均值和标准差（训练集）
y_full = dataset._data.y  # [130791, 19]
y_train = y_full[train_idx][:, [0, 1, 3, 4]]
y_mean = y_train.mean(dim=0)  # [4]
y_std = y_train.std(dim=0) + 1e-8

# print(f'y_mean: {y_mean}')
# print(f'y_std:  {y_std}')

'''划分batch'''
trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# print(len(trainloader), len(validloader), len(testloader))


'''GNN模型，两层GAT+一层线性层'''
class QM9_GNN(nn.Module):
    def __init__(self,in_channels=11,edge_dim=4,hidden=64,heads=4,num_tasks=4):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden, heads=heads, edge_dim=edge_dim, concat=True)  # 64*4=256
        self.gat2 = GATConv(hidden * heads, hidden, heads=heads, edge_dim=edge_dim, concat=True)
        self.lin = nn.Linear(hidden * heads, num_tasks)

    def forward(self, x, edge_index, edge_attr, batch, return_v=False):
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = global_mean_pool(x, batch)          # → [batch, 256], 这就是 V_monomer
        out = self.lin(x)
        if return_v:
            return out, x
        return out


'''training'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = QM9_GNN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_epoch(loader):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y[:, [0, 1, 3, 4]]
        y_norm = (y - y_mean.to(device)) / y_std.to(device)
        loss = F.mse_loss(out, y_norm)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def valid_epoch(loader):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y[:, [0, 1, 3, 4]]
        y_norm = (y - y_mean.to(device)) / y_std.to(device)
        loss = F.mse_loss(out, y_norm)
        total_loss += loss.item() * batch.num_graphs
        preds = out * y_std.to(device) + y_mean.to(device)
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())
    return total_loss / len(loader.dataset), torch.cat(all_preds), torch.cat(all_targets)


def main():
    epochs = 50
    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(trainloader)
        val_loss, preds, targets = valid_epoch(validloader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')

        if epoch % 5 == 0 or epoch == 1:
            print(f'Epoch {epoch:2d}/{epochs}  '
                  f'train_loss={train_loss:.4f}  val_loss={val_loss:.4f}')

    # 测试集评估
    model.load_state_dict(torch.load('best_model.pt'))
    test_loss, preds, targets = valid_epoch(testloader)
    print(f'\nTest  loss={test_loss:.4f}')


if __name__ == '__main__':
    main()