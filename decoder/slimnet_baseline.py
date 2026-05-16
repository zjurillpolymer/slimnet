"""SLIMNet baseline (frozen encoder, no fine-tune)"""
from torch import nn
from data_loader import PI1070
import torch, sys, os, numpy as np
sys.path.append('base_model_molecule_encoder')
from Schnet_model_monomer import Schnet_monomer
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

# 根目录检测
_script_dir = os.path.dirname(os.path.abspath(__file__))
for _d in [_script_dir, os.path.dirname(_script_dir)]:
    if os.path.isdir(os.path.join(_d, 'base_model_molecule_encoder')):
        ROOT = _d; break
else:
    ROOT = _script_dir
os.environ['SLIMNET_ROOT'] = ROOT
torch.manual_seed(42)

# 数据
_csv = os.path.join(ROOT, 'data/PI1070.csv')
if not os.path.exists(_csv): _csv = os.path.join(ROOT, 'PI1070.csv')
dataset = PI1070(_csv)
np.random.seed(42); idx = np.random.permutation(len(dataset))
n = len(dataset)
train_ds = dataset[idx[:int(0.8*n)]]
valid_ds = dataset[idx[int(0.8*n):int(0.9*n)]]
test_ds = dataset[idx[int(0.9*n):]]
train_loader = DataLoader(train_ds, 32, shuffle=True)
valid_loader = DataLoader(valid_ds, 32)
test_loader = DataLoader(test_ds, 32)

y_train = torch.cat([b.y for b in train_loader])
y_mean = y_train.mean(dim=0); y_std = y_train.std(dim=0) + 1e-8
print(f'y_mean: {y_mean}\ny_std:  {y_std}')


class SlimNet(nn.Module):
    def __init__(self, v_dim=128, out_channels=3):
        super().__init__()
        self.linear1 = nn.Linear(v_dim, out_channels)
        self.linear2 = nn.Linear(v_dim + 3, out_channels)
        self.linear3 = nn.Linear(v_dim + 3, out_channels)
        self.mlp = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, out_channels))

    def forward(self, x, v_monomer):
        v_polymer = torch.cat([v_monomer, x.chain], dim=-1)
        v_polymer = F.dropout(v_polymer, 0.1, self.training)
        a = torch.sigmoid(self.linear1(v_monomer))
        b = F.softplus(self.linear2(v_polymer)) + 1e-6
        g = F.softplus(self.linear3(v_polymer)).clamp(max=5)
        return a * torch.clamp(b ** g, max=1e6) + self.mlp(x.order)


# 模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
encoder = Schnet_monomer(hidden_dim=128, n_layers=6)
_pt = os.path.join(ROOT, 'base_model_molecule_encoder/best_schnet.pt')
if not os.path.exists(_pt): _pt = os.path.join(ROOT, 'best_schnet.pt')
print(f'Loading: {_pt}')
encoder.load_state_dict(torch.load(_pt, map_location=device))
encoder.eval()
for p in encoder.parameters(): p.requires_grad = False  # 冻结
model = SlimNet(v_dim=128).to(device)
encoder.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


def train_epoch(loader):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            _, v = encoder(batch.z, batch.pos, batch.edge_index, batch.batch, return_v=True)
        optimizer.zero_grad()
        out = model(batch, v)
        loss = F.mse_loss(out, (batch.y - y_mean.to(device)) / y_std.to(device))
        loss.backward(); optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def valid_epoch(loader):
    model.eval()
    total_loss = 0
    all_p, all_t = [], []
    for batch in loader:
        batch = batch.to(device)
        _, v = encoder(batch.z, batch.pos, batch.edge_index, batch.batch, return_v=True)
        out = model(batch, v)
        loss = F.mse_loss(out, (batch.y - y_mean.to(device)) / y_std.to(device))
        total_loss += loss.item() * batch.num_graphs
        all_p.append((out * y_std.to(device) + y_mean.to(device)).cpu())
        all_t.append(batch.y.cpu())
    return total_loss / len(loader.dataset), torch.cat(all_p), torch.cat(all_t)


def main():
    epochs = 300
    train_h, val_h = [], []
    best_val = float('inf')
    for epoch in range(1, epochs + 1):
        tl = train_epoch(train_loader)
        vl, _, _ = valid_epoch(valid_loader)
        train_h.append(tl); val_h.append(vl)
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), os.path.join(ROOT, 'decoder/best_slimnet_baseline.pt'))
        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch:3d}/{epochs}  train={tl:.4f}  val={vl:.4f}')
    model.load_state_dict(torch.load(os.path.join(ROOT, 'decoder/best_slimnet_baseline.pt')))
    tl, p, t = valid_epoch(test_loader)
    print(f'\nTest  loss={tl:.4f}')
    from plot_results import plot_all
    plot_all(train_h, val_h, p.numpy(), t.numpy())

if __name__ == '__main__':
    main()
