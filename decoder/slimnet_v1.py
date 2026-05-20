from torch import nn

from data_loader import PI1070
import torch
import sys
sys.path.append('base_model_molecule_encoder')
from Schnet_model_monomer import Schnet_monomer
import numpy as np
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import os

# 自动检测项目根目录（兼容扁平结构和子目录结构）
_script_dir = os.path.dirname(os.path.abspath(__file__))      # slimnet.py 所在目录
if os.path.isdir(os.path.join(_script_dir, 'base_model_molecule_encoder')):
    ROOT = _script_dir                                        # 扁平结构: /root/
elif os.path.isdir(os.path.join(os.path.dirname(_script_dir), 'base_model_molecule_encoder')):
    ROOT = os.path.dirname(_script_dir)                        # 子目录结构: .../Slimnet/
else:
    ROOT = _script_dir  # fallback
os.environ['SLIMNET_ROOT'] = ROOT
torch.set_float32_matmul_precision('high')
torch.manual_seed(42)

'''准备数据'''
_csv_path = os.path.join(ROOT, 'data/PI1070.csv')
if not os.path.exists(_csv_path):
    _csv_path = os.path.join(ROOT, 'PI1070.csv')  # 扁平结构兜底
dataset = PI1070(_csv_path)
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
trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

y_train = torch.cat([batch.y for batch in trainloader])
y_mean = y_train.mean(dim=0)
y_std = y_train.std(dim=0) + 1e-8
print(f'y_mean: {y_mean}')
print(f'y_std:  {y_std}')



class SlimNet(nn.Module):
    def __init__(self, v_dim=128, out_channels=3):
        super().__init__()
        self.linear1 = nn.Linear(v_dim, out_channels)           # α: V_monomer → 3
        self.linear2 = nn.Linear(v_dim + 3, out_channels)       # β: V_monomer + chain
        self.linear3 = nn.Linear(v_dim + 3, out_channels)       # γ: V_monomer + chain
        self.mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels)
        )

    def forward(self, x, v_monomer, return_components=False):
        # 替换 FAN 为 0（少数 GPU 上 SchNet 值域溢出）
        v_monomer = torch.nan_to_num(v_monomer, nan=0.0, posinf=50.0, neginf=-50.0)
        v_polymer = torch.cat([v_monomer, x.chain], dim=-1)
        v_polymer = F.dropout(v_polymer, p=0.1, training=self.training)
        alpha = torch.sigmoid(self.linear1(v_monomer))
        beta = F.softplus(self.linear2(v_polymer)).clamp(max=5)
        gamma = F.softplus(self.linear3(v_polymer)).clamp(max=2)

        attr_disordered = alpha * torch.clamp(beta ** gamma, max=100)
        attr_ordered = self.mlp(x.order)
        out = torch.nan_to_num(attr_disordered + attr_ordered)
        if return_components == 'components':
            return out, attr_disordered.detach(), attr_ordered.detach()
        if return_components == 'params':
            return out, alpha.detach(), beta.detach(), gamma.detach()
        return out


'''加载权重，训练模型'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

encoder = Schnet_monomer(hidden_dim=128, n_layers=6)
_enc_path = os.path.join(ROOT, 'base_model_molecule_encoder/best_schnet.pt')
if not os.path.exists(_enc_path):
    _enc_path = os.path.join(ROOT, 'best_schnet.pt')
print(f'Loading encoder from: {_enc_path}')
encoder.load_state_dict(torch.load(_enc_path, map_location=device))
model = SlimNet(v_dim=128).to(device)
encoder.to(device)

optimizer = torch.optim.Adam([
    {'params': model.parameters(), 'lr': 0.001},
    {'params': encoder.parameters(), 'lr': 1e-5},
], weight_decay=1e-4)


def train_epoch(loader):
    model.train()
    encoder.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        _, v_monomer = encoder(batch.z, batch.pos, batch.edge_index, batch.batch, return_v=True)
        # 清理 encoder 输出的 NaN（部分 GPU 上 SchNet 值域溢出）
        v_monomer = torch.nan_to_num(v_monomer, nan=0.0)
        optimizer.zero_grad()
        output = model(batch, v_monomer)
        y = batch.y
        loss_polymer = F.mse_loss(output, (y - y_mean.to(device)) / y_std.to(device))

        loss = loss_polymer
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def valid_epoch(loader):
    model.eval()
    encoder.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    all_ordered, all_disordered = [], []
    for batch in loader:
        batch = batch.to(device)
        out, v_monomer = encoder(batch.z, batch.pos, batch.edge_index, batch.batch, return_v=True)
        out, ordered, disordered = model(batch, v_monomer, return_components='components')
        y = batch.y
        loss = F.mse_loss(out, (y - y_mean.to(device)) / y_std.to(device))
        total_loss += loss.item() * batch.num_graphs
        preds = out * y_std.to(device) + y_mean.to(device)
        all_ordered.append(ordered.cpu())
        all_disordered.append(disordered.cpu())
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())
    ordered = torch.cat(all_ordered)
    disordered = torch.cat(all_disordered)
    ratio = (ordered.abs().mean() / (disordered.abs().mean() + 1e-8)).item()
    if not (0 < ratio < 1e6): ratio = 0
    return total_loss / len(loader.dataset), torch.cat(all_preds), torch.cat(all_targets), ratio


def main():
    epochs = 300
    train_hist, val_hist = [], []
    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(trainloader)
        val_loss, preds, targets, ratio = valid_epoch(validloader)
        train_hist.append(train_loss)
        val_hist.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(ROOT, 'decoder/best_slimnet.pt'))

        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch:3d}/{epochs}  '
                  f'train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  '
                  f'|ϕ_ordered|/|ϕ_disordered|={ratio:.3f}')

    # 测试集
    model.load_state_dict(torch.load(os.path.join(ROOT, 'decoder/best_slimnet.pt')))
    test_loss, preds, targets, ratio = valid_epoch(testloader)
    print(f'\nTest  loss={test_loss:.4f}  |ϕ_ordered|/|ϕ_disordered|={ratio:.3f}')

    # 提取标度律参数
    model.eval()
    all_a, all_b, all_g = [], [], []
    for batch in testloader:
        batch = batch.to(device)
        _, v_m = encoder(batch.z, batch.pos, batch.edge_index, batch.batch, return_v=True)
        _, a, b, g = model(batch, v_m, return_components='params')
        all_a.append(a), all_b.append(b), all_g.append(g)
    alpha = torch.cat(all_a); beta = torch.cat(all_b); gamma = torch.cat(all_g)
    print(f'\n标度律参数 (mean ± std per task):')
    names = ['thermal_diffusivity', 'dielectric_const', 'linear_expansion']
    for i, name in enumerate(names):
        print(f'  {name:25s}  α={alpha[:,i].mean():.3f}±{alpha[:,i].std():.3f}  '
              f'β={beta[:,i].mean():.3f}±{beta[:,i].std():.3f}  '
              f'γ={gamma[:,i].mean():.3f}±{gamma[:,i].std():.3f}')

    # 画图
    from plot_results import plot_all
    plot_all(train_hist, val_hist, preds.numpy(), targets.numpy())


if __name__ == '__main__':
    main()
