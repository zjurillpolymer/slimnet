from torch import nn
from torch_geometric.loader import DataLoader
import torch
from torch_geometric.datasets import QM9
from torch_geometric.nn import global_mean_pool
import numpy as np
import torch.nn.functional as F
import os

_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 临时 patch PyG 的 QM9 处理代码，跳过 RDKit 无法解析的分子
try:
    import torch_geometric.datasets.qm9 as _qm9_mod
    with open(_qm9_mod.__file__, 'r') as f:
        _c = f.read()
    _old = 'if i in skip:\n                continue\n\n            N = mol.GetNumAtoms()'
    _new = 'if i in skip:\n                continue\n            if mol is None:\n                continue\n\n            N = mol.GetNumAtoms()'
    if _old in _c:
        with open(_qm9_mod.__file__, 'w') as f:
            f.write(_c.replace(_old, _new))
        print('[Patch] QM9: skip None molecules')
except: pass

'''划分train/valid/test三个集合'''
if __name__ == '__main__':
    dataset = QM9(root=os.path.join(_SCRIPT_DIR, 'data/QM9'))
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

    y_full = dataset._data.y
    y_train = y_full[train_idx][:, [0, 1, 2, 3]]
    y_mean = y_train.mean(dim=0)
    y_std = y_train.std(dim=0) + 1e-8

    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class ShiftedSoftplus(nn.Module):
    """SchNet 使用的激活函数: ln(0.5 + 0.5*exp(x))，数值稳定版本"""
    def forward(self, x):
        return F.softplus(x) - 0.693147  # ln(2)



class RBFExpansion(nn.Module):
    def __init__(self,n_rbf=32,cutoff=4.0):
        super().__init__()
        centers=torch.linspace(0,cutoff,n_rbf)
        self.register_buffer('centers',centers)
        self.gamma=0.5/(cutoff/n_rbf)**2

    def forward(self,dist):
        d=dist
        c=self.centers.unsqueeze(0)
        return torch.exp(-self.gamma*torch.pow(d-c,2))


class SchNetConv(nn.Module):
    '''连续滤波卷积层'''
    def __init__(self,hidden_dim,rbf_dim=32):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(rbf_dim, hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)


    def forward(self, h, x, edge_index, rbf):
        '''x:positions of atoms;h:characters of atoms'''
        n=h.size(0) # h:(num_atoms,hidden)
        i,j=edge_index #edge_index (2,num_edges)
        rel_pos=x[i]-x[j] # 位置向量
        dist=rel_pos.norm(dim=-1,keepdim=True) # L2范数
        rbf_feat=rbf(dist)

        W=self.filter_net(rbf_feat) # 将向量滤波，得到权重矩阵(num_edges,hidden_dim)
        m_ij=h[j]*W # m_ij (num_edges,hidden_dim)

        aggr=torch.zeros(n,h.size(1),device=h.device,dtype=h.dtype) # (n,hidden)
        aggr=aggr.index_add(0,i,m_ij) # 聚合周围原子特征 (n,hidden)
        deg=torch.zeros(n,device=h.device,dtype=h.dtype) # (n,）
        deg=deg.index_add(0,i,torch.ones(m_ij.size(0),device=h.device,dtype=h.dtype)) # 计算自己的化学键数目
        aggr=aggr/deg.clamp(min=1).view(-1,1) #求平均

        h_new=self.update_net(h+aggr)#聚合周围原子+自己
        return self.norm(h_new)



class OutputBlock1(nn.Module):
    """逐原子 MLP + 求和预测极化率"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), ShiftedSoftplus(),
            nn.Linear(hidden_dim, hidden_dim), ShiftedSoftplus(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h, batch=None):
        E_atom = self.net(h).squeeze(-1)
        if batch is not None: # 并行计算
            n_graphs = batch.max().item() + 1
            energy = torch.zeros(n_graphs, device=h.device, dtype=h.dtype)
            energy = energy.index_add(0, batch, E_atom)
        else:
            energy = E_atom.sum()
        return energy


class OutputBlock2(nn.Module):
    '''逐层mlp+预测分子的homo和lumo'''
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), ShiftedSoftplus(),
            nn.Linear(hidden_dim, hidden_dim), ShiftedSoftplus(),
            nn.Linear(hidden_dim,1)
        )

    def forward(self, h, batch=None):
        atom_homo = self.net(h).squeeze(-1)
        if batch is not None:
            n_graphs = batch.max().item() + 1
            mol_homo = torch.zeros(n_graphs, device=h.device, dtype=h.dtype)
            mol_homo = mol_homo.index_add(0, batch, atom_homo) / n_graphs
        else:
            mol_homo = atom_homo.mean(dim=0)
        return mol_homo

class OutputBlock4(nn.Module):
    """逐原子 MLP + 求和预测极化率"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), ShiftedSoftplus(),
            nn.Linear(hidden_dim, hidden_dim), ShiftedSoftplus(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h, batch=None):
        E_atom = self.net(h).squeeze(-1)
        if batch is not None: # 并行计算
            n_graphs = batch.max().item() + 1
            energy = torch.zeros(n_graphs, device=h.device, dtype=h.dtype)
            energy = energy.index_add(0, batch, E_atom)
        else:
            energy = E_atom.sum()
        return energy


class OutputBlock3(nn.Module):
    '''逐层mlp+预测分子的homo和lumo'''

    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), ShiftedSoftplus(),
            nn.Linear(hidden_dim, hidden_dim), ShiftedSoftplus(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h, batch=None):
        atom_lumo = self.net(h).squeeze(-1)
        if batch is not None:
            n_graphs = batch.max().item() + 1
            mol_lumo = torch.zeros(n_graphs, device=h.device, dtype=h.dtype)
            mol_lumo = mol_lumo.index_add(0, batch, atom_lumo) / n_graphs
        else:
            mol_lumo = atom_lumo.mean(dim=0)
        return mol_lumo



class Schnet_monomer(nn.Module):
    def __init__(self, num_atom_types=10, hidden_dim=128, n_layers=6,num_tasks=3):
        super().__init__()
        self.atom_embed = nn.Embedding(num_atom_types + 1, hidden_dim)
        self.rbf = RBFExpansion(n_rbf=32, cutoff=4.0)
        self.convs = nn.ModuleList([
            SchNetConv(hidden_dim) for _ in range(n_layers)
        ])
        self.alpha = OutputBlock1(hidden_dim)
        self.homo=OutputBlock2(hidden_dim)
        self.lumo=OutputBlock3(hidden_dim)
        self.mu=OutputBlock4(hidden_dim)

    def forward(self, z, pos, edge_index, batch=None, return_v=False):
        h = self.atom_embed(z)
        for conv in self.convs:
            h = conv(h, pos, edge_index, self.rbf)
        v = global_mean_pool(h, batch)  # [batch, hidden_dim] V_monomer
        alpha = self.alpha(h, batch).unsqueeze(-1)
        homo = self.homo(h, batch).unsqueeze(-1)
        lumo = self.lumo(h, batch).unsqueeze(-1)
        mu = self.mu(h, batch).unsqueeze(-1)
        out = torch.cat([mu, alpha, homo, lumo], dim=-1)
        if return_v:
            return out, v
        return out


'''训练'''
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = Schnet_monomer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_epoch(loader):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.z, batch.pos, batch.edge_index, batch.batch)
        y = batch.y[:, [0, 1, 2, 3]]
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
        out = model(batch.z, batch.pos, batch.edge_index, batch.batch)
        y = batch.y[:, [0, 1, 2, 3]]
        y_norm = (y - y_mean.to(device)) / y_std.to(device)
        loss = F.mse_loss(out, y_norm)
        total_loss += loss.item() * batch.num_graphs
        preds = out * y_std.to(device) + y_mean.to(device)
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())
    return total_loss / len(loader.dataset), torch.cat(all_preds), torch.cat(all_targets)


def main():
    epochs = 100
    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(trainloader)
        val_loss, _, _ = valid_epoch(validloader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(_SCRIPT_DIR, 'base_model_molecule_encoder/best_schnet.pt'))

        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch:3d}/{epochs}  '
                  f'train_loss={train_loss:.4f}  val_loss={val_loss:.4f}')

    model.load_state_dict(torch.load(os.path.join(_SCRIPT_DIR, 'base_model_molecule_encoder/best_schnet.pt')))
    test_loss, preds, targets = valid_epoch(testloader)
    print(f'\nTest  loss={test_loss:.4f}')


if __name__ == '__main__':
    main()