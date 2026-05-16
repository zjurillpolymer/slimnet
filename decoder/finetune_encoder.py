"""Stage 2: 在 PI1070 单体属性上 fine-tune SchNet encoder"""
import torch
import sys, os
sys.path.append('base_model_molecule_encoder')
from Schnet_model_monomer import Schnet_monomer
from data_loader import PI1070
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据路径：兼容扁平结构和子目录结构
_script_dir = os.path.dirname(os.path.abspath(__file__))
_csv = os.path.join(_script_dir, '..', 'data', 'PI1070.csv')
if not os.path.exists(_csv):
    _csv = os.path.join(_script_dir, '..', 'PI1070.csv')  # 扁平结构
_csv = os.path.abspath(_csv)
print(f'Data: {_csv}')
_dataset = PI1070(_csv)
np.random.seed(42)
n = len(_dataset)
idx = np.random.permutation(n)
train_ds = _dataset[idx[:int(0.8*n)]]
valid_ds = _dataset[idx[int(0.8*n):int(0.9*n)]]
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=False)

# qm 标准化
qm_train = torch.cat([b.qm for b in train_loader])
qm_mean = qm_train.mean(dim=0)
qm_std = qm_train.std(dim=0) + 1e-8

# 加载预训练 encoder
encoder = Schnet_monomer(hidden_dim=128, n_layers=6)
_pt = os.path.join(_script_dir, '..', 'base_model_molecule_encoder', 'best_schnet.pt')
if not os.path.exists(_pt):
    _pt = os.path.join(_script_dir, '..', 'best_schnet.pt')
_pt = os.path.abspath(_pt)
print(f'Loading from: {_pt}')
encoder.load_state_dict(torch.load(_pt, map_location='cpu'))
encoder.to(DEVICE)

optimizer = torch.optim.Adam([
    {'params': encoder.atom_embed.parameters(), 'lr': 1e-6},
    {'params': encoder.convs[:2].parameters(), 'lr': 1e-6},
    {'params': encoder.convs[2:4].parameters(), 'lr': 5e-6},
    {'params': encoder.convs[4:].parameters(), 'lr': 1e-5},
    {'params': encoder.alpha.parameters(), 'lr': 1e-5},
    {'params': encoder.homo.parameters(), 'lr': 1e-5},
    {'params': encoder.lumo.parameters(), 'lr': 1e-5},
    {'params': encoder.mu.parameters(), 'lr': 1e-5},
], weight_decay=1e-4)
best_val = float('inf')

for epoch in range(1, 51):
    encoder.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out = encoder(batch.z, batch.pos, batch.edge_index, batch.batch)
        target = (batch.qm - qm_mean.to(DEVICE)) / qm_std.to(DEVICE)
        loss = F.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    train_loss = total_loss / len(train_loader.dataset)

    if epoch % 10 == 0 or epoch == 1:
        encoder.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(DEVICE)
                out = encoder(batch.z, batch.pos, batch.edge_index, batch.batch)
                target = (batch.qm - qm_mean.to(DEVICE)) / qm_std.to(DEVICE)
                val_loss += F.mse_loss(out, target).item() * batch.num_graphs
        val_loss /= len(valid_loader.dataset)
        print(f'Epoch {epoch:2d}/50  train={train_loss:.4f}  val={val_loss:.4f}')

    # 保存（在全部训练中只保存最好的）
    if epoch == 1 or val_loss < best_val:
        best_val = val_loss
        save_path = os.path.abspath(os.path.join(_script_dir, '..', 'best_schnet_ft.pt'))
        torch.save(encoder.state_dict(), save_path)

print(f'Best val loss: {best_val:.4f}, saved to {save_path}')
