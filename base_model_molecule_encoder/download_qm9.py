"""下载 QM9 数据集并查看数据结构"""
import torch
from torch_geometric.datasets import QM9

print("正在下载 QM9 数据集（~300MB）...")
dataset = QM9(root='../data/QM9')
print(f'QM9 样本数: {len(dataset)}')
print(f'每个样本特征维度: {dataset.num_features}')
print(f'每个样本标签维度: {dataset.num_classes}')

data = dataset[0]
print(f'\n=== 第一个样本结构 ===')
print(f'原子数 (num_nodes): {data.num_nodes}')
print(f'边数 (num_edges): {data.num_edges}')
print(f'原子特征 (x) shape: {data.x.shape}')
print(f'原子特征 (x): 5原子类型 one-hot + 6实数值(原子序数,芳香性,sp,sp2,sp3,H数) = {data.x.shape[1]}维')
print(f'边索引 (edge_index) shape: {data.edge_index.shape}')
print(f'边属性 (edge_attr) shape: {data.edge_attr.shape}')
print(f'位置坐标 (pos) shape: {data.pos.shape}')
print(f'标签 (y) shape: {data.y.shape}')
print(f'标签 (y) 值: {data.y}')
print(f'所有属性: {data.keys()}')

# QM9 的 19 个量子化学属性
qm9_targets = [
    'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve',
    'U0', 'U', 'H', 'G', 'Cv',
    'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'A_atom', 'B_atom', 'C_atom'
]
print(f'\n=== QM9 19 个标签属性对照 ===')
for i, name in enumerate(qm9_targets):
    vals = dataset._data.y[:, i]
    print(f'  [{i:2d}] {name:12s}  mean={vals.mean():.4f}  std={vals.std():.4f}  '
          f'min={vals.min():.4f}  max={vals.max():.4f}')

# 论文中 Base Model 监督目标: HOMO(3), LUMO(4), mu(0/ dipole), alpha(1/ polarizability)
print(f'\n=== 论文关注的 monomer 属性 ===')
target_indices = {
    'mu (dipole moment)': 0,
    'alpha (polarizability)': 1,
    'homo': 3,
    'lumo': 4,
}
for name, idx in target_indices.items():
    vals = dataset._data.y[:, idx]
    print(f'  {name:30s} mean={vals.mean():.4f}  std={vals.std():.4f}')

# 查看第一个样本的原子详情
print(f'\n=== 第一个样本原子详情 ===')
for i in range(data.num_nodes):
    feats = data.x[i]
    atom_types = ['H','C','N','O','F']
    atom_idx = int(feats[:5].argmax().item())
    h_conn = int(feats[10].item())
    print(f'  原子{i}: {atom_types[atom_idx]:>3}  atomic#={int(feats[5].item()):2d}  '
          f'aromatic={int(feats[6].item())}  sp={int(feats[7].item())}  '
          f'sp2={int(feats[8].item())}  sp3={int(feats[9].item())}  Hs={h_conn}')
print(f'  位置坐标:\n{data.pos}')
print(f'  y(标签): homo={data.y[0, 3]:.4f}, lumo={data.y[0, 4]:.4f}, '
      f'dipole={data.y[0, 0]:.4f}, alpha={data.y[0, 1]:.4f}')
