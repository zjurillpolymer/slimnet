"""PI1070 dataset: 读取 + p-SMILES转图 + 构建PyG Dataset"""
import pandas as pd
import torch
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
from torch_geometric.data import Dataset, Data
from torch_geometric.nn import global_mean_pool

# 特征列索引
QM_COLS = ['qm_homo_monomer', 'qm_lumo_monomer', 'qm_dipole_monomer', 'qm_polarizability_monomer']
CHAIN_COLS = ['mol_weight_monomer', 'DP', 'density']
ORDER_COLS = ['nematic_order_parameter']
TARGET_COLS = ['thermal_diffusivity',
               'static_dielectric_const', 'linear_expansion']


def parse_polymer_smiles(smi, cap_with='[H]'):
    """将 p-SMILES（带 * 标记）转为 RDKit 可解析的 SMILES。"""
    smi = smi.replace('*', cap_with)
    return smi


def mol_from_polymer_smiles(smi):
    """p-SMILES → RDKit Mol"""
    smi_clean = parse_polymer_smiles(smi)
    mol = Chem.MolFromSmiles(smi_clean)
    if mol is None:
        mol = Chem.MolFromSmiles(smi.replace('*', ''))
    return mol


# ── QM9 风格的特征（11 维原子 + 4 维边）──
ATOM_TYPES = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
BOND_TYPES = [Chem.rdchem.BondType.SINGLE,
              Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE,
              Chem.rdchem.BondType.AROMATIC]


def one_hot(x, num_classes):
    return torch.eye(num_classes)[x]


# SchNet 支持的原子序数（与 QM9 一致），超出部分映射到 0
SCHNET_MAX_Z = 10
SCHNET_OOV_Z = 0  # 未知原子映射到 H(1) 的位置

def get_3d_structure(mol):
    """用 RDKit 生成 3D 构象，返回原子序数 z 和坐标 pos"""
    from rdkit.Chem import AllChem
    mol_copy = Chem.RWMol(mol)

    succeeded = AllChem.EmbedMolecule(mol_copy, randomSeed=42)
    if succeeded != 0:
        succeeded = AllChem.EmbedMolecule(mol_copy, useRandomCoords=True, randomSeed=42)

    if succeeded != 0 or mol_copy.GetNumConformers() == 0:
        n = mol_copy.GetNumAtoms()
        z = torch.tensor([a.GetAtomicNum() if a.GetAtomicNum() <= SCHNET_MAX_Z else SCHNET_OOV_Z
                          for a in mol_copy.GetAtoms()], dtype=torch.long)
        pos = torch.randn(n, 3, dtype=torch.float)
        return z, pos

    conf = mol_copy.GetConformer()
    z = torch.tensor([a.GetAtomicNum() if a.GetAtomicNum() <= SCHNET_MAX_Z else SCHNET_OOV_Z
                      for a in mol_copy.GetAtoms()], dtype=torch.long)
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float)
    return z, pos


def qm9_style_features(mol):
    """提取与 QM9 相同的 11 维原子特征 + 4 维边特征"""
    type_idx, aromatic, sp, sp2, sp3, num_hs = [], [], [], [], [], []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym in ATOM_TYPES:
            type_idx.append(ATOM_TYPES[sym])
        else:
            type_idx.append(len(ATOM_TYPES))  # 其他原子统一丢到 extra
            if len(ATOM_TYPES) not in type_idx:
                pass  # 已有 extra class
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        h = atom.GetHybridization()
        sp.append(1 if h == Chem.rdchem.HybridizationType.SP else 0)
        sp2.append(1 if h == Chem.rdchem.HybridizationType.SP2 else 0)
        sp3.append(1 if h == Chem.rdchem.HybridizationType.SP3 else 0)
        num_hs.append(atom.GetTotalNumHs())

    N = mol.GetNumAtoms()
    x1 = one_hot(torch.tensor(type_idx), num_classes=len(ATOM_TYPES) + 1)  # 6
    x2 = torch.tensor([aromatic, sp, sp2, sp3, num_hs], dtype=torch.float).t()  # 5
    x = torch.cat([x1, x2], dim=-1)  # [N, 11]

    # 边特征
    src, dst, edge_types = [], [], []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [u, v]
        dst += [v, u]
        edge_types += 2 * [BOND_TYPES.index(bond.GetBondType())]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = one_hot(torch.tensor(edge_types), num_classes=len(BOND_TYPES)).float()

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class PI1070(Dataset):
    def __init__(self, csv_path, transform=None, pre_transform=None):
        self.df = pd.read_csv(csv_path)

        # 丢弃无效 SMILES
        valid_mask = []
        for smi in self.df['smiles']:
            mol = mol_from_polymer_smiles(smi)
            valid_mask.append(mol is not None)
        self.df = self.df[valid_mask].reset_index(drop=True)
        print(f"PI1070: {len(self.df)} valid samples (dropped {sum(~np.array(valid_mask))})")

        self.qm = torch.tensor(self.df[QM_COLS].values.astype(np.float32))
        self.chain = torch.tensor(self.df[CHAIN_COLS].values.astype(np.float32))
        self.order = torch.tensor(self.df[ORDER_COLS].values.astype(np.float32))
        self.targets = torch.tensor(self.df[TARGET_COLS].values.astype(np.float32))

        # NAN 填充为 0
        for t in [self.qm, self.chain, self.order, self.targets]:
            t[torch.isnan(t)] = 0.0

        # 标准化 chain 和 order（防止 beta**gamma 溢出）
        self.chain = (self.chain - self.chain.mean(dim=0)) / (self.chain.std(dim=0) + 1e-8)
        self.order = (self.order - self.order.mean(dim=0)) / (self.order.std(dim=0) + 1e-8)

        super().__init__(transform, pre_transform)

    def len(self):
        return len(self.df)

    def get(self, idx):
        row = self.df.iloc[idx]
        mol = mol_from_polymer_smiles(row['smiles'])

        # QM9 风格特征（GAT 编码器用）
        monomer_graph = qm9_style_features(mol)

        # 3D 结构（SchNet 编码器用）
        z, pos = get_3d_structure(mol)

        # 验证一致性：edge_index 不能超出 pos 的原子数
        if monomer_graph.edge_index.numel() > 0 and monomer_graph.edge_index.max() >= z.size(0):
            n = mol.GetNumAtoms()
            z = torch.tensor([a.GetAtomicNum() if a.GetAtomicNum() <= SCHNET_MAX_Z else SCHNET_OOV_Z
                              for a in mol.GetAtoms()], dtype=torch.long)
            pos = torch.randn(n, 3, dtype=torch.float)

        data = Data(
            x=monomer_graph.x,
            edge_index=monomer_graph.edge_index,
            edge_attr=monomer_graph.edge_attr,
            z=z,
            pos=pos,
            qm=self.qm[idx].unsqueeze(0),
            chain=self.chain[idx].unsqueeze(0),
            order=self.order[idx].unsqueeze(0),
            y=self.targets[idx].unsqueeze(0),
        )
        return data


if __name__ == '__main__':
    dataset = PI1070('../data/PI1070.csv')
    print(f'\n样本总数: {len(dataset)}')

    d = dataset[0]
    print(f'\n第一个样本:')
    print(f'  x (atom feats):       {d.x.shape}')
    print(f'  edge_index:           {d.edge_index.shape}')
    print(f'  edge_attr:            {d.edge_attr.shape}')
    print(f'  qm (monomer props):   {d.qm.shape} → {d.qm.tolist()}')
    print(f'  chain (MW,DP,density): {d.chain.shape} → {d.chain.tolist()}')
    print(f'  order (nematic):      {d.order.shape} → {d.order.tolist()}')
    print(f'  y (targets):          {d.y.shape} → {d.y.tolist()}')
