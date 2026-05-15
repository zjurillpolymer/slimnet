from torch import nn

from data_loader import PI1070
import torch
import sys
sys.path.append('base_model_molecule_encoder')
from monomer_predict_GNN import QM9_GNN
import numpy as np
from torch_geometric.loader import DataLoader
import torch.nn.functional as F


'''准备数据'''
'''
  每个样本返回：
  - x: 分子图（送入 GNN Base Model）
  - qm: 4个单体量子属性（Base Model 监督信号）
  - chain: [分子量, 聚合度, 密度]（链参数）
  - order: [nematic_order]（序参数）
  - y: [热导率, 热扩散率, 介电常数, 线膨胀]（预测目标）
'''

dataset = PI1070('/Users/arcadio/Slimnet/data/PI1070.csv')
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

y_train = torch.cat([batch.y for batch in trainloader])  # 收集所有训练标签
y_mean = y_train.mean(dim=0)
y_std = y_train.std(dim=0) + 1e-8
print(f'y_mean: {y_mean}')
print(f'y_std:  {y_std}')


class SlimNet(nn.Module):
    def __init__(self,in_channels=260,out_channels=3,num_tasks=3):
        super().__init__()
        self.linear1=nn.Linear(256,out_channels)
        self.linear2=nn.Linear(in_channels,out_channels)
        self.linear3=nn.Linear(in_channels,out_channels)
        self.mlp=nn.Sequential(
            nn.Linear(1,64),
            nn.ReLU(),
            nn.Linear(64,3)
        )



    def forward(self,x,v_monomer):
        v_polymer=torch.cat([v_monomer,x.chain,x.order],dim=-1)
        v_polymer=F.dropout(v_polymer,p=0.1,training=self.training)
        alpha=torch.sigmoid(self.linear1(v_monomer))
        beta=F.softplus(self.linear2(v_polymer))
        gamma=F.softplus(self.linear3(v_polymer))

        attr_disordered=alpha*(beta**gamma)
        attr_ordered=self.mlp(x.order)

        attr_final=attr_disordered+attr_ordered
        return attr_final








'''加载权重，训练模型'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder=QM9_GNN(in_channels=11, edge_dim=4, hidden=64, heads=4, num_tasks=4)
encoder.load_state_dict(torch.load('/Users/arcadio/Slimnet/base_model_molecule_encoder/best_model.pt'))
model=SlimNet().to(device)
encoder.to(device)

# encoder 用小 lr，SLIMNet 用正常 lr
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
        out, v_momomer = encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch, return_v=True)
        optimizer.zero_grad()
        output=model(batch,v_momomer)
        y=batch.y
        loss = F.mse_loss(output, (y - y_mean.to(device)) / y_std.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)







@torch.no_grad()
def valid_epoch(loader):
    model.eval()
    encoder.eval()
    total_loss = 0
    all_preds,all_targets=[],[]
    for batch in loader:
        batch = batch.to(device)
        out, v_momomer = encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch, return_v=True)
        out=model(batch,v_momomer)
        y=batch.y
        loss=F.mse_loss(out, (y - y_mean.to(device)) / y_std.to(device))
        total_loss += loss.item() * batch.num_graphs
        preds = out * y_std.to(device) + y_mean.to(device)
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())
    return total_loss / len(loader.dataset), torch.cat(all_preds), torch.cat(all_targets)


def main():
    epochs = 300
    train_hist, val_hist = [], []
    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(trainloader)
        val_loss, preds, targets = valid_epoch(validloader)
        train_hist.append(train_loss)
        val_hist.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '/Users/arcadio/Slimnet/decoder/best_slimnet.pt')

        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch:3d}/{epochs}  '
                  f'train_loss={train_loss:.4f}  val_loss={val_loss:.4f}')

    # 测试集
    model.load_state_dict(torch.load('/Users/arcadio/Slimnet/decoder/best_slimnet.pt'))
    test_loss, preds, targets = valid_epoch(testloader)
    print(f'\nTest  loss={test_loss:.4f}')

    # 画图
    from plot_results import plot_all
    plot_all(train_hist, val_hist, preds.numpy(), targets.numpy())


if __name__ == '__main__':
    main()