import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from rdkit import Chem


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'aqsoldb_data', 'esol_backup.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'aqsoldb')
RESULT_DIR = os.path.join(BASE_DIR, 'results', 'aqsoldb')

for dir_path in [MODEL_DIR, RESULT_DIR]:
    os.makedirs(dir_path, exist_ok=True)



class GINEncoder(nn.Module):
    """Graph Isomorphism Network Encoder"""
    def __init__(self, num_features=9, hidden_dim=128, num_layers=4, dropout=0.2):
        super(GINEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.atom_embedding = nn.Linear(num_features, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
    def forward(self, x, edge_index, batch):
        x = self.atom_embedding(x)
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)



def get_atom_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        atom.GetNumRadicalElectrons(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        atom.GetMass() / 100.0,
    ]


def smiles_to_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
        if len(atom_features) == 0:
            return None
            
        x = torch.tensor(atom_features, dtype=torch.float)
        
        edge_index = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.extend([[i, j], [j, i]])
        
        # 孤儿节点: 使用空边张量
        if len(edge_index) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        return Data(x=x, edge_index=edge_index)
    except Exception:
        return None



class ESOLDataset(Dataset):
    """ESOL/AqSolDB 数据集"""
    
    def __init__(self, data_path=DATA_PATH):
        self.graphs = []
        self.targets = []
        
        self._load_data(data_path)
        self._normalize()
        
        print(f"ESOL 数据集: {len(self.graphs)} 个有效样本")
    
    def _load_data(self, data_path):
        df = pd.read_csv(data_path)
        
        # ESOL 列名
        smiles_col = 'smiles'
        target_col = 'measured log solubility in mols per litre'
        
        print(f"加载 {data_path}...")
        print(f"列名: {df.columns.tolist()}")
        
        success = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc="处理 ESOL 数据"):
            try:
                smiles = str(row[smiles_col])
                target = float(row[target_col])
                
                if pd.isna(target):
                    continue
                
                graph = smiles_to_graph(smiles)
                if graph is not None:
                    self.graphs.append(graph)
                    self.targets.append(target)
                    success += 1
            except Exception:
                continue
        
        print(f"成功加载: {success}/{len(df)}")
    
    def _normalize(self):
        self.targets = np.array(self.targets)
        self.target_mean = np.mean(self.targets)
        self.target_std = np.std(self.targets) + 1e-8
        self.targets = (self.targets - self.target_mean) / self.target_std
        self.targets = self.targets.tolist()
        
        print(f"LogS: mean={self.target_mean:.3f}, std={self.target_std:.3f}")
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx], torch.tensor([self.targets[idx]], dtype=torch.float)


def collate_fn(batch):
    graphs, targets = zip(*batch)
    
    x_list, edge_list, batch_list = [], [], []
    node_offset = 0
    
    for i, g in enumerate(graphs):
        x_list.append(g.x)
        edge_list.append(g.edge_index + node_offset)
        batch_list.extend([i] * g.x.size(0))
        node_offset += g.x.size(0)
    
    return (torch.cat(x_list, 0), torch.cat(edge_list, 1), 
            torch.tensor(batch_list, dtype=torch.long), torch.stack(targets))



class PretrainModel(nn.Module):
    
    def __init__(self, hidden_dim=128, num_layers=4, dropout=0.2):
        super(PretrainModel, self).__init__()
        self.encoder = GINEncoder(9, hidden_dim, num_layers, dropout)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, edge_index, batch):
        emb = self.encoder(x, edge_index, batch)
        return self.predictor(emb)
    
    def get_encoder(self):
        return self.encoder



def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, num_samples = 0, 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        x, edge_index, batch_idx, targets = batch
        x, edge_index, batch_idx = x.to(device), edge_index.to(device), batch_idx.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        pred = model(x, edge_index, batch_idx)
        loss = criterion(pred, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * targets.size(0)
        num_samples += targets.size(0)
    
    return total_loss / num_samples


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, predictions, targets_list, num_samples = 0, [], [], 0
    
    with torch.no_grad():
        for batch in loader:
            x, edge_index, batch_idx, targets = batch
            x, edge_index, batch_idx = x.to(device), edge_index.to(device), batch_idx.to(device)
            targets = targets.to(device)
            
            pred = model(x, edge_index, batch_idx)
            total_loss += criterion(pred, targets).item() * targets.size(0)
            num_samples += targets.size(0)
            predictions.extend(pred.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())
    
    predictions = np.array(predictions).flatten()
    targets_arr = np.array(targets_list).flatten()
    
    ss_res = np.sum((targets_arr - predictions) ** 2)
    ss_tot = np.sum((targets_arr - np.mean(targets_arr)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean((targets_arr - predictions) ** 2))
    
    return total_loss / num_samples, r2, rmse


def plot_results(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('ESOL Pretrain - Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history['val_r2'], color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('R²')
    axes[1].set_title('ESOL Pretrain - Validation R²')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    config = {
        'hidden_dim': 128,
        'num_layers': 4,
        'dropout': 0.2,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': 1e-5,
        'epochs': args.epochs,
        'early_stopping_patience': 20,
        'seed': args.seed,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    print("\n" + "="*60)
    print("AqSolDB/ESOL 预训练 (单分子溶解度)")
    print("="*60)
    
    dataset = ESOLDataset(DATA_PATH)
    
    n = len(dataset)
    indices = np.random.permutation(n).tolist()
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    print(f"划分: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")
    
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), 
                             batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), 
                           batch_size=config['batch_size'], collate_fn=collate_fn)
    test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx), 
                            batch_size=config['batch_size'], collate_fn=collate_fn)
    
    model = PretrainModel(config['hidden_dim'], config['num_layers'], config['dropout']).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
    best_val_loss, patience_counter = float('inf'), 0
    best_model_state = None
    
    print("\n开始预训练...")
    for epoch in range(config['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_r2, val_rmse = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_r2)
        
        print(f"Epoch {epoch+1}/{config['epochs']}: "
              f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val R²={val_r2:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= config['early_stopping_patience']:
            print(f"\n早停: {config['early_stopping_patience']} 个 epoch 没有改善")
            break
    
    model.load_state_dict(best_model_state)
    test_loss, test_r2, test_rmse = evaluate(model, test_loader, criterion, device)
    
    print(f"\n{'='*60}")
    print(f"ESOL 预训练完成!")
    print(f"{'='*60}")
    print(f"测试结果: R²={test_r2:.4f}, RMSE={test_rmse:.4f}")
    
    encoder_path = os.path.join(MODEL_DIR, 'drug_encoder.pth')
    torch.save(model.encoder.state_dict(), encoder_path)
    print(f"\nDrug Encoder 保存至: {encoder_path}")
    
    full_model_path = os.path.join(MODEL_DIR, 'full_model.pth')
    torch.save(model.state_dict(), full_model_path)
    
    plot_path = os.path.join(RESULT_DIR, 'training_history.png')
    plot_results(history, plot_path)
    print(f"训练曲线保存至: {plot_path}")
    
    return {'test_r2': test_r2, 'test_rmse': test_rmse}


if __name__ == "__main__":
    main()
