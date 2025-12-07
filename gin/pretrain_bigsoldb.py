import os
import sys
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
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'bigsoldb')
RESULT_DIR = os.path.join(BASE_DIR, 'results', 'bigsoldb')
DATA_DIR = os.path.join(BASE_DIR, 'bigsoldb_data')

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
    """获取原子特征向量 - 9维"""
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
    """将 SMILES 转换为图数据"""
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
        
        if len(edge_index) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        return Data(x=x, edge_index=edge_index)
    except Exception:
        return None


class BigSolDBDataset(Dataset):
    """
    BigSolDB v2.0 数据集
    学习 Drug + Solvent + Temperature -> LogS
    """
    
    def __init__(self, data_path, max_samples=None):
        self.drug_graphs = []
        self.solvent_graphs = []
        self.temperatures = []
        self.targets = []
        
        self._load_data(data_path, max_samples)
        self._normalize()
        
        print(f"BigSolDB 数据集: {len(self.drug_graphs)} 个样本")
    
    def _load_data(self, data_path, max_samples):
        print(f"加载 {data_path}...")
        df = pd.read_csv(data_path)
        
        solute_col = 'SMILES_Solute'
        solvent_col = 'SMILES_Solvent'
        temp_col = 'Temperature_K'
        target_col = 'LogS(mol/L)'
        
        if max_samples and len(df) > max_samples:
            print(f"采样 {max_samples} 条数据 (总共 {len(df)} 条)")
            df = df.sample(n=max_samples, random_state=42)
        
        smiles_cache = {}
        
        success = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc="处理 BigSolDB"):
            try:
                solute_smiles = str(row[solute_col])
                solvent_smiles = str(row[solvent_col])
                temperature = float(row[temp_col])
                target = float(row[target_col])
                
                if pd.isna(target) or pd.isna(temperature):
                    continue
                
                if solute_smiles not in smiles_cache:
                    smiles_cache[solute_smiles] = smiles_to_graph(solute_smiles)
                if solvent_smiles not in smiles_cache:
                    smiles_cache[solvent_smiles] = smiles_to_graph(solvent_smiles)
                
                drug_graph = smiles_cache[solute_smiles]
                solvent_graph = smiles_cache[solvent_smiles]
                
                if drug_graph is not None and solvent_graph is not None:
                    self.drug_graphs.append(drug_graph)
                    self.solvent_graphs.append(solvent_graph)
                    self.temperatures.append(temperature)
                    self.targets.append(target)
                    success += 1
            except Exception:
                continue
        
        print(f"成功加载: {success}/{len(df)}")
    
    def _normalize(self):
        self.temperatures = np.array(self.temperatures)
        self.temp_mean = np.mean(self.temperatures)
        self.temp_std = np.std(self.temperatures) + 1e-8
        self.temperatures = (self.temperatures - self.temp_mean) / self.temp_std
        self.temperatures = self.temperatures.tolist()
        
        self.targets = np.array(self.targets)
        self.target_mean = np.mean(self.targets)
        self.target_std = np.std(self.targets) + 1e-8
        self.targets = (self.targets - self.target_mean) / self.target_std
        self.targets = self.targets.tolist()
        
        print(f"温度: mean={self.temp_mean:.2f}K, std={self.temp_std:.2f}")
        print(f"LogS: mean={self.target_mean:.3f}, std={self.target_std:.3f}")
    
    def __len__(self):
        return len(self.drug_graphs)
    
    def __getitem__(self, idx):
        return {
            'drug_graph': self.drug_graphs[idx],
            'solvent_graph': self.solvent_graphs[idx],
            'temperature': torch.tensor([self.temperatures[idx]], dtype=torch.float),
            'target': torch.tensor([self.targets[idx]], dtype=torch.float)
        }


def collate_fn(batch):
    """批处理函数 - 处理 Drug + Solvent 对"""
    drug_graphs = [item['drug_graph'] for item in batch]
    solvent_graphs = [item['solvent_graph'] for item in batch]
    temperatures = torch.stack([item['temperature'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    
    drug_x_list, drug_edge_list, drug_batch = [], [], []
    node_offset = 0
    for i, g in enumerate(drug_graphs):
        drug_x_list.append(g.x)
        drug_edge_list.append(g.edge_index + node_offset)
        drug_batch.extend([i] * g.x.size(0))
        node_offset += g.x.size(0)
    
    drug_x = torch.cat(drug_x_list, 0)
    drug_edge_index = torch.cat(drug_edge_list, 1)
    drug_batch = torch.tensor(drug_batch, dtype=torch.long)
    
    solvent_x_list, solvent_edge_list, solvent_batch = [], [], []
    node_offset = 0
    for i, g in enumerate(solvent_graphs):
        solvent_x_list.append(g.x)
        solvent_edge_list.append(g.edge_index + node_offset)
        solvent_batch.extend([i] * g.x.size(0))
        node_offset += g.x.size(0)
    
    solvent_x = torch.cat(solvent_x_list, 0)
    solvent_edge_index = torch.cat(solvent_edge_list, 1)
    solvent_batch = torch.tensor(solvent_batch, dtype=torch.long)
    
    return (drug_x, drug_edge_index, drug_batch,
            solvent_x, solvent_edge_index, solvent_batch,
            temperatures, targets)


class BigSolDBPretrainModel(nn.Module):
    """
    BigSolDB 预训练模型
    Drug Encoder + Solvent Encoder + Temperature -> LogS
    """
    
    def __init__(self, num_features=9, hidden_dim=128, num_layers=4, dropout=0.2):
        super(BigSolDBPretrainModel, self).__init__()
        
        self.drug_encoder = GINEncoder(num_features, hidden_dim, num_layers, dropout)
        self.solvent_encoder = GINEncoder(num_features, hidden_dim, num_layers, dropout)
        
        self.temp_embedding = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 32, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, drug_x, drug_edge_index, drug_batch,
                solvent_x, solvent_edge_index, solvent_batch, temperature):
        drug_emb = self.drug_encoder(drug_x, drug_edge_index, drug_batch)
        solvent_emb = self.solvent_encoder(solvent_x, solvent_edge_index, solvent_batch)
        temp_emb = self.temp_embedding(temperature)
        
        combined = torch.cat([drug_emb, solvent_emb, temp_emb], dim=1)
        return self.predictor(combined)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, num_samples = 0, 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        drug_x, drug_edge, drug_batch, solvent_x, solvent_edge, solvent_batch, temps, targets = batch
        
        drug_x = drug_x.to(device)
        drug_edge = drug_edge.to(device)
        drug_batch = drug_batch.to(device)
        solvent_x = solvent_x.to(device)
        solvent_edge = solvent_edge.to(device)
        solvent_batch = solvent_batch.to(device)
        temps = temps.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        pred = model(drug_x, drug_edge, drug_batch, solvent_x, solvent_edge, solvent_batch, temps)
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
            drug_x, drug_edge, drug_batch, solvent_x, solvent_edge, solvent_batch, temps, targets = batch
            
            drug_x = drug_x.to(device)
            drug_edge = drug_edge.to(device)
            drug_batch = drug_batch.to(device)
            solvent_x = solvent_x.to(device)
            solvent_edge = solvent_edge.to(device)
            solvent_batch = solvent_batch.to(device)
            temps = temps.to(device)
            targets = targets.to(device)
            
            pred = model(drug_x, drug_edge, drug_batch, solvent_x, solvent_edge, solvent_batch, temps)
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


def plot_history(history, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('BigSolDB - Training Loss')
    axes[0].legend(); axes[0].grid(True)
    
    axes[1].plot(history['val_r2'], color='green')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('R²')
    axes[1].set_title('BigSolDB - Validation R²')
    axes[1].grid(True)
    
    axes[2].plot(history['val_rmse'], color='orange')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('RMSE')
    axes[2].set_title('BigSolDB - Validation RMSE')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()
    print(f"训练曲线已保存至: {save_dir}/training_history.png")


def plot_predictions(predictions, targets, save_dir):
    """绘制预测 vs 真实散点图"""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(targets, predictions, alpha=0.3, s=5)
    
    min_val = min(min(targets), min(predictions))
    max_val = max(max(targets), max(predictions))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    ax.set_xlabel('True LogS (standardized)')
    ax.set_ylabel('Predicted LogS (standardized)')
    ax.set_title('BigSolDB - Predictions vs True')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'predictions.png'), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-samples', type=int, default=None, help='最大样本数')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=128, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    args = parser.parse_args()
    
    config = {
        'hidden_dim': 128,
        'num_layers': 4,
        'dropout': 0.2,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': 1e-5,
        'epochs': args.epochs,
        'early_stopping_patience': 15,
        'seed': 42,
        'max_samples': args.max_samples,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    print("\n" + "="*60)
    print("BigSolDB v2.0 预训练: Drug + Solvent + Temp -> LogS")
    print("="*60)
    
    data_path = os.path.join(DATA_DIR, 'BigSolDBv2.0.csv')
    dataset = BigSolDBDataset(data_path, max_samples=config['max_samples'])
    
    n = len(dataset)
    indices = np.random.permutation(n)
    train_size = int(0.9 * n)
    
    train_dataset = torch.utils.data.Subset(dataset, indices[:train_size])
    val_dataset = torch.utils.data.Subset(dataset, indices[train_size:])
    
    print(f"\n训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           collate_fn=collate_fn, num_workers=0)
    
    model = BigSolDBPretrainModel(
        9, config['hidden_dim'], config['num_layers'], config['dropout']
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': [], 'val_r2': [], 'val_rmse': []}
    best_val_loss, best_epoch, patience_counter = float('inf'), 0, 0
    best_predictions, best_targets = None, None
    
    print("\n开始训练...")
    for epoch in range(config['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_r2, val_rmse = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_r2)
        history['val_rmse'].append(val_rmse)
        
        print(f"Epoch {epoch+1}/{config['epochs']}: "
              f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val R²={val_r2:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss, best_epoch, patience_counter = val_loss, epoch, 0
            
            torch.save(model.drug_encoder.state_dict(), 
                      os.path.join(MODEL_DIR, 'drug_encoder.pth'))
            torch.save(model.solvent_encoder.state_dict(), 
                      os.path.join(MODEL_DIR, 'solvent_encoder.pth'))
            torch.save(model.state_dict(), 
                      os.path.join(MODEL_DIR, 'full_model.pth'))
            
            print(f"  -> 保存最佳模型 (Drug + Solvent Encoder)")
        else:
            patience_counter += 1
        
        if patience_counter >= config['early_stopping_patience']:
            print(f"\n早停: {config['early_stopping_patience']} 个 epoch 没有改善")
            break
    
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'full_model.pth')))
    _, final_r2, final_rmse = evaluate(model, val_loader, criterion, device)
    
    print(f"\n{'='*60}")
    print(f"BigSolDB v2.0 预训练完成!")
    print(f"最佳 epoch: {best_epoch + 1}")
    print(f"最佳 Val R²: {history['val_r2'][best_epoch]:.4f}")
    print(f"最佳 Val RMSE: {history['val_rmse'][best_epoch]:.4f}")
    print(f"\n模型保存至:")
    print(f"  - {MODEL_DIR}/drug_encoder.pth")
    print(f"  - {MODEL_DIR}/solvent_encoder.pth")
    print(f"{'='*60}")
    
    plot_history(history, RESULT_DIR)
    
    return model, history


if __name__ == "__main__":
    main()
