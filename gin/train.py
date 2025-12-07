import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import SaltRemover
from collections import defaultdict
from sklearn.model_selection import KFold

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'finetune')
RESULT_DIR = os.path.join(BASE_DIR, 'results', 'finetune')

for dir_path in [MODEL_DIR, RESULT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

remover = SaltRemover.SaltRemover()

class GINEncoder(nn.Module):
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
        
        try:
            mol = remover.StripMol(mol)
        except:
            pass
            
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumAtoms())
        
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

class SolubilityDataset(Dataset):
    def __init__(self, data_path):
        self.drug_graphs = []
        self.solvent_graphs = []
        self.temperatures = []
        self.targets = []
        self.drug_smiles_list = []
        
        self._load_data(data_path)
        self._normalize()
        
        print(f"数据集: {len(self.drug_graphs)} 个样本")
    
    def _load_data(self, data_path):
        try:
            df = pd.read_csv(data_path, encoding='utf-8')
        except:
            df = pd.read_csv(data_path, encoding='gbk')
        
        drug_col = 'Drug SMILES'
        solvent_col = 'Solvent SMILES'
        temp_col = 'Temperature'
        target_col = 'Solubility_log10'
        
        smiles_cache = {}
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="加载数据"):
            try:
                drug_smiles = str(row[drug_col])
                solvent_smiles = str(row[solvent_col])
                temperature = float(row[temp_col])
                target = float(row[target_col])
                
                if pd.isna(target):
                    continue
                
                mol = Chem.MolFromSmiles(drug_smiles)
                if mol:
                    try: mol = remover.StripMol(mol)
                    except: pass
                    frags = Chem.GetMolFrags(mol, asMols=True)
                    if len(frags) > 1: mol = max(frags, key=lambda m: m.GetNumAtoms())
                    drug_smiles_clean = Chem.MolToSmiles(mol, canonical=True)
                else:
                    continue

                if drug_smiles_clean not in smiles_cache:
                    smiles_cache[drug_smiles_clean] = smiles_to_graph(drug_smiles_clean)
                if solvent_smiles not in smiles_cache:
                    smiles_cache[solvent_smiles] = smiles_to_graph(solvent_smiles)
                
                drug_graph = smiles_cache[drug_smiles_clean]
                solvent_graph = smiles_cache[solvent_smiles]
                
                if drug_graph is not None and solvent_graph is not None:
                    self.drug_graphs.append(drug_graph)
                    self.solvent_graphs.append(solvent_graph)
                    self.temperatures.append(temperature)
                    self.targets.append(target)
                    self.drug_smiles_list.append(drug_smiles_clean)
            except Exception:
                continue
    
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
        
        print(f"温度: mean={self.temp_mean:.2f}, std={self.temp_std:.2f}")
        print(f"LogS (Norm): mean={self.target_mean:.3f}, std={self.target_std:.3f}")
    
    def __len__(self):
        return len(self.drug_graphs)
    
    def __getitem__(self, idx):
        return {
            'drug_graph': self.drug_graphs[idx],
            'solvent_graph': self.solvent_graphs[idx],
            'temperature': torch.tensor([self.temperatures[idx]], dtype=torch.float),
            'target': torch.tensor([self.targets[idx]], dtype=torch.float),
        }

def collate_fn(batch):
    drug_graphs = [item['drug_graph'] for item in batch]
    solvent_graphs = [item['solvent_graph'] for item in batch]
    temperatures = torch.stack([item['temperature'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    
    def batch_graphs(graphs):
        x_list, edge_list, batch_list = [], [], []
        offset = 0
        for i, g in enumerate(graphs):
            x_list.append(g.x)
            edge_list.append(g.edge_index + offset)
            batch_list.extend([i] * g.x.size(0))
            offset += g.x.size(0)
        return torch.cat(x_list, 0), torch.cat(edge_list, 1), torch.tensor(batch_list, dtype=torch.long)

    dx, de, db = batch_graphs(drug_graphs)
    sx, se, sb = batch_graphs(solvent_graphs)
    
    return dx, de, db, sx, se, sb, temperatures, targets

class DualBranchFusionModel(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=4, dropout=0.2):
        super(DualBranchFusionModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        self.drug_encoder_aq = GINEncoder(9, hidden_dim, num_layers, dropout)
        
        self.drug_encoder_bs = GINEncoder(9, hidden_dim, num_layers, dropout)
        self.solvent_encoder = GINEncoder(9, hidden_dim, num_layers, dropout)
        
        self.temp_embedding = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 32))
        
        self.interaction_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.fusion_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def load_pretrained(self, device='cpu', mode='both'):
        print(f"\n📥 正在加载预训练权重 (模式: {mode})...")
        
        if mode in ['aqsoldb', 'both']:
            aq_path = os.path.join(BASE_DIR, 'models', 'aqsoldb', 'drug_encoder.pth')
            if os.path.exists(aq_path):
                self.drug_encoder_aq.load_state_dict(torch.load(aq_path, map_location=device))
                print(f"  ✓ 分支1: Loaded AqSolDB (Drug_AQ)")
            else:
                print(f"  ✗ 分支1: 未找到 {aq_path}")
        
        if mode in ['bigsoldb', 'both']:
            bs_drug_path = os.path.join(BASE_DIR, 'models', 'bigsoldb', 'drug_encoder.pth')
            bs_solvent_path = os.path.join(BASE_DIR, 'models', 'bigsoldb', 'solvent_encoder.pth')
            
            if os.path.exists(bs_drug_path):
                self.drug_encoder_bs.load_state_dict(torch.load(bs_drug_path, map_location=device))
                print(f"  ✓ 分支2: Loaded BigSolDB (Drug_BS)")
            else:
                print(f"  ✗ 分支2: 未找到 {bs_drug_path}")
                
            if os.path.exists(bs_solvent_path):
                self.solvent_encoder.load_state_dict(torch.load(bs_solvent_path, map_location=device))
                print(f"  ✓ 分支2: Loaded BigSolDB (Solvent)")
            else:
                print(f"  ✗ 分支2: 未找到 {bs_solvent_path}")
    
    def forward(self, drug_x, drug_edge_index, drug_batch,
                solvent_x, solvent_edge_index, solvent_batch, temperature):
        
        drug_emb_aq = self.drug_encoder_aq(drug_x, drug_edge_index, drug_batch)
        drug_emb_bs = self.drug_encoder_bs(drug_x, drug_edge_index, drug_batch)
        solvent_emb = self.solvent_encoder(solvent_x, solvent_edge_index, solvent_batch)
        
        interaction_input = torch.cat([drug_emb_bs, solvent_emb], dim=1)
        interaction_emb = self.interaction_layer(interaction_input)
        temp_emb = self.temp_embedding(temperature)
        
        fusion = torch.cat([drug_emb_aq, interaction_emb, temp_emb], dim=1)
        return self.fusion_predictor(fusion)
    
    def get_param_groups(self, encoder_lr=1e-5, predictor_lr=1e-3):
        return [
            {'params': self.drug_encoder_aq.parameters(), 'lr': encoder_lr},
            {'params': self.drug_encoder_bs.parameters(), 'lr': encoder_lr},
            {'params': self.solvent_encoder.parameters(), 'lr': encoder_lr},
            {'params': self.temp_embedding.parameters(), 'lr': predictor_lr},
            {'params': self.interaction_layer.parameters(), 'lr': predictor_lr},
            {'params': self.fusion_predictor.parameters(), 'lr': predictor_lr},
        ]
    
    def freeze_encoders(self):
        print("❄️  冻结 Encoders...")
        for enc in [self.drug_encoder_aq, self.drug_encoder_bs, self.solvent_encoder]:
            for p in enc.parameters(): p.requires_grad = False
    
    def unfreeze_encoders(self):
        print("🔥 解冻 Encoders...")
        for enc in [self.drug_encoder_aq, self.drug_encoder_bs, self.solvent_encoder]:
            for p in enc.parameters(): p.requires_grad = True

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        dx, de, db, sx, se, sb, t, y = batch
        dx, de, db = dx.to(device), de.to(device), db.to(device)
        sx, se, sb = sx.to(device), se.to(device), sb.to(device)
        t, y = t.to(device), y.to(device)
        
        optimizer.zero_grad()
        pred = model(dx, de, db, sx, se, sb, t)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * y.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds, trues = 0, [], []
    with torch.no_grad():
        for batch in loader:
            dx, de, db, sx, se, sb, t, y = batch
            dx, de, db = dx.to(device), de.to(device), db.to(device)
            sx, se, sb = sx.to(device), se.to(device), sb.to(device)
            t, y = t.to(device), y.to(device)
            
            pred = model(dx, de, db, sx, se, sb, t)
            total_loss += criterion(pred, y).item() * y.size(0)
            preds.extend(pred.cpu().numpy())
            trues.extend(y.cpu().numpy())
            
    preds = np.array(preds).flatten()
    trues = np.array(trues).flatten()
    
    ss_res = np.sum((trues - preds)**2)
    ss_tot = np.sum((trues - np.mean(trues))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    rmse = np.sqrt(np.mean((trues - preds)**2))
    mae = np.mean(np.abs(trues - preds))
    return total_loss / len(loader.dataset), r2, rmse, mae, preds, trues

def plot_results(history, preds, trues, save_dir, prefix):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].legend(); axes[0].set_title('Loss')
    axes[1].plot(history['val_r2'], color='green'); axes[1].set_title('R2')
    
    axes[2].scatter(trues, preds, alpha=0.5, s=10)
    mi, ma = min(trues.min(), preds.min()), max(trues.max(), preds.max())
    axes[2].plot([mi, ma], [mi, ma], 'r--')
    axes[2].set_title('Pred vs True')
    plt.savefig(os.path.join(save_dir, f'{prefix}_results.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data_2025.csv')
    parser.add_argument('--split', type=str, default='random', choices=['drug', 'random'])
    parser.add_argument('--pretrain', type=str, default='both', 
                        choices=['none', 'aqsoldb', 'bigsoldb', 'both'])
    parser.add_argument('--no-pretrain', action='store_true')
    parser.add_argument('--k-folds', type=int, default=5, help='K-Fold Cross Validation Folds')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--encoder-lr', type=float, default=1e-5)
    parser.add_argument('--predictor-lr', type=float, default=1e-3)
    parser.add_argument('--freeze-epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Run Info: {args.split} split, K-Folds={args.k_folds}, Pretrain={args.pretrain}, Device={device}")
    
    dataset = SolubilityDataset(os.path.join(BASE_DIR, args.data))
    
    k_fold_results = {'r2': [], 'rmse': [], 'mae': []}
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    if args.split == 'drug':
        drug_to_indices = defaultdict(list)
        for idx, drug in enumerate(dataset.drug_smiles_list):
            drug_to_indices[drug].append(idx)
        unique_drugs = list(drug_to_indices.keys())
        split_source = np.array(unique_drugs)
        print(f"Executing Drug-Based K-Fold on {len(unique_drugs)} unique drugs")
    else:
        split_source = np.arange(len(dataset))
        print("Executing Random K-Fold on all samples")

    for fold, (train_split_idx, val_split_idx) in enumerate(kf.split(split_source)):
        print(f"\n{'='*20} Fold {fold+1}/{args.k_folds} {'='*20}")
        
        if args.split == 'drug':
            train_drugs = set(split_source[train_split_idx])
            val_drugs = set(split_source[val_split_idx])
            
            train_idx = []
            val_idx = []
            
            for drug in train_drugs:
                train_idx.extend(drug_to_indices[drug])
            for drug in val_drugs:
                val_idx.extend(drug_to_indices[drug])
        else:
            train_idx = split_source[train_split_idx]
            val_idx = split_source[val_split_idx]
            
        train_loader = DataLoader(Subset(dataset, train_idx), 
                                  batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(Subset(dataset, val_idx), 
                                batch_size=args.batch_size, collate_fn=collate_fn)
        
        model = DualBranchFusionModel(hidden_dim=128, num_layers=4, dropout=0.2).to(device)
        
        pretrain_mode = args.pretrain
        if args.no_pretrain:
            pretrain_mode = 'none'
            
        if pretrain_mode != 'none':
            model.load_pretrained(device, mode=pretrain_mode)
            if args.freeze_epochs > 0:
                model.freeze_encoders()
        
        params = filter(lambda p: p.requires_grad, model.parameters())
        opt = optim.AdamW(params, lr=args.predictor_lr, weight_decay=1e-5)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)
        crit = nn.MSELoss()
        
        history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
        best_loss = float('inf')
        best_metrics = None
        best_state = None
        
        for epoch in range(args.epochs):
            if pretrain_mode != 'none' and epoch == args.freeze_epochs:
                model.unfreeze_encoders()
                param_groups = model.get_param_groups(args.encoder_lr, args.predictor_lr)
                opt = optim.AdamW(param_groups, weight_decay=1e-5)
            
            tr_loss = train_epoch(model, train_loader, opt, crit, device)
            val_loss, val_r2, val_rmse, val_mae, _, _ = evaluate(model, val_loader, crit, device)
            sched.step(val_loss)
            
            history['train_loss'].append(tr_loss)
            history['val_loss'].append(val_loss)
            history['val_r2'].append(val_r2)
            
            frozen = " [❄️]" if (pretrain_mode != 'none' and epoch < args.freeze_epochs) else ""
            if (epoch + 1) % 10 == 0:
                print(f"Ep {epoch+1}{frozen}: Tr={tr_loss:.4f}, Val={val_loss:.4f}, R2={val_r2:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_metrics = (val_r2, val_rmse, val_mae)
                best_state = model.state_dict().copy()
        
        print(f"Fold {fold+1} Best: R2={best_metrics[0]:.4f}, RMSE={best_metrics[1]:.4f}")
        k_fold_results['r2'].append(best_metrics[0])
        k_fold_results['rmse'].append(best_metrics[1])
        k_fold_results['mae'].append(best_metrics[2])
        
        exp_name = f"{args.split}_{pretrain_mode}_v6_fold{fold+1}"
        torch.save(best_state, os.path.join(MODEL_DIR, f"{exp_name}.pth"))
        
        model.load_state_dict(best_state)
        _, _, _, _, preds, trues = evaluate(model, val_loader, crit, device)
        plot_results(history, preds, trues, RESULT_DIR, exp_name)

    print("\n" + "="*60)
    print("K-Fold Cross Validation Results")
    print("="*60)
    mean_r2 = np.mean(k_fold_results['r2'])
    std_r2 = np.std(k_fold_results['r2'])
    mean_rmse = np.mean(k_fold_results['rmse'])
    std_rmse = np.std(k_fold_results['rmse'])
    
    print(f"R2:   {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
    print(f"MAE:  {np.mean(k_fold_results['mae']):.4f}")
    print("="*60)

if __name__ == "__main__":
    main()