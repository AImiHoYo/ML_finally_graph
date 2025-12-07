import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, LayerNorm, Module, ModuleList, Sequential, ReLU, Dropout
from torch_geometric.nn import GINConv, global_mean_pool

class GINEncoder(Module):
    def __init__(self, num_features=9, hidden_dim=128, num_layers=4, dropout=0.2):
        super(GINEncoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.atom_embedding = Linear(num_features, hidden_dim)
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        
        for _ in range(num_layers):
            mlp = Sequential(
                Linear(hidden_dim, hidden_dim * 2),
                BatchNorm1d(hidden_dim * 2),
                ReLU(),
                Linear(hidden_dim * 2, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(BatchNorm1d(hidden_dim))
    
    def forward(self, x, edge_index, batch):
        x = self.atom_embedding(x)
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)


class DualBranchFusionModel(Module):
    def __init__(self, hidden_dim=128, num_layers=4, dropout=0.2):
        super(DualBranchFusionModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        self.drug_encoder_aq = GINEncoder(9, hidden_dim, num_layers, dropout)
        self.drug_encoder_bs = GINEncoder(9, hidden_dim, num_layers, dropout)
        self.solvent_encoder = GINEncoder(9, hidden_dim, num_layers, dropout)
        
        self.temp_embedding = Sequential(Linear(1, 32), ReLU(), Linear(32, 32))
        
        self.interaction_layer = Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            LayerNorm(hidden_dim),
            ReLU(),
            Dropout(dropout)
        )
        
        self.fusion_predictor = Sequential(
            Linear(hidden_dim * 2 + 32, hidden_dim),
            LayerNorm(hidden_dim),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, hidden_dim // 2),
            LayerNorm(hidden_dim // 2),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, dx, de, db, sx, se, sb, t):
        drug_emb_aq = self.drug_encoder_aq(dx, de, db)
        drug_emb_bs = self.drug_encoder_bs(dx, de, db)
        solvent_emb = self.solvent_encoder(sx, se, sb)
        
        interaction_input = torch.cat([drug_emb_bs, solvent_emb], dim=1)
        interaction_emb = self.interaction_layer(interaction_input)
        temp_emb = self.temp_embedding(t)
        
        fusion = torch.cat([drug_emb_aq, interaction_emb, temp_emb], dim=1)
        return self.fusion_predictor(fusion)
