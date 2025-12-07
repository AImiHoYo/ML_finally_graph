import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
import json
from tqdm import tqdm
import sys
sys.path.append('.')
from smiles.models import SmilesPredictor # Import from models.py

class SmilesDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train_smiles_model(data_path, vocab_path):
    with open(vocab_path, 'r') as f:
        vocab_size = len(json.load(f))
    
    dataset = SmilesDataset(data_path)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SmilesPredictor(vocab_size=vocab_size, embedding_dim=128, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    print("Starting SMILES sequence model training...")
    for epoch in range(1, 31): # 30 epochs might be enough for this model
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            drug_seq = batch['drug_seq'].to(device)
            solvent_seq = batch['solvent_seq'].to(device)
            temp = batch['temperature'].to(device)
            solubility = batch['solubility'].to(device)
            
            optimizer.zero_grad()
            out = model(drug_seq, solvent_seq, temp) # Forward pass is still the same
            loss = criterion(out, solubility)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * drug_seq.size(0)
        
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
        
    print("Training complete. Evaluating on test set...")
    model.eval()
    total_mse = 0
    with torch.no_grad():
        for batch in test_loader:
            drug_seq = batch['drug_seq'].to(device)
            solvent_seq = batch['solvent_seq'].to(device)
            temp = batch['temperature'].to(device)
            solubility = batch['solubility'].to(device)
            
            out = model(drug_seq, solvent_seq, temp)
            total_mse += nn.functional.mse_loss(out, solubility, reduction='sum').item()
            
    avg_mse = total_mse / len(test_loader.dataset)
    print(f"Mean Squared Error on test set: {avg_mse:.4f}")
    
    torch.save(model.state_dict(), 'smiles/saved_model.pth')
    print("SMILES sequence model saved to 'smiles/saved_model.pth'.")

if __name__ == '__main__':
    train_smiles_model('smiles/data.pt', 'smiles/vocab.json')