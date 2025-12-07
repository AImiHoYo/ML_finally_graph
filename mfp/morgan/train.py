import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import sys
sys.path.append('.')
from mfp.morgan.models import SolubilityPredictor # Import from models.py

class MorganDataset(Dataset):
    def __init__(self, features_file, target_file):
        self.features = pd.read_csv(features_file).values
        self.target = pd.read_csv(target_file).values

    def __len__(self):
        return len(self.features) # Corrected line

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor(self.target[idx], dtype=torch.float32)
        return features, target

def train_morgan_model(features_file, target_file, input_size, num_epochs=50, batch_size=64, learning_rate=0.001):
    dataset = MorganDataset(features_file, target_file)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SolubilityPredictor(input_size=input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    print("Starting Morgan model re-training...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    print("Re-training complete. Evaluating on test set...")
    model.eval()
    total_mse = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            total_mse += nn.functional.mse_loss(outputs, labels, reduction='sum').item()
            
    avg_mse = total_mse / len(test_dataset)
    print(f"Mean Squared Error on test set: {avg_mse:.4f}")

    torch.save(model.state_dict(), 'mfp/morgan/saved_model.pth')
    print("Morgan model re-trained and saved to 'mfp/morgan/saved_model.pth'.")

if __name__ == '__main__':
    train_morgan_model('data/features.csv', 'data/target.csv', input_size=4097)