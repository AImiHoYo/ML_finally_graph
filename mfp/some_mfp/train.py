import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import sys
sys.path.append('.')
from mfp.some_mfp.models import SolubilityPredictor # Import from models.py

# 1. Dataset Class (remains the same)
class SolubilityDataset(Dataset):
    def __init__(self, features_file, target_file):
        self.features = pd.read_csv(features_file).values
        self.target = pd.read_csv(target_file).values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor(self.target[idx], dtype=torch.float32)
        return features, target

# 2. Model Definition is now in models.py

# 3. Training and Evaluation Function
def train_model(features_file, target_file, input_size, num_epochs=50, batch_size=64, learning_rate=0.001):
    dataset = SolubilityDataset(features_file, target_file)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = SolubilityPredictor(input_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for i, (features, labels) in enumerate(train_loader):
            outputs = model(features) # Forward pass is still the same
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        total_mse = 0
        for features, labels in test_loader:
            outputs = model(features)
            total_mse += criterion(outputs, labels).item() * len(labels)
        avg_mse = total_mse / len(test_dataset)
        print(f'Mean Squared Error on test set: {avg_mse:.4f}')

    torch.save(model.state_dict(), 'mfp/some_mfp/saved_model.pth')
    print("Model training complete and saved to mfp/some_mfp/saved_model.pth")

if __name__ == '__main__':
    input_size = 4851
    train_model('mfp/some_mfp/features.csv', 'data/target.csv', input_size)
