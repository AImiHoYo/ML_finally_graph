import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import sys
sys.path.append('.')
from multimodal.models import MultimodalPredictor

def train_multimodal_model(data_path):
    dataset = torch.load(data_path)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_size = dataset[0][0].shape[0]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalPredictor(input_size=input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    print("Starting Multimodal Fusion model training...")
    for epoch in range(1, 51):
        model.train()
        total_loss = 0
        for features, solubility in tqdm(train_loader, desc=f"Epoch {epoch}"):
            features = features.to(device)
            solubility = solubility.to(device)
            
            optimizer.zero_grad()
            out = model(features)
            loss = criterion(out, solubility)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * features.size(0)
        
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
        
    print("Training complete. Evaluating on test set...")
    model.eval()
    total_mse = 0
    with torch.no_grad():
        for features, solubility in test_loader:
            features = features.to(device)
            solubility = solubility.to(device)
            
            out = model(features)
            total_mse += nn.functional.mse_loss(out, solubility, reduction='sum').item()
            
    avg_mse = total_mse / len(test_loader.dataset)
    print(f"Mean Squared Error on test set: {avg_mse:.4f}")
    
    torch.save(model.state_dict(), 'multimodal/saved_model.pth')
    print("Multimodal Fusion model saved to 'multimodal/saved_model.pth'.")

if __name__ == '__main__':
    train_multimodal_model('multimodal/features.pt')
