import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import joblib
import json
import sys
sys.path.append('.')

# Correct Imports for Models
from mfp.some_mfp.models import SolubilityPredictor as MfpPredictor
from gcn.models import GCNPredictor
from smiles.models import SmilesPredictor

# Other necessary imports
from mfp.some_mfp.train import SolubilityDataset as MFDataset
from smiles.train import SmilesDataset
from gcn.preprocess import smiles_to_graph # This is used for smiles_to_graph helper function

def generate_multimodal_features():
    print("Loading all datasets and trained base models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load Data ---
    # Using the paths we standardized on
    mfp_dataset = MFDataset('mfp/some_mfp/features.csv', 'data/target.csv')
    gcn_dataset = torch.load('gcn/data.pt')
    smiles_dataset = SmilesDataset('smiles/data.pt')

    # Use batch_size=1 and shuffle=False to process in order
    mfp_loader = DataLoader(mfp_dataset, batch_size=1, shuffle=False)
    # The collate_fn is needed because the default collate function can't handle PyG's Data objects in a list
    gcn_loader = DataLoader(gcn_dataset, batch_size=1, shuffle=False, collate_fn=lambda batch: batch[0]) 
    smiles_loader = DataLoader(smiles_dataset, batch_size=1, shuffle=False)

    # --- Load Models ---
    model_mfp = MfpPredictor(4851).to(device)
    model_mfp.load_state_dict(torch.load('mfp/some_mfp/saved_model.pth'))
    model_mfp.eval()

    model_gcn = GCNPredictor(node_features=14, hidden_channels=64, graph_out_features=64).to(device)
    model_gcn.load_state_dict(torch.load('gcn/saved_model.pth'))
    model_gcn.eval()

    with open('smiles/vocab.json', 'r') as f:
        vocab_size = len(json.load(f))
    model_smiles = SmilesPredictor(vocab_size=vocab_size, embedding_dim=128, hidden_dim=128).to(device)
    model_smiles.load_state_dict(torch.load('smiles/saved_model.pth'))
    model_smiles.eval()

    # --- Generate Features ---
    multimodal_features = []
    solubilities = []
    
    print("Extracting features from base models...")
    with torch.no_grad():
        for mfp_batch, gcn_batch, smiles_batch in tqdm(zip(mfp_loader, gcn_loader, smiles_loader), total=len(mfp_dataset)):
            # MFP features
            mfp_features, _ = mfp_batch
            extracted_mfp = model_mfp(mfp_features.to(device), extract_features=True)

            # GCN features
            drug_graph = gcn_batch['drug_graph'].to(device)
            solvent_graph = gcn_batch['solvent_graph'].to(device)
            temp_gcn = gcn_batch['temperature'].to(device)
            extracted_gcn = model_gcn(drug_graph, solvent_graph, temp_gcn, extract_features=True)

            # SMILES features
            drug_seq = smiles_batch['drug_seq'].to(device)
            solvent_seq = smiles_batch['solvent_seq'].to(device)
            temp_smiles = smiles_batch['temperature'].to(device)
            extracted_smiles = model_smiles(drug_seq, solvent_seq, temp_smiles, extract_features=True)
            
            fused_features = torch.cat([extracted_mfp, extracted_gcn, extracted_smiles], dim=1)
            multimodal_features.append(fused_features.cpu())
            
            solubilities.append(smiles_batch['solubility'])

    all_features = torch.cat(multimodal_features, dim=0)
    all_solubilities = torch.cat(solubilities, dim=0)
    
    print(f"Generated multimodal features with shape: {all_features.shape}")
    
    multimodal_dataset = TensorDataset(all_features, all_solubilities)
    torch.save(multimodal_dataset, 'multimodal/features.pt')
    
    print("Multimodal feature generation complete. Saved to 'multimodal/features.pt'.")

if __name__ == '__main__':
    generate_multimodal_features()
