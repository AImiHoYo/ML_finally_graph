import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

def get_morgan_features(drug_smiles, solvent_smiles, temp):
    drug_mol = Chem.MolFromSmiles(drug_smiles)
    drug_fp = AllChem.GetMorganFingerprintAsBitVect(drug_mol, 2, nBits=2048).ToList() if drug_mol else [0]*2048
    solvent_mol = Chem.MolFromSmiles(solvent_smiles)
    solvent_fp = AllChem.GetMorganFingerprintAsBitVect(solvent_mol, 2, nBits=2048).ToList() if solvent_mol else [0]*2048
    return np.array(drug_fp + solvent_fp + [temp])

def preprocess_morgan(input_file, features_out, target_out):
    print("Loading data for Morgan preprocessing...")
    try:
        df = pd.read_csv(input_file, encoding='gbk')
    except UnicodeDecodeError:
        df = pd.read_csv(input_file, encoding='latin1')
    
    df.columns = df.columns.str.strip()
    
    features_list = []
    targets_list = []
    
    print("Generating Morgan fingerprints...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        drug_smiles = row['Drug SMILES']
        solvent_smiles = row['Solvent SMILES']
        temp = row['Temperature']
        solubility = row['Solubility_log10']
        
        # Ensure valid data
        if pd.isna(drug_smiles) or pd.isna(solvent_smiles) or pd.isna(temp) or pd.isna(solubility):
            continue
            
        features = get_morgan_features(drug_smiles, solvent_smiles, temp)
        features_list.append(features)
        targets_list.append(solubility)
        
    # Convert to DataFrames and save
    print("Saving features and targets...")
    
    # Ensure all feature arrays have the same length (4097)
    features_array = np.array(features_list)
    pd.DataFrame(features_array).to_csv(features_out, index=False)
    pd.DataFrame(targets_list, columns=['Solubility_log10']).to_csv(target_out, index=False)
    
    print("Morgan preprocessing complete.")

if __name__ == '__main__':
    preprocess_morgan('data/data_2025.csv', 'data/features.csv', 'data/target.csv')
