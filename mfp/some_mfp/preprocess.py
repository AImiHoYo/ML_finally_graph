import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def generate_morgan_fingerprints(smiles_series, n_bits=2048):
    """Generate Morgan fingerprints."""
    fingerprints = []
    for smiles in smiles_series:
        mol = Chem.MolFromSmiles(str(smiles)) if smiles and pd.notna(smiles) else None
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
            fingerprints.append(fp.ToList())
        else:
            fingerprints.append([0] * n_bits)
    return np.array(fingerprints)

def generate_maccs_keys(smiles_series):
    """Generate MACCS keys."""
    keys = []
    for smiles in smiles_series:
        mol = Chem.MolFromSmiles(str(smiles)) if smiles and pd.notna(smiles) else None
        if mol:
            fp = MACCSkeys.GenMACCSKeys(mol)
            keys.append(fp.ToList())
        else:
            keys.append([0] * 167) # MACCS keys are 167 bits
    return np.array(keys)

def generate_rdkit_descriptors(smiles_series):
    """Generate RDKit 2D descriptors."""
    desc_names = [desc[0] for desc in Descriptors._descList]
    descriptors = []
    for smiles in smiles_series:
        mol = Chem.MolFromSmiles(str(smiles)) if smiles and pd.notna(smiles) else None
        if mol:
            desc_values = [desc[1](mol) for desc in Descriptors._descList]
            descriptors.append(desc_values)
        else:
            descriptors.append([0] * len(desc_names))
    return np.array(descriptors)

def preprocess_data_mfp(file_path):
    """Load, preprocess data with multiple fingerprints, and save."""
    print("Loading data for Multi-Fingerprint model...")
    try:
        df = pd.read_csv(file_path, encoding='gbk')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin1')

    df.columns = df.columns.str.strip()

    print("Generating Morgan fingerprints (Drug & Solvent)...")
    drug_morgan_fps = pd.DataFrame(generate_morgan_fingerprints(df['Drug SMILES']))
    solvent_morgan_fps = pd.DataFrame(generate_morgan_fingerprints(df['Solvent SMILES']))

    print("Generating MACCS keys (Drug & Solvent)...")
    drug_maccs_keys = pd.DataFrame(generate_maccs_keys(df['Drug SMILES']))
    solvent_maccs_keys = pd.DataFrame(generate_maccs_keys(df['Solvent SMILES']))

    print("Generating RDKit descriptors (Drug & Solvent)...")
    drug_descriptors = pd.DataFrame(generate_rdkit_descriptors(df['Drug SMILES']))
    solvent_descriptors = pd.DataFrame(generate_rdkit_descriptors(df['Solvent SMILES']))

    print("Combining all features...")
    temp_df = df[['Temperature']].copy()
    
    features_df = pd.concat([
        drug_morgan_fps, drug_maccs_keys, drug_descriptors,
        solvent_morgan_fps, solvent_maccs_keys, solvent_descriptors,
        temp_df.reset_index(drop=True)
    ], axis=1)

    features_df.columns = features_df.columns.astype(str)
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_df.fillna(0, inplace=True)

    # --- Fit scaler and save it ---
    # Important: In a real scenario, you would fit the scaler ONLY on the training data
    # and use it to transform both train and test sets.
    # For simplicity here, we fit on the whole dataset, which is not best practice but will work for this demo.
    print("Fitting StandardScaler and saving it...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)
    joblib.dump(scaler, 'mfp/some_mfp/scaler.joblib') # Save the scaler
    
    scaled_features_df = pd.DataFrame(scaled_features, columns=features_df.columns)
    
    print(f"Saving processed multi-fingerprint features... Shape: {scaled_features_df.shape}")
    scaled_features_df.to_csv('mfp/some_mfp/features.csv', index=False)

    print("Multi-Fingerprint preprocessing complete. 'mfp/some_mfp/features.csv' and 'mfp/some_mfp/scaler.joblib' created.")

if __name__ == '__main__':
    preprocess_data_mfp('data/data_2025.csv')