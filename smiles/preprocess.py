import pandas as pd
import torch
import json
from tqdm import tqdm
import sys
sys.path.append('.')

def build_vocab(smiles_series):
    """Builds a vocabulary from a series of SMILES strings."""
    chars = set()
    for smiles in smiles_series:
        if isinstance(smiles, str):
            chars.update(list(smiles))
    
    # Add special tokens
    vocab = {'<pad>': 0, '<unk>': 1}
    for i, char in enumerate(sorted(list(chars)), start=2):
        vocab[char] = i
        
    return vocab

def smiles_to_sequence(smiles, vocab, max_len):
    """Converts a SMILES string to a padded sequence of integers."""
    sequence = [vocab.get(char, vocab['<unk>']) for char in smiles]
    
    # Padding
    padded_sequence = sequence + [vocab['<pad>']] * (max_len - len(sequence))
    
    # Truncating
    return padded_sequence[:max_len]

def preprocess_smiles_data(file_path):
    print("Loading data for SMILES sequence model preprocessing...")
    try:
        df = pd.read_csv(file_path, encoding='gbk')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin1')
    
    df.columns = df.columns.str.strip()
    
    # --- Vocabulary ---
    print("Building vocabulary...")
    all_smiles = pd.concat([df['Drug SMILES'], df['Solvent SMILES']]).dropna().unique()
    vocab = build_vocab(all_smiles)
    with open('smiles/vocab.json', 'w') as f:
        json.dump(vocab, f)
    print(f"Vocabulary built with {len(vocab)} tokens and saved to 'smiles/vocab.json'.")

    # --- Tokenization and Padding ---
    max_len_drug = df['Drug SMILES'].str.len().max()
    max_len_solvent = df['Solvent SMILES'].str.len().max()
    max_len = max(max_len_drug, max_len_solvent)
    print(f"Max sequence length set to: {max_len}")
    
    processed_data = []
    print("Converting SMILES to padded sequences...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        drug_smiles = row['Drug SMILES']
        solvent_smiles = row['Solvent SMILES']
        
        if not isinstance(drug_smiles, str) or not isinstance(solvent_smiles, str):
            continue
            
        drug_seq = smiles_to_sequence(drug_smiles, vocab, max_len)
        solvent_seq = smiles_to_sequence(solvent_smiles, vocab, max_len)
        
        temperature = torch.tensor([row['Temperature']], dtype=torch.float)
        solubility = torch.tensor([row['Solubility_log10']], dtype=torch.float)
        
        processed_data.append({
            'drug_seq': torch.tensor(drug_seq, dtype=torch.long),
            'solvent_seq': torch.tensor(solvent_seq, dtype=torch.long),
            'temperature': temperature,
            'solubility': solubility
        })
        
    print(f"Successfully processed {len(processed_data)} data points.")
    print("Saving processed SMILES sequence data to 'smiles/data.pt'...")
    torch.save(processed_data, 'smiles/data.pt')
    # Save max_len for use in prediction
    with open('smiles/config.json', 'w') as f:
        json.dump({'max_len': max_len}, f)
    print("SMILES sequence preprocessing complete.")

if __name__ == '__main__':
    preprocess_smiles_data('data/data_2025.csv')
