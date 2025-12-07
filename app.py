import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify, render_template
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, AdditionalOutput
from rdkit import RDLogger
from collections import defaultdict
import joblib
import json
from torch_geometric.data import Data

from mfp.some_mfp.models import SolubilityPredictor as MfpPredictor
from mfp.morgan.models import SolubilityPredictor as MorganPredictor
from gin.models import DualBranchFusionModel
from smiles.models import SmilesPredictor
from multimodal.models import MultimodalPredictor
from gin.preprocess import get_gin_features, get_gin_atom_contributions, generate_gin_visualization

# --- Helper Functions for Feature Generation ---
RDLogger.DisableLog('rdApp.warning')
_MORGAN_GEN = GetMorganGenerator(radius=2, fpSize=2048)

def get_morgan_features(drug_smiles, solvent_smiles, temp):
    drug_mol = Chem.MolFromSmiles(drug_smiles)
    solvent_mol = Chem.MolFromSmiles(solvent_smiles)
    drug_fp = list(_MORGAN_GEN.GetFingerprint(drug_mol)) if drug_mol else [0]*2048
    solvent_fp = list(_MORGAN_GEN.GetFingerprint(solvent_mol)) if solvent_mol else [0]*2048
    return np.array(drug_fp + solvent_fp + [temp])

def get_mfp_features(drug_smiles, solvent_smiles, temp, scaler):
    drug_mol = Chem.MolFromSmiles(drug_smiles)
    solvent_mol = Chem.MolFromSmiles(solvent_smiles)
    
    drug_morgan = np.array(list(_MORGAN_GEN.GetFingerprint(drug_mol))) if drug_mol else np.zeros(2048)
    solvent_morgan = np.array(list(_MORGAN_GEN.GetFingerprint(solvent_mol))) if solvent_mol else np.zeros(2048)
    drug_maccs = np.array(MACCSkeys.GenMACCSKeys(drug_mol).ToList()) if drug_mol else np.zeros(167)
    solvent_maccs = np.array(MACCSkeys.GenMACCSKeys(solvent_mol).ToList()) if solvent_mol else np.zeros(167)

    expected_len = len(getattr(scaler, 'feature_names_in_', [])) if hasattr(scaler, 'feature_names_in_') else 4851
    fixed_part = 2048*2 + 167*2 + 1
    expected_desc_len = max(0, int((expected_len - fixed_part) // 2))

    drug_descs_full = [desc[1](drug_mol) if drug_mol else 0.0 for desc in Descriptors._descList]
    solvent_descs_full = [desc[1](solvent_mol) if solvent_mol else 0.0 for desc in Descriptors._descList]
    if len(drug_descs_full) >= expected_desc_len:
        drug_descs = np.array(drug_descs_full[:expected_desc_len])
    else:
        drug_descs = np.pad(np.array(drug_descs_full, dtype=float), (0, expected_desc_len - len(drug_descs_full)), 'constant')
    if len(solvent_descs_full) >= expected_desc_len:
        solvent_descs = np.array(solvent_descs_full[:expected_desc_len])
    else:
        solvent_descs = np.pad(np.array(solvent_descs_full, dtype=float), (0, expected_desc_len - len(solvent_descs_full)), 'constant')

    features_vec = np.concatenate([
        drug_morgan, drug_maccs, drug_descs,
        solvent_morgan, solvent_maccs, solvent_descs,
        np.array([temp])
    ])

    features_vec = np.nan_to_num(features_vec, nan=0.0, posinf=0.0, neginf=0.0)

    if hasattr(scaler, 'feature_names_in_'):
        import pandas as pd
        cols = list(scaler.feature_names_in_)
        df = pd.DataFrame([features_vec], columns=cols)
        scaled = scaler.transform(df)
    else:
        scaled = scaler.transform(features_vec.reshape(1, -1))
    return scaled.flatten()

# get_gcn_features removed: GIN features are obtained using `gin.preprocess.get_gin_features`

def get_smiles_features(smiles, vocab, max_len):
    sequence = [vocab.get(char, vocab['<unk>']) for char in smiles]
    padded = sequence + [vocab['<pad>']] * (max_len - len(sequence))
    return torch.tensor(padded[:max_len], dtype=torch.long).unsqueeze(0)

def get_atom_contributions(model, smiles):
    """Compute simple atom importance for Morgan fingerprint model using the first layer weights.
    This function now aligns preprocessing with GIN's salt removal and fragment selection.
    """
    feature_weights = model.layer1.weight.data.sum(axis=0)
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None, None
    # Original behavior: do not alter mol here (no salt strip), keep default Morgan fingerprint mapping
    ao = AdditionalOutput()
    ao.AllocateBitInfoMap()
    _ = _MORGAN_GEN.GetFingerprint(mol, additionalOutput=ao)
    bit_info_map = ao.GetBitInfoMap()
    atom_to_bits = defaultdict(list)
    for bit_id, entries in bit_info_map.items():
        for center_atom, radius in entries:
            atom_to_bits[center_atom].append(bit_id)
    atom_scores = [0.0] * mol.GetNumAtoms()
    for atom_id, bit_list in atom_to_bits.items():
        score = sum(float(feature_weights[int(bit)]) for bit in bit_list if int(bit) < feature_weights.shape[0])
        atom_scores[atom_id] = score
    return atom_scores, mol

# --- Flask App & Model Loading ---
app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SOLVENT_DATA = {'tert-Pentyl alcohol': 'CCC(C)(C)O', 'Acetone': 'CC(=O)C', 'Carbon Tetrachloride': 'C(Cl)(Cl)(Cl)Cl', 'Ethylbenzene': 'CCC1=CC=CC=C1', 'Chlorobenzene': 'C1=CC=C(C=C1)Cl', 'Pentyl acetate': 'CCCCCOC(=O)C', '2-Methyl-1-Propanol': 'CC(C)CO', 'Diethylene glycol': 'C(COCCO)O', '1-Hexanol': 'CCCCCCO', 'N,N-Dimethylacetamide': 'CC(=O)N(C)C', 'Dimethyl Sulfoxide': 'CS(=O)C', 'Chloroform': 'C(Cl)(Cl)Cl', '1-Octanol': 'CCCCCCCCO', 'Tristearin': 'CCCCCCCCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCCCCCCCC)OC(=O)CCCCCCCCCCCCCCCCC', 'Nonane': 'CCCCCCCCC', '3-Methyl-1-butanol': 'CC(C)CCO', 'tert-Butyl Acetate': 'CC(=O)OC(C)(C)C', 'Methyl cyclohexane': 'COC(=O)C1CCCCC1', '2-Pyrrolidone': 'C1CC(=O)NC1', 'Cyclohexane': 'C1CCCCC1', 'Ethyl Formate': 'CCOC=O', 'Acetophenone': 'CC(=O)C1=CC=CC=C1', 'Butyl Acetate': 'CCCCOC(=O)C', 'Cyclopentane': 'C1CCCC1', 'Ethylene Dichloride': 'C(CCl)Cl', 'N-Methyl formamide': 'CNC=O', 'Pyridine': 'C1=CC=NC=C1', '2-Phenylethanol': 'C1=CC=C(C=C1)CCO', 'Tricaprylin': 'CCCCCCCC(=O)OCC(COC(=O)CCCCCCC)OC(=O)CCCCCCC', 'Ethylene Glycol': 'C(CO)O', 'Butyl butyrate': 'CCCCOC(=O)CCC', '1-Decanol': 'CCCCCCCCCCCO', '1-Propanol': 'CCCO', '2-Butanol': 'CCC(C)O', 'Dimethyl Formamide': 'CN(C)C=O', 'Anisole': 'COC1=CC=CC=C1', 'Butyl ether': 'CCCCOCCCC', 'Acetic Acid': 'CC(=O)O', 'Methyl Acetate': 'CC(=O)OC', 'Acetic anhydride': 'CC(=O)OC(=O)C', 'Ethyl ether': 'CCOCC', 'Hexyl acetate': 'CCCCCCOC(=O)C', 'Dodecane': 'CCCCCCCCCCCC', 'Diisopropyl ether': 'CC(C)OC(C)C', '1-Heptanol': 'CCCCCCCO', 'Triolein': 'CCCCCCCCC=CCCCCCCCC(=O)OCC(COC(=O)CCCCCCCC=CCCCCCCCC)OC(=O)CCCCCCCC=CCCCCCCCC', '1,3-Butanediol': 'CC(CCO)O', 'Tributyrin': 'CCCC(=O)OCC(COC(=O)CCC)OC(=O)CCC', '1-Butanol': 'CCCCO', '2,2,4-Trimethyl pentane': 'CC(C)CC(C)(C)C', 'Methanol': 'CO'}

# Load all models and supporting files
model_morgan = MorganPredictor(4097).to(device)
model_morgan.load_state_dict(torch.load('mfp/morgan/saved_model.pth'))
model_morgan.eval()

model_mfp = MfpPredictor(4851).to(device)
model_mfp.load_state_dict(torch.load('mfp/some_mfp/saved_model.pth'))
model_mfp.eval()
scaler_mfp = joblib.load('mfp/some_mfp/scaler.joblib')

model_gcn = DualBranchFusionModel(hidden_dim=128, num_layers=4, dropout=0.2).to(device)
_gin_weight_path = None
for p in ['gin/best_v5.pth', 'gin/gin.pth']:
    try:
        model_gcn.load_state_dict(torch.load(p, map_location=device))
        _gin_weight_path = p
        break
    except Exception as _e:
        pass
model_gcn.eval()
print(f"Loaded GIN weights: {_gin_weight_path}")

GCN_TEMP_MEAN = 30.02
GCN_TEMP_STD = 15.20
GCN_LOGS_MEAN = -1.263
GCN_LOGS_STD = 1.239

with open('smiles/vocab.json', 'r') as f: vocab = json.load(f)
with open('smiles/config.json', 'r') as f: smiles_config = json.load(f)
model_smiles = SmilesPredictor(vocab_size=len(vocab), embedding_dim=128, hidden_dim=128).to(device)
model_smiles.load_state_dict(torch.load('smiles/saved_model.pth'))
model_smiles.eval()

model_multimodal = MultimodalPredictor(input_size=384).to(device)
model_multimodal.load_state_dict(torch.load('multimodal/saved_model.pth'))
model_multimodal.eval()

# --- Routes ---
@app.route('/')
def landing(): return render_template('landing.html')
@app.route('/predictor')
def predictor():
    def sort_key(item):
        name = item[0]
        return (0 if name[:1].isdigit() else 1, name.lower())
    solvents_sorted = sorted(SOLVENT_DATA.items(), key=sort_key)
    return render_template('predictor.html', solvents_sorted=solvents_sorted)
@app.route('/analysis')
def analysis():
    try:
        df = pd.read_csv('data/data_2025.csv', encoding='gbk')
    except UnicodeDecodeError:
        df = pd.read_csv('data/data_2025.csv', encoding='latin1')
    solvent_counts = df['Solvent'].value_counts()
    inner_color_palette, outer_color_palette = ['#3498db', '#2ecc71', '#e74c3c', '#f1c40f', '#9b59b6', '#8e44ad', '#16a085', '#27ae60', '#f39c12', '#d35400'], ['#34495e', '#1abc9c', '#e67e22', '#95a5a6', '#c0392b', '#7f8c8d', '#bdc3c7', '#7b241c', '#a93226', '#cb4335']
    
    top_10 = solvent_counts.nlargest(10)
    inner_labels, inner_values = list(top_10.keys()), [int(v) for v in top_10.values]
    inner_colors = inner_color_palette[:len(inner_labels)]
    inner_legend_data = [{'label': l, 'value': v, 'color': c} for l, v, c in zip(inner_labels, inner_values, inner_colors)]
    
    next_10, others_count = solvent_counts.iloc[10:20], solvent_counts.iloc[20:].sum()
    outer_temp_data = next_10.to_dict()
    if others_count > 0: outer_temp_data['Others'] = others_count
    
    outer_labels, outer_values = list(outer_temp_data.keys()), [int(v) for v in outer_temp_data.values()]
    outer_colors = outer_color_palette[:len(outer_labels)]
    outer_legend_data = [{'label': l, 'value': v, 'color': c} for l, v, c in zip(outer_labels, outer_values, outer_colors)]
    
    return render_template('analysis.html', all_labels=inner_labels + outer_labels, inner_values=inner_values, inner_colors=inner_colors, outer_values=outer_values, outer_colors=outer_colors, inner_legend_data=inner_legend_data, outer_legend_data=outer_legend_data)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    drug_smiles, solvent_smiles, temperature = data.get('drug_smiles'), data.get('solvent_smiles'), float(data.get('temperature'))

    if not all([drug_smiles, solvent_smiles, Chem.MolFromSmiles(drug_smiles), Chem.MolFromSmiles(solvent_smiles)]):
        return jsonify({'error': 'Invalid SMILES string provided.'}), 400

    # --- Get features from base models ---
    with torch.no_grad():
        pred_morgan_val, pred_mfp_val, pred_gcn_val, pred_smiles_val, pred_multimodal_val = None, None, None, None, None
        # Morgan
        features_morgan = torch.tensor(get_morgan_features(drug_smiles, solvent_smiles, temperature), dtype=torch.float32).unsqueeze(0).to(device)
        pred_morgan_val = model_morgan(features_morgan).item()
        
        # MFP
        features_mfp = torch.tensor(get_mfp_features(drug_smiles, solvent_smiles, temperature, scaler_mfp), dtype=torch.float32).unsqueeze(0).to(device)
        extracted_mfp = model_mfp(features_mfp, extract_features=True)
        pred_mfp_val = model_mfp.output_layer(extracted_mfp).item()

        # GIN
        try:
            drug_graph_gin, solvent_graph_gin, temp_gin = get_gin_features(
                drug_smiles, solvent_smiles, temperature, GCN_TEMP_MEAN, GCN_TEMP_STD
            )
            if drug_graph_gin is not None and solvent_graph_gin is not None:
                dx = drug_graph_gin.x.to(device)
                de = drug_graph_gin.edge_index.to(device)
                db = torch.zeros(dx.size(0), dtype=torch.long, device=device)
                
                sx = solvent_graph_gin.x.to(device)
                se = solvent_graph_gin.edge_index.to(device)
                sb = torch.zeros(sx.size(0), dtype=torch.long, device=device)
                
                t_gin = temp_gin.to(device)
                
                pred_norm = model_gcn(dx, de, db, sx, se, sb, t_gin).item()
                pred_gcn_val = pred_norm * GCN_LOGS_STD + GCN_LOGS_MEAN
                print(f"GIN debug -> drug_atoms: {dx.size(0)}, solvent_atoms: {sx.size(0)}, temp_norm: {t_gin.item():.4f}, pred_norm: {pred_norm:.4f}, pred_denorm: {pred_gcn_val:.4f}")
        except Exception as e:
            print(f"GCN (GIN) model error: {e}")

        # SMILES
        drug_seq = get_smiles_features(drug_smiles, vocab, smiles_config['max_len']).to(device)
        solvent_seq = get_smiles_features(solvent_smiles, vocab, smiles_config['max_len']).to(device)
        temp_smiles = torch.tensor([temperature], dtype=torch.float).unsqueeze(0).to(device)
        extracted_smiles = model_smiles(drug_seq, solvent_seq, temp_smiles, extract_features=True)
        pred_smiles_val = model_smiles.output_layer(extracted_smiles).item()
        
        # --- Multimodal Prediction ---
        pred_multimodal_val = None

    # --- Visualizations ---
    atom_scores, mol = None, None
    try:
        atom_scores, mol = get_atom_contributions(model_morgan, drug_smiles)
    except Exception as e:
        print(f"Morgan atom contribution error: {e}")
        atom_scores, mol = None, None
    contrib_svg, plain_svg = "", ""
    if mol:
        plain_drawer = Draw.rdMolDraw2D.MolDraw2DSVG(300, 250); plain_drawer.DrawMolecule(mol); plain_drawer.FinishDrawing(); plain_svg = plain_drawer.GetDrawingText()
        if atom_scores:
            max_abs = max(abs(c) for c in atom_scores) if any(c != 0 for c in atom_scores) else 1
            norm_scores = [c / max_abs for c in atom_scores]
            atom_colors = {i: (1, 1-s, 1-s) if s > 0 else (1+s, 1+s, 1) for i, s in enumerate(norm_scores)}
            contrib_drawer = Draw.rdMolDraw2D.MolDraw2DSVG(300, 250); contrib_drawer.DrawMolecule(mol, highlightAtoms=list(range(mol.GetNumAtoms())), highlightAtomColors=atom_colors); contrib_drawer.FinishDrawing(); contrib_svg = contrib_drawer.GetDrawingText()

    def classify(score):
        if score is None: return "N/A", "#95a5a6"
        if score > 0: return "High", "#27ae60"
        if score >= -3: return "Medium", "#f39c12"
        return "Low", "#c0392b"
    
    gin_contrib_svg, gin_plain_svg = "", ""
    orig_dev = next(model_gcn.parameters()).device
    cpu_dev = torch.device('cpu')
    try:
        model_gcn.to(cpu_dev)
        gin_atom_scores, gin_mol = get_gin_atom_contributions(
            model_gcn, drug_smiles, solvent_smiles, temperature, 
            GCN_TEMP_MEAN, GCN_TEMP_STD, cpu_dev
        )
        gin_contrib_svg, gin_plain_svg = generate_gin_visualization(gin_atom_scores, gin_mol)
    except Exception as e:
        print(f"GIN visualization error: {e}")
    finally:
        try:
            model_gcn.to(orig_dev)
        except:
            pass

    return jsonify({
        'morgan': {'prediction': pred_morgan_val, 'level': classify(pred_morgan_val)[0], 'color': classify(pred_morgan_val)[1], 'contrib_svg': contrib_svg},
        'mfp': {'prediction': pred_mfp_val, 'level': classify(pred_mfp_val)[0], 'color': classify(pred_mfp_val)[1]},
        'gcn': {'prediction': pred_gcn_val, 'level': classify(pred_gcn_val)[0], 'color': classify(pred_gcn_val)[1], 'gin_contrib_svg': gin_contrib_svg, 'gin_plain_svg': gin_plain_svg},
        'smiles': {'prediction': pred_smiles_val, 'level': classify(pred_smiles_val)[0], 'color': classify(pred_smiles_val)[1]},
        'multimodal': {'prediction': pred_multimodal_val, 'level': classify(pred_multimodal_val)[0], 'color': classify(pred_multimodal_val)[1]},
        'plain_svg': plain_svg
    })

if __name__ == '__main__':
    app.run(debug=True)
