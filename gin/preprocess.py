import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import SaltRemover, Draw
from torch_geometric.data import Data

remover = SaltRemover.SaltRemover()


def salt_remove_and_main_fragment(mol):
    try:
        m = remover.StripMol(mol)
    except:
        m = mol
    try:
        frags = Chem.GetMolFrags(m, asMols=True)
        if len(frags) > 1:
            m = max(frags, key=lambda x: x.GetNumAtoms())
    except:
        pass
    return m

def normalize_scores(scores):
    max_abs = max(abs(s) for s in scores) if any(s != 0 for s in scores) else 1.0
    return [s / max_abs for s in scores]

def scores_to_atom_color_map(norm_scores):
    return {i: ((1, 1 - s, 1 - s) if s > 0 else (1 + s, 1 + s, 1)) for i, s in enumerate(norm_scores)}

def get_atom_features_gin(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        atom.GetNumRadicalElectrons(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        atom.GetMass() / 100.0,
    ]


def smiles_to_graph_gin(smiles, clean_mol=True):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        if clean_mol:
            mol = salt_remove_and_main_fragment(mol)
        
        try:
            smiles_can = Chem.MolToSmiles(mol, canonical=True)
            mol = Chem.MolFromSmiles(smiles_can)
        except:
            pass
        
        atom_features = [get_atom_features_gin(atom) for atom in mol.GetAtoms()]
        if len(atom_features) == 0:
            return None
        
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.extend([[i, j], [j, i]])
        
        if len(edge_index) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        return Data(x=x, edge_index=edge_index)
    except:
        return None


def get_gin_features(drug_smiles, solvent_smiles, temp, temp_mean=298.0, temp_std=15.0):
    drug_graph = smiles_to_graph_gin(drug_smiles, clean_mol=True)
    solvent_graph = smiles_to_graph_gin(solvent_smiles, clean_mol=True)
    
    if drug_graph is None or solvent_graph is None:
        return None, None, None
    
    temp_normalized = (temp - temp_mean) / temp_std
    temp_tensor = torch.tensor([[temp_normalized]], dtype=torch.float)
    
    return drug_graph, solvent_graph, temp_tensor


def get_gin_atom_contributions(model, drug_smiles, solvent_smiles, temp, temp_mean, temp_std, device):
    try:
        mol = Chem.MolFromSmiles(drug_smiles)
        if mol is None:
            return None, None
        
        mol = salt_remove_and_main_fragment(mol)
        
        atom_features = [get_atom_features_gin(atom) for atom in mol.GetAtoms()]
        if len(atom_features) == 0:
            return None, None
        
        drug_x = torch.tensor(atom_features, dtype=torch.float, device=device, requires_grad=True)
        edge_index = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.extend([[i, j], [j, i]])
        if len(edge_index) == 0:
            drug_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        else:
            drug_edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
        
        solvent_graph = smiles_to_graph_gin(solvent_smiles, clean_mol=True)
        if solvent_graph is None:
            return None, None
        
        temp_normalized = (temp - temp_mean) / temp_std
        temp_tensor = torch.tensor([[temp_normalized]], dtype=torch.float, device=device)
        
        drug_batch = torch.zeros(drug_x.size(0), dtype=torch.long, device=device)
        
        solvent_x = solvent_graph.x.to(device)
        solvent_edge_index = solvent_graph.edge_index.to(device)
        solvent_batch = torch.zeros(solvent_x.size(0), dtype=torch.long, device=device)
        
        model.eval()
        with torch.enable_grad():
            pred = model(drug_x, drug_edge_index, drug_batch,
                         solvent_x, solvent_edge_index, solvent_batch, temp_tensor)
            pred.backward()
        
        if drug_x.grad is not None:
            atom_grads = drug_x.grad.detach().cpu()
            atom_inputs = drug_x.detach().cpu()
            grad_input = (atom_grads * atom_inputs).sum(dim=1).numpy()
            atom_scores = [float(grad_input[i]) for i in range(len(grad_input))]
        else:
            atom_scores = [0.0] * mol.GetNumAtoms()
        
        return atom_scores, mol
    
    except Exception as e:
        print(f"GIN atom contribution error: {e}")
        return None, None


def generate_gin_visualization(gin_atom_scores, gin_mol):
    gin_contrib_svg = ""
    gin_plain_svg = ""
    
    if gin_mol is None:
        return gin_contrib_svg, gin_plain_svg
    
    gin_plain_drawer = Draw.rdMolDraw2D.MolDraw2DSVG(300, 250)
    gin_plain_drawer.DrawMolecule(gin_mol)
    gin_plain_drawer.FinishDrawing()
    gin_plain_svg = gin_plain_drawer.GetDrawingText()
    
    if gin_atom_scores and any(s != 0 for s in gin_atom_scores):
        norm_scores = normalize_scores(gin_atom_scores)
        gin_atom_colors = scores_to_atom_color_map(norm_scores)
        
        gin_contrib_drawer = Draw.rdMolDraw2D.MolDraw2DSVG(300, 250)
        gin_contrib_drawer.DrawMolecule(
            gin_mol,
            highlightAtoms=list(range(gin_mol.GetNumAtoms())),
            highlightAtomColors=gin_atom_colors
        )
        gin_contrib_drawer.FinishDrawing()
        gin_contrib_svg = gin_contrib_drawer.GetDrawingText()
    
    return gin_contrib_svg, gin_plain_svg
