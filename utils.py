import torch
import numpy as np
import random
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

# https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097
def seed_all(seed=42):
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return None

# move torch/GPU tensor to numpy/CPU
def toCPU(data):
    return data.cpu().detach().numpy()

# count number of free parameters in the network
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sec2min_sec(t):
    mins = int(t) // 60
    secs = int(t) % 60
    
    return f'{mins}[min]{secs}[sec]'

def smiles2graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    
    return mol

if __name__ == '__main__':
    smi = "CCOC(=O)C1=C(CN2CCC(C)CC2)NC(=O)NC1c1ccco1"
    mol = smiles2graph(smi)
    print(mol)
    n_nodes = mol.GetNumAtoms()
    n_edges = 2*mol.GetNumBonds()
    print(n_nodes)
    print(n_edges)
    