import torch
import torch_geometric as pyg
from torch_geometric.data import Dataset
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import GetAdjacencyMatrix
from rdkit import RDLogger
import numpy as np

RDLogger.DisableLog('rdApp.*')

def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """

    if x not in permitted_list:
        x = permitted_list[-1]

    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]

    return binary_encoding

class SynthGraphDataset(Dataset):
    def __init__(self, 
                 data_file='data/cid-smile-label170.txt',
                 preprocessed_path=None):
        
        if preprocessed_path is None:
            self.raw_data = []
            with open(data_file) as f:
                lines = f.readlines()
            for line in lines:
                cid, smiles, label = line.strip().split()
                self.raw_data.append([cid, smiles, int(label)])
            self.graphs = self.gen_graphs(self.raw_data)
        else:
            self.graphs = self.load_graphs(preprocessed_path)
    
    
    def __getitem__(self, idx):
        return self.graphs[idx]
    
    def __len__(self):
        return len(self.graphs)

    def gen_graphs(self, raw_data):
        mol_graphs = []
        for cid, smiles, label in tqdm(self.raw_data, desc='gen graphs', dynamic_ncols=True):
            try:
                graph = self.smiles2graph(smiles, cid, label)
                mol_graphs.append(graph)
            except:
                pass
            
        print(f'Number of generated molecule graphs: {len(mol_graphs)}')
        
        return mol_graphs
    
    def save_graphs(self, save_path):
        save_data = []
        for data in self.graphs:
            save_data.append({'x': data.x,
                              'edge_index': data.edge_index,
                              'edge_attr': data.edge_attr,
                              'y': data.y,
                              'cid': data.cid})
        print(f'save dataset to {save_path}')
        torch.save(save_data, save_path)
    
    def load_graphs(self, preprocessed_path):
        all_data = torch.load(preprocessed_path)
        all_graphs = []
        print(f'Loading graphs from {preprocessed_path}')
        for data in tqdm(all_data, desc='loading graphs', dynamic_ncols=True):
            graph = pyg.data.Data(x=data['x'], 
                                  edge_index=data['edge_index'], 
                                  edge_attr=data['edge_attr'], 
                                  y=data['y'].type(torch.long),
                                  cid=data['cid'])
            all_graphs.append(graph)
        
        print(f'number of loaded molecule graphs: {len(all_graphs)}')
        return all_graphs
    
    # TODO: 
    # https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/
    def smiles2graph(self, smiles, cid, label):
        # construct rdkit molecule from smiles
        mol = Chem.MolFromSmiles(smiles)
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        
        # get feature dimensions
        n_nodes = mol.GetNumAtoms()
        n_edges = 2*mol.GetNumBonds()
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(self.get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(self.get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))
        
        # construct node feature matrix X of shape (n_nodes, n_node_features)
        X = np.zeros((n_nodes, n_node_features))
        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = self.get_atom_features(atom)
        X = torch.tensor(X, dtype = torch.float32)
        
        # construct edge index array E of shape (2, n_edges)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim = 0)
        
        # construct edge feature array EF of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, n_edge_features))
        for (k, (i,j)) in enumerate(zip(rows, cols)):
            EF[k] = self.get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
        EF = torch.tensor(EF, dtype = torch.float32)
        
        # construct y tensor
        y_tensor = torch.tensor(label, dtype=torch.long)
        
        data = pyg.data.Data(x=X, 
                             edge_index=E, 
                             edge_attr=EF, 
                             y=y_tensor,
                             cid=cid)
        
        return data
        
    def get_atom_features(self, atom, use_chirality = True, hydrogens_implicit = True):
        """
        Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
        """

        # define list of permitted atoms
        
        permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']
        
        if hydrogens_implicit == False:
            permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
        
        # compute atom features
        
        atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
        
        n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
        
        formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
        
        hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
        
        is_in_a_ring_enc = [int(atom.IsInRing())]
        
        is_aromatic_enc = [int(atom.GetIsAromatic())]
        
        atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
        
        vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
        
        covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]

        atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled
                                        
        if use_chirality == True:
            chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
            atom_feature_vector += chirality_type_enc
        
        if hydrogens_implicit == True:
            n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
            atom_feature_vector += n_hydrogens_enc

        return np.array(atom_feature_vector)

    def get_bond_features(self, bond, use_stereochemistry = True):
        """
        Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
        """

        permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

        bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
        
        bond_is_conj_enc = [int(bond.GetIsConjugated())]
        
        bond_is_in_ring_enc = [int(bond.IsInRing())]
        
        bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
        
        if use_stereochemistry == True:
            stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
            bond_feature_vector += stereo_type_enc

        return np.array(bond_feature_vector)

    # TODO
    def get_idx2label(self):
        idx2label = []
        for data in self.graphs:
            idx2label.append(data.y)

        return idx2label

if __name__ == '__main__':
    dataset = SynthGraphDataset()
    print(dataset[0])
    dataset.save_graphs(save_path='data/pubchem340k.pt')
    
    # loaded_dataset = SynthGraphDataset(preprocessed_path='data/pubchem340k.pt')
    # for i in range(1):
    #     print(loaded_dataset[i])
    #     print(loaded_dataset[i].x)
    #     print(loaded_dataset[i].y, loaded_dataset[i].y.dtype)
    
