import os
import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm


class ToxicDataset(TorchDataset):
    def __init__(self, endpoint_dir, desc_list):
        super(ToxicDataset, self).__init__()
        self.endpoint_name = os.path.basename(endpoint_dir)
        self.train_df = pd.read_csv(os.path.join(endpoint_dir, f'{self.endpoint_name}_train.csv'))
        self.val_df = pd.read_csv(os.path.join(endpoint_dir, f'{self.endpoint_name}_val.csv'))
        self.test_df = pd.read_csv(os.path.join(endpoint_dir, f'{self.endpoint_name}_test.csv'))

        self.desc_list = desc_list

    @staticmethod
    def atom_features(atom):
        return torch.tensor([
            atom.GetAtomicNum(),
            atom.GetTotalDegree(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            int(atom.GetIsAromatic()),
            atom.GetMass() * 0.01,
            atom.GetTotalNumHs(),
            int(atom.IsInRing()),
        ], dtype=torch.float)

    @staticmethod
    def bond_features(bond):
        return torch.tensor([
            bond.GetBondTypeAsDouble(),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
            int(bond.GetStereo()),
        ], dtype=torch.float)

    @staticmethod
    def get_rdkit_descriptors(mol, desc_list):
        descriptors = [getattr(Descriptors, desc)(mol) for desc in desc_list]
        return torch.tensor(descriptors, dtype=torch.float)

    def smiles_to_data(self, smiles, label):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None  # skip invalid molecules

        atom_feats = [self.atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.stack(atom_feats) if atom_feats else torch.zeros((1, 8))

        edge_index = []
        edge_attr = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            features = self.bond_features(bond)

            if features.shape[0] != 4:
                 continue

            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_attr.append(features)
            edge_attr.append(features)

        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.stack(edge_attr)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 3), dtype=torch.float)

        global_desc = self.get_rdkit_descriptors(mol, self.desc_list)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label], dtype=torch.float), global_desc=global_desc)

    def process_dataframe(self, df):
        data_list = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing molecules"):
            data = self.smiles_to_data(row['smiles_standarized'], row['label'])
            if data is not None:
                data_list.append(data)
        return data_list

    def get_dataloaders(self, batch_size=32, shuffle=True):
        train_data = self.process_dataframe(self.train_df)
        val_data = self.process_dataframe(self.val_df)
        test_data = self.process_dataframe(self.test_df)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
