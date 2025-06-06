import os
import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

class GeneratedDescToxicDataset(TorchDataset):
    def __init__(self, endpoint_dir, descriptor_names=None):
        super(GeneratedDescToxicDataset, self).__init__()
        self.endpoint_name = os.path.basename(endpoint_dir)
        self.train_df = pd.read_csv(f'{endpoint_dir}/{self.endpoint_name}_train_with_descriptors.csv')
        self.val_df = pd.read_csv(f'{endpoint_dir}/{self.endpoint_name}_val_with_descriptors.csv')
        self.test_df = pd.read_csv(f'{endpoint_dir}/{self.endpoint_name}_test_with_descriptors.csv')

        self.train_df.fillna(0)
        self.test_df.fillna(0)
        self.val_df.fillna(0)

        if descriptor_names:
            self.descriptor_cols = descriptor_names
        else:
            self.descriptor_cols = [col for col in self.train_df.columns if col not in ['smiles_standarized', 'label', 'group']]

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

    def smiles_to_data(self, smiles, label, global_desc):
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

        global_desc_array = np.array(global_desc.values if hasattr(global_desc, 'values') else global_desc,
                                     dtype=np.float32).reshape(1, -1)
        global_desc_tensor = torch.tensor(global_desc_array, dtype=torch.float)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([label], dtype=torch.float),
            descriptors=global_desc_tensor)

        return data

    def process_dataframe(self, df):
        data_list = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing molecules"):
            data = self.smiles_to_data(row['smiles_standarized'], row['label'], row[self.descriptor_cols])
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


if __name__ == '__main__':
    dataset = GeneratedDescToxicDataset('dataset_with_desc/eye_corrosion')
    dataset.get_dataloaders()