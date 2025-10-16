import os
from tqdm import tqdm

import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.transforms import ToUndirected


class DataSet(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [""]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        raw_folder = os.path.join(self.root, "raw")

        data_list = []
        for target in tqdm(os.listdir((raw_folder))):
            target_path = os.path.join(raw_folder, target)
            if target not in affinity_dict.keys():
                print(f"Binding affinity not found for target {target}. Skipping.")
                continue
            affinity = float(affinity_dict[target])
            data = self.process_data_point(
                target_path, bbox_size=self.bbox_size, affinity=affinity
            )
            if data is not None:
                data_list.append(data)

        data_list = [data for data in data_list if data is not None]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    @staticmethod
    def process_data_point(target_path, bbox_size, affinity):
        try:
            ligand_path = os.path.join(target_path, "ligand.sdf")
            grail_path = os.path.join(target_path, "map.cdf")
            code = os.path.basename(target_path)

            # Read in sdf files and extract coordinates and types
            mol_reader = Chem.MoleculeReader(ligand_path)
            mol = Chem.BasicMolecule()
            mol_reader.read(mol)
            pos, types, edge_index, edge_attr = DataSet.ligand_to_tensor(mol)

            # Create a PyG Data object
            data = Data(
                x=types,
                pos=pos,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )
            transform = ToUndirected()
            data = transform(data)
            return data

        except Exception as e:
            print(f"Error processing {target_path}: {e}")
            return None

    @staticmethod
    def ligand_to_tensor(mol, conf_idx=None):
        num_atoms = mol.getNumAtoms()
        num_bonds = mol.getNumBonds()
        pos = torch.zeros((num_atoms, 3), dtype=torch.float32)
        types = torch.zeros((num_atoms, 1), dtype=torch.long)
        edge_index = torch.zeros((2, num_bonds), dtype=torch.long)
        edge_attr = torch.zeros((num_bonds, 5), dtype=torch.long)

        Chem.calcBasicProperties(mol, False)

        for (
            atom
        ) in (
            mol.atoms
        ):  # iterate of structure data entries consisting of a header line and the actual data
            idx = atom.getIndex()
            if conf_idx is not None:
                pos[idx] = torch.tensor(
                    Chem.getConformer3DCoordinates(atom, conf_idx).toArray(),
                    dtype=torch.float32,
                )
            else:
                pos[idx] = torch.tensor(
                    Chem.get3DCoordinates(atom).toArray(), dtype=torch.float32
                )
            types[idx] = Chem.getType(atom)

        for i, bond in enumerate(mol.bonds):
            edge_index[0, i] = bond.getBegin().getIndex()
            edge_index[1, i] = bond.getEnd().getIndex()
            edge_attr[i, Chem.getOrder(bond) - 1] = 1
            edge_attr[i, 3] = Chem.getAromaticityFlag(bond)
            edge_attr[i, 4] = Chem.getRingFlag(bond)

        return pos, types, edge_index, edge_attr
