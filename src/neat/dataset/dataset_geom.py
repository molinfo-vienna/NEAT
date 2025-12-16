import logging
import os
import pickle
import subprocess

import networkx as nx
import torch
import yaml
from rdkit import RDLogger
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


RDLogger.DisableLog("rdApp.*")

SEED = 0


def process_molecule(smiles, conformers, vocabulary, num_conformers=30):
    try:
        conformer_list = []
        for mol in conformers:
            if len(conformer_list) >= num_conformers:
                break

            if mol is None:
                logging.warning(
                    f"RDKit molecule is None for {smiles}, skipping conformer."
                )
                continue

            # Create a tensor for atomic numbers and hybridization states
            x = torch.tensor(
                [vocabulary[atom.GetAtomicNum()] for atom in mol.GetAtoms()],
                dtype=torch.long,
            )

            # 3D coordinates centered at origin
            conf = mol.GetConformer()
            n = mol.GetNumAtoms()
            pos = torch.zeros((n, 3), dtype=torch.float32)
            for i in range(n):
                p = conf.GetAtomPosition(i)  # returns RDKit Point3D
                pos[i, 0] = p.x
                pos[i, 1] = p.y
                pos[i, 2] = p.z
            pos = pos - pos.mean(dim=0, keepdim=True)

            # Bond information for data augmentation during training
            # Note that the generation does not use bond information
            edge_index = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_index.append((i, j))
                edge_index.append((j, i))

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

            # Initialize molecular graph
            G = nx.Graph()
            for i, j in edge_index.t().tolist():
                G.add_edge(i, j)

            # Compute eccentricities for all nodes
            eccentricities = nx.eccentricity(G)
            eccentricity_tensor = torch.tensor(
                [eccentricities[node] for node in range(len(G.nodes))],
                dtype=torch.long,
            )

            data = Data(
                x=x,
                pos=pos,
                edge_index=edge_index,
                eccentricity=eccentricity_tensor,
                smiles=smiles,
            )

            conformer_list.append(data)

        return conformer_list

    except Exception as e:
        print(f"Error processing {smiles}: {e}")
        return None


class GEOMDataSet(InMemoryDataset):
    """GEOM dataset.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            torch_geometric.data.Data object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an torch_geometric.data.Data object and returns a transformed
            version. The data object will be transformed before being saved to
            disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in
            an torch_geometric.data.Data object and returns a boolean value,
            indicating whether the data object should be included in the final
            dataset. (default: :obj:`None`)
    """

    DRUGS_URL = "https://bits.csb.pitt.edu/files/geom_raw/"

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        split="train",
        num_conformers=30,
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        self.num_conformers = num_conformers
        if split == "train":
            self.load(self.processed_paths[0])
        elif split == "val":
            self.load(self.processed_paths[1])
        elif split == "test":
            self.load(self.processed_paths[2])
        else:
            raise ValueError(f"Unknown split: {split}")

    def download(self):
        raw_path = os.path.join(self.root, "raw")
        os.makedirs(raw_path, exist_ok=True)

        for raw_file in self.raw_file_names:
            if not os.path.exists(os.path.join(raw_path, raw_file)):
                subprocess.run(
                    [
                        "wget",
                        "-r",
                        "-np",
                        "-nH",
                        "--cut-dirs=2",
                        "--reject",
                        "index.html*",
                        "-P",
                        raw_path,
                        os.path.join(self.DRUGS_URL, raw_file),
                    ],
                    check=True,
                )

                print("Downloaded GEOM dataset.")

    @property
    def raw_file_names(self):
        return ["train_data.pickle", "val_data.pickle", "test_data.pickle"]

    @property
    def processed_file_names(self):
        return ["train_data.pt", "val_data.pt", "test_data.pt"]

    def process(self):
        # Load vocabulary YAML
        vocab_path = os.path.join(
            os.path.dirname(os.path.dirname(self.root)), "scripts", "geom_vocab.yaml"
        )
        try:
            with open(vocab_path, "r") as file:
                vocabulary = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f"Error loading vocabulary YAML file: {e}")

        for i, raw_path in enumerate(self.raw_paths):
            raw_path = self.raw_paths[i]
            with open(raw_path, "rb") as f:
                mol_list = pickle.load(f)

            print(f"Processing {len(mol_list)} molecules from {raw_path}...")

            data_list = []
            for smiles, conformers in tqdm(mol_list):
                mol_data = process_molecule(
                    smiles,
                    conformers,
                    vocabulary,
                    num_conformers=self.num_conformers,
                )
                if mol_data is not None:
                    data_list.extend(mol_data)

            data_list = [data for data in data_list if data is not None]

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            self.save(data_list, self.processed_paths[i])
