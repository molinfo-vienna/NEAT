import logging
import os
from typing import Dict

import networkx as nx
import numpy as np
import torch
import yaml
from rdkit import RDLogger
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

import random
import pickle
import subprocess

import os
import pickle
from tqdm import tqdm
import os
import pickle
import json, urllib

RDLogger.DisableLog("rdApp.*")

SEED = 0


def process_molecule(mol_dict, vocabulary, num_conformers=30):
    try:
        conformers_info = mol_dict.get("conformers", [])
        if len(conformers_info) == 0:
            logging.warning(
                f"No conformers found for {mol_dict['smiles']}, skipping molecule."
            )
            return None

        energies = np.array(
            [conformer["totalenergy"] for conformer in conformers_info],
            dtype=float,
        )
        # Sort conformers by energy and select the top `num_conformers`
        # This is probably not necessary, since the conformers should
        # already be sorted by energy, but just in case.
        order = np.argsort(energies)
        keep = order[:num_conformers]

        data_list = []  # to store Data objects for each conformer

        for i in keep:
            conformer_info = conformers_info[i]
            mol = conformer_info["rd_mol"]
            if mol is None:
                logging.warning(
                    f"RDKit molecule is None for {mol_dict['smiles']}, conformer {i}, skipping conformer."
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
                smiles=mol_dict["smiles"],
            )

            data_list.append(data)

        return data_list

    except Exception as e:
        print(f"Error processing {mol_dict['smiles']}: {e}")
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

    DRUGS_URL = "https://dataverse.harvard.edu/api/access/datafile/4327252?version=4.0"

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        num_conformers=30,
    ):
        self.root = root
        self.num_conformers = num_conformers
        self.mols = {}
        if not os.path.exists(self.root):
            os.makedirs(self.root, exist_ok=True)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    def download(self):
        raw_path = os.path.join(self.root, "raw")
        os.makedirs(raw_path, exist_ok=True)

        extracted_rdkit_folder_path = os.path.join(raw_path, "rdkit_folder")

        if not os.path.exists(extracted_rdkit_folder_path):
            # First check if rdkit_folder.tar.gz exists
            archive_rdkit_folder_path = os.path.join(raw_path, "rdkit_folder.tar.gz")
            if os.path.exists(archive_rdkit_folder_path):
                logging.info(f"Extracting existing GEOM RDKit folder to {raw_path}...")
                try:
                    subprocess.run(
                        ["tar", "-xvf", archive_rdkit_folder_path, "-C", raw_path],
                        check=True,
                    )
                    logging.info("Existing GEOM RDKit folder extraction complete.")
                except subprocess.CalledProcessError as e:
                    raise subprocess.CalledProcessError(
                        f"Error extracting existing GEOM RDKit folder: {e}"
                    )
            # If not, download it from the web
            else:
                logging.info(
                    f"Downloading GEOM RDKit folder to {archive_rdkit_folder_path}..."
                )
                try:
                    try:
                        urllib.request.urlretrieve(
                            self.DRUGS_URL,
                            filename=archive_rdkit_folder_path,
                        )
                    except Exception:
                        subprocess.run(
                            [
                                "wget",
                                "-O",
                                archive_rdkit_folder_path,
                                self.DRUGS_URL,
                            ],
                            check=True,
                        )
                except Exception as e:
                    raise RuntimeError(
                        f"Error downloading GEOM RDKit from " f"{self.DRUGS_URL}: {e}"
                    )
                logging.info(f"Extracting GEOM RDKit folder to {raw_path}...")
                try:
                    subprocess.run(
                        ["tar", "-xvf", archive_rdkit_folder_path, "-C", raw_path],
                        check=True,
                    )
                    logging.info("GEOM RDKit folder extraction complete.")
                except subprocess.CalledProcessError as e:
                    raise subprocess.CalledProcessError(
                        f"Error extracting GEOM RDKit folder: {e}"
                    )

        else:
            logging.info(
                f"GEOM RDKit folder already exists at {extracted_rdkit_folder_path}, skipping download and extraction."
            )

    @property
    def raw_file_names(self):
        return ["rdkit_folder"]

    @property
    def processed_file_names(self):
        return ["geom.pt"]

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

        # Gather all molecule pickle paths
        drugs_summary_path = os.path.join(
            self.root, "raw", "rdkit_folder", "summary_drugs.json"
        )
        with open(drugs_summary_path, "r") as f:
            drugs_summary = json.load(f)

        # Get all pickle paths
        pickle_paths = []
        for smiles, sub_dict in tqdm(drugs_summary.items()):
            try:
                assert "pickle_path" in sub_dict, f"pickle_path not found"
                pickle_path = os.path.join(
                    self.root, "raw", "rdkit_folder", sub_dict["pickle_path"]
                )
                pickle_paths.append(pickle_path)
            except Exception as e:
                print(f"Error processing {smiles}: {e}")

        # drug_path = os.path.join(self.root, "raw", "rdkit_folder", "drugs")

        data_list = []
        # Load all molecule pickles into self.mols
        for pickle_path in tqdm(pickle_paths):
            try:
                with open(pickle_path, "rb") as f:
                    mol_dict = pickle.load(f)

                molecule_data_list = process_molecule(mol_dict, vocabulary=vocabulary)
                if molecule_data_list is not None:
                    data_list.extend(molecule_data_list)

            except Exception as e:
                print(f"Error loading pickle {pickle_path}: {e}")
                continue

        data_list = [data for data in data_list if data is not None]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    def get_random_splits(
        self,
        test_fraction: float = 0.1,
        validation_fraction: float = 0.1,
    ) -> Dict[str, np.ndarray]:

        num_mols = len(self)
        num_test = int(num_mols * test_fraction)
        num_val = int(num_mols * validation_fraction)

        idx_lst = [i for i in range(num_mols)]
        random.seed(0)
        random.shuffle(idx_lst)

        test_idx = np.array(idx_lst[num_test:])
        val_idx = np.array(idx_lst[num_test : num_test + num_val])
        train_idx = np.array(idx_lst[num_test + num_val :])

        logging.info(f"Number of training molecules: {len(train_idx)}")
        logging.info(f"Number of validation molecules: {len(val_idx)}")
        logging.info(f"Number of test molecules: {len(test_idx)}")

        splits = {"train": train_idx, "val": val_idx, "test": test_idx}

        return splits
