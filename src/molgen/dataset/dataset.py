# credit:
# https://github.com/atomicarchitects/symphony/blob/590621f27fdf74d7ca13939185d9cfb1e881b775/symphony/data/datasets/qm9.py
# https://github.com/atomicarchitects/symphony/blob/590621f27fdf74d7ca13939185d9cfb1e881b775/symphony/data/datasets/utils.py
# https://github.com/aspuru-guzik-group/quetzal/blob/main/qm9.py

import logging
import os
import urllib
import zipfile
from typing import Dict

import networkx as nx
import numpy as np
import torch
import yaml
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import HybridizationType
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")  # Disables all RDKit warnings and informational messages


class DataSet(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        if not os.path.exists(self.root):
            os.makedirs(self.root, exist_ok=True)
        self.qm9_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip"
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["gdb9.sdf"]

    @property
    def processed_file_names(self):
        return ["qm9.pt"]

    def download(self):
        raw_path = os.path.join(self.root, "raw")
        if not os.path.exists(raw_path):
            os.makedirs(raw_path, exist_ok=True)
        path = DataSet.download_url(self.qm9_url, raw_path)
        DataSet.extract_zip(path, raw_path)

    @staticmethod
    def download_url(url: str, root: str) -> str:
        """Download if file does not exist in root already. Returns path to file."""
        filename = url.rpartition("/")[2]
        file_path = os.path.join(root, filename)

        try:
            if os.path.exists(file_path):
                logging.info(f"Using downloaded file: {file_path}")
                return file_path
            data = urllib.request.urlopen(url)
        except urllib.error.URLError:
            # No internet connection
            if os.path.exists(file_path):
                logging.info(
                    f"No internet connection! Using downloaded file: {file_path}"
                )
                return file_path

            raise ValueError(f"Could not download {url}")

        chunk_size = 1024
        total_size = int(data.info()["Content-Length"].strip())

        if os.path.exists(file_path):
            if os.path.getsize(file_path) == total_size:
                logging.info(f"Using downloaded and verified file: {file_path}")
                return file_path

        logging.info(f"Downloading {url} to {file_path}")

        with open(file_path, "wb") as f:
            with tqdm(total=total_size) as pbar:
                while True:
                    chunk = data.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(chunk_size)

        return file_path

    @staticmethod
    def extract_zip(path: str, root: str):
        """Extract zip if content does not exist in root already."""
        logging.info(f"Extracting {path} to {root}...")
        with zipfile.ZipFile(path, "r") as f:
            for name in f.namelist():
                if name.endswith("/"):
                    logging.info(f"Skip directory {name}")
                    continue
                out_path = os.path.join(root, name)
                file_size = f.getinfo(name).file_size
                if os.path.exists(out_path) and os.path.getsize(out_path) == file_size:
                    logging.info(f"Skip existing file {name}")
                    continue
                logging.info(f"Extracting {name} to {root}...")
                f.extract(name, root)

    def process(self):
        raw_path = self.raw_paths[0]
        vocab_path = os.path.join(
            os.path.dirname(os.path.dirname(self.root)), "scripts", "qm9_small_vocab.yaml"
        )
        try:
            with open(vocab_path, "r") as file:
                self.vocabulary = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f"Error loading YAML file: {e}")
        data_list = []
        not_sanitized = []
        supplier = Chem.SDMolSupplier(raw_path, removeHs=False, sanitize=False)
        for idx, molecule in tqdm(enumerate(supplier)):
            data, sanitized = self.process_molecule(molecule)
            if data is not None:
                data_list.append(data)
                if not sanitized:
                    not_sanitized.append(idx)

        data_list = [data for data in data_list if data is not None]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
        not_sanitized = torch.tensor(not_sanitized, dtype=torch.long)
        torch.save(
            not_sanitized,
            os.path.join(self.root, "processed", "_not_sanitized_idx.pt"),
        )

    @staticmethod
    def try_sanitize_molecule(mol):
        try:
            # Attempt to sanitize the molecule
            Chem.SanitizeMol(mol)
            return (
                mol,
                True,
            )  # Return the molecule and a flag indicating successful sanitization
        except Exception as e:
            # If sanitization fails, flag it and continue
            logging.warning(f"Sanitization failed for molecule: {e}")
            return (
                mol,
                False,
            )  # Return the molecule and a flag indicating sanitization failure

    def process_molecule(self, mol):
        try:
            # Some molecules contain multiple fragements, here we pic the largest one
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            mol = max(frags, key=lambda frag: frag.GetNumAtoms())
            # mol = Chem.AddHs(mol)
            mol, sanitized = DataSet.try_sanitize_molecule(mol)

            # Define a mapping for hybridization states to integers
            hybridization_mapping = {
                HybridizationType.S: 0,
                HybridizationType.SP: 1,
                HybridizationType.SP2: 2,
                HybridizationType.SP3: 3,
                HybridizationType.SP3D: 4,
                HybridizationType.SP3D2: 5,
                HybridizationType.UNSPECIFIED: -1,  # Optional: handle unspecified hybridization
            }

            # Create a tensor for atomic numbers and hybridization states
            x = torch.tensor(
                [
                    self.vocabulary[
                        f"({atom.GetAtomicNum()}, {hybridization_mapping[atom.GetHybridization()]})"
                    ]
                    for atom in mol.GetAtoms()
                ],
                dtype=torch.long,
            )

            # Node positions: 3D coordinates
            pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
            pos = pos - pos.mean(dim=0, keepdim=True)

            # Edge index: Bond connections
            edge_index = []
            edge_attr = []

            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_index.append((i, j))
                edge_index.append((j, i))  # Add reverse direction for undirected graph

                # Bond attributes: OHE for bond type, aromaticity, and ring membership
                bond_type = bond.GetBondType()
                bond_type_ohe = {
                    Chem.rdchem.BondType.SINGLE: [1, 0, 0, 0],
                    Chem.rdchem.BondType.DOUBLE: [0, 1, 0, 0],
                    Chem.rdchem.BondType.TRIPLE: [0, 0, 1, 0],
                    Chem.rdchem.BondType.AROMATIC: [0, 0, 0, 1],
                }.get(
                    bond_type, [0, 0, 0, 0]
                )  # Default to [0, 0, 0, 0] if bond type is unknown

                is_aromatic = int(bond.GetIsAromatic())
                in_ring = int(bond.IsInRing())

                # Combine bond attributes
                edge_attr.append(bond_type_ohe + [is_aromatic, in_ring])
                edge_attr.append(
                    bond_type_ohe + [is_aromatic, in_ring]
                )  # Reverse direction

            # Convert edge_index and edge_attr to tensors
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

            # Generate the Minimum Spanning Tree (MST)
            G = nx.Graph()
            for i, j in edge_index.t().tolist():
                G.add_edge(i, j)

            mst = nx.minimum_spanning_tree(G)  # Compute MST using NetworkX
            edge_index_mst = (
                torch.tensor(list(mst.edges), dtype=torch.long).t().contiguous()
            )

            # Also store the tree depth in the data object
            root_node = list(mst.nodes)[0]
            depths = nx.single_source_shortest_path_length(mst, root_node)

            # Compute edge hierarchy levels
            edge_hierarchy = []
            for i, j in mst.edges:
                # The hierarchical level of the edge is the maximum depth of its two nodes
                edge_hierarchy.append(max(depths[i], depths[j]))

            edge_hierarchy = torch.tensor(edge_hierarchy, dtype=torch.long)
            diameter = nx.diameter(mst)

            # Compute eccentricities for all nodes
            eccentricities = nx.eccentricity(G)  # Dictionary {node: eccentricity}
            eccentricity_tensor = torch.tensor(
                [eccentricities[node] for node in range(len(G.nodes))], dtype=torch.long
            )

            # Create PyG Data object
            data = Data(
                x=x,
                pos=pos,
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_index_mst=edge_index_mst,
                edge_attr_mst=edge_hierarchy,
                diameter=diameter,
                eccentricity=eccentricity_tensor,
            )

            return data, sanitized

        except Exception as e:
            print(f"Error processing {mol}: {e}")
            return None

    def get_qm9_splits(
        self,
        edm_splits: bool,
    ) -> Dict[str, np.ndarray]:
        """Adapted from https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/qm9/data/prepare/qm9.py."""

        def is_int(string: str) -> bool:
            try:
                int(string)
                return True
            except ValueError:
                return False

        logging.info("Dropping uncharacterized molecules.")
        gdb9_url_excluded = (
            "https://springernature.figshare.com/ndownloader/files/3195404"
        )
        gdb9_txt_excluded = os.path.join(self.root, "uncharacterized.txt")
        urllib.request.urlretrieve(gdb9_url_excluded, filename=gdb9_txt_excluded)

        # First, get list of excluded indices.
        excluded_strings = []
        with open(gdb9_txt_excluded) as f:
            lines = f.readlines()
            excluded_strings = [
                line.split()[0] for line in lines if len(line.split()) > 0
            ]

        excluded_idxs = [int(idx) - 1 for idx in excluded_strings if is_int(idx)]

        assert (
            len(excluded_idxs) == 3054
        ), "There should be exactly 3054 excluded atoms. Found {}".format(
            len(excluded_idxs)
        )

        # Now, create a list of included indices.
        Ngdb9 = 133885
        Nexcluded = 3054

        included_idxs = np.array(sorted(list(set(range(Ngdb9)) - set(excluded_idxs))))

        # Now, generate random permutations to assign molecules to training/valation/test sets.
        Nmols = Ngdb9 - Nexcluded
        assert Nmols == len(
            included_idxs
        ), "Number of included molecules should be equal to Ngdb9 - Nexcluded. Found {} {}".format(
            Nmols, len(included_idxs)
        )

        Ntrain = 100000
        Ntest = int(0.1 * Nmols)
        Nval = Nmols - (Ntrain + Ntest)

        # Generate random permutation.
        np.random.seed(0)
        if edm_splits:
            data_permutation = np.random.permutation(Nmols)
        else:
            data_permutation = np.arange(Nmols)

        train, val, test, extra = np.split(
            data_permutation, [Ntrain, Ntrain + Nval, Ntrain + Nval + Ntest]
        )

        assert len(extra) == 0, "Split was inexact {} {} {} {}".format(
            len(train), len(val), len(test), len(extra)
        )

        train = included_idxs[train]
        val = included_idxs[val]
        test = included_idxs[test]

        # Load sanitized indices and exclude them from splits
        not_sanitized = torch.load(
            os.path.join(self.root, "processed", "_not_sanitized_idx.pt")
        ).numpy()
        train = np.array([idx for idx in train if idx not in not_sanitized])
        val = np.array([idx for idx in val if idx not in not_sanitized])
        test = np.array([idx for idx in test if idx not in not_sanitized])

        splits = {"train": train, "val": val, "test": test}

        # Cleanup file.
        try:
            os.remove(gdb9_txt_excluded)
        except OSError:
            pass

        return splits
