# credit:
# https://github.com/atomicarchitects/symphony/blob/590621f27fdf74d7ca13939185d9cfb1e881b775/symphony/data/datasets/qm9.py
# https://github.com/atomicarchitects/symphony/blob/590621f27fdf74d7ca13939185d9cfb1e881b775/symphony/data/datasets/utils.py
# https://github.com/aspuru-guzik-group/quetzal/blob/main/qm9.py

import os
import logging
import zipfile
import urllib
from typing import Dict
import numpy as np

from tqdm import tqdm
from rdkit import Chem
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.transforms import ToUndirected


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
        path = DataSet.download_url(self.qm9_url, self.root)
        DataSet.extract_zip(path, self.root)

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
        data_list = []
        supplier = Chem.SDMolSupplier(raw_path, removeHs=False, sanitize=False)
        for molecule in tqdm(supplier):
            data = self.process_molecule(molecule)
            if data is not None:
                data_list.append(data)

        data_list = [data for data in data_list if data is not None]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    @staticmethod
    def process_molecule(mol):
        try:
            x = torch.tensor(
                [atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.int32
            )
            pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
            pos = pos - pos.mean(dim=0, keepdim=True)

            # zero-center coords and do PCA
            # U, _, _ = np.linalg.svd(pos.T)
            # if np.linalg.det(U) < 0:
            #     U[:, -1] *= -1
            # pos = pos @ U

            # Create a PyG Data object
            data = Data(
                x=x,
                pos=pos,
                #    edge_index=edge_index,
                #    edge_attr=edge_attr,
            )
            transform = ToUndirected()
            data = transform(data)
            return data

        except Exception as e:
            print(f"Error processing {mol}: {e}")
            return None

    def get_qm9_splits(
        self,
        root_dir: str,
        edm_splits: bool,
    ) -> Dict[str, np.ndarray]:
        """Adapted from https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/qm9/data/prepare/qm9.py."""

        def is_int(string: str) -> bool:
            try:
                int(string)
                return True
            except:
                return False

        logging.info("Dropping uncharacterized molecules.")
        gdb9_url_excluded = (
            "https://springernature.figshare.com/ndownloader/files/3195404"
        )
        gdb9_txt_excluded = os.path.join(root_dir, "uncharacterized.txt")
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

        splits = {"train": train, "val": val, "test": test}

        # Cleanup file.
        try:
            os.remove(gdb9_txt_excluded)
        except OSError:
            pass

        return splits
