# credit:
# https://github.com/atomicarchitects/symphony/blob/590621f27fdf74d7ca13939185d9cfb1e881b775/symphony/data/datasets/qm9.py
# https://github.com/atomicarchitects/symphony/blob/590621f27fdf74d7ca13939185d9cfb1e881b775/symphony/data/datasets/utils.py

import torch
import os
import urllib
import numpy as np
from rdkit import Chem

PTABLE = Chem.GetPeriodicTable()

from typing import Dict

import tqdm
import zipfile
import tarfile
import logging

# from chem import GEN, STOP


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
            logging.info(f"No internet connection! Using downloaded file: {file_path}")
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
        with tqdm.tqdm(total=total_size) as pbar:
            while True:
                chunk = data.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(chunk_size)

    return file_path


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


def extract_tar(path: str, root: str):
    """Extract tar."""
    logging.info(f"Extracting {path} to {root}...")
    with tarfile.TarFile(path, "r") as f:
        f.extractall(path=root)


def get_qm9_splits(
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
    gdb9_url_excluded = "https://springernature.figshare.com/ndownloader/files/3195404"
    gdb9_txt_excluded = os.path.join(root_dir, "uncharacterized.txt")
    urllib.request.urlretrieve(gdb9_url_excluded, filename=gdb9_txt_excluded)

    # First, get list of excluded indices.
    excluded_strings = []
    with open(gdb9_txt_excluded) as f:
        lines = f.readlines()
        excluded_strings = [line.split()[0] for line in lines if len(line.split()) > 0]

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


####################################################################################################

QM9_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip"
)

root_dir = "data"
if not os.path.exists(root_dir):
    os.makedirs(root_dir, exist_ok=True)

raw_mols_path = os.path.join(root_dir, "gdb9.sdf")

if not os.path.exists(raw_mols_path):
    path = download_url(QM9_URL, root_dir)
    extract_zip(path, root_dir)

if not os.path.exists(os.path.join(root_dir, "qm9_train_atoms.npy")):

    supplier = Chem.SDMolSupplier(raw_mols_path, removeHs=False, sanitize=False)

    all_coords = []
    all_atoms = []
    all_sizes = []

    for mol in tqdm.tqdm(supplier):
        if mol is None:
            continue

        coords = np.array(mol.GetConformer().GetPositions(), dtype=np.float32)
        atoms = np.array(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int64
        )

        # zero-center coords and do PCA
        coords -= coords.mean(0)
        U, _, _ = np.linalg.svd(coords.T)
        if np.linalg.det(U) < 0:
            U[:, -1] *= -1
        coords = coords @ U

        size = atoms.shape[0]

        all_atoms.append(atoms)
        all_coords.append(coords)
        all_sizes.append(size)

    print("Saving...")

    splits = get_qm9_splits(root_dir, edm_splits=True)

    for split in splits:
        atoms = np.concatenate([all_atoms[i] for i in splits[split]])
        coords = np.concatenate([all_coords[i] for i in splits[split]])
        sizes = np.array([all_sizes[i] for i in splits[split]])

        np.save(os.path.join(root_dir, f"qm9_{split}_atoms.npy"), atoms)
        np.save(os.path.join(root_dir, f"qm9_{split}_coords.npy"), coords)
        np.save(os.path.join(root_dir, f"qm9_{split}_sizes.npy"), sizes)
