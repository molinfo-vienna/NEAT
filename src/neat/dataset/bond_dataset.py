"""Dataset and transform for bond prediction: adds pair_labels from SMILES."""

from typing import Optional

import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")

# Bond type mapping: RDKit BondType -> int (0=no bond, 1=single, 2=double, 3=triple, 4=aromatic)
RDKIT_BOND_TO_ID = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 4,
}


def _get_bond_matrix_from_smiles(smiles: str, n_atoms: int) -> Optional[torch.Tensor]:
    """Build bond type matrix from SMILES.

    Returns matrix of shape (n, n) with values in {0,1,2,3,4}.
    Only upper triangle is filled; lower triangle and diagonal are zeros.
    Symmetric pairs (i,j) and (j,i) have the same bond type.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if mol.GetNumAtoms(onlyExplicit=False) != n_atoms:
        return None

    bond_matrix = torch.zeros(n_atoms, n_atoms, dtype=torch.long)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        bid = RDKIT_BOND_TO_ID.get(bt, 0)
        if bid == 0:
            continue
        bond_matrix[i, j] = bid
        bond_matrix[j, i] = bid

    return bond_matrix


def add_bond_labels(data: Data) -> Optional[Data]:
    """Add pair_labels to a Data object for bond prediction.

    Uses the smiles field to get ground-truth bond types from RDKit.
    Adds data.pair_labels: tensor of shape (num_pairs,) for all (i,j) with i<j,
    with values in {0,1,2,3,4}.

    Returns None if smiles is missing or cannot be parsed.
    """
    smiles = getattr(data, "smiles", None)
    if smiles is None:
        return None
    n = data.x.shape[0]
    bond_matrix = _get_bond_matrix_from_smiles(smiles, n_atoms=n)
    if bond_matrix is None:
        return None

    pair_labels = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_labels.append(bond_matrix[i, j].item())
    data.pair_labels = torch.tensor(pair_labels, dtype=torch.long)
    return data


class BondPredictionTransform:
    """Transform that adds pair_labels to Data for bond prediction training."""

    def __call__(self, data: Data) -> Optional[Data]:
        return add_bond_labels(data)


class BondPredictionDataset(torch.utils.data.Dataset):
    """Wrapper dataset that adds bond labels on the fly.

    Wraps a base dataset (e.g. QM9DataSet, GEOMDataSet) and applies
    add_bond_labels when __getitem__ is called. Skips samples where
    bond labels cannot be computed.
    """

    def __init__(self, base_dataset: torch.utils.data.Dataset, filter_failures: bool = True):
        self.base_dataset = base_dataset
        self.filter_failures = filter_failures
        self._valid_indices = None

        if filter_failures:
            self._build_valid_indices()

    def _build_valid_indices(self):
        valid = []
        for i in range(len(self.base_dataset)):
            data = self.base_dataset[i].clone()
            if add_bond_labels(data) is not None:
                valid.append(i)
            else:
                pass
        self._valid_indices = valid

    @property
    def valid_indices(self):
        if self._valid_indices is None:
            self._build_valid_indices()
        return self._valid_indices

    def __len__(self) -> int:
        if self.filter_failures:
            return len(self.valid_indices)
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Data:
        if self.filter_failures:
            real_idx = self.valid_indices[idx]
        else:
            real_idx = idx
        data = self.base_dataset[real_idx].clone()
        result = add_bond_labels(data)
        if result is None and not self.filter_failures:
            # Return with dummy labels (should not happen if filter_failures)
            data.pair_labels = torch.zeros(
                data.x.shape[0] * (data.x.shape[0] - 1) // 2, dtype=torch.long
            )
            return data
        return result
