import os

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = os.path.join(os.getcwd(), "data", "prefixes")


def compute_prefix_x_pos_batch(
    prefix_smiles: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute x, pos, and batch tensors from the prefix smiles."""

    atomic_num_to_atom_type = {
        1: 1,
        6: 2,
        7: 3,
        8: 4,
        9: 5,
    }

    # Create RDKit molecule from SMILES
    mol = Chem.MolFromSmiles(prefix_smiles)

    # Add hydrogens (needed for 3D embedding)
    mol = Chem.AddHs(mol)

    # Generate a 3D conformer
    params = AllChem.ETKDGv3()
    res = AllChem.EmbedMolecule(mol, params)
    if res != 0:
        raise RuntimeError("Embedding failed")

    # Geometry optimization
    AllChem.UFFOptimizeMolecule(mol)

    # Get 3D conformer
    conf = mol.GetConformer()

    # Get categorical atom types
    x = torch.tensor(
        [atomic_num_to_atom_type[atom.GetAtomicNum()] for atom in mol.GetAtoms()],
        dtype=torch.long,
    )

    # Get 3D atom positions
    pos = torch.tensor(
        [tuple(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
        dtype=torch.float,
    )
    pos -= pos.mean(dim=0)  # Center the molecule at the origin

    # Compute dummy batch tensor for compatibility with MoleculeBuilder.load_tensor_from_file()
    batch = torch.zeros(x.size(0), dtype=torch.long)

    return x, pos, batch


def save_prefix_tensors(
    x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor, path: str
):
    """Save prefix tensors for different molecules."""
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(x, os.path.join(path, "x.pt"))
    torch.save(pos, os.path.join(path, "pos.pt"))
    torch.save(batch, os.path.join(path, "batch.pt"))


# ------------- Prepare benzene -------------

prefix_smiles = "C1=CC=CC=C1"
x, pos, batch = compute_prefix_x_pos_batch(prefix_smiles)

# Remove hydrogen atoms to allow for growth
mask = x != 1  # Atomic type 1 corresponds to hydrogen
x = x[mask]
pos = pos[mask]
batch = batch[mask]

# Shuffle order of atoms to test permutation invariance
perm = torch.randperm(x.size(0))
x = x[perm]
pos = pos[perm]
batch = batch[perm]

# Save tensors
path = os.path.join(ROOT, "benzene")
save_prefix_tensors(x, pos, batch, path)

# ------------- Prepare 1,2,4-triazole -------------

prefix_smiles = "C1=NC=NN1"
x, pos, batch = compute_prefix_x_pos_batch(prefix_smiles)

# Remove hydrogen atoms to allow for growth
mask = x != 1  # Atomic type 1 corresponds to hydrogen
x = x[mask]
pos = pos[mask]
batch = batch[mask]

# Shuffle order of atoms to test permutation invariance
perm = torch.randperm(x.size(0))
x = x[perm]
pos = pos[perm]
batch = batch[perm]

# Save tensors
path = os.path.join(ROOT, "triazole")
save_prefix_tensors(x, pos, batch, path)

# ------------- Prepare cyclohexane -------------

prefix_smiles = "C1CCCCC1"
x, pos, batch = compute_prefix_x_pos_batch(prefix_smiles)

# Remove hydrogen atoms to allow for growth
mask = x != 1  # Atomic type 1 corresponds to hydrogen
x = x[mask]
pos = pos[mask]
batch = batch[mask]

# Shuffle order of atoms to test permutation invariance
perm = torch.randperm(x.size(0))
x = x[perm]
pos = pos[perm]
batch = batch[perm]

# Save tensors
path = os.path.join(ROOT, "cyclohexane")
save_prefix_tensors(x, pos, batch, path)

# ------------- Prepare cyclopropane -------------

prefix_smiles = "C1CC1"
x, pos, batch = compute_prefix_x_pos_batch(prefix_smiles)

# Remove hydrogen atoms to allow for growth
mask = x != 1  # Atomic type 1 corresponds to hydrogen
x = x[mask]
pos = pos[mask]
batch = batch[mask]

# Shuffle order of atoms to test permutation invariance
perm = torch.randperm(x.size(0))
x = x[perm]
pos = pos[perm]
batch = batch[perm]

# Save tensors
path = os.path.join(ROOT, "cyclopropane")
save_prefix_tensors(x, pos, batch, path)
