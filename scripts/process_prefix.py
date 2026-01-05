import os

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = os.path.join(os.getcwd(), "data", "prefixes")


def compute_prefix_x_pos_batch(
    prefix_smiles: list[str],
    xylene: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute x, pos, and batch tensors from the prefix smiles."""

    atomic_num_to_atom_type = {
        1: 1,  # H
        5: 2,  # B
        6: 3,  # C
        7: 4,  # N
        8: 5,  # O
        9: 6,  # F
        13: 7,  # Al
        14: 8,  # Si
        15: 9,  # P
        16: 10,  # S
        17: 11,  # Cl
        33: 12,  # As
        35: 13,  # Br
        53: 14,  # I
        80: 15,  # Hg
        83: 16,  # Bi
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

    if xylene:
        xylene_hydrogen_indices = []
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetSymbol() == "H":
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetSymbol() == "C" and not neighbor.GetIsAromatic():
                        xylene_hydrogen_indices.append(i)
        return x, pos, batch, xylene_hydrogen_indices
    else:
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

# ------------- Prepare ortho, meta and para-"empty" benzenes -------------

prefix_smiles = "C1=CC=CC=C1"
x, pos, batch = compute_prefix_x_pos_batch(prefix_smiles)

# Remove ortho-hydrogen atoms to allow for growth
mask = torch.ones_like(x, dtype=torch.bool)
mask[6] = False  # Remove hydrogen atom at index 6
mask[7] = False  # Remove hydrogen atom at index 7
x_ortho = x[mask]
pos_ortho = pos[mask]
batch_ortho = batch[mask]

# # Shuffle order of atoms to test permutation invariance
# perm = torch.randperm(x_ortho.size(0))
# x_ortho = x_ortho[perm]
# pos_ortho = pos_ortho[perm]
# batch_ortho = batch_ortho[perm]

# Remove meta-hydrogen atoms to allow for growth
mask = torch.ones_like(x, dtype=torch.bool)
mask[6] = False  # Remove hydrogen atom at index 6
mask[8] = False  # Remove hydrogen atom at index 8
x_meta = x[mask]
pos_meta = pos[mask]
batch_meta = batch[mask]

# # Shuffle order of atoms to test permutation invariance
# perm = torch.randperm(x_meta.size(0))
# x_meta = x_meta[perm]
# pos_meta = pos_meta[perm]
# batch_meta = batch_meta[perm]

# Remove para-hydrogen atoms to allow for growth
mask = torch.ones_like(x, dtype=torch.bool)
mask[6] = False  # Remove hydrogen atom at index 6
mask[9] = False  # Remove hydrogen atom at index 9
x_para = x[mask]
pos_para = pos[mask]
batch_para = batch[mask]

# # Shuffle order of atoms to test permutation invariance
# perm = torch.randperm(x_para.size(0))
# x_para = x_para[perm]
# pos_para = pos_para[perm]
# batch_para = batch_para[perm]

# Save tensors
path = os.path.join(ROOT, "benzene_ortho")
save_prefix_tensors(x_ortho, pos_ortho, batch_ortho, path)
path = os.path.join(ROOT, "benzene_meta")
save_prefix_tensors(x_meta, pos_meta, batch_meta, path)
path = os.path.join(ROOT, "benzene_para")
save_prefix_tensors(x_para, pos_para, batch_para, path)

# ------------- Prepare ortho-, meta- and para-xylene -------------

# Initialize ortho-xylene and remove methyl hydrogen atoms to allow for growth
prefix_smiles = "C1(C)=C(C)C=CC=C1"
x, pos, batch, xylene_hydrogen_indices = compute_prefix_x_pos_batch(prefix_smiles, xylene=True)
mask = torch.ones_like(x, dtype=torch.bool)
for idx in xylene_hydrogen_indices:
    mask[idx] = False
x_ortho = x[mask]
pos_ortho = pos[mask]
batch_ortho = batch[mask]

# Initialize meta-xylene and remove methyl hydrogen atoms to allow for growth
prefix_smiles = "C1(C)=CC(C)=CC=C1"
x, pos, batch, xylene_hydrogen_indices = compute_prefix_x_pos_batch(prefix_smiles, xylene=True)
mask = torch.ones_like(x, dtype=torch.bool)
for idx in xylene_hydrogen_indices:
    mask[idx] = False
x_meta = x[mask]
pos_meta = pos[mask]
batch_meta = batch[mask]

# Initialize para-xylene and remove methyl hydrogen atoms to allow for growth
prefix_smiles = "C1(C)=CC=C(C)C=C1"
x, pos, batch, xylene_hydrogen_indices = compute_prefix_x_pos_batch(prefix_smiles, xylene=True)
mask = torch.ones_like(x, dtype=torch.bool)
for idx in xylene_hydrogen_indices:
    mask[idx] = False
x_para = x[mask]
pos_para = pos[mask]
batch_para = batch[mask]

# Save tensors
path = os.path.join(ROOT, "xylene_ortho")
save_prefix_tensors(x_ortho, pos_ortho, batch_ortho, path)
path = os.path.join(ROOT, "xylene_meta")
save_prefix_tensors(x_meta, pos_meta, batch_meta, path)
path = os.path.join(ROOT, "xylene_para")
save_prefix_tensors(x_para, pos_para, batch_para, path)
