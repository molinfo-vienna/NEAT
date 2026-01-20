import logging
import os

import torch
from rdkit.Chem import Mol, MolToSmiles, rdDetermineBonds, rdmolfiles
from tqdm import tqdm


class MoleculeBuilder:
    """Build RDKit molecules from tensors.

    Args:
        vocab (str): The vocabulary to use. Options are "QM9" and "GEOM".
    """

    def __init__(self, vocab="QM9") -> None:
        super().__init__()
        if vocab == "QM9":
            self.atom_type_to_element = {
                1: "H",
                2: "C",
                3: "N",
                4: "O",
                5: "F",
            }
        elif vocab == "GEOM":
            self.atom_type_to_element = {
                1: "H",
                2: "B",
                3: "C",
                4: "N",
                5: "O",
                6: "F",
                7: "Al",
                8: "Si",
                9: "P",
                10: "S",
                11: "Cl",
                12: "As",
                13: "Br",
                14: "I",
                15: "Hg",
                16: "Bi",
            }
        else:
            raise ValueError(f"Unsupported vocabulary: {vocab}")

    def load_tensor_from_file(
        self, files_path: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load atom types, positions, and batch indices from a generated_mols.pt file.

        Args:
            files_path (str): The path to the generated_mols.pt file.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the atom types, positions, and batch indices.
        """
        generated_mols = (
            torch.load(
                os.path.join(files_path, "generated_mols.pt"), weights_only=False
            )
            .detach()
            .cpu()
        )
        return generated_mols.x, generated_mols.pos, generated_mols.batch

    def create_xyz_block(self, x: torch.Tensor, pos: torch.Tensor) -> str:
        """Create an XYZ block from a tensor of atom types and positions.

        Args:
            x (torch.Tensor): A tensor of shape (n_atoms,).
            pos (torch.Tensor): A tensor of shape (n_atoms, 3).

        Returns:
            str: An XYZ block.
        """
        xyz_lines = []
        num_atoms = x.size(0)
        xyz_lines.append(f"{num_atoms}")
        xyz_lines.append("")

        for i in range(num_atoms):
            atom_type = x[i].item()
            element = self.atom_type_to_element.get(atom_type, "X")
            x_coord, y_coord, z_coord = pos[i].tolist()
            xyz_lines.append(f"{element}\t{x_coord:.4f}\t{y_coord:.4f}\t{z_coord:.4f}")

        return "\n".join(xyz_lines)

    def generate_rdkit_molecules(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        progress_bar: bool = False,
        break_after_k_mols: int = None,
    ) -> list[Mol]:
        """Generate RDKit molecules from tensors of atom types and positions.

        Args:
            x (torch.Tensor): A tensor of shape (n_atoms,).
            pos (torch.Tensor): A tensor of shape (n_atoms, 3).
            batch (torch.Tensor): A tensor of shape (n_atoms,).

        Returns:
            list[Mol]: A list of RDKit molecules.
        """
        mols = []
        unique_batches = batch.unique().tolist()

        iterator = unique_batches
        if progress_bar:
            iterator = tqdm(unique_batches, desc="Generating RDKit molecules")
        for batch_id in iterator:
            mask = batch == batch_id
            x_mol = x[mask]
            pos_mol = pos[mask]

            xyz_block = self.create_xyz_block(x_mol, pos_mol)
            mol = rdmolfiles.MolFromXYZBlock(xyz_block)
            try:
                rdDetermineBonds.DetermineBonds(mol, charge=0, maxIterations=100000)
            except ValueError:
                # logging.warning(
                #     f"Could not determine bonds for molecule in batch {batch_id} with neutral total charge."
                # )
                mol = None
            except Exception as e:
                logging.warning(
                    f"An error occurred while determining bonds for molecule in batch {batch_id}: {e}"
                )
                mol = None

            mols.append(mol)

            if break_after_k_mols is not None and len(mols) >= break_after_k_mols:
                break

        return mols

    def generate_smiles_from_rdkit_molecules(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        progress_bar: bool = False,
    ) -> list[str]:
        """Generate SMILES strings from tensors of atom types and positions.

        Args:
            x (torch.Tensor): A tensor of shape (n_atoms,).
            pos (torch.Tensor): A tensor of shape (n_atoms, 3).
            batch (torch.Tensor): A tensor of shape (n_atoms,).
            progress_bar (bool): Whether to show a progress bar.

        Returns:
            list[str]: A list of SMILES strings.
        """
        smiles = []
        unique_batches = batch.unique().tolist()

        iterator = unique_batches
        if progress_bar:
            iterator = tqdm(unique_batches, desc="Generating RDKit molecules")
        for batch_id in iterator:
            mask = batch == batch_id
            x_mol = x[mask]
            pos_mol = pos[mask]

            xyz_block = self.create_xyz_block(x_mol, pos_mol)
            mol = rdmolfiles.MolFromXYZBlock(xyz_block)
            try:
                rdDetermineBonds.DetermineBonds(mol, charge=0, maxIterations=100000)
            except ValueError:
                mol = None
            except Exception as e:
                logging.warning(
                    f"An error occurred while determining bonds for molecule in batch {batch_id}: {e}"
                )
                mol = None
            if mol is not None:
                smiles_str = MolToSmiles(mol, canonical=True)
                smiles.append(smiles_str)
            else:
                smiles.append(None)

        return smiles
