import logging
import os

import torch
from rdkit.Chem import Mol, MolFromXYZBlock, rdDetermineBonds


class MoleculeBuilder:
    """
    Builds RDKit molecules from tensors.
    """

    def __init__(self):
        super().__init__()
        self.atom_type_to_element = {
            1: "H",  # Hydrogen
            2: "C",  # Carbon
            3: "N",  # Nitrogen
            4: "O",  # Oxygen
            5: "F",  # Fluorine
        }
        # Only for QUETZAL (different vocabulary)
        # self.atom_type_to_element = {
        #     1: "H",  # Hydrogen
        #     6: "C",  # Carbon
        #     7: "N",  # Nitrogen
        #     8: "O",  # Oxygen
        #     9: "F",  # Fluorine
        # }

    def load_tensor_from_file(
        self, files_path: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Loads tensors from a file.
        Args:
            files_path (str): The path to the file.
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the atom types, positions, and batch indices.
        """
        x = torch.load(os.path.join(files_path, "x.pt")).detach().cpu()
        pos = torch.load(os.path.join(files_path, "pos.pt")).detach().cpu()
        batch = torch.load(os.path.join(files_path, "batch.pt")).detach().cpu()
        return x, pos, batch

    def create_xyz_block(self, x: torch.Tensor, pos: torch.Tensor) -> str:
        """
        Creates an XYZ block from a tensor of atom types and positions.
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
    ) -> list[Mol]:
        """
        Generates RDKit molecules from tensors of atom types and positions.
        Args:
            x (torch.Tensor): A tensor of shape (n_atoms,).
            pos (torch.Tensor): A tensor of shape (n_atoms, 3).
            batch (torch.Tensor): A tensor of shape (n_atoms,).
        Returns:
            list[Mol]: A list of RDKit molecules.
        """
        mols = []
        unique_batches = batch.unique().tolist()

        for batch_id in unique_batches:
            mask = batch == batch_id
            x_mol = x[mask]
            pos_mol = pos[mask]

            xyz_block = self.create_xyz_block(x_mol, pos_mol)
            mol = MolFromXYZBlock(xyz_block)
            try:
                rdDetermineBonds.DetermineBonds(mol, charge=0)
            except ValueError:
                logging.warning(
                    f"Could not determine bonds for molecule in batch {batch_id} with neutral total charge."
                )
                mol = None
            mols.append(mol)

        return mols
