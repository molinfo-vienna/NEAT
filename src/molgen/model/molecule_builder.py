import logging
import os
import torch
from rdkit.Chem import Mol, MolFromXYZBlock, rdDetermineBonds


class MoleculeBuilder:
    def __init__(self):
        super().__init__()
        self.atom_type_to_element = {
            1: "H",  # Hydrogen
            2: "H",  # Hydrogen
            3: "C",  # Carbon
            4: "C",  # Carbon
            5: "C",  # Carbon
            6: "C",  # Carbon
            7: "N",  # Nitrogen
            8: "N",  # Nitrogen
            9: "N",  # Nitrogen
            10: "N",  # Nitrogen
            11: "O",  # Oxygen
            12: "O",  # Oxygen
            13: "O",  # Oxygen
            14: "F",  # Fluorine
            15: "F",  # Fluorine
        }

    def load_tensor_from_file(self, files_path):
        x = torch.load(os.path.join(files_path, "x.pt")).detach().cpu()
        pos = torch.load(os.path.join(files_path, "pos.pt")).detach().cpu()
        batch = torch.load(os.path.join(files_path, "batch.pt")).detach().cpu()
        return x, pos, batch

    def create_xyz_block(self, x, pos):
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
        self, x, pos, batch,
    ):
        mols = []
        unique_batches = batch.unique().tolist()

        for batch_id in unique_batches:
            mask = batch == batch_id
            x_mol = x[mask]
            pos_mol = pos[mask]

            try:
                xyz_block = self.create_xyz_block(x_mol, pos_mol)
                raw_mol = MolFromXYZBlock(xyz_block)
                mol = Mol(raw_mol)
                rdDetermineBonds.DetermineBonds(mol, charge=0)
                mols.append(mol)
            except Exception as e:
                logging.warning(f"Error processing molecule: {e}")
                mols.append(None)

        return mols
