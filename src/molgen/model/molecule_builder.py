import logging
import os
import torch
from rdkit.Chem import Mol, MolFromXYZBlock, rdDetermineBonds, rdDepictor, RemoveHs


class MoleculeBuilder():
    def __init__(self, data_path, num_molecules):
        super().__init__()
        self.data_path = data_path
        self.num_molecules = num_molecules
        self.x = torch.load(os.path.join(data_path, f"x_{num_molecules}.pt")).detach().cpu()
        self.pos = torch.load(os.path.join(data_path, f"pos_{num_molecules}.pt")).detach().cpu()
        self.batch = torch.load(os.path.join(data_path, f"batch_{num_molecules}.pt")).detach().cpu()
        self.atom_type_to_element = {
            1: 'H',  # Hydrogen
            2: 'H',  # Hydrogen
            3: 'C',  # Carbon
            4: 'C',  # Carbon
            5: 'C',  # Carbon
            6: 'C',  # Carbon
            7: 'N',  # Nitrogen
            8: 'N',  # Nitrogen
            9: 'N',  # Nitrogen
            10: 'N',  # Nitrogen
            11: 'O',  # Oxygen
            12: 'O',  # Oxygen
            13: 'O',  # Oxygen
            14: 'F',  # Fluorine
            15: 'F',  # Fluorine
        }

    def create_xyz_block(self, x, pos):
        xyz_lines = []
        num_atoms = x.size(0)
        xyz_lines.append(f"{num_atoms}")
        xyz_lines.append("")

        for i in range(num_atoms):
            atom_type = x[i].item()
            element = self.atom_type_to_element.get(atom_type, 'X')
            x_coord, y_coord, z_coord = pos[i].tolist()
            xyz_lines.append(f"{element}\t{x_coord:.4f}\t{y_coord:.4f}\t{z_coord:.4f}")

        return "\n".join(xyz_lines)

    def generate_rdkit_molecules(self, optimized_for_2d=False, remove_hydrogens=False):
        mols = []
        unique_batches = self.batch.unique().tolist()

        for batch_id in unique_batches:
            mask = (self.batch == batch_id)
            x_mol = self.x[mask]
            pos_mol = self.pos[mask]

            try:
                xyz_block = self.create_xyz_block(x_mol, pos_mol)
                raw_mol = MolFromXYZBlock(xyz_block)
                mol = Mol(raw_mol)
                rdDetermineBonds.DetermineBonds(mol, charge=0)
                if optimized_for_2d:
                    rdDepictor.Compute2DCoords(mol)
                if remove_hydrogens:
                    mol = RemoveHs(mol)
                mols.append(mol)
            except Exception as e:
                logging.warning(f"Error processing molecule: {e}")
                mols.append(None)

        return mols
    

    
    