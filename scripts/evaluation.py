import argparse
import os
from pathlib import Path

import py3Dmol
import rdkit
import yaml
from rdkit.Chem import AllChem, Draw, MolToSmiles, rdDepictor, SDMolSupplier

from molgen.dataset import DataModule
from molgen.model.molecule_builder import MoleculeBuilder
from molgen.utils.edm_metrics import edm_metrics

RESOLUTION = 400
NUM_MOLECULES_PLOTTED = 100
NUM_MOLECULES_PER_ROW = 5


def compute_validity(mols):
    num_valid = 0
    for mol in mols:
        if mol is not None:
            num_valid += 1
    return num_valid


def compute_uniqueness(mols):
    unique_smiles = set()
    for mol in mols:
        if mol is not None:
            smiles = MolToSmiles(mol, canonical=True)
            unique_smiles.add(smiles)
    return len(unique_smiles)


def compute_novelty(generated_mols, reference_mols):
    unique_generated_smiles = set()
    for mol in generated_mols:
        if mol is not None:
            smiles = MolToSmiles(mol, canonical=True)
            unique_generated_smiles.add(smiles)
    reference_smiles = set()
    for mol in reference_mols:
        if mol is not None:
            smiles = MolToSmiles(mol, canonical=True)
            reference_smiles.add(smiles)
    intersection = unique_generated_smiles.intersection(reference_smiles)
    num_novel = len(unique_generated_smiles) - len(intersection)
    return num_novel


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        dest="config_file",
        required=False,
        metavar="<file>",
        help="Config file for evaluation.",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parseArgs()

    ROOT = os.getcwd()
    if args.config_file is not None:
        CONFIG_FILE_PATH = args.config_file
        print(f"Using config file: {CONFIG_FILE_PATH}")
    else:
        CONFIG_FILE_PATH = os.path.join(ROOT, "scripts", "config_evaluation.yaml")
        print(f"Using default config file: {CONFIG_FILE_PATH}")

    params = yaml.load(
        open(CONFIG_FILE_PATH, "r"),
        Loader=yaml.FullLoader,
    )

    DATA_ROOT = os.path.join(ROOT, "data", params["data_set"])
    datamodule = DataModule(DATA_ROOT)
    datamodule.setup()
    supplier = SDMolSupplier(
        os.path.join(DATA_ROOT, "processed", "qm9.sdf"),
        removeHs=False,
        sanitize=False,
    )
    reference_mols = [mol for mol in supplier if mol is not None]

    data_path = Path(params["data_path"])
    for subdir in data_path.iterdir():
        if subdir.is_dir() and subdir.name.startswith("seed"):
            subdata_path = os.path.join(data_path, subdir.name)
            builder = MoleculeBuilder()
            x, pos, batch = builder.load_tensor_from_file(subdata_path)

            (
                atom_stability,
                mol_stability,
                edm_validity,
                edm_uniqueness,
                edm_invalid_idxs,
            ) = edm_metrics(x, pos, batch, "qm9")

            mols = builder.generate_rdkit_molecules(x, pos, batch)

            n_valid = compute_validity(mols)
            n_unique = compute_uniqueness(mols)
            n_novel = compute_novelty(mols, reference_mols)

            with open(os.path.join(subdata_path, "evaluation_results.txt"), "w") as f:
                f.write(f"Atom stability: {atom_stability*100:.2f}%\n")
                f.write(f"Molecule stability: {mol_stability*100:.2f}%\n")
                f.write(f"EDM valid: {edm_validity*100:.2f}%\n")
                f.write(f"EDM unique: {edm_uniqueness*100:.2f}%\n")
                f.write(
                    f"EDM valid x unique: {(edm_validity*edm_uniqueness)*100:.2f}%\n"
                )
                f.write(f"xyz2mol valid: {n_valid/len(mols)*100:.2f}%\n")
                f.write(f"xyz2mol unique: {n_unique/n_valid*100:.2f}%\n")
                f.write(
                    f"xyz2mol valid x unique: {(n_valid/len(mols))*(n_unique/n_valid)*100:.2f}%\n"
                )
                f.write(f"xyz2mol novel: {n_novel/len(mols)*100:.2f}%\n")
                f.write(f"Data set: {params['data_set']}\n")
                f.write(f"RDKit version: {rdkit.__version__}\n")

            mols = mols[:NUM_MOLECULES_PLOTTED]
            img = Draw.MolsToGridImage(
                mols,
                molsPerRow=NUM_MOLECULES_PER_ROW,
                subImgSize=(RESOLUTION, RESOLUTION),
            )
            img.save(os.path.join(subdata_path, "generated_molecules.png"))

            mols_2d = []
            for mol in mols:
                if mol is None:
                    mols_2d.append(None)
                else:
                    rdDepictor.Compute2DCoords(mol)
                    mol = AllChem.RemoveHs(mol)
                    mols_2d.append(mol)

            img = Draw.MolsToGridImage(mols_2d, molsPerRow=5, subImgSize=(400, 400))
            img.save(os.path.join(subdata_path, "generated_molecules_2d.png"))

            print(f"Saved generated molecules images to {os.path.join(subdata_path)}.")

            view = py3Dmol.view(
                width=NUM_MOLECULES_PER_ROW * RESOLUTION,
                height=NUM_MOLECULES_PLOTTED * RESOLUTION,
                viewergrid=(NUM_MOLECULES_PLOTTED, NUM_MOLECULES_PER_ROW),
            )

            for i in range(NUM_MOLECULES_PLOTTED):
                row = i // NUM_MOLECULES_PER_ROW
                col = i % NUM_MOLECULES_PER_ROW

                x_sub = x[batch == i]
                pos_sub = pos[batch == i]

                xyz = builder.create_xyz_block(x_sub, pos_sub)

                view.addModel(xyz, "xyz", viewer=(row, col))
                view.setStyle(
                    {"model": -1},
                    {"stick": {"radius": 0.2}, "sphere": {"scale": 0.3}},
                    viewer=(row, col),
                )

            view.zoomTo()
            view.show()

            with open(
                os.path.join(subdata_path, "generated_molecules_3d.html"), "w"
            ) as f:
                f.write(view._make_html())

            print(
                f"Saved 3D visualization to {os.path.join(subdata_path, 'generated_molecules_3d.html')}"
            )
