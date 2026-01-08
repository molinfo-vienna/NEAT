import argparse
import os
from pathlib import Path

import numpy as np
import py3Dmol
import rdkit
import yaml
from rdkit.Chem import AllChem, Draw, MolToSmiles, rdDepictor

from neat.dataset import DataModule
from neat.model.molecule_builder import MoleculeBuilder
from neat.utils.edm_metrics import edm_metrics

RESOLUTION = 400
NUM_MOLECULES_PLOTTED = 100
NUM_MOLECULES_PER_ROW = 5


def compute_validity_uniqueness_novelty(mols, reference_smiles):
    """Compute validity, uniqueness and novelty ratio of generated molecules.

    Args:
        mols (List(Mol)): generated molecules
        reference_smiles (List[str]): reference canonical SMILES strings for novelty computation

    Returns:
        Tuple[float, float, float]: validity, uniqueness, novelty ratios
    """
    ref_set = set(reference_smiles)
    unique_smiles = set()
    num_valid = 0

    for mol in mols:
        if mol is None:
            continue
        num_valid += 1
        smiles = MolToSmiles(mol, canonical=True)
        unique_smiles.add(smiles)

    num_unique = len(unique_smiles)
    num_novel = len(unique_smiles - ref_set)

    p_valid = num_valid / len(mols)
    p_valid_unique = num_unique / len(mols)
    p_valid_unique_novel = num_novel / len(mols)
    return p_valid, p_valid_unique, p_valid_unique_novel


def compute_mean_and_95_ci(data):
    mean = np.mean(data)
    std_err = np.std(data) / np.sqrt(len(data))
    margin_of_error = 1.96 * std_err
    return mean, margin_of_error


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

    # Load config file
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

    # Load preprocessed training data for computing novelty
    DATA_ROOT = os.path.join(ROOT, "data")
    datamodule = DataModule(DATA_ROOT, data_set=params["data_set"])
    datamodule.setup()
    reference_smiles = datamodule.training_data.smiles

    # Evaluate generated molecules across all available seeds
    data_path = Path(os.path.join(ROOT, params["data_path"]))
    for subdir in data_path.iterdir():
        if subdir.is_dir() and subdir.name.startswith("seed"):
            subdata_path = os.path.join(data_path, subdir.name)
            builder = MoleculeBuilder(vocab=params["data_set"])
            x, pos, batch = builder.load_tensor_from_file(subdata_path)

            (
                atom_stability,
                mol_stability,
                edm_valid,
                edm_unique,
                edm_invalid_idxs,
            ) = edm_metrics(x, pos, batch, params["data_set"])

            mols = builder.generate_rdkit_molecules(x, pos, batch, progress_bar=True)

            (xyz2mol_valid, xyz2mol_valid_x_unique, xyz2mol_valid_x_unique_x_novel) = (
                compute_validity_uniqueness_novelty(mols, reference_smiles)
            )

            with open(os.path.join(subdata_path, "evaluation_results.txt"), "w") as f:
                f.write(f"Atom stability: {atom_stability*100:.2f}%\n")
                f.write(f"Molecule stability: {mol_stability*100:.2f}%\n")
                f.write(f"Lookup valid: {edm_valid*100:.2f}%\n")
                f.write(
                    f"Lookup valid x unique: { edm_valid * edm_unique * 100:.2f}%\n"
                )
                f.write(f"xyz2mol valid: {xyz2mol_valid*100:.2f}%\n")
                f.write(
                    f"xyz2mol valid x unique: {xyz2mol_valid_x_unique * 100:.2f}%\n"
                )
                f.write(
                    f"xyz2mol valid x unique x novel: { xyz2mol_valid_x_unique_x_novel *100:.2f}%\n"
                )
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

    # Collect and average results across all seeds
    atom_stability_lst = []
    molecule_stability_lst = []
    lookup_valid_lst = []
    lookup_valid_x_unique_lst = []
    xyz2mol_valid_lst = []
    xyz2mol_valid_x_unique_lst = []
    xyz2mol_valid_x_unique_x_novel_lst = []

    for subdir in data_path.iterdir():
        if Path.is_dir(subdir) and subdir.name.startswith("seed"):
            results_file = os.path.join(subdir, "evaluation_results.txt")
        else:
            continue
        with open(results_file, "r") as f:
            lines = f.readlines()
            atom_stability_lst.append(float(lines[0].strip().split(": ")[1].strip("%")))
            molecule_stability_lst.append(
                float(lines[1].strip().split(": ")[1].strip("%"))
            )
            lookup_valid_lst.append(float(lines[2].strip().split(": ")[1].strip("%")))
            lookup_valid_x_unique_lst.append(
                float(lines[3].strip().split(": ")[1].strip("%"))
            )
            xyz2mol_valid_lst.append(float(lines[4].strip().split(": ")[1].strip("%")))
            xyz2mol_valid_x_unique_lst.append(
                float(lines[5].strip().split(": ")[1].strip("%"))
            )
            xyz2mol_valid_x_unique_x_novel_lst.append(
                float(lines[6].strip().split(": ")[1].strip("%"))
            )

    atom_stability_mean, atom_stability_ci = compute_mean_and_95_ci(atom_stability_lst)
    molecule_stability_mean, molecule_stability_ci = compute_mean_and_95_ci(
        molecule_stability_lst
    )
    lookup_valid_mean, lookup_valid_ci = compute_mean_and_95_ci(lookup_valid_lst)
    lookup_valid_x_unique_mean, lookup_valid_x_unique_ci = compute_mean_and_95_ci(
        lookup_valid_x_unique_lst
    )
    xyz2mol_valid_mean, xyz2mol_valid_ci = compute_mean_and_95_ci(xyz2mol_valid_lst)
    xyz2mol_valid_x_unique_mean, xyz2mol_valid_x_unique_ci = compute_mean_and_95_ci(
        xyz2mol_valid_x_unique_lst
    )
    xyz2mol_valid_x_unique_x_novel_mean, xyz2mol_valid_x_unique_x_novel_ci = (
        compute_mean_and_95_ci(xyz2mol_valid_x_unique_x_novel_lst)
    )

    with open(os.path.join(data_path, "evaluation_summary.txt"), "w") as f:
        f.write(
            f"Atom stability: {atom_stability_mean:.2f}% ± {atom_stability_ci:.2f}%\n"
        )
        f.write(
            f"Molecule stability: {molecule_stability_mean:.2f}% ± {molecule_stability_ci:.2f}%\n"
        )
        f.write(f"Lookup valid: {lookup_valid_mean:.2f}% ± {lookup_valid_ci:.2f}%\n")
        f.write(
            f"Lookup valid x unique: {lookup_valid_x_unique_mean:.2f}% ± {lookup_valid_x_unique_ci:.2f}%\n"
        )
        f.write(f"xyz2mol valid: {xyz2mol_valid_mean:.2f}% ± {xyz2mol_valid_ci:.2f}%\n")
        f.write(
            f"xyz2mol valid x unique: {xyz2mol_valid_x_unique_mean:.2f}% ± {xyz2mol_valid_x_unique_ci:.2f}%\n"
        )
        f.write(
            f"xyz2mol valid x unique x novel: {xyz2mol_valid_x_unique_x_novel_mean:.2f}% ± {xyz2mol_valid_x_unique_x_novel_ci:.2f}%\n"
        )
        f.write(f"Data set: {params['data_set']}\n")
        f.write(f"RDKit version: {rdkit.__version__}\n")
