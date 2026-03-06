import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import py3Dmol
import rdkit
import yaml
from rdkit.Chem import AllChem, Draw, MolToSmiles, rdDepictor

from neat.dataset import DataModule
from neat.model.molecule_builder import MoleculeBuilder
from neat.utils.edm_metrics import edm_metrics

NUM_MOLECULES_PLOTTED = 100
NUM_MOLECULES_PER_ROW = 5
RESOLUTION = 400
ROOT = os.getcwd()


def compute_validity_uniqueness_novelty(
    smiles: list[str], reference_smiles: list[str] = None
) -> tuple[float, float, float]:
    """Compute validity, uniqueness and novelty ratio of generated molecules.

    Args:
        smiles (List[str]): generated molecules as SMILES strings
        reference_smiles (List[str]): reference canonical SMILES strings for novelty computation

    Returns:
        Tuple[float, float, float]: validity, uniqueness, and novelty ratios
    """
    unique_smiles = set()
    num_valid = 0

    for smile in smiles:
        if smile is None:
            continue
        num_valid += 1
        unique_smiles.add(smile)
    num_unique = len(unique_smiles)

    p_valid = num_valid / len(smiles)
    p_valid_unique = num_unique / len(smiles)
    p_valid_unique_novel = None

    if reference_smiles is not None:
        ref_set = set(reference_smiles)
        num_novel = len(unique_smiles - ref_set)
        p_valid_unique_novel = num_novel / len(smiles)

    return p_valid, p_valid_unique, p_valid_unique_novel


def compute_mean_and_95_ci(data: list[float]) -> tuple[float, float]:
    """Compute mean and 95% confidence interval for a list of data.

    Args:
        data (List[float]): list of data points.

    Returns:
        Tuple[float, float]: mean and 95% confidence interval.
    """
    mean = np.mean(data)
    std_err = np.std(data) / np.sqrt(len(data))
    margin_of_error = 1.96 * std_err
    return mean, margin_of_error


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate generated molecules using various metrics.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        None
    """
    # Load config file
    if args.config_file is not None:
        config_file_path = args.config_file
        print(f"Using config file: {config_file_path}")
    else:
        config_file_path = os.path.join(ROOT, "scripts", "config_evaluation.yaml")
        print(f"Using default config file: {config_file_path}")
    params = yaml.load(
        open(config_file_path, "r"),
        Loader=yaml.FullLoader,
    )

    # Load preprocessed training data for computing novelty
    if params["compute_novelty"]:
        data_root = os.path.join(ROOT, "data")
        datamodule = DataModule(data_root, data_set=params["data_set"])
        datamodule.setup()
        reference_smiles = datamodule.training_data.smiles
    else:
        reference_smiles = None

    # Evaluate generated molecules across all available seeds or prefixes
    atom_stability_lst = []
    molecule_stability_lst = []
    lookup_valid_lst = []
    lookup_valid_x_unique_lst = []
    xyz2mol_valid_lst = []
    xyz2mol_valid_x_unique_lst = []
    xyz2mol_valid_x_unique_x_novel_lst = []
    bond_predictor_valid_lst = []
    bond_predictor_valid_x_unique_lst = []
    bond_predictor_valid_x_unique_x_novel_lst = []
    use_bond_predictor = params.get("bond_predictor_path") is not None
    data_path = Path(os.path.join(ROOT, params["data_path"]))
    for subdir in data_path.iterdir():
        if subdir.is_dir() and (
            subdir.name.startswith("seed") or subdir.name.startswith("prefix")
        ):
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
            edm_valid_x_unique = edm_valid * edm_unique
            atom_stability_lst.append(atom_stability)
            molecule_stability_lst.append(mol_stability)
            lookup_valid_lst.append(edm_valid)
            lookup_valid_x_unique_lst.append(edm_valid_x_unique)

            mols = builder.generate_rdkit_molecules_via_xyz2mol(
                x, pos, batch, progress_bar=True
            )
            smiles = [
                MolToSmiles(mol, canonical=True) if mol is not None else None for mol in mols
            ]
            (
                xyz2mol_valid, 
                xyz2mol_valid_x_unique, 
                xyz2mol_valid_x_unique_x_novel 
            )= compute_validity_uniqueness_novelty(smiles, reference_smiles)
            xyz2mol_valid_lst.append(xyz2mol_valid)
            xyz2mol_valid_x_unique_lst.append(xyz2mol_valid_x_unique)
            xyz2mol_valid_x_unique_x_novel_lst.append(xyz2mol_valid_x_unique_x_novel)

            if use_bond_predictor:
                mols_bp = builder.generate_rdkit_molecules_via_bond_predictor(
                    x, pos, batch,
                    bond_predictor_path=params["bond_predictor_path"],
                    progress_bar=True,
                )
                smiles_bp = [
                    MolToSmiles(mol, canonical=True) if mol is not None else None
                    for mol in mols_bp
                ]
                (
                    bp_valid,
                    bp_valid_x_unique,
                    bp_valid_x_unique_x_novel,
                ) = compute_validity_uniqueness_novelty(smiles_bp, reference_smiles)
                bond_predictor_valid_lst.append(bp_valid)
                bond_predictor_valid_x_unique_lst.append(bp_valid_x_unique)
                bond_predictor_valid_x_unique_x_novel_lst.append(bp_valid_x_unique_x_novel)

            with open(os.path.join(subdata_path, "evaluation_results.txt"), "w") as f:
                f.write(f"Atom stability: {atom_stability*100:.2f}%\n")
                f.write(f"Molecule stability: {mol_stability*100:.2f}%\n")
                f.write(f"Lookup valid: {edm_valid*100:.2f}%\n")
                f.write(f"Lookup valid x unique: { edm_valid_x_unique * 100:.2f}%\n")
                f.write(f"xyz2mol valid: {xyz2mol_valid*100:.2f}%\n")
                f.write(
                    f"xyz2mol valid x unique: {xyz2mol_valid_x_unique * 100:.2f}%\n"
                )
                if params["compute_novelty"]:
                    f.write(f"xyz2mol valid x unique x novel: { xyz2mol_valid_x_unique_x_novel *100:.2f}%\n")
                if use_bond_predictor:
                    f.write(f"bond_predictor valid: {bp_valid*100:.2f}%\n")
                    f.write(f"bond_predictor valid x unique: {bp_valid_x_unique*100:.2f}%\n")
                    if params["compute_novelty"]:
                        f.write(f"bond_predictor valid x unique x novel: {bp_valid_x_unique_x_novel*100:.2f}%\n")
                f.write(f"Data set: {params['data_set']}\n")
                f.write(f"RDKit version: {rdkit.__version__}\n")

            mols = builder.generate_rdkit_molecules_via_xyz2mol(
                x, pos, batch, break_after_k_mols=NUM_MOLECULES_PLOTTED
            )
            for mol in mols:
                if mol is not None:
                    rdDepictor.Compute2DCoords(
                        mol
                    )  # Optimize 2D coordinates for better visualization
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
                    mol = AllChem.RemoveHs(mol)
                    rdDepictor.Compute2DCoords(mol)
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

            with open(
                os.path.join(subdata_path, "generated_molecules_3d.html"), "w"
            ) as f:
                f.write(view._make_html())

            print(
                f"Saved 3D visualization to {os.path.join(subdata_path, 'generated_molecules_3d.html')}"
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
    if params["compute_novelty"]:
        xyz2mol_valid_x_unique_x_novel_mean, xyz2mol_valid_x_unique_x_novel_ci = (
            compute_mean_and_95_ci(xyz2mol_valid_x_unique_x_novel_lst)
        )
    if use_bond_predictor:
        bond_predictor_valid_mean, bond_predictor_valid_ci = compute_mean_and_95_ci(
            bond_predictor_valid_lst
        )
        bond_predictor_valid_x_unique_mean, bond_predictor_valid_x_unique_ci = (
            compute_mean_and_95_ci(bond_predictor_valid_x_unique_lst)
        )
        if params["compute_novelty"]:
            (
                bond_predictor_valid_x_unique_x_novel_mean,
                bond_predictor_valid_x_unique_x_novel_ci,
            ) = compute_mean_and_95_ci(bond_predictor_valid_x_unique_x_novel_lst)

    with open(os.path.join(data_path, "evaluation_summary.txt"), "w") as f:
        f.write(
            f"Atom stability: {atom_stability_mean*100:.2f}% ± {atom_stability_ci*100:.2f}%\n"
        )
        f.write(
            f"Molecule stability: {molecule_stability_mean*100:.2f}% ± {molecule_stability_ci*100:.2f}%\n"
        )
        f.write(
            f"Lookup valid: {lookup_valid_mean*100:.2f}% ± {lookup_valid_ci*100:.2f}%\n"
        )
        f.write(
            f"Lookup valid x unique: {lookup_valid_x_unique_mean*100:.2f}% ± {lookup_valid_x_unique_ci*100:.2f}%\n"
        )
        f.write(
            f"xyz2mol valid: {xyz2mol_valid_mean*100:.2f}% ± {xyz2mol_valid_ci*100:.2f}%\n"
        )
        f.write(
            f"xyz2mol valid x unique: {xyz2mol_valid_x_unique_mean*100:.2f}% ± {xyz2mol_valid_x_unique_ci*100:.2f}%\n"
        )
        if params["compute_novelty"]:
            f.write(
                f"xyz2mol valid x unique x novel: {xyz2mol_valid_x_unique_x_novel_mean*100:.2f}% ± {xyz2mol_valid_x_unique_x_novel_ci*100:.2f}%\n"
            )
        if use_bond_predictor:
            f.write(
                f"bond_predictor valid: {bond_predictor_valid_mean*100:.2f}% ± {bond_predictor_valid_ci*100:.2f}%\n"
            )
            f.write(
                f"bond_predictor valid x unique: {bond_predictor_valid_x_unique_mean*100:.2f}% ± {bond_predictor_valid_x_unique_ci*100:.2f}%\n"
            )
            if params["compute_novelty"]:
                f.write(
                    f"bond_predictor valid x unique x novel: {bond_predictor_valid_x_unique_x_novel_mean*100:.2f}% ± {bond_predictor_valid_x_unique_x_novel_ci*100:.2f}%\n"
                )
        f.write(f"Data set: {params['data_set']}\n")
        f.write(f"RDKit version: {rdkit.__version__}\n")


if __name__ == "__main__":
    start_time = datetime.now()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        dest="config_file",
        required=False,
        metavar="<file>",
        help="Config file for evaluation.",
    )

    args = parser.parse_args()

    evaluate(args)

    end_time = datetime.now()
    print(f"Total evaluation time: {end_time - start_time}")
