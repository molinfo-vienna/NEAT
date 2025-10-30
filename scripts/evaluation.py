import argparse
import os
import yaml

from rdkit import Chem

from molgen.model.molecule_builder import MoleculeBuilder


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
            smiles = Chem.MolToSmiles(mol, canonical=True)
            unique_smiles.add(smiles)
    return len(unique_smiles)


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
        CONFIG_FILE_PATH = os.path.join(ROOT, "scripts", "config_evaluate.yaml")
        print(f"Using default config file: {CONFIG_FILE_PATH}")

    params = yaml.load(
        open(CONFIG_FILE_PATH, "r"),
        Loader=yaml.FullLoader,
    )

    builder = MoleculeBuilder()
    x, pos, batch = builder.load_tensor_from_file(params["data_path"])
    mols = builder.generate_rdkit_molecules(x, pos, batch)

    n_valid = compute_validity(mols)
    n_unique = compute_uniqueness(mols)

    with open(os.path.join(params["data_path"], "evaluation_results.txt"), "w") as f:
        f.write(
            f"Number of valid molecules: {n_valid} out of {len(mols)} ({n_valid/len(mols)*100:.2f}%)\n"
        )
        f.write(
            f"Number of unique molecules: {n_unique} out of {n_valid} valid molecules ({n_unique/n_valid*100:.2f}%)\n"
        )

    img = Chem.Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(400, 400))
    img.save(os.path.join(params["data_path"], "generated_molecules.png"))

    mols_2d = builder.generate_rdkit_molecules(
        x, pos, batch, optimized_for_2d=True, remove_hydrogens=True
    )
    img = Chem.Draw.MolsToGridImage(mols_2d, molsPerRow=5, subImgSize=(400, 400))
    img.save(os.path.join(params["data_path"], "generated_molecules_2d.png"))

    print(f"Saved generated molecules images to {os.path.join(params['data_path'])}.")
