import argparse
import ast
import os

from lightning import seed_everything
import numpy as np
from rdkit import Chem
import torch
import torch_geometric
import yaml

from neat.model import NEAT
from neat.dataset import DataModule, GEOMDataSet
from neat.utils.edm_metrics import edm_metrics

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch_geometric.seed_everything(0)
seed_everything(0)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mol_to_tensor(mol, vocabulary):
    n = mol.GetNumAtoms()

    x = torch.tensor(
        [vocabulary[atom.GetAtomicNum()] for atom in mol.GetAtoms()],
        dtype=torch.long,
    )

    conf = mol.GetConformer()
    pos = torch.zeros((n, 3), dtype=torch.float32)
    for i in range(n):
        position = conf.GetAtomPosition(i)
        pos[i, 0] = position.x
        pos[i, 1] = position.y
        pos[i, 2] = position.z

    pos -= pos.mean(dim=0, keepdim=True)

    return x, pos


def read_sdf_dummy_indices_csv(sdf_path, vocab):
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mols, dummy_indices = [], []
    for mol in suppl:
        if mol is None:
            continue
        mols.append(mol_to_tensor(mol, vocab))
        if mol.HasProp("R_group_indices"):
            idxs = mol.GetProp("R_group_indices")
            idxs = ast.literal_eval(idxs)
        else:
            idxs = []
        dummy_indices.append(idxs)
    return mols, dummy_indices


def compute_mean_and_95_ci(data):
    mean = np.mean(data)
    std_err = np.std(data) / np.sqrt(len(data))
    margin_of_error = 1.96 * std_err
    return mean, margin_of_error


def generate(args: argparse.Namespace) -> None:
    ROOT = os.getcwd()
    if args.config_file is not None:
        CONFIG_FILE_PATH = args.config_file
        print(f"Using config file: {CONFIG_FILE_PATH}")
    else:
        CONFIG_FILE_PATH = os.path.join(ROOT, "scripts", "config_generation.yaml")
        print(f"Using default config file: {CONFIG_FILE_PATH}")

    params = yaml.load(
        open(CONFIG_FILE_PATH, "r"),
        Loader=yaml.FullLoader,
    )

    checkpoints_dir = os.path.join(params["checkpoints_path"], "checkpoints")
    pt_files = [
        f
        for f in os.listdir(checkpoints_dir)
        if f.endswith(".ckpt") and f.startswith("best-val-loss")
    ]
    if not pt_files:
        raise FileNotFoundError(f"No .ckpt files found in {checkpoints_dir}")

    # Load preprocessed training data for computing novelty
    DATA_ROOT = os.path.join(ROOT, "data")
    datamodule = DataModule(DATA_ROOT, data_set=params["data_set"])
    datamodule.setup()

    # Load prefix molecules from file
    prefix_path = os.path.join(os.getcwd(), "data", "GEOM", "prefixes.sdf")
    vocab = GEOMDataSet.VOCABULARY
    mols, dummy_idxs = read_sdf_dummy_indices_csv(prefix_path, vocab)

    CHECKPOINTS_PATH = os.path.join(checkpoints_dir, pt_files[0])
    print(f"Using checkpoint file: {CHECKPOINTS_PATH}")

    MODEL = NEAT
    model = MODEL.load_from_checkpoint(CHECKPOINTS_PATH, map_location=device)

    stability = []
    validity = []
    uniqueness = []

    for mol_index, ((x, pos), dummy_idx) in enumerate(zip(mols, dummy_idxs)):
        with torch.no_grad():
            model.eval()
            n_atoms = x.size(0)
            mask = torch.ones(n_atoms, dtype=torch.bool)
            if len(dummy_idx) > 0:
                mask[dummy_idx] = False
            prefix_x = x[mask]
            prefix_pos = pos[mask]
            prefix_pos -= prefix_pos.mean(dim=0, keepdim=True)
            x, pos, batch = model.generate(
                batch_size=params["num_molecules"],
                max_atoms=params["max_atoms"],
                num_time_steps=params["num_time_steps"],
                prefix_x=prefix_x,
                prefix_pos=prefix_pos,
                time_step_spacing=params["time_step_spacing"],
            )

            (
                atom_stability,
                mol_stability,
                edm_valid,
                edm_unique,
                edm_invalid_idxs,
            ) = edm_metrics(x.cpu(), pos.cpu(), batch.cpu(), params["data_set"])

            edm_unique = edm_unique * edm_valid
            stability.append(atom_stability)
            validity.append(edm_valid)
            uniqueness.append(edm_unique)
            print(
                f"Prefix with {n_atoms} atoms: Atom stability: {atom_stability*100:.2f}%, Validity: {edm_valid*100:.2f}%, Uniqueness: {edm_unique*100:.2f}%"
            )

            out_dir = os.path.join(
                params["output_path"], "prefix_generation", f"prefix_{mol_index}"
            )
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            torch.save(x, os.path.join(out_dir, "x.pt"))
            torch.save(pos, os.path.join(out_dir, "pos.pt"))
            torch.save(batch, os.path.join(out_dir, "batch.pt"))
            with open(os.path.join(out_dir, "evaluation_results.txt"), "w") as f:
                f.write(f"Atom stability: {atom_stability*100:.2f}%\n")
                f.write(f"Molecule stability: {mol_stability*100:.2f}%\n")
                f.write(f"Lookup valid: {edm_valid*100:.2f}%\n")
                f.write(f"Lookup valid x unique: { edm_unique * 100:.2f}%\n")

        atom_stability_mean, atom_stability_ci = compute_mean_and_95_ci(stability)
        lookup_valid_mean, lookup_valid_ci = compute_mean_and_95_ci(validity)
        lookup_valid_x_unique_mean, lookup_valid_x_unique_ci = compute_mean_and_95_ci(
            uniqueness
        )
        out_dir = os.path.join(params["output_path"], "prefix_generation")
        with open(os.path.join(out_dir, "evaluation_summary.txt"), "w") as f:
            f.write(
                f"Atom stability: {atom_stability_mean:.2f}% ± {atom_stability_ci:.2f}%\n"
            )
            f.write(
                f"Lookup valid: {lookup_valid_mean:.2f}% ± {lookup_valid_ci:.2f}%\n"
            )
            f.write(
                f"Lookup valid x unique: {lookup_valid_x_unique_mean:.2f}% ± {lookup_valid_x_unique_ci:.2f}%\n"
            )
        print(
            f"Average Atom stability: {sum(stability)/len(stability)*100:.2f}%, Average Validity: {sum(validity)/len(validity)*100:.2f}%, Average Uniqueness: {sum(uniqueness)/len(uniqueness)*100:.2f}%"
        )


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        dest="config_file",
        required=False,
        metavar="<file>",
        help="Config file for generation.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    generate(parseArgs())
