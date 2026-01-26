import argparse
import ast
import os
from datetime import datetime

import torch
import torch_geometric
import yaml
from lightning import seed_everything
from rdkit import Chem
from torch_geometric.data import Batch

from neat.dataset import GEOMDataSet
from neat.model import NEAT

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch_geometric.seed_everything(0)
seed_everything(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.getcwd()


def mol_to_tensor(
    mol: Chem.Mol, vocabulary: dict[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert RDKit Mol to PyTorch tensors for atom types and positions.

    Args:
        mol (Chem.Mol): RDKit molecule.
        vocabulary (dict[int, int]): Mapping from atomic numbers to vocabulary indices.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Atom types tensor and positions tensor.
    """
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


def read_sdf_dummy_indices(
    sdf_path: str, vocabulary: dict[int, int]
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[list[int]]]:
    """Read molecules and dummy atom indices from an SDF file.

    Args:
        sdf_path (str): Path to the SDF file.
        vocabulary (dict[int, int]): Mapping from atomic numbers to vocabulary indices.

    Returns:
        tuple[list[tuple[torch.Tensor, torch.Tensor]], list[list[int]]]:
            - List of tuples containing atom types and positions tensors for each molecule.
            - List of lists containing dummy atom indices for each molecule.
    """
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mols, dummy_indices = [], []
    for mol in suppl:
        if mol is None:
            continue
        mols.append(mol_to_tensor(mol, vocabulary))
        if mol.HasProp("R_group_indices"):
            idxs = mol.GetProp("R_group_indices")
            idxs = ast.literal_eval(idxs)
        else:
            idxs = []
        dummy_indices.append(idxs)
    return mols, dummy_indices


def generate(args: argparse.Namespace) -> None:
    """Complete prefixes using the NEAT model.

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
        config_file_path = os.path.join(ROOT, "scripts", "config_generation.yaml")
        print(f"Using default config file: {config_file_path}")

    params = yaml.load(
        open(config_file_path, "r"),
        Loader=yaml.FullLoader,
    )

    # Load model
    checkpoints_dir = os.path.join(params["checkpoints_path"], "checkpoints")
    pt_files = [
        f
        for f in os.listdir(checkpoints_dir)
        if f.endswith(".ckpt") and f.startswith("best-val-loss")
    ]
    if not pt_files:
        raise FileNotFoundError(f"No .ckpt files found in {checkpoints_dir}")

    checkpoint_path = os.path.join(checkpoints_dir, pt_files[0])
    print(f"Using checkpoint file: {checkpoint_path}")

    MODEL = NEAT
    model = MODEL.load_from_checkpoint(checkpoint_path, map_location=DEVICE)

    # Load prefix molecules from file
    prefix_path = os.path.join(os.getcwd(), "prefixes", "prefixes.sdf")
    vocabulary = GEOMDataSet.VOCABULARY
    mols, dummy_idxs = read_sdf_dummy_indices(prefix_path, vocabulary)

    # Set up batching
    num_molecules = params["num_molecules"]
    batch_size = params["batch_size"]
    num_batches = (num_molecules + batch_size - 1) // batch_size
    if (num_molecules % batch_size) == 0:
        num_mols_per_batch = [batch_size] * num_batches
    else:
        num_mols_per_batch = [batch_size] * (num_batches - 1) + [
            num_molecules % batch_size
        ]

    # Generate molecules for each prefix
    for mol_index, ((x, pos), dummy_idx) in enumerate(zip(mols, dummy_idxs)):

        prefix_start_time = datetime.now()

        generated_batches = []
        for batch_idx in range(num_batches):
            num_mols_batch = num_mols_per_batch[batch_idx]

            with torch.no_grad():
                model.eval()
                n_atoms = x.size(0)
                mask = torch.ones(n_atoms, dtype=torch.bool)
                if len(dummy_idx) > 0:
                    mask[dummy_idx] = False
                prefix_x = x[mask]
                prefix_pos = pos[mask]
                generated_batch = model.generate(
                    batch_size=num_mols_batch,
                    max_atoms=params["max_atoms"],
                    num_time_steps=params["num_time_steps"],
                    prefix_x=prefix_x,
                    prefix_pos=prefix_pos,
                    time_step_spacing=params["time_step_spacing"],
                )
            generated_batches.append(generated_batch)

        generated_mols = Batch.from_data_list(generated_batches)

        out_dir = os.path.join(params["output_path"], "prefix", f"prefix_{mol_index}")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        torch.save(generated_mols, os.path.join(out_dir, "generated_mols.pt"))

        prefix_end_time = datetime.now()
        print(
            f"Generation time for prefix {mol_index}: {prefix_end_time - prefix_start_time}"
        )


if __name__ == "__main__":
    start_time = datetime.now()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        dest="config_file",
        required=False,
        metavar="<file>",
        help="Config file for generation.",
    )

    args = parser.parse_args()

    generate(args)

    end_time = datetime.now()
    print(f"Total generation time: {end_time - start_time}")
