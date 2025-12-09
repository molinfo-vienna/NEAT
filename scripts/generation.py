import argparse
import os

import torch
import torch_geometric
import yaml
from lightning import seed_everything

from molgen.model import MolGen
from molgen.model.molecule_builder import MoleculeBuilder

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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

    CHECKPOINTS_PATH = os.path.join(checkpoints_dir, pt_files[0])
    print(f"Using checkpoint file: {CHECKPOINTS_PATH}")

    MODEL = MolGen
    model = MODEL.load_from_checkpoint(CHECKPOINTS_PATH, map_location=device)

    seeds = [i for i in range(params.get("num_runs", 1))]
    for seed in seeds:
        torch_geometric.seed_everything(seed)
        seed_everything(seed)

        with torch.no_grad():
            model.eval()
            if "prefix_path" in params:
                builder = MoleculeBuilder()
                prefix_x, prefix_pos, _ = builder.load_tensor_from_file(
                    params["prefix_path"]
                )
                x, pos, batch = model.generate(
                    batch_size=params["num_molecules"],
                    max_atoms=params["max_atoms"],
                    num_time_steps=params["num_time_steps"],
                    prefix_x=prefix_x,
                    prefix_pos=prefix_pos,
                )
            else:
                x, pos, batch = model.generate(
                    batch_size=params["num_molecules"],
                    max_atoms=params["max_atoms"],
                    num_time_steps=params["num_time_steps"],
                )

        out_dir = os.path.join(params["output_path"], f"seed_{seed}")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        torch.save(x, os.path.join(out_dir, "x.pt"))
        torch.save(pos, os.path.join(out_dir, "pos.pt"))
        torch.save(batch, os.path.join(out_dir, "batch.pt"))


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
