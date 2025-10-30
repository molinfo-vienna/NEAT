import argparse
import os
import yaml

import torch
import torch_geometric

from lightning import seed_everything

from molgen.model import MolGen

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set device for model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# settings for deterministic generation
torch.set_float32_matmul_precision("medium")
torch_geometric.seed_everything(42)
seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def generate(args: argparse.Namespace) -> None:
    # Load settings
    ROOT = os.getcwd()
    if args.config_file is not None:
        CONFIG_FILE_PATH = args.config_file
        print(f"Using config file: {CONFIG_FILE_PATH}")
    else:
        CONFIG_FILE_PATH = os.path.join(ROOT, "scripts", "config_generate.yaml")
        print(f"Using default config file: {CONFIG_FILE_PATH}")

    # Generation configs
    MODEL = MolGen
    params = yaml.load(
        open(CONFIG_FILE_PATH, "r"),
        Loader=yaml.FullLoader,
    )

    # Checkpoints path (find the first .ckpt file in the checkpoints folder)
    checkpoints_dir = os.path.join(params["checkpoints_path"], "checkpoints")
    pt_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".ckpt")]
    if not pt_files:
        raise FileNotFoundError(f"No .ckpt files found in {checkpoints_dir}")

    CHECKPOINTS_PATH = os.path.join(checkpoints_dir, pt_files[0])
    print(f"Using checkpoint file: {CHECKPOINTS_PATH}")

    # Load model
    model = MODEL.load_from_checkpoint(CHECKPOINTS_PATH, map_location=device)

    with torch.no_grad():
        model.eval()
        # Generate molecules
        x, pos, batch = model.generate(
            batch_size=params["num_molecules"],
            max_atoms=params["max_atoms"],
            temperature=params["temperature"],
            top_k=params["top_k"],
            num_time_steps=params["num_time_steps"],
        )

    # Save molecules to output file
    if not os.path.exists(params["output_path"]):
        os.makedirs(params["output_path"])
    torch.save(x, os.path.join(params["output_path"], "x.pt"))
    torch.save(pos, os.path.join(params["output_path"], "pos.pt"))
    torch.save(batch, os.path.join(params["output_path"], "batch.pt"))


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
