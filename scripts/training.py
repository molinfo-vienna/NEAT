import argparse
import os

import torch
import torch_geometric
import yaml
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from neat.dataset import DataModule
from neat.model import GenerationMonitor, NEAT

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.set_float32_matmul_precision("medium")
torch_geometric.seed_everything(42)
seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def training(args: argparse.Namespace) -> None:
    ROOT = os.getcwd()
    if args.config_file is not None:
        CONFIG_FILE_PATH = args.config_file
        print(f"Using config file: {CONFIG_FILE_PATH}")
    else:
        CONFIG_FILE_PATH = os.path.join(ROOT, "scripts", "config_training.yaml")
        print(f"Using default config file: {CONFIG_FILE_PATH}")

    MODEL = NEAT
    params = yaml.load(
        open(CONFIG_FILE_PATH, "r"),
        Loader=yaml.FullLoader,
    )

    DATA_ROOT = os.path.join(ROOT, "data", params["data_set"])
    datamodule = DataModule(
        DATA_ROOT,
        batch_size=params["batch_size"],
        source_target_split=params["source_target_split"],
        noise_std=params["noise_std"],
        num_workers=8,
    )
    datamodule.setup()

    accumulate_grad_batches = params.pop("accumulate_grad_batches")

    # ------- Model initialization -----------------------------

    # Initialize a new model instance...
    model = MODEL(**params)

    # ... OR load and fine-tune model
    # MODEL_NUMBER = 85
    # MODEL_PATH = f"{ROOT}/logs/{MODEL.__name__}/version_{MODEL_NUMBER}/"
    # checkpoints_dir = os.path.join(MODEL_PATH, "checkpoints")
    # pt_files = [
    #     f
    #     for f in os.listdir(checkpoints_dir)
    #     if f.endswith(".ckpt") and f.startswith("best-val-validity")
    # ]
    # if not pt_files:
    #     raise FileNotFoundError(f"No .ckpt files found in {checkpoints_dir}")

    # CHECKPOINTS_PATH = os.path.join(checkpoints_dir, pt_files[0])
    # print(f"Using checkpoint file: {CHECKPOINTS_PATH}")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = MODEL.load_from_checkpoint(CHECKPOINTS_PATH, map_location=device)

    # CAREFUL:
    # The trainer.fit() method needs to be called with the right arguments,
    # depending on whether we are initializing a new model or loading one.

    # ----------------------------------------------------------

    tb_logger = TensorBoardLogger(
        os.path.join(ROOT, "logs"),
        name=f"{MODEL.__name__}",
        default_hp_metric=False,
    )
    # Define the first ModelCheckpoint for validation loss
    checkpoint_val_loss = ModelCheckpoint(
        monitor="val/val_loss",
        mode="min",
        filename="best-val-loss-{epoch:02d}",
        save_top_k=1,
        every_n_epochs=10,
    )

    # Define the second ModelCheckpoint for molecular validity
    generate_every_n_epochs = 50
    checkpoint_validity = ModelCheckpoint(
        monitor="val/validity",
        mode="max",
        filename="best-val-validity-{epoch:02d}",
        save_top_k=1,
        every_n_epochs=generate_every_n_epochs,
    )
    callbacks = [
        GenerationMonitor(num_samples=10000, every_n_epochs=generate_every_n_epochs),
        checkpoint_val_loss,
        checkpoint_validity,
        LearningRateMonitor(logging_interval="epoch"),
    ]
    trainer = Trainer(
        devices=[3],
        max_epochs=params["max_epochs"],
        accelerator="gpu",
        logger=tb_logger,
        log_every_n_steps=8,
        callbacks=callbacks,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=1.0,
        precision="bf16-mixed",
    )

    trainer.fit(model=model, datamodule=datamodule)  # For new model training
    # trainer.fit(model=model, datamodule=datamodule, ckpt_path=CHECKPOINTS_PATH)  # For fine-tuning


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        dest="config_file",
        required=False,
        metavar="<file>",
        help="Config file for training.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    training(parseArgs())
