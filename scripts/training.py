import argparse
import os
import yaml

from lightning import Trainer, seed_everything
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import torch_geometric

from molgen.dataset import DataModule
from molgen.model import MolGen, CurriculumLearningScheduler

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Settings for deterministic training
torch.set_float32_matmul_precision("medium")
torch_geometric.seed_everything(42)
seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def training(args: argparse.Namespace) -> None:
    # Load settings
    ROOT = os.getcwd()
    if args.config_file is not None:
        CONFIG_FILE_PATH = args.config_file
        print(f"Using config file: {CONFIG_FILE_PATH}")
    else:
        CONFIG_FILE_PATH = os.path.join(ROOT, "scripts", "config_training.yaml")
        print(f"Using default config file: {CONFIG_FILE_PATH}")

    # Training configs
    MODEL = MolGen
    params = yaml.load(
        open(CONFIG_FILE_PATH, "r"),
        Loader=yaml.FullLoader,
    )

    DATA_ROOT = os.path.join(ROOT, "data", params["data_set"])
    datamodule = DataModule(
        DATA_ROOT,
        batch_size=params["batch_size"],
        split=params["data_split"],
    )
    datamodule.setup()

    accumulate_grad_batches = params.pop("accumulate_grad_batches")

    # Initialize and train model
    model = MODEL(**params)
    # MODEL_NUMBER = 0
    # MODEL_PATH = f"{ROOT}/logs/{MODEL.__name__}/version_{MODEL_NUMBER}/"
    # model = load_model_from_path(MODEL_PATH, MODEL)
    tb_logger = TensorBoardLogger(
        os.path.join(ROOT, "logs"),
        # "/data/local/MolGen",
        name=f"{MODEL.__name__}",
        default_hp_metric=False,
    )
    callbacks = [
        # CurriculumLearningScheduler(1, 25, 1.01),
        ModelCheckpoint(
            monitor="val/val_loss",
            mode="min",
            every_n_epochs=10,
        ),
    ]
    trainer = Trainer(
        devices=1,
        max_epochs=params["max_epochs"],
        accelerator="gpu",
        logger=tb_logger,
        log_every_n_steps=8,
        callbacks=callbacks,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )
    trainer.fit(model=model, datamodule=datamodule)


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
