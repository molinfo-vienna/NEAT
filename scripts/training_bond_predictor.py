"""Train bond predictor models on QM9 and GEOM datasets.

Trains a BondPredictor GNN that, given atom types and 3D coordinates,
predicts bond types (no bond, single, double, triple, aromatic) between
all atom pairs. One model is trained per dataset.

Usage:
    python scripts/training_bond_predictor.py --config scripts/config_bond_predictor.yaml
"""

import argparse
import os

import torch
import torch_geometric
import yaml
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from neat.dataset import GEOMDataSet, QM9DataSet
from neat.dataset.bond_dataset import BondPredictionDataset
from neat.model import BondPredictor

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch_geometric.seed_everything(42)
seed_everything(42)

ROOT = os.getcwd()

def bond_collate_fn(batch: list) -> Batch:
    """Collate function that batches Data objects and concatenates pair_labels."""
    valid = [d for d in batch if hasattr(d, "pair_labels")]
    if len(valid) == 0:
        raise RuntimeError("Empty batch after filtering; check add_bond_labels function.")
    return Batch.from_data_list(valid)


def get_datasets(dataset_name: str):
    """Load train/val/test for QM9 or GEOM and wrap with BondPredictionDataset."""
    data_dir = os.path.join(ROOT, "data")
    data_path = os.path.join(data_dir, dataset_name.upper())

    if dataset_name.upper() == "QM9":
        full_data = QM9DataSet(data_path)
        splits = full_data.get_splits()
        train_data = BondPredictionDataset(full_data[splits["train"]])
        val_data = BondPredictionDataset(full_data[splits["val"]])
        test_data = BondPredictionDataset(full_data[splits["test"]])
        vocab_size = len(QM9DataSet.VOCABULARY) + 1
    elif dataset_name.upper() == "GEOM":
        train_data = BondPredictionDataset(GEOMDataSet(data_path, split="train"))
        val_data = BondPredictionDataset(GEOMDataSet(data_path, split="val"))
        test_data = BondPredictionDataset(GEOMDataSet(data_path, split="test"))
        vocab_size = len(GEOMDataSet.VOCABULARY) + 1
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_data, val_data, test_data, vocab_size


def train(args: argparse.Namespace) -> None:
    """Train bond predictor models on QM9 and GEOM datasets.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        None
    """
    config_path = args.config_file or os.path.join(ROOT, "scripts", "config_bond_predictor.yaml")
    print(f"Using config: {config_path}")

    params = yaml.load(
        open(config_path, "r"),
        Loader=yaml.FullLoader,
    )

    dataset_name = params.get("data_set", "QM9")
    print(f"Loading {dataset_name}...")
    train_data, val_data, test_data, vocab_size = get_datasets(dataset_name)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    train_loader = DataLoader(
        train_data,
        batch_size=params.get("batch_size", 64),
        shuffle=True,
        drop_last=True,
        collate_fn=bond_collate_fn,
        persistent_workers=True,
        num_workers=8,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=params.get("batch_size", 64),
        shuffle=False,
        drop_last=False,
        collate_fn=bond_collate_fn,
        persistent_workers=True,
        num_workers=8,
    )

    model_params = {
        "vocab_size": vocab_size,
        "n_embd": params.get("n_embd", 256),
        "n_conv_layers": params.get("n_conv_layers", 4),
        "dropout": params.get("dropout", 0.1),
        "learning_rate": params.get("learning_rate", 1e-3),
        "weight_decay": params.get("weight_decay", 1e-6),
        "max_epochs": params.get("max_epochs", 100),
        "lr_warmup_epochs": params.get("lr_warmup_epochs", 5),
        "lr_min_ratio": params.get("lr_min_ratio", 0.1),
    }

    model = BondPredictor(**model_params)

    log_dir = os.path.join(ROOT, "logs", "BondPredictor")
    tb_logger = TensorBoardLogger(log_dir, name=dataset_name, default_hp_metric=False)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        filename=f"bond_predictor_{dataset_name}-{{epoch:02d}}",
        save_top_k=1,
        every_n_epochs=1,
    )

    early_stopping = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=params.get("early_stopping_patience", 10),
        min_delta=params.get("early_stopping_min_delta", 1e-3),
    )

    callbacks = [
        checkpoint_callback,
        early_stopping,
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = Trainer(
        devices=[0] if torch.cuda.is_available() else "auto",
        max_epochs=model_params["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=tb_logger,
        log_every_n_steps=10,
        callbacks=callbacks,
        gradient_clip_val=1.0,
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", 
        dest="config_file", 
        required=False, 
        metavar="<file>", 
        help="Config file for training.",
    )

    args = parser.parse_args()

    train(args)
