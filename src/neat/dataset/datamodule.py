import functools
import os

import torch
from lightning import LightningDataModule
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from .augmentation import RandomRotationAugmentation
from .dataset_qm9 import QM9DataSet
from .dataset_geom import GEOMDataSet
from .splitting import SourceTargetSplitter


def batch_transform(batch, source_target_split, noise_std):
    rotation_augmentation = RandomRotationAugmentation()
    batch.pos = rotation_augmentation.rotate_graphs_randomly(batch.pos, batch.batch)
    splitter = SourceTargetSplitter(splitting_mode=source_target_split)

    (
        x_source,  # [n_source_atoms]
        pos_source,  # [n_source_atoms, 3]
        batch_source,  # [n_source_atoms]
        x_target,  # [n_target_atoms]
        pos_target,  # [n_target_atoms, 3]
        batch_target,  # [n_target_atoms]
        stop_tokens,  # [n_molecules]
        source_ptr,
        target_ptr,
    ) = splitter.create_source_target_split(batch)

    batch.stop_tokens = stop_tokens
    batch.source_ptr = source_ptr
    batch.target_ptr = target_ptr

    batch_target = batch_target.long()
    pos_random = noise_std * torch.randn_like(pos_target)
    target_idx = torch.unique(batch_target)
    for idx in target_idx:
        cost_matrix = torch.cdist(
            pos_target[batch_target == idx], pos_random[batch_target == idx], p=2
        )
        _, prior_idx = linear_sum_assignment(cost_matrix.cpu())

        # Reorder prior according to optimal assignment
        pos_random[batch_target == idx] = pos_random[batch_target == idx][prior_idx]
    batch.pos_random = pos_random

    return batch


def custom_collate_fn(batch, source_target_split, noise_std):
    batch = Batch.from_data_list(batch)
    return batch_transform(batch, source_target_split, noise_std)


class DataModule(LightningDataModule):
    """
    DataModule for loading and transforming the data for the MolGen model.

    Args:
        training_data_dir (str): Directory containing the training data.
        batch_size (int): Batch size for the data loader.
        source_target_split (str): Source-target split mode.
        num_workers (int): Number of workers for the data loader.

    Returns:
        DataModule for loading and transforming the data for the MolGen model.
    """

    def __init__(
        self,
        data_dir: str,
        data_set: str = "QM9",
        batch_size: int = 32,
        source_target_split: str = "cyclic",
        noise_std: float = 1.4,
        num_workers: int = 1,
    ) -> None:
        super(DataModule, self).__init__()
        self.data_path = os.path.join(data_dir, data_set.upper())
        self.data_set = data_set.upper()
        self.batch_size = batch_size
        self.source_target_split = source_target_split
        self.noise_std = noise_std
        self.num_workers = num_workers

        self.source_target_split_fn = functools.partial(
            custom_collate_fn,
            source_target_split=self.source_target_split,
            noise_std=self.noise_std,
        )

    def setup(self, stage: str = "fit") -> None:
        if stage == "fit":
            if self.data_set == "QM9":
                self.full_data = QM9DataSet(self.data_path)
                splits = self.full_data.get_splits()
                self.training_data = self.full_data[splits["train"]]
                self.validation_data = self.full_data[splits["val"]]
                self.test_data = self.full_data[splits["test"]]
                print(f"Number of training graphs: {len(self.training_data)}")
                print(f"Number of validation graphs: {len(self.validation_data)}")
                print(f"Number of test graphs: {len(self.test_data)}")
            elif self.data_set == "GEOM":
                print("Using GEOM dataset.")
                self.training_data = GEOMDataSet(self.data_path, split="train")
                self.validation_data = GEOMDataSet(self.data_path, split="val")
                self.test_data = GEOMDataSet(self.data_path, split="test")

                print(f"Number of training graphs: {len(self.training_data)}")
                print(f"Number of validation graphs: {len(self.validation_data)}")
                print(f"Number of test graphs: {len(self.test_data)}")

            else:
                raise ValueError(f"Unknown data_set: {self.data_set}")

    def train_dataloader(self, shuffle_data=True) -> DataLoader:
        return DataLoader(
            self.training_data,
            batch_size=self.batch_size,
            shuffle=shuffle_data,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=self.source_target_split_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=True,
            collate_fn=self.source_target_split_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=self.source_target_split_fn,
        )

    def full_dataloader(self) -> DataLoader:
        return DataLoader(
            self.full_data,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=self.source_target_split_fn,
        )
