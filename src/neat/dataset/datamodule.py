import functools
import os

import torch
from lightning import LightningDataModule
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from .augmentation import RandomRotationAugmentation
from .dataset_geom import GEOMDataSet
from .dataset_qm9 import QM9DataSet
from .splitting import SourceTargetSplitter


def batch_transform(batch: Batch, source_target_split: str, noise_std: float) -> Batch:
    """Transform a batch of graphs by:

        1.applying random rotation augmentation,
        2. creating source-target splits,
        3. initializing stop tokens, and
        4. coupling positions in the target set with random positions.

    Args:
        batch (Batch): Batch of graphs.
        source_target_split (str): Source-target split mode.
        noise_std (float): Standard deviation of the initial Gaussian noise in the flow matching process.

    Returns:
        Batch: Transformed batch of graphs.
    """
    # (1) Apply random rotation augmentation
    rotation_augmentation = RandomRotationAugmentation()
    batch.pos = rotation_augmentation.rotate_graphs_randomly(batch.pos, batch.batch)

    # (2) Create source-target split
    splitter = SourceTargetSplitter(splitting_mode=source_target_split)

    source_ptr, target_ptr = splitter.create_source_target_split(batch)
    batch.source_ptr = source_ptr
    batch.target_ptr = target_ptr

    # (3) Determine source sets with empty target sets, these have stop tokens
    target_set_mask = torch.zeros_like(
        batch.batch, device=batch.batch.device, dtype=torch.bool
    )
    target_set_mask[target_ptr] = 1
    batch_target = batch.batch[target_set_mask]
    stop_tokens = ~(torch.isin(torch.arange(0, len(batch)), torch.unique(batch_target)))
    batch.stop_tokens = stop_tokens

    # (4) Couple positions in the target set with random positions via linear sum assignment
    pos_target = batch.pos[target_set_mask]
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


def custom_collate_fn(batch: list, source_target_split: str, noise_std: float) -> Batch:
    batch = Batch.from_data_list(batch)
    return batch_transform(batch, source_target_split, noise_std)


class DataModule(LightningDataModule):
    """DataModule for loading and transforming the data for the NEAT model.

    Args:
        training_data_dir (str): Directory containing the training data.
        data_set (str): Dataset to use ("QM9" or "GEOM").
        batch_size (int): Batch size for the data loader.
        source_target_split (str): Source-target split mode.
        noise_std (float): Standard deviation of the initial Gaussian noise in the flow matching process
        num_workers (int): Number of workers for the data loader.

    Returns:
        DataModule for loading and transforming the data for the NEAT model.
    """

    def __init__(
        self,
        data_dir: str,
        data_set: str = "QM9",
        batch_size: int = 32,
        source_target_split: str = "neighborhood",
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
        if self.data_set == "QM9":
            self.vocab_size = len(QM9DataSet.VOCABULARY) + 1
            self.vocab = QM9DataSet.VOCABULARY
        elif self.data_set == "GEOM":
            self.vocab_size = len(GEOMDataSet.VOCABULARY) + 1
            self.vocab = GEOMDataSet.VOCABULARY
        else:
            raise ValueError(f"Unknown data_set: {self.data_set}")

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
