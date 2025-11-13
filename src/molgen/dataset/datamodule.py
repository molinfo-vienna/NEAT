import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from .dataset import DataSet
from ..model.splitting import SourceTargetSplitter
from ..model.augmentation import RandomRotationAugmentation


def batch_transform(batch):
    # Apply your batch-level augmentation logic here
    # For example, add random noise to all node features in the batch
    # data augmentation by random rotation
    rotation_augmentation = RandomRotationAugmentation()
    batch.pos = rotation_augmentation.rotate_graphs_randomly(batch.pos, batch.batch)
    splitter = SourceTargetSplitter(splitting_mode="cyclic")

    (
        x_source,  # [n_source_atoms]
        pos_source,  # [n_source_atoms, 3]
        batch_source,  # [n_source_atoms]
        atom_count_source,
        x_target,  # [n_target_atoms]
        pos_target,  # [n_target_atoms, 3]
        batch_target,  # [n_target_atoms]
        stop_tokens,  # [n_molecules]
    ) = splitter.create_source_target_split(batch)

    batch.x_source = x_source
    batch.pos_source = pos_source
    batch.batch_source = batch_source
    batch.atom_count_source = atom_count_source
    batch.x_target = x_target
    batch.pos_target = pos_target
    batch.batch_target = batch_target
    batch.stop_tokens = stop_tokens

    return batch


# Create a custom DataLoader with a batch-level transform
def custom_collate_fn(batch):
    batch = Batch.from_data_list(batch)  # Collate individual samples into a batch
    return batch_transform(batch)  # Apply the batch-level transform


class DataModule(LightningDataModule):
    def __init__(
        self,
        training_data_dir: str,
        batch_size: int = None,
        split: str = "random",
        num_workers: int = 4,
    ) -> None:
        super(DataModule, self).__init__()
        self.training_data_dir = training_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split

    def setup(self, stage: str = "fit") -> None:
        if stage == "fit":
            self.full_data = DataSet(self.training_data_dir, transform=None)
            if self.split == "random":
                seed = 42  # or any fixed number
                generator = torch.Generator().manual_seed(seed)
                perm = torch.randperm(len(self.full_data), generator=generator)
                self.full_data = self.full_data[perm]

                self.training_data = self.full_data[: int(0.8 * len(self.full_data))]
                self.validation_data = self.full_data[
                    int(0.8 * len(self.full_data)) : int(0.9 * len(self.full_data))
                ]
                self.test_data = self.full_data[int(0.9 * len(self.full_data)) :]

                print(f"Number of training graphs: {len(self.training_data)}")
                print(f"Number of validation graphs: {len(self.validation_data)}")
                print(f"Number of test graphs: {len(self.test_data)}")

            elif self.split == "edm_split":
                splits = self.full_data.get_qm9_splits(edm_splits=True)
                print("Using predefined EDM splits.")
                self.training_data = self.full_data[splits["train"]]
                self.validation_data = self.full_data[splits["val"]]
                self.test_data = self.full_data[splits["test"]]

                print(f"Number of training graphs: {len(self.training_data)}")
                print(f"Number of validation graphs: {len(self.validation_data)}")
                print(f"Number of test graphs: {len(self.test_data)}")
            else:
                raise ValueError(f"Unknown data split: {self.split}")

    def train_dataloader(self, shuffle_data=True) -> DataLoader:
        if self.batch_size is None:
            return DataLoader(
                self.training_data,
                batch_size=len(self.training_data),
                shuffle=shuffle_data,
                drop_last=True,
                num_workers=self.num_workers,
                persistent_workers=True,
                collate_fn=custom_collate_fn,
            )
        else:
            return DataLoader(
                self.training_data,
                batch_size=self.batch_size,
                shuffle=shuffle_data,
                drop_last=True,
                num_workers=self.num_workers,
                persistent_workers=True,
                collate_fn=custom_collate_fn,
            )

    def val_dataloader(self) -> DataLoader:
        if self.batch_size is None:
            return DataLoader(
                self.validation_data,
                batch_size=len(self.validation_data),
                shuffle=False,
                drop_last=True,
                num_workers=self.num_workers,
                persistent_workers=True,
                collate_fn=custom_collate_fn,
            )
        else:
            return DataLoader(
                self.validation_data,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=True,
                drop_last=True,
                collate_fn=custom_collate_fn,
            )

    def test_dataloader(self) -> DataLoader:
        if self.batch_size is None:
            return DataLoader(
                self.test_data,
                batch_size=len(self.test_data),
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
                persistent_workers=True,
                collate_fn=custom_collate_fn,
            )
        else:
            return DataLoader(
                self.test_data,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
                persistent_workers=True,
                collate_fn=custom_collate_fn,
            )

    def full_dataloader(self) -> DataLoader:
        if self.batch_size is None:
            return DataLoader(
                self.full_data,
                batch_size=len(self.full_data),
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
                persistent_workers=True,
                collate_fn=custom_collate_fn,
            )
        else:
            return DataLoader(
                self.full_data,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
                persistent_workers=True,
                collate_fn=custom_collate_fn,
            )
