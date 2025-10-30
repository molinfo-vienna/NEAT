import torch
from lightning import LightningDataModule
from torch_geometric.loader import DataLoader

from .dataset import DataSet


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
            )
        else:
            return DataLoader(
                self.training_data,
                batch_size=self.batch_size,
                shuffle=shuffle_data,
                drop_last=True,
                num_workers=self.num_workers,
                persistent_workers=True,
            )

    def val_dataloader(self) -> DataLoader:
        if self.batch_size is None:
            return DataLoader(
                self.validation_data,
                batch_size=len(self.validation_data),
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
                persistent_workers=True,
            )
        else:
            return DataLoader(
                self.validation_data,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
                persistent_workers=True,
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
            )
        else:
            return DataLoader(
                self.test_data,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
                persistent_workers=True,
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
            )
        else:
            return DataLoader(
                self.full_data,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
                persistent_workers=True,
            )
