import torch
from lightning import LightningModule
from torch_geometric.data import Data

from .decoder import Decoder


class MolGen(LightningModule):
    def __init__(self):
        super(MolGen, self).__init__()
        self.save_hyperparameters()
        # self.hparams.setdefault("key", "value")

        # Define model layers here
        self.decoder1 = torch.nn.Identity()
        self.decoder2 = torch.nn.Identity()

        # Define loss functions here
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.fm_loss = FlowMatchingLoss()

    def forward(self, data: Data) -> Data:
        token_prob = self.decoder1(data)
        token = torch.argmax(token_prob, dim=-1)
        data.x.append(token)
        condition = self.decoder2(data)
        position = self.fm_loss(condition, data)
        data.pos.append(position)
        return data

    def data_augmentation(self, data: Data) -> Data:
        # Implement data augmentation logic here
        return data

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict]]:
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        return [optimizer]

    def shared_step(self, batch: Data, batch_idx: int) -> Tensor:
        y_hat = self(batch)
        y = batch.y

        return self.loss(y_hat, y)

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        """Training step and logging"""
        batch = self.flip_sign_and_voxel(batch)
        loss = self.shared_step(batch, batch_idx)

        self.log(
            "train/train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=len(batch),
            reduce_fx="mean",
        )

        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> Tensor:
        loss = self.shared_step(batch, batch_idx)

        self.log(
            "val/val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )

        return loss

    def predict_step(
        self, batch: Data, batch_idx: int = 0
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        return self(batch)
