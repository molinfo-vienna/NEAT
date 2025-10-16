import torch
from lightning import LightningModule
from torch_geometric.data import Data


class MolGen(LightningModule):
    def __init__(self):
        super(MolGen, self).__init__()
        self.save_hyperparameters()
        # self.hparams.setdefault("key", "value")

        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, data: Data) -> Data:
        # Define the forward pass
        return x

    def training_step(self, batch, batch_idx):
        # Define a single training step
        loss = torch.nn.functional.mse_loss(batch, batch)  # Dummy loss
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        # Define optimizers and learning-rate schedulers
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
