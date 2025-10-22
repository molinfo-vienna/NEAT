from lightning import LightningModule, Callback, Trainer
import torch
from torch_geometric import transforms as T


class CurriculumLearningScheduler(Callback):
    def __init__(
        self,
        set_size_at_start: int = 1,
        num_epochs_before_increase: int = 5,
        minimum_improvement_threshold: float = 1.05,
    ) -> None:
        super().__init__()
        self.set_size_at_start = set_size_at_start
        self.num_epochs_before_increase = num_epochs_before_increase
        self.minimum_improvement_threshold = minimum_improvement_threshold
        self.loss_at_reference_point = torch.inf
        self.counter = 0

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module.splitter.target_set_max_size = self.set_size_at_start

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:

        pl_module.log(
            "target_set_max_size",
            pl_module.splitter.target_set_max_size,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )

        if trainer.current_epoch == 0:
            return

        current_loss = trainer.logged_metrics["val/val_loss"]

        if (
            current_loss * self.minimum_improvement_threshold
        ) < self.loss_at_reference_point:
            self.loss_at_reference_point = current_loss
            self.counter = 0
        elif self.counter < self.num_epochs_before_increase:
            self.counter += 1
        else:
            pl_module.splitter.target_set_max_size += 1
            self.loss_at_reference_point = torch.inf
            self.counter = 0
            print(
                f"Graph size increased to {pl_module.splitter.target_set_max_size} at epoch {trainer.current_epoch}"
            )
