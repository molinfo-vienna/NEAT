import torch
from lightning import Callback, LightningModule, Trainer
from rdkit import Chem

from molgen.model.molecule_builder import MoleculeBuilder


class GenerationMonitor(Callback):
    def __init__(self, num_samples: int = 100, every_n_epochs: int = 50) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if (
            trainer.current_epoch % self.every_n_epochs != 0
            or trainer.current_epoch == 0
        ):
            return
        x, pos, batch = pl_module.generate(batch_size=self.num_samples)
        builder = MoleculeBuilder()
        mols = builder.generate_rdkit_molecules(x, pos, batch)
        n_valid = self.compute_validity(mols)
        n_unique = self.compute_uniqueness(mols)
        frac_valid = n_valid / self.num_samples
        frac_unique = n_unique / n_valid if n_valid > 0 else 0.0

        pl_module.log(
            "val/validity",
            frac_valid,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        pl_module.log(
            "val/uniqueness",
            frac_unique,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def compute_validity(self, mols):
        num_valid = 0
        for mol in mols:
            if mol is not None:
                num_valid += 1
        return num_valid

    def compute_uniqueness(self, mols):
        unique_smiles = set()
        for mol in mols:
            if mol is not None:
                smiles = Chem.MolToSmiles(mol, canonical=True)
                unique_smiles.add(smiles)
        return len(unique_smiles)


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
