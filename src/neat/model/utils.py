import os

import torch
from lightning import LightningModule


def load_model_from_path(
    folder_path: str, model_class: LightningModule
) -> LightningModule:
    """Load the model from the checkpoint path."""
    if not os.path.exists(folder_path):
        return None
    else:
        folder_path = os.path.join(folder_path, "checkpoints")

    model_path = None
    for file in os.listdir(folder_path):
        if file.endswith(".ckpt"):
            model_path = os.path.join(folder_path, file)

    if model_path:
        return model_class.load_from_checkpoint(
            model_path, map_location=torch.device("cpu")
        )
    else:
        return None
