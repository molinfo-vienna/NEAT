"""Bond predictor GNN: given atom types and coordinates, predict bond types for edges."""

import math

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, radius_graph

from .positional_encoding import FourierPositionEncoding

# Bond types: 0=no bond, 1=single, 2=double, 3=triple, 4=aromatic
NUM_BOND_TYPES = 5


class BondPredictor(LightningModule):
    """GNN to predict bond types for edges in a molecular graph.

    Expects data with edge_index (radius graph) and edge_labels (precomputed).
    Predicts one of 5 bond types per edge: no bond, single, double, triple, aromatic.
    """

    def __init__(self, **params) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.hparams.setdefault("radius", 2.5)  # for inference when building radius graph

        n_embd = self.hparams.n_embd
        n_conv_layers = self.hparams.n_conv_layers

        self.atom_type_embedding = nn.Embedding(
            num_embeddings=self.hparams.vocab_size, embedding_dim=n_embd
        )
        self.fourier_encoding_layer = FourierPositionEncoding(out_dim=n_embd)

        self.conv_layers = nn.ModuleList()
        for _ in range(n_conv_layers):
            nn_module = nn.Sequential(
                nn.Linear(n_embd, n_embd * 2),
                nn.ReLU(),
                nn.Dropout(self.hparams.dropout),
                nn.Linear(n_embd * 2, n_embd),
            )
            self.conv_layers.append(GINConv(nn=nn_module, eps=0.0, train_eps=True))
        self.layer_norm = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(self.hparams.dropout)

        self.bond_mlp = nn.Sequential(
            nn.Linear(n_embd * 2, n_embd),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, NUM_BOND_TYPES),
        )

    def forward(self, data: Data) -> Tensor:
        """Forward pass.

        Args:
            data: PyG Batch with x, pos, batch, edge_index, edge_labels.
                  edge_index and edge_labels are precomputed by the datamodule.

        Returns:
            bond_logits: [num_edges, 5] logits for each edge.
        """
        # (1) Node embeddings
        atom_type_emb = self.atom_type_embedding(data.x)
        pos_emb = self.fourier_encoding_layer(data.pos)
        x = atom_type_emb + pos_emb
        x = self.dropout(x)

        # Step 2: Message passing
        for conv in self.conv_layers:
            x = conv(x, data.edge_index) + x
            x = F.relu(x)
        x = self.layer_norm(x)

        # Step 3: Bond prediction per edge
        src, dst = data.edge_index[0], data.edge_index[1]
        h_src = x[src]  # [num_edges, n_embd]
        h_dst = x[dst]
        edge_features = torch.cat([h_src, h_dst], dim=-1)  # [num_edges, n_embd*2]
        bond_logits = self.bond_mlp(edge_features)  # [num_edges, 5]

        return bond_logits

    @torch.no_grad()
    def predict_bonds(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor | None = None,
        device: torch.device | None = None,
        radius: float | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Predict bond types for inference. Builds radius graph from pos/batch.

        Returns:
            bond_types: [num_edges] predicted class (0-4) per edge.
            pair_indices: [num_edges, 2] (src, dst) for each edge.
        """
        if device is None:
            device = x.device
        radius = radius or getattr(self.hparams, "radius", 2.5)

        data = Data(x=x, pos=pos)
        if batch is not None:
            data.batch = batch
        else:
            data.batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)
        data = data.to(device)

        # Build radius graph (same as datamodule's bond_prediction_batch_transform)
        data.edge_index = radius_graph(data.pos, r=radius, batch=data.batch, loop=False)

        logits = self(data)
        bond_types = logits.argmax(dim=1)
        pair_indices = data.edge_index.t()  # [num_edges, 2]

        return bond_types, pair_indices

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        bond_logits = self(batch)
        labels = batch.edge_labels
        loss = F.cross_entropy(bond_logits, labels.long(), reduction="mean")
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> Tensor:
        bond_logits = self(batch)
        labels = batch.edge_labels
        loss = F.cross_entropy(bond_logits, labels.long(), reduction="mean")
        pred = bond_logits.argmax(dim=1)
        acc = (pred == labels).float().mean()
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """AdamW; cosine LR schedule with warmup."""
        decay_params = [p for n, p in self.named_parameters() if p.dim() >= 2]
        no_decay_params = [p for n, p in self.named_parameters() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.hparams.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        optimizer = AdamW(optim_groups, lr=self.hparams.learning_rate)

        def lr_lambda(epoch):
            # Linear warmup, then cosine decay down to lr_min_ratio * base_lr
            if epoch < self.hparams.lr_warmup_epochs:
                return (epoch + 1) / (self.hparams.lr_warmup_epochs + 1)
            progress = (epoch - self.hparams.lr_warmup_epochs) / (
                self.hparams.max_epochs - self.hparams.lr_warmup_epochs
            )
            progress = min(progress, 1.0)
            return self.hparams.lr_min_ratio + (1 - self.hparams.lr_min_ratio) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [scheduler]
