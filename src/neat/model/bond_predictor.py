"""Bond predictor GNN: given atom types and coordinates, predict bond types between atom pairs.

High-level logic:
1. Each atom is represented by: atom_type_emb + fourier(pos). This encodes both chemical
   identity and 3D coordinates.
2. We build a fully connected graph within each molecule and perform message passing via GINConv layers.
3. For every pair (i, j) with i < j in each molecule, we concatenate their node embeddings
   [h_i; h_j] and predict one of 5 bond types: no bond, single, double, triple, aromatic.
"""

import math

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW
from torch_geometric.data import Data
from torch_geometric.nn import GINConv

from .positional_encoding import FourierPositionEncoding

# Bond types: 0=no bond, 1=single, 2=double, 3=triple, 4=aromatic
NUM_BOND_TYPES = 5


class BondPredictor(LightningModule):
    """GNN to predict bond types between atoms in a point cloud.

    Given atom types (x) and coordinates (pos), predicts for each atom pair (i,j)
    the bond type: no bond (0), single (1), double (2), triple (3), or aromatic (4).
    """

    def __init__(self, **params) -> None:
        super().__init__()
        self.save_hyperparameters()

        n_embd = self.hparams.n_embd
        n_conv_layers = self.hparams.n_conv_layers

        # Atom type embedding: map discrete atom type (1=C, 2=N, ...) to a dense vector
        self.atom_type_embedding = nn.Embedding(
            num_embeddings=self.hparams.vocab_size, embedding_dim=n_embd
        )

        # Fourier encoding: map (x,y,z) coordinates to a vector. Encodes absolute position;
        # relative geometry between two atoms is implicit when we concatenate their encodings.
        self.fourier_encoding_layer = FourierPositionEncoding(out_dim=n_embd)

        # GINConv layers: message passing over the fully connected graph. 
        # Each layer aggregates neighbor features and combines them with the node's own features.
        # Residual + ReLU after each layer.
        self.conv_layers = nn.ModuleList()
        for _ in range(n_conv_layers):
            nn_module = nn.Sequential(
                nn.Linear(n_embd, n_embd * 2),
                nn.ReLU(),
                nn.Dropout(self.hparams.dropout),
                nn.Linear(n_embd * 2, n_embd),
            )
            self.conv_layers.append(GINConv(nn=nn_module, eps=0.0, train_eps=True))
        # Layer normalization and dropout after the GINConv layers
        self.layer_norm = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(self.hparams.dropout)

        # Bond prediction head: for a pair (i,j), input is [h_i; h_j], output is 5-way logits
        self.bond_mlp = nn.Sequential(
            nn.Linear(n_embd * 2, n_embd),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, NUM_BOND_TYPES),
        )

    def forward(self, data: Data) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            data: PyG Data or Batch with x (atom types), pos (coordinates), batch (molecule id per atom).

        Returns:
            bond_logits: [num_pairs, 5] logits for each pair (i<j).
            pair_batch: [num_pairs] molecule id for each pair (for loss aggregation).
        """
        device = data.x.device
        x_atom = data.x  # [n_atoms]
        pos = data.pos  # [n_atoms, 3]
        batch_idx = data.batch  # [n_atoms]

        # ---- Step 1: Initial node embeddings ----
        # Combine atom type and position. Each atom gets a vector that encodes both
        # what element it is and where it sits in 3D space.
        atom_type_emb = self.atom_type_embedding(x_atom)  # [n_atoms, n_embd]
        pos_emb = self.fourier_encoding_layer(pos)  # [n_atoms, n_embd]
        x = atom_type_emb + pos_emb  # [n_atoms, n_embd]

        # ---- Step 2: Build fully connected graph (within each molecule) ----
        # edge_index: [2, num_edges], each column is (begin, end).
        # We connect every atom to every other atom within the same molecule only.
        # This lets GIN aggregate information from all atoms in the molecule.
        row, col = [], []
        for b in batch_idx.unique():
            mask = batch_idx == b
            indices = torch.where(mask)[0]
            n = indices.shape[0]  
            for i in range(n):
                for j in range(n):
                    if i != j:
                        row.append(indices[i].item())
                        col.append(indices[j].item())

        edge_index = torch.tensor([row, col], dtype=torch.long, device=device)

        # ---- Step 3: Graph convolutions (message passing) ----
        for conv in self.conv_layers:
            x = conv(x, edge_index) + x
            x = F.relu(x)
        x = self.layer_norm(x)

        # ---- Step 4: Collect all pairs (i, j) with i < j ----
        # We only need one ordering (i<j) since bonds are symmetric.
        # Pairs are grouped by molecule: mol0 pairs, then mol1 pairs, ...
        pair_rows, pair_cols = [], []
        pair_batch_list = []
        for b in batch_idx.unique():
            mask = batch_idx == b
            indices = torch.where(mask)[0]
            n = indices.shape[0]
            for i in range(n):
                for j in range(i + 1, n):  # i < j only
                    pair_rows.append(indices[i].item())
                    pair_cols.append(indices[j].item())
                    pair_batch_list.append(b.item())

        pair_i = torch.tensor(pair_rows, device=device)
        pair_j = torch.tensor(pair_cols, device=device)
        pair_batch = torch.tensor(pair_batch_list, device=device)

        # ---- Step 5: Bond prediction ----
        # For each pair, concatenate embeddings and predict bond type.
        # h_i and h_j already encode geometry (from Fourier) and context (from GIN).
        h_i = x[pair_i]  # [num_pairs, n_embd]
        h_j = x[pair_j]  # [num_pairs, n_embd]
        pair_features = torch.cat([h_i, h_j], dim=-1)  # [num_pairs, n_embd*2]
        bond_logits = self.bond_mlp(pair_features)  # [num_pairs, 5]

        return bond_logits, pair_batch

    @torch.no_grad()
    def predict_bonds(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor | None = None,
        device: torch.device | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Predict bond types for atom pairs. For inference use.

        Wraps forward() and returns argmax predictions plus pair indices so the caller
        can map each prediction back to the corresponding (i, j) atom pair.

        Returns:
            bond_types: [num_pairs] predicted class (0-4) per pair.
            pair_indices: [num_pairs, 2] with (i, j) for each pair, same order as bond_types.
        """
        if device is None:
            device = x.device
        data = Data(x=x, pos=pos)
        if batch is not None:
            data.batch = batch
        data = data.to(device)
        logits, pair_batch = self(data)
        bond_types = logits.argmax(dim=1)

        # Rebuild (i, j) indices in the same order as bond_types (mol0 pairs, then mol1, ...)
        batch_idx = getattr(data, "batch", torch.zeros(x.shape[0], dtype=torch.long, device=device))
        pair_rows, pair_cols = [], []
        for b in batch_idx.unique():
            mask = batch_idx == b
            indices = torch.where(mask)[0]
            n = indices.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    pair_rows.append(indices[i].item())
                    pair_cols.append(indices[j].item())
        pair_indices = torch.stack(
            [torch.tensor(pair_rows, device=device), torch.tensor(pair_cols, device=device)],
            dim=1,
        )
        return bond_types, pair_indices

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        """Cross-entropy loss on bond type predictions."""
        bond_logits, _ = self(batch)
        labels = batch.pair_labels  # [num_pairs], ground-truth bond type per pair of atoms (i, j) with i < j
        loss = F.cross_entropy(bond_logits, labels.long(), reduction="mean")
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> Tensor:
        """Same as training_step; also log accuracy (fraction of pairs predicted correctly)."""
        bond_logits, _ = self(batch)
        labels = batch.pair_labels
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
