from typing import Union
import inspect
import math
from abc import ABC

import torch
from torch import Tensor
from lightning import LightningModule
from torch_geometric.data import Data
from torch.optim import Optimizer, AdamW
from torch.nn import functional as F
from torch_geometric.nn.models import MLP as GeoMLP

from .decoder import Decoder
from .modules import (
    LayerNorm,
    MLP,
    Block,
    SinusoidalPositionalEncoding,
    pad_and_mask_sequences,
    create_time_embeddings,
    rotate_graphs_randomly,
)


class MolGen(LightningModule):
    def __init__(self, **params) -> None:
        super(MolGen, self).__init__()
        self.save_hyperparameters()
        # self.hparams.setdefault("key", "value")

        # Atom type embedding layer
        self.atom_type_embedding = torch.nn.Embedding(
            num_embeddings=self.hparams.vocab_size, embedding_dim=self.hparams.n_embd
        )

        # Fourier features for embedding of Cartesian coordinates
        self.cartesian_positional_embedding = SinusoidalPositionalEncoding(
            out_dim=self.hparams.n_embd
        )

        # A linear layer for projecting the Cartesian coordinates (additional to the Fourier features) --> Probably not necessary
        self.coord_proj = torch.nn.Identity()

        # Positional embedding layer for sequences (only important for causal transformer) --> Probably not necessary
        self.sequential_positional_embedding = torch.nn.Embedding(
            self.hparams.block_size, self.hparams.n_embd
        )

        # Dropout layer
        self.dropout_layer = torch.nn.Dropout(self.hparams.dropout)

        # Transformer blocks
        self.transformer_blocks = torch.nn.ModuleList(
            [
                Block(
                    self.hparams.n_embd,
                    self.hparams.n_head,
                    self.hparams.dropout,
                    self.hparams.bias,
                )
                for _ in range(self.hparams.n_layer)
            ]
        )
        self.output_layer_norm = LayerNorm(self.hparams.n_embd, bias=self.hparams.bias)

        # Linear layer to map the final embeddings to atom vocabulary logits
        self.linear_output_head = torch.nn.Linear(
            self.hparams.n_embd, self.hparams.vocab_size, bias=False
        )

        # The atom types are supervised with a cross-entropy loss
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

        # Weight tying
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.atom_type_embedding.weight = self.linear_output_head.weight

        # So far it is the GPT logic, here comes additional stuff

        # # A second transformer block
        # self.transformer_block_2 = torch.nn.ModuleList(
        #     [
        #         Block(
        #             self.hparams.n_embd,
        #             self.hparams.n_head,
        #             self.hparams.dropout,
        #             self.hparams.bias,
        #         )
        #         for _ in range(self.hparams.n_layer)
        #     ]
        # )

        # self.output_layer_norm_2 = LayerNorm(
        #     self.hparams.n_embd, bias=self.hparams.bias
        # )

        # Positional embedding for flow matching
        self.cartesian_positional_embedding_fm = SinusoidalPositionalEncoding(
            out_dim=self.hparams.n_embd
        )

        # A simple MLP with layer norm used for the denoising step
        self.flow_matching_mlp = GeoMLP(
            channel_list=[self.hparams.n_embd, self.hparams.n_embd, 3],
            dropout=self.hparams.dropout,
            bias=self.hparams.bias,
        )

        # Define loss functions here
        # self.fm_loss = FlowMatchingLoss()

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.hparams.n_layer)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.sequential_positional_embedding.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, data, targets=None):
        device = data.x.device
        atom_counts = torch.bincount(data.batch)
        batch_size = atom_counts.size(0)

        sequence_length = atom_counts.max()
        assert (
            sequence_length <= self.hparams.block_size
        ), f"Cannot forward sequence of length {sequence_length}, block size is only {self.hparams.block_size}"

        # Randomly select a subset of atoms per molecule
        uniform_distribution = torch.rand(atom_counts.shape, device=device) * 0.999
        deletion_limit = torch.ones_like(atom_counts, device=device)
        atoms_to_delete = ((deletion_limit.float() + 1) * uniform_distribution).int()
        # atoms_to_delete = ((atom_counts.float() + 1) * uniform_distribution).int()
        atoms_to_keep = atom_counts - atoms_to_delete
        random_indices = torch.cat(
            [
                (torch.randperm(i, device=device) + k)
                for i, k in zip(atom_counts, data.ptr[0:-1])
            ]
        )
        subset_idx = torch.cat(
            [random_indices[j : j + k] for j, k in zip(data.ptr[0:-1], atoms_to_keep)]
        )

        target_idx = torch.tensor(
            [
                random_indices[j + k].item() if i > 0 else -1
                for i, j, k in zip(atoms_to_delete, data.ptr[0:-1], atoms_to_keep)
            ],
            device=device,
            dtype=torch.long,
        )
        target_types = data.x[target_idx]
        target_types[target_idx == -1] = (
            0  # Stop token for molecules without deleted nodes
        )
        target_pos = data.pos[target_idx]
        target_pos[target_idx == -1] = 0.0

        atom_mask = torch.zeros_like(data.batch, device=device, dtype=torch.bool)
        atom_mask[subset_idx] = 1
        data.x = data.x[atom_mask]
        data.pos = data.pos[atom_mask]
        data.batch = data.batch[atom_mask]

        # forward the GPT model itself
        atom_type_embeddings = self.atom_type_embedding(
            data.x
        )  # token embeddings of shape (b, t, n_embd)
        cartesian_positional_embeddings = self.cartesian_positional_embedding(
            data.pos
        )  # position embeddings of shape (t, n_embd)
        x = self.dropout_layer(atom_type_embeddings + cartesian_positional_embeddings)

        # Here I need to reshape the input to (batch_size, max_seq_length, n_embd)
        x, mask = pad_and_mask_sequences(x, data.batch)
        attn_mask = mask.unsqueeze(1).unsqueeze(2)
        attn_mask = attn_mask.expand(-1, self.hparams.n_head, -1, -1)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attn_mask=attn_mask)
        x = self.output_layer_norm(x)
        x = x * mask.unsqueeze(-1)
        output = x.sum(dim=1)  # / atom_counts.unsqueeze(-1)

        logits = self.linear_output_head(output)
        loss_ce = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_types.view(-1).long(),
            ignore_index=-1,
        )

        # --- Here comes the flow matching logic ---

        time_step = torch.rand(batch_size, device=device)
        time_embeddings = create_time_embeddings(time_step, self.hparams.n_embd)
        pos_random = torch.randn(batch_size, 3, device=device)  # Shape: (num_graphs, 3)
        interpolation = target_pos - pos_random
        # t = 0 --> pos_random, t=1 --> target_pos
        interpolated_pos = pos_random + interpolation * time_step.unsqueeze(1)
        positional_embedding = self.cartesian_positional_embedding_fm(
            interpolated_pos
        )  # Shape: (num_graphs * seq_length, n_embd)
        x = positional_embedding + time_embeddings
        x = x + output
        output_fm = self.flow_matching_mlp(x)
        loss_fm = F.mse_loss(output_fm, interpolation, reduction="mean")

        # if targets is not None:
        #     # if we are given some desired targets also calculate the loss
        #     logits = self.linear_output_head(x)
        #     loss = F.cross_entropy(
        #         logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        #     )
        # else:
        #     # inference-time mini-optimization: only forward the lm_head on the very last position
        #     logits = self.linear_output_head(
        #         x[:, [-1], :]
        #     )  # note: using list [-1] to preserve the time dim
        #     loss = None

        return logits, loss_ce, loss_fm

    def data_augmentation(self, data: Data) -> Data:
        # Implement data augmentation logic here
        return data

    def configure_optimizers(self, betas=(0.9, 0.999)) -> Optimizer:
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.hparams.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        # fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        # use_fused = fused_available and device_type == "cuda"
        # extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.hparams.learning_rate,
            betas=betas,
            fused=True,  # **extra_args
        )
        # print(f"using fused AdamW: {use_fused}")

        return optimizer

    def shared_step(self, batch: Data, batch_idx: int) -> Tensor:
        logits, loss_ce, loss_fm = self(batch)

        return loss_ce, loss_fm

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        """Training step and logging"""
        # data augmentation by random rotation
        batch.pos = rotate_graphs_randomly(batch.pos, batch.batch)
        loss_ce, loss_fm = self.shared_step(batch, batch_idx)

        self.log(
            "train/train_loss",
            loss_ce + loss_fm,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=len(batch),
            reduce_fx="mean",
        )
        self.log(
            "train/train_loss_ce",
            loss_ce,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=len(batch),
            reduce_fx="mean",
        )
        self.log(
            "train/train_loss_fm",
            loss_fm,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=len(batch),
            reduce_fx="mean",
        )

        return loss_ce + loss_fm

    def validation_step(self, batch: Data, batch_idx: int) -> Tensor:
        loss_ce, loss_fm = self.shared_step(batch, batch_idx)

        self.log(
            "val/val_loss",
            loss_ce + loss_fm,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )
        self.log(
            "val/val_loss_fm",
            loss_fm,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )
        self.log(
            "val/val_loss_ce",
            loss_ce,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )

        return loss_ce + loss_fm

    def predict_step(
        self, batch: Data, batch_idx: int = 0
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        return self(batch)
