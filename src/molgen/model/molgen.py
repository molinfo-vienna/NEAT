from typing import Union
import math

import torch
from torch import Tensor
from lightning import LightningModule
from torch_geometric.data import Data
from torch.optim import Optimizer
from torch.nn import functional as F
from torch_geometric.nn.models import MLP
from torch_geometric.nn.pool import global_mean_pool
from torch.nn.functional import one_hot

from .attention import (
    LayerNorm,
    Block,
)
from .augmentation import RandomRotationAugmentation
from .utils import pad_and_mask_sequences, create_time_embeddings
from .positional_encoding import AxialRotaryPositionEncoding, FourierPositionEncoding
from .splitting import SourceTargetSplitter


class MolGen(LightningModule):
    def __init__(self, **params) -> None:
        super(MolGen, self).__init__()
        self.save_hyperparameters()
        # This will be handy when we introduce more hyper parameters
        # self.hparams.setdefault("key", "value")

        # Atom type embedding layer
        self.atom_type_embedding = torch.nn.Embedding(
            num_embeddings=self.hparams.vocab_size, embedding_dim=self.hparams.n_embd
        )

        # Fourier features for embedding of Cartesian coordinates
        self.fourier_embedding_layer = FourierPositionEncoding(
            out_dim=self.hparams.n_embd
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
                    AxialRotaryPositionEncoding(
                        embed_dim=self.hparams.n_embd,
                        num_heads=self.hparams.n_head,
                    ),
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

        # So far it is the GPT logic, Flow Matching only needs an additional MLP
        # TODO: Do we really need a seperate positional embedding layer here?
        self.fourier_embedding_layer_fm = FourierPositionEncoding(
            out_dim=self.hparams.n_embd
        )

        # A simple MLP with layer norm used for the flow network
        self.flow_matching_mlp = MLP(
            channel_list=[self.hparams.n_embd, self.hparams.n_embd, 3],
            dropout=self.hparams.dropout,
            bias=self.hparams.bias,
        )

        # init all weights, this is again from the GPT-2 code
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.hparams.n_layer)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

        self.splitter = SourceTargetSplitter(splitting_mode="cyclic")
        self.rotation_augmentation = RandomRotationAugmentation()
        self.target_set_max_size = -1

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())

        return n_params

    def _init_weights(self, module):
        """Initialize weights as in NanoGPT"""
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, data):
        device = data.x.device
        atom_counts = torch.bincount(data.batch)
        batch_size = atom_counts.size(0)
        max_atom_count = atom_counts.max()
        assert (
            max_atom_count <= self.hparams.block_size
        ), f"Cannot forward sequence of length {max_atom_count}, block size is only {self.hparams.block_size}"

        # We split the molecular data into source and target atom sets.
        # The indexing tensors point to the same molecules as in the original batch.
        # The source set contains at least one atom, and at most all atoms.
        # If it contains all atoms, then the target set will be empty.
        # The stop tokens mask indicates which molecules have empty target sets.
        # TODO: Make sure this sampling procedure really does what we want.
        (
            x_source,  # [n_source_atoms]
            pos_source,  # [n_source_atoms, 3]
            batch_source,  # [n_source_atoms]
            x_target,  # [n_target_atoms]
            pos_target,  # [n_target_atoms, 3]
            batch_target,  # [n_target_atoms]
            stop_tokens,  # [n_molecules]
        ) = self.splitter.create_source_target_split(data, device=device)

        # Now we compute the representation of the source atom sets with the transformer
        source_set_representation = self.source_set_representation(
            x_source, pos_source, batch_source, device
        )  # [n_molecules, n_embd]

        # From this representation, we can calculate a cross-entropy loss for atom type
        # prediction, and a flow matching loss for the target atom positions.
        # Note that these two objectives are disentangled and independent of each other.

        # --- Atom type / Stop token prediction loss ---

        logits = self.linear_output_head(
            source_set_representation
        )  # [n_target_sets, 118]

        loss_ce = self.compute_atom_type_loss(
            logits, x_target, batch_target, stop_tokens, device
        )

        loss_fm = self.compute_flow_matching_loss(
            x_target,
            pos_target,
            batch_target,
            stop_tokens,
            source_set_representation,
            device,
        )

        # --- Aggregate CE and FM losses over each target atom set ---

        # Now we simply add the two losses together. This could be weighted in future.
        loss = loss_ce + loss_fm

        return loss, loss_ce, loss_fm

    def source_set_representation(self, x_source, pos_source, batch_source, device):
        x_source = x_source.to(device)
        pos_source = pos_source.to(device)
        batch_source = batch_source.to(device)

        # Embedding layers for atom types and positions
        atom_type_embeddings = self.atom_type_embedding(
            x_source
        )  # [n_source_atoms, n_embd]
        fourier_positional_embeddings = self.fourier_embedding_layer(
            pos_source
        )  # [n_source_atoms, n_embd]
        x = self.dropout_layer(
            atom_type_embeddings + fourier_positional_embeddings
        )  # [n_source_atoms, n_embd]

        # Here we need to reshape the input to [batch_size, max_atom_count, n_embd].
        # This could also be done with sequence packing, but for now we keep it simple.
        # The output tensor is padded with zeros for all source sets with less atoms
        # than the largest source atom set in the batch. The atom mask keeps track of
        # which entries correspond to atoms and padding.
        x, atom_mask = pad_and_mask_sequences(
            x, batch_source
        )  # [n_molecules, max_atom_count, n_embd], [n_molecules, max_atom_count]

        # The attention mask corresponds to the atom mask, but needs to be broadcasted
        # to the number of attention heads.
        attn_mask = atom_mask.unsqueeze(1).unsqueeze(
            2
        )  # [n_molecules, 1, 1, max_atom_count]
        attn_mask = attn_mask.expand(
            -1, self.hparams.n_head, -1, -1
        )  # [n_molecules, n_head, 1, max_atom_count]

        # The positions need to be padded in the same way as the atom embeddings.
        # This will be needed for applying the rotary positional embeddings in the
        # transformer blocks.
        pos, _ = pad_and_mask_sequences(
            pos_source, batch_source
        )  # [n_molecules, max_atom_count, 3]

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(
                x, attn_mask=attn_mask, pos=pos
            )  # [n_molecules, max_atom_count, n_embd]
        x = self.output_layer_norm(x)  # [n_molecules, max_atom_count, n_embd]

        # During the forward pass through the trandformer layers, the zero-paddings
        # get filled with non-zero values. This should not be a problem, since these
        # are masked out in the attention mechanism, but before pooling the atom
        # embeddings into a molecule embedding, we re-apply the atom mask.
        # TODO: Investigate where this behavior comes from, maybe it influences batch statistics of the MLP and LayerNorms?
        x = x * atom_mask.unsqueeze(-1)  # [n_molecules, max_atom_count, n_embd]

        # Now we can pool the atom embeddings by summation along the max_atom_count dimension
        # TODO: Investigate how attention pooling works here.
        source_set_representation = x.sum(dim=1)  # [n_molecules, n_embd]

        return source_set_representation

    def compute_atom_type_loss(
        self, logits, x_target, batch_target, stop_tokens, device
    ):
        logits.to(device)
        x_target.to(device)
        batch_target.to(device)
        stop_tokens.to(device)
        # Atom type prediction is done with a cross-entropy loss.
        # Importantly, since we can have multiple atoms in the target set per source set,
        # we are modelling a target type *distribution*. This distribution is the mean
        # over the one-hot encodings of the target atom types.

        # (1) Map target atom indices to contiguous indices to avoid errors in the aggregation step.
        _, batch_target_contiguous = torch.unique(
            batch_target.clone(), return_inverse=True
        )  # [n_target_atoms]
        # (2) Take the mean over the one-hot encodings of the target atom types
        # TODO: Using all atom types in not necessary. However, it could be interesting to differentiate atoms with different valences.
        x_target_prob = one_hot(x_target.long(), 118).float()  # [n_target_atoms, 118]
        x_target_prob = global_mean_pool(
            x_target_prob.float(), batch_target_contiguous
        )  # [n_target_sets, 118]

        # (3) Incorporate the stop tokens into the target type distributions
        combined_prob = torch.zeros((stop_tokens.shape[0], 118), dtype=torch.float, device=device)
        combined_prob[stop_tokens, 0] = 1.0
        combined_prob[~stop_tokens] = x_target_prob

        # (4) Compute the cross-entropy loss between predicted logits and target type distributions
        loss_ce = F.cross_entropy(
            logits,
            combined_prob,
            reduction="mean",
        )  # [1]

        return loss_ce

    def compute_flow_matching_loss(
        self,
        x_target,
        pos_target,
        batch_target,
        stop_tokens,
        source_set_representation,
        device,
    ):
        x_target.to(device)
        pos_target.to(device)
        batch_target.to(device)
        stop_tokens.to(device)
        source_set_representation.to(device)
        # --- Here comes the flow matching logic ---

        # Map target atom indices to contiguous indices to avoid errors in the aggregation step.
        batch_size = len(source_set_representation)
        _, batch_target_contiguous = torch.unique(
            batch_target.clone(), return_inverse=True
        )  # [n_target_atoms]
        n_target_sets = (
            batch_size - stop_tokens.sum()
        )  # Number of non-empty target sets
        n_target_atoms = len(
            batch_target_contiguous
        )  # Num atoms in non-empty target sets

        # (1) We sample a random position for each non_empty target set
        pos_random = torch.randn(n_target_sets, 3, device=device)  # [n_target_sets, 3]

        # (2) We need to expand this to the number of atoms in the target sets, this is
        # number conditional paths we are regressing in the CFM objective. For each
        # possible path, we draw a random time step, and compute the interpolated position
        # between the random and target position.
        # Interpolation: t = 0 --> pos_random, t=1 --> target_pos
        pos_random_expanded = pos_random[batch_target_contiguous]  # [n_target_atoms, 3]
        time_step = torch.rand(n_target_atoms, device=device)  # [n_target_atoms]
        interpolation = pos_target - pos_random_expanded  # [n_target_atoms, 3]
        interpolated_pos = pos_random_expanded + interpolation * time_step.unsqueeze(
            1
        )  # [n_target_atoms, 3]

        # (3) Now we can compute the vector field output of the flow network at the
        # interpolated positions and time steps.
        output_fm = self.compute_vector_field(
            x_target,
            interpolated_pos,
            time_step,
            batch_target,
            source_set_representation,
            device,
        )  # [n_target_atoms, 3]

        # (4) The flow matching loss is the MSE between the predicted vector field and
        # the interpolation (pos_1 - pos_0).
        # TODO: Maybe we should aggregate this over source atom sets, and not atoms? But weighting each path individually is also a valid strategy
        loss_fm = torch.mean((output_fm - interpolation) ** 2)

        return loss_fm

    def compute_vector_field(
        self,
        x_target,
        pos_t,
        time_step,
        batch_target,
        source_set_representation,
        device,
    ):
        x_target.to(device)
        pos_t.to(device)
        time_step.to(device)
        batch_target.to(device)
        source_set_representation.to(device)

        # (1) Embed time steps with sinusoidal embeddings
        time_embeddings = create_time_embeddings(
            time_step, self.hparams.n_embd
        )  # [n_target_atoms, n_embd]

        # (2) Embed the given positions at time t with Fourier features
        # TODO: Maybe use weight tying with the other positional embedding layer?
        positional_embedding = self.fourier_embedding_layer_fm(
            pos_t
        )  # [n_target_atoms, n_embd]

        # (3) CFM paths are conditioned on the type of the respective target atoms,
        # so we need to include this information in the flow matching condition.
        target_atom_type_embeddings = self.atom_type_embedding(
            x_target
        )  # [n_target_atoms, n_embd]

        # (4) Add embeddings up and predict the vector field
        x = (
            positional_embedding
            + time_embeddings
            + target_atom_type_embeddings
            + source_set_representation[batch_target]  # [n_target_atoms, n_embd]
        )  # [n_target_atoms, n_embd]
        output_fm = self.flow_matching_mlp(x)  # [n_target_atoms, 3]

        return output_fm

    def configure_optimizers(self, betas=(0.9, 0.999)) -> Optimizer:
        """Same configurations as in NanoGPT"""
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
            fused=True,
        )

        return optimizer

    def shared_step(self, batch: Data, batch_idx: int) -> Tensor:
        loss, loss_ce, loss_fm = self(batch)

        return loss, loss_ce, loss_fm

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        """Training step and logging"""
        # data augmentation by random rotation
        batch.pos = self.rotation_augmentation.rotate_graphs_randomly(
            batch.pos, batch.batch
        )
        loss, loss_ce, loss_fm = self.shared_step(batch, batch_idx)

        self.log(
            "train/train_loss",
            loss,
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

        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> Tensor:
        loss, loss_ce, loss_fm = self.shared_step(batch, batch_idx)

        self.log(
            "val/val_loss",
            loss,
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

        return loss

    def predict_step(
        self, batch: Data, batch_idx: int = 0
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        return self(batch)
