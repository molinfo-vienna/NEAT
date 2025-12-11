"""
Taken and modified from the nanoGPT repository:
https://github.com/karpathy/nanoGPT/blob/master/model.py
With contributions from:
https://arxiv.org/pdf/2403.03206.pdf
"""

import math
import types

import torch
import torch.nn as nn
from lightning import LightningModule
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from torch_geometric.data import Data
from torch_geometric.nn.pool import global_add_pool, global_mean_pool

from .attention import Block
from .positional_encoding import AxialRotaryPositionEncoding, FourierPositionEncoding
from .simple_mlp import SimpleMLPAdaLN


class MolGen(LightningModule):
    def __init__(self, **params) -> None:
        super(MolGen, self).__init__()
        self.hparams.setdefault("noise_std", 1.0)
        self.save_hyperparameters()

        # Atom type embedding layer
        self.atom_type_embedding = nn.Embedding(
            num_embeddings=self.hparams.vocab_size, embedding_dim=self.hparams.n_embd
        )

        # Fourier features for embedding of Cartesian coordinates
        self.fourier_embedding_layer = FourierPositionEncoding(
            out_dim=self.hparams.n_embd
        )

        # Dropout layer
        self.dropout_layer = nn.Dropout(self.hparams.dropout)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                Block(
                    self.hparams.n_embd,
                    self.hparams.n_head,
                    self.hparams.dropout,
                    self.hparams.bias,
                    (
                        AxialRotaryPositionEncoding(
                            embed_dim=self.hparams.n_embd,
                            num_heads=self.hparams.n_head,
                        )
                        if self.hparams.rope
                        else None
                    ),
                )
                for _ in range(self.hparams.n_layer)
            ]
        )

        # Layer normalization after the transformer blocks
        self.layer_norm_after_transformer = nn.LayerNorm(
            self.hparams.n_embd, bias=False
        )

        # Linear prediction head for atom type prediction
        self.atom_type_prediction_head = nn.Linear(
            self.hparams.n_embd,
            self.hparams.vocab_size,
            bias=self.hparams.bias,
        )

        # Init all weights (taken from the nanoGPT repository)
        self.apply(self._init_weights)
        # Apply special scaled initialization to the residual projections
        # (taken from the nanoGPT repository)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.hparams.n_layer)
                )

        # This is the Diffusion MLP with AdaLN conditioning.s
        # It was used in the original diffusion loss paper, and QUETZAL also uses it.
        config = types.SimpleNamespace(
            diff_w=self.hparams.n_embd_fm,  # model hidden width
            n_embd=self.hparams.n_embd,  # dimension of conditioning vector c
            diff_fourier=512,  # number of Fourier channels for coord embedding
            coord_bandwidth=20.0,  # frequency bandwidth for Fourier features
            diff_d=self.hparams.n_layers_fm,  # number of residual blocks
            diff_mlp="mlp",  # use MLP feedforward
            diff_mlp_expand=1,  # expansion factor for MLP
        )
        self.ada_mlp = SimpleMLPAdaLN(config)

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights as in NanoGPT"""
        if isinstance(module, nn.Linear):
            # std is chosen w.r.t. sqrt(embd_dim)
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())

        return n_params

    def forward(self, data: Data) -> tuple[Tensor, Tensor, Tensor]:
        device = data.x.device

        # We split the molecular data into source and target atom sets.
        # The indexing tensors point to the same molecules as in the original batch.
        # The source set contains at least one atom, and at most all atoms.
        # If it contains all atoms, then the target set will be empty.
        # The stop tokens mask indicates which molecules have empty target sets.

        # (1) Compute the representation of the source atom sets with the transformer
        source_set_representation = self.compute_source_set_representation(
            data.x_source, data.pos_source, data.batch_source, device
        )  # [batch_size, n_embd]

        # (2) Compute the logits for the atom type prediction
        logits = self.atom_type_prediction_head(
            source_set_representation
        )  # [n_target_sets, vocab_size]

        # (3) Calculate a cross-entropy loss for atom type prediction
        loss_ce = self.compute_atom_type_loss(
            logits, data.x_target, data.batch_target, data.stop_tokens, device
        )

        # (4) Calculate a flow matching loss for the target atom positions
        loss_fm = self.compute_flow_matching_loss(
            data.x_target,
            data.pos_target,
            data.pos_random,
            data.batch_target,
            source_set_representation,
            device,
        )

        # (5) Add the two losses together
        # Note that these two objectives are disentangled and independent of each other.
        loss = loss_ce + loss_fm

        return loss, loss_ce, loss_fm

    def compute_source_set_representation(
        self,
        x_source: Tensor,
        pos_source: Tensor,
        batch_source: Tensor,
        device: torch.device,
    ) -> Tensor:
        """
        Compute the representation of the source atom sets.

        Args:
            x_source (Tensor): The atom types of the source atoms. shape: [n_source_atoms]
            pos_source (Tensor): The positions of the source atoms. shape: [n_source_atoms, 3]
            batch_source (Tensor): The batch indices of the source atoms. shape: [n_source_atoms]
            device (torch.device): The device to use for computations.

        Returns:
            Tensor: The representation of the source atom sets. shape: [batch_size, n_embd]
        """
        x_source = x_source.to(device)
        pos_source = pos_source.to(device)
        batch_source = batch_source.to(device)

        # (1) Compute atom counts of the source sets
        atom_count_source = torch.bincount(batch_source)

        # (2) Reshape the input to [batch_size, max_atom_count, n_embd].
        # This could also be done with sequence packing, but for now we keep it simple.
        # The output tensor is padded with zeros for all source sets with less atoms
        # than the largest source atom set in the batch. The atom mask keeps track of
        # which entries correspond to atoms and padding.
        dim = [len(atom_count_source), atom_count_source.max(), self.hparams.n_embd]
        x = torch.zeros(dim, device=device)  # [batch_size, max_atom_count, n_embd]
        context_range = torch.arange(
            atom_count_source.max(), device=atom_count_source.device
        ).unsqueeze(0)
        atom_mask = context_range < atom_count_source.unsqueeze(
            1
        )  # [batch_size, max_atom_count]
        # The attention mask is used in the transformer blocks and is the outer product of the atom mask.
        attn_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(
            2
        )  # [batch_size, max_atom_count, max_atom_count]
        attn_mask = attn_mask.unsqueeze(1).expand(
            -1, self.hparams.n_head, -1, -1
        )  # [batch_size, n_head, max_atom_count, max_atom_count]

        # (3) Embed the atom types and positions
        atom_type_embedding = self.atom_type_embedding(
            x_source
        )  # [n_source_atoms, n_embd]
        positional_embedding = self.fourier_embedding_layer(
            pos_source
        )  # [n_source_atoms, n_embd]

        # (4) Combine the atom type embedding and the positional embedding
        input_embedding = (
            atom_type_embedding + positional_embedding
        )  # [n_source_atoms, n_embd]

        # (5) Apply the dropout layer
        input_embedding = self.dropout_layer(
            input_embedding
        )  # [n_source_atoms, n_embd]

        # (6) Apply the atom mask to the input embedding
        x[atom_mask] = input_embedding  # [batch_size, max_atom_count, n_embd]

        # (7) Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(
                x, attn_mask=attn_mask, pos=pos_source
            )  # [batch_size, max_atom_count, n_embd]

        # (8) Apply the output layer normalization
        x = self.layer_norm_after_transformer(x)  # [batch_size, max_atom_count, n_embd]

        # During the forward pass through the trandformer layers, the zero-paddings
        # get filled with non-zero values. This should not be a problem, since these
        # are masked out in the attention mechanism, but before pooling the atom
        # embeddings into a molecule embedding, we re-apply the atom mask.
        # TODO: Investigate where this behavior comes from, maybe it influences batch statistics of the MLP and LayerNorms?
        # (9) Apply the atom mask to the input embedding
        x = x * atom_mask.unsqueeze(-1)  # [batch_size, max_atom_count, n_embd]

        # (10) Pool the atom embeddings into a molecule embedding
        source_set_representation = x.sum(dim=1)  # [batch_size, n_embd]

        return source_set_representation

    def compute_atom_type_loss(
        self,
        logits: Tensor,
        x_target: Tensor,
        batch_target: Tensor,
        stop_tokens: Tensor,
        device: torch.device,
    ) -> Tensor:
        """
        Compute the atom type prediction loss.

        Args:
            logits (Tensor): The logits of the atom type predictions. shape: [n_target_sets, vocab_size]
            x_target (Tensor): The target atom types. shape: [n_target_atoms]
            batch_target (Tensor): The batch indices of the target atoms. shape: [n_target_atoms]
            stop_tokens (Tensor): The stop tokens. shape: [batch_size]
            device (torch.device): The device to use for computations.

        Returns:
            Tensor: The atom type prediction loss. shape: [1]

        """
        logits = logits.to(device)
        x_target = x_target.to(device)
        batch_target = batch_target.to(device)
        stop_tokens = stop_tokens.to(device)

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
        x_target_prob = F.one_hot(
            x_target.long(), self.hparams.vocab_size
        ).float()  # [n_target_atoms, vocab_size]
        x_target_prob = global_mean_pool(
            x_target_prob.float(), batch_target_contiguous
        )  # [n_target_sets, vocab_size]

        # (3) Incorporate the stop tokens into the target type distributions
        combined_prob = torch.zeros(
            (stop_tokens.shape[0], self.hparams.vocab_size),
            dtype=torch.float,
            device=device,
        )
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
        x_target: Tensor,
        pos_target: Tensor,
        pos_random: Tensor,
        batch_target: Tensor,
        source_set_representation: Tensor,
        device: torch.device,
        resampling=4,
    ) -> Tensor:
        """
        Compute the flow matching loss.

        Args:
            x_target (Tensor): The target atom types. shape: [n_target_atoms]
            pos_target (Tensor): The target positions. shape: [n_target_atoms, 3]
            batch_target (Tensor): The batch indices of the target atoms. shape: [n_target_atoms]
            stop_tokens (Tensor): The stop tokens. shape: [batch_size]
            source_set_representation (Tensor): The representation of the source sets. shape: [batch_size, n_embd]
            device (torch.device): The device to use for computations.
            resampling (int): The number of resampling steps.

        Returns:
            Tensor: The flow matching loss. shape: [1]
        """
        x_target = x_target.to(device)
        pos_target = pos_target.to(device)
        batch_target = batch_target.to(device)
        source_set_representation = source_set_representation.to(device)
        pos_random = pos_random.to(device)
        batch_target = batch_target.long()

        # # (1) We sample a random position for each non_empty target set
        # #     with optimal transport correction.

        # # (1.1) Compute path indices. Atoms of the same type in the same molecule have the same index.
        # _, idx = torch.unique(
        #     batch_target * 100 + x_target, return_inverse=True
        # )  # [n_paths]
        # # (1.2) Compute number of paths. Number of paths is the sum of unique atom types over all molecules.
        # n_paths = idx.max() + 1  # [1]
        # # (1.3) Sample random positions for each path.
        # pos_random = self.hparams.noise_std * torch.randn(
        #     n_paths, 3, device=device
        # )  # [n_paths, 3]
        # # (1.4) Expand the random positions to the number of atoms in the target sets.
        # #       This is done by indexing the random positions with the path indices.
        # #       It is necessary to find which atom of the correct type in the target set
        # #       is closest to the randomly sampled positions (optimal transport).
        # pos_random_expanded = pos_random[idx]  # [n_target_atoms, 3]
        # # (1.5) Compute the distance between the target positions and the randomly sampled positions.
        # distance = torch.norm(
        #     pos_target - pos_random_expanded, dim=1
        # )  # [n_target_atoms]
        # # (1.6) Find the indices of the minimum distances.
        # #       This returns a tensor of length n_paths pointing to the atom
        # #       in the target set that is closest to the randomly sampled position.
        # #       This returns an index tensor.
        # min_indices = self.global_argmin_pool(distance, idx)  # [n_paths]
        # # (1.7) Compute the reweighting factor. This is the number of atoms of the same type in the target set.
        # reweighting = global_add_pool(torch.ones_like(distance), idx)  # [n_paths]
        # # (1.8) Select the closest target positions given the minimum distance indices.
        # pos_target = pos_target[min_indices]  # [n_paths, 3]

        n_paths = pos_target.shape[0]
        # batch_target = batch_target.long()
        # pos_random = self.hparams.noise_std * torch.randn_like(pos_target)
        # target_idx = torch.unique(batch_target)
        # for idx in target_idx:
        #     cost_matrix = torch.cdist(
        #         pos_target[batch_target == idx], pos_random[batch_target == idx], p=2
        #     )
        #     _, prior_idx = linear_sum_assignment(cost_matrix.cpu())

        #     # reorder prior to according to optimal assignment
        #     pos_random[batch_target == idx] = pos_random[batch_target == idx][prior_idx]

        # (2) Interpolation: t = 0 --> pos_random, t=1 --> target_pos
        interpolation = pos_target - pos_random  # [n_paths, 3]

        # (3) For each path, draw k random time steps
        resampling = self.hparams.time_step_resampling
        if self.hparams.time_step_sampling == "uniform":
            time_step = self.sample_timesteps_uniform(
                n_paths * resampling, device=device
            )  # [n_paths * k]
        elif self.hparams.time_step_sampling == "logit_normal":
            time_step = 0.98 * self.sample_timesteps_logit_normal(
                n_paths * resampling, device=device, m=0.8, s=1.7
            ) + 0.02 * self.sample_timesteps_uniform(
                n_paths * resampling, device=device
            )  # [n_paths * k]

        # (4) Since we sample k time steps per path, we need to expand all other tensors accordingly
        # x_target = x_target[min_indices]
        x_target = torch.cat([x_target for _ in range(resampling)], dim=0)
        pos_random = torch.cat([pos_random for _ in range(resampling)], dim=0)
        pos_target = torch.cat([pos_target for _ in range(resampling)], dim=0)
        interpolation = torch.cat([interpolation for _ in range(resampling)], dim=0)
        source_set_representations = source_set_representation[batch_target]
        # source_set_representations = source_set_representations[min_indices]
        source_set_representations = torch.cat(
            [source_set_representations for _ in range(resampling)], dim=0
        )
        # reweighting = torch.cat([reweighting for _ in range(resampling)], dim=0)

        # (5) Calculate k interpolated positions per path given the sampled time steps
        interpolated_pos = pos_random + interpolation * time_step.unsqueeze(
            1
        )  # [n_paths * k, 3]

        # (6) Compute the vector field output of the flow network at the
        # interpolated positions and time steps.
        output_fm = self.compute_vector_field(
            x_target,
            interpolated_pos,
            time_step,
            source_set_representations,
            device,
        )  # [n_paths * k, 3]

        # (7) Compute the flow matching loss.
        # This is the MSE between the predicted vector field and
        # the interpolation (pos_1 - pos_0) for each path.
        loss_fm = torch.mean((output_fm - interpolation) ** 2, dim=1)  # [n_paths * k]

        # (8) Reweight the loss by the number of atoms of the same type in the target set.
        # loss_fm = loss_fm * reweighting  # [n_paths * k]

        # (9) Return the mean loss over all paths and time steps.
        return loss_fm.mean()  # [1]

    def global_argmin_pool(self, x: Tensor, batch: Tensor) -> Tensor:
        """
        Performs a global pooling operation to find the indices of the minimum values
        for each group in the batch.

        Args:
            x (Tensor): The input tensor of shape `(n_atoms,)` (e.g., float values).
            batch (Tensor): A tensor of shape `(n_atoms,)` that maps each atom to a molecule.

        Returns:
            Tensor: Indices of the minimum values for each molecule.
        """
        # Get the number of unique molecules
        n_molecules = batch.max().item() + 1

        # Create a large tensor to store the values for each molecule
        max_value = x.max() + 1
        expanded_x = torch.full((n_molecules, x.size(0)), max_value, device=x.device)

        # Scatter the values of x into the expanded tensor
        expanded_x[batch, torch.arange(x.size(0))] = x

        # Find the minimum values and their indices
        _, min_indices = expanded_x.min(dim=1)

        return min_indices

    def sample_timesteps_uniform(
        self, num_samples: int, device: torch.device
    ) -> Tensor:
        """
        Sample timesteps from a uniform distribution.

        Args:
            num_samples (int): The number of timesteps to sample.
            device (torch.device): The device to use for computations.

        Returns:
            Tensor: The sampled timesteps. shape: [num_samples]
        """
        return torch.rand(num_samples, device=device)

    def sample_timesteps_logit_normal(
        self, num_samples: int, device: torch.device, m: float = 0.8, s: float = 1.7
    ) -> Tensor:
        """
        Sample timesteps from a logit-normal distribution.
        Adapated from https://arxiv.org/pdf/2403.03206.pdf

        Args:
            num_samples (int): The number of timesteps to sample.
            device (torch.device): The device to use for computations.
            m (float): The mean of the logit-normal distribution.
            s (float): The standard deviation of the logit-normal distribution.

        Returns:
            Tensor: The sampled timesteps. shape: [num_samples]
        """
        u = torch.randn(num_samples, device=device) * s + m
        t = 1 / (1 + torch.exp(-u))
        return t

    def compute_vector_field(
        self,
        x: Tensor,
        pos_t: Tensor,
        time_step: Tensor,
        source_set_representation: Tensor,
        device: torch.device,
    ) -> Tensor:
        """Method to compute the vector field of the flow matching network.

        Args:
            x (Tensor): The atom types of the noisy atoms. shape: [n_atoms, 1]
            pos_t (Tensor): The noisy positions at time t. shape: [n_atoms, 3]
            time_step (Tensor): The current time step. shape: [n_atoms], values in [0, 1]
            source_set_representation (Tensor): Learned representation of the source sets.
                shape: [n_atoms, n_embd]
            device (torch.device): cuda or cpu.

        Returns:
            Tensor: Vector field of shape [n_atoms, 3]
        """
        x = x.to(device)
        pos_t = pos_t.to(device)
        time_step = time_step.to(device)
        source_set_representation = source_set_representation.to(device)

        # CFM paths are conditioned on the type of the respective target atoms,
        # so we need to include this information in the flow matching condition.
        target_atom_type_embeddings = self.atom_type_embedding(
            x
        )  # [n_target_atoms, n_embd]

        condition = (
            target_atom_type_embeddings + source_set_representation
        )  # [n_target_atoms, n_embd]

        output_fm = self.ada_mlp(
            pos_t, time_step, condition
        )  # [n_target_atoms, n_embd]

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
            # fused=True,
        )

        def lr_lambda(epoch):
            # 1) linear warmup for warmup_iters steps
            warmup_epochs = self.hparams.lr_warmup_epochs
            min_lr = self.hparams.lr_min_ratio
            lr_decay_epochs = self.hparams.max_epochs
            if epoch < warmup_epochs:
                return (epoch + 1) / (warmup_epochs + 1)
            # 2) if it > lr_decay_iters, return min learning rate
            if epoch > lr_decay_epochs:
                return min_lr
            # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (epoch - warmup_epochs) / (lr_decay_epochs - warmup_epochs)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
            return min_lr + coeff * (1.0 - min_lr)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return [optimizer], [scheduler]

    def on_before_optimizer_step(
        self, optimizer: Optimizer, optimizer_idx: int = None
    ) -> None:
        """
        Compute the gradient norm before clipping.

        Args:
            optimizer (Optimizer): The optimizer to use.
            optimizer_idx (int): The index of the optimizer.

        Returns:
            None
        """
        grad_norm = 0
        for param in self.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm(2).item() ** 2
        grad_norm = grad_norm**0.5

        self.log(
            "train/grad_norm",
            grad_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

    def shared_step(self, batch: Data, batch_idx: int) -> Tensor:
        loss, loss_ce, loss_fm = self(batch)

        return loss, loss_ce, loss_fm

    def on_train_start(self) -> None:
        """Initialization of the logger"""
        self.logger.log_hyperparams(
            self.hparams,
            {"train/train_loss": torch.inf, "val/val_loss": torch.inf},
        )

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        """Training step and logging"""
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

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        max_atoms: int = 100,
        num_time_steps: int = 30,
        device: torch.device = torch.device("cuda"),
        prefix_x: Tensor = None,
        prefix_pos: Tensor = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Generate a molecule using the flow matching network.

        Args:
            batch_size (int): The number of molecules to generate.
            max_atoms (int): The maximum number of atoms to generate.
            num_time_steps (int): The number of time steps to use for the flow matching.
            device (torch.device): The device to use for computations.

        Returns:
            tuple[Tensor, Tensor, Tensor]: The generated molecule, its positions, and the batch indices.
        """
        if prefix_x is not None and prefix_pos is not None:
            # (1) Initialize starting atom types with the provided prefix
            x = torch.cat([prefix_x for _ in range(batch_size)]).to(device)
            # (2) Initialize starting positions with the provided prefix
            pos = torch.cat([prefix_pos for _ in range(batch_size)], dim=0).to(device)
            # (3) Initialize the batch source tensor with the provided prefix
            batch_source = torch.cat(
                [torch.ones_like(prefix_x) * i for i in range(batch_size)]
            ).to(device)
        else:
            # (1) Sample initial atoms from the prior distribution of atom types in QM9
            dist = torch.tensor(
                [0.0000, 0.5109, 0.3517, 0.0580, 0.0780, 0.0014], device=device
            )
            x = torch.multinomial(dist, batch_size, replacement=True)  # [batch_size]
            # (2) Initialize starting positions with random ones
            pos = self.hparams.noise_std * torch.randn(batch_size, 3, device=device)
            # (3) Initialize the batch source tensor
            batch_source = torch.arange(batch_size, device=device)
        # (4) Create a mask for the stop tokens that will be used to track which molecules have a stop token
        stop_token_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
        # (5) Create a tensor of molecule indices that do not have a stop token
        active_mol_idx = torch.arange(batch_size, device=device)[~stop_token_mask]

        # (6) Iterate over the maximum number of atoms to generate
        for i in range(max_atoms):
            # (6.1) Compute source set representation
            expanded_mask = torch.isin(batch_source, active_mol_idx)
            masked_x = x[expanded_mask]
            masked_pos = pos[expanded_mask]
            masked_batch_source = batch_source[expanded_mask]
            _, batch_source_remapped = torch.unique(
                masked_batch_source.clone(), return_inverse=True
            )
            source_set_representation = self.compute_source_set_representation(
                masked_x, masked_pos, batch_source_remapped, device
            )  # [active_mol_count, n_embd]

            # (6.2) Compute logits
            logits = self.atom_type_prediction_head(
                source_set_representation
            )  # [active_mol_count, vocab_size]

            # (6.3) Compute probabilities
            probabilities = F.softmax(logits, dim=-1)  # [active_mol_count, vocab_size]

            # (6.4) Sample next atom types from the resulting distribution
            x_next = torch.argmax(probabilities, dim=1)

            # (6.5) Create a mask on the active molecules given the newly predicted atom types
            x_next_mask = x_next == 0  # [active_mol_count]

            print(f"Generating atom {i + 2} for {(~x_next_mask).sum()} molecules.")

            # (6.6) Update the stop token mask with the newly predicted stop tokens
            stop_token_mask[active_mol_idx] += x_next_mask  # [batch_size]

            # (6.7) Count the number of stop tokens and break if all molecules
            # have a stop token also update the active molecule indices and count
            n_stop_tokens = stop_token_mask.sum()
            active_mol_idx = torch.arange(batch_size, device=device)[
                ~stop_token_mask
            ]  # [active_mol_count] carefull, this might be shorter than before, if stop tokens were predicted!
            if n_stop_tokens == batch_size:
                break

            # (6.8) Select only the source set representations for the active molecules
            x_next = x_next[~x_next_mask]
            source_set_representation = source_set_representation[~x_next_mask]
            # (6.9) Calculate the positions of the newly predicted atoms with flow matching
            pos_next = self.calculate_positions(
                x_next, source_set_representation, num_time_steps, device
            )

            # (6.10) Update the x, pos, and batch source tensors
            updated_x = []
            updated_pos = []
            updated_batch = []
            for idx in range(batch_size):
                if idx in active_mol_idx:
                    active_idx = torch.where(active_mol_idx == idx)[0]
                    updated_x.append(
                        torch.cat(
                            (x[batch_source == idx], x_next[active_idx].view(1)), dim=0
                        )
                    )  # [num_atoms+1]
                    updated_pos.append(
                        torch.cat(
                            (pos[batch_source == idx], pos_next[active_idx].view(1, 3)),
                            dim=0,
                        )
                    )  # [num_atoms+1, 3]
                    updated_batch.append(
                        torch.cat(
                            (
                                batch_source[batch_source == idx],
                                torch.tensor(idx, device=device).view(1),
                            ),
                            dim=0,
                        )
                    )  # [num_atoms+1]
                else:
                    updated_x.append(x[batch_source == idx])
                    updated_pos.append(pos[batch_source == idx])
                    updated_batch.append(batch_source[batch_source == idx])

            x = torch.cat(updated_x, dim=0)  # [batch_size]
            pos = torch.cat(updated_pos, dim=0)  # [batch_size, 3]
            pos -= pos.mean(dim=0, keepdim=True)
            batch_source = torch.cat(updated_batch, dim=0)  # [batch_size]

        return x, pos, batch_source

    def calculate_positions(
        self,
        x_next: Tensor,
        source_set_representation: Tensor,
        num_time_steps: int,
        device: torch.device,
        integration_method: str = "euler",
    ) -> Tensor:
        """
        Method to calculate the positions of the newly predicted atoms with flow matching.

        Args:
            x_next (Tensor): The atom types of the newly predicted atoms. shape: [n_atoms, 1]
            source_set_representation (Tensor): The representation of the source sets. shape: [n_atoms, n_embd]
            num_time_steps (int): The number of time steps to use for the flow matching.
            device (torch.device): The device to use for computations.
            integration_method (str): The integration method to use for the flow matching.

        Returns:
            Tensor: The positions of the newly predicted atoms. shape: [n_atoms, 3]
        """
        # (1) Initialize next atoms' position with a random position
        pos_next = self.hparams.noise_std * torch.randn(
            x_next.shape[0], 3, device=device
        )

        time_steps = torch.linspace(0, 1, num_time_steps, device=device)
        # time_steps = torch.cumsum(
        #    torch.arange(num_time_steps, 0, -1, device=device, dtype=torch.long), dim=0
        # )
        # time_steps = torch.cat([torch.tensor([0], device=device), time_steps])
        # time_steps = time_steps / time_steps[-1]
        dts = time_steps[1:] - time_steps[:-1]

        if integration_method == "euler":
            # (2) Find position of the atoms via flow matching using Euler method
            for dt, time_step in zip(dts, time_steps[:-1]):
                # for time_step in torch.linspace(0, 1, num_time_steps, device=device)[:-1]:
                time_step = time_step.expand(x_next.shape[0])
                delta_pos = dt * self.compute_vector_field(
                    x_next,
                    pos_next,
                    time_step,
                    source_set_representation,
                    device=device,
                )
                pos_next = pos_next + delta_pos

        elif integration_method == "euler_maruyama":

            def diffusion_coefficient(t, epsilon=0.1, w_cutoff=0.9):
                # determine diffusion coefficient
                w = (1.0 - t) / (t + epsilon)
                if t >= w_cutoff:
                    w = torch.zeros_like(t)
                return w

            # (2) Find position of the atoms via flow matching using Euler method
            for time_step in torch.linspace(0, 1, num_time_steps, device=device)[:-1]:
                dt = 1 / num_time_steps
                eps = torch.randn_like(pos_next)
                velocity = self.compute_vector_field(
                    x_next,
                    pos_next,
                    time_step.expand(x_next.shape[0]),
                    source_set_representation,
                    device=device,
                )
                # score = time_step * velocity - pos_next / (1 - time_step)
                score = self.compute_score_from_velocity(velocity, pos_next, time_step)
                diff_coeff = diffusion_coefficient(time_step)
                drift = velocity + diff_coeff * score
                mean_pos = pos_next + drift * dt
                pos_next = mean_pos + torch.sqrt(2.0 * diff_coeff * dt * 0.3) * eps

        elif integration_method == "rk4":
            # (2) Find position of the atoms via flow matching using RK4 method
            for time_step in torch.linspace(0, 1, num_time_steps, device=device)[:-1]:
                time_step = time_step.expand(x_next.shape[0])
                dt = 1 / num_time_steps  # Time step size

                # Compute RK4 intermediate steps
                k1 = self.compute_vector_field(
                    x_next,
                    pos_next,
                    time_step,
                    source_set_representation,
                    device=device,
                )
                k2 = self.compute_vector_field(
                    x_next,
                    pos_next + 0.5 * dt * k1,
                    time_step + 0.5 * dt,
                    source_set_representation,
                    device=device,
                )
                k3 = self.compute_vector_field(
                    x_next,
                    pos_next + 0.5 * dt * k2,
                    time_step + 0.5 * dt,
                    source_set_representation,
                    device=device,
                )
                k4 = self.compute_vector_field(
                    x_next,
                    pos_next + dt * k3,
                    time_step + dt,
                    source_set_representation,
                    device=device,
                )

                # Update position using RK4 formula
                delta_pos = (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
                pos_next = pos_next + delta_pos

            else:
                raise ValueError(f"Unknown integration method: {integration_method}")

        return pos_next

    def compute_score_from_velocity(self, v_t, y_t, t):
        # t = right_pad_dims_to(y_t, t)
        alpha_t, d_alpha_t = t, 1
        sigma_t, d_sigma_t = 1 - t, -1
        mean = y_t
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * v_t - mean) / var
        return score

    def calculate_positional_trajectory(
        self,
        x_next: Tensor,
        source_set_representation: Tensor,
        num_time_steps: int,
        device: torch.device,
        grid: Tensor,
    ) -> Tensor:
        """
        Method to calculate the positions of the newly predicted atoms with flow matching.

        Args:
            x_next (Tensor): The atom types of the newly predicted atoms. shape: [n_atoms, 1]
            source_set_representation (Tensor): The representation of the source sets. shape: [n_atoms, n_embd]
            num_time_steps (int): The number of time steps to use for the flow matching.
            device (torch.device): The device to use for computations.
            integration_method (str): The integration method to use for the flow matching.

        Returns:
            Tensor: The positions of the newly predicted atoms. shape: [n_atoms, 3]
        """
        # (1) Initialize next atoms' position with a random position
        pos_next = self.hparams.noise_std * torch.randn(
            x_next.shape[0], 3, device=device
        )
        pos_trajectory = []
        grid_trajectory = []

        for i, time_step in enumerate(
            torch.linspace(0, 1, num_time_steps, device=device)[:-1]
        ):
            time_step = time_step.expand(x_next.shape[0])
            dt = 1 / num_time_steps
            velocity = self.compute_vector_field(
                x_next,
                pos_next,
                time_step,
                source_set_representation,
                device=device,
            )
            if i % 5 == 0:
                grid_velocity = self.compute_vector_field(
                    x_next.expand(grid.shape[0]),
                    grid,
                    time_step.expand(grid.shape[0]),
                    source_set_representation.expand(grid.shape[0], -1),
                    device=device,
                )
                grid_trajectory.append(grid_velocity.unsqueeze(0))
                pos_trajectory.append(pos_next)
            delta_pos = dt * velocity
            pos_next = pos_next + delta_pos

        pos_trajectory = torch.cat(pos_trajectory, dim=0)
        grid_trajectory = torch.cat(grid_trajectory, dim=0)

        return pos_next, pos_trajectory, grid_trajectory
