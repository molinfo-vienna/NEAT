from typing import Union
import inspect
import math

import torch
from torch import Tensor
from lightning import LightningModule
from torch_geometric.data import Data
from torch.optim import Optimizer, AdamW
from torch.nn import functional as F

from .decoder import Decoder
from .modules import LayerNorm, MLP, Block


class MolGen(LightningModule):
    def __init__(self):
        super(MolGen, self).__init__()
        self.save_hyperparameters()
        # self.hparams.setdefault("key", "value")

        # Atom type embedding layer
        self.wte = torch.nn.Embedding(self.vocab_size, self.n_embd)

        # Fourier features for embedding of Cartesian coordinates
        self.fourier_features = torch.nn.Identity()

        # A linear layer for projecting the Cartesian coordinates (additional to the Fourier features)
        self.coord_proj = torch.nn.Identity()

        # Positional embedding layer for sequences (only important for causal transformer)
        self.wpe = torch.nn.Embedding(self.block_size, self.n_embd)

        # Dropout layer
        self.drop = torch.nn.Dropout(self.dropout)

        # Transformer blocks
        self.h = (
            torch.nn.ModuleList(
                [
                    Block(self.n_embd, self.n_head, self.dropout, self.bias)
                    for _ in range(self.n_layer)
                ]
            ),
        )
        self.ln_f = LayerNorm(self.n_embd, bias=self.bias)

        # Linear layer to map the final embeddings to atom vocabulary logits
        self.lm_head = torch.nn.Linear(self.n_embd, self.vocab_size, bias=False)

        # The atom types are supervised with a cross-entropy loss
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

        # Weight tying
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.wte.weight = self.lm_head.weight

        # So far it is the GPT logic, here comes additional stuff

        # A second transformer block
        self.h2 = (
            torch.nn.ModuleList(
                [
                    Block(self.n_embd, self.n_head, self.dropout, self.bias)
                    for _ in range(self.n_layer)
                ]
            ),
        )
        self.ln_f2 = LayerNorm(self.n_embd, bias=self.bias)

        # A simple MLP with layer norm used for the denoising step
        self.flow_matching_mlp = torch.nn.Identity()

        # Define loss functions here
        # self.fm_loss = FlowMatchingLoss()

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layer)
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
            n_params -= self.wpe.weight.numel()
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
        batch_size, sequence_length = data.x.size()
        assert (
            sequence_length <= self.config.block_size
        ), f"Cannot forward sequence of length {sequence_length}, block size is only {self.config.block_size}"
        sequence_position = torch.arange(
            0, sequence_length, dtype=torch.long, device=device
        )  # shape (t)

        # forward the GPT model itself
        tok_emb = self.wte(data.x)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(
            sequence_position
        )  # position embeddings of shape (t, n_embd)
        x = self.drop(tok_emb + pos_emb)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def data_augmentation(self, data: Data) -> Data:
        # Implement data augmentation logic here
        return data

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
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
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

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
