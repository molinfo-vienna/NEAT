"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MaskedBidirectionalAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, bias, pos_embedder=None, qk_norm=False):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.pos_embedder = pos_embedder
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.RMSNorm(n_embd // n_head, eps=1e-8)
            self.k_norm = nn.RMSNorm(n_embd // n_head, eps=1e-8)

    def forward(self, x, attn_mask=None, pos=None):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # Create the right mask dimensions
        attn_mask = attn_mask.unsqueeze(1) * attn_mask.unsqueeze(2)
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.n_head, -1, -1)

        # This was the previous approach.
        # I leave this here for reference, but we can delete it later.
        # attn_mask = attn_mask.unsqueeze(1).unsqueeze(
        #     2
        # )  # [n_molecules, 1, 1, max_atom_count]
        # attn_mask = attn_mask.expand(
        #     -1, self.hparams.n_head, -1, -1
        # )  # [n_molecules, n_head, 1, max_atom_count]

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        if self.pos_embedder and pos is not None:
            q, k = self.pos_embedder(q, k, pos)

        # Apply scaled dot-product attention with the provided attention mask
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,  # Pass the attention mask here
            dropout_p=self.dropout if self.training else 0,
            is_causal=False,  # Bidirectional attention, not causal
        )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class AttentionPooling(nn.Module):
    def __init__(self, n_embd, n_head, dropout, bias):
        super().__init__()
        assert (
            n_embd % n_head == 0
        ), "Embedding dimension must be divisible by the number of heads"

        # Learnable query vector (1 x 1 x n_embd)
        self.query = nn.Parameter(torch.randn(1, 1, n_embd))

        # Key, query, value projections for all heads
        self.c_attn = nn.Linear(n_embd, 2 * n_embd, bias=bias)

        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, C)
                - B: Batch size
                - T: Sequence length (number of atoms)
                - C: Embedding dimension (n_embd)
        Returns:
            pooled_output: Tensor of shape (B, C)
                - Aggregated molecular representation for each molecule in the batch
        """
        B, T, C = (
            x.size()
        )  # Batch size, sequence length, embedding dimensionality (n_embd)

        # Expand the learnable query vector to match the batch size
        query = self.query.expand(B, -1, -1)  # Shape: (B, 1, C)

        # Calculate query, key, values for all heads
        k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = query.view(B, 1, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, 1, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # Apply scaled dot-product attention
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,  # No mask needed for pooling
            dropout_p=self.dropout if self.training else 0,
            is_causal=False,  # Not causal, as this is pooling
        )

        # Reshape the output back to (B, 1, C)
        y = y.transpose(1, 2).contiguous().view(B, 1, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))

        # Remove the singleton dimension (B, 1, C) -> (B, C)
        pooled_output = y.squeeze(1)

        return pooled_output


class MLP(nn.Module):
    def __init__(self, n_embd, dropout, bias):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, bias, pos_embedder=None, qk_norm=False):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = MaskedBidirectionalAttention(
            n_embd, n_head, dropout, bias, pos_embedder, qk_norm=qk_norm
        )
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, dropout, bias)

    def forward(self, x, attn_mask=None, pos=None):
        x = x + self.attn(self.ln_1(x), attn_mask=attn_mask, pos=pos)
        x = x + self.mlp(self.ln_2(x))
        return x
