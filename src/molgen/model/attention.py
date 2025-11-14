"""
Taken and modified from the nanoGPT repository:
https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

from typing import Optional

import torch
import torch.nn as nn


class MaskedBidirectionalAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
        bias: bool,
        bias_zero: bool,
        pos_embedder: Optional[nn.Module] = None,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias_zero)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.pos_embedder = pos_embedder

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
    """
    A transformer block with masked bidirectional attention.
    Args:
        n_embd (int): The number of embedding dimensions.
        n_head (int): The number of attention heads.
        dropout (float): The dropout rate.
        bias (bool): Whether to use bias in the layers.
        pos_embedder (Optional[nn.Module]): The positional embedder to use. Relates to rope embeddings.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
        bias: bool,
        bias_zero: bool,
        pos_embedder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, bias=bias_zero)
        self.attn = MaskedBidirectionalAttention(
            n_embd, n_head, dropout, bias, bias_zero, pos_embedder
        )
        self.ln_2 = nn.LayerNorm(n_embd, bias=bias_zero)
        self.mlp = MLP(n_embd, dropout, bias_zero)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
    ):
        x = x + self.attn(self.ln_1(x), attn_mask=attn_mask, pos=pos)
        x = x + self.mlp(self.ln_2(x))
        return x
