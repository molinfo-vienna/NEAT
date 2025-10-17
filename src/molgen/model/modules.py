"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

from abc import ABC

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
    def __init__(self, n_embd, n_head, dropout, bias):
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

    def forward(self, x, attn_mask=None):
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


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, bias):
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

    def forward(self, x):
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

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
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
    def __init__(self, n_embd, n_head, dropout, bias):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = MaskedBidirectionalAttention(n_embd, n_head, dropout, bias)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, dropout, bias)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class PositionalEncodingMixin(ABC):
    def sinusoidal_positional_encoding(
        self, positions, pos_embedding_dim, frequency: float = 10000.0
    ):
        assert pos_embedding_dim % 2 == 0, "Embedding dimension must be even."

        n_nodes, coord_dim = positions.shape
        assert coord_dim == 3, "Input positions must have shape (n_nodes, 3)."

        # Ensure the embedding dimension is divisible by 3
        if pos_embedding_dim % 3 != 0:
            raise ValueError(
                f"Embedding dimension ({pos_embedding_dim}) must be divisible by 3 for x, y, z coordinates."
            )

        if pos_embedding_dim == 0:
            return torch.empty((n_nodes, 0), device=positions.device)

        # Create a range of frequencies for the sinusoidal encoding
        div_term = torch.exp(
            torch.arange(0, pos_embedding_dim // 2, device=positions.device)
            * -(torch.log(torch.tensor(frequency)) / (pos_embedding_dim // 2))
        )

        # Initialize the embedding matrix
        embeddings = torch.zeros((n_nodes, pos_embedding_dim), device=positions.device)

        # Number of dimensions allocated to each coordinate
        coord_embedding_dim = pos_embedding_dim // 3

        # Apply sinusoidal encoding for each coordinate (x, y, z)
        for i in range(3):  # Loop over the 3 coordinates
            pos = positions[:, i][:, torch.newaxis]  # Shape: (n_nodes, 1)
            sinusoidal = pos * div_term[: coord_embedding_dim // 2]  # Match the size
            embeddings[
                :,
                i * coord_embedding_dim : (i + 1) * coord_embedding_dim,
            ] = torch.cat([torch.sin(sinusoidal), torch.cos(sinusoidal)], axis=-1)

        return embeddings


class SinusoidalPositionalEncoding(torch.nn.Module, PositionalEncodingMixin):
    def __init__(self, out_dim: int, num_freq: int = 256):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.num_freq = num_freq
        self.out_dim = out_dim
        self.mlp = torch.nn.Linear(3 * num_freq, out_dim)

    def forward(self, positions):
        sinusoidal_encoding = self.sinusoidal_positional_encoding(
            positions, self.num_freq * 3
        )
        return self.mlp(sinusoidal_encoding)


def pad_and_mask_sequences(sequences, batch_indices):
    """
    Pad sequences to the maximum length in the batch and create an attention mask without using a for loop.
    Args:
        sequences (torch.Tensor): A tensor of shape (n_atoms, embedding_dim).
        batch_indices (torch.Tensor): A tensor indicating the batch index for each atom (shape: (n_atoms,)).
    Returns:
        padded_sequences (torch.Tensor): A tensor of shape (batch_size, max_seq_length, embedding_dim).
        attention_mask (torch.Tensor): A binary mask of shape (batch_size, max_seq_length).
    """
    # Determine the batch size and maximum sequence length
    batch_size = batch_indices.max().item() + 1
    max_seq_length = torch.bincount(batch_indices).max().item()

    # Create a padded tensor for sequences
    padded_sequences = torch.zeros(
        batch_size, max_seq_length, sequences.size(-1), device=sequences.device
    )

    # Create an attention mask
    attention_mask = torch.zeros(
        batch_size, max_seq_length, dtype=torch.bool, device=sequences.device
    )

    # Compute the indices for padding
    seq_lengths = torch.bincount(batch_indices)  # Number of atoms per molecule
    if seq_lengths.size(0) < batch_size:
        padding = torch.zeros(
            batch_size - seq_lengths.size(0),
            device=batch_indices.device,
            dtype=seq_lengths.dtype,
        )
        seq_lengths = torch.cat([seq_lengths, padding])
    cum_lengths = torch.cat(
        [torch.tensor([0], device=sequences.device), seq_lengths.cumsum(0)]
    )  # Cumulative lengths

    # Use advanced indexing to fill the padded tensor and mask
    idx = torch.arange(
        batch_indices.size(0), device=sequences.device
    )  # Indices for all atoms
    batch_offsets = batch_indices * max_seq_length  # Offset for each batch
    flat_indices = batch_offsets + (
        idx - cum_lengths[batch_indices]
    )  # Flattened indices for padded tensor

    # Flatten padded_sequences and attention_mask for efficient assignment
    flat_padded_sequences = padded_sequences.view(-1, sequences.size(-1))
    flat_attention_mask = attention_mask.view(-1)

    # Assign values to the flattened tensors
    flat_padded_sequences[flat_indices] = sequences
    flat_attention_mask[flat_indices] = 1

    # Reshape back to the original padded shape
    padded_sequences = flat_padded_sequences.view(
        batch_size, max_seq_length, sequences.size(-1)
    )
    attention_mask = flat_attention_mask.view(batch_size, max_seq_length)

    return padded_sequences, attention_mask
