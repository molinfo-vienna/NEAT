"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

from abc import ABC
import math

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
    def __init__(self, n_embd, n_head, dropout, bias, pos_embedder=None):
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
        self.pos_embedder = pos_embedder  # Placeholder for positional embedding module

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
    def __init__(self, n_embd, n_head, dropout, bias, pos_embedder=None):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = MaskedBidirectionalAttention(
            n_embd, n_head, dropout, bias, pos_embedder
        )
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, dropout, bias)

    def forward(self, x, attn_mask=None, pos=None):
        x = x + self.attn(self.ln_1(x), attn_mask=attn_mask, pos=pos)
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


def create_time_embeddings(timesteps, embedding_dim):
    """
    Creates sinusoidal time embeddings for a batch of timesteps.

    Args:
        timesteps (torch.Tensor): A tensor of shape (n,) containing timesteps in the range [0, 1].
        embedding_dim (int): The dimensionality of the time embeddings (must be even).

    Returns:
        torch.Tensor: A tensor of shape (n, embedding_dim) containing the time embeddings.
    """
    assert embedding_dim % 2 == 0, "Embedding dimension must be even."

    # Scale timesteps to the range [0, 1] (if not already scaled)
    timesteps = timesteps.unsqueeze(1)  # Shape: (n, 1)

    # Compute the sinusoidal frequencies
    half_dim = embedding_dim // 2
    frequencies = torch.exp(
        -torch.arange(half_dim, device=timesteps.device, dtype=timesteps.dtype)
        * torch.log(torch.tensor(10000.0))
        / half_dim
    )  # Shape: (half_dim,)

    # Compute the sinusoidal embeddings
    angular_frequencies = timesteps * frequencies  # Shape: (n, half_dim)
    sin_embeddings = torch.sin(angular_frequencies)  # Shape: (n, half_dim)
    cos_embeddings = torch.cos(angular_frequencies)  # Shape: (n, half_dim)

    # Concatenate sine and cosine embeddings
    time_embeddings = torch.cat(
        [sin_embeddings, cos_embeddings], dim=1
    )  # Shape: (n, embedding_dim)

    return time_embeddings


def quaternions_to_rotation_matrices(quaternions):
    """
    Converts a batch of quaternions to a batch of 3x3 rotation matrices.

    Args:
        quaternions (torch.Tensor): A tensor of shape (batch_size, 4) representing a batch of quaternions [q_w, q_x, q_y, q_z].
                                    The quaternions are assumed to be normalized.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, 3, 3) containing the rotation matrices.
    """
    # Ensure the quaternions are normalized
    quaternions = quaternions / quaternions.norm(dim=1, keepdim=True)

    # Extract components of the quaternions
    q_w, q_x, q_y, q_z = (
        quaternions[:, 0],
        quaternions[:, 1],
        quaternions[:, 2],
        quaternions[:, 3],
    )

    # Compute the rotation matrices
    rotation_matrices = torch.stack(
        [
            torch.stack(
                [
                    1 - 2 * (q_y**2 + q_z**2),
                    2 * (q_x * q_y - q_z * q_w),
                    2 * (q_x * q_z + q_y * q_w),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2 * (q_x * q_y + q_z * q_w),
                    1 - 2 * (q_x**2 + q_z**2),
                    2 * (q_y * q_z - q_x * q_w),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2 * (q_x * q_z - q_y * q_w),
                    2 * (q_y * q_z + q_x * q_w),
                    1 - 2 * (q_x**2 + q_y**2),
                ],
                dim=-1,
            ),
        ],
        dim=1,
    )  # Shape: (batch_size, 3, 3)

    return rotation_matrices


def compute_graph_centers(positions, batch_idx):
    """
    Computes the center (mean position) of each graph in the batch.

    Args:
        positions (torch.Tensor): A tensor of shape (num_nodes, 3) containing the 3D positions of all nodes.
        batch_idx (torch.Tensor): A tensor of shape (num_nodes,) indicating the graph index for each node.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, 3) containing the center of each graph.
    """
    # Sum the positions for each graph
    graph_sums = torch.zeros(
        batch_idx.max() + 1, 3, device=positions.device, dtype=positions.dtype
    )
    graph_counts = torch.zeros(
        batch_idx.max() + 1, device=positions.device, dtype=positions.dtype
    )

    graph_sums.index_add_(0, batch_idx, positions)  # Sum positions for each graph
    graph_counts.index_add_(
        0, batch_idx, torch.ones_like(batch_idx, dtype=positions.dtype)
    )  # Count nodes per graph

    # Compute the mean (center) for each graph
    graph_centers = graph_sums / graph_counts.unsqueeze(1)  # Shape: (batch_size, 3)

    return graph_centers


def generate_uniform_quaternions(batch_size, device):
    """
    Generate uniformly sampled unit quaternions.

    Args:
        batch_size (int): Number of quaternions to generate.
        device (torch.device): Device for computations (e.g., 'cpu' or 'cuda').

    Returns:
        torch.Tensor: Tensor of shape (batch_size, 4) containing unit quaternions.
    """
    N = batch_size

    # Step 1: Generate random rotation axes
    random_axes = torch.randn(N, 3, device=device)
    # Add a small epsilon to the norm to avoid division by zero
    epsilon = 1e-8
    random_axes = random_axes / (
        random_axes.norm(dim=1, keepdim=True) + epsilon
    )  # Normalize to unit vectors

    # Step 2: Generate random rotation angles
    random_angles = (torch.rand(N, device=device) * (math.pi - 0.1)) + (0.05 * math.pi)

    # Step 3: Construct random quaternions
    half_angles = random_angles / 2
    qw = torch.cos(half_angles)
    qxyz = random_axes * torch.sin(half_angles).unsqueeze(1)
    random_quaternions = torch.cat([qw.unsqueeze(1), qxyz], dim=1)  # Shape (N, 4)

    return random_quaternions


def draw_random_quaternions(batch_size, device):
    # Generate random rotation quaternions for each graph
    rot = generate_uniform_quaternions(batch_size, device)
    rot = rot / rot.norm(dim=1, keepdim=True)
    mask = rot[:, 0] < 0  # Check where w < 0
    rot[mask] = -rot[mask]  # Flip the sign of the entire quaternion

    return rot


def rotate_graphs_randomly(positions, batch_idx):
    """
    Rotates and translates the positions of nodes in a batch of PyG graphs w.r.t. their centers.

    Args:
        positions (torch.Tensor): A tensor of shape (num_nodes, 3) containing the 3D positions of all nodes.
        batch_idx (torch.Tensor): A tensor of shape (num_nodes,) indicating the graph index for each node.
        quaternions (torch.Tensor): A tensor of shape (batch_size, 4) containing the rotation quaternions for each graph.
        translation_vectors (torch.Tensor): A tensor of shape (batch_size, 3) containing the translation vectors for each graph.

    Returns:
        torch.Tensor: A tensor of shape (num_nodes, 3) containing the rotated and translated positions.
    """
    # Step 0: Draw random quaternions for each graph
    batch_size = batch_idx.max().item() + 1
    device = positions.device
    rot_operator = draw_random_quaternions(batch_size, device)

    # Step 1: Convert quaternions to rotation matrices
    rot_operator = quaternions_to_rotation_matrices(
        rot_operator.clone()
    )  # Shape: (batch_size, 3, 3)

    # Step 2: Compute the center of each graph
    graph_centers = compute_graph_centers(
        positions, batch_idx
    )  # Shape: (batch_size, 3)

    # Step 3: Center the positions (subtract the graph center)
    centered_positions = positions - graph_centers[batch_idx]  # Shape: (num_nodes, 3)

    # Step 4: Apply rotation
    node_rotation_matrices = rot_operator[batch_idx]  # Shape: (num_nodes, 3, 3)
    rotated_positions = torch.bmm(
        node_rotation_matrices, centered_positions.unsqueeze(-1)
    ).squeeze(
        -1
    )  # Shape: (num_nodes, 3)

    # Step 5: Recenter the positions (add the graph center back)
    recentered_positions = (
        rotated_positions + graph_centers[batch_idx]
    )  # Shape: (num_nodes, 3)

    return recentered_positions
