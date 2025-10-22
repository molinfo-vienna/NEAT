import torch


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
