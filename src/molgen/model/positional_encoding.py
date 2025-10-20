# Credit: https://github.com/apple/ml-simplefold/blob/main/src/simplefold/model/torch/pos_embed.py
# This file was taken from the SimpleFold repository as is.

import math
import torch
from torch import nn


class FourierPositionEncoding(torch.nn.Module):
    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        include_input: bool = True,
        min_freq_log2: float = 0,
        max_freq_log2: float = 12,
        num_freqs: int = 128,
        log_sampling: bool = True,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.include_input = include_input
        self.min_freq_log2 = min_freq_log2
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.create_embedding_fn()
        intermediary_dim = 2 * in_dim * num_freqs
        if include_input:
            intermediary_dim += in_dim
        self.linear = nn.Linear(intermediary_dim, out_dim)

    def create_embedding_fn(self):
        d = self.in_dim
        dim_out = 0
        if self.include_input:
            dim_out += d

        min_freq = self.min_freq_log2
        max_freq = self.max_freq_log2
        N_freqs = self.num_freqs

        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(
                min_freq, max_freq, steps=N_freqs
            )  # (nf,)
        else:
            freq_bands = torch.linspace(
                2.0**min_freq, 2.0**max_freq, steps=N_freqs
            )  # (nf,)

        assert (
            freq_bands.isfinite().all()
        ), f"nan: {freq_bands.isnan().any()} inf: {freq_bands.isinf().any()}"

        self.register_buffer("freq_bands", freq_bands)  # (nf,)
        self.embed_dim = dim_out + d * self.freq_bands.numel() * 2

    def forward(
        self,
        pos: torch.Tensor,
    ):
        """
        Get the positional encoding for each coordinate.
        Args:
            pos:
                (*, in_dim)
        Returns:
            out:
                (*, in_dimitional_encoding)
        """

        out = []
        if self.include_input:
            out = [pos]  # (*, in_dim)

        pos = pos.unsqueeze(-1) * self.freq_bands  # (*b, d, nf)

        out += [
            torch.sin(pos).flatten(start_dim=-2),  # (*b, d*nf)
            torch.cos(pos).flatten(start_dim=-2),  # (*b, d*nf)
        ]

        out = torch.cat(out, dim=-1)  # (*b, 2 * in_dim * nf (+ in_dim))
        return self.linear(out)  # (*, out_dim)


def compute_axial_cis(
    ts: torch.Tensor,
    in_dim: int,
    dim: int,
    theta: float = 100.0,
):
    B, N, D = ts.shape
    freqs_all = []
    interval = 2 * in_dim
    for i in range(in_dim):
        freq = 1.0 / (
            theta ** (torch.arange(0, dim, interval)[: (dim // interval)].float() / dim)
        ).to(ts.device)
        t = ts[..., i].flatten()
        freq_i = torch.outer(t, freq)
        freq_cis_i = torch.polar(torch.ones_like(freq_i), freq_i)
        freq_cis_i = freq_cis_i.view(B, N, -1)
        freqs_all.append(freq_cis_i)
    freqs_cis = torch.cat(freqs_all, dim=-1)
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


class AxialRotaryPositionEncoding(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        in_dim=3,
        base=100.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.embed_dim = embed_dim // num_heads
        self.base = base

    def forward(self, xq, xk, pos):
        """
        xq: [B, H, N, D]
        xk: [B, H, N, D]
        pos: [B, N, in_dim]
        """
        if pos.ndim == 2:
            pos = pos.unsqueeze(-1)
        freqs_cis = compute_axial_cis(pos, self.in_dim, self.embed_dim, self.base)
        freqs_cis = freqs_cis.unsqueeze(1)
        return apply_rotary_emb(xq, xk, freqs_cis.to(xq.device))
