"""
Taken from
https://github.com/LTH14/mar/blob/fe470ac24afbee924668d8c5c83e9fec60af3a73/models/diffloss.py
with modifications following
https://github.com/aspuru-guzik-group/quetzal/blob/main/simple_mlp.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class FourierCoords(nn.Module):
    def __init__(self, in_features, num_channels, out_features, bandwidth=1):
        super().__init__()

        self.fourier = MPFourier(num_channels, bandwidth=bandwidth)
        self.final_linear = nn.Linear(num_channels * in_features, out_features)

    def forward(self, coords):
        coords = self.fourier(coords.unsqueeze(-1))
        coords = coords.flatten(-2)
        return self.final_linear(coords)


# https://github.com/NVlabs/edm2/blob/4bf8162f601bcc09472ce8a32dd0cbe8889dc8fc/training/networks_edm2.py#L73
class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y * self.freqs.to(torch.float32)
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """

    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            model_channels, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    """

    def __init__(
        self,
        model_channels,
        condition_channels,
        fourier_features_channels=512,
        fourier_features_bandwidth=20.0,
        n_layer_mlp=6,
        coordinate_channels=3,
    ):
        super().__init__()

        self.time_embed = MPFourier(model_channels)
        self.cond_embed = nn.Linear(condition_channels, model_channels)

        self.input_proj = nn.Linear(coordinate_channels, model_channels)
        if fourier_features_channels > 0:
            self.embed_fourier = FourierCoords(
                in_features=3,
                num_channels=fourier_features_channels,
                out_features=model_channels,
                bandwidth=fourier_features_bandwidth,
            )
        else:
            self.embed_fourier = lambda x: torch.zeros(
                *x.shape[:-1], model_channels, device=x.device, dtype=x.dtype
            )

        res_blocks = []
        for _ in range(n_layer_mlp):
            res_blocks.append(
                ResBlock(
                    model_channels,
                )
            )

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, coordinate_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x) + self.embed_fourier(x)
        t = self.time_embed(t.unsqueeze(1)).reshape(x.shape)
        c = self.cond_embed(c)

        y = t + c

        for block in self.res_blocks:
            x = block(x, y)

        return self.final_layer(x, y)
