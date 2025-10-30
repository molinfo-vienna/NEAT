# https://github.com/LTH14/mar/blob/fe470ac24afbee924668d8c5c83e9fec60af3a73/models/diffloss.py
# https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/networks.py
# https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/loss.py

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/layers/swiglu_ffn.py
class SwiGLUFFN(nn.Module):
    def __init__(self, channels, bias):
        super().__init__()
        self.w12 = nn.Linear(channels, 2 * channels * 2, bias=bias)
        self.w3 = nn.Linear(2 * channels, channels, bias=bias)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


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

    def __init__(self, channels, mlp_type="mlp", expand=4):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)

        if mlp_type == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(channels, expand * channels, bias=True),
                nn.SiLU(),
                nn.Linear(expand * channels, channels, bias=True),
            )
        elif mlp_type == "swiglu":  # expand=2
            self.mlp = SwiGLUFFN(channels, bias=True)

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
        config,
        channels=3,
    ):
        super().__init__()

        self.time_embed = MPFourier(config.diff_w)
        self.cond_embed = nn.Linear(config.n_embd, config.diff_w)

        self.input_proj = nn.Linear(channels, config.diff_w)
        if config.diff_fourier > 0:
            self.embed_fourier = FourierCoords(
                in_features=3,
                num_channels=config.diff_fourier,
                out_features=config.diff_w,
                bandwidth=config.coord_bandwidth,
            )
        else:
            self.embed_fourier = lambda x: torch.zeros(
                *x.shape[:-1], config.diff_w, device=x.device, dtype=x.dtype
            )

        res_blocks = []
        for i in range(config.diff_d):
            res_blocks.append(
                ResBlock(
                    config.diff_w,
                    mlp_type=config.diff_mlp,
                    expand=config.diff_mlp_expand,
                )
            )

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(config.diff_w, channels)

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
