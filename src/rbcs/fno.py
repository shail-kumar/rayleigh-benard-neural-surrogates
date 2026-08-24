"""Fourier Neural Operator for one-step field prediction.

Input and output are (batch, 2, nx, nz): temperature perturbation and vorticity.
"""

import torch
import torch.nn.functional as F
from torch import nn


class SpectralConv2d(nn.Module):
    """Truncated spectral convolution: rfft2, keep the lowest modes, apply a
    learned complex multiplier, invert."""

    def __init__(self, in_ch, out_ch, modes_x, modes_z):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.modes_x, self.modes_z = modes_x, modes_z
        scale = 1.0 / (in_ch * out_ch)
        # Two blocks: rfft2 halves the last axis but keeps both signs on the
        # first, so positive and negative x wavenumbers need separate weights.
        self.w_pos = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes_x, modes_z, dtype=torch.cfloat)
        )
        self.w_neg = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes_x, modes_z, dtype=torch.cfloat)
        )

    def forward(self, x):
        b, _, nx, nz = x.shape
        xf = torch.fft.rfft2(x, norm="ortho")
        out = torch.zeros(
            b, self.out_ch, nx, nz // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        mx, mz = min(self.modes_x, nx // 2), min(self.modes_z, nz // 2 + 1)
        out[:, :, :mx, :mz] = torch.einsum(
            "bixz,ioxz->boxz", xf[:, :, :mx, :mz], self.w_pos[:, :, :mx, :mz]
        )
        out[:, :, -mx:, :mz] = torch.einsum(
            "bixz,ioxz->boxz", xf[:, :, -mx:, :mz], self.w_neg[:, :, :mx, :mz]
        )
        return torch.fft.irfft2(out, s=(nx, nz), norm="ortho")


class FNOBlock(nn.Module):
    """Spectral convolution plus a pointwise bypass."""

    def __init__(self, width, modes_x, modes_z):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_x, modes_z)
        self.pointwise = nn.Conv2d(width, width, 1)

    def forward(self, x):
        return F.gelu(self.spectral(x) + self.pointwise(x))


class FNO2d(nn.Module):
    """Maps one field to the next.

    Predicts the increment rather than the state, so zero output reproduces
    persistence and training starts from that baseline.

    A z-coordinate channel is appended but an x-coordinate one is not: the
    walls make z inhomogeneous, while x is periodic and should stay
    translation-equivariant, matching the shift augmentation used in training.
    """

    def __init__(
        self,
        channels=2,
        width=20,
        modes_x=16,
        modes_z=12,
        depth=4,
        z_pad=8,
        residual=True,
    ):
        super().__init__()
        self.z_pad, self.residual = z_pad, residual
        self.lift = nn.Conv2d(channels + 1, width, 1)
        self.blocks = nn.ModuleList(FNOBlock(width, modes_x, modes_z) for _ in range(depth))
        self.project = nn.Sequential(
            nn.Conv2d(width, 2 * width, 1), nn.GELU(), nn.Conv2d(2 * width, channels, 1)
        )
        # Per-channel normalisation; theta and omega differ by ~100x in scale.
        self.register_buffer("mean", torch.zeros(1, channels, 1, 1))
        self.register_buffer("std", torch.ones(1, channels, 1, 1))

    def set_normalisation(self, mean, std):
        self.mean.copy_(torch.as_tensor(mean).reshape_as(self.mean))
        self.std.copy_(torch.as_tensor(std).reshape_as(self.std))

    def _with_z(self, x):
        b, _, nx, nz = x.shape
        z = torch.linspace(0.0, 1.0, nz, device=x.device, dtype=x.dtype)
        return torch.cat([x, z.view(1, 1, 1, nz).expand(b, 1, nx, nz)], dim=1)

    def forward(self, x):
        xn = (x - self.mean) / self.std
        h = self._with_z(xn)
        # z is bounded by walls, not periodic, so pad before the FFT and crop
        # after. x is periodic and needs no padding.
        if self.z_pad:
            h = F.pad(h, (0, self.z_pad))
        h = self.lift(h)
        for block in self.blocks:
            h = block(h)
        h = self.project(h)
        if self.z_pad:
            h = h[..., : -self.z_pad]
        return x + h * self.std if self.residual else h * self.std + self.mean
