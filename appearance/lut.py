"""Tissue-conditional, fixed-HU-grid RGB appearance (linear-light output)."""
import torch
from torch import nn


def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).pow(2.4))


def linear_to_srgb(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, 12.92 * x,
                       1.055 * x.clamp_min(1e-12).pow(1 / 2.4) - 0.055)


class TissueRGBLUT(nn.Module):
    """IDs are contiguous LUT row indices, not anatomical layer numbers.

    HU bounds and uniformly spaced knot locations are fixed buffers. Every
    RGB knot is an independent sigmoid-parameterized learnable value.
    """

    def __init__(self, initial_rgb, hu_bounds):
        super().__init__()
        initial_rgb = torch.as_tensor(initial_rgb, dtype=torch.float32)
        bounds = torch.as_tensor(hu_bounds, dtype=torch.float32)
        if initial_rgb.ndim != 3 or initial_rgb.shape[1] < 3 or initial_rgb.shape[2] != 3:
            raise ValueError('Expected [tissues, >=3 knots, RGB]')
        if bounds.shape != (initial_rgb.shape[0], 2):
            raise ValueError('Expected HU bounds [tissues, 2]')
        if not torch.isfinite(bounds).all() or not (bounds[:, 1] > bounds[:, 0]).all():
            raise ValueError('HU bounds must be finite and increasing')
        if not torch.isfinite(initial_rgb).all() or ((initial_rgb < 0) | (initial_rgb > 1)).any():
            raise ValueError('Initial linear RGB must be finite and in [0, 1]')
        self.register_buffer('hu_bounds', bounds.clone())
        self.logits = nn.Parameter(torch.logit(initial_rgb.clamp(1e-6, 1 - 1e-6)))

    @property
    def rgb(self):
        return self.logits.sigmoid()

    @property
    def hu_knots(self):
        u = torch.linspace(0, 1, self.logits.shape[1], device=self.logits.device)
        return self.hu_bounds[:, :1] + u * (self.hu_bounds[:, 1:] - self.hu_bounds[:, :1])

    def forward(self, hu, tissue):
        if hu.ndim != 1 or tissue.shape != hu.shape or tissue.dtype != torch.long:
            raise ValueError('Expected 1D HU and matching int64 tissue row indices')
        if not torch.isfinite(hu).all():
            raise ValueError('Nonfinite HU')
        if ((tissue < 0) | (tissue >= self.logits.shape[0])).any():
            raise ValueError('Unknown tissue row')
        lo, hi = self.hu_bounds[tissue].unbind(-1)
        u = ((hu - lo) / (hi - lo)).clamp(0, 1) * (self.logits.shape[1] - 1)
        j = u.floor().long().clamp(max=self.logits.shape[1] - 2)
        t = (u - j).unsqueeze(-1)
        rgb = self.rgb
        return (1 - t) * rgb[tissue, j] + t * rgb[tissue, j + 1]

    def smoothness(self):
        rgb = self.rgb
        return (rgb[:, 2:] - 2 * rgb[:, 1:-1] + rgb[:, :-2]).square().sum()


def from_manual_config(config, knots):
    """Resample explicit manual HU/RGB anchors, decoding sRGB exactly once.

    Interpolation of the initialized/learned LUT itself is always linear light.
    The anchor curve can have nonuniform locations; the learnable grid is uniform.
    """
    rows, bounds = [], []
    for spec in config['tissues']:
        lo, hi = spec['hu_range']
        anchors = torch.tensor(spec['anchors'], dtype=torch.float32)
        if anchors.ndim != 2 or anchors.shape[1] != 4 or anchors.shape[0] < 2:
            raise ValueError('Manual anchors must be [HU,R,G,B] rows')
        if not torch.isfinite(anchors).all() or not (anchors[1:, 0] > anchors[:-1, 0]).all():
            raise ValueError('Manual anchors must be finite with increasing HU')
        if ((anchors[:, 1:] < 0) | (anchors[:, 1:] > 1)).any():
            raise ValueError('Invalid manual RGB')
        x = torch.linspace(lo, hi, knots).clamp(anchors[0, 0], anchors[-1, 0])
        j = torch.searchsorted(anchors[:, 0].contiguous(), x, right=True) - 1
        j = j.clamp(0, anchors.shape[0] - 2)
        t = ((x - anchors[j, 0]) / (anchors[j + 1, 0] - anchors[j, 0])).unsqueeze(-1)
        rgb = torch.lerp(anchors[j, 1:], anchors[j + 1, 1:], t)
        if config['anchor_color_space'] == 'srgb':
            rgb = srgb_to_linear(rgb)
        elif config['anchor_color_space'] != 'linear':
            raise ValueError('anchor_color_space must be srgb or linear')
        rows.append(rgb)
        bounds.append([lo, hi])
    return TissueRGBLUT(torch.stack(rows), bounds)
