"""Independent, view-invariant RGB per selected Gaussian, with frozen geometry."""
import torch
from torch import nn


class GaussianRGB(nn.Module):
    def __init__(self, initial_rgb):
        super().__init__()
        if initial_rgb.ndim != 2 or initial_rgb.shape[1] != 3 or not len(initial_rgb):
            raise ValueError('Expected nonempty [Gaussian, RGB] initialization')
        if not torch.isfinite(initial_rgb).all() or (initial_rgb < 0).any() or (initial_rgb > 1).any():
            raise ValueError('Expected finite linear RGB in [0,1]')
        self.logits = nn.Parameter(torch.logit(initial_rgb.detach().clone().clamp(1e-6, 1-1e-6)))

    @property
    def rgb(self):
        return self.logits.sigmoid()

    def forward(self, hu, tissue_rows):
        # HU never determines learned color. Input order is the fixed selected PLY order.
        if len(hu) != len(self.logits) or len(tissue_rows) != len(self.logits):
            raise ValueError('Selected Gaussian count/order contract violated')
        return self.rgb

    def smoothness(self):
        # Baseline has no spatial regularizer. PLY adjacency is NOT surface adjacency.
        return self.logits.new_zeros(())
