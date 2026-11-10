"""Small tissue-conditioned discriminator on linear-RGB patches."""
import math
import torch
from torch import nn


class TissuePatchDiscriminator(nn.Module):
    def __init__(self, tissues=1, channels=32):
        super().__init__()
        if tissues < 1 or channels < 4:
            raise ValueError('Invalid discriminator dimensions')
        layers, source = [], 3
        for target in [channels, channels*2, channels*4, channels*4]:
            layers.extend([nn.Conv2d(source, target, 4, 2, 1), nn.LeakyReLU(.2)])
            source = target
        # No batch statistics, dropout or mutable spectral-norm state during G steps.
        self.features = nn.Sequential(*layers)
        self.score = nn.Linear(source, 1)
        self.condition = nn.Embedding(tissues, source)

    def forward(self, linear_rgb, tissue):
        if linear_rgb.ndim != 4 or linear_rgb.shape[1] != 3 or min(linear_rgb.shape[-2:]) < 32:
            raise ValueError('Expected BCHW RGB patches at least 32px')
        if tissue.shape != (len(linear_rgb),) or tissue.dtype != torch.long:
            raise ValueError('Expected one int64 tissue row per patch')
        feature = self.features(linear_rgb*2-1).mean((-2, -1))
        return self.score(feature).squeeze(1) + (feature*self.condition(tissue)).sum(1)/math.sqrt(feature.shape[1])
