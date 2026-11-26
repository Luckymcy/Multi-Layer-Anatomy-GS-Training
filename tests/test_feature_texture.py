import unittest
import torch
from pathlib import Path
from appearance.feature_texture import MaskedFeatureSW


class MaskedFeaturesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path.home()/'.cache/torch/hub/checkpoints/vgg16-397923af.pth'
        if not path.exists(): raise unittest.SkipTest('Local pretrained VGG16 unavailable')
        cls.model=MaskedFeatureSW(path)

    def test_no_gradient_outside_valid_tissue(self):
        image=torch.rand(3,32,32,requires_grad=True)
        mask=torch.zeros(32,32,dtype=torch.bool);mask[2:30,2:30]=True
        tokens=self.model.tokens(image,mask)
        sum(v.square().mean() for v in tokens.values()).backward()
        self.assertGreater(float(image.grad.norm()),0)
        self.assertEqual(float(image.grad[:,~mask].abs().max()),0)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_empty_mask_has_no_tokens(self):
        self.assertFalse(self.model.tokens(torch.rand(3,32,32),torch.zeros(32,32,dtype=torch.bool)))
