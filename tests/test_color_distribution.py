import unittest
import torch
from appearance.color_distribution import distribution_loss


class ColorDistributionTests(unittest.TestCase):
    def test_unpaired_permutation_is_invariant(self):
        torch.manual_seed(3)
        rgb = torch.rand(64, 3)
        directions = torch.nn.functional.normalize(torch.randn(12, 3), dim=-1)
        self.assertLess(float(distribution_loss(rgb, rgb.flip(0), directions)), 1e-12)

    def test_gradient_reduces_color_mismatch(self):
        fake = torch.full((32, 3), .2, requires_grad=True)
        real = torch.full((32, 3), .6)
        directions = torch.eye(3)
        before = distribution_loss(fake, real, directions)
        before.backward()
        self.assertTrue(torch.isfinite(fake.grad).all())
        self.assertGreater(float(fake.grad.norm()), 0)
        after = distribution_loss(fake.detach() - fake.grad, real, directions)
        self.assertLess(float(after), float(before))

    def test_different_sample_sizes_rejected(self):
        with self.assertRaises(ValueError):
            distribution_loss(torch.zeros(3, 3), torch.zeros(4, 3), torch.eye(3))


if __name__ == '__main__':
    unittest.main()
