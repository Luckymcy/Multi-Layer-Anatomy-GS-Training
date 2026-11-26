import unittest
import torch
from appearance.gaussian_rgb import GaussianRGB


class GaussianRGBTests(unittest.TestCase):
    def test_equal_hu_can_learn_different_colors_and_unseen_stays_fixed(self):
        model = GaussianRGB(torch.full((3, 3), .5))
        hu, rows = torch.zeros(3), torch.zeros(3, dtype=torch.long)
        optimizer = torch.optim.Adam(model.parameters(), lr=.1)
        colors = model(hu, rows)
        loss = (colors[0, 0]-1).square() + colors[1, 0].square()
        loss.backward()
        self.assertTrue(torch.equal(model.logits.grad[2], torch.zeros(3)))
        optimizer.step()
        self.assertGreater(float(model.rgb[0, 0]), float(model.rgb[1, 0]))
        torch.testing.assert_close(model.rgb[2], torch.full((3,), .5), rtol=0, atol=0)
        self.assertTrue(((model.rgb >= 0) & (model.rgb <= 1)).all())

    def test_state_roundtrip_and_count_validation(self):
        model = GaussianRGB(torch.rand(7, 3))
        restored = GaussianRGB(torch.full((7, 3), .5))
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(model.rgb, restored.rgb, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            model(torch.zeros(6), torch.zeros(6))
        for bad in [torch.zeros(0, 3), torch.zeros(3), torch.full((2, 3), float('nan')),
                    torch.full((2, 3), 1.1)]:
            with self.assertRaises(ValueError):
                GaussianRGB(bad)


if __name__ == '__main__':
    unittest.main()
