"""CPU mathematical checks; these do not substitute for the CUDA render smoke."""
import io
import unittest
import torch
from appearance.lut import TissueRGBLUT, srgb_to_linear, linear_to_srgb


class LUTTests(unittest.TestCase):
    def setUp(self):
        self.lut = TissueRGBLUT([[[.1, .2, .3], [.4, .5, .6], [.7, .8, .9]],
                                [[.9, .8, .7], [.6, .5, .4], [.3, .2, .1]]],
                               [[-100, 100], [0, 20]])

    def test_endpoints_clamping_and_tissue(self):
        out = self.lut(torch.tensor([-200., -50., 200., 5.]), torch.tensor([0, 0, 0, 1]))
        expected = torch.tensor([[.1, .2, .3], [.25, .35, .45], [.7, .8, .9], [.75, .65, .55]])
        torch.testing.assert_close(out, expected)

    def test_render_independent_finite_difference(self):
        lut = self.lut.double()
        hu, tissue = torch.tensor([-50., 50.], dtype=torch.double), torch.tensor([0, 0])
        def loss():
            return lut(hu, tissue).square().sum()
        loss().backward()
        analytic = lut.logits.grad[0, 1, 1].item()
        with torch.no_grad():
            old = lut.logits[0, 1, 1].item()
            lut.logits[0, 1, 1] = old + 1e-5
            plus = loss().item()
            lut.logits[0, 1, 1] = old - 1e-5
            minus = loss().item()
            lut.logits[0, 1, 1] = old
        self.assertAlmostEqual(analytic, (plus - minus) / 2e-5, places=8)
        self.assertEqual(lut.logits.grad[1].abs().sum().item(), 0)

    def test_smoothness_and_serialization(self):
        self.assertLess(self.lut.smoothness().item(), 1e-12)
        stream = io.BytesIO()
        torch.save(self.lut.state_dict(), stream)
        stream.seek(0)
        self.lut.load_state_dict(torch.load(stream))
        self.assertTrue(((self.lut.rgb > 0) & (self.lut.rgb < 1)).all())

    def test_gamma_roundtrip(self):
        x = torch.linspace(0, 1, 100)
        torch.testing.assert_close(linear_to_srgb(srgb_to_linear(x)), x)

    def test_reject_invalid_inputs(self):
        with self.assertRaises(ValueError):
            self.lut(torch.tensor([float('nan')]), torch.tensor([0]))
        with self.assertRaises(ValueError):
            self.lut(torch.tensor([0.]), torch.tensor([2]))


if __name__ == '__main__':
    unittest.main()
