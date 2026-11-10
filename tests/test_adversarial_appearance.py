import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from appearance.lut import TissueRGBLUT
from appearance.discriminator import TissuePatchDiscriminator
from appearance.patches import RealPatchBank, crop_render, eligible_boxes, file_hash
from appearance.trainer import gan_step
from appearance.checkpoint import save_checkpoint, restore_checkpoint, state_equal


class AdversarialTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)

    def make_models(self):
        lut = TissueRGBLUT(torch.linspace(.1, .8, 24).reshape(1, 8, 3), [[-100, 100]])
        d = TissuePatchDiscriminator(tissues=2, channels=4)
        return lut, d, torch.optim.Adam(lut.parameters(), lr=.01), torch.optim.Adam(d.parameters(), lr=.001)

    def test_alternating_updates_and_input_gradient(self):
        lut, d, go, do = self.make_models()
        before_lut = lut.logits.detach().clone()
        before_d = [p.detach().clone() for p in d.parameters()]
        hu = torch.linspace(-100, 100, 32*32)
        fake = lut(hu, torch.zeros_like(hu, dtype=torch.long)).T.reshape(1, 3, 32, 32).repeat(2, 1, 1, 1)
        stats = gan_step(lut, d, go, do, torch.rand_like(fake), fake, torch.tensor([0, 1]),
                         0., lut.rgb.detach().clone(), check_adversarial_gradient=True)
        self.assertGreater(stats['adversarial_gradient_norm'], 0)
        self.assertFalse(torch.equal(lut.logits, before_lut))
        self.assertTrue(any(not torch.equal(p, old) for p, old in zip(d.parameters(), before_d)))
        self.assertTrue(all(p.grad is None for p in d.parameters()))

    def test_visibility_and_differentiable_crops(self):
        alpha = torch.ones(32, 32)
        contribution = torch.ones(32, 32)
        contribution[:, 16:] = .5  # In-frustum target points could still be behind other tissues.
        boxes, valid = eligible_boxes(contribution, alpha, 8, .98, .95, 1)
        self.assertTrue(boxes)
        self.assertTrue(all(x1 <= 16 for _, _, x1, _ in boxes))
        self.assertFalse(valid[0, 4])
        image = torch.rand(3, 32, 32, requires_grad=True)
        crop_render(image, boxes).sum().backward()
        self.assertGreater(float(image.grad.sum()), 0)
        self.assertEqual(float(image.grad[:, :, 16:].sum()), 0)

    def test_manifest_acceptance_hashes_and_color(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image, annotation = root/'original.png', root/'label.json'
            Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8)).save(image)
            annotation.write_text('{}')
            patch = root/'patch.png'
            patch.write_bytes(image.read_bytes())
            row = {'id': 'one', 'image': patch.name, 'image_sha256': file_hash(patch),
                   'size': 64, 'tissue_label': 'fat', 'frame_id': 'f1', 'patient_group': 'case1',
                   'split': 'unassigned', 'valid_fraction': 1.,
                   'source_image': str(image), 'source_image_sha256': file_hash(image),
                   'source_annotation': str(annotation), 'source_annotation_sha256': file_hash(annotation)}
            manifest = root/'patch_manifest.json'
            manifest.write_text(json.dumps([row]))
            (root/'summary.json').write_text('{"training_ready": false}')
            with self.assertRaises(ValueError):
                RealPatchBank(manifest, 64, 'fat', 'cpu')
            bank = RealPatchBank(manifest, 64, 'fat', 'cpu', allow_unreviewed=True)
            self.assertAlmostEqual(float(bank.linear_rgb.mean()), .2158605, places=6)
            self.assertTrue(bank.verify_unchanged())
            annotation.write_text('{"changed": true}')
            with self.assertRaises(RuntimeError):
                bank.verify_unchanged()
            with self.assertRaises(ValueError):
                RealPatchBank(manifest, 64, 'fat', 'cpu', allow_unreviewed=True)

    def test_checkpoint_optimizers_rng_and_fingerprint(self):
        lut, d, go, do = self.make_models()
        hu = torch.linspace(-100, 100, 32*32)
        fake = lut(hu, torch.zeros_like(hu, dtype=torch.long)).T.reshape(1, 3, 32, 32).repeat(2, 1, 1, 1)
        gan_step(lut, d, go, do, torch.rand_like(fake), fake, torch.tensor([0, 1]), 0., lut.rgb.detach())
        rng = np.random.default_rng(8)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.pt'
            save_checkpoint(path, 1, lut, d, go, do, lut.rgb.detach().clone(), {'input': 'fixed'}, rng, [])
            expected_next = int(rng.integers(100000))
            new_lut, new_d, new_go, new_do = self.make_models()
            restored = restore_checkpoint(path, new_lut, new_d, new_go, new_do, {'input': 'fixed'}, rng)
            self.assertEqual(restored['step'], 1)
            self.assertEqual(int(rng.integers(100000)), expected_next)
            self.assertTrue(state_equal(lut.state_dict(), new_lut.state_dict()))
            self.assertTrue(state_equal(go.state_dict(), new_go.state_dict()))
            self.assertTrue(state_equal(do.state_dict(), new_do.state_dict()))
            with self.assertRaises(ValueError):
                restore_checkpoint(path, new_lut, new_d, new_go, new_do, {'input': 'other'}, rng)


if __name__ == '__main__':
    unittest.main()
