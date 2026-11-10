"""Frozen layered Gaussian model with LUT replacement and occlusion-aware masks."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from plyfile import PlyData
import torch

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.cameras import MiniCam
from utils.graphics_utils import getProjectionMatrix
from utils.sh_utils import eval_sh
from .lut import linear_to_srgb
from .patches import file_hash, eligible_boxes


FIELDS = ('_xyz', '_scaling', '_rotation', '_opacity', '_features_dc', '_features_rest')


def load_camera(entry, width, xyz):
    c2w = np.eye(4)
    c2w[:3, :3], c2w[:3, 3] = entry['rotation'], entry['position']
    if not np.isfinite(c2w).all() or min(entry['fx'], entry['fy']) <= 0:
        raise ValueError('Invalid camera')
    view = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device='cuda').T.contiguous()
    fovx = 2*math.atan(entry['width']/(2*entry['fx']))
    fovy = 2*math.atan(entry['height']/(2*entry['fy']))
    depth = (xyz@view[:3, :3]+view[3, :3])[:, 2]
    positive = depth[depth > 0]
    if not len(positive):
        raise ValueError('No geometry in front of camera')
    near, far = .005, max(5., float(positive.max())*2)
    projection = getProjectionMatrix(near, far, fovx, fovy).T.cuda()
    height = round(width*entry['height']/entry['width'])
    camera = MiniCam(width, height, fovy, fovx, near, far, view, view@projection)
    torch.testing.assert_close(camera.camera_center,
        torch.tensor(entry['position'], dtype=torch.float32, device='cuda'), rtol=1e-5, atol=1e-3)
    return camera


def save_rgb(path, image, srgb=True):
    image = image.detach().clamp(0, 1)
    if srgb:
        image = linear_to_srgb(image)
    Image.fromarray((image.permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)).save(path)


class FrozenGaussianScene:
    def __init__(self, ply, camera_path, layer):
        self.path, self.camera_path = Path(ply).resolve(), Path(camera_path).resolve()
        self.model_hash, self.camera_hash = file_hash(self.path), file_hash(self.camera_path)
        vertex = PlyData.read(str(self.path))['vertex'].data
        if not {'hu', 'layer'}.issubset(vertex.dtype.names):
            raise ValueError('Model needs hu and explicit layer properties')
        labels = np.asarray(vertex['layer'])
        if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
            raise ValueError('Invalid model layer IDs')
        selected = labels == layer
        if not selected.any():
            raise ValueError(f'Layer {layer} absent')
        rest = len([name for name in vertex.dtype.names if name.startswith('f_rest_')])
        degree = math.isqrt(rest//3+1)-1
        if rest != 3*((degree+1)**2-1):
            raise ValueError('Unexpected SH layout')
        self.pc = GaussianModel(degree)
        self.pc.load_ply(str(self.path))  # All PLY rows in the original order, no layer_save duplication.
        np.testing.assert_array_equal(self.pc.get_xyz.detach().cpu().numpy(),
            np.column_stack([vertex[key] for key in ('x', 'y', 'z')]).astype(np.float32))
        self.frozen = {}
        for key in FIELDS:
            value = getattr(self.pc, key)
            if not torch.isfinite(value).all():
                raise ValueError(f'Nonfinite model field: {key}')
            value.requires_grad_(False)
            self.frozen[key] = value.detach().clone()
        self.indices = torch.tensor(np.flatnonzero(selected), device='cuda', dtype=torch.long)
        self.hu = torch.tensor(np.array(vertex['hu'][selected], dtype=np.float32), device='cuda')
        if not torch.isfinite(self.hu).all():
            raise ValueError('Nonfinite selected HU')
        self.tissue_rows = torch.zeros(len(self.indices), device='cuda', dtype=torch.long)
        self.camera_entries = json.loads(self.camera_path.read_text(encoding='utf-8'))
        self.pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
        self.background = torch.zeros(3, device='cuda')
        self.layer = layer
        self.views, self.coverage = [], []

    def _render(self, camera, colors=None):
        if colors is not None and colors.shape != self.pc.get_xyz_all.shape:
            raise RuntimeError('Override color concatenation mismatch')
        return render(camera, self.pc, self.pipe, self.background, override_color=colors)

    def prepare_views(self, indices, width, patch_size, purity, minimum_weight, margin):
        if not indices or len(set(indices)) != len(indices):
            raise ValueError('Expected distinct camera indices')
        indicator = torch.zeros_like(self.pc.get_xyz).index_fill(0, self.indices, 1.)
        with torch.no_grad():
            for index in indices:
                if not 0 <= index < len(self.camera_entries):
                    raise ValueError(f'Camera index outside model camera list: {index}')
                camera = load_camera(self.camera_entries[index], width, self.pc.get_xyz)
                weights = self._render(camera, indicator)
                boxes, valid = eligible_boxes(weights['render'][0], weights['accumulation'],
                    patch_size, purity, minimum_weight, margin)
                self.coverage.append({'camera_index': index, 'eligible_boxes': len(boxes),
                                      'valid_pixels': int(valid.sum()),
                                      'near_far': [camera.znear, camera.zfar]})
                if not boxes:
                    continue
                direction = torch.nn.functional.normalize(self.pc.get_xyz-camera.camera_center, dim=-1)
                context = (eval_sh(self.pc.max_sh_degree, self.pc.get_features.transpose(1, 2), direction)+.5).clamp_min(0)
                self.views.append({'index': index, 'camera': camera, 'context': context,
                                   'boxes': boxes, 'mask': valid.cpu()})
        return self.coverage

    def draw(self, view, lut):
        colors = view['context'].index_copy(0, self.indices, lut(self.hu, self.tissue_rows))
        return self._render(view['camera'], colors)['render']

    def original(self, view):
        return self._render(view['camera'])['render']

    def verify_frozen(self):
        for key, initial in self.frozen.items():
            current = getattr(self.pc, key)
            if not torch.equal(current, initial) or current.grad is not None or current.requires_grad:
                raise RuntimeError(f'Frozen model field changed/received gradients: {key}')
        if self.pc.optimizer is not None:
            raise RuntimeError('Gaussian optimizer must never be configured')
        if file_hash(self.path) != self.model_hash or file_hash(self.camera_path) != self.camera_hash:
            raise RuntimeError('Input model/camera files changed')
        return True
