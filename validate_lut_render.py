"""Bounded synthetic gradient validation on an existing HU Gaussian PLY.

No real-image distribution learning is performed. Baseline training is untouched.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
from PIL import Image
from plyfile import PlyData
import torch

from appearance_lut import from_manual_config, linear_to_srgb
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.cameras import MiniCam
from utils.graphics_utils import getProjectionMatrix, getWorld2View2
from utils.sh_utils import eval_sh


FIELDS = ('_xyz', '_scaling', '_rotation', '_opacity', '_features_dc', '_features_rest')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_camera(entry, width, xyz):
    # cameras.json rotation/position encode camera-to-world, as camera_to_JSON.
    c2w = np.eye(4)
    c2w[:3, :3], c2w[:3, 3] = entry['rotation'], entry['position']
    w2c = np.linalg.inv(c2w)
    reference = getWorld2View2(w2c[:3, :3].T, w2c[:3, 3])
    np.testing.assert_allclose(reference, w2c, rtol=1e-5, atol=1e-4)
    fovx = 2 * math.atan(entry['width'] / (2 * entry['fx']))
    fovy = 2 * math.atan(entry['height'] / (2 * entry['fy']))
    view = torch.tensor(reference, device='cuda').T.contiguous()
    depth = (xyz @ view[:3, :3] + view[3, :3])[:, 2]
    positive = depth[depth > 0]
    if not len(positive):
        raise ValueError('No points in front of camera; check coordinate conventions')
    # Existing data uses millimeter-scale coordinates. Do not inherit Camera.zfar=5.
    near, far = .005, max(5., float(positive.max()) * 2)
    projection = getProjectionMatrix(near, far, fovx, fovy).T.cuda()
    height = round(width * entry['height'] / entry['width'])
    cam = MiniCam(width, height, fovy, fovx, near, far, view, view @ projection)
    torch.testing.assert_close(cam.camera_center, torch.tensor(entry['position'], device='cuda', dtype=torch.float32),
                               rtol=1e-5, atol=1e-3)
    return cam


def save_image(path, tensor, display_transform=True):
    # Only visualization leaves autograd; this function is never in the loss path.
    value = tensor.detach().clamp(0, 1)
    if display_transform:
        value = linear_to_srgb(value)
    Image.fromarray((value.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ply', type=Path, required=True)
    parser.add_argument('--cameras', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'experiments/lut_l4.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--knots', type=int, choices=[16, 32], default=16)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--view-indices', type=int, nargs='+', default=[0, 40, 80])
    parser.add_argument('--seed', type=int, default=20261108)
    parser.add_argument('--lr', type=float, default=.03)
    parser.add_argument('--smooth-weight', type=float, default=1e-4)
    args = parser.parse_args()
    if not 1 <= args.steps <= 50 or not 32 <= args.width <= 512:
        raise ValueError('Smoke validation is bounded to 1..50 steps and 32..512 pixels')
    if args.lr <= 0 or args.smooth_weight < 0:
        raise ValueError('Invalid optimization settings')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA rasterizer required; CPU previews cannot validate this path')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config = json.loads(args.config.read_text(encoding='utf-8'))
    entries = json.loads(args.cameras.read_text(encoding='utf-8'))
    if not args.view_indices or min(args.view_indices) < 0 or max(args.view_indices) >= len(entries):
        raise ValueError('Invalid camera indices')
    input_model_hash = sha256(args.ply)
    vertex = PlyData.read(str(args.ply))['vertex'].data
    if not {'hu', 'layer'}.issubset(vertex.dtype.names):
        raise ValueError('PLY must contain explicit hu and layer properties')
    layer_values = vertex['layer']
    if not np.isfinite(layer_values).all() or not np.equal(layer_values, np.floor(layer_values)).all():
        raise ValueError('Invalid PLY layer labels')
    selected = np.zeros(len(vertex), dtype=bool)
    rows = np.full(len(vertex), -1, dtype=np.int64)
    layer_ids = [spec['layer'] for spec in config['tissues']]
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError('Duplicate tissue layer mapping')
    for k, layer in enumerate(layer_ids):
        mask = layer_values == layer
        if not mask.any():
            raise ValueError(f'Missing configured tissue/layer {layer}')
        rows[mask], selected[mask] = k, True
    # Model.load_ply loads every row into the active array, without reordering.
    rest = len([name for name in vertex.dtype.names if name.startswith('f_rest_')])
    degree = int(math.isqrt(rest // 3 + 1) - 1)
    if rest != 3 * ((degree + 1) ** 2 - 1):
        raise ValueError('Invalid SH coefficient count')
    pc = GaussianModel(degree)
    pc.load_ply(str(args.ply))
    for name in FIELDS:
        getattr(pc, name).requires_grad_(False)
    frozen = {name: getattr(pc, name).detach().clone() for name in FIELDS}
    xyz_input = np.column_stack([vertex[x] for x in ('x', 'y', 'z')]).astype(np.float32)
    np.testing.assert_array_equal(pc.get_xyz.detach().cpu().numpy(), xyz_input)
    indices = torch.tensor(np.flatnonzero(selected), device='cuda')
    hu = torch.tensor(np.array(vertex['hu'][selected], dtype=np.float32), device='cuda')
    tissue = torch.tensor(rows[selected], device='cuda')
    lut = from_manual_config(config, args.knots).cuda()
    initial_state = {k: v.detach().clone() for k, v in lut.state_dict().items()}
    optimizer = torch.optim.Adam(lut.parameters(), lr=args.lr)
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == {id(lut.logits)}
    views = [load_camera(entries[i], args.width, pc.get_xyz) for i in args.view_indices]
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    background = torch.zeros(3, device='cuda')
    fixed_sh = []
    with torch.no_grad():
        for view in views:
            direction = torch.nn.functional.normalize(pc.get_xyz - view.camera_center, dim=-1)
            fixed_sh.append((eval_sh(degree, pc.get_features.transpose(1, 2), direction) + .5).clamp_min(0))

    def draw(i, colors=None):
        # Selected tissues REPLACE all SH contributions. Context uses frozen SH.
        colors = lut(hu, tissue) if colors is None else colors
        override = fixed_sh[i].index_copy(0, indices, colors)
        if override.shape != pc.get_xyz_all.shape:
            raise RuntimeError('Override color/model concatenation count mismatch')
        return render(views[i], pc, pipe, background, override_color=override)

    args.output.mkdir(parents=True, exist_ok=False)
    # Synthetic teacher is ONLY a plumbing test, generated in the same renderer.
    with torch.no_grad():
        initial_rgb = lut.rgb.clone()
        lut.logits.add_(torch.tensor([.5, -.3, .4], device='cuda'))
        targets = [draw(i)['render'].detach() for i in range(len(views))]
        lut.load_state_dict(initial_state)
        initial_packages = [draw(i) for i in range(len(views))]
        for i, package in enumerate(initial_packages):
            if not package['visibility_filter_all'][indices].any():
                raise RuntimeError(f'No selected tissue visible in view {i}')
            save_image(args.output / f'view_{i}_initial.png', package['render'])
            save_image(args.output / f'view_{i}_synthetic_target.png', targets[i])
            # Historical SH colors have no established linear/sRGB contract.
            # Save their raw numeric display exactly as the original renderer.
            old = render(views[i], pc, pipe, background)['render']
            save_image(args.output / f'view_{i}_original_sh_as_stored.png', old, False)

    def objective(i):
        prediction = draw(i)['render']
        return (prediction - targets[i]).square().mean()

    # Strongest image-loss gradient, independent of smoothness, checked through CUDA.
    first_loss = objective(0)
    first_loss.backward()
    grad = lut.logits.grad.detach().clone()
    if not torch.isfinite(grad).all() or grad.abs().max() <= 0:
        raise RuntimeError('Missing/nonfinite image -> LUT gradient')
    flat_index = int(grad.abs().argmax())
    analytic = float(grad.flatten()[flat_index])
    eps = .01
    with torch.no_grad():
        flat = lut.logits.view(-1)
        original = flat[flat_index].item()
        flat[flat_index] = original + eps
        plus = float(objective(0))
        flat[flat_index] = original - eps
        minus = float(objective(0))
        flat[flat_index] = original
    numeric = (plus - minus) / (2 * eps)
    relative_error = abs(analytic - numeric) / max(abs(analytic), abs(numeric), 1e-12)
    if relative_error > .05:
        raise RuntimeError(f'CUDA finite difference mismatch: {analytic}, {numeric}')
    optimizer.zero_grad(set_to_none=True)
    history = []
    with torch.no_grad():
        before = [float(objective(i)) for i in range(len(views))]
    for step in range(args.steps):
        i = step % len(views)
        optimizer.zero_grad(set_to_none=True)
        image_loss = objective(i)
        smooth = lut.smoothness()
        loss = image_loss + args.smooth_weight * smooth
        loss.backward()
        if not torch.isfinite(lut.logits.grad).all():
            raise RuntimeError('Nonfinite gradient')
        history.append({'step': step + 1, 'view': i, 'image_mse': float(image_loss),
                        'smoothness': float(smooth), 'gradient_norm': float(lut.logits.grad.norm())})
        optimizer.step()
    with torch.no_grad():
        after = [float(objective(i)) for i in range(len(views))]
        for name, original in frozen.items():
            if not torch.equal(getattr(pc, name), original) or getattr(pc, name).grad is not None:
                raise RuntimeError(f'Frozen model changed or received gradients: {name}')
        for i in range(len(views)):
            save_image(args.output / f'view_{i}_final.png', draw(i)['render'])
        final_rgb = lut.rgb.detach().clone()
        if torch.equal(final_rgb, initial_rgb) or np.mean(after) >= np.mean(before):
            raise RuntimeError('Synthetic optimization failed to change colors and lower mean loss')
    if sha256(args.ply) != input_model_hash:
        raise RuntimeError('Input model file changed during validation')
    checkpoint_path = args.output / 'lut_checkpoint.pt'
    torch.save({'lut': lut.state_dict(), 'optimizer': optimizer.state_dict(),
                'initial_lut': initial_state, 'config': config, 'knots': args.knots,
                'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'model_path': str(args.ply.resolve()), 'model_sha256': input_model_hash,
                'camera_sha256': sha256(args.cameras), 'step': args.steps,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
                'discriminator': None, 'scope': 'synthetic_gradient_validation_only'}, checkpoint_path)
    restored = torch.load(checkpoint_path)
    check_lut = from_manual_config(config, args.knots).cuda()
    check_lut.load_state_dict(restored['lut'])
    check_optimizer = torch.optim.Adam(check_lut.parameters(), lr=args.lr)
    check_optimizer.load_state_dict(restored['optimizer'])
    original_adam = optimizer.state[lut.logits]
    restored_adam = check_optimizer.state[check_lut.logits]
    for key in ('step', 'exp_avg', 'exp_avg_sq'):
        torch.testing.assert_close(restored_adam[key], original_adam[key], rtol=0, atol=0)
    if not torch.equal(check_lut(hu, tissue), lut(hu, tissue)):
        raise RuntimeError('Checkpoint color roundtrip mismatch')
    with torch.no_grad():
        for i in range(len(views)):
            if not torch.equal(draw(i, check_lut(hu, tissue))['render'], draw(i)['render']):
                raise RuntimeError('Checkpoint render roundtrip mismatch')
    report = {
        'scope': 'Synthetic CUDA gradient validation; NOT unpaired realism training',
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
        'model': str(args.ply.resolve()), 'model_sha256': restored['model_sha256'],
        'camera_path': str(args.cameras.resolve()), 'view_indices': args.view_indices,
        'seed': args.seed, 'knots': args.knots, 'steps': args.steps,
        'gaussians_total': len(vertex), 'selected_gaussians': len(indices),
        'tissue_mapping': config['tissues'], 'selected_hu_percentiles': np.percentile(hu.cpu(), [0, 5, 50, 95, 100]).tolist(),
        'selected_hu_below_range': int((hu < lut.hu_bounds[tissue, 0]).sum()),
        'selected_hu_above_range': int((hu > lut.hu_bounds[tissue, 1]).sum()),
        'frustum_selected_per_view': [int(p['visibility_filter_all'][indices].sum()) for p in initial_packages],
        'camera_near_far': [[v.znear, v.zfar] for v in views],
        'image_mse_before': before, 'image_mse_after': after,
        'image_gradient_norm': float(grad.norm()), 'finite_difference_analytic': analytic,
        'finite_difference_numeric': numeric, 'finite_difference_relative_error': relative_error,
        'knots_with_image_gradient': (grad.abs().sum(-1) > 0).sum(-1).tolist(),
        'all_model_fields_bitwise_unchanged': True, 'checkpoint_color_and_render_roundtrip': True,
        'input_model_hash_unchanged': True, 'checkpoint_optimizer_roundtrip': True,
        'linear_rgb_min_max': [float(final_rgb.min()), float(final_rgb.max())],
        'max_control_rgb_change': float((final_rgb - initial_rgb).abs().max()),
        'color_contract': 'Selected LUT: linear RGB; PNG: one sRGB encoding. Context SH: historical numeric values, colorimetry unverified.',
        'missing': ['Verified target-domain tissue crops/masks and patient split',
                    'Layer-to-tissue semantic validation against CT masks',
                    'Real-image color/exposure calibration'],
        'history': history}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    curves = {'hu': lut.hu_knots.cpu().tolist(), 'initial_linear_rgb': initial_rgb.cpu().tolist(),
              'final_linear_rgb': final_rgb.cpu().tolist()}
    (args.output / 'lut_curves.json').write_text(json.dumps(curves, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
