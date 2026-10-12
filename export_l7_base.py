"""Export frozen L6 RGB/alpha/expected depth at the existing teacher cameras."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.cameras import MiniCam
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--teachers', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scale', type=int, default=2)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.teachers)
    extrinsics = read_extrinsics_text(str(source / 'sparse/0/images.txt'))
    intrinsics = read_intrinsics_text(str(source / 'sparse/0/cameras.txt'))
    model = GaussianModel(3)
    model.load_ply(str(Path(args.model) / 'point_cloud/iteration_30000/point_cloud.ply'))
    pipeline = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    background = torch.zeros(3, device='cuda')
    records = []
    with torch.no_grad():
        for index, extrinsic in enumerate(sorted(extrinsics.values(), key=lambda e: e.name)):
            intrinsic = intrinsics[extrinsic.camera_id]
            if intrinsic.model != 'SIMPLE_PINHOLE':
                raise ValueError('Expected the existing SIMPLE_PINHOLE teacher cameras')
            fov_x = focal2fov(intrinsic.params[0], intrinsic.width)
            fov_y = focal2fov(intrinsic.params[0], intrinsic.height)
            rotation = qvec2rotmat(extrinsic.qvec)
            view = torch.tensor(getWorld2View2(rotation.T, extrinsic.tvec)).T.cuda()
            projection = getProjectionMatrix(0.005, 2000.0, fov_x, fov_y).T.cuda()
            camera = MiniCam(intrinsic.width * args.scale, intrinsic.height * args.scale,
                             fov_y, fov_x, 0.005, 2000.0, view, view @ projection)
            result = render(camera, model, pipeline, background)
            alpha = result['accumulation'].squeeze().cpu().numpy()
            inverse_depth = result['depth'].squeeze().cpu().numpy()
            depth = np.divide(alpha, inverse_depth, out=np.full_like(alpha, 1e6),
                              where=inverse_depth > 1e-8)
            rgb = result['render'].permute(1, 2, 0).cpu().numpy()
            name = Path(extrinsic.name).stem
            np.savez_compressed(output / (name + '.npz'),
                                rgb=rgb.astype(np.float16), alpha=alpha.astype(np.float16), depth=depth)
            records.append({'name': name, 'position': (-rotation.T @ extrinsic.tvec).tolist(),
                            'forward': rotation.T[:, 2].tolist(), 'up': (-rotation.T[:, 1]).tolist(),
                            'fov_y_degrees': float(np.rad2deg(fov_y)),
                            'width': intrinsic.width * args.scale, 'height': intrinsic.height * args.scale})
            if (index + 1) % 20 == 0:
                print(f'Exported {index + 1}/{len(extrinsics)}', flush=True)
    (output / 'cameras.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
