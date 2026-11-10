"""Refine L7 against smooth teachers with bounded scale and surface distance."""
import argparse
import json
from pathlib import Path
import random
import shutil

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
import torch
from torch import nn

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.image_utils import flatten_rgba, psnr
from utils.loss_utils import l1_loss, ssim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    optimization = OptimizationParams(parser)
    parser.add_argument('--seed-model', type=Path, required=True)
    parser.add_argument('--base-checkpoint', type=Path, required=True)
    parser.add_argument('--surface', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--max-surface-distance-mm', type=float, default=0.6)
    parser.add_argument('--max-long-axis-mm', type=float, default=2.0)
    parser.add_argument('--max-long-middle-ratio', type=float, default=8.0)
    args = parser.parse_args()
    random.seed(49)
    np.random.seed(49)
    torch.manual_seed(49)
    output = Path(args.model_path)
    if output.exists():
        raise ValueError('Use a new output directory')
    if args.steps <= 0 or min(args.max_surface_distance_mm, args.max_long_axis_mm,
                             args.max_long_middle_ratio) <= 0:
        raise ValueError('Refinement steps and bounds must be positive')
    output.mkdir(parents=True)
    seed_ply = args.seed_model / 'point_cloud/iteration_30000/point_cloud.ply'
    seed_dir = output / 'point_cloud/iteration_30000'
    seed_dir.mkdir(parents=True)
    shutil.copy2(seed_ply, seed_dir / 'point_cloud.ply')
    dataset, pipe, opt = model.extract(args), pipeline.extract(args), optimization.extract(args)
    (output / 'cfg_args').write_text(str(argparse.Namespace(**vars(dataset))), encoding='utf-8')
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=30000, shuffle=False)
    vertex = PlyData.read(str(seed_ply))['vertex'].data
    mask = torch.tensor(vertex['layer'] == 7, device='cuda')
    for field in ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity'):
        setattr(gaussians, field, nn.Parameter(getattr(gaussians, field)[mask].detach().clone()))
    gaussians.restore_layer(str(args.base_checkpoint))
    gaussians.spatial_lr_scale = scene.cameras_extent
    opt.position_lr_init = 0.000002
    opt.position_lr_final = 0.0000005
    opt.position_lr_max_steps = args.steps
    opt.feature_lr = 0.001
    opt.opacity_lr = 0.01
    opt.scaling_lr = 0.001
    opt.rotation_lr = 0.0002
    gaussians.training_setup(opt)
    mesh = PlyData.read(str(args.surface))['vertex'].data
    mesh_xyz = np.column_stack([mesh[key] for key in ('x', 'y', 'z')])
    tree = cKDTree(mesh_xyz)
    initial_distance, nearest = tree.query(gaussians.get_xyz.detach().cpu().numpy(), workers=-1)
    anchors = torch.tensor(mesh_xyz[nearest], dtype=torch.float32, device='cuda')

    def constrain():
        with torch.no_grad():
            # Preserve naturally thin surface discs; bound the longest axis
            # against the middle axis rather than against the surface normal.
            scales = gaussians.get_scaling
            ordered, indices = scales.sort(dim=1)
            cap = torch.clamp(args.max_long_middle_ratio * ordered[:, 1], min=0.25,
                              max=args.max_long_axis_mm)
            limit = torch.minimum(ordered[:, 2], cap)
            gaussians._scaling.scatter_(1, indices[:, 2:3], limit.log()[:, None])
            delta = gaussians._xyz - anchors
            distance = delta.norm(dim=1, keepdim=True).clamp_min(1e-12)
            factor = (args.max_surface_distance_mm / distance).clamp(max=1)
            gaussians._xyz.copy_(anchors + delta * factor)

    constrain()
    black = torch.zeros(3, device='cuda')

    def evaluate():
        scores, similarities = [], []
        with torch.no_grad():
            for view in scene.getTestCameras():
                predicted = render(view, gaussians, pipe, black,
                                   antialiasing=dataset.antialiasing)['render'].clamp(0, 1)
                truth = flatten_rgba(view, black.cpu()).cuda().clamp(0, 1)
                scores.append(float(psnr(predicted, truth).mean()))
                similarities.append(float(ssim(predicted, truth)))
        return {'psnr': float(np.mean(scores)), 'ssim': float(np.mean(similarities))}

    baseline = evaluate()
    print('Initial constrained model:', baseline, flush=True)
    views = []
    for step in range(1, args.steps + 1):
        gaussians.update_learning_rate(step)
        if not views:
            views = scene.getTrainCameras().copy()
            random.shuffle(views)
        view = views.pop()
        bg = torch.rand(3, device='cuda')
        package = render(view, gaussians, pipe, bg, antialiasing=dataset.antialiasing)
        predicted = package['render']
        truth = flatten_rgba(view, bg.cpu()).cuda()
        if dataset.alpha_train:
            predicted = torch.cat((predicted, package['accumulation'].unsqueeze(0)), dim=0)
            truth = torch.cat((truth, view.original_alpha_mask.cuda()), dim=0)
        loss = (1 - opt.lambda_dssim) * l1_loss(predicted, truth) + opt.lambda_dssim * (1 - ssim(predicted, truth))
        loss.backward()
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)
        constrain()
        if step % 250 == 0:
            print(f'Refinement {step}/{args.steps}, loss={float(loss):.6f}', flush=True)
    final = evaluate()
    iteration = 30000 + args.steps
    scene.save(iteration)
    torch.save(gaussians.capture_layer(), output / f'layer_save_{iteration}.pth')
    distance, _ = tree.query(gaussians.get_xyz.detach().cpu().numpy(), workers=-1)
    report = {
        'seed_model': str(args.seed_model), 'surface': str(args.surface),
        'teachers': dataset.source_path, 'steps': args.steps, 'iteration': iteration,
        'initial_surface_distance_p50_p95_max_mm': np.percentile(initial_distance, [50, 95, 100]).tolist(),
        'final_surface_distance_p50_p95_max_mm': np.percentile(distance, [50, 95, 100]).tolist(),
        'max_surface_distance_mm': args.max_surface_distance_mm,
        'max_long_axis_mm': args.max_long_axis_mm,
        'max_long_middle_ratio': args.max_long_middle_ratio,
        'initial_test_metrics': baseline, 'final_test_metrics': final,
        'metric_scope': 'Fit to smooth teacher images, not anatomical accuracy.',
        'gaussian_count_layer7': len(gaussians.get_xyz),
        'hu_status': 'Must be resampled at the refined positions before Unity import.'
    }
    (output / 'refinement_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
