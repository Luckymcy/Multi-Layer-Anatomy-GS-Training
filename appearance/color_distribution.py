"""Short non-adversarial masked color-distribution validation, fixed Gaussian RGB."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .lut import srgb_to_linear, linear_to_srgb
from .gaussian_rgb import GaussianRGB
from .patches import file_hash


def distribution_loss(fake, real, directions):
    """Equal-sized RGB point clouds: moments plus sliced Wasserstein squared."""
    if fake.shape != real.shape or fake.ndim != 2 or fake.shape[1] != 3:
        raise ValueError('Expected equal nonempty Nx3 samples')
    fm, rm = fake.mean(0), real.mean(0)
    fc, rc = fake - fm, real - rm
    covariance = F.mse_loss(fc.T @ fc / len(fake), rc.T @ rc / len(real))
    wasserstein = F.mse_loss((fake @ directions.T).sort(0).values,
                            (real @ directions.T).sort(0).values)
    return F.mse_loss(fm, rm) + covariance + wasserstein


def load_real(root, output, device, tissue_label='myocardium'):
    # Reuse the audited polygon/exclusion and quality filtering implementation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.annotations.prepare_fat_dataset import rasterize, sampling_mask, overlay
    manifest = root / 'manifest.json'
    rows = json.loads(manifest.read_text(encoding='utf-8'))
    bank, report, hashes = [], [], {str(manifest): file_hash(manifest)}
    for row in rows:
        image = (root / row['image']).resolve()
        label = root / 'labels' / (row['id'] + '.json')
        if not label.exists():
            raise ValueError(f'Missing annotation: {label}')
        data = json.loads(label.read_text(encoding='utf-8'))
        if (label.parent / data['imagePath'].replace('\\', '/')).resolve() != image:
            raise ValueError('Annotation/image link mismatch')
        if file_hash(image) != row['image_sha256']:
            raise ValueError('Source image hash mismatch')
        rgb = np.asarray(Image.open(image).convert('RGB')).copy()
        if (data['imageWidth'], data['imageHeight']) != (rgb.shape[1], rgb.shape[0]):
            raise ValueError('Annotation/image dimensions differ')
        shapes = []
        for shape in data['shapes']:
            if shape['label'] not in (tissue_label, 'exclude'):
                raise ValueError(f'Unexpected label: {shape["label"]}')
            shapes.append(dict(shape, label='fat' if shape['label'] == tissue_label else 'exclude'))
        tissue, exclude, valid, counts = rasterize(shapes, (rgb.shape[1], rgb.shape[0]))
        safe, bright = sampling_mask(rgb, valid, 2, 250, 220, 1)
        if safe.sum() < 32:
            raise ValueError(f'Insufficient valid tissue: {row["id"]}')
        Image.fromarray(safe.astype(np.uint8)*255).save(output / f'{row["id"]}_mask.png')
        preview = overlay(rgb, valid, exclude, bright, safe)
        preview.thumbnail((960, 540))
        preview.save(output / f'{row["id"]}_overlay.jpg')
        bank.append(srgb_to_linear(torch.tensor(rgb[safe].copy(), device=device).float()/255))
        report.append(dict(id=row['id'], tissue_pixels=int(tissue.sum()),
                           excluded_pixels=int(exclude.sum()), valid_pixels=int(safe.sum()),
                           valid_fraction=float(safe.mean())))
        hashes[str(image)], hashes[str(label)] = file_hash(image), file_hash(label)
    if not bank:
        raise ValueError('No annotated frames')
    return bank, report, hashes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--labeling', type=Path, required=True)
    ap.add_argument('--ply', type=Path, required=True)
    ap.add_argument('--cameras', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=5)
    ap.add_argument('--width', type=int, default=768)
    ap.add_argument('--layer', type=int, default=4)
    ap.add_argument('--tissue-label', choices=['myocardium','fat'], default='myocardium')
    ap.add_argument('--source-color-space', choices=['legacy-linear','srgb'],default='legacy-linear',
                    help='Numeric SH RGB encoding of source PLY; source geometry training uses srgb')
    ap.add_argument('--seed', type=int, default=20261117)
    ap.add_argument('--long-run', action='store_true')
    ap.add_argument('--train-split', type=Path)
    ap.add_argument('--test-split', type=Path)
    ap.add_argument('--lr', type=float, default=.001)
    ap.add_argument('--save-every', type=int, default=250)
    ap.add_argument('--surface-graph', type=Path)
    ap.add_argument('--surface-weight', type=float, default=0.)
    ap.add_argument('--surface-huber-delta', type=float, default=.05)
    ap.add_argument('--feature-weights',type=Path)
    ap.add_argument('--feature-weight',type=float,default=0.)
    args = ap.parse_args()
    if not 1 <= args.steps <= (100000 if args.long_run else 20) or not 128 <= args.width <= 2048:
        ap.error('Initial validation supports 1..20 steps and width 128..2048')
    if bool(args.train_split) != bool(args.test_split) or args.lr <= 0 or args.save_every < 1:
        ap.error('Supply both splits and positive learning rate/save interval')
    if args.output.exists():
        ap.error('Use a new output directory')
    if (not np.isfinite(args.surface_weight) or args.surface_weight < 0
            or not np.isfinite(args.surface_huber_delta) or args.surface_huber_delta <= 0
            or (args.surface_weight > 0 and args.surface_graph is None)):
        ap.error('Surface weight must be nonnegative; positive weight requires a graph and positive Huber delta')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output.mkdir(parents=True)
    (args.output / 'real_masks').mkdir()
    bank, masks, hashes = load_real(args.labeling.resolve(), args.output / 'real_masks', 'cuda',args.tissue_label)
    feature_model, feature_bank = None, []
    if args.feature_weight>0:
        if args.feature_weights is None: ap.error('Feature weights required')
        from .feature_texture import MaskedFeatureSW
        feature_model=MaskedFeatureSW(args.feature_weights).cuda().eval()
        hashes[str(args.feature_weights.resolve())]=file_hash(args.feature_weights)
        rows=json.loads((args.labeling/'manifest.json').read_text(encoding='utf-8'))
        with torch.no_grad():
            for row in rows:
                rgb=torch.tensor(np.array(Image.open(args.labeling/row['image']).convert('RGB')).copy(),device='cuda').float().permute(2,0,1)/255
                valid=torch.tensor(np.array(Image.open(args.output/'real_masks'/f'{row["id"]}_mask.png')).copy(),device='cuda').float()/255
                height=round(args.width*rgb.shape[1]/rgb.shape[2])
                rgb=F.interpolate(rgb[None],size=(height,args.width),mode='bilinear',align_corners=False)[0]
                valid=F.interpolate(valid[None,None],size=(height,args.width),mode='area')[0,0]>.999
                tokens=feature_model.tokens(rgb,valid)
                if not tokens: raise ValueError(f'No safe feature tokens: {row["id"]}')
                feature_bank.append(tokens)
        torch.save([{k:v.cpu() for k,v in item.items()} for item in feature_bank],args.output/'real_feature_bank.pt')
    (args.output / 'annotation_report.json').write_text(json.dumps(masks, indent=2), encoding='utf-8')
    from .scene import FrozenGaussianScene, save_rgb
    scene = FrozenGaussianScene(args.ply, args.cameras, args.layer)
    surface_graph, surface_metadata = None, None
    if args.surface_graph:
        from .surface_graph import SurfaceColorGraph
        surface_graph, surface_metadata = SurfaceColorGraph.load(
            args.surface_graph, scene.model_hash, scene.indices, args.surface_huber_delta)
        surface_graph = surface_graph.cuda()
        hashes[str(args.surface_graph.resolve())] = file_hash(args.surface_graph)
    if args.train_split:
        from .views import split_view_indices
        indices, held_out = split_view_indices(scene.camera_entries, args.train_split, args.test_split)
    else:
        indices = np.linspace(0, len(scene.camera_entries)-1, min(24, len(scene.camera_entries)), dtype=int).tolist()
        held_out = []
    scene.prepare_pixel_views(indices, args.width)
    (args.output / 'view_coverage.json').write_text(json.dumps(scene.coverage, indent=2), encoding='utf-8')
    if not scene.views:
        raise ValueError(f'No visible L{args.layer} pixels at unchanged purity/weight thresholds')
    # Start from stored SH DC, interpreted as historical linear numeric RGB.
    # Higher-order SH on the selected layer is deliberately bypassed.
    initial = (scene.pc._features_dc[scene.indices, 0] * .28209479177387814 + .5).clamp(0, 1)
    if args.source_color_space == 'srgb':
        initial=srgb_to_linear(initial)
        for view in scene.views:
            view['context']=srgb_to_linear(view['context'].clamp(0,1))
    colors = GaussianRGB(initial).cuda()
    initial = colors.rgb.detach().clone()
    initial_surface_loss = float(surface_graph(initial)) if surface_graph is not None else None
    optimizer = torch.optim.Adam(colors.parameters(), lr=args.lr)
    directions = F.normalize(torch.randn(32, 3, device='cuda'), dim=-1)
    fixed = scene.views[:3]
    evaluation_real = torch.cat([b[torch.randint(len(b), (512,), device='cuda')] for b in bank])
    evaluation_indices = []
    for i, view in enumerate(fixed):
        ids = view['mask'].flatten().nonzero().flatten()
        evaluation_indices.append(ids[torch.randint(len(ids), (len(evaluation_real),), device='cuda')])
        save_rgb(args.output / f'view_{view["index"]}_before.png', scene.draw(view, colors))
        Image.fromarray(view['mask'].cpu().numpy().astype(np.uint8)*255).save(args.output / f'view_{view["index"]}_mask.png')
    def evaluate():
        with torch.no_grad():
            return [float(distribution_loss(scene.draw(v, colors).permute(1, 2, 0).reshape(-1, 3)[ids],
                                            evaluation_real, directions)) for v, ids in zip(fixed, evaluation_indices)]
    before = evaluate()
    history = []
    first_gradient = None
    first_surface_gradient = None
    first_feature_gradient = None
    rng = np.random.default_rng(args.seed)
    def save_state(path, completed_steps):
        torch.save(dict(format='gaussian-linear-rgb-v1', layer=args.layer,
                        ply_row_indices=scene.indices.cpu(), linear_rgb=colors.rgb.detach().cpu(),
                        colors=colors.state_dict(), optimizer=optimizer.state_dict(),
                        directions=directions.cpu(), completed_steps=completed_steps,
                        initial_rgb=initial.cpu(), torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=rng.bit_generator.state,
                        model_sha256=scene.model_hash, camera_sha256=scene.camera_hash,
                        input_hashes=hashes,
                        surface_graph_state=surface_graph.state_dict() if surface_graph is not None else None,
                        surface_graph_metadata=surface_metadata,
                        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}), path)
    for step in range(args.steps):
        from .views import epoch_view_index
        view = scene.views[epoch_view_index(step+1, len(scene.views), args.seed)]
        real_index = int(rng.integers(len(bank)))
        real = bank[real_index]
        real = real[torch.randint(len(real), (2048,), device='cuda')]
        rendered = scene.draw(view, colors)
        fake = rendered.permute(1, 2, 0)[view['mask']]
        fake = fake[torch.randint(len(fake), (2048,), device='cuda')]
        data_loss = distribution_loss(fake, real, directions)
        if step == 0:
            grad = torch.autograd.grad(data_loss, colors.logits, retain_graph=True)[0]
            first_gradient = dict(norm=float(grad.norm()), nonzero_gaussians=int((grad.abs().sum(-1)>0).sum()))
            if not torch.isfinite(grad).all() or first_gradient['norm'] <= 0:
                raise RuntimeError('No finite data-only color gradient')
        anchor = (colors.rgb-initial).square().mean()
        surface_loss = surface_graph(colors.rgb) if surface_graph is not None else colors.logits.new_zeros(())
        feature_loss=colors.logits.new_zeros(())
        if feature_model is not None:
            fake_features=feature_model.tokens(linear_to_srgb(rendered.clamp(0,1)),view['mask'])
            feature_loss=feature_model.loss(fake_features,feature_bank[real_index])
            if step==0:
                fg=torch.autograd.grad(feature_loss,colors.logits,retain_graph=True)[0]
                first_feature_gradient=float(fg.norm())
                if not torch.isfinite(fg).all() or first_feature_gradient<=0: raise RuntimeError('No finite feature-to-color gradient')
        if step == 0 and surface_graph is not None:
            graph_grad = torch.autograd.grad(surface_loss, colors.logits, retain_graph=True)[0]
            if not torch.isfinite(graph_grad).all():
                raise RuntimeError('Nonfinite surface regularization gradient')
            first_surface_gradient = float(graph_grad.norm())
        loss = data_loss + .01*anchor + args.surface_weight*surface_loss + args.feature_weight*feature_loss
        optimizer.zero_grad()
        loss.backward()
        if not torch.isfinite(colors.logits.grad).all():
            raise RuntimeError('Nonfinite gradient')
        optimizer.step()
        history.append(dict(step=step+1, camera=view['index'], real_frame=masks[real_index]['id'],
                            data_loss=float(data_loss), anchor=float(anchor), surface_loss=float(surface_loss),feature_loss=float(feature_loss),
                            weighted_surface_loss=float(args.surface_weight*surface_loss), total_loss=float(loss)))
        if step == 0 or (step+1) % args.save_every == 0:
            scene.verify_frozen()
            save_state(args.output / 'latest_checkpoint.pt', step+1)
            (args.output / 'progress.json').write_text(json.dumps(dict(completed_steps=step+1,
                total_steps=args.steps, last=history[-1], fixed_sample_loss=evaluate()), indent=2), encoding='utf-8')
            print(history[-1], flush=True)
    after = evaluate()
    for view in fixed:
        save_rgb(args.output / f'view_{view["index"]}_after.png', scene.draw(view, colors))
    unchanged = all(file_hash(Path(p)) == h for p, h in hashes.items())
    if not unchanged:
        raise RuntimeError('Source annotation data changed during validation')
    scene.verify_frozen()
    delta = (colors.rgb.detach()-initial).abs()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    report = dict(config=config, method='masked_RGB_and_VGG_feature_SW_with_surface_graph' if feature_model is not None else 'masked_linear_RGB_moments_and_sliced_Wasserstein',
                  discriminator=False, patch_cropping=False, steps=args.steps,
                  annotated_frames=len(bank), eligible_views=len(scene.views),
                  screened_views=len(indices), data_only_gradient=first_gradient,
                  held_out_camera_indices=held_out,
                  structure_unchanged=True, source_inputs_unchanged=unchanged,
                  fixed_sample_loss_before=before, fixed_sample_loss_after=after,
                  changed_gaussians=int((delta.max(-1).values>1e-6).sum()),
                  maximum_rgb_change=float(delta.max()), metrics=history,
                  input_hashes=hashes, model_sha256=scene.model_hash,
                  camera_sha256=scene.camera_hash,
                  feature_texture={'enabled':feature_model is not None,'weight':args.feature_weight,
                    'first_gradient_norm':first_feature_gradient,'layers':['relu1_2','relu2_2'],
                    'receptive_fields_pixels':[5,14],'weights_sha256':file_hash(args.feature_weights) if args.feature_weights else None,
                    'masks':'Full receptive field valid; frame resized to render width. Missing feature levels skipped, empty frames rejected.'},
                  surface_regularization={'enabled': surface_graph is not None and args.surface_weight > 0,
                      'weight': args.surface_weight, 'huber_delta': args.surface_huber_delta,
                      'initial_loss': initial_surface_loss,
                      'final_loss': float(surface_graph(colors.rgb).detach()) if surface_graph is not None else None,
                      'first_unweighted_gradient_norm': first_surface_gradient, 'graph': surface_metadata},
                  limitations=f'One-operation experiment without independent realism evaluation. Surface graph, when enabled, is a CT surface prior, not verified anatomical connectivity of every Gaussian. Source SH DC encoding: {args.source_color_space}; selected-layer higher SH bypassed. Color-distribution loss does not establish spatial texture realism.')
    save_state(args.output / 'gaussian_colors.pt', args.steps)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Completed short non-adversarial validation', flush=True)


if __name__ == '__main__':
    main()
