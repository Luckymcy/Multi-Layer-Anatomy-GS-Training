"""Unpaired real-patch guidance for a single frozen Gaussian tissue LUT.

This independent entry point never calls geometry.train or a Gaussian optimizer.
"""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch

from appearance.lut import from_manual_config
from appearance.discriminator import TissuePatchDiscriminator
from appearance.patches import RealPatchBank, crop_render, file_hash
from appearance.scene import FrozenGaussianScene, save_rgb
from appearance.trainer import gan_step
from appearance.checkpoint import save_checkpoint, restore_checkpoint, state_equal


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ply', type=Path, required=True)
    parser.add_argument('--cameras', type=Path, required=True)
    parser.add_argument('--real-manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent/'configs/lut_l7.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=10, help='Total completed steps, including resumed steps')
    parser.add_argument('--knots', type=int, choices=[16, 32], default=16)
    parser.add_argument('--patch-size', type=int, choices=[64, 96, 128], default=64)
    parser.add_argument('--render-width', type=int, default=2048)
    parser.add_argument('--view-indices', nargs='+', type=int, default=[0, 20, 40, 60, 80, 100, 120, 140, 160])
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--minimum-tissue-purity', type=float, default=.98)
    parser.add_argument('--minimum-tissue-weight', type=float, default=.95)
    parser.add_argument('--mask-margin', type=int, default=2)
    parser.add_argument('--lut-lr', type=float, default=.01)
    parser.add_argument('--d-lr', type=float, default=.0001)
    parser.add_argument('--d-channels', type=int, default=32)
    parser.add_argument('--smooth-weight', type=float, default=.001)
    parser.add_argument('--anchor-weight', type=float, default=0.)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--preview-every', type=int, default=0, help='Periodic preview interval; zero saves start/final only')
    parser.add_argument('--seed', type=int, default=20261110)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--allow-unreviewed-data', action='store_true')
    parser.add_argument('--smoke-test', action='store_true', help='Engineering check limited to <=5 total steps')
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.save_every < 1:
        parser.error('Steps/batch size/save interval must be positive')
    if args.preview_every < 0:
        parser.error('Preview interval must be nonnegative')
    if not args.patch_size <= args.render_width <= 2048 or args.mask_margin < 0:
        parser.error('Invalid render size or margin')
    if not 0 < args.minimum_tissue_purity <= 1 or not 0 < args.minimum_tissue_weight <= 1:
        parser.error('Visibility thresholds must be in (0,1]')
    if min(args.lut_lr, args.d_lr) <= 0 or min(args.smooth_weight, args.anchor_weight) < 0:
        parser.error('Invalid learning rate or regularization weight')
    if args.smoke_test and args.steps > 5:
        parser.error('Smoke validation is limited to five total steps')
    if args.output.exists():
        parser.error('Use a new output directory, including for resumed runs')
    if not torch.cuda.is_available():
        parser.error('A working CUDA Gaussian rasterizer is required')
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if len(config['tissues']) != 1:
        raise ValueError('This first entry point learns one tissue; use one LUT row in config')
    spec = config['tissues'][0]
    bank = RealPatchBank(args.real_manifest, args.patch_size, spec['name'], 'cuda', args.allow_unreviewed_data)
    scene = FrozenGaussianScene(args.ply, args.cameras, spec['layer'])
    lut = from_manual_config(config, args.knots).cuda()
    discriminator = TissuePatchDiscriminator(1, args.d_channels).cuda()
    lut_optimizer = torch.optim.Adam(lut.parameters(), lr=args.lut_lr)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(0., .99))
    if {id(p) for g in lut_optimizer.param_groups for p in g['params']} != {id(lut.logits)}:
        raise RuntimeError('Unexpected LUT optimizer parameters')
    initial_rgb = lut.rgb.detach().clone()
    args.output.mkdir(parents=True, exist_ok=False)
    coverage = scene.prepare_views(args.view_indices, args.render_width, args.patch_size,
                                  args.minimum_tissue_purity, args.minimum_tissue_weight, args.mask_margin)
    (args.output/'view_coverage.json').write_text(json.dumps(coverage, indent=2), encoding='utf-8')
    if not scene.views:
        raise RuntimeError('No fully visible target-tissue patches. Inspect view_coverage.json; adjust view/resolution rather than silently lowering tissue purity.')
    immutable_options = {k: v for k,v in vars(args).items() if k not in
                         {'output', 'steps', 'resume', 'save_every', 'preview_every', 'ply', 'cameras', 'real_manifest', 'config'}}
    signature = {'model_sha256': scene.model_hash, 'camera_sha256': scene.camera_hash,
                 'real_data': bank.signature, 'lut_config': config, 'options': immutable_options,
                 'eligible_views': [(v['index'], len(v['boxes'])) for v in scene.views],
                 'implementation': {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                 'entry_sha256': file_hash(Path(__file__)), 'torch': torch.__version__, 'cuda': torch.version.cuda}
    step, history = 0, []
    if args.resume:
        state = restore_checkpoint(args.resume, lut, discriminator, lut_optimizer, d_optimizer, signature, rng)
        step, history = state['step'], state['history']
        initial_rgb = state['initial_rgb'].to('cuda')
        if args.steps <= step:
            raise ValueError('--steps must exceed the checkpoint completed-step count')
    config_out = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config_out.update({'signature': signature, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                       'gpu': torch.cuda.get_device_name(), 'start_step': step,
                       'dataset_review_accepted': bank.accepted,
                       'scope': 'engineering_smoke_only' if args.smoke_test else 'single-tissue_unpaired_experiment'})
    (args.output/'config.json').write_text(json.dumps(config_out, indent=2), encoding='utf-8')
    previews = scene.views[:min(3, len(scene.views))]
    (args.output/'previews').mkdir()
    with torch.no_grad():
        for view in previews:
            name = f'view_{view["index"]:03d}'
            save_rgb(args.output/'previews'/f'{name}_start.png', scene.draw(view, lut))
            Image.fromarray(view['mask'].numpy().astype(np.uint8)*255).save(args.output/'previews'/f'{name}_sampling_mask.png')
        save_rgb(args.output/'previews/original_sh_as_stored.png', scene.original(previews[0]), srgb=False)
    start_rgb = lut.rgb.detach().clone()
    run_started = time.monotonic()
    last_checkpoint, last_fake, last_real, last_tissue = None, None, None, None
    with (args.output/'metrics.jsonl').open('w', encoding='utf-8') as log:
        for current in range(step+1, args.steps+1):
            # Cycle available views so even a short run exercises shared 3D appearance.
            view = scene.views[(current-1) % len(scene.views)]
            picks = rng.choice(len(view['boxes']), size=args.batch_size,
                               replace=len(view['boxes']) < args.batch_size).tolist()
            boxes = [view['boxes'][i] for i in picks]
            rendered = scene.draw(view, lut)  # Continuous RGB graph, never PNG/NumPy in the loss path.
            fake = crop_render(rendered, boxes).clamp(0, 1)
            real, real_indices = bank.sample(args.batch_size, rng)
            tissue = torch.zeros(args.batch_size, dtype=torch.long, device='cuda')
            values = gan_step(lut, discriminator, lut_optimizer, d_optimizer, real, fake, tissue,
                              args.smooth_weight, initial_rgb, args.anchor_weight,
                              check_adversarial_gradient=current == step+1)
            values.update({'step': current, 'camera_index': view['index'], 'boxes_xyxy': boxes,
                           'real_patch_ids': [bank.rows[i]['id'] for i in real_indices],
                           'elapsed_seconds_this_run': time.monotonic()-run_started})
            history.append(values)
            log.write(json.dumps(values)+'\n')
            log.flush()
            last_fake, last_real, last_tissue = fake.detach(), real.detach(), tissue
            print(json.dumps({k: values[k] for k in ['step', 'camera_index', 'd_loss', 'g_adversarial', 'lut_gradient_norm']}), flush=True)
            if current % args.save_every == 0 or current == args.steps:
                scene.verify_frozen()
                bank.verify_unchanged()
                last_checkpoint = args.output/f'checkpoint_{current:06d}.pt'
                save_checkpoint(last_checkpoint, current, lut, discriminator, lut_optimizer, d_optimizer,
                                initial_rgb, signature, rng, history)
                live_curves = {'step': current, 'hu': lut.hu_knots.detach().cpu().tolist(),
                               'initial_linear_rgb': initial_rgb.cpu().tolist(),
                               'final_linear_rgb': lut.rgb.detach().cpu().tolist()}
                (args.output/'lut_curves_latest.json').write_text(json.dumps(live_curves, indent=2), encoding='utf-8')
            if args.preview_every and (current % args.preview_every == 0 or current == args.steps):
                with torch.no_grad():
                    for fixed_view in previews:
                        save_rgb(args.output/'previews'/f'step_{current:06d}_view_{fixed_view["index"]:03d}.png',
                                 scene.draw(fixed_view, lut))
    scene.verify_frozen()
    bank.verify_unchanged()
    changed = float((lut.rgb.detach()-start_rgb).abs().max())
    if not changed > 0:
        raise RuntimeError('LUT did not change during run')
    with torch.no_grad():
        for view in previews:
            save_rgb(args.output/'previews'/f'view_{view["index"]:03d}_final.png', scene.draw(view, lut))
        save_rgb(args.output/'previews/real_patch_example.png', last_real[0])
        save_rgb(args.output/'previews/generated_patch_last_step.png', last_fake[0])
    # Reconstruct models/optimizers and verify saved tensors and rendered output.
    check_lut = from_manual_config(config, args.knots).cuda()
    check_d = TissuePatchDiscriminator(1, args.d_channels).cuda()
    check_g_optimizer = torch.optim.Adam(check_lut.parameters(), lr=args.lut_lr)
    check_d_optimizer = torch.optim.Adam(check_d.parameters(), lr=args.d_lr, betas=(0., .99))
    restored = restore_checkpoint(last_checkpoint, check_lut, check_d, check_g_optimizer, check_d_optimizer,
                                  signature, np.random.default_rng())
    for original, loaded in [(lut.state_dict(), check_lut.state_dict()),
                             (discriminator.state_dict(), check_d.state_dict()),
                             (lut_optimizer.state_dict(), check_g_optimizer.state_dict()),
                             (d_optimizer.state_dict(), check_d_optimizer.state_dict())]:
        if not state_equal(original, loaded):
            raise RuntimeError('Checkpoint state roundtrip mismatch')
    with torch.no_grad():
        torch.testing.assert_close(scene.draw(previews[0], lut), scene.draw(previews[0], check_lut), rtol=0, atol=0)
        torch.testing.assert_close(discriminator(last_fake, last_tissue), check_d(last_fake, last_tissue), rtol=0, atol=0)
    curves = {'hu': lut.hu_knots.detach().cpu().tolist(), 'initial_linear_rgb': initial_rgb.cpu().tolist(),
              'final_linear_rgb': lut.rgb.detach().cpu().tolist()}
    (args.output/'lut_curves.json').write_text(json.dumps(curves, indent=2), encoding='utf-8')
    report = {'scope': config_out['scope'], 'completed_steps': args.steps, 'start_step': step,
              'eligible_views': len(scene.views), 'real_patches': len(bank.rows),
              'real_source_frames': len(bank.groups), 'layer': spec['layer'], 'knots': args.knots,
              'dataset_review_accepted': bank.accepted, 'allow_unreviewed_data': args.allow_unreviewed_data,
              'model_fields_bitwise_unchanged': True, 'model_and_real_input_hashes_unchanged': True,
              'checkpoint_model_optimizer_render_roundtrip': True,
              'max_rgb_control_change_this_run': changed,
              'adversarial_gradient_norm_first_step': history[step]['adversarial_gradient_norm'],
              'hu_below_range': int((scene.hu < lut.hu_bounds[0, 0]).sum()),
              'hu_above_range': int((scene.hu > lut.hu_bounds[0, 1]).sum()),
              'checkpoint': last_checkpoint.name, 'last_losses': history[-1],
              'elapsed_seconds_this_run': time.monotonic()-run_started,
              'interpretation': 'Loss/gradient validation does not establish realism or cross-patient generalization. '
                                'Patch pixel size is fixed but physical scale and exposure remain uncalibrated. '
                                'Selected LUT uses linear RGB; historical SH context colorimetry is unverified.'}
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
