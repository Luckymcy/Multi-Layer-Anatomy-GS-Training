"""Bounded local surface densification pilot; never overwrites an input model."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw
from plyfile import PlyData, PlyElement
import torch
from torch import nn
import torch.nn.functional as F

from appearance.scene import FrozenGaussianScene, FIELDS, load_camera
from appearance.patches import file_hash
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.general_utils import build_rotation
from utils.loss_utils import ssim


def split_ellipsoid(xyz, scales, rotation, cap):
    """Deterministic binary moment-matching split along each oversized axis."""
    centers = xyz[None].copy()
    child_scale = scales.copy()
    while child_scale.max() > cap:
        axis = int(child_scale.argmax())
        child_scale[axis] /= np.sqrt(2)
        offset = rotation[:, axis]*child_scale[axis]
        centers = np.concatenate([centers-offset, centers+offset])
        if len(centers) > 1024:
            raise ValueError('Unreasonably large parent: split budget exceeded')
    return centers, child_scale


def edge_loss(image, target, mask):
    dx = ((image[:, :, 1:]-image[:, :, :-1])-(target[:, :, 1:]-target[:, :, :-1])).abs()
    dy = ((image[:, 1:, :]-image[:, :-1, :])-(target[:, 1:, :]-target[:, :-1, :])).abs()
    mx, my = mask[:, 1:]*mask[:, :-1], mask[1:]*mask[:-1]
    return (dx*mx).sum()/(3*mx.sum().clamp_min(1)) + (dy*my).sum()/(3*my.sum().clamp_min(1))


def make_model(fields, degree):
    pc = GaussianModel(degree)
    for name, value in fields.items():
        setattr(pc, name, value)
    pc.active_sh_degree = degree
    return pc


def large_coverage(pc, camera, pipe, background, threshold=5):
    """Appearance-independent oversized-Gaussian contribution diagnostic."""
    with torch.no_grad():
        large = (pc.get_scaling.max(1).values > threshold)[:, None].expand(-1, 3).float()
        package = render(camera, pc, pipe, background, override_color=large)
        alpha, mass = package['accumulation'], package['render'][0]
        valid = alpha > .95
        return dict(contribution_fraction=float(mass.sum()/alpha.sum()),
                    majority_area_fraction=float(((mass/alpha.clamp_min(.01) > .5) & valid).sum()/valid.sum()))


def differentiable_alpha(pc, camera, pipe):
    """White splats on black equal alpha; RGB backward supports geometry/opacity.

    The pinned rasterizer's separate accumulation output has no alpha backward.
    Do not use that output as an optimization target.
    """
    return render(camera, pc, pipe, torch.zeros(3, device=pc.get_xyz.device),
                  override_color=torch.ones_like(pc.get_xyz))['render'][0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['ply', 'cameras', 'teachers', 'surface', 'output']:
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--max-parents', type=int, default=512)
    ap.add_argument('--parent-sigma', type=float, default=5)
    ap.add_argument('--child-sigma', type=float, default=2.5)
    ap.add_argument('--initialization', choices=['preserve', 'surface'], default='preserve',
                    help='Preserve original Gaussian support, or experimental hard CT-surface projection')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skip-control', action='store_true', help='Run refinement only; omit the independent color-only control')
    ap.add_argument('--model-name', default='L4_refined.ply')
    ap.add_argument('--lr-final-factor', type=float, default=1., help='Exponential LR decay, 1 keeps pilot behavior')
    ap.add_argument('--checkpoint', type=Path, help='Optional recovery state outside the compact result directory')
    args = ap.parse_args()
    if args.output.exists() or args.steps < 1 or args.max_parents < 1 or not 0 < args.child_sigma < args.parent_sigma:
        ap.error('Use a new output, positive budgets and child-sigma < parent-sigma')
    if Path(args.model_name).name != args.model_name or not args.model_name.endswith('.ply') or not 0 < args.lr_final_factor <= 1:
        ap.error('Expected a PLY filename and 0 < lr-final-factor <= 1')
    if args.checkpoint and (args.checkpoint.exists() or args.checkpoint.resolve().is_relative_to(args.output.resolve())):
        ap.error('Checkpoint must be new and outside the compact output directory')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    scene = FrozenGaussianScene(args.ply, args.cameras, 4)
    original = {k: getattr(scene.pc, k)[scene.indices].detach().clone() for k in FIELDS}
    base = make_model(original, scene.pc.max_sh_degree)
    surface = np.load(args.surface)
    vertices, normals = surface['vertices'], surface['normals']
    tree = cKDTree(vertices)
    xyz = original['_xyz'].cpu().numpy()
    scales = original['_scaling'].exp().cpu().numpy()
    rotations = build_rotation(original['_rotation']).cpu().numpy()
    distance, _ = tree.query(xyz)
    train = (args.teachers/'train.txt').read_text().split()
    test = (args.teachers/'test.txt').read_text().split()
    assert train and test and not set(train) & set(test)
    entries = {e['img_name']+'.png': e for e in scene.camera_entries}
    views = {}
    for name in train+test:
        a = np.array(Image.open(args.teachers/'images'/name).convert('RGBA'), dtype=np.float32)/255
        alpha = torch.tensor(a[:, :, 3], device='cuda')
        target = torch.tensor(a[:, :, :3], device='cuda').permute(2, 0, 1)*alpha
        mask = -F.max_pool2d(-alpha[None, None], 5, 1, 2)[0, 0] > .99
        views[name] = dict(camera=load_camera(entries[name], a.shape[1], base.get_xyz), target=target, alpha=alpha, mask=mask)

    def draw(pc, view, colors=None):
        return render(view['camera'], pc, scene.pipe, scene.background, override_color=colors)

    # Select only using training cameras, with actual composited contribution.
    contribution = torch.zeros(len(xyz), device='cuda')
    for name in train:
        colors = torch.ones_like(base.get_xyz, requires_grad=True)
        draw(base, views[name], colors)['render'][0].sum().backward()
        contribution += colors.grad[:, 0].detach()/len(train)
    score = contribution.cpu().numpy()
    candidates = np.flatnonzero((scales.max(1) > args.parent_sigma) & (distance < 5) & (score > .05))
    selected = candidates[np.argsort(score[candidates])[::-1][:args.max_parents]]
    if not len(selected):
        raise RuntimeError('No visible large near-surface Gaussians')
    keep = np.setdiff1d(np.arange(len(xyz)), selected)
    parent_ids, child_xyz, child_scales, child_rots, child_opacity = [], [], [], [], []
    for parent in selected:
        centers, child_scale = split_ellipsoid(xyz[parent], scales[parent], rotations[parent], args.child_sigma)
        _, anchor_ids = tree.query(centers)
        n = normals[anchor_ids].copy()
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        order = np.argsort(child_scale)
        tangent = rotations[parent, :, order[2]][None].repeat(len(n), axis=0)
        tangent -= (tangent*n).sum(1, keepdims=True)*n
        bad = np.linalg.norm(tangent, axis=1) < 1e-6
        if bad.any():
            tangent[bad] = np.cross(n[bad], np.eye(3)[np.argmin(np.abs(n[bad]), axis=1)])
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        r = np.zeros((len(n), 3, 3))
        r[:, :, order[0]], r[:, :, order[2]] = n, tangent
        r[:, :, order[1]] = np.cross(tangent, n)
        r[np.linalg.det(r) < 0, :, order[1]] *= -1
        q = Rotation.from_matrix(r).as_quat()[:, [3, 0, 1, 2]]
        child_scale[order[0]] = np.clip(child_scale[order[0]], .03, .3)
        if args.initialization == 'preserve':
            # Preserve the original ellipsoid's spatial moments and orientation.
            # CT guides only the normal-motion bound, not a jump to a new surface.
            _, child_scale = split_ellipsoid(xyz[parent], scales[parent], rotations[parent], args.child_sigma)
            q = np.tile(original['_rotation'][parent].cpu().numpy(), (len(n), 1))
        parent_area = np.prod(np.sort(scales[parent])[1:])
        child_area = np.prod(np.sort(child_scale)[1:])
        parent_alpha = float(base.get_opacity[parent])
        # Approximate optical-depth mass allocation, NOT exact alpha equivalence.
        opacity = np.clip(1-(1-parent_alpha)**(parent_area/(len(n)*child_area)), .05, .95)
        parent_ids.extend([parent]*len(n))
        child_xyz.extend(centers if args.initialization == 'preserve' else vertices[anchor_ids])
        child_scales.extend(np.tile(child_scale, (len(n), 1)))
        child_rots.extend(q); child_opacity.extend([opacity]*len(n))
    parent_ids = np.array(parent_ids)
    def tensor(a):
        return torch.tensor(np.array(a), dtype=torch.float32, device='cuda')
    child_initial = {k: v[parent_ids].clone() for k, v in original.items()}
    child_initial['_xyz'] = tensor(child_xyz)
    child_initial['_scaling'] = tensor(child_scales).log()
    child_initial['_rotation'] = tensor(child_rots)
    child_initial['_opacity'] = torch.logit(tensor(child_opacity)[:, None])
    print(json.dumps({'selected_parents': len(selected), 'children': len(parent_ids), 'train': len(train), 'test': len(test)}), flush=True)

    def evaluate(pc, names):
        rows = []
        with torch.no_grad():
            for name in names:
                v = views[name]; p = draw(pc, v); im = p['render']; a = p['accumulation']
                truth = v['alpha'] > .5; predicted = a > .5
                rows.append(dict(foreground_l1=float((im-v['target']).abs()[:, truth].mean()),
                                 ssim=float(ssim(im, v['target'])), edge_l1=float(edge_loss(im, v['target'], v['mask'])),
                                 silhouette_iou=float((predicted & truth).sum()/(predicted | truth).sum())))
        return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

    display = ['view_000001.png', 'view_000060.png', 'view_000117.png']
    with torch.no_grad():
        before = {n: draw(base, views[n])['render'].cpu() for n in display}
    baseline = evaluate(base, test)
    report = dict(source_model=str(args.ply.resolve()), source_sha256=scene.model_hash,
                  surface_sha256=file_hash(args.surface), camera_sha256=scene.camera_hash,
                  teacher_hashes={n: file_hash(args.teachers/'images'/n) for n in train+test},
                  config={k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  train_views=len(train), test_views=len(test), baseline=baseline,
                  original_L4_count=len(xyz), selected_parents=len(selected), children=len(parent_ids),
                  selection='train-view contribution > .05px, max sigma > threshold, nearest CT vertex <5mm; rank by contribution',
                  loss='0.8 RGB L1 + 0.2 (1-SSIM) + 0.2 interior image-gradient L1 + normalized alpha L1 (surface:0.1 full-frame; preserve:0.3 foreground+dilated border) + 0.01 child anchor displacement squared',
                  alpha_gradient='White RGB render on black, equivalent alpha forward with RGB backward; fixed after the historical L4new run',
                  constraints='Only selected children geometry/SH update; displacement <=0.6mm; max sigma bounded by child-sigma; alpha <=0.98. Surface initialization anchors at CT vertices, normal-axis <=0.3mm. Preserve initialization retains ellipsoid moments/orientation and original normal thickness, with additional CT-normal motion <=0.1mm.',
                  hu_status='Child HU fields inherited from parent, NOT resampled; pilot only, HU mapping not validated',
                  note='Synthetic L4 geometry refinement, not real-image appearance training. Optional independent color-only control uses same camera sequence and steps.')
    final_models = {}
    for arm in (['surface_split'] if args.skip_control else ['color_control', 'surface_split']):
        init = {k: v[selected].clone() for k, v in original.items()} if arm == 'color_control' else child_initial
        moving = {k: nn.Parameter(v.clone(), requires_grad=(k in ('_features_dc', '_features_rest') or (arm == 'surface_split' and k in ('_xyz', '_scaling', '_opacity')))) for k, v in init.items()}
        frozen = {k: v[keep].clone() for k, v in original.items()}
        def assembled():
            return make_model({k: torch.cat([frozen[k], moving[k]]) for k in FIELDS}, base.max_sh_degree)
        groups = [{'params': [moving['_features_dc']], 'lr': .0025}, {'params': [moving['_features_rest']], 'lr': .000125}]
        if arm == 'surface_split':
            groups += [{'params': [moving['_xyz']], 'lr': .002}, {'params': [moving['_scaling']], 'lr': .001}, {'params': [moving['_opacity']], 'lr': .01}]
        optimizer = torch.optim.Adam(groups, eps=1e-15)
        initial_lrs = [group['lr'] for group in optimizer.param_groups]
        short_axis = init['_scaling'].argmin(1, keepdim=True)
        short_cap = init['_scaling'].gather(1, short_axis).clamp(min=np.log(.01))
        _, near_ids = tree.query(init['_xyz'].cpu().numpy())
        motion_normals = F.normalize(tensor(normals[near_ids]), dim=1)
        initial_metrics = evaluate(assembled(), test)
        rng = random.Random(args.seed); stack = []; history = []
        for step in range(1, args.steps+1):
            factor = args.lr_final_factor**((step-1)/max(1, args.steps-1))
            for group, initial_lr in zip(optimizer.param_groups, initial_lrs):
                group['lr'] = initial_lr*factor
            if not stack:
                stack = rng.sample(train, len(train))
            view = views[stack.pop()]
            optimizer.zero_grad(set_to_none=True)
            pc = assembled(); p = draw(pc, view); im = p['render']
            loss = .8*(im-view['target']).abs().mean() + .2*(1-ssim(im, view['target']))
            loss = loss + .2*edge_loss(im, view['target'], view['mask'])
            if arm == 'surface_split':
                alpha = differentiable_alpha(pc, view['camera'], scene.pipe)
            else:
                alpha = p['accumulation'].detach()  # Geometry is fixed in the color-only control.
            alpha_error = (alpha-view['alpha']).abs()
            if args.initialization == 'preserve':
                alpha_region = F.max_pool2d(view['alpha'][None, None], 21, 1, 10)[0, 0] > .01
                loss = loss + .3*alpha_error[alpha_region].mean()
            else:
                loss = loss + .1*alpha_error.mean()
            if arm == 'surface_split':
                loss = loss + .01*(moving['_xyz']-init['_xyz']).square().sum(1).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            if step == 1:
                gradients = {k: float(v.grad.norm()) for k, v in moving.items() if v.requires_grad}
                if not all(np.isfinite(list(gradients.values()))) or not any(v > 0 for v in gradients.values()):
                    raise RuntimeError('Invalid gradients')
            optimizer.step()
            with torch.no_grad():
                if arm == 'surface_split':
                    delta = moving['_xyz']-init['_xyz']
                    if args.initialization == 'preserve':
                        normal_step = (delta*motion_normals).sum(1, keepdim=True)
                        delta = delta+(normal_step.clamp(-.1, .1)-normal_step)*motion_normals
                    moving['_xyz'].copy_(init['_xyz']+delta*(.6/delta.norm(dim=1, keepdim=True).clamp_min(.6)))
                    moving['_scaling'].clamp_(min=np.log(.01), max=np.log(args.child_sigma))
                    small = moving['_scaling'].gather(1, short_axis)
                    small = torch.minimum(small, short_cap) if args.initialization == 'preserve' else small.clamp(max=np.log(.3))
                    moving['_scaling'].scatter_(1, short_axis, small)
                    moving['_opacity'].clamp_(max=float(np.log(.98/.02)))
            if step % 100 == 0 or step == args.steps:
                row = dict(arm=arm, step=step, loss=float(loss))
                history.append(row); print(json.dumps(row), flush=True)
            if args.checkpoint and arm == 'surface_split' and (step % 1000 == 0 or step == args.steps):
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(step=step, source_sha256=scene.model_hash, selected=selected,
                                keep=keep, parent_ids=parent_ids, moving={k: v.detach().cpu() for k, v in moving.items()},
                                initial={k: v.detach().cpu() for k, v in init.items()}, optimizer=optimizer.state_dict(),
                                camera_stack=stack, rng_state=rng.getstate(), config=report['config']), args.checkpoint)
        pc = assembled()
        report[arm] = dict(before=initial_metrics, after=evaluate(pc, test), first_gradients=gradients, history=history,
                           display_views={n: evaluate(pc, [n]) for n in display})
        final_models[arm] = {k: v.detach().clone() for k, v in ((k, getattr(pc, k)) for k in FIELDS)}
        for k in FIELDS:
            torch.testing.assert_close(getattr(pc, k)[:len(keep)], original[k][keep], atol=0, rtol=0)

    # Output only a full-scene PLY, comparison and report; other layers byte-identical.
    args.output.mkdir(parents=True)
    full = PlyData.read(str(args.ply), mmap=False)['vertex'].data
    selected_full = scene.indices.cpu().numpy()[selected]
    kept_full = np.setdiff1d(np.arange(len(full)), selected_full)
    children = full[scene.indices.cpu().numpy()[parent_ids]].copy()
    final = final_models['surface_split']
    mapping = {'_xyz': ['x', 'y', 'z'], '_scaling': ['scale_'+str(i) for i in range(3)],
               '_rotation': ['rot_'+str(i) for i in range(4)], '_opacity': ['opacity']}
    for field, keys in mapping.items():
        a = final[field][len(keep):].cpu().numpy()
        for i, key in enumerate(keys):
            children[key] = a[:, i]
    for field, prefix in [('_features_dc', 'f_dc_'), ('_features_rest', 'f_rest_')]:
        a = final[field][len(keep):].transpose(1, 2).reshape(len(parent_ids), -1).cpu().numpy()
        for i in range(a.shape[1]):
            children[prefix+str(i)] = a[:, i]
    # PLY nx/ny/nz are unused by the renderer; set child metadata to anchor normals.
    _, anchor = tree.query(np.array(child_xyz))
    for i, key in enumerate(['nx', 'ny', 'nz']):
        children[key] = normals[anchor, i]
    output_array = np.concatenate([full[kept_full], children])
    destination = args.output/args.model_name
    PlyData([PlyElement.describe(output_array, 'vertex')], text=False).write(str(destination))
    reloaded = PlyData.read(str(destination))['vertex'].data
    np.testing.assert_array_equal(reloaded[:len(kept_full)], full[kept_full])
    reload_scene = FrozenGaussianScene(destination, args.cameras, 4)
    reload_pc = make_model({k: getattr(reload_scene.pc, k)[reload_scene.indices] for k in FIELDS}, base.max_sh_degree)
    columns = 3 if args.skip_control else 4
    sheet = Image.new('RGB', (columns*440, len(display)*480), '#202020'); painter = ImageDraw.Draw(sheet)
    with torch.no_grad():
        for row, name in enumerate(display):
            view = views[name]
            fitted = make_model(final_models['surface_split'], base.max_sh_degree)
            after = draw(fitted, view)['render']
            torch.testing.assert_close(after, draw(reload_pc, view)['render'], atol=2e-5, rtol=2e-4)
            yy, xx = torch.where(view['alpha'] > .5)
            cx, cy = int((xx.min()+xx.max())/2), int((yy.min()+yy.max())/2)
            box = (cx-150, cy-150, cx+150, cy+150)
            panels = [('Teacher', view['target']), ('Original', before[name])]
            if not args.skip_control:
                control = make_model(final_models['color_control'], base.max_sh_degree)
                panels.append(('Color-only control', draw(control, view)['render']))
            panels.append(('Local surface split', after))
            for col, (label, rgb) in enumerate(panels):
                a = (rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).round().astype('uint8')
                sheet.paste(Image.fromarray(a).crop(box).resize((440, 440)), (col*440, row*480+40))
                painter.text((col*440+8, row*480+8), name+' / '+label, fill='white')
    sheet.save(args.output/'comparison.jpg', quality=94)
    scene.verify_frozen()
    report.update(output_sha256=file_hash(destination), other_layers_and_unselected_L4_bitwise_unchanged=True,
                  final_L4_count=len(keep)+len(parent_ids), export_render_verified=True,
                  child_anchor_displacement_max_mm=float((final['_xyz'][len(keep):]-child_initial['_xyz']).norm(dim=1).max()))
    report['status'] = 'pilot_only; increased spatial capacity is not evidence of improved natural texture or HU validity'
    report['large_gaussian_diagnostic'] = {n: {
        'before': large_coverage(base, views[n]['camera'], scene.pipe, scene.background),
        'after': large_coverage(reload_pc, views[n]['camera'], scene.pipe, scene.background)} for n in display}
    (args.output/'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['baseline', 'color_control', 'surface_split'] if k in report}), flush=True)


if __name__ == '__main__':
    main()
