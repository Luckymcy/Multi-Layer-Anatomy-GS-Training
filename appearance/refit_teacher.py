"""Refit one layer's SH to paired synthetic teachers, preserving the full source PLY."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData
import torch

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from utils.loss_utils import ssim
from utils.sh_utils import eval_sh
from .scene import FrozenGaussianScene, FIELDS, load_camera
from .patches import file_hash


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ply',type=Path,required=True)
    ap.add_argument('--cameras',type=Path,required=True)
    ap.add_argument('--teachers',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=5000)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--layer',type=int,default=4)
    ap.add_argument('--context',action='store_true',help='Render frozen other layers for composite teachers')
    ap.add_argument('--model-name',default='L4_new.ply')
    args=ap.parse_args()
    if args.output.exists() or args.steps<1: ap.error('Use new output and positive steps')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if Path(args.model_name).name != args.model_name: ap.error('Model name must be a filename')
    scene=FrozenGaussianScene(args.ply,args.cameras,args.layer)
    pc=GaussianModel(scene.pc.max_sh_degree)
    for field in FIELDS:
        setattr(pc,field,torch.nn.Parameter(getattr(scene.pc,field)[scene.indices].detach().clone(),
            requires_grad=field in ('_features_dc','_features_rest')))
    pc.active_sh_degree=pc.max_sh_degree
    frozen={k:getattr(pc,k).detach().clone() for k in FIELDS if k not in ('_features_dc','_features_rest')}
    initial_dc=pc._features_dc.detach().clone(); initial_rest=pc._features_rest.detach().clone()
    by_name={entry['img_name']+'.png':entry for entry in scene.camera_entries}
    views={}
    for path in sorted((args.teachers/'images').glob('*.png')):
        entry=by_name[path.name]
        rgba=np.array(Image.open(path).convert('RGBA'),dtype=np.float32)/255
        target=torch.tensor(rgba[:,:,:3]*rgba[:,:,3:],device='cuda').permute(2,0,1)
        camera=load_camera(entry,rgba.shape[1],pc.get_xyz)
        direction=torch.nn.functional.normalize(pc.get_xyz-camera.camera_center,dim=-1)
        views[path.name]=dict(camera=camera,target=target,direction=direction)
        if args.context:
            with torch.no_grad():
                full_direction=torch.nn.functional.normalize(scene.pc.get_xyz-camera.camera_center,dim=-1)
                views[path.name]['context']=(eval_sh(scene.pc.max_sh_degree,
                    scene.pc.get_features.transpose(1,2),full_direction)+.5).clamp_min(0)
    train=(args.teachers/'train.txt').read_text().split()
    test=(args.teachers/'test.txt').read_text().split()
    assert not set(train)&set(test) and set(train)|set(test)==set(views)
    def draw(view):
        raw=eval_sh(pc.max_sh_degree,pc.get_features.transpose(1,2),view['direction'])+.5
        # Exact normal SH forward color, with straight-through lower clamp to
        # allow recovery of inherited negative SH colors in the old black side.
        color=raw+(raw.clamp_min(0)-raw).detach()
        if args.context:
            color=view['context'].index_copy(0,scene.indices,color)
        return render(view['camera'],scene.pc if args.context else pc,scene.pipe,scene.background,override_color=color)['render']
    def evaluate(names):
        with torch.no_grad():
            values=[float((draw(views[n])-views[n]['target']).abs().mean()) for n in names]
        return float(np.mean(values))
    display_names=['view_000001.png','view_000060.png','view_000117.png']
    with torch.no_grad(): before={n:draw(views[n]).cpu() for n in display_names}
    baseline=dict(train_l1=evaluate(train),test_l1=evaluate(test))
    print(json.dumps(dict(baseline=baseline,train_views=len(train),test_views=len(test))),flush=True)
    optimizer=torch.optim.Adam([{'params':[pc._features_dc],'lr':.0025},
                                {'params':[pc._features_rest],'lr':.0025/20}],eps=1e-15)
    stack=[]; history=[]; first_gradient=None
    for step in range(1,args.steps+1):
        if not stack: stack=random.sample(train,len(train))
        name=stack.pop(); view=views[name]
        optimizer.zero_grad(set_to_none=True)
        image=draw(view); l1=(image-view['target']).abs().mean()
        loss=.8*l1+.2*(1-ssim(image,view['target']))
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
        loss.backward()
        if first_gradient is None:
            first_gradient=float(pc._features_dc.grad.norm())
            if not np.isfinite(first_gradient) or first_gradient<=0: raise RuntimeError('No SH gradient')
        optimizer.step()
        if step%250==0 or step==args.steps:
            row=dict(step=step,loss=float(loss),l1=float(l1))
            history.append(row); print(json.dumps(row),flush=True)
    final=dict(train_l1=evaluate(train),test_l1=evaluate(test))
    scene.verify_frozen()
    for key,value in frozen.items():
        if not torch.equal(value,getattr(pc,key)) or getattr(pc,key).grad is not None:
            raise RuntimeError('Structure changed')
    # Full-scene output, retaining every field and original row order.
    ply=PlyData.read(str(args.ply),mmap=False); vertices=ply['vertex'].data
    original=vertices.copy(); selected=scene.indices.cpu().numpy()
    dc=pc._features_dc.detach().cpu().numpy().reshape(-1,3)
    rest=pc._features_rest.detach().cpu().numpy().transpose(0,2,1).reshape(len(selected),-1)
    for i in range(3): vertices[f'f_dc_{i}'][selected]=dc[:,i]
    for i in range(rest.shape[1]): vertices[f'f_rest_{i}'][selected]=rest[:,i]
    args.output.mkdir(parents=True)
    destination=args.output/args.model_name; ply.write(str(destination))
    reloaded=PlyData.read(str(destination))['vertex'].data
    other=original['layer']!=args.layer
    for key in original.dtype.names:
        np.testing.assert_array_equal(reloaded[key][other],original[key][other])
        if not key.startswith(('f_dc_','f_rest_')):
            np.testing.assert_array_equal(reloaded[key],original[key])
    sheet=Image.new('RGB',(1536,len(display_names)*550),'#181818'); text=ImageDraw.Draw(sheet)
    with torch.no_grad():
        # Verify export layout and standard CUDA SH forward match custom training forward.
        exported=GaussianModel(pc.max_sh_degree); exported.load_ply(str(destination))
        for row,name in enumerate(display_names):
            view=views[name]; after=draw(view)
            standard=render(view['camera'],exported if args.context else pc,scene.pipe,scene.background)['render']
            torch.testing.assert_close(after,standard,atol=2e-5,rtol=2e-4)
            all_sh=exported.get_features[scene.indices]
            torch.testing.assert_close(all_sh,pc.get_features,atol=0,rtol=0)
            for col,(label,tensor) in enumerate((('New teacher',view['target']),(f'Original L{args.layer}',before[name]),(f'Refitted L{args.layer}',after))):
                a=(tensor.detach().cpu().clamp(0,1).permute(1,2,0).numpy()*255).round().astype(np.uint8)
                sheet.paste(Image.fromarray(a),(col*512,row*550+30))
                text.text((col*512+8,row*550+8),name+' / '+label,fill='white')
    sheet.save(args.output/'comparison.jpg',quality=94)
    report=dict(task=f'Paired synthetic-teacher SH refit of L{args.layer} within full layered model',
        source_model=str(args.ply.resolve()),source_sha256=scene.model_hash,
        teachers=str(args.teachers.resolve()),teacher_manifest_sha256=file_hash(args.teachers/'manifest.json'),
        camera_sha256=scene.camera_hash,output_model=args.model_name,output_sha256=file_hash(destination),
        steps=args.steps,seed=args.seed,train_views=len(train),test_views=len(test),
        loss='0.8 RGB L1 + 0.2 (1-SSIM)',color_space='Original geometry-training RGB convention; no extra gamma conversion',
        optimizer=dict(dc_lr=.0025,rest_lr=.000125),first_dc_gradient_norm=first_gradient,
        lower_clamp_gradient='Straight-through: normal nonnegative SH forward, identity derivative below zero',
        selected_gaussians=len(selected),geometry_opacity_hu_order_unchanged=True,other_layers_bitwise_unchanged=True,
        baseline=baseline,final=final,history=history,
        dc_change_mean=float((pc._features_dc-initial_dc).abs().mean()),
        rest_change_mean=float((pc._features_rest-initial_rest).abs().mean()),
        layer=args.layer,render_frozen_context=args.context,
        note=f'Full layered PLY; only L{args.layer} SH changed. Frozen context: {args.context}. Synthetic illumination refitting, not real-image appearance learning or geometry retraining.')
    (args.output/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(final=final,output=str(args.output))),flush=True)


if __name__=='__main__': main()
