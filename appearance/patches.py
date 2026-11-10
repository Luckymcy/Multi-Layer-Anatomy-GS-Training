"""Verified real patch loading and visibility-based differentiable crop extraction."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .lut import srgb_to_linear


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


class RealPatchBank:
    def __init__(self, manifest, size, tissue_name, device, allow_unreviewed=False):
        self.manifest = Path(manifest).resolve()
        self.size = size
        summary_path = self.manifest.parent/'summary.json'
        if not summary_path.is_file():
            raise ValueError('Dataset summary.json is required to record review status')
        self.summary = json.loads(summary_path.read_text(encoding='utf-8'))
        self.accepted = self.summary.get('training_ready') is True
        if not self.accepted and not allow_unreviewed:
            raise ValueError('Dataset is pending review. Use --allow-unreviewed-data only for an explicitly acknowledged engineering check.')
        all_rows = json.loads(self.manifest.read_text(encoding='utf-8'))
        self.rows = [r for r in all_rows if r['size'] == size and r['tissue_label'] == tissue_name]
        if not self.rows:
            raise ValueError(f'No {size}px patches for tissue {tissue_name!r}')
        images, ids, groups, references = [], set(), defaultdict(list), []
        self.input_hashes = {self.manifest: file_hash(self.manifest), summary_path: file_hash(summary_path)}
        for index, row in enumerate(self.rows):
            if row['id'] in ids:
                raise ValueError('Duplicate real patch ID')
            ids.add(row['id'])
            if row.get('split') in ('test', 'validation', 'val'):
                raise ValueError('Evaluation patches cannot be used for training')
            if row.get('valid_fraction') != 1.:
                raise ValueError('Real crops must be fully inside the supplied valid region')
            path = (self.manifest.parent/row['image']).resolve()
            if self.manifest.parent not in path.parents:
                raise ValueError('Patch path escapes dataset')
            checksum = file_hash(path)
            if checksum != row['image_sha256']:
                raise ValueError(f'Real patch checksum mismatch: {row["id"]}')
            self.input_hashes[path] = checksum
            for key in ('source_image', 'source_annotation'):
                original, expected = Path(row[key]), row[key+'_sha256']
                if original not in self.input_hashes:
                    if not original.is_file() or file_hash(original) != expected:
                        raise ValueError(f'Changed/missing source input; regenerate dataset: {key}')
                    self.input_hashes[original] = expected
                elif self.input_hashes[original] != expected:
                    raise ValueError('Inconsistent source checksums in patch manifest')
            with Image.open(path) as im:
                if im.size != (size, size) or im.mode != 'RGB':
                    raise ValueError('Real patch size/mode mismatch; no silent resizing/conversion')
                images.append(torch.from_numpy(np.array(im, copy=True)).permute(2, 0, 1).float()/255)
            group = row.get('frame_id')
            if not group or not row.get('patient_group'):
                raise ValueError('Source frame and patient grouping are required')
            groups[group].append(index)
            references.append((row['id'], checksum))
        self.linear_rgb = srgb_to_linear(torch.stack(images).to(device))
        self.groups = list(groups.values())
        self.group_names = list(groups)
        self.signature = {'manifest_sha256': file_hash(self.manifest),
                          'summary_sha256': file_hash(summary_path), 'size': size,
                          'tissue': tissue_name, 'patches': references,
                          'dataset_accepted': self.accepted}

    def sample(self, count, rng):
        # Equal frame probability; a frame with more crops does not dominate.
        indices = [int(rng.choice(self.groups[int(rng.integers(len(self.groups)))])) for _ in range(count)]
        return self.linear_rgb[indices], indices

    def verify_unchanged(self):
        if any(file_hash(path) != checksum for path, checksum in self.input_hashes.items()):
            raise RuntimeError('Real dataset/source inputs changed during run')
        return True


def eligible_boxes(target_weight, total_alpha, size, purity=.98, minimum_weight=.95, margin=2):
    """Mask from full-scene compositing contribution, not Gaussian centers/radii.

    The indicator render measures sum_i(T_i*alpha_i) for target tissue under
    all-layer occlusion; total_alpha is used only in this frozen sampling path.
    """
    target_weight, total_alpha = target_weight.squeeze(), total_alpha.squeeze()
    if target_weight.ndim != 2 or target_weight.shape != total_alpha.shape:
        raise ValueError('Expected matching HxW contribution/alpha maps')
    valid = ((target_weight >= minimum_weight) & (total_alpha >= minimum_weight)
             & (target_weight/total_alpha.clamp_min(1e-8) >= purity))
    if margin:
        invalid = F.pad((~valid).float()[None, None], (margin,)*4, value=1.)
        valid = F.max_pool2d(invalid, 2*margin+1, stride=1)[0, 0] == 0
    if min(valid.shape) < size:
        return [], valid
    stride = max(1, size//2)
    complete = F.avg_pool2d(valid.float()[None, None], size, stride=stride)[0, 0] == 1
    points = complete.nonzero(as_tuple=False).cpu().tolist()
    boxes = [(x*stride, y*stride, x*stride+size, y*stride+size) for y, x in points]
    return boxes, valid


def crop_render(image, boxes):
    if not boxes:
        raise ValueError('No generated crop boxes')
    return torch.stack([image[:, y0:y1, x0:x1] for x0, y0, x1, y1 in boxes])
