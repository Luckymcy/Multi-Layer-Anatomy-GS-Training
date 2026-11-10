"""Check frozen layers, finite Gaussian fields, and optional HU augmentation."""
import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData


def read(path):
    return PlyData.read(str(path))['vertex'].data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--iteration', type=int, default=30000)
    args = parser.parse_args()
    directory = args.model / 'point_cloud' / f'iteration_{args.iteration}'
    model = read(directory / 'point_cloud.ply')
    base = read(args.base / 'point_cloud/iteration_30000/point_cloud.ply')
    layers, counts = np.unique(model['layer'], return_counts=True)
    if set(layers.tolist()) != set(range(8)):
        raise ValueError(f'Expected layers 0..7, got {layers.tolist()}')
    frozen = model[model['layer'] < 7]
    if len(frozen) != len(base):
        raise ValueError('Frozen Gaussian count changed')
    changed = [name for name in base.dtype.names
               if not np.array_equal(base[name], frozen[name])]
    if changed:
        raise ValueError(f'Frozen Gaussian attributes changed: {changed}')
    nonfinite = [name for name in model.dtype.names
                 if not np.isfinite(model[name]).all()]
    if nonfinite:
        raise ValueError(f'Nonfinite Gaussian fields: {nonfinite}')
    report = {
        'iteration': args.iteration,
        'total_gaussians': len(model),
        'layer_counts': {str(layer): int(count) for layer, count in zip(layers, counts)},
        'frozen_layers_0_to_6_identical_to_base': True,
        'all_gaussian_fields_finite': True,
        'hu_augmented': False,
    }
    hu_path = directory / 'point_cloud_hu.ply'
    if hu_path.exists():
        augmented = read(hu_path)
        if len(augmented) != len(model):
            raise ValueError('HU augmentation changed the Gaussian count')
        changed = [name for name in model.dtype.names
                   if not np.array_equal(model[name], augmented[name])]
        if changed:
            raise ValueError(f'HU augmentation changed existing fields: {changed}')
        required = {'hu', 'hu_macro', 'hu_detail'}
        if not required.issubset(augmented.dtype.names):
            raise ValueError('Missing HU fields')
        if not all(np.isfinite(augmented[name]).all() for name in required):
            raise ValueError('Nonfinite HU fields')
        l7 = augmented[augmented['layer'] == 7]
        low, center, high = np.percentile(l7['hu'], [5, 50, 95])
        report.update({
            'hu_augmented': True,
            'hu_augmentation_preserved_existing_fields': True,
            'layer7_raw_hu_p05_p50_p95': [float(low), float(center), float(high)],
            'layer7_centers_in_fat_hu_band_percent': float(np.mean(
                (l7['hu'] >= -190) & (l7['hu'] <= -30)) * 100),
        })
    destination = args.model / f'verification_{args.iteration}.json'
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
