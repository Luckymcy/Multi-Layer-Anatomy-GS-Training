"""Create a separate L7 scale-limited candidate without deleting Gaussians."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from plyfile import PlyData


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-long-middle-ratio', type=float, default=8.0)
    parser.add_argument('--max-long-axis-mm', type=float, default=2.0)
    parser.add_argument('--min-long-cap-mm', type=float, default=0.25)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve() or args.output.exists():
        raise ValueError('Output must be a new model directory')
    if min(args.max_long_middle_ratio, args.max_long_axis_mm, args.min_long_cap_mm) <= 0:
        raise ValueError('Scale limits must be positive')
    source_dir = args.source / 'point_cloud/iteration_30000'
    args.output.mkdir(parents=True)
    output_dir = args.output / 'point_cloud/iteration_30000'
    output_dir.mkdir(parents=True)
    shutil.copy2(args.source / 'cfg_args', args.output / 'cfg_args')
    report = {}
    for name in ('point_cloud.ply', 'point_cloud_hu.ply'):
        path = source_dir / name
        if not path.exists():
            continue
        ply = PlyData.read(str(path))
        vertex = ply['vertex'].data
        original = vertex.copy()
        mask = vertex['layer'] == 7
        scales = np.exp(np.column_stack([vertex[f'scale_{i}'][mask] for i in range(3)]))
        order = np.argsort(scales, axis=1)
        longest = scales[np.arange(len(scales)), order[:, 2]]
        middle = scales[np.arange(len(scales)), order[:, 1]]
        cap = np.minimum(args.max_long_axis_mm,
                         np.maximum(args.min_long_cap_mm, args.max_long_middle_ratio * middle))
        clipped = np.minimum(longest, cap)
        for i in range(3):
            values = vertex[f'scale_{i}'][mask].copy()
            selected = order[:, 2] == i
            values[selected] = np.log(clipped[selected])
            vertex[f'scale_{i}'][mask] = values
        for field in vertex.dtype.names:
            check_mask = ~mask if field.startswith('scale_') else np.ones(len(vertex), dtype=bool)
            if not np.array_equal(vertex[field][check_mask], original[field][check_mask]):
                raise ValueError(f'Unexpected modification: {field}')
        ply.write(str(output_dir / name))
        report[name] = {'layer7_count': int(mask.sum()),
                        'long_axes_changed': int((clipped < longest).sum()),
                        'frozen_layers_identical': True,
                        'positions_opacity_sh_hu_preserved': True}
    report['limits'] = vars(args).copy()
    report['limits']['source'] = str(args.source)
    report['limits']['output'] = str(args.output)
    report['status'] = 'Postprocessed candidate; requires visual review for coverage loss.'
    (args.output / 'scale_refinement.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
