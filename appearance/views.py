"""Explicit geometry split matching and reproducible shuffled camera epochs."""
from pathlib import Path
import numpy as np


def split_view_indices(cameras, train_path, test_path):
    def names(path):
        lines = [Path(line.strip()).stem for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        if not lines or len(lines) != len(set(lines)):
            raise ValueError('Camera split must be nonempty without duplicate names')
        return set(lines)
    train, test = names(train_path), names(test_path)
    if train & test:
        raise ValueError('Training/test camera splits overlap')
    camera_names = [Path(c['img_name']).stem for c in cameras]
    if len(camera_names) != len(set(camera_names)) or set(camera_names) != train | test:
        raise ValueError('Camera JSON must match geometry train/test splits exactly')
    return [i for i, name in enumerate(camera_names) if name in train], [i for i, name in enumerate(camera_names) if name in test]


def epoch_view_index(step, count, seed):
    if step < 1 or count < 1:
        raise ValueError('Expected positive step and camera count')
    epoch, offset = divmod(step-1, count)
    # Separate from patch RNG; absolute step makes resumed runs identical.
    return int(np.random.default_rng(np.random.SeedSequence([seed, epoch, 731])).permutation(count)[offset])
