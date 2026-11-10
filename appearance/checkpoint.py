"""Full-state checkpoints for trusted local experiments."""
import random
import numpy as np
import torch


def save_checkpoint(path, step, lut, discriminator, lut_optimizer, d_optimizer,
                    initial_rgb, signature, rng, history):
    if path.exists():
        raise FileExistsError('Checkpoint path exists; do not overwrite another run')
    torch.save({'format_version': 1, 'step': step, 'lut': lut.state_dict(),
                'discriminator': discriminator.state_dict(), 'lut_optimizer': lut_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(), 'initial_rgb': initial_rgb,
                'signature': signature, 'history': history, 'sampler_rng': rng.bit_generator.state,
                'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, path)


def restore_checkpoint(path, lut, discriminator, lut_optimizer, d_optimizer, signature, rng):
    # Load only trusted local .pt files; this torch version uses pickle.
    state = torch.load(path, map_location='cpu')
    if state.get('format_version') != 1 or state['signature'] != signature:
        raise ValueError('Resume input/config fingerprint differs from checkpoint')
    lut.load_state_dict(state['lut'])
    discriminator.load_state_dict(state['discriminator'])
    lut_optimizer.load_state_dict(state['lut_optimizer'])
    d_optimizer.load_state_dict(state['d_optimizer'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    rng.bit_generator.state = state['sampler_rng']
    return state


def state_equal(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)) and isinstance(b, type(a)):
        return len(a) == len(b) and all(state_equal(x, y) for x, y in zip(a, b))
    return a == b
