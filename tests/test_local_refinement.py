import numpy as np
from scipy.spatial.transform import Rotation

from geometry.refine_local import split_ellipsoid


def test_split_preserves_center_and_covariance_before_surface_projection():
    xyz = np.array([4., -3., 2.])
    scales = np.array([.1, 5., 9.])
    rotation = Rotation.from_euler('xyz', [20, 30, 10], degrees=True).as_matrix()
    centers, child_scale = split_ellipsoid(xyz, scales, rotation, 2.5)
    assert child_scale.max() <= 2.5
    assert len(centers) > 1
    np.testing.assert_allclose(centers.mean(0), xyz, atol=1e-12)
    delta = centers-xyz
    mixture_covariance = delta.T@delta/len(delta) + (rotation*child_scale**2)@rotation.T
    np.testing.assert_allclose(mixture_covariance, (rotation*scales**2)@rotation.T, atol=1e-10)


def test_small_gaussian_is_not_split():
    xyz, scales = np.ones(3), np.array([.1, 1., 2.])
    centers, result_scales = split_ellipsoid(xyz, scales, np.eye(3), 2.5)
    np.testing.assert_array_equal(centers, xyz[None])
    np.testing.assert_array_equal(result_scales, scales)


def test_alpha_loss_updates_opacity_and_scale_on_cuda():
    import pytest
    import torch
    from types import SimpleNamespace
    from appearance.scene import load_camera
    from gaussian_renderer import render
    from geometry.refine_local import make_model, differentiable_alpha
    if not torch.cuda.is_available():
        pytest.skip('CUDA renderer required')
    def t(value):
        return torch.tensor(value, dtype=torch.float32, device='cuda')
    fields = dict(_xyz=t([[0., 0., 2.], [.15, 0., 2.1]]),
                  _scaling=torch.nn.Parameter(t([[.1, .1, .02], [.1, .1, .02]]).log()),
                  _rotation=t([[1, 0, 0, 0], [1, 0, 0, 0]]),
                  _opacity=torch.nn.Parameter(t([[0.], [0.]])),
                  _features_dc=torch.zeros(2, 1, 3, device='cuda'),
                  _features_rest=torch.zeros(2, 15, 3, device='cuda'))
    pc = make_model(fields, 3)
    camera = load_camera(dict(rotation=np.eye(3).tolist(), position=[0, 0, 0],
                             fx=64, fy=64, width=64, height=64), 64, pc.get_xyz)
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    alpha = differentiable_alpha(pc, camera, pipe)
    with torch.no_grad():
        legacy = render(camera, pc, pipe, t([0, 0, 0]))['accumulation']
    torch.testing.assert_close(alpha, legacy, atol=2e-6, rtol=2e-5)
    alpha.square().mean().backward()
    assert float(fields['_opacity'].grad.norm()) > 0
    assert float(fields['_scaling'].grad.norm()) > 0
