# Local rasterizer and Windows build adaptations

The training renderer calls a locally adapted rasterizer. The nested repositories remain at their upstream commits; these patch files preserve the existing local source differences without publishing compiled binaries or requiring new remote forks.

| Submodule | Pinned base | Patch |
| --- | --- | --- |
| `submodules/diff-gaussian-rasterization` | `9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0` | `rasterizer-local-compat.patch` |
| `submodules/simple-knn` | `44f764299fa305faf6ec5ebd99939e0508331503` | `simple-knn-windows.patch` |

On a **fresh checkout**, from the training repository, initialize those submodules and then apply:

```powershell
git -C submodules/diff-gaussian-rasterization apply --check ../../patches/rasterizer-local-compat.patch
git -C submodules/diff-gaussian-rasterization apply ../../patches/rasterizer-local-compat.patch
git -C submodules/simple-knn apply --check ../../patches/simple-knn-windows.patch
git -C submodules/simple-knn apply ../../patches/simple-knn-windows.patch
```

The original development checkout already contains these modifications. Do not apply them twice. `git apply --reverse --check` can verify that a patch is present. Rebuild extensions after applying, using the existing PyTorch/CUDA toolchain. The workspace's `build_cuda_extensions.cmd` is a Windows helper; toolchain paths are configurable.

The rasterizer patch lowers the hard-coded near culling distance from 0.2 to 0.005, adds a fourth `accumulation` return extracted from the pinned image-buffer layout, and records local NVCC compatibility flags. The KNN patch records the same NVCC compatibility flags. The expected return contract is `(color, radii, inverse_depth, accumulation)`.

**Important limitation:** the existing patch's custom backward does not propagate a gradient from the extra accumulation output. The fixed-structure LUT validator optimizes RGB image loss and does not depend on that gradient. This patch is a record of the working local interface, not a completed differentiable-alpha implementation. Any alpha-only objective requires separate backward support and validation.

Compiler compatibility overrides do not establish support for arbitrary CUDA/MSVC combinations. The local RGB gradient checks used PyTorch 1.12/CUDA 11.3 and an RTX 2080 Ti; a different environment must run its own validation. No CUDA rebuild or long training is performed merely by checking out these patches.
