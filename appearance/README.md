# Appearance-only LUT training

This package is separate from `geometry/train.py` and never configures a Gaussian optimizer. The adversarial entry point is `appearance/train.py` and learns **one tissue LUT**, currently using the L7/fat example. Geometry and all original SH parameters stay frozen; selected-tissue RGB replaces SH and other tissues keep fixed SH context.

```text
appearance/
├─ train.py            adversarial LUT executable module
├─ configs/            manual tissue LUT configurations
├─ lut.py              fixed-grid HU-to-RGB LUT and color transforms
├─ discriminator.py    tissue-conditioned patch discriminator
├─ patches.py          real patch verification, visibility masks and RGB crops
├─ scene.py            frozen model, cameras, indicator renders and checks
├─ trainer.py          alternating D/LUT updates
└─ checkpoint.py       full optimizer/model/RNG state and resume checks
run.py appearance train        unified workspace entry point
appearance/configs/lut_l7.json manual fat anchors and fixed HU bounds
tests/test_adversarial_appearance.py
```

## Inputs and color contract

Supply a Gaussian PLY with `layer` and `hu`, its `cameras.json`, and a prepared `patch_manifest.json` with `summary.json` in the same directory. The real loader selects one tissue and one native patch size, verifies checksums, source image/annotation checksums and dimensions, and samples frames uniformly before sampling their patches. It refuses evaluation splits and does not resize real images.

Real PNG values are assumed sRGB and decoded once into linear RGB. The LUT is initialized from sRGB manual anchors and interpolated in linear RGB. D sees both domains in linear RGB (normalized to [-1,1] inside D). Saved LUT previews are encoded to sRGB once; the original-SH comparison is saved as historical numeric values. This is an explicit working assumption, not radiometric calibration of surgical video or historical SH.

`lut_l7.json` fixes the HU input range to [-190,50] and uses Unity's layer-7 color anchors. Outside input HU is clamped to the table endpoints and counted in the report; this does not prove every layer-7 point is fat. Control-point locations, HU, Gaussian structure and opacity are not optimized.

## Generated-patch visibility

A full-scene indicator render uses RGB=1 for target-layer Gaussians and RGB=0 for other layers and background. Its red channel measures the target tissue's actual front-to-back compositing weight under all-layer occlusion. Together with the full-scene accumulation it defines a fixed sampling mask:

- target contribution >=0.95 and total accumulation >=0.95;
- target contribution / total accumulation >=0.98;
- two-pixel interior margin;
- every pixel of a generated crop must pass the mask.

The accumulation output is used only for frozen sampling, so the local rasterizer's missing accumulation backward is not used in the loss. Gaussian center projections, frustum/radius flags or an isolated-layer render would not provide this all-layer visible contribution test.

The default render width is 2048. On the current L7 model, the checked 1024px views had no eligible 64px crops; keeping the same purity thresholds and using 2048px yielded candidates. Empty views are recorded and skipped; no eligible views produces an error. Pixel-size equality alone does not establish matched physical observation scale. Use view choice/resolution and real ROI review to control that confound, rather than silently admitting other tissues.

## Alternating updates

D returns logits and has no batch statistics or dropout. The baseline uses:

```text
L_D   = mean softplus(-D(real,k)) + mean softplus(D(fake.detach(),k))
L_LUT = mean softplus(-D(fake,k)) + lambda_smooth * LUT second-difference sum
        + optional lambda_anchor * mean squared distance to initial LUT RGB
```

D is updated first. For the LUT step, D parameters are frozen while the input graph remains attached. The first new step explicitly checks adversarial-only LUT gradients, independently of regularization. Only LUT logits are in its optimizer. No cross-patient image L1/SSIM is used. One eligible camera is used per step and cameras cycle, sharing the same 3D LUT across views. Boxes are sampled from eligible regions, with replacement only when fewer boxes exist than the batch size.

## Run and resume

Use the training repository as working directory. Dependencies are the existing PyTorch/CUDA renderer environment; see `patches/README.md` for the local rasterizer contract.

```powershell
python -m unittest discover -s tests -v

# Three-step engineering check. Explicitly acknowledges pending dataset review.
python run.py appearance train --ply ../output/L7_v4/point_cloud/iteration_30000/point_cloud_hu.ply --cameras ../output/L7_v4/cameras.json --real-manifest ../data/real/surgery_001/training_dataset/v001/patch_manifest.json --output ../output/appearance/l7/new_smoke_run --steps 3 --patch-size 64 --render-width 2048 --view-indices 0 20 80 120 140 --d-channels 16 --smoke-test --allow-unreviewed-data
```

The default total step count is 10; `--smoke-test` is limited to <=5 total steps. Output must be new. The dataset summary must have `training_ready=true` for normal use; `--allow-unreviewed-data` explicitly records a departure from that status and never edits it. An engineering smoke run is not acceptance of the tissue labels or evidence of realism. No long run is started by importing or installing this package.

Resume by supplying `--resume /path/to/checkpoint_000003.pt`, a new output directory and `--steps 4`, while keeping the remaining arguments of that run. Steps are a total, so this executes one additional step. Resume verifies input, patch, config, code and environment fingerprints before restoring LUT, D, both Adam states and Python/NumPy/Torch/CUDA/sampler RNG state. Use only trusted local checkpoints (the supported PyTorch version uses pickle).

Run `python -m appearance.train ...` as an alternative to run.py. Old root-level entry points and the compatibility LUT shim were removed during task-based reorganization. Since source fingerprints changed, checkpoints from the previous layout are historical engineering records and will fail the strict new-code resume check; new runs resume normally within this layout. Existing Gaussian model/optimizer snapshots used by geometry training are not changed by the reorganization.

Outputs include `config.json`, `view_coverage.json`, per-step `metrics.jsonl`, checkpoints, LUT curves, a small set of fixed-view/patch previews and `report.json`. Checkpoints embed input signatures and loss history. The end-of-run verification reconstructs both networks/optimizers, checks saved tensors and a rendered output, checks the frozen model fields bitwise and rechecks model/data hashes. GPU atomics can still limit reproducibility across machines; state restoration is not a claim of bitwise-identical future training on every environment.

For longer runs, `--save-every 250 --preview-every 250` saves periodic checkpoints, latest LUT controls and fixed-view previews. `--preview-every 0` keeps only start/final previews. Per-step metrics include elapsed seconds within this run. Keep all source code unchanged during a run; a workspace launcher can retain a source snapshot alongside the output.

## Limits and current validation

The implementation has CPU tests for update separation/input gradients, occlusion-aware sampling, source integrity, review handling and checkpoint/RNG restore. A three-step actual-renderer/real-patch engineering check and one additional resumed step are recorded locally under ignored archive paths. These runs do not establish a useful learned appearance.

The current data come from one surgery, remain pending visual acceptance, and include uncertain red/vascular-looking regions in some fat crops. L7 itself is an anatomical proxy rather than a validated epicardial segmentation. No independent-patient result is available. Local D can exploit morphology and image quality, not just color; a shared HU LUT cannot create independent spatial texture or separate lighting/material. Smoothness and an optional anchor help constrain color but do not solve these identifiability issues.
