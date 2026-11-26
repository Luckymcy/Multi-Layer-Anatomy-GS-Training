# Appearance-only LUT training

## Non-adversarial L4 initial validation

### Optional CT surface-neighbor regularization

`appearance/surface_graph.py` builds a fixed graph for one layer in the source PLY row order. An explicit physical-LPS/mm CT mesh provides nearest-surface vertices and consistently oriented normals. Candidate Gaussian KNN edges must satisfy a 2.5 mm Euclidean radius, 1.5 mm maximum Gaussian-to-surface distance, <=45 degree surface-normal angle, and <=2.5 mm path length along actual mesh triangle edges. Bounded Dijkstra rejects connections across nearby but geodesically distant folds. These are CT surface priors, not verified Gaussian anatomical connectivity; no PCA normal or zero PLY normal is silently substituted.

The optional loss is `sum_e w_e * mean_RGB(smooth_L1(c_i-c_j, beta=delta)) / sum_e w_e`, where `w_e=exp(-0.5*(geodesic_distance/radius)^2)` and default delta is 0.05 linear RGB. It regularizes final colors, not PLY index adjacency or geometry. Unmapped/isolated Gaussians get no graph gradient. All graph edges/weights are fixed buffers. A constant color gives zero loss; graph-size normalization keeps loss scale independent of edge count. Strong weights can oversmooth real texture, so 0.1 is only an initial trial value.

Prepare a mesh with the separate SimpleITK/VTK environment (workspace root):

```powershell
& C:/ProgramData/Anaconda3/envs/heart-teachers/python.exe tools/training/extract_ct_surface.py --mask data/ct/masks/total_free/heart.nii.gz --output archive/2026-11-20/L4_surface_regularization/ct_surface.npz
```

Then from this training repository with the existing PyTorch/SciPy environment:

```powershell
python -m appearance.surface_graph --ply ../output/geometry/L7_v4/point_cloud/iteration_30000/point_cloud_hu.ply --surface ../archive/2026-11-20/L4_surface_regularization/ct_surface.npz --output ../archive/2026-11-20/L4_surface_regularization/l4_graph.npz
```

Add `--surface-graph PATH --surface-weight 0.1 --surface-huber-delta 0.05` to `appearance color-distribution`. Weight defaults to zero, preserving the previous baseline. Providing the graph with weight zero records its diagnostic while retaining identical baseline updates. Graphs bind to the exact source model SHA256 and selected PLY rows; regenerate if the source changes. Checkpoints include graph buffers and metadata, and reports include initial/final loss and the separate graph gradient. The graph is not wired into the GAN entry point in this first experiment.

Current graph: 164,979 edges covering 38,445/58,289 L4 Gaussians. The original teacher's L4 outer-mask path is unavailable; this graph uses the existing CT whole-heart mask as an approximate prior, with a strict distance gate. Do not enlarge the mapping threshold just to force full coverage. `tests/test_surface_graph.py` checks folds, normals, disconnected color components, isolated points, normalized loss and model-bound reload.

`python run.py appearance color-distribution --labeling ../data/real/L4_heartmuscle/labeling --ply ../output/geometry/L4_v4/point_cloud/iteration_30000/point_cloud_hu.ply --cameras ../output/geometry/L4_v4/cameras.json --output ../output/appearance/l4/gaussian_train/new_validation --steps 8`

This separate entry point reads myocardium/exclude polygons directly, samples masked pixels without patches, and optimizes independent L4 linear RGB using mean/covariance and 32-direction sliced Wasserstein squared distance. Geometry, opacity and source SH stay frozen; selected-layer SH is replaced by view-independent RGB initialized from source SH DC (historical values assumed linear). The data-only gradient is checked separately from the weak initialization anchor. Fixed-sample before/after losses use the same random pixels and projections. Masks retain the existing 0.95 contribution/alpha and 0.98 purity thresholds under all-layer occlusion. Real masks are eroded by two pixels and heuristic bright pixels are excluded. Other blood/vessels/sutures still require manual exclusion.

By default limited to 1–20 engineering steps. Explicit `--long-run --steps 5000 --lr 0.001 --train-split PATH --test-split PATH` enables longer experiments with exact geometry train/test camera matching. All requested training cameras are screened, eligible views shuffle without replacement, and real frames are independently sampled uniformly. Without splits, 24 evenly spaced cameras are screened. There is no discriminator, spatial-neighbor regularizer, texture correspondence, or claim of realism/generalization. The default learning rate is 0.001. `gaussian_colors.pt` exports the model-bound `gaussian-linear-rgb-v1` color format supported by `tools/training/export_gaussian_rgb.py`; it is not a PLY. Periodic `latest_checkpoint.pt` also saves optimizer and RNG state, but a production resume CLI is not implemented. Source and annotation hashes, configuration, gradients, fixed-view images and frozen-field verification are recorded in `report.json` and adjacent outputs. Fixed-view losses use training cameras, not independent real validation.

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

### Independent Gaussian RGB baseline

L4 results live under `output/appearance/l4/guassian_train/` (non-adversarial) and `output/appearance/l4/gan_train/` (GAN). Keep only comparison.jpg, result.json and trained_l7.ply in each final result directory; checkpoints and detailed logs live in archive. L4 and L7 result roots identify the optimized tissue, not the highest layer in the source PLY. The updated low-glare L4 dataset has 75 native 32px patches covering all five annotated frames; 64px only covers three frames. `--patch-size 32` is supported by the existing discriminator; larger default sizes remain unchanged. This scale change must be recorded when comparing runs.

For L4 myocardium GAN comparison, use `--config appearance/configs/myocardium_l4.json --appearance-mode gaussian-rgb --rgb-initialization sh-dc --color-lr 0.001 --anchor-weight 0.01`. This matches the non-adversarial run's stored-DC initialization and bypasses selected-layer high-order SH; no HU LUT is used. The target-only myocardium config is not a valid LUT config. Default RGB initialization remains the manual LUT for older experiments. `--log-every 250` reduces console output while preserving every-step metrics. Current L4 data has 25 safe 64px patches from four annotated frames; 163/165 train cameras pass complete-patch screening at width 2048, with the same visibility thresholds. These candidates remain explicitly unreviewed and are used as an experiment, not certified tissue ground truth.

To use the same camera split as geometry training, supply `--train-split ../data/ct/teachers/v4/L7/train.txt --test-split ../data/ct/teachers/v4/L7/test.txt --view-order shuffle` instead of `--view-indices`. Names are matched to cameras.json (whose order includes test cameras); exact split matching and disjointness are required. All train cameras are screened using the existing visibility thresholds; cameras without a complete eligible patch are logged and skipped. Eligible views are shuffled without replacement each epoch, deterministically from seed/absolute step for reproducible resume. Test cameras never enter the loss. The report distinguishes requested, eligible, skipped and held-out view counts. A fresh run is required when changing splits or ordering.

Store independent RGB runs under `output/appearance/l7/guassian_train/` (workspace root). From this training repository, pass `--output ../output/appearance/l7/guassian_train/run_001`, choosing a new run subdirectory each time. LUT outputs remain separate under `output/appearance/l7/lut_train/`.

Use `--appearance-mode gaussian-rgb --color-lr 0.001` with the same input/view/data options below. This creates one sigmoid-parameterized linear RGB triplet per selected-layer Gaussian, initialized from the manual LUT, and then optimizes the colors independently. HU is used only for initialization, never for the learned color mapping. Identical HU values can learn different colors. This is view-independent RGB, not optimization of higher-order SH or Gaussian geometry. All stored source SH fields remain frozen and are bypassed on the selected layer.

The discriminator, patch sampling, occlusion tests and GAN losses are shared with the LUT baseline. `--smooth-weight` has no effect in RGB mode (effective weight zero): adjacency in PLY order is not spatial adjacency. `--anchor-weight` optionally constrains colors to their initialization. There is currently no surface-neighbor regularizer. Invisible/unobserved Gaussians are not guaranteed to change; the report records how many colors changed. Dataset acceptance and physical-scale limitations still apply.

Final RGB output is `gaussian_colors.pt`, with original PLY row indices, source model hash, layer and linear RGB values. It is bound to this exact model; it is not a HU LUT JSON and cannot be loaded through Unity's LUT JSON interface. Full resumable state is also saved. The shared checkpoint/metric field names retain the legacy `lut` prefix for compatibility; `appearance_mode` in config/signature/report distinguishes the representations and prevents cross-mode resume. LUT curve JSON files are not emitted in RGB mode. The LUT-specific summary plotting tool should not be used for this mode.

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
python run.py appearance train --ply ../output/geometry/L7_v4/point_cloud/iteration_30000/point_cloud_hu.ply --cameras ../output/geometry/L7_v4/cameras.json --real-manifest ../data/real/L7_fat/train/patch_manifest.json --output ../output/appearance/l7/new_smoke_run --steps 3 --patch-size 64 --render-width 2048 --view-indices 0 20 80 120 140 --d-channels 16 --smoke-test --allow-unreviewed-data
```

The default total step count is 10; `--smoke-test` is limited to <=5 total steps. Output must be new. The dataset summary must have `training_ready=true` for normal use; `--allow-unreviewed-data` explicitly records a departure from that status and never edits it. An engineering smoke run is not acceptance of the tissue labels or evidence of realism. No long run is started by importing or installing this package.

Resume by supplying `--resume /path/to/checkpoint_000003.pt`, a new output directory and `--steps 4`, while keeping the remaining arguments of that run. Steps are a total, so this executes one additional step. Resume verifies input, patch, config, code and environment fingerprints before restoring LUT, D, both Adam states and Python/NumPy/Torch/CUDA/sampler RNG state. Use only trusted local checkpoints (the supported PyTorch version uses pickle).

Run `python -m appearance.train ...` as an alternative to run.py. Old root-level entry points and the compatibility LUT shim were removed during task-based reorganization. Since source fingerprints changed, checkpoints from the previous layout are historical engineering records and will fail the strict new-code resume check; new runs resume normally within this layout. Existing Gaussian model/optimizer snapshots used by geometry training are not changed by the reorganization.

Outputs include `config.json`, `view_coverage.json`, per-step `metrics.jsonl`, checkpoints, LUT curves, a small set of fixed-view/patch previews and `report.json`. Checkpoints embed input signatures and loss history. The end-of-run verification reconstructs both networks/optimizers, checks saved tensors and a rendered output, checks the frozen model fields bitwise and rechecks model/data hashes. GPU atomics can still limit reproducibility across machines; state restoration is not a claim of bitwise-identical future training on every environment.

For longer runs, `--save-every 250 --preview-every 250` saves periodic checkpoints, latest LUT controls and fixed-view previews. `--preview-every 0` keeps only start/final previews. Per-step metrics include elapsed seconds within this run. Keep all source code unchanged during a run; a workspace launcher can retain a source snapshot alongside the output.

## Limits and current validation

The implementation has CPU tests for update separation/input gradients, occlusion-aware sampling, source integrity, review handling and checkpoint/RNG restore. A three-step actual-renderer/real-patch engineering check and one additional resumed step are recorded locally under ignored archive paths. These runs do not establish a useful learned appearance.

The current data come from one surgery, remain pending visual acceptance, and include uncertain red/vascular-looking regions in some fat crops. L7 itself is an anatomical proxy rather than a validated epicardial segmentation. No independent-patient result is available. Local D can exploit morphology and image quality, not just color; a shared HU LUT cannot create independent spatial texture or separate lighting/material. Smoothness and an optional anchor help constrain color but do not solve these identifiability issues.
