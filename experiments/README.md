# Fixed-structure RGB LUT experiments

`appearance_lut.py` implements one learnable RGB table per tissue, with fixed HU bounds and uniformly spaced control points. Sigmoid logits keep colors in range; neighboring entries are linearly interpolated. Only LUT parameters are optimized.

`validate_lut_render.py` runs a bounded **synthetic-target** CUDA gradient check on a supplied Gaussian PLY. It does not implement unpaired real-image GAN training. The original `train.py` is unchanged.

## Inputs and renderer

- Supply a Gaussian PLY containing SH coefficients, `layer`, and `hu`, plus the model's saved `cameras.json`.
- Supply a JSON tissue mapping; `lut_l4.json` is an example using the manual myocardium color anchors. Layer IDs are dataset-specific proxies and require semantic validation.
- The renderer expects the local rasterizer adaptations documented in `patches/README.md`, including its four-output interface. Apply the patches before building a fresh checkout.
- Camera coordinates must use the model's coordinate system. The validator derives its far plane from positive scene depth instead of assuming a meter-scale scene.

From the training repository:

```powershell
python -m unittest test_appearance_lut -v
python validate_lut_render.py --ply /path/to/model_hu.ply --cameras /path/to/cameras.json --config experiments/lut_l4.json --output /path/to/new_run --knots 16 --steps 10
```

Use `--knots 32` for the second control-point count. The CLI limits runs to 1–50 steps and 32–512 pixels and refuses an existing output directory. Supply a new output for every run.

Selected tissue colors replace its original SH contribution; other tissues keep frozen SH context. All original Gaussian positions, scales, rotations, opacities and SH parameters are frozen and checked bitwise. Optimizer parameters are explicitly limited to the LUT.

The validator checks an image-loss gradient against a central finite difference, checks multi-view output and synthetic loss improvement, and saves/reloads LUT and Adam state. Input model hashes, RNG state, configuration, fixed-view images and LUT curves are recorded. Unsampled/invisible control points need not receive nonzero gradients.

## Color and interpretation

Manual anchor numbers in the example are interpreted as sRGB and decoded once. Learned LUT interpolation and selected-tissue rendering use linear RGB; saved previews receive one sRGB encoding. Historical SH context has uncalibrated colorimetry, so this is not a full radiometric calibration or Unity framebuffer match.

A shared HU-to-RGB table changes appearance across views but cannot create independent spatial texture or identify material/lighting separately. Synthetic optimization success does not establish anatomical accuracy, real-image realism or cross-patient generalization. Real-domain patches still require matched tissue, observation scale and source-level splits.

Additional scripts in this repository are optional CT teacher-fitting or geometry diagnostics (`finetune_l7_surface.py`, `refine_l7_scales.py`, `preview_l7_refinement.py`, `verify_l7_training_output.py`). They are separate experiments and are not called by the fixed-structure LUT validator.
