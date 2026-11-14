# SOLFIRE: Solar Filament Segmentation 2026

**SOLFIRE** (SOLar FIlament Refined Ensemble) is the solution to the IEEE BigData Cup 2026 Solar Filament Segmentation
challenge. It does instance segmentation of solar filaments in GONG H-alpha full-disk observations (MAGFiLO 1.0).
Code, training recipe and the same weights (Git LFS, `weights/final/`):
https://github.com/vhrabar/solfire

## Method

- Pretrained-encoder U-Nets trained on 512 px crops: U-Net++ ResNet34 (GONG-pretrained), U-Net MiT-B2 and
  U-Net HRNet-W32, plus an HRNet-W32 at 2× input scale ("zoom"). They train with BCE + Dice for 40 epochs,
  then 8 epochs with the Lovász hinge added. The published models are trained on all 707 training observations
  and keep their final epoch.
- The ensemble averages the probability maps (tiled 512 px windows, flip TTA). Instances are connected components cut at a
  calibrated threshold and filtered on their mean probability.
- A crop refiner (U-Net ResNet34 with an IoU head) re-segments each instance in a crop around it and drops those
  with a low predicted IoU.
- Barbs: thin dark protrusions are grown from the image into each instance (contrast + ridge filter).
- Cut-offs are tuned on pooled out-of-fold Panoptic Quality of the five-fold models.

## Variations

Every variation is one speed/quality trade-off of the same method, with float16 weights and barbs on.

| Variation       | Models                                                 | TTA     | Refiner | Out-of-fold PQ¹               | Public LB² | s / image³ | Size   |
|-----------------|--------------------------------------------------------|---------|---------|-------------------------------|------------|------------|--------|
| `solfire-final` | ResNet34 (GONG SSL), MiT-B2, HRNet-W32, zoom HRNet-W32 | D4⁴     | yes     | 0.4668                        | 0.40       | 10.8       | 303 MB |
| `solfire-fast`  | the same 4                                             | 4 flips | yes     | 0.4642                        | 0.40       | 4.1        | 303 MB |
| `solfire-lite`  | ResNet34 (ImageNet), MiT-B2, HRNet-W32                 | 4 flips | yes     | 0.4614                        | 0.39       | 2.0        | 230 MB |
| `solfire-tiny`  | HRNet-W32                                              | 4 flips | no      | 0.4508                        | 0.39       | 1.2        | 73 MB  |

1. Pooled over the 707 training observations, from the matching five-fold models through the same cut-offs, with
   barbs. Without barbs the scores are 0.4692, 0.4664, 0.4637 and 0.4531. Barbs cost about 0.002 PQ (0.0024 on
   `final`); the organisers asked for them in the masks.
2. Rounded to two decimals by the leaderboard, measured on the released variations. They keep one refiner (fold 0):
   on the out-of-fold maps every observation went through a single refiner anyway, and its masks match the
   five-refiner average at foreground IoU 0.97.
3. The 180 test observations including image loading, RTX 5070 Ti, fp16 autocast, batch 1.
4. The four flips with tile overlap 0.5, plus the four flips of the quarter-turned window.

`final` is the evaluated submission. Pick `fast` for about the same PQ at 2.6× the speed, and `tiny` for a
single small model.

## Files

- `<run>-best.pt`: segmentation checkpoint (`model_state` in float16, `model_config`, training settings).
- `refiner-<set>-fold0.pt`: crop refiner (`state_dict` in float16); `<set>` names the maps it was trained on.
- `<preset>.calibration.json`: cut-offs, inference settings (`variants`, `precision`, `skip_dark`, `barbs`), the
  refiner block, and a `preset` block with the scores above and the source calibration.
- `manifest.json`: source, SHA-256 and training settings of every file.

Weights are stored as float16: tensors of two or more dimensions; biases and norms stay float32. Under the
pipeline's fp16 autocast the predictions are identical to the float32 training checkpoints.
