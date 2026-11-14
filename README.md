# SOLFIRE: Solar Filament Instance Segmentation in GONG H-alpha

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](LICENSE)
[![Python 3.15](https://img.shields.io/badge/python-3.15-blue.svg?style=flat-square)](pyproject.toml)
[![DOI](https://zenodo.org/badge/1318428092.svg)](https://doi.org/10.5281/zenodo.24232262)

SOLFIRE segments solar filaments in GONG H-alpha full-disk observations (MAGFiLO 1.0). It is a solution to the
[IEEE BigData Cup 2026 — Solar Filament Segmentation](https://www.kaggle.com/competitions/filament-segmentation-2026)
challenge. One annotator scored against another reaches a Panoptic Quality (PQ) of only 0.34 on the 296 training
observations with several records, so each change is judged against a matching control on pooled out-of-fold maps.

The `final` variant averages four networks from three encoder families (ResNet34, MiT-B2, HRNet-W32, the last also
at twice the input resolution). A crop refiner then redraws and scores every proposal, and barbs, which the
organisers want in the masks, are grown from the observation without labels. It reaches an out-of-fold PQ of 0.4668
over 1154 annotation records and 0.40 on the public leaderboard. Four released variants span 0.4508 to 0.4668 PQ at
1.2 to 10.8 s per observation; see [Usage](#usage) to run them, or open the
[Kaggle notebook](https://www.kaggle.com/code/vhrabar/solfire-final-pipeline) that predicts the test set with the released weights.

## Results

| Submission | Models                                                             | Out-of-fold PQ | Public LB | s / img | Peak GPU |
|------------|--------------------------------------------------------------------|----------------|-----------|---------|----------|
| `final`    | 4 (SSL ResNet34, MiT-B2, HRNet-W32, zoom HRNet) + D4 TTA + refiner | 0.4668         | 0.40      | 10.8    | 4.0 GiB  |
| `fast`     | the same 4, flips only, + refiner                                  | 0.4642         | 0.40      | 4.1     | 4.0 GiB  |
| `lite`     | 3 (ImageNet ResNet34, MiT-B2, HRNet-W32) + refiner                 | 0.4614         | 0.39      | 2.0     | 3.9 GiB  |
| `tiny`     | HRNet-W32                                                          | 0.4508         | 0.39      | 1.2     | 1.8 GiB  |

- Out-of-fold PQ is pooled over all 707 training observations with the organisers' exact scoring
  (verified identical to `get_pq_score` from their self-evaluation notebook). It includes grown barbs, as the
  submissions do; the table under [Weights](#weights) gives it without them.
- Timing and peak GPU memory are for the full 180-image test set on an RTX 5070 Ti, including image loading.
- Annotator-vs-annotator PQ is 0.34 on the 296 training observations with several records. On the same observations
  the final model scores 0.468 out of fold against those records, and higher than the other annotators on 82 % of them.

Quality of the released presets beyond PQ, over the 1154 records of the training set (out-of-fold):

| Preset  | PQ with barbs | PQ without barbs |    SQ |    RQ | Dice per record | Dice per matched pair | Missed | Split | Spurious | Merged |
|---------|--------------:|-----------------:|------:|------:|----------------:|----------------------:|-------:|------:|---------:|-------:|
| `final` |        0.4668 |           0.4692 | 0.689 | 0.678 |           0.662 |                 0.813 | 24.3 % | 2.1 % |   11.6 % | 2.1 %  |
| `fast`  |        0.4642 |           0.4664 | 0.688 | 0.675 |           0.660 |                 0.812 | 22.9 % | 2.6 % |   12.1 % | 2.0 %  |
| `lite`  |        0.4614 |           0.4637 | 0.687 | 0.672 |           0.668 |                 0.811 | 21.7 % | 2.6 % |   13.1 % | 2.1 %  |
| `tiny`  |        0.4508 |           0.4531 | 0.679 | 0.664 |           0.667 |                 0.806 | 18.2 % | 2.9 % |   14.8 % | 3.0 %  |

- A filament is *missed* or a prediction *spurious* when it touches nothing; a filament is *split* when several
  predictions touch it and a prediction *merged* when it touches several filaments. Overlaps below the IoU match
  are in no column.
- Dice is averaged per record and over matched pairs (IoU > 0.5) only.

## Method

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/vhrabar/solfire/raw/main/paper/out/pipeline-dark.png">
  <img alt="SOLFIRE pipeline of the final variant: SSL pretraining, two-stage training, out-of-fold calibration and refiner training (top) feed the inference chain of 4-model ensemble, proposals, crop refiner, assembly and barb growth (bottom)" src="https://github.com/vhrabar/solfire/raw/main/paper/out/pipeline.png">
</picture>

*Pipeline of the `final` variant, dashed arrows carry training outputs into inference.*

### Data

- 707 observations, 1,154 annotation sets (up to 3 annotators per image), 2048×2048 greyscale, read as one channel
  in [0, 1] and standardised with ImageNet statistics; no contrast enhancement.
- 5-fold cross-validation grouped by observation.
- Soft targets: polygons are rasterised with pycocotools, as in the evaluator, and the annotators' masks are averaged,
  so each pixel holds the fraction of annotators who marked it.

### Models

| Member     | Decoder | Encoder   | Pretraining                                      | Input |
|------------|---------|-----------|--------------------------------------------------|-------|
| SSL-R34    | U-Net++ | ResNet34  | ImageNet, then SimMIM-style on 1,300 GONG frames | 1×    |
| MiT-B2     | U-Net   | MiT-B2    | ImageNet                                         | 1×    |
| HRNet-W32  | U-Net   | HRNet-W32 | ImageNet                                         | 1×    |
| zoom HRNet | U-Net   | HRNet-W32 | starts from the HRNet-W32 member                 | 2×    |

`lite` uses the ImageNet ResNet34 instead of SSL-R34 and drops the zoom member; `tiny` is the HRNet-W32 alone.

### Training

- 512×512 crops at native resolution, 7 per observation per epoch; with probability 0.7 a crop is placed on a
  filament vertex. Flips, quarter turns, gamma, brightness and contrast jitter.
- Base stage: 40 epochs of BCE (positive weight 5) + soft Dice, AdamW (batch 8, weight decay 0.01), peak lr 4e-4
  with the encoder at 0.25×, 3 warm-up epochs, cosine decay.
- Fine-tune: 8 epochs at lr 1e-4 with a Lovász hinge term (weight 0.5). The zoom member is fine-tuned from the base
  HRNet for 12 epochs at 2× input.
- The shipped models repeat both stages on all 707 observations and keep the final epoch (on out-of-fold maps the
  final epoch scores 0.4602 vs 0.4593 for the validation-selected one).

### Inference and post-processing

- **Ensemble:** 512 px tiles, probabilities averaged over the members; `final` uses D4 TTA (the four flips at 50 % tile
  overlap and the four flips of the quarter-turned image at 25 %), the other presets the four flips at 25 % overlap. fp16, dark (off-disk) tiles skipped.
- **Proposals:** threshold the mean map (0.35 in `final`), erode 1 px, split into connected components.
- **Crop refiner** (all presets but `tiny`): for each proposal, a 256 px crop of a window 1.5× its box (image,
  ensemble map, proposal) goes through a refiner network that predicts the instance mask and its IoU. It is trained
  on out-of-fold proposals; fold 0's model ships.
- **Assembly:** refined masks are thresholded (0.45) and filtered by predicted IoU (≥ 0.32 in `final`; its out-of-fold best is 0.28, the stricter cut follows
  leaderboard probes), then placed in order
  of IoU; a mask is kept if at least half of its pixels are still free.
- **Barb growth:** thin, long side structures are grown from each mask using local contrast and a ridge filter
  within a 20 px band. It costs about 0.002 PQ but is the only method that adds barbs, which the organisers want.
- All cuts are grid-searched on pooled out-of-fold maps; the output is one RLE mask per instance.

### Morphology on H-alpha

![Three held-out observations: H-alpha image, the annotators' outlines, the `final` instances with grown barbs in orange, and a 12× zoom on the barbs](https://github.com/vhrabar/solfire/raw/main/paper/out/qualitative.png)

*Held-out observations with at least two annotator records. They were chosen for their grown barbs (among those
whose barb pixels mostly fall inside the annotators' masks, the ones that read clearly at 12×), so they are not
typical; their PQ (0.38, 0.62, 0.54) is printed beside each.

- Long, dark, well-contrasted filaments are found and outlined close to where the annotators put them, and the
  refiner gives smooth contours, as in all three rows. Where annotators draw one filament as several pieces or split
  a long one differently, one prediction can match only part of them.
- Barbs are hard to judge from the labels. Pixels thinner than about 5 px (barbs) are 1.9 % of the annotated pixels.
  A model's recall on them is 0.38, against 0.67 on the body of a filament. One annotator's thin pixels fall inside
  another's mask only 0.40 of the time (0.54 within 2 px), so the model is already at the level of a second
  annotator. Filling in every missed thin pixel would raise PQ by only 0.015. (Measured on the out-of-fold maps of the
  three-family fine-tuned ensemble, before the zoom model and the refiner.)
- Barb growth adds the structures in orange. They are short, 419 px in total over the three examples above, and the
  figure shows them only in the 12× column. Only the clearest barbs are added. Growing them costs about 0.002 PQ,
  because the annotations hardly label barbs.
- On the same maps the hit rate is lowest for small, barb-heavy filaments (0.60, mean best IoU 0.45) and next lowest
  for the largest ones (0.62, IoU 0.49), against 0.65–0.68 and about 0.51 in between. For the largest, the problem is
  where a filament ends, not whether it is found.
- Chirality (left, right, unidentifiable) is read from the barbs, and annotators agree on it for 69 % of matched
  pairs. A straight left/right flip happens in 14 % of the left/right pairs.

### Efficiency

Measured over the 180 test images, including image loading, on an NVIDIA GeForce RTX 5070 Ti.
The same runs on a Kaggle Tesla T4 reproduce the predictions (99.8 % of masks have
IoU ≥ 0.99 against the local ones) but are 3 to 5 times slower per image.

| Preset  | s / img (5070 Ti) | s / img (T4) | Peak GPU | Out-of-fold PQ |
|---------|------------------:|-------------:|---------:|---------------:|
| `final` |              10.8 |         50.6 |  4.0 GiB |         0.4668 |
| `fast`  |               4.1 |         17.7 |  4.0 GiB |         0.4642 |
| `lite`  |               2.0 |          7.8 |  3.9 GiB |         0.4614 |
| `tiny`  |               1.2 |          4.0 |  1.8 GiB |         0.4508 |

One HRNet-W32 gets 0.4508. Three models plus the refiner add 0.0106 for 1.7× the time (95 % interval of the paired
difference 0.0064 to 0.0152), and the zoom member with D4 TTA adds another 0.0054 for 5.4× the time of `lite`
(0.0021 to 0.0086). `fast` and `final` run the same four models, with flips and 25 % tile overlap against D4 TTA and
50 %: `final` takes 2.6× the time for 0.0026 PQ, and its interval (−0.0005 to 0.0056) includes zero, so that gain is
not established. The same holds for `fast` over `lite` (+0.0028, −0.0003 to 0.0059).

## What didn't work

Changes that were tested and rejected, as pooled out-of-fold PQ after calibration against a control with the same
setup. Rerun noise is about 0.0006 on average for the three-family ensemble (four comparisons that should have no
effect) and 0.007 at most (one single-family five-model set), and ±0.001 for the crop refiner. A change is therefore
adopted only if it beats its control by more than 0.002 after the full pipeline, with the nested score agreeing.
None did.

| Change                             |     PQ | Control |    Gain |
|------------------------------------|-------:|--------:|--------:|
| CLAHE contrast enhancement         | 0.4367 |  0.4390 | −0.0023 |
| GONG pseudo-labels                 | 0.4353 |  0.4390 | −0.0037 |
| Lovász hinge from scratch          | 0.4446 |  0.4503 | −0.0057 |
| 1024×1024 training crops           | 0.4458 |  0.4503 | −0.0045 |
| Ensemble distillation              | 0.4465 |  0.4503 | −0.0038 |
| Tversky loss in the fine-tune      | 0.4585 |  0.4594 | −0.0009 |
| Longer fine-tune                   | 0.4589 |  0.4594 | −0.0005 |
| Thin-aware fine-tune               | 0.4599 |  0.4594 | +0.0005 |
| 35 models instead of 15            | 0.4558 |  0.4564 | −0.0006 |
| Magnetogram head, in ensemble      | 0.4550 |  0.4549 | +0.0001 |
| Watershed instance splitting       |  0.388 |   0.443 |  −0.055 |
| YOLO26m-seg member, score-ordered  | 0.4521 |  0.4641 | −0.0120 |
| Spine auxiliary head, with refiner | 0.4650 |  0.4637 | +0.0013 |
| Second refiner pass                | 0.4692 |  0.4700 | −0.0008 |
| Other refiner encoders, best       | 0.4692 |  0.4700 | −0.0008 |

Losses with clDice or differentiable morphology terms were also tried and dropped. A dedicated barb head gave no PQ
gain (best +0.0001) and barely raised thin-structure recall: it extended filament edges rather than adding barbs.

## Limitations

- Validated and shipped systems differ. Every out-of-fold score belongs to the five-fold system (models on about
  80 % of the observations, one refiner per fold). The released presets are the same recipe retrained once on all 707
  observations, with the final epoch kept and the refiner of fold 0. No labelled data can score them without leakage.
- Folds are not separated in time: the median out-of-fold observation lies 2.0 days from a training one, the median
  test observation 1.5 days.
- Each configuration is one training run; rerun noise is about 0.0006 on average for the three-family ensemble
  and 0.007 at most for a single-family five-model set.
- Only the IoU cut of `final` (0.32 against an out-of-fold best of 0.28) follows leaderboard probes; no test image or
  label enters training. The out-of-fold to leaderboard gap is about 0.07 and unexplained, so out-of-fold PQ ranks
  methods rather than forecasts the test score.

## Setup

The code is the installable Python package `solfire` (SOLar FIlament Refined Ensemble; source in
`src/solfire/`). It requires Python 3.15.

With [uv](https://docs.astral.sh/uv/) (recommended), pick the dependency group that matches your hardware; this
installs the package in editable mode together with the pinned torch stack:

```bash
uv sync --group cuda-134   # or rocm-100 (AMD, x86_64), cpu
```
or via pip, from the repository root; this also installs `solfire` in editable mode. The same file serves every Linux setup, and the index picks the torch build:
```bash
pip install -r requirements.txt                                                               # NVIDIA CUDA
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/rocm10.0   # AMD ROCm (x86_64)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu        # CPU-only
```

`uv build` produces a wheel and sdist in `dist/`. Installing the package adds the `solfire` command
(the same CLI as `python -m solfire.main`) and `solfire-lab`, the same CLI with the dropped
experiments' options (`src/solfire/experiments/README.md`). `data/` and `runs/` are resolved against the source checkout
when the package is imported from one, otherwise against the working directory; set `FILAMENT_PROJECT_ROOT` to
override, or `FILAMENT_DATA_ROOT` to point at the data alone.

Place the competition data under `data/`:

```
data/
  train/
    MAGFiLO_1.0_Annotations_kaggle2026_train.json
    train_images/
  test/
```

## Usage

`uv run --group cuda-134` below stands for any dependency group from [Setup](#setup). Every sweep spec in `sweeps/`
(see `sweeps/README.md`) is run with `scripts/sweep.py`; `--dry-run` prints the commands without running them.

### Predict with the shipped weights

```bash
uv run --group cuda-134 solfire predict \
  --calibration weights/final/final.calibration.json \
  --checkpoint weights/final/c02_full-lovasz-ssl-r34_seed1-best.pt weights/final/c04_full-lovasz-mitb2_seed1-best.pt \
               weights/final/c06_full-lovasz-hrnet32_seed1-best.pt weights/final/c01_full-zoom-hrnet32_seed1-best.pt \
  --output-dir out/final
```

The calibration holds the cut-offs, TTA variants, fp16, barbs and the refiner; `fast`, `lite` and `tiny` work the same way
with their own calibration. `notebook/Final pipeline.ipynb` does the same on Kaggle:
[SOLFIRE Final Pipeline](https://www.kaggle.com/code/vhrabar/solfire-final-pipeline), with the weights attached as the
[Kaggle model](https://www.kaggle.com/models/vhrabar/filament-segmentation-2026).

### Reproduce the `final` preset from scratch

Place the data as in [Setup](#setup).

```bash
# 1. SSL pretraining of the ResNet34 encoder on unlabelled GONG frames -> runs/ssl/mim-unetpp-r34/
#    needs the unlabelled GONG H-alpha frames of the Kaggle dataset vhrabar/gong-solar-filaments,
#    Training only: nothing external is used at inference.
uv run --group cuda-134 python scripts/pretrain_mim.py --output runs/ssl/mim-unetpp-r34

# 2. five-fold models: 40 epochs of BCE + Dice, then 8 epochs with the Lovász hinge, then the 2x zoom HRNet
for spec in cv5_ssl cv5_mitb2 cv5_hrnet finetune_ssl finetune_mitb2 finetune_hrnet finetune_zoom_hrnet; do
  uv run --group cuda-134 python scripts/sweep.py sweeps/$spec.json
done

# 3. calibrate on the pooled out-of-fold maps of the 20 fold checkpoints (D4-style TTA: overlap 0.5 and turned)
uv run --group cuda-134 solfire calibrate --variants o0.5 turned \
  --checkpoint runs/finetune_{ssl,mitb2,hrnet,zoom_hrnet}/*/best.pt \
  --output runs/ensembles/cv20-zoom-ssl-tta.calibration.json

# 4. crop refiner, trained on the held-out proposals of that ensemble (one model per fold)
uv run --group cuda-134 python scripts/refine.py all --calibration runs/ensembles/cv20-zoom-ssl-tta.calibration.json \
  --name zoom-ssl-tta-rt --crops crops-zoom-ssl-tta

# 5. full-data models (every record, final epoch kept)
for spec in full_ssl full full_zoom; do uv run --group cuda-134 python scripts/sweep.py sweeps/$spec.json; done

# 6. float16 weights, one calibration per preset in weights/final
uv run --group cuda-134 python scripts/release_weights.py
```

### Other commands

```bash
# train one fold
uv run --group cuda-134 solfire train --arch pretrained --decoder unetplusplus --encoder resnet34 --fold 0 ...

# final-rubric metrics
uv run --group cuda-134 solfire evaluate --checkpoint ...

# predict the test set and submit it to Kaggle (credentials must be cached)
uv run --group cuda-134 python scripts/submit.py <checkpoints> --calibration <calibration.json>
```

## Weights

`weights/final/` holds the shipped weights in float16 (plain files in the repository, about 430 MB in total) and one calibration per preset. Out-of-fold PQ is
without barbs; s/img is on an RTX 5070 Ti.

| Preset  | Models                                                             | OOF PQ | s / img | Size   |
|---------|--------------------------------------------------------------------|--------|---------|--------|
| `final` | 4 (SSL ResNet34, MiT-B2, HRNet-W32, zoom HRNet) + D4 TTA + refiner | 0.4692 | 10.8    | 303 MB |
| `fast`  | the same 4, flips only, + refiner                                  | 0.4664 | 4.1     | 303 MB |
| `lite`  | 3 (ImageNet ResNet34, MiT-B2, HRNet-W32) + refiner                 | 0.4637 | 2.0     | 230 MB |
| `tiny`  | HRNet-W32                                                          | 0.4531 | 1.2     | 73 MB  |


## Repository layout

```
src/solfire/         the package: the final method
  main.py            CLI (`solfire`): train, predict, calibrate, evaluate
  data/              dataset, crop sampling, preprocessing cache
  networks/          pretrained-encoder U-Net / U-Net++ (smp)
  training/          losses, engine, calibration, evaluation, crop refiner training
  inference/         tiling, TTA, prediction, crop refiner, barb growth, post-processing
  experiments/       dropped experiments, runnable with `solfire-lab` (see its README.md)
scripts/             sweeps, SSL pretraining, release and Kaggle upload, submission, ledger, notebook build
  experiments/       data and analysis scripts of the dropped experiments
sweeps/              specs of the final method's runs (JSON); experiments/ for the dropped ones
tests/               unit tests; experiments/ for the dropped experiments
weights/final/       shipped fp16 weights, refiners and one calibration per preset
notebook/            Kaggle notebook (pipeline.py builds `Final pipeline.ipynb`)
kaggle/              metadata for the Kaggle dataset, model and notebook
results/             submission.csv of each released preset on the 180 test observations
paper/               the paper (LaTeX source, figures, the scripts behind its tables)
docs/                submissions.csv (every Kaggle submission)
external/            submodule: FITS to JPEG converter for the GONG frames
```

## Tests

```bash
uv run --group cuda-134 pytest
```

## Reproducibility

- Environment: `uv.lock` pins every package, including the torch build per hardware group ( [Setup](#setup));
  `requirements.txt` is its export and CI checks that they agree.
- Weights: `weights/final/` (in the repository) and the Kaggle models `solfire-<preset>/1` are the same fp16 files, one
  calibration JSON per preset.
- Predicting the test set: `notebook/Final pipeline.ipynb` (`PRESET = "final"`, also on Kaggle as
  [solfire-final-pipeline](https://www.kaggle.com/code/vhrabar/solfire-final-pipeline)) loads those weights and writes the
  submission. It gives the same output as `solfire predict` on all 180 test images.
- Determinism: inference is not bit-exact across GPUs; masks can differ by 1–2 px at the borders, which does not
  change the score at the reported precision.
- Retraining: see [Usage](#usage) and `sweeps/` for the exact configurations.
- Seeds: every shipped model and refiner was trained with seed 1 (`--seed`, the default), which seeds `random`,
  `numpy` and `torch` (CPU and CUDA) and the augmentation and crop sampling.
- Folds: 5 grouped folds (all records of an observation stay together), observations shuffled by file name with
  seed 0. The assignment does not depend on `--seed`, so out-of-fold predictions of different seeds can be pooled.
- Determinism of training: cuDNN is not set to deterministic, so retraining with the same seed gives close but not
  identical weights (refiner runs vary by about ±0.001 PQ).
- Training time (NVIDIA GeForce RTX 5070 Ti): the four shipped models take about 4 h on all 707 observations
  (SSL-R34 77 min, MiT-B2 61 min, HRNet-W32 88 min, zoom HRNet 22 min), plus about 1.8 h of SSL pretraining. The 5-fold
  out-of-fold runs used for calibration and for training the refiner cost roughly 5× that per model.

## Citation

```bibtex
@techreport{hrabar2026solfire,
  title       = {Beyond Annotator Agreement: A Calibrated Three-Family Ensemble with Crop Refinement for Solar Filaments},
  author      = {Hrabar, Vedran},
  institution = {Faculty of Electrical Engineering and Computing, University of Zagreb},
  address     = {Zagreb, Croatia},
  year        = {2026},
  note        = {A Solution to the Solar Filament Segmentation Challenge 2026},
  url         = {https://github.com/vhrabar/solfire}
}
```

## Disclaimer on the use of the LLM tools

This project was developed with the help of a LLM coding assistant,
[Claude Code](https://claude.com/claude-code), using Claude Opus 5 and 5.5 models.
It was used for:

- debugging and analysing run-time logs;
- launching and monitoring training runs and sweeps, and summarising their results;
- linguistic editing and proofreading of the documentation.

## License

- **Code:** MIT License. See [`LICENSE`](https://github.com/vhrabar/solfire/blob/main/LICENSE).
- **Model weights:**  MIT License. See [`LICENSE`](https://github.com/vhrabar/solfire/blob/main/LICENSE).
- **Paper (text and figures):** CC BY 4.0. See [`LICENSE-paper`](https://github.com/vhrabar/solfire/blob/main/LICENSE-paper).

- **Pretraining frames:** published separately as [GONG Multi-Channel Data for Solar Filaments](https://doi.org/10.34740/KAGGLE/DSV/21047890) under its own terms.

Copyright © 2026 Vedran Hrabar
