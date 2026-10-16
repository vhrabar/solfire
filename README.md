# Solar Filament Segmentation Challenge

<!-- TODO: summary -->

Solution to the [IEEE BigData Cup 2026 — Solar Filament Segmentation](https://www.kaggle.com/competitions/filament-segmentation-2026)
challenge: instance segmentation of solar filaments in GONG H-alpha full-disk observations (MAGFiLO 1.0).

## Results

<!-- TODO: final numbers -->

| Submission | Models | Out-of-fold PQ | Public LB |
|------------|--------|----------------|-----------|
| TODO       |        |                |           |

- Out-of-fold PQ is pooled over all 707 training observations with the organizers' exact scoring
  (verified identical to `get_pq_score` from their self-evaluation notebook).
- Annotator-vs-annotator PQ on the training set is 0.34, which bounds what the labels can support.

<!-- TODO: final-rubric metrics (mean Dice, Dice/IoU distributions, 1:n / n:1 relations) from `evaluate`. -->

## Method

<!-- TODO: best pipeline figure (observation → tiles → ensemble → threshold/erode → instances → RLE). -->

### Data

- 707 observations, 1,154 annotation sets (up to 3 annotators per image), 2048×2048 grayscale.
- 5-fold cross-validation grouped by observation.
- Soft targets: the annotators' masks are averaged, polygons rasterised like pycocotools.

### Models

<!-- TODO: architectures desc -->

| Family | Decoder | Encoder                    | Pretraining |
|--------|---------|----------------------------|-------------|
| 1      | U-Net++ | ResNet34                   | ImageNet    |
| 2      | U-Net   | EfficientNet-B3            | ImageNet    |
| 3      | U-Net   | MiT-B2 (SegFormer encoder) | ImageNet    |

### Training

<!-- TODO: losses, augmentations, schedule, Lovász fine-tune and why it beats Lovász-from-start. -->

- 512×512 crops at native resolution, foreground-biased sampling.
- 40 epochs BCE + Dice (positive weight 5), AdamW, warm-up, SWA; encoder at 0.25× learning rate.
- 8-epoch fine-tune at lr 1e-4 with a Lovász hinge term (weight 0.5).

### Inference and post-processing

<!-- TODO: tiling, TTA, ensembling, calibration on out-of-fold maps, instance extraction. -->

- Tiled 512 windows with overlap, flip TTA, probabilities averaged over the ensemble.
- Threshold, morphological offset and instance mean-probability cut calibrated on the pooled out-of-fold maps.

### Morphology on H-alpha

<!-- TODO: qualitative examples and discus -->

### Efficiency

<!-- TODO: measured per-image inference time and peak memory; the models-vs-accuracy trade-off. -->

## What didn't work

<!-- TODO: measured effect of each. -->

Morphology-aware losses??? (clDice, opening/closing), GONG pseudo-labels, hard-negative mining, an auxiliary
category head, CLAHE preprocessing, instance-segmentation models, Lovász from the first epoch with longer
training, and further post-processing.

## Setup

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/). Pick the dependency group that matches your hardware:

```bash
uv sync --group cuda-132   # or rocm-714 (AMD, x86_64), cpu
```

Place the competition data under `data/`:

```
data/
  train/
    MAGFiLO_1.0_Annotations_kaggle2026_train.json
    train_images/
  test/
```

## Usage

```bash
# train one fold
uv run --group cuda-132 python -m src.main train --arch pretrained --decoder unetplusplus --encoder resnet34 --fold 0 ...

# run a sweep spec (train + calibrate every config)
uv run --group cuda-132 python scripts/sweep.py sweeps/cv5.json --only fold

# calibrate post-processing on the pooled out-of-fold maps of several checkpoints
uv run --group cuda-132 python -m src.main calibrate --checkpoint runs/<sweep>/*/best.pt --output runs/ensembles/<name>.calibration.json

# final-rubric metrics
uv run --group cuda-132 python -m src.main evaluate --checkpoint ...

# predict the test set (and submit)
uv run --group cuda-132 python scripts/submit.py <checkpoints> --calibration runs/ensembles/<name>.calibration.json --no-submit
```

<!-- TODO: exact commands to reproduce the final submission from scratch+ notebook for it-->

## Repository layout

```
src/
  main.py            CLI: train, predict, calibrate, evaluate, fit-filter
  data/              dataset, crop sampling, target rasterisation, preprocessing
  networks/          smp wrappers (pretrained encoders), TransUNet baseline
  training/          losses, engine, calibration, evaluation, metrics, instance filter
  inference/         tiling, prediction, post-processing
scripts/             sweeps, submission, model upload, data tools
sweeps/              experiment specs (JSON)
tests/               unit tests
```

## Tests

```bash
uv run --group cuda-132 pytest
```

<!-- TODO: list of tests ??? -->

## Reproducibility

<!-- TODO: seeds, fold assignment, training time per model, Kaggle model link and notebook. -->

## Disclaimer on the use of the LLM tools

This project was developed with the help of a LLM coding assistant,
[Claude Code](https://claude.com/claude-code), using Claude Opus 5 and 5.5 models.
It was used for:

- debugging and analysing run-time logs;
- launching and monitoring training runs and sweeps, and summarising their results;
- linguistic editing and proofreading of the documentation.

## License

Released under the MIT License. See [`LICENSE`](LICENSE).

Copyright © 2026 Vedran Hrabar
