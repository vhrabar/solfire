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

The code is the installable Python package `solfire` (**SOL**ar **FI**lament **R**efined **E**nsemble; source in
`src/solfire/`). It requires Python 3.15.

With [uv](https://docs.astral.sh/uv/) (recommended), pick the dependency group that matches your hardware; this
installs the package in editable mode together with the pinned torch stack:

```bash
uv sync --group cuda-134   # or rocm-100 (AMD, x86_64), cpu
```
or via pip
```bash
pip install -r requirements.txt
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

```bash
# train one fold
uv run --group cuda-134 solfire train --arch pretrained --decoder unetplusplus --encoder resnet34 --fold 0 ...

# run a sweep spec (train + calibrate every config)
uv run --group cuda-134 python scripts/sweep.py sweeps/cv5_hrnet.json --only fold

# calibrate post-processing on the pooled out-of-fold maps of several checkpoints
uv run --group cuda-134 solfire calibrate --checkpoint runs/<sweep>/*/best.pt --output runs/ensembles/<name>.calibration.json

# final-rubric metrics
uv run --group cuda-134 solfire evaluate --checkpoint ...

# predict the test set (and submit)
uv run --group cuda-134 python scripts/submit.py <checkpoints> --calibration runs/ensembles/<name>.calibration.json --no-submit
```

<!-- TODO: exact commands to reproduce the final submission from scratch+ notebook for it-->

## Repository layout

```
src/solfire/           the package: the final method
  main.py              CLI (`solfire`): train, predict, calibrate, evaluate
  extension.py         hooks the experiments plug into
  data/                dataset, crop sampling, target rasterisation, preprocessing cache
  networks/            pretrained-encoder U-Net / U-Net++ (smp)
  training/            losses, engine, calibration, evaluation, metrics, crop refiner training
  inference/           tiling, TTA, prediction, crop refiner, barb growth, post-processing
  experiments/         every dropped experiment, still runnable with `solfire-lab` (see its README.md)
scripts/               sweeps, submission, model upload, notebook build, SSL pretraining, refiner
scripts/experiments/   data and analysis scripts of the dropped experiments
sweeps/                specs of the final method's runs (JSON)
sweeps/experiments/    specs of the dropped experiments, run through solfire-lab
tests/                 unit tests; tests/experiments/ for the experiments
```

## Tests

```bash
uv run --group cuda-134 pytest
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
