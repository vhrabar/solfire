"""Annotator-vs-annotator and final model, scored on the same observations (all records)."""

import itertools
import json
import sys
import types
from pathlib import Path

import numpy as np

import solfire.inference.refiner as refiner_module
from solfire.main import held_out_maps, parse_args, resolve_device
from solfire.training.evaluate import evaluate
from solfire.training.metrics import SUBMISSION_SIZE, annotation_pixels, pq_counts, pq_from_counts


def alias_renamed_package():
    """Refined-instance pickles name the package ``src``, as it was before the rename to ``solfire``."""
    sys.modules.setdefault("src", types.ModuleType("src"))
    sys.modules.setdefault("src.inference", types.ModuleType("src.inference"))
    sys.modules["src.inference.refiner"] = refiner_module


alias_renamed_package()

cal_path = "runs/ensembles/cv20-zoom-ssl-tta-refined.calibration.json"
cal = json.loads(Path(cal_path).read_text())
args = parse_args(["evaluate", "--checkpoint", *cal["checkpoints"], "--calibration", cal_path, "--variants", "o0.5", "turned"])
device = resolve_device(None)
sources, per_file, _, method = held_out_maps(args, device)
names = sorted(sources)
print("observations", len(names), "records", sum(len(per_file[n]) for n in names))


def model_counts(calibration, refined_dir="runs/refiner/zoom-ssl-tta-rt/refined"):
    _, overlaps = evaluate(sources, per_file, calibration, workers=8, refined_dir=refined_dir)
    out, i = {}, 0
    for n in names:
        k = len(per_file[n])
        out[n] = np.array([pq_counts(overlaps[i + j][0]) for j in range(k)])
        i += k
    return out


best = model_counts(cal)
ship = json.loads(json.dumps(cal))
ship["refiner"]["min_score"] = 0.32
shipped = model_counts(ship)

# annotator vs annotator: ordered pairs of records of one observation
H = SUBMISSION_SIZE


def masks(ann):
    px = annotation_pixels(ann, H, H)
    m = np.zeros((len(px), H * H), np.float32)
    for r, p in enumerate(px):
        m[r, p] = 1
    return m


inter = {}
for n in names:
    recs = per_file[n]
    if len(recs) < 2:
        continue
    ms = [masks(a) for a in recs]
    rows = []
    for a, b in itertools.permutations(range(len(ms)), 2):
        A, B = ms[a], ms[b]
        if len(A) == 0 or len(B) == 0:
            iou = np.zeros((len(B), len(A)))  # rows GT(B), cols pred(A)
        else:
            i_ = B @ A.T
            u = B.sum(1)[:, None] + A.sum(1)[None] - i_
            iou = i_ / np.maximum(u, 1)
        rows.append(pq_counts(iou))
    inter[n] = np.array(rows)
multi = sorted(inter)
print("multi-annotator observations", len(multi))


def pq(arr):
    return pq_from_counts(*arr.sum(0)) if len(arr) else float("nan")


def agg(d, keys):
    return np.concatenate([d[k] for k in keys])


rng = np.random.default_rng(0)


def boot(fn, keys, n=1000):
    v = [fn(list(rng.choice(keys, len(keys)))) for _ in range(n)]
    return np.percentile(v, [2.5, 97.5])


single = [n for n in names if n not in inter]
print(
    f"annotator vs annotator ({len(multi)} obs):          {pq(agg(inter, multi)):.4f}  CI {boot(lambda k: pq(agg(inter, k)), multi)}"
)
for label, d in (("model best cuts (0.4700 run)", best), ("model shipped cuts (0.32)", shipped)):
    print(f"{label}")
    print(f"   all {len(names)} obs:        {pq(agg(d, names)):.4f}")
    print(f"   multi-annotator {len(multi)} obs: {pq(agg(d, multi)):.4f}  CI {boot(lambda k, d=d: pq(agg(d, k)), multi)}")
    print(f"   single-annotator {len(single)} obs: {pq(agg(d, single)):.4f}")
# paired: same observations, same bootstrap draw


def diff(k):
    return pq(agg(best, k)) - pq(agg(inter, k))


print("paired diff (model best - annotators) on multi:", round(diff(multi), 4), "CI", boot(diff, multi))
# model vs a single random record per obs (one-to-one like an annotator pair)
sel = np.array([agg_ for agg_ in [0]])
for seed in range(3):
    r = np.random.default_rng(seed)
    rows = np.array([best[n][r.integers(len(best[n]))] for n in multi])
    print("model vs one random record per multi obs, seed", seed, round(pq(rows), 4))
# per-record-pair apples to apples: model vs record, restricted to pairs
tp = agg(best, multi)
print("model counts multi (matched,tp,fp,fn):", tp.sum(0).round(1))
ia = agg(inter, multi)
print("annot  counts multi:", ia.sum(0).round(1))
