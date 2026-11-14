import collections
import json
import sys
import types
from pathlib import Path

import numpy as np

import solfire.inference.refiner as refiner_module
from solfire.data import load_annotations
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
sources, per_file, _, _ = held_out_maps(args, resolve_device(None))
_, by_image = load_annotations()
idof = {tuple(sorted(a["id"] for a in v)): k for k, v in by_image.items() if v}
names = sorted(sources)
_, ov = evaluate(sources, per_file, cal, workers=8, refined_dir="runs/refiner/zoom-ssl-tta-rt/refined")
H = SUBMISSION_SIZE
rows = []  # per record: id, model counts, others-vs-it counts
i = 0
for n in names:
    recs = per_file[n]
    k = len(recs)
    mod = [pq_counts(ov[i + j][0]) for j in range(k)]
    i += k
    if k < 2:
        continue
    ms = []
    for a in recs:
        px = annotation_pixels(a, H, H)
        m = np.zeros((len(px), H * H), np.float32)
        for r, p in enumerate(px):
            m[r, p] = 1
        ms.append(m)
    for b in range(k):
        oth = []
        for a in range(k):
            if a == b:
                continue
            A, B = ms[a], ms[b]
            iou = (
                np.zeros((len(B), len(A)))
                if len(A) == 0 or len(B) == 0
                else (lambda x: x / np.maximum(B.sum(1)[:, None] + A.sum(1)[None] - x, 1))(B @ A.T)
            )
            oth.append(pq_counts(iou))
        rows.append((idof.get(tuple(sorted(a["id"] for a in recs[b])), "empty-0"), n, np.array(mod[b]), np.array(oth).sum(0)))


def P(c):
    return pq_from_counts(*c)


mp = np.array([P(r[2]) for r in rows])
ap = np.array([P(r[3]) for r in rows])
print(
    "records",
    len(rows),
    "model > others:",
    (mp > ap).mean().round(3),
    "tie",
    (mp == ap).mean().round(3),
    "model<others",
    (mp < ap).mean().round(3),
)
print("mean per-record PQ model", mp.mean().round(4), "others", ap.mean().round(4), "median diff", np.median(mp - ap).round(3))
print("pooled model", round(P(sum(r[2] for r in rows)), 4), "pooled others", round(P(sum(r[3] for r in rows)), 4))
# by prefix (rater code)
g = collections.defaultdict(list)
for r in rows:
    g[r[0].split("-")[0]].append(r)
print("prefix  nrec  model  others")
res = []
for k_, v in sorted(g.items()):
    res.append((k_, len(v), P(sum(x[2] for x in v)), P(sum(x[3] for x in v))))
for x in res:
    print(f"{x[0]} {x[1]:4d} {x[2]:.3f} {x[3]:.3f}")
big = [x for x in res if x[1] >= 15]
print("prefixes with >=15 records:", len(big), "model > others in", sum(x[2] > x[3] for x in big))
