"""
Fine-structure diagnostic on the pooled out-of-fold maps of the cv15-hrnet ensemble.

"""

import itertools
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.morphology import disk, skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataset import kfold_split, load_annotations
from src.inference.postprocess import connected_components, erode, instance_scores, keep_labels
from src.training.calibrate import load_probabilities
from src.training.metrics import annotation_pixels, instance_iou, pq_counts

# FINE_SET=thin measures the thin-aware fine-tunes (sweeps/finetune_thin*.json) instead
SET = os.environ.get("FINE_SET", "lovasz")
THRESHOLD = float(os.environ.get("FINE_THRESHOLD", 0.375))
MIN_SCORE = float(os.environ.get("FINE_MIN_SCORE", 0.675))
RADII = (1, 2)
SETS = {
    "lovasz": {
        "r34": "runs/finetune/c{:02d}_lovasz-fold{k}_seed1/best.val",
        "mitb2": "runs/finetune_mitb2/c{:02d}_lovasz-mitb2-fold{k}_seed1/best.val",
        "hrnet": "runs/finetune_hrnet/c{:02d}_lovasz-hrnet32-fold{k}_seed1/best.val",
    },
    "thin": {
        "r34": "runs/finetune_thin/c{:02d}_thin-r34-fold{k}_seed1/best.val",
        "mitb2": "runs/finetune_thin_mitb2/c{:02d}_thin-mitb2-fold{k}_seed1/best.val",
        "hrnet": "runs/finetune_thin_hrnet/c{:02d}_thin-hrnet32-fold{k}_seed1/best.val",
    },
}
FAMILIES = SETS[SET]
FIRST = {"r34": 6, "mitb2": 1, "hrnet": 1} if SET == "lovasz" else {"r34": 1, "mitb2": 1, "hrnet": 1}
H = W = 2048


def cache_dirs(fold):
    return [FAMILIES[f].format(FIRST[f] + fold, k=fold) for f in FAMILIES]


def labels_from_mask(mask, probabilities, min_score):
    labels, n = connected_components(mask)
    if min_score > 0 and n:
        labels, n = keep_labels(labels, n, instance_scores(labels, n, probabilities) >= min_score)
    return labels, n


def thin_mask(mask, radius):
    return mask & ~ndimage.binary_opening(mask, structure=disk(radius))


def job(args):
    fold, file_name, records = args
    p = load_probabilities(cache_dirs(fold), file_name)
    p0 = p >= THRESHOLD
    p1 = erode(p0, 1)
    skel = skeletonize(p0)
    variants = {
        "thr0.375_noerode": p0,
        "thr0.375_erode1 (shipped)": p1,
        "erode1 + skeleton of thr mask": p1 | skel,
    }
    for r in RADII:
        variants[f"erode1 + thin(r={r}) of thr mask"] = p1 | thin_mask(p0, r)
    out = {
        "pix": defaultdict(lambda: np.zeros(4)),
        "pq": defaultdict(lambda: np.zeros(4)),
        "oracle": defaultdict(lambda: np.zeros(4)),
        "inst": [],
    }
    labelled = {k: labels_from_mask(m, p, MIN_SCORE) for k, m in variants.items()}
    for record in records:
        gt_pixels = annotation_pixels(record)
        if not gt_pixels:
            continue
        union = np.zeros(H * W, dtype=bool)
        for px in gt_pixels:
            union[px] = True
        union = union.reshape(H, W)
        thin = {r: thin_mask(union, r).ravel() for r in RADII}
        for name, (labels, n) in labelled.items():
            iou = instance_iou(gt_pixels, labels, n)
            out["pq"][name] += np.array(pq_counts(iou))
            pred = labels.ravel() > 0
            for r in RADII:
                t = thin[r]
                body = union.ravel() & ~t
                out["pix"][(name, r)] += np.array([pred[t].sum(), t.sum(), pred[body].sum(), body.sum()])
            # oracle: add every missed thin pixel of each annotation to its best prediction
            if name.endswith("(shipped)"):
                for r in RADII:
                    iou2 = iou.copy()
                    flat = labels.ravel()
                    for j, px in enumerate(gt_pixels):
                        if n == 0:
                            break
                        k = int(np.argmax(iou[j]))
                        if iou[j, k] <= 0:
                            continue
                        inter = int((flat[px] == k + 1).sum())
                        missed_thin = int((thin[r][px] & (flat[px] != k + 1)).sum())
                        area_k = int((flat == k + 1).sum())
                        union_jk = len(px) + area_k - inter
                        iou2[j, k] = (inter + missed_thin) / union_jk
                    out["oracle"][(r,)] += np.array(pq_counts(iou2))
                # per-annotation: thin fraction and hit
                for j, px in enumerate(gt_pixels):
                    hit = bool(n and iou[j].max() > 0.5)
                    out["inst"].append((len(px), float(thin[2][px].mean()), hit, float(iou[j].max()) if n else 0.0))
    return {k: {kk: v.tolist() for kk, v in d.items()} if isinstance(d, dict) else d for k, d in out.items()}


def main():
    images, by_id = load_annotations()
    by_file = defaultdict(list)
    for im in images:
        by_file[im["file_name"]].append(by_id[im["id"]])
    jobs = []
    for fold in range(5):
        _, val_ids = kfold_split(images, fold)
        files = sorted({im["file_name"] for im in images if im["id"] in set(val_ids)})
        jobs += [(fold, f, by_file[f]) for f in files]
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    jobs = jobs[:limit] if limit else jobs
    pix, pq, oracle, inst = (
        defaultdict(lambda: np.zeros(4)),
        defaultdict(lambda: np.zeros(4)),
        defaultdict(lambda: np.zeros(4)),
        [],
    )
    with ProcessPoolExecutor(8) as pool:
        for i, res in enumerate(pool.map(job, jobs, chunksize=4)):
            for k, v in res["pix"].items():
                pix[k] += v
            for k, v in res["pq"].items():
                pq[k] += v
            for k, v in res["oracle"].items():
                oracle[k] += v
            inst += res["inst"]
            if i % 100 == 0:
                print(f"{i}/{len(jobs)}", file=sys.stderr, flush=True)

    def pq_of(v):
        s, tp, fp, fn = v
        return s / (tp + 0.5 * fp + 0.5 * fn), int(tp), int(fp), int(fn), s / max(tp, 1)

    print(f"\n{len(jobs)} files; threshold {THRESHOLD}, min_score {MIN_SCORE}\n")
    print("GT pixel share that is thin (removed by opening):")
    for r in RADII:
        k = ("thr0.375_erode1 (shipped)", r)
        print(f"  r={r} (<~{2 * r + 1} px): {pix[k][1] / (pix[k][1] + pix[k][3]):.3f}")
    print("\nPixel recall on thin vs body GT pixels, and PQ / TP / FP / FN / SQ per variant:")
    for name in pq:
        row = [f"{name:38s}"]
        for r in RADII:
            k = (name, r)
            row.append(f"thin{r} {pix[k][0] / pix[k][1]:.3f}  body{r} {pix[k][2] / pix[k][3]:.3f}")
        p, tp, fp, fn, sq = pq_of(pq[name])
        row.append(f"PQ {p:.4f} TP {tp} FP {fp} FN {fn} SQ {sq:.3f}")
        print("  " + " | ".join(row))
    print("\nOracle: shipped prediction + every missed thin GT pixel added to the best-overlapping prediction:")
    for r in RADII:
        p, tp, fp, fn, sq = pq_of(oracle[(r,)])
        print(f"  r={r}: PQ {p:.4f} TP {tp} FP {fp} FN {fn} SQ {sq:.3f}")
    inst = np.array(inst)
    q = np.quantile(inst[:, 1], [0.2, 0.4, 0.6, 0.8])
    print("\nPer-annotation hit rate by thin-fraction quintile (r=2), shipped variant:")
    edges = [-1, *q, 2]
    for a, b in itertools.pairwise(edges):
        sel = (inst[:, 1] > a) & (inst[:, 1] <= b)
        print(
            f"  thin frac ({max(a, 0):.2f}, {min(b, 1):.2f}]: n {sel.sum():5d}  hit {inst[sel, 2].mean():.3f}  median area {np.median(inst[sel, 0]):.0f}  mean best IoU {inst[sel, 3].mean():.3f}"
        )
    with open(Path("runs/ensembles/fine_structure.json"), "w") as fh:
        json.dump(
            {
                "pix": {str(k): v.tolist() for k, v in pix.items()},
                "pq": {k: v.tolist() for k, v in pq.items()},
                "oracle": {str(k): v.tolist() for k, v in oracle.items()},
            },
            fh,
            indent=1,
        )


if __name__ == "__main__":
    main()
