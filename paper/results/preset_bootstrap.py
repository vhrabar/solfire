"""Bootstrap confidence intervals of the PQ of the released presets (pooled out-of-fold, with and without barbs),
and of the paired differences between presets. Run: python paper/results/preset_bootstrap.py paper/results/preset_bootstrap.json"""

import json
import multiprocessing
import pickle
import sys
from pathlib import Path

import numpy as np

# refined pickles were written when the package was called src
import solfire
import solfire.inference
import solfire.inference.refiner
from solfire.data import TRAIN_ANNOTATIONS, TRAIN_IMAGES, load_image
from solfire.data.dataset import load_annotations
from solfire.data.preprocess import observation_path
from solfire.inference.barbs import grow_barbs
from solfire.inference.postprocess import SUBMISSION_SIZE, label_instances
from solfire.inference.refiner import assemble
from solfire.training.calibrate import load_probabilities
from solfire.training.evaluate import bootstrap_pq, evaluate_observation, summarise
from solfire.training.metrics import annotation_pixels

sys.modules["src"] = solfire
sys.modules["src.inference"] = solfire.inference
sys.modules["src.inference.refiner"] = solfire.inference.refiner

BARBS = {"contrast": 0.05, "ridge": 0.025, "band": 20, "min_ratio": 4.0, "max_width": 4.0}
PRESETS = {
    "final": dict(dir="runs/refiner/zoom-ssl-tta-rt/refined", threshold=0.45, min_score=0.32, expect=0.46917),
    "fast": dict(dir="runs/refiner/zoom-ssl/refined", threshold=0.5, min_score=0.32, expect=0.46635),
    "lite": dict(dir="runs/refiner/r34/refined", threshold=0.45, min_score=0.28, expect=0.46374),
    "tiny": dict(cal="runs/ensembles/ctl-hrnet-oof.calibration.json", expect=0.45308),
}
_ctx = {}


def labels_of(preset, name):
    p = PRESETS[preset]
    if "dir" in p:
        with open(Path(p["dir"]) / f"{Path(name).stem}.pkl", "rb") as h:
            stored = pickle.load(h)
        labels, n, _ = assemble(
            stored["refined"], SUBMISSION_SIZE, SUBMISSION_SIZE, threshold=p["threshold"], min_score=p["min_score"]
        )
        return labels, n
    c = _ctx["cal"]
    return label_instances(
        load_probabilities([_ctx["hrnet"][Path(name).stem]], name), c["threshold"], c["offset"], c["min_area"], c["min_score"]
    )


def one(job):
    preset, name, annotations = job
    labels, n = labels_of(preset, name)
    gt = [annotation_pixels(r) for r in annotations]
    plain = evaluate_observation(labels, n, gt)
    image = load_image(observation_path(TRAIN_IMAGES, name)).numpy()[0]
    grown = grow_barbs(labels, n, image, BARBS)
    return plain, evaluate_observation(grown, n, gt), int((grown > 0).sum() - (labels > 0).sum())


def main():
    images, by_image = load_annotations(TRAIN_ANNOTATIONS)
    per_file = {}
    for image in images:
        per_file.setdefault(image["file_name"], []).append(by_image[image["id"]])
    assert len(per_file) == 707 and sum(map(len, per_file.values())) == 1154
    with open(PRESETS["tiny"]["cal"]) as handle:
        cal = json.load(handle)
    hrnet = {}
    for ck in cal["checkpoints"]:
        d = Path(ck).parent / "best.val"
        for f in d.glob("*.npy"):
            hrnet[f.stem] = d
    assert len(hrnet) == 707, len(hrnet)
    _ctx.update(cal=cal, hrnet=hrnet)
    names = sorted(per_file)
    counts = {}
    with multiprocessing.get_context("fork").Pool(4) as pool:
        for preset, p in PRESETS.items():
            res = pool.map(one, [(preset, n, per_file[n]) for n in names], chunksize=8)
            for key, index in (("plain", 0), ("barbs", 1)):
                counts[preset, key] = np.array([r[index]["counts"] for r in res])
            pq = summarise([r[0] for r in res])["pq"]
            print(preset, "pq", round(pq, 5), "expect", p["expect"], flush=True)
            assert abs(pq - p["expect"]) < 1e-4, "does not reproduce"

    def pq_of(c):
        t = c.sum(0)
        return t[0] / (t[1] + 0.5 * t[2] + 0.5 * t[3])

    out = {"observations": len(names), "draws": 2000, "unit": "observation", "presets": {}, "paired_barbs": {}}
    for preset in PRESETS:
        out["presets"][preset] = {
            key: {"pq": pq_of(counts[preset, key]), "ci": bootstrap_pq(counts[preset, key])} for key in ("plain", "barbs")
        }
    order = list(PRESETS)
    for i, a in enumerate(order):
        for b in order[i + 1 :]:
            out["paired_barbs"][f"{a} - {b}"] = {
                "diff": pq_of(counts[a, "barbs"]) - pq_of(counts[b, "barbs"]),
                "ci": bootstrap_pq(counts[a, "barbs"], counts[b, "barbs"]),
            }
    with open(sys.argv[1], "w") as handle:
        json.dump(out, handle, indent=1)
    np.savez_compressed(
        sys.argv[1].replace(".json", "_counts.npz"), names=names, **{f"{k[0]}_{k[1]}": v for k, v in counts.items()}
    )
    for preset, v in out["presets"].items():
        print(
            f"{preset:6s} barbs {v['barbs']['pq']:.4f} [{v['barbs']['ci'][0]:.4f}, {v['barbs']['ci'][1]:.4f}]  plain {v['plain']['pq']:.4f} [{v['plain']['ci'][0]:.4f}, {v['plain']['ci'][1]:.4f}]"
        )
    for k, v in out["paired_barbs"].items():
        print(f"{k:14s} {v['diff']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]")


main()
