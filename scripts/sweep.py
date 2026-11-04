"""
Parameter sweep over ``solfire.main train`` + ``calibrate``
"""

import argparse
import codecs
import csv
import itertools
import json
import shlex
import statistics
import subprocess
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.pick_checkpoint import best_checkpoint

REPO = Path(__file__).resolve().parent.parent
BOOL_OPTIONAL = {"bilinear", "tta"}
CALIBRATION_FIELDS = ("pq", "threshold", "offset", "min_area")
DEFAULT_LAUNCHER = "uv run --group cuda-134 python"


def parse_scalar(text: str):
    """
    Read a command-line value as int, then float, then bool, else leave it as text.
    :param text: The raw string
    :return: The parsed value
    """
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    return text


def load_spec(path: Path) -> dict:
    """
    Read a sweep spec from JSON or TOML.
    :param path: ``.json`` or ``.toml`` file
    :return: The spec dict
    :raises SystemExit: unknown extension or unreadable file
    """
    if not path.is_file():
        sys.exit(f"spec {path} not found")
    if path.suffix == ".toml":
        return tomllib.loads(path.read_text())
    if path.suffix == ".json":
        return json.loads(path.read_text())
    sys.exit(f"spec must be .json or .toml, got {path.suffix!r}")


def render_args(params: dict) -> list[str]:
    """
    Turn a ``{snake_key: value}`` config into a ``solfire.main`` argument list.
    :param params: Config parameters; ``None`` values are dropped
    :return: Flat list of ``--kebab-key`` / value tokens
    """
    tokens: list[str] = []
    for key, value in params.items():
        flag = "--" + key.replace("_", "-")
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                tokens.append(flag)
            elif key in BOOL_OPTIONAL:
                tokens.append("--no-" + key.replace("_", "-"))
            continue
        if isinstance(value, (list, tuple)):
            tokens.append(flag)
            tokens.extend(str(item) for item in value)
            continue
        tokens += [flag, str(value)]
    return tokens


def slug(value) -> str:
    """Filesystem-safe rendering of one parameter value."""
    text = ",".join(str(v) for v in value) if isinstance(value, (list, tuple)) else str(value)
    return "".join(c if c.isalnum() or c in ".-" else "-" for c in text)


def expand_configs(spec: dict) -> list[dict]:
    """
    Build the ordered list of configs from ``base``, ``grid`` and ``configs``.

    :param spec: The sweep spec
    :return: One dict per config, each with ``_name`` (label), ``_keys`` (the keys
        that vary across the sweep) and ``_params`` (full train args)
    """
    base = dict(spec.get("base", {}))
    grid = spec.get("grid", {})
    grid_keys = sorted(grid)
    varying: set[str] = set(grid_keys)

    combos = [dict(zip(grid_keys, choice)) for choice in itertools.product(*(grid[k] for k in grid_keys))] or [{}]
    raw = [("grid", combo) for combo in combos]
    for entry in spec.get("configs", []):
        entry = dict(entry)
        name = entry.pop("name", None)
        varying.update(entry)
        raw.append((name, entry))

    configs = []
    for index, (name, overrides) in enumerate(raw):
        params = {**base, **overrides}
        if name is None or name == "grid":
            label_bits = [f"{k}{slug(overrides[k])}" for k in sorted(overrides)] or ["base"]
            name = "-".join(label_bits)[:70]
        configs.append(
            {
                "_name": name,
                "_tag": f"c{index:02d}",
                "_keys": sorted(varying),
                "_overrides": overrides,
                "_params": params,
            }
        )
    return configs


def read_calibration(checkpoint: Path) -> dict | None:
    """
    Load the calibration JSON that sits next to ``checkpoint``.
    :param checkpoint: ``*.pt`` path
    :return: ``{"pq", "threshold", "offset", "min_area"}`` or ``None`` when absent
    """
    path = checkpoint.with_name(checkpoint.stem + ".calibration.json")
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    return {field: data[field] for field in CALIBRATION_FIELDS}


def stream(command: list[str], log_path: Path) -> int:
    """
    Run ``command`` from the repo root, mirroring its raw bytes to our stdout and
    logging only completed lines to ``log_path``.

    :param command: argv
    :param log_path: File the completed-line log is written to
    :return: The process exit code
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    stdout = sys.stdout.buffer if hasattr(sys.stdout, "buffer") else sys.stdout
    with (
        subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0) as process,
        open(log_path, "w", encoding="utf-8") as log,
    ):
        assert process.stdout is not None
        line = ""
        while chunk := process.stdout.read(4096):
            stdout.write(chunk)
            stdout.flush()
            for char in decoder.decode(chunk):
                if char == "\n":
                    log.write(line + "\n")
                    line = ""
                elif char == "\r":
                    line = ""
                else:
                    line += char
        if line:
            log.write(line + "\n")
        return process.wait()


def attempt(command: list[str], log_path: Path, retries: int) -> bool:
    """
    Run ``command``, and again after a failure; training resumes from its ``last.pt``, so a
    crash costs at most the epoch it happened in.
    :param command: argv
    :param log_path: Log of the latest attempt; earlier ones are kept as ``<log>.<n>``
    :param retries: Further attempts after the first failure
    :return: Whether an attempt succeeded
    """
    for number in range(retries + 1):
        if number:
            log_path.replace(log_path.with_name(f"{log_path.name}.{number}"))
            print(f"  attempt {number + 1}/{retries + 1} after a failure; see {log_path.name}.{number}")
        if stream(command, log_path) == 0:
            return True
    return False


def run_one(config: dict, seed: int, sweep_dir: Path, calibrate_args: list[str] | None, device: list[str], args) -> dict:
    """
    Train + calibrate one config at one seed, or reuse an existing result.
    :return: A results row (status is ``ok``, ``cached``, ``trained`` (no calibration asked for),
        ``train-failed`` or ``calibrate-failed``)
    """
    run_dir = sweep_dir / f"{config['_tag']}_{config['_name']}_seed{seed}"
    row = {"config": config["_name"], "tag": config["_tag"], "seed": seed, "run_dir": run_dir.relative_to(REPO)}
    for key in config["_keys"]:
        row[key] = config["_params"].get(key)

    found = best_checkpoint(run_dir)
    if found is not None and not args.force:
        checkpoint, train_pq = found
        calibration = read_calibration(checkpoint)
        if calibration is not None:
            print(f"  cached: {run_dir.relative_to(REPO)}  calibrated pq {calibration['pq']:.4f}")
            return {
                **row,
                "checkpoint": checkpoint.relative_to(REPO),
                "train_pq": train_pq,
                "calibrated_pq": calibration["pq"],
                "threshold": calibration["threshold"],
                "offset": calibration["offset"],
                "min_area": calibration["min_area"],
                "status": "cached",
            }
    launcher = shlex.split(args.launcher)
    train_cmd = [
        *launcher,
        "-m",
        "solfire.main",
        "--seed",
        str(seed),
        *device,
        "train",
        *render_args(config["_params"]),
        "--output-dir",
        str(run_dir),
        "--resume",
    ]
    calibrate_cmd_prefix = [*launcher, "-m", "solfire.main", *device, "calibrate"]

    if args.dry_run:
        print("  train:    " + " ".join(train_cmd))
        if calibrate_args is not None:
            print(
                "  calibrate: " + " ".join([*calibrate_cmd_prefix, "--checkpoint", str(run_dir / "<best>.pt"), *calibrate_args])
            )
        return {**row, "status": "dry-run"}

    if not attempt(train_cmd, run_dir.with_suffix(".train.log"), args.retries):
        return {**row, "status": "train-failed"}
    found = best_checkpoint(run_dir)
    if found is None:
        return {**row, "status": "train-failed"}
    checkpoint, train_pq = found
    if calibrate_args is None:
        return {**row, "checkpoint": checkpoint.relative_to(REPO), "train_pq": train_pq, "status": "trained"}

    calibrated = attempt(
        [*calibrate_cmd_prefix, "--checkpoint", str(checkpoint), *calibrate_args],
        run_dir.with_suffix(".calibrate.log"),
        args.retries,
    )
    calibration = read_calibration(checkpoint) if calibrated else None
    if calibration is None:
        return {**row, "checkpoint": checkpoint.relative_to(REPO), "train_pq": train_pq, "status": "calibrate-failed"}

    return {
        **row,
        "checkpoint": checkpoint.relative_to(REPO),
        "train_pq": train_pq,
        "calibrated_pq": calibration["pq"],
        "threshold": calibration["threshold"],
        "offset": calibration["offset"],
        "min_area": calibration["min_area"],
        "status": "ok",
    }


def write_csv(path: Path, rows: list[dict], keys: list[str]) -> None:
    """Write every row so far, so an interrupted sweep keeps its partial results."""
    columns = [
        "config",
        "tag",
        "seed",
        *keys,
        "train_pq",
        "calibrated_pq",
        "threshold",
        "offset",
        "min_area",
        "checkpoint",
        "run_dir",
        "status",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], keys: list[str]) -> None:
    """Rank configs by mean calibrated PQ across their seeds and print a table."""
    scored: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("status") in ("ok", "cached"):
            scored.setdefault(row["config"], []).append(row)
    if not scored:
        print("\nno completed runs to rank")
        return

    ranked = sorted(
        scored.items(),
        key=lambda item: statistics.mean(r["calibrated_pq"] for r in item[1]),
        reverse=True,
    )
    width = max(len(name) for name in scored)
    header = f"{'config':<{width}}  {'n':>2}  {'calib_pq mean':>13}  {'std':>6}  {'best':>6}  " + "  ".join(k for k in keys)
    print("\n" + header)
    print("-" * len(header))
    for name, group in ranked:
        values = [r["calibrated_pq"] for r in group]
        params = group[0]
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        cells = "  ".join(str(params.get(k)) for k in keys)
        print(f"{name:<{width}}  {len(values):>2}  {statistics.mean(values):>13.4f}  {std:>6.4f}  {max(values):>6.4f}  {cells}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec", nargs="?", type=Path, help="sweep spec, .json or .toml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="add/override a base train arg")
    parser.add_argument("--grid", action="append", default=[], metavar="KEY=V1,V2", help="add/override a swept arg")
    parser.add_argument("--seeds", help="comma-separated, overrides the spec")
    parser.add_argument("--name", help="sweep name -> runs/<name>/, overrides the spec")
    parser.add_argument("--device", help="passed through to solfire.main")
    parser.add_argument(
        "--launcher",
        default=DEFAULT_LAUNCHER,
        help="Python command the runs start with, e.g. 'python' where uv is unavailable (Kaggle)",
    )
    parser.add_argument("--only", help="run only configs whose name contains one of these comma-separated substrings")
    parser.add_argument("--max-runs", type=int, default=0, help="stop after this many (config, seed) runs; 0 is no limit")
    parser.add_argument("--force", action="store_true", help="retrain configs that already have a calibration.json")
    parser.add_argument("--retries", type=int, default=2, help="re-run a failed train/calibrate this many times")
    parser.add_argument("--continue-on-error", action="store_true", help="keep going after a failed train/calibrate")
    parser.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    args = parser.parse_args()

    spec = load_spec(args.spec) if args.spec else {}
    for item in args.set:
        key, _, value = item.partition("=")
        spec.setdefault("base", {})[key] = parse_scalar(value)
    for item in args.grid:
        key, _, value = item.partition("=")
        spec.setdefault("grid", {})[key] = [parse_scalar(v) for v in value.split(",")]

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else spec.get("seeds", [1])
    sweep_name = args.name or spec.get("name", "sweep")
    # "calibrate": false trains only, for runs with nothing held out to calibrate on
    calibrate_spec = spec.get("calibrate", {})
    calibrate_args = None if calibrate_spec is False else render_args(calibrate_spec)
    device = ["--device", args.device] if args.device else []

    configs = expand_configs(spec)
    if args.only:
        wanted = args.only.split(",")
        configs = [c for c in configs if any(w in c["_name"] for w in wanted)]
    if not configs:
        sys.exit("no configs to run")
    keys = configs[0]["_keys"]

    sweep_dir = REPO / "runs" / sweep_name
    results_csv = sweep_dir / "results.csv"
    if not args.dry_run:
        sweep_dir.mkdir(parents=True, exist_ok=True)
        (sweep_dir / "configs.json").write_text(
            json.dumps([{"tag": c["_tag"], "name": c["_name"], "params": c["_params"]} for c in configs], indent=2)
        )

    plan = [(c, s) for c in configs for s in seeds]
    if args.max_runs:
        plan = plan[: args.max_runs]
    print(
        f"sweep '{sweep_name}': {len(configs)} config(s) x {len(seeds)} seed(s) = {len(plan)} run(s) -> {sweep_dir.relative_to(REPO)}"
    )
    print(f"varying: {', '.join(keys) or '(nothing)'}")

    rows: list[dict] = []
    for index, (config, seed) in enumerate(plan, 1):
        print(f"\n[{index}/{len(plan)}] {config['_tag']} {config['_name']}  seed {seed}")
        row = run_one(config, seed, sweep_dir, calibrate_args, device, args)
        rows.append(row)
        if not args.dry_run:
            write_csv(results_csv, rows, keys)
        if row["status"] not in ("ok", "cached", "trained", "dry-run"):
            print(f"  {row['status']}")
            if not args.continue_on_error:
                sys.exit(f"stopping: {config['_name']} seed {seed} {row['status']} (see {sweep_dir.relative_to(REPO)})")

    if args.dry_run:
        return
    summarize(rows, keys)
    print(f"\nwrote {results_csv.relative_to(REPO)}")


if __name__ == "__main__":
    main()
