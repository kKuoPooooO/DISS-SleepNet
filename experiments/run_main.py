"""
Main pipeline runner — edf78 + isruc1 (+ seq-len=1 variant on all 3 datasets).

Tasks per dataset:
  1. SSL pretrain  (barlow_truepinball_adaptive)  x 10 folds
  2. Proxy probe   (checkpoint selection)          x 10 folds
  3. finetune_v1   (supervised, fixed target IDs)  x 10 folds x 6 ratios
  4. finetune_v2   (+ soft-label correction)       x 10 folds x 6 ratios
  5. seq1_finetune (seq-len=1, reuse proxy ckpt)   x 10 folds x 6 ratios

edf20 v1/v2 results already exist in legacy_repro — referenced but not re-run.
Idempotent: tasks whose test_metrics.json / pretrain_metrics.json exist are skipped.
"""
from __future__ import annotations

import argparse, json, os, shutil, subprocess, time
from datetime import datetime
from pathlib import Path

PY   = "/path/to/python"
REPO = Path(__file__).resolve().parents[1]
PROJ = Path("/path/to/project")

# ── Dataset configs ────────────────────────────────────────────────────────────
DATASETS = {
    "edf78": {
        "h5":     PROJ / "data/processed/sleepedf78_patient_meta.h5",
        "raw":    PROJ / "data/processed/sleepedf78_raw_patient_memmap",
        "rhythm": PROJ / "data/processed/sleepedf78_patient_memmap",
        "subsets": PROJ / "newworld0804/ssl_module_selection/label_subsets_edf78",
    },
    "isruc1": {
        "h5":     PROJ / "data/processed/isruc1.h5",
        "raw":    PROJ / "data/processed/isruc1_raw_patient_memmap",
        "rhythm": PROJ / "data/processed/isruc1_patient_memmap",
        "subsets": PROJ / "newworld0804/ssl_module_selection/label_subsets_isruc1",
    },
}

# edf20 reference (results already exist — used only for seq-len=1 variant)
EDF20 = {
    "h5":     PROJ / "data/processed/sleepedf20_patient_meta.h5",
    "raw":    PROJ / "data/processed/sleepedf20_raw_patient_memmap",
    "rhythm": PROJ / "data/processed/sleepedf20_patient_memmap",
    "subsets": PROJ / "newworld0804/ssl_module_selection/label_subsets",
    "proxy":   PROJ / "newworld0804/ssl_module_selection/proxy/barlow_pinball_adaptive",
}

RATIOS     = [1.0, 0.50, 0.25, 0.10, 0.05, 0.01]
RATIO_TAG  = {r: f"label{int(round(r*100))}pct" for r in RATIOS}
FOLDS      = list(range(10))

OUT_ROOT   = Path("/path/to/output/main_experiments")
LOGS       = OUT_ROOT / "logs"

# ── Helpers ────────────────────────────────────────────────────────────────────
def ts() -> str:
    return datetime.now().strftime("%F %T")

def readj(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None

CUDA_FAILURE_MARKERS = (
    "CUDA error",
    "CUDA-capable device(s) is/are busy or unavailable",
    "couldn't communicate with the NVIDIA driver",
    "Failed to initialize NVML",
    "unspecified launch failure",
)


def gpu_available() -> bool:
    try:
        r = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
        return r.returncode == 0
    except Exception:
        return False


def wait_for_gpu(label: str, retry_delay: int = 300) -> None:
    while not gpu_available():
        print(f"  GPU unavailable after {label}; retrying nvidia-smi in {retry_delay}s", flush=True)
        time.sleep(retry_delay)


def is_cuda_failure(logfile: Path) -> bool:
    if not logfile.exists():
        return False
    text = logfile.read_text(encoding="utf-8", errors="ignore")[-12000:]
    return any(marker in text for marker in CUDA_FAILURE_MARKERS)


def clean_incomplete_output(marker: Path) -> None:
    out_dir = marker.parent
    if out_dir.exists() and not marker.exists():
        shutil.rmtree(out_dir)


def run_once(cmd: list, logfile: Path, label: str, gpu_id: int = 0) -> int:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES=str(gpu_id))
    t0 = time.time()
    with logfile.open("a", encoding="utf-8") as fh:
        fh.write(f"\n===== {ts()} {label}\n")
        fh.write(" ".join(str(x) for x in cmd) + "\n")
        fh.flush()
        r = subprocess.run([str(x) for x in cmd], cwd=str(REPO),
                           env=env, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode == 0:
        print(f"[OK] {label}  {time.time()-t0:.0f}s", flush=True)
    return r.returncode


def run(cmd: list, logfile: Path, label: str, marker: Path, gpu_id: int = 0) -> None:
    attempts = 0
    while True:
        attempts += 1
        rc = run_once(cmd, logfile, label, gpu_id)
        if rc == 0:
            return
        if not is_cuda_failure(logfile):
            raise RuntimeError(f"FAILED rc={rc} [{label}] -> {logfile}")
        print(f"[CUDA FAIL] {label} attempt {attempts} rc={rc}; cleaning incomplete output", flush=True)
        clean_incomplete_output(marker)
        wait_for_gpu(label)
        print(f"  RETRY {label}", flush=True)

def write_status(path: Path, done: int, total: int, last: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"now": ts(), "done": done, "total": total, "last": last}, indent=2))

# ── Command builders ───────────────────────────────────────────────────────────
def pretrain_cmd(out: Path, ds: dict, fold: int) -> list:
    return [
        PY, "-u", REPO / "scripts/pretrain.py",
        "--h5", ds["h5"], "--raw-memmap-dir", ds["raw"],
        "--rhythm-memmap-dir", ds["rhythm"],
        "--out-dir", out, "--fold", fold,
        "--method", "barlow_truepinball_adaptive",
        "--epochs", 50, "--batch-size", 256, "--num-workers", 8,
        "--prefetch-factor", 4, "--lr", "3e-4", "--weight-decay", "1e-4",
        "--save-every", 10, "--seed", 42,
    ]

def proxy_cmd(out: Path, pre_dir: Path, ds: dict, fold: int) -> list:
    return [
        PY, "-u", REPO / "scripts/proxy_probe.py",
        "--h5", ds["h5"], "--raw-memmap-dir", ds["raw"],
        "--rhythm-memmap-dir", ds["rhythm"],
        "--pretrain-dir", pre_dir, "--out-dir", out, "--fold", fold,
        "--epochs", 15, "--batch-size", 256, "--num-workers", 8, "--seed", 42,
    ]

def finetune_v1_cmd(out: Path, ds: dict, fold: int, ratio: float, ssl_ckpt: str) -> list:
    ids = str(ds["subsets"] / f"fold{fold}_{RATIO_TAG[ratio]}_target_ids.npy")
    return [
        PY, "-u", REPO / "scripts/finetune_nested.py",
        "--h5", ds["h5"], "--raw-memmap-dir", ds["raw"],
        "--rhythm-memmap-dir", ds["rhythm"],
        "--out-dir", out, "--fold", fold,
        "--label-ratio", ratio, "--label-target-ids", ids,
        "--pretrained-ckpt", ssl_ckpt,
        "--epochs", 80, "--batch-size", 96, "--num-workers", 8,
        "--val-num-workers", 4, "--prefetch-factor", 4,
        "--lr", "3e-4", "--weight-decay", "1e-4",
        "--patience", 10, "--selection-metric", "acc", "--seed", 42,
    ]

def finetune_v2_cmd(out: Path, ds: dict, fold: int, ratio: float,
                    ssl_ckpt: str, v1_best: str) -> list:
    ids = str(ds["subsets"] / f"fold{fold}_{RATIO_TAG[ratio]}_target_ids.npy")
    return [
        PY, "-u", REPO / "scripts/finetune.py",
        "--h5", ds["h5"], "--raw-memmap-dir", ds["raw"],
        "--rhythm-memmap-dir", ds["rhythm"],
        "--out-dir", out, "--fold", fold,
        "--label-ratio", ratio, "--label-target-ids", ids,
        "--pretrained-ckpt", ssl_ckpt,
        "--soft-teacher", v1_best,
        "--soft-mode", "pinball_disp",
        "--soft-warmup", 5, "--pinball-disp-quantile", 0.30,
        "--soft-weight", 0.1, "--soft-temp", 2.0,
        "--epochs", 80, "--batch-size", 96, "--num-workers", 8,
        "--val-num-workers", 4, "--prefetch-factor", 4,
        "--lr", "3e-4", "--weight-decay", "1e-4",
        "--patience", 10, "--selection-metric", "acc", "--seed", 42,
    ]

def seq1_cmd(out: Path, ds: dict, fold: int, ratio: float, ssl_ckpt: str,
             subsets_dir: Path) -> list:
    """seq-len=1 variant: single-epoch input, no temporal context."""
    ids = str(subsets_dir / f"fold{fold}_{RATIO_TAG[ratio]}_target_ids.npy")
    return [
        PY, "-u", REPO / "scripts/finetune_nested.py",
        "--h5", ds["h5"], "--raw-memmap-dir", ds["raw"],
        "--rhythm-memmap-dir", ds["rhythm"],
        "--out-dir", out, "--fold", fold,
        "--label-ratio", ratio, "--label-target-ids", ids,
        "--pretrained-ckpt", ssl_ckpt,
        "--seq-len", 1,
        "--epochs", 80, "--batch-size", 96, "--num-workers", 8,
        "--val-num-workers", 4, "--prefetch-factor", 4,
        "--lr", "3e-4", "--weight-decay", "1e-4",
        "--patience", 10, "--selection-metric", "acc", "--seed", 42,
    ]

# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of parallel shards (one per GPU)")
    parser.add_argument("--shard-id", type=int, default=0,
                        help="Which shard this process handles (0-indexed)")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="CUDA device index to use (CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print task list without running anything")
    args = parser.parse_args()

    assert 0 <= args.shard_id < args.num_shards, \
        f"shard-id {args.shard_id} out of range for num-shards {args.num_shards}"

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    shard_suffix = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    status = OUT_ROOT / f"STATUS{shard_suffix}.json"

    tasks: list[tuple[str, Path, list]] = []

    # ── Build task list for edf78 + isruc1 ────────────────────────────────────
    for ds_name, ds in DATASETS.items():
        pre_base   = OUT_ROOT / ds_name / "pretrain"
        proxy_base = OUT_ROOT / ds_name / "proxy"
        ft_base    = OUT_ROOT / ds_name / "finetune"
        s1_base    = OUT_ROOT / ds_name / "seq1_finetune"

        for fold in FOLDS:
            pre_dir   = pre_base / f"fold{fold}"
            proxy_dir = proxy_base / f"fold{fold}"

            # pretrain
            tasks.append((
                f"pretrain/{ds_name}/fold{fold}",
                pre_dir / "pretrain_metrics.json",
                pretrain_cmd(pre_dir, ds, fold),
            ))
            # proxy probe — depends on pretrain; runner is sequential so order handles this
            tasks.append((
                f"proxy/{ds_name}/fold{fold}",
                proxy_dir / "proxy_selection.json",
                proxy_cmd(proxy_dir, pre_dir, ds, fold),
            ))

            for ratio in RATIOS:
                rtag  = RATIO_TAG[ratio]
                v1dir = ft_base / "v1" / rtag / f"fold{fold}"
                v2dir = ft_base / "v2" / rtag / f"fold{fold}"
                s1dir = s1_base / rtag / f"fold{fold}"

                # v1 — proxy ckpt resolved at run time from proxy_selection.json
                tasks.append((
                    f"v1/{ds_name}/{rtag}/fold{fold}",
                    v1dir / "test_metrics.json",
                    None,   # command resolved lazily below
                ))
                # v2
                tasks.append((
                    f"v2/{ds_name}/{rtag}/fold{fold}",
                    v2dir / "test_metrics.json",
                    None,
                ))
                # seq-len=1
                tasks.append((
                    f"seq1/{ds_name}/{rtag}/fold{fold}",
                    s1dir / "test_metrics.json",
                    None,
                ))

    # ── seq-len=1 for edf20 (using existing proxy ckpts) ──────────────────────
    for fold in FOLDS:
        sel = readj(EDF20["proxy"] / f"fold{fold}/proxy_selection.json")
        edf20_ckpt = sel["best"]["ckpt"] if sel else ""
        for ratio in RATIOS:
            rtag  = RATIO_TAG[ratio]
            s1dir = OUT_ROOT / "edf20" / "seq1_finetune" / rtag / f"fold{fold}"
            tasks.append((
                f"seq1/edf20/{rtag}/fold{fold}",
                s1dir / "test_metrics.json",
                seq1_cmd(s1dir, EDF20, fold, ratio, edf20_ckpt, EDF20["subsets"]),
            ))

    # ── Shard filtering ────────────────────────────────────────────────────────
    all_tasks = tasks
    tasks = [t for i, t in enumerate(all_tasks) if i % args.num_shards == args.shard_id]

    total_global = len(all_tasks)
    total = len(tasks)
    print(f"Total tasks (global): {total_global}  |  This shard ({args.shard_id}/{args.num_shards}): {total}  |  GPU: {args.gpu_id}", flush=True)

    if args.dry_run:
        for label, marker, _ in tasks:
            status_str = "DONE" if marker.exists() else "PENDING"
            print(f"  [{status_str}] {label}")
        return

    write_status(status, 0, total, "starting")

    # ── Execute ────────────────────────────────────────────────────────────────
    done = 0
    for label, marker, cmd in tasks:
        parts = label.split("/")
        stage = parts[0]

        if marker.exists():
            print(f"  SKIP {label}", flush=True)
            done += 1
            write_status(status, done, total, label)
            continue

        # Resolve lazy commands for finetune tasks
        if cmd is None:
            ds_name = parts[1]
            rtag    = parts[2]
            fold    = int(parts[3].replace("fold", ""))
            ratio   = next(r for r in RATIOS if RATIO_TAG[r] == rtag)
            ds      = DATASETS[ds_name]
            proxy_j = readj(OUT_ROOT / ds_name / "proxy" / f"fold{fold}" / "proxy_selection.json")
            if proxy_j is None:
                raise RuntimeError(f"proxy_selection.json missing for {ds_name}/fold{fold}")
            ssl_ckpt = proxy_j["best"]["ckpt"]

            ft_base = OUT_ROOT / ds_name / "finetune"
            s1_base = OUT_ROOT / ds_name / "seq1_finetune"
            v1dir   = ft_base / "v1" / rtag / f"fold{fold}"
            v2dir   = ft_base / "v2" / rtag / f"fold{fold}"
            s1dir   = s1_base / rtag / f"fold{fold}"

            if stage == "v1":
                cmd = finetune_v1_cmd(v1dir, ds, fold, ratio, ssl_ckpt)
            elif stage == "v2":
                v1best = str(v1dir / "best.pt")
                cmd = finetune_v2_cmd(v2dir, ds, fold, ratio, ssl_ckpt, v1best)
            else:  # seq1
                cmd = seq1_cmd(s1dir, ds, fold, ratio, ssl_ckpt, ds["subsets"])

        print(f"  RUN  {label}", flush=True)
        log = LOGS / (label.replace("/", "_") + f"{shard_suffix}.log")
        run(cmd, log, label, marker, args.gpu_id)
        done += 1
        write_status(status, done, total, label)

    print("\nAll main experiment tasks complete.", flush=True)
    write_status(status, done, total, "DONE")


if __name__ == "__main__":
    main()
