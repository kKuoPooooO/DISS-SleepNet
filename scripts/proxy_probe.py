from __future__ import annotations

import argparse, json, re
from pathlib import Path

import h5py, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sleepssl.utils import compute_metrics, save_run_config, set_seed
from scripts.pretrain import DualSSLTruePinballModel


class ProxyEpochDataset(Dataset):
    def __init__(self, h5_path: str, raw_dir: str, rhythm_dir: str, fold: int, part: str, val_subject_ratio: float = 0.1, seed: int = 42) -> None:
        self.raw_dir, self.rhythm_dir = Path(raw_dir).resolve(), Path(rhythm_dir).resolve()
        self.xraw_path, self.xrhy_path, self.y_path = self.raw_dir / "x_raw.npy", self.rhythm_dir / "x_rhythm.npy", self.raw_dir / "y.npy"
        with h5py.File(h5_path, "r") as f:
            split = np.asarray(f[f"split_fold_{fold}"])
            subjects = np.asarray(f["subject_id"], dtype=np.int64)
            train_subjects = np.unique(subjects[split == 0])
            rng = np.random.default_rng(seed + 313 * fold)
            shuffled = train_subjects.copy()
            rng.shuffle(shuffled)
            n_val = max(1, int(round(len(shuffled) * val_subject_ratio)))
            val = set(int(x) for x in shuffled[:n_val])
            train = set(int(x) for x in shuffled[n_val:])
            keep = train if part == "train" else val
            self.indices = np.where((split == 0) & np.isin(subjects, list(keep)))[0].astype(np.int64)
        self._raw = None
        self._rhy = None
        self._y = None

    def __len__(self): return len(self.indices)

    def _ensure(self):
        if self._raw is None: self._raw = np.load(self.xraw_path, mmap_mode="r")
        if self._rhy is None: self._rhy = np.load(self.xrhy_path, mmap_mode="r")
        if self._y is None: self._y = np.load(self.y_path, mmap_mode="r")
        return self._raw, self._rhy, self._y

    def __getitem__(self, idx):
        raw, rhy, y = self._ensure()
        i = int(self.indices[idx])
        return {"raw": torch.from_numpy(np.asarray(raw[i], dtype=np.float32)), "rhythm": torch.from_numpy(np.asarray(rhy[i], dtype=np.float32)), "label": torch.tensor(int(y[i]), dtype=torch.long)}


class ProbeModel(nn.Module):
    def __init__(self, ssl: DualSSLTruePinballModel):
        super().__init__()
        self.ssl = ssl
        self.head = nn.Linear(320, 5)
        for p in self.ssl.parameters():
            p.requires_grad = False

    def forward(self, raw, rhythm):
        with torch.no_grad():
            raw_z = self.ssl.encode_raw(raw)
            rhy_z = self.ssl.encode_rhythm(rhythm)
            z = self.ssl.fuse(raw_z, rhy_z)
        return self.head(z)


def parse_args():
    p = argparse.ArgumentParser("True pinball proxy checkpoint selector")
    p.add_argument("--h5", required=True)
    p.add_argument("--raw-memmap-dir", required=True)
    p.add_argument("--rhythm-memmap-dir", required=True)
    p.add_argument("--pretrain-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def ckpts(pretrain_dir: Path):
    rows = []
    for p in pretrain_dir.glob("*.pt"):
        if p.name in {"last.pt", "best_trainloss.pt"} or re.match(r"epoch_\d+\.pt", p.name):
            rows.append(p)
    return sorted(rows, key=lambda x: (0 if x.name.startswith("epoch_") else 1, x.name))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ys = []; ps = []
    for b in loader:
        raw = b["raw"].to(device, non_blocking=True)
        rhy = b["rhythm"].to(device, non_blocking=True)
        y = b["label"].to(device, non_blocking=True)
        logits = model(raw, rhy)
        ys.append(y.cpu().numpy())
        ps.append(logits.argmax(-1).cpu().numpy())
    yt, yp = np.concatenate(ys), np.concatenate(ps)
    m = compute_metrics(yt, yp)
    return {"acc": m.acc, "macro_f1": m.macro_f1, "kappa": m.kappa, "per_class_f1": m.per_class_f1}


def run_one(ckpt_path, args, train_loader, val_loader, device):
    ssl = DualSSLTruePinballModel()
    ck = torch.load(ckpt_path, map_location="cpu")
    ssl.load_state_dict(ck["model"], strict=True)
    model = ProbeModel(ssl).to(device)
    opt = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for _ in range(args.epochs):
        model.train()
        for b in tqdm(train_loader):
            raw = b["raw"].to(device, non_blocking=True)
            rhy = b["rhythm"].to(device, non_blocking=True)
            y = b["label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(raw, rhy), y)
            loss.backward()
            opt.step()
    return evaluate(model, val_loader, device)


def main():
    a = parse_args()
    set_seed(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pdir = Path(a.pretrain_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = ProxyEpochDataset(a.h5, a.raw_memmap_dir, a.rhythm_memmap_dir, a.fold, "train", seed=a.seed)
    va = ProxyEpochDataset(a.h5, a.raw_memmap_dir, a.rhythm_memmap_dir, a.fold, "val", seed=a.seed)
    trl = DataLoader(tr, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers, pin_memory=device.type == "cuda", persistent_workers=a.num_workers > 0, drop_last=True)
    val = DataLoader(va, batch_size=a.batch_size, shuffle=False, num_workers=max(1, a.num_workers // 2), pin_memory=device.type == "cuda", persistent_workers=a.num_workers > 0)
    save_run_config(out, {"stage": "truepinball_proxy_probe_select", **vars(a), "n_proxy_train": len(tr), "n_proxy_val": len(va)})
    rows = []
    for c in ckpts(pdir):
        met = run_one(c, a, trl, val, device)
        row = {"ckpt": str(c), **met}
        rows.append(row)
        print(row, flush=True)
        (out / "proxy_log.jsonl").open("a", encoding="utf-8").write(json.dumps(row, ensure_ascii=False) + "\n")
    best = max(rows, key=lambda r: (r["macro_f1"], r["acc"]))
    (out / "proxy_selection.json").write_text(json.dumps({"best": best, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print({"best_proxy_ckpt": best["ckpt"], "macro_f1": best["macro_f1"], "acc": best["acc"]}, flush=True)


if __name__ == "__main__":
    main()
