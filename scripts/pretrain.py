from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import Optional

import h5py, numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sleepssl.losses import GroupedBarlowTemporalLoss, masked_pinball_loss
from sleepssl.models import RhythmEncoder
from sleepssl.utils import epoch_zscore, save_run_config, set_seed
from sleepssl.dual_input import RawEncoder


class DualEpochDataset(Dataset):
    def __init__(
        self, h5_path: str, raw_dir: str, rhythm_dir: str, fold: int,
        split_name: str = "train", temporal_window: int = 1,
    ) -> None:
        if temporal_window < 1 or temporal_window % 2 == 0:
            raise ValueError("temporal_window must be a positive odd integer")
        self.temporal_window = temporal_window
        self.raw_dir, self.rhythm_dir = Path(raw_dir).resolve(), Path(rhythm_dir).resolve()
        self.xraw_path, self.xrhy_path = self.raw_dir / "x_raw.npy", self.rhythm_dir / "x_rhythm.npy"
        if not (self.raw_dir / "alignment_ok.json").exists():
            raise FileNotFoundError(self.raw_dir / "alignment_ok.json")
        for p in [self.xraw_path, self.xrhy_path]:
            if not p.exists():
                raise FileNotFoundError(p)
        with h5py.File(h5_path, "r") as f:
            split = np.asarray(f[f"split_fold_{int(fold)}"])
            subjects = np.asarray(f["subject_id"], dtype=np.int64)
            records = np.asarray(f["record_id"], dtype=np.int64)
            code = {"train": 0, "val": 1, "test": 2}[split_name]
        if temporal_window == 1:
            self.windows = np.where(split == code)[0].astype(np.int64)[:, None]
        else:
            radius = temporal_window // 2
            centers = np.where(split == code)[0].astype(np.int64)
            centers = centers[(centers >= radius) & (centers < len(split) - radius)]
            offsets = np.arange(-radius, radius + 1, dtype=np.int64)
            windows = centers[:, None] + offsets[None, :]
            valid = np.all(split[windows] == code, axis=1)
            valid &= np.all(subjects[windows] == subjects[windows[:, :1]], axis=1)
            valid &= np.all(records[windows] == records[windows[:, :1]], axis=1)
            self.windows = windows[valid]
        self.indices = self.windows[:, temporal_window // 2]
        self._raw: Optional[np.ndarray] = None
        self._rhy: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.indices)

    def _ensure(self):
        if self._raw is None:
            self._raw = np.load(self.xraw_path, mmap_mode="r")
        if self._rhy is None:
            self._rhy = np.load(self.xrhy_path, mmap_mode="r")
        return self._raw, self._rhy

    def __getitem__(self, idx: int):
        raw, rhy = self._ensure()
        window = self.windows[idx]
        center = self.temporal_window // 2
        item = {
            "raw": torch.from_numpy(np.asarray(raw[int(window[center])], dtype=np.float32)),
            "rhythm": torch.from_numpy(np.asarray(rhy[int(window[center])], dtype=np.float32)),
        }
        if self.temporal_window > 1:
            item["raw_temporal"] = torch.from_numpy(np.asarray(raw[window], dtype=np.float32))
        return item


def temporal_group_loss(
    center: torch.Tensor,
    neighbors: torch.Tensor,
    group_mask: torch.Tensor,
    lambda_group: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Average grouped Barlow loss between the center and each neighbor."""
    if neighbors.ndim != 3 or neighbors.shape[0] != center.shape[0]:
        raise ValueError("neighbors must have shape (B, K, D)")
    center_norm = (center - center.mean(0, keepdim=True)) / (
        center.std(0, keepdim=True, unbiased=False) + eps
    )
    terms = []
    for offset in range(neighbors.shape[1]):
        neighbor = neighbors[:, offset]
        neighbor_norm = (neighbor - neighbor.mean(0, keepdim=True)) / (
            neighbor.std(0, keepdim=True, unbiased=False) + eps
        )
        corr = center_norm.T @ neighbor_norm / max(center.shape[0], 1)
        on_diag = torch.diagonal(corr).sub(1.0).pow(2).mean()
        off_group = corr[group_mask].pow(2).mean()
        terms.append(on_diag + lambda_group * off_group)
    return torch.stack(terms).mean() if terms else center.new_zeros(())


class DualSSLTruePinballModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.raw_encoder = RawEncoder()
        self.rhythm_encoder = RhythmEncoder()
        self.gate = nn.Sequential(nn.Linear(640, 320), nn.Sigmoid())
        self.norm = nn.LayerNorm(320)
        self.quant_pred = nn.Sequential(
            nn.Linear(320, 320), nn.GELU(), nn.LayerNorm(320), nn.Linear(320, 25)
        )
        self.log_sigma_b = nn.Parameter(torch.zeros(()))
        self.log_sigma_p = nn.Parameter(torch.zeros(()))

    def encode_raw(self, raw: torch.Tensor) -> torch.Tensor:
        return self.raw_encoder(epoch_zscore(raw.reshape(raw.size(0), 1, 3000)))

    def encode_rhythm_all(self, rhythm: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.rhythm_encoder(epoch_zscore(rhythm.reshape(rhythm.size(0), 5, 3000)))

    def encode_rhythm(self, rhythm: torch.Tensor) -> torch.Tensor:
        return self.encode_rhythm_all(rhythm)["z_hidden"]

    def fuse(self, raw_z: torch.Tensor, rhythm_z: torch.Tensor) -> torch.Tensor:
        return self.norm(raw_z + self.gate(torch.cat([raw_z, rhythm_z], dim=-1)) * rhythm_z)

    def forward(self, raw: torch.Tensor, rhythm: torch.Tensor, mask_ratio: float = 0.4):
        raw_z = self.encode_raw(raw)
        rhythm_full = self.encode_rhythm_all(rhythm)
        rhythm_z = rhythm_full["z_hidden"]
        masked = rhythm.clone()
        mask = torch.rand(masked.size(0), 5, device=masked.device) < mask_ratio
        if not mask.any():
            bi = torch.randint(0, masked.size(0), (1,), device=masked.device)
            ri = torch.randint(0, 5, (1,), device=masked.device)
            mask[bi, ri] = True
        masked[mask.unsqueeze(-1).expand_as(masked)] = 0.0
        masked_z = self.encode_rhythm(masked)
        fused_masked = self.fuse(raw_z, masked_z)
        pred_quant = self.quant_pred(fused_masked).reshape(raw.size(0), 5, 5)
        return {
            "raw_z": raw_z,
            "rhythm_z": rhythm_z,
            "pred_quant": pred_quant,
            "mask": mask,
        }


def parse_args():
    p = argparse.ArgumentParser("Dual raw-rhythm true quantile pinball SSL pretraining")
    p.add_argument("--h5", required=True)
    p.add_argument("--raw-memmap-dir", required=True)
    p.add_argument("--rhythm-memmap-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--method", choices=["barlow_only", "true_pinball", "barlow_truepinball", "barlow_truepinball_adaptive"], required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--mask-ratio", type=float, default=0.4)
    p.add_argument("--pinball-weight", type=float, default=0.5)
    p.add_argument("--lambda-group", type=float, default=0.005)
    p.add_argument("--lambda-temp", type=float, default=0.0)
    p.add_argument("--temporal-window", type=int, default=1)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    a = parse_args()
    set_seed(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    temporal_on = a.temporal_window > 1 and a.lambda_temp > 0
    effective_window = a.temporal_window if temporal_on else 1
    ds = DualEpochDataset(
        a.h5, a.raw_memmap_dir, a.rhythm_memmap_dir, a.fold, "train",
        temporal_window=effective_window,
    )
    pin = bool(a.pin_memory and device.type == "cuda")
    dl = DataLoader(
        ds,
        batch_size=a.batch_size,
        shuffle=True,
        num_workers=a.num_workers,
        pin_memory=pin,
        persistent_workers=a.num_workers > 0,
        drop_last=True,
        prefetch_factor=a.prefetch_factor,
    )
    model = DualSSLTruePinballModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    barlow = GroupedBarlowTemporalLoss(lambda_group=a.lambda_group, lambda_temp=a.lambda_temp).to(device)
    taus = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=device)
    save_run_config(out, {"stage": "dual_ssl_truepinball_pretrain", **vars(a), "n_train_epochs": len(ds)})
    best_loss = math.inf
    logp = out / "pretrain_log.jsonl"
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    for epoch in range(1, a.epochs + 1):
        model.train()
        total = btot = ptot = ttot = 0.0
        count = 0
        for batch in tqdm(dl):
            raw = batch["raw"].to(device, non_blocking=True)
            rhythm = batch["rhythm"].to(device, non_blocking=True)
            raw_temporal = batch.get("raw_temporal")
            if raw_temporal is not None:
                raw_temporal = raw_temporal.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                rhythm_norm = epoch_zscore(rhythm.reshape(rhythm.size(0), 5, 3000))
                target_quant = torch.quantile(rhythm_norm, q=taus, dim=-1).permute(1, 2, 0).contiguous()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=dtype):
                o = model(raw, rhythm, a.mask_ratio)
                use_b = a.method in {"barlow_only", "barlow_truepinball", "barlow_truepinball_adaptive"}
                use_p = a.method in {"true_pinball", "barlow_truepinball", "barlow_truepinball_adaptive"}
                loss_b_base = barlow(o["raw_z"], o["rhythm_z"]) if use_b else raw.new_zeros(())
                loss_temp = raw.new_zeros(())
                if use_b and temporal_on and raw_temporal is not None:
                    batch_size, window_size = raw_temporal.shape[:2]
                    center_idx = window_size // 2
                    neighbor_raw = torch.cat(
                        [raw_temporal[:, :center_idx], raw_temporal[:, center_idx + 1:]], dim=1
                    )
                    neighbor_z = model.encode_raw(neighbor_raw.reshape(batch_size * (window_size - 1), 1, 3000))
                    neighbor_z = neighbor_z.reshape(batch_size, window_size - 1, -1)
                    loss_temp = temporal_group_loss(
                        o["raw_z"], neighbor_z, barlow.group_mask, a.lambda_group
                    )
                loss_b = loss_b_base + a.lambda_temp * loss_temp
                loss_p = masked_pinball_loss(o["pred_quant"], target_quant, o["mask"]) if use_p else raw.new_zeros(())
                if a.method == "barlow_truepinball_adaptive":
                    loss = (
                        torch.exp(-model.log_sigma_b) * loss_b + model.log_sigma_b
                        + torch.exp(-model.log_sigma_p) * loss_p + model.log_sigma_p
                    )
                else:
                    loss = loss_b + a.pinball_weight * loss_p
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss.item())
            btot += float(loss_b.item())
            ptot += float(loss_p.item())
            ttot += float(loss_temp.item())
            count += 1
        row = {
            "epoch": epoch,
            "loss": total / max(count, 1),
            "barlow_loss": btot / max(count, 1),
            "true_pinball_loss": ptot / max(count, 1),
            "temporal_loss": ttot / max(count, 1),
            "temporal_enabled": temporal_on,
            "temporal_window": effective_window,
            "lambda_temp": a.lambda_temp,
            "method": a.method,
            "pinball_weight": a.pinball_weight,
        }
        if a.method == "barlow_truepinball_adaptive":
            row["w_barlow"] = float(torch.exp(-model.log_sigma_b).item())
            row["w_pinball"] = float(torch.exp(-model.log_sigma_p).item())
        logp.open("a", encoding="utf-8").write(json.dumps(row, ensure_ascii=False) + "\n")
        print(row, flush=True)
        ck = {"epoch": epoch, "model": model.state_dict(), "args": vars(a), "loss": row["loss"]}
        torch.save(ck, out / "last.pt")
        if epoch % max(1, a.save_every) == 0:
            torch.save(ck, out / f"epoch_{epoch}.pt")
        if row["loss"] < best_loss:
            best_loss = row["loss"]
            torch.save(ck, out / "best_trainloss.pt")
    (out / "pretrain_metrics.json").write_text(
        json.dumps({"best_trainloss": best_loss, "last_epoch": a.epochs, "method": a.method, "pinball_weight": a.pinball_weight}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
