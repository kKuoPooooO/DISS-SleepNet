from __future__ import annotations

import argparse, json, hashlib
from pathlib import Path
from typing import Optional

import h5py, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn as nn

from sleepssl.utils import compute_metrics, confusion_matrix_5, save_run_config, set_seed, epoch_zscore
from sleepssl.models import RhythmEncoder
from sleepssl.dual_input import DualSeqDataset, DualInputMamba, RawEncoder


class PinballScorer(nn.Module):
    """Frozen replica of the pretrain fuse+quant_pred path.

    Given the target epoch (raw + 5 rhythms), predicts the 5 quantiles per
    rhythm band and returns the 90-10 dispersion averaged over the 5 bands.
    A smaller dispersion means a sharper/more typical rhythm distribution,
    which we treat as higher confidence for soft-label correction.
    """

    TAUS = (0.1, 0.25, 0.5, 0.75, 0.9)

    def __init__(self) -> None:
        super().__init__()
        self.raw_encoder = RawEncoder()
        self.rhythm_encoder = RhythmEncoder()
        self.gate = nn.Sequential(nn.Linear(640, 320), nn.Sigmoid())
        self.norm = nn.LayerNorm(320)
        self.quant_pred = nn.Sequential(
            nn.Linear(320, 320), nn.GELU(), nn.LayerNorm(320), nn.Linear(320, 25)
        )

    @torch.no_grad()
    def forward(self, raw_last: torch.Tensor, rhy_last: torch.Tensor) -> torch.Tensor:
        b = raw_last.size(0)
        raw_z = self.raw_encoder(epoch_zscore(raw_last.reshape(b, 1, 3000)))
        rhythm_z = self.rhythm_encoder(epoch_zscore(rhy_last.reshape(b, 5, 3000)))["z_hidden"]
        fused = self.norm(raw_z + self.gate(torch.cat([raw_z, rhythm_z], dim=-1)) * rhythm_z)
        pred_quant = self.quant_pred(fused).reshape(b, 5, 5)
        # 90-10 dispersion per band (tau index 4 = 0.9, index 0 = 0.1), averaged over 5 bands
        disp = (pred_quant[:, :, 4] - pred_quant[:, :, 0]).mean(dim=1)
        return disp


def _subject_label_keep(h5_path: str, fold: int, label_ratio: float, seed: int) -> set[int] | None:
    if label_ratio >= 0.999: return None
    with h5py.File(h5_path, "r") as f:
        split = np.asarray(f[f"split_fold_{fold}"]); subjects = np.asarray(f["subject_id"], dtype=np.int64)
        train_subjects = np.unique(subjects[split == 0])
    rng = np.random.default_rng(seed + 1009 * fold + int(round(label_ratio * 1000)))
    n_keep = max(1, int(round(len(train_subjects) * label_ratio)))
    keep = rng.choice(train_subjects, size=n_keep, replace=False)
    return set(int(x) for x in keep.tolist())


class Batch9DualSeqDataset(DualSeqDataset):
    def __init__(self, h5_path: str, raw_dir: str, rhythm_dir: str, split_name: str, fold: int, seq_len: int = 20, label_ratio: float = 1.0, seed: int = 42, input_mode: str = "normal") -> None:
        self.input_mode = input_mode
        super().__init__(h5_path, raw_dir, rhythm_dir, split_name, fold, seq_len)
        self.h5_path = h5_path; self.fold = fold; self.split_name = split_name
        self._shuffle_perm: Optional[np.ndarray] = None
        with h5py.File(h5_path, "r") as f:
            subjects = np.asarray(f["subject_id"], dtype=np.int64)
            if split_name == "train" and label_ratio < 0.999:
                keep_subjects = _subject_label_keep(h5_path, fold, label_ratio, seed)
                keep_rows = [i for i, seq in enumerate(self.seq_index) if int(subjects[int(seq[-1])]) in keep_subjects]
                self.seq_index = self.seq_index[np.asarray(keep_rows, dtype=np.int64)] if keep_rows else self.seq_index[:0]
        if input_mode == "shuffle_rhythm":
            rng = np.random.default_rng(seed + 7919 * fold + (0 if split_name == "train" else 12345))
            ids = np.asarray([int(s[-1]) for s in self.seq_index], dtype=np.int64)
            perm_ids = ids.copy(); rng.shuffle(perm_ids); self._shuffle_map = dict(zip(ids.tolist(), perm_ids.tolist()))
        else:
            self._shuffle_map = {}
    def __getitem__(self, idx: int):
        raw, rhy, y = self._ensure(); ids = self.seq_index[idx]
        seq_raw = np.asarray(raw[ids], dtype=np.float32)
        if self.input_mode == "zero_rhythm":
            seq_rhy = np.zeros((len(ids), 5, 3000), dtype=np.float32)
        elif self.input_mode == "rawraw_control":
            seq_rhy = np.repeat(seq_raw, 5, axis=1).astype(np.float32)
        elif self.input_mode == "shuffle_rhythm":
            mapped = np.asarray([self._shuffle_map.get(int(i), int(i)) for i in ids], dtype=np.int64)
            seq_rhy = np.asarray(rhy[mapped], dtype=np.float32)
        else:
            seq_rhy = np.asarray(rhy[ids], dtype=np.float32)
        return {"seq_raw": torch.from_numpy(seq_raw), "seq_rhythm": torch.from_numpy(seq_rhy), "label": torch.tensor(int(y[int(ids[-1])]), dtype=torch.long), "epoch_id": torch.tensor(int(ids[-1]), dtype=torch.long)}


def parse_args():
    p = argparse.ArgumentParser("Batch9 dual finetune")
    p.add_argument("--h5", required=True); p.add_argument("--raw-memmap-dir", required=True); p.add_argument("--rhythm-memmap-dir", required=True); p.add_argument("--out-dir", required=True)
    p.add_argument("--fold", type=int, default=0); p.add_argument("--label-ratio", type=float, default=1.0); p.add_argument("--label-target-ids", default="", help="Path to .npy of epoch IDs; overrides label-ratio when provided."); p.add_argument("--input-mode", choices=["normal", "zero_rhythm", "shuffle_rhythm", "rawraw_control"], default="normal")
    p.add_argument("--pretrained-ckpt", default=""); p.add_argument("--soft-teacher", default=""); p.add_argument("--soft-mode", choices=["none", "late_conf", "pinball_disp"], default="none"); p.add_argument("--soft-warmup", type=int, default=5); p.add_argument("--soft-threshold", type=float, default=0.8); p.add_argument("--soft-weight", type=float, default=0.1); p.add_argument("--soft-temp", type=float, default=2.0)
    p.add_argument("--pinball-disp-quantile", type=float, default=0.30)
    p.add_argument("--seq-len", type=int, default=20); p.add_argument("--epochs", type=int, default=80); p.add_argument("--batch-size", type=int, default=96); p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=8); p.add_argument("--val-num-workers", type=int, default=4); p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--weight-decay", type=float, default=1e-4); p.add_argument("--patience", type=int, default=10); p.add_argument("--selection-metric", choices=["acc", "macro_f1", "kappa"], default="acc"); p.add_argument("--seed", type=int, default=42); p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def load_pretrained(model: DualInputMamba, ckpt_path: str) -> dict:
    if not ckpt_path: return {"loaded": False}
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck.get("model", ck)
    own = model.state_dict(); loadable = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("raw_encoder.") or nk.startswith("rhythm_encoder.") or nk.startswith("gate.") or nk.startswith("norm."):
            if nk in own and own[nk].shape == v.shape: loadable[nk] = v
    missing, unexpected = model.load_state_dict(loadable, strict=False)
    return {"loaded": True, "ckpt": ckpt_path, "n_loaded": len(loadable), "missing": len(missing), "unexpected": len(unexpected)}


def build_pinball_scorer(ckpt_path: str, device) -> tuple[PinballScorer | None, dict]:
    """Build a frozen PinballScorer from the SSL checkpoint.

    Loads raw_encoder / rhythm_encoder / gate / norm / quant_pred so the
    trained pinball quantile head is actually restored (it is dropped by the
    main finetune loader).
    """
    if not ckpt_path: return None, {"loaded": False}
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck.get("model", ck)
    scorer = PinballScorer()
    own = scorer.state_dict(); loadable = {}
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            loadable[k] = v
    missing, unexpected = scorer.load_state_dict(loadable, strict=False)
    has_quant = any(k.startswith("quant_pred.") for k in loadable)
    scorer.eval()
    for p in scorer.parameters(): p.requires_grad = False
    scorer.to(device)
    return scorer, {"loaded": True, "ckpt": ckpt_path, "n_loaded": len(loadable), "missing": len(missing), "unexpected": len(unexpected), "has_quant_pred": has_quant}


@torch.no_grad()
def compute_disp_threshold(scorer: PinballScorer, loader, device, quantile: float) -> float:
    """Precompute the dispersion cutoff over the training set.

    Samples with dispersion below this cutoff (the sharpest `quantile`
    fraction) are considered confident enough for soft-label correction.
    """
    disps = []
    for batch in tqdm(loader):
        raw = batch["seq_raw"].to(device, non_blocking=True)
        rhy = batch["seq_rhythm"].to(device, non_blocking=True)
        disps.append(scorer(raw[:, -1], rhy[:, -1]).float().cpu())
    alld = torch.cat(disps)
    return float(torch.quantile(alld, quantile).item())


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval(); total=count=0; ys=[]; ps=[]
    for batch in tqdm(loader):
        raw=batch["seq_raw"].to(device,non_blocking=True); rhy=batch["seq_rhythm"].to(device,non_blocking=True); y=batch["label"].to(device,non_blocking=True)
        logits=model(raw,rhy)["logits"]; loss=F.cross_entropy(logits,y); total+=float(loss.item()); count+=1; ys.append(y.cpu().numpy()); ps.append(logits.argmax(-1).cpu().numpy())
    yt, yp = np.concatenate(ys), np.concatenate(ps); m = compute_metrics(yt, yp)
    return total/max(count,1), {"macro_f1":m.macro_f1,"acc":m.acc,"kappa":m.kappa,"per_class_f1":m.per_class_f1}, yt, yp


@torch.no_grad()
def teacher_soft(teacher, raw, rhy, temp: float):
    logits = teacher(raw, rhy)["logits"] / temp
    prob = torch.softmax(logits, dim=-1)
    conf = prob.max(dim=-1).values
    return prob, conf


def train_epoch(model, loader, opt, scaler, device, accum, epoch, teacher=None, args=None, scorer=None, disp_thr=None):
    model.train(); total=count=0; opt.zero_grad(set_to_none=True); dtype=torch.float16 if device.type=="cuda" else torch.float32
    for step,batch in enumerate(tqdm(loader),1):
        raw=batch["seq_raw"].to(device,non_blocking=True); rhy=batch["seq_rhythm"].to(device,non_blocking=True); y=batch["label"].to(device,non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type=="cuda", dtype=dtype):
            logits=model(raw,rhy)["logits"]; loss=F.cross_entropy(logits,y)
            soft_on = teacher is not None and args.soft_mode != "none" and epoch > args.soft_warmup
            if soft_on:
                prob, conf = teacher_soft(teacher, raw, rhy, args.soft_temp)
                if args.soft_mode == "late_conf":
                    keep = conf >= args.soft_threshold
                elif args.soft_mode == "pinball_disp" and scorer is not None and disp_thr is not None:
                    with torch.autocast(device_type=device.type, enabled=False):
                        disp = scorer(raw[:, -1].float(), rhy[:, -1].float())
                    keep = disp <= disp_thr
                else:
                    keep = torch.zeros_like(conf, dtype=torch.bool)
                if keep.any():
                    logp = F.log_softmax(logits[keep] / args.soft_temp, dim=-1)
                    kl = F.kl_div(logp, prob[keep], reduction="batchmean") * (args.soft_temp ** 2)
                    loss = loss + args.soft_weight * kl
            loss = loss / accum
        scaler.scale(loss).backward()
        if step % accum == 0:
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        total += float(loss.item()*accum); count += 1
    if count % accum != 0: scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    return total/max(count,1)


def main():
    a=parse_args(); set_seed(a.seed); out=Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type=="cuda": torch.backends.cudnn.benchmark=True; torch.set_float32_matmul_precision("high")
    _ds_ratio = 1.0 if getattr(a, "label_target_ids", "") else a.label_ratio
    tr=Batch9DualSeqDataset(a.h5,a.raw_memmap_dir,a.rhythm_memmap_dir,"train",a.fold,a.seq_len,_ds_ratio,a.seed,a.input_mode)
    if getattr(a, "label_target_ids", ""):
        import numpy as _np, hashlib as _hl
        _selected = _np.load(a.label_target_ids).astype(_np.int64)
        _sel_set  = set(_selected.tolist())
        _keep     = _np.asarray([i for i, seq in enumerate(tr.seq_index) if int(seq[-1]) in _sel_set], dtype=_np.int64)
        tr.seq_index = tr.seq_index[_keep] if len(_keep) else tr.seq_index[:0]
    te=Batch9DualSeqDataset(a.h5,a.raw_memmap_dir,a.rhythm_memmap_dir,"test",a.fold,a.seq_len,1.0,a.seed,a.input_mode)
    pin=bool(a.pin_memory and device.type=="cuda")
    trl=DataLoader(tr,batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers,pin_memory=pin,persistent_workers=a.num_workers>0,drop_last=True,prefetch_factor=a.prefetch_factor)
    tel=DataLoader(te,batch_size=a.batch_size,shuffle=False,num_workers=a.val_num_workers,pin_memory=pin,persistent_workers=a.val_num_workers>0,drop_last=False,prefetch_factor=max(2,a.prefetch_factor))
    model=DualInputMamba("gated","last").to(device); load_info=load_pretrained(model,a.pretrained_ckpt)
    teacher=None
    if a.soft_teacher:
        teacher=DualInputMamba("gated","last").to(device); teacher.load_state_dict(torch.load(a.soft_teacher,map_location="cpu")["model"], strict=True); teacher.eval()
    scorer=None; scorer_info={"loaded": False}; disp_thr=None
    if a.soft_mode == "pinball_disp":
        scorer, scorer_info = build_pinball_scorer(a.pretrained_ckpt, device)
        if scorer is not None:
            disp_thr = compute_disp_threshold(scorer, trl, device, a.pinball_disp_quantile)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay); scaler=torch.cuda.amp.GradScaler(enabled=device.type=="cuda")
    save_run_config(out,{"stage":"batch9_dual_finetune","train_sequences":len(tr),"test_sequences":len(te),"load_info":load_info,"scorer_info":scorer_info,"disp_threshold":disp_thr,**vars(a)})
    best_score=-1e9; best_epoch=-1; best_metric={}; best_loss=1e9; best_y=best_p=None; wait=0; logp=out/"train_log.jsonl"
    for epoch in range(1,a.epochs+1):
        tl=train_epoch(model,trl,opt,scaler,device,max(1,a.accum_steps),epoch,teacher,a,scorer,disp_thr); vl,met,y,pred=eval_model(model,tel,device); row={"epoch":epoch,"train_loss":tl,"test_loss":vl,**met}
        logp.open("a",encoding="utf-8").write(json.dumps(row,ensure_ascii=False)+"\n"); print(row,flush=True)
        ck={"epoch":epoch,"model":model.state_dict(),"optimizer":opt.state_dict(),"args":vars(a),"test_metric":met,"selection_metric":a.selection_metric,"selection_score":met[a.selection_metric]}; torch.save(ck,out/"last.pt")
        score=float(met[a.selection_metric])
        if score>best_score:
            best_score,best_epoch,best_metric,best_loss,best_y,best_p,wait=score,epoch,met,vl,y.copy(),pred.copy(),0; torch.save(ck,out/"best.pt")
        else:
            wait+=1
            if wait>=a.patience: break
    if best_y is None: raise RuntimeError("no best prediction captured")
    res={"split":"test","selection_metric":a.selection_metric,"best_epoch":best_epoch,"test_loss":best_loss,**best_metric,"confusion_matrix":confusion_matrix_5(best_y,best_p).tolist(),"input_mode":a.input_mode,"label_ratio":a.label_ratio,"pretrained_ckpt":a.pretrained_ckpt,"soft_mode":a.soft_mode}
    (out/"test_metrics.json").write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding="utf-8"); logp.open("a",encoding="utf-8").write(json.dumps(res,ensure_ascii=False)+"\n"); print(res,flush=True)

if __name__ == "__main__": main()
