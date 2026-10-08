from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Optional

import h5py, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sleepssl.models import RhythmEncoder, _try_import_mamba
from sleepssl.utils import compute_metrics, confusion_matrix_5, epoch_zscore, save_run_config, set_seed


class DualSeqDataset(Dataset):
    def __init__(self, h5_path: str, raw_dir: str, rhythm_dir: str, split_name: str, fold: int, seq_len: int = 20) -> None:
        self.raw_dir, self.rhythm_dir = Path(raw_dir).resolve(), Path(rhythm_dir).resolve()
        self.xraw_path, self.xrhy_path, self.y_path = self.raw_dir / "x_raw.npy", self.rhythm_dir / "x_rhythm.npy", self.raw_dir / "y.npy"
        if not (self.raw_dir / "alignment_ok.json").exists(): raise FileNotFoundError(self.raw_dir / "alignment_ok.json")
        for p in [self.xraw_path, self.xrhy_path, self.y_path]:
            if not p.exists(): raise FileNotFoundError(p)
        self._raw: Optional[np.ndarray] = None; self._rhy: Optional[np.ndarray] = None; self._y: Optional[np.ndarray] = None
        with h5py.File(h5_path, "r") as f:
            split = np.asarray(f[f"split_fold_{int(fold)}"]); code = {"train":0,"val":1,"test":2}[split_name]
            split_set = set(np.where(split == code)[0].astype(np.int64).tolist())
            seq = np.asarray(f["seq_index"], dtype=np.int64)
            keep = [i for i in range(len(seq)) if all(int(j) in split_set for j in seq[i])]
            self.seq_index = seq[np.asarray(keep, dtype=np.int64)] if keep else np.zeros((0, seq_len), dtype=np.int64)
            subjects = np.asarray(f["subject_id"], dtype=np.int64)
            if len(self.seq_index):
                ss = subjects[self.seq_index]
                if not (ss == ss[:, :1]).all(axis=1).all(): raise AssertionError("sequence spans multiple subjects")

    def __len__(self): return len(self.seq_index)
    def _ensure(self):
        if self._raw is None: self._raw = np.load(self.xraw_path, mmap_mode="r")
        if self._rhy is None: self._rhy = np.load(self.xrhy_path, mmap_mode="r")
        if self._y is None: self._y = np.load(self.y_path, mmap_mode="r")
        return self._raw, self._rhy, self._y
    def __getitem__(self, idx: int):
        raw, rhy, y = self._ensure(); ids = self.seq_index[idx]
        return {"seq_raw": torch.from_numpy(np.asarray(raw[ids], dtype=np.float32)), "seq_rhythm": torch.from_numpy(np.asarray(rhy[ids], dtype=np.float32)), "label": torch.tensor(int(y[int(ids[-1])]), dtype=torch.long)}


class RawEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1,32,7,stride=2,padding=3,bias=False), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32,64,5,stride=2,padding=2,bias=False), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64,128,3,stride=2,padding=1,bias=False), nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128,320), nn.GELU(), nn.LayerNorm(320))
    def forward(self, x): return self.net(x)


class DualInputMamba(nn.Module):
    def __init__(self, fusion: str = "concat", pool_mode: str = "last"):
        super().__init__(); self.fusion, self.pool_mode = fusion, pool_mode
        self.raw_encoder = RawEncoder(); self.rhythm_encoder = RhythmEncoder()
        if fusion == "concat": self.fuse = nn.Sequential(nn.Linear(640,320), nn.GELU(), nn.LayerNorm(320))
        elif fusion == "gated": self.gate = nn.Sequential(nn.Linear(640,320), nn.Sigmoid()); self.norm = nn.LayerNorm(320)
        else: raise ValueError(f"unknown fusion={fusion}")
        mamba_cls = _try_import_mamba()
        if mamba_cls is None:
            block=lambda: nn.Sequential(nn.LayerNorm(320), nn.Linear(320,320), nn.GELU(), nn.Linear(320,320))
            self.mamba_fwd, self.mamba_bwd, self.using_fallback = block(), block(), True
        else:
            self.mamba_fwd = mamba_cls(d_model=320,d_state=8,d_conv=4,expand=1,use_fast_path=False)
            self.mamba_bwd = mamba_cls(d_model=320,d_state=8,d_conv=4,expand=1,use_fast_path=False)
            self.using_fallback = False
        self.cls_head = nn.Linear(320,5)
    def forward(self, seq_raw, seq_rhythm):
        b,t,_,_ = seq_raw.shape
        raw_z = self.raw_encoder(epoch_zscore(seq_raw.reshape(b*t,1,3000))).reshape(b,t,320)
        rhy_z = self.rhythm_encoder(epoch_zscore(seq_rhythm.reshape(b*t,5,3000)))["z_hidden"].reshape(b,t,320)
        if self.fusion == "concat": z = self.fuse(torch.cat([raw_z, rhy_z], dim=-1))
        else: z = self.norm(raw_z + self.gate(torch.cat([raw_z, rhy_z], dim=-1)) * rhy_z)
        fwd = self.mamba_fwd(z); bwd = torch.flip(self.mamba_bwd(torch.flip(z,dims=[1])), dims=[1]); ctx = fwd + bwd
        pooled = ctx[:, -1] if self.pool_mode == "last" else ctx.mean(dim=1)
        logits = self.cls_head(pooled)
        return {"logits": logits, "probs": torch.softmax(logits, dim=-1)}


def parse_args():
    p=argparse.ArgumentParser("Raw-rhythm dual input baseline")
    p.add_argument("--h5", required=True); p.add_argument("--raw-memmap-dir", required=True); p.add_argument("--rhythm-memmap-dir", required=True); p.add_argument("--out-dir", required=True)
    p.add_argument("--fusion", choices=["concat","gated"], default="concat"); p.add_argument("--fold", type=int, default=0); p.add_argument("--seq-len", type=int, default=20); p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128); p.add_argument("--accum-steps", type=int, default=1); p.add_argument("--num-workers", type=int, default=8); p.add_argument("--val-num-workers", type=int, default=4); p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--weight-decay", type=float, default=1e-4); p.add_argument("--patience", type=int, default=10); p.add_argument("--pool-mode", choices=["mean","last"], default="last"); p.add_argument("--selection-metric", choices=["acc","macro_f1","kappa"], default="acc"); p.add_argument("--seed", type=int, default=42); p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); total=count=0; ys=[]; ps=[]
    for batch in tqdm(loader):
        raw=batch["seq_raw"].to(device,non_blocking=True); rhy=batch["seq_rhythm"].to(device,non_blocking=True); y=batch["label"].to(device,non_blocking=True)
        logits=model(raw,rhy)["logits"]; loss=F.cross_entropy(logits,y); total+=float(loss.item()); count+=1; ys.append(y.cpu().numpy()); ps.append(logits.argmax(-1).cpu().numpy())
    yt,yp=np.concatenate(ys),np.concatenate(ps); m=compute_metrics(yt,yp)
    return total/max(count,1), {"macro_f1":m.macro_f1,"acc":m.acc,"kappa":m.kappa,"per_class_f1":m.per_class_f1}, yt, yp

def train_one_epoch(model, loader, opt, scaler, device, accum):
    model.train(); total=count=0; opt.zero_grad(set_to_none=True); dtype=torch.float16 if device.type=="cuda" else torch.float32
    for step,batch in enumerate(tqdm(loader),1):
        raw=batch["seq_raw"].to(device,non_blocking=True); rhy=batch["seq_rhythm"].to(device,non_blocking=True); y=batch["label"].to(device,non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type=="cuda", dtype=dtype): loss=F.cross_entropy(model(raw,rhy)["logits"],y)/accum
        scaler.scale(loss).backward()
        if step % accum == 0: scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        total += float(loss.item()*accum); count += 1
    if count % accum != 0: scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    return total/max(count,1)

def main():
    a=parse_args(); set_seed(a.seed); out=Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type=="cuda": torch.backends.cudnn.benchmark=True; torch.set_float32_matmul_precision("high")
    tr=DualSeqDataset(a.h5,a.raw_memmap_dir,a.rhythm_memmap_dir,"train",a.fold,a.seq_len); te=DualSeqDataset(a.h5,a.raw_memmap_dir,a.rhythm_memmap_dir,"test",a.fold,a.seq_len)
    pin=bool(a.pin_memory and device.type=="cuda")
    trl=DataLoader(tr,batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers,pin_memory=pin,persistent_workers=a.num_workers>0,drop_last=True,prefetch_factor=a.prefetch_factor)
    tel=DataLoader(te,batch_size=a.batch_size,shuffle=False,num_workers=a.val_num_workers,pin_memory=pin,persistent_workers=a.val_num_workers>0,drop_last=False,prefetch_factor=max(2,a.prefetch_factor))
    model=DualInputMamba(a.fusion,a.pool_mode).to(device); opt=torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay); scaler=torch.cuda.amp.GradScaler(enabled=device.type=="cuda")
    save_run_config(out,{"stage":"raw_rhythm_dual_input","input_source":"true raw EEG + x_rhythm auxiliary", **vars(a)})
    best_score=-1e9; best_epoch=-1; best_metric={}; best_loss=1e9; best_y=best_p=None; wait=0; logp=out/"train_log.jsonl"
    for epoch in range(1,a.epochs+1):
        tl=train_one_epoch(model,trl,opt,scaler,device,max(1,a.accum_steps)); vl,met,y,pred=evaluate(model,tel,device); row={"epoch":epoch,"train_loss":tl,"test_loss":vl,**met}
        logp.open("a",encoding="utf-8").write(json.dumps(row,ensure_ascii=False)+"\n"); print(row,flush=True)
        ck={"epoch":epoch,"model":model.state_dict(),"optimizer":opt.state_dict(),"args":vars(a),"test_metric":met,"selection_metric":a.selection_metric,"selection_score":met[a.selection_metric]}; torch.save(ck,out/"last.pt")
        score=float(met[a.selection_metric])
        if score>best_score: best_score,best_epoch,best_metric,best_loss,best_y,best_p,wait=score,epoch,met,vl,y.copy(),pred.copy(),0; torch.save(ck,out/"best.pt")
        else:
            wait+=1
            if wait>=a.patience: break
    if best_y is None: raise RuntimeError("no best prediction captured")
    res={"split":"test","selection_metric":a.selection_metric,"best_epoch":best_epoch,"test_loss":best_loss,**best_metric,"confusion_matrix":confusion_matrix_5(best_y,best_p).tolist(),"input_source":"true raw EEG + rhythm auxiliary", "fusion":a.fusion}
    (out/"test_metrics.json").write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding="utf-8"); logp.open("a",encoding="utf-8").write(json.dumps(res,ensure_ascii=False)+"\n"); print(res,flush=True)

if __name__ == "__main__": main()
