"""通用工具模块。

包含：随机种子、按 epoch 的 z-score、评估指标、混淆矩阵、切分校验、
以及规范化输出目录/配置记录等。这些函数本身不训练模型，但决定了实验
是否可复现、评估是否可信。
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def epoch_zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-epoch z-score. 输入形如 (..., 3000)，沿最后一维归一化。"""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / (std + eps)


@dataclass
class EvalResult:
    macro_f1: float
    acc: float
    kappa: float
    per_class_f1: list[float]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> EvalResult:
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)
    return EvalResult(macro_f1=float(macro), acc=float(acc), kappa=float(kappa), per_class_f1=per_class.tolist())


def verify_subject_independent_split(subject_ids: np.ndarray, split_codes: np.ndarray) -> Dict[str, int]:
    """校验同一受试者不会出现在多个 split。

    split 编码: 0=train, 2=test（本项目无独立 val；保留 1=val 的兼容校验）。
    """
    subject_ids = np.asarray(subject_ids)
    split_codes = np.asarray(split_codes)

    if subject_ids.shape != split_codes.shape:
        raise ValueError(
            f"subject_ids and split_codes must have same shape, got {subject_ids.shape} vs {split_codes.shape}"
        )
    if subject_ids.ndim != 1:
        raise ValueError(f"expected 1D arrays, got ndim={subject_ids.ndim}")

    allowed = {0, 1, 2}
    observed = set(np.unique(split_codes).tolist())
    if not observed.issubset(allowed):
        raise ValueError(f"split_codes contains invalid values: {sorted(observed - allowed)}")

    train_sub = set(np.unique(subject_ids[split_codes == 0]).tolist())
    val_sub = set(np.unique(subject_ids[split_codes == 1]).tolist())
    test_sub = set(np.unique(subject_ids[split_codes == 2]).tolist())

    overlap_train_val = train_sub & val_sub
    overlap_train_test = train_sub & test_sub
    overlap_val_test = val_sub & test_sub

    if overlap_train_val:
        raise AssertionError(f"Data leakage: train/val overlap subjects={sorted(overlap_train_val)}")
    if overlap_train_test:
        raise AssertionError(f"Data leakage: train/test overlap subjects={sorted(overlap_train_test)}")
    if overlap_val_test:
        raise AssertionError(f"Data leakage: val/test overlap subjects={sorted(overlap_val_test)}")

    return {
        "train_subjects": len(train_sub),
        "val_subjects": len(val_sub),
        "test_subjects": len(test_sub),
        "total_subjects": len(np.unique(subject_ids)),
    }


def build_soft_targets(hard_labels: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    return torch.nn.functional.one_hot(hard_labels, num_classes=num_classes).float()


def confusion_matrix_5(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> np.ndarray:
    """5 类混淆矩阵（计数版）。行=真实标签，列=预测标签。"""
    y_true = np.asarray(y_true).astype(np.int64).ravel()
    y_pred = np.asarray(y_pred).astype(np.int64).ravel()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def topk_iqr_indices(iqr_scores: torch.Tensor, ratio: float = 0.1) -> torch.Tensor:
    k = max(1, int(round(iqr_scores.numel() * ratio)))
    return torch.topk(iqr_scores, k=k, largest=True).indices


def build_run_dir(base: str, stage: str, dataset: str, tag: str = "") -> Path:
    """生成规范命名的输出目录并返回其路径。

    命名格式: <base>/<stage>/<YYYYMMDD-HHMMSS>__<dataset>[__<tag>]
    - stage:   pretrain / finetune / linear_probe / gridsearch
    - dataset: 如 sleepedf20
    - tag:     关键参数串, 如 fold0_seq10_label0.10_lr1e-5
    旧目录一律不受影响; 时间戳保证唯一、可排序。
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{ts}__{dataset}" + (f"__{tag}" if tag else "")
    run_dir = Path(base) / stage / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_config(run_dir: Path, config: dict) -> None:
    """把关键参数写入 run_dir/run_config.json, 便于审查与复现。"""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"saved_at": datetime.now().isoformat(timespec="seconds"), **config}
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2, default=str)
