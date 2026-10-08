"""模型定义模块。

包含：
- `RhythmEncoder`：把单个 epoch 的 5 路节律编码成 320 维表示
- `SleepBTPretrainModel`：预训练包装器（仅 encoder）
- `SleepBTMambaClassifier`：时序微调模型（双向 Mamba + 节律门控 + 分类头）
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt


def _try_import_mamba():
    try:
        from mamba_ssm import Mamba

        return Mamba
    except Exception:
        return None


class _FallbackSequenceBlock(nn.Module):
    """Fallback used only when mamba_ssm is unavailable in environment."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class RhythmEncoder(nn.Module):
    """Shared rhythm encoder + rhythm-specific dual heads.

    Input: rhythms (B, 5, 3000)
    Output:
    - hidden_per_rhythm: (B, 5, 64)
    - quant_per_rhythm: (B, 5, 5)
    - z_hidden: (B, 320)
    - z_quant: (B, 25)
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.hidden_heads = nn.ModuleList([nn.Linear(64, 64) for _ in range(5)])
        self.quant_heads = nn.ModuleList([nn.Linear(64, 5) for _ in range(5)])

    def forward(self, rhythms: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz = rhythms.size(0)
        x = rhythms.reshape(bsz * 5, 1, 3000)
        feat = self.backbone(x).squeeze(-1)
        feat = feat.reshape(bsz, 5, 64)

        hidden_parts = []
        quant_parts = []
        for r in range(5):
            hidden_parts.append(self.hidden_heads[r](feat[:, r]))
            quant_parts.append(self.quant_heads[r](feat[:, r]))

        hidden_per_rhythm = torch.stack(hidden_parts, dim=1)
        quant_per_rhythm = torch.stack(quant_parts, dim=1)

        return {
            "hidden_per_rhythm": hidden_per_rhythm,
            "quant_per_rhythm": quant_per_rhythm,
            "z_hidden": hidden_per_rhythm.reshape(bsz, 320),
            "z_quant": quant_per_rhythm.reshape(bsz, 25),
        }


class CrossRhythmGate(nn.Module):
    """Sigmoid(mean over 64 dim) rhythm gate.

    Input: hidden_seq (B, T, 5, 64)
    Output:
    - gated_flat: (B, T, 320)
    - gate: (B, T, 5)
    """

    def forward(self, hidden_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(hidden_seq.mean(dim=-1))
        gated = hidden_seq * gate.unsqueeze(-1)
        return gated.reshape(hidden_seq.size(0), hidden_seq.size(1), 320), gate


class SleepBTMambaClassifier(nn.Module):
    """Fine-tuning model.

    Input: seq_rhythms (B, T, 5, 3000)
    Output logits: (B, 5)
    """

    def __init__(
        self,
        encoder: RhythmEncoder,
        d_model: int = 320,
        d_state: int = 8,
        d_conv: int = 4,
        expand: int = 1,
        use_checkpoint: bool = True,
        pool_mode: str = "mean",
        tensor_probe: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.use_checkpoint = use_checkpoint
        self.pool_mode = pool_mode
        self.tensor_probe = tensor_probe

        mamba_cls = _try_import_mamba()
        if mamba_cls is None:
            self.mamba_fwd = _FallbackSequenceBlock(d_model)
            self.mamba_bwd = _FallbackSequenceBlock(d_model)
            self.using_fallback = True
        else:
            self.mamba_fwd = mamba_cls(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                use_fast_path=False,
            )
            self.mamba_bwd = mamba_cls(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                use_fast_path=False,
            )
            self.using_fallback = False

        self.gate = CrossRhythmGate()
        self.cls_head = nn.Linear(d_model, 5)

        if self.pool_mode not in {"mean", "last"}:
            raise ValueError(f"pool_mode must be 'mean' or 'last', got {self.pool_mode}")

    def extract_seq_features(self, seq_rhythms: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, tlen, _, _ = seq_rhythms.shape
        flat = seq_rhythms.reshape(bsz * tlen, 5, 3000)
        enc = self.encoder(flat)
        z_hidden = enc["z_hidden"].reshape(bsz, tlen, 320)
        rhythm_hidden = enc["hidden_per_rhythm"].reshape(bsz, tlen, 5, 64)
        return {"z_hidden": z_hidden, "rhythm_hidden": rhythm_hidden}

    def _bidirectional_mamba(self, x: torch.Tensor) -> torch.Tensor:
        out_fwd = self.mamba_fwd(x)
        out_bwd = self.mamba_bwd(torch.flip(x, dims=[1]))
        out_bwd = torch.flip(out_bwd, dims=[1])
        return out_fwd + out_bwd

    def forward(self, seq_rhythms: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.extract_seq_features(seq_rhythms)
        seq_hidden = feats["z_hidden"]
        rhythm_hidden = feats["rhythm_hidden"]

        if self.training and self.use_checkpoint:
            seq_ctx = ckpt.checkpoint(self._bidirectional_mamba, seq_hidden, use_reentrant=False)
        else:
            seq_ctx = self._bidirectional_mamba(seq_hidden)

        gated_flat, gate = self.gate(rhythm_hidden)
        fused = seq_ctx + gated_flat

        if self.tensor_probe:
            print(
                f"[TensorProbe] fused.shape={tuple(fused.shape)}, "
                f"fused_t_last.shape={tuple(fused[:, -1, :].shape)}"
            )

        if self.pool_mode == "mean":
            pooled = fused.mean(dim=1)
        else:
            pooled = fused[:, -1, :]

        logits = self.cls_head(pooled)

        return {
            "logits": logits,
            "probs": torch.softmax(logits, dim=-1),
            "gate": gate,
            "seq_hidden": seq_hidden,
            "fused": fused,
            "pooled": pooled,
        }

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder_last_block(self) -> None:
        # Last block in shared backbone: Conv(32->64) + BN(64)
        for idx in [6, 7]:
            for p in self.encoder.backbone[idx].parameters():
                p.requires_grad = True


class SleepBTPretrainModel(nn.Module):
    """Pretraining wrapper with shared encoder only."""

    def __init__(self, encoder: RhythmEncoder | None = None) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else RhythmEncoder()

    def forward(self, rhythms: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.encoder(rhythms)
