"""损失函数模块。



包括：

- Grouped Barlow Temporal loss（跨节律去冗余 + 邻域平滑）

- Masked Pinball loss（仅对被遮蔽节律回归分位数）

- 微调阶段 soft-target 分类损失

"""

from __future__ import annotations



from typing import Optional



import torch

import torch.nn as nn

import torch.nn.functional as F





def _build_cross_rhythm_mask(num_rhythms: int = 5, dim_per_rhythm: int = 64) -> torch.Tensor:

    total = num_rhythms * dim_per_rhythm

    mask = torch.zeros(total, total, dtype=torch.bool)

    for ra in range(num_rhythms):

        for rb in range(num_rhythms):

            if ra == rb:

                continue

            a0, a1 = ra * dim_per_rhythm, (ra + 1) * dim_per_rhythm

            b0, b1 = rb * dim_per_rhythm, (rb + 1) * dim_per_rhythm

            mask[a0:a1, b0:b1] = True

    return mask





class GroupedBarlowTemporalLoss(nn.Module):

    """Barlow Twins with cross-rhythm grouping and temporal smoothing.



    Inputs:

    - z_a: (B, 320)

    - z_b: (B, 320)

    - z_prev: optional (B, 320) for C_{t,t-1}

    - z_next: optional (B, 320) for C_{t,t+1}

    """



    def __init__(

        self,

        dim: int = 320,

        lambda_group: float = 0.005,

        lambda_temp: float = 0.001,

        eps: float = 1e-6,

    ) -> None:

        super().__init__()

        self.dim = dim

        self.lambda_group = lambda_group

        self.lambda_temp = lambda_temp

        self.eps = eps

        self.register_buffer("group_mask", _build_cross_rhythm_mask(5, 64), persistent=False)



    def _normalize(self, z: torch.Tensor) -> torch.Tensor:

        mean = z.mean(dim=0, keepdim=True)

        std = z.std(dim=0, keepdim=True, unbiased=False)

        return (z - mean) / (std + self.eps)



    def _corr(self, zx: torch.Tensor, zy: torch.Tensor) -> torch.Tensor:

        n = zx.size(0)

        return (zx.T @ zy) / max(n, 1)



    def forward(

        self,

        z_a: torch.Tensor,

        z_b: torch.Tensor,

        z_prev: Optional[torch.Tensor] = None,

        z_next: Optional[torch.Tensor] = None,

        valid_prev: Optional[torch.Tensor] = None,

        valid_next: Optional[torch.Tensor] = None,

    ) -> torch.Tensor:

        za = self._normalize(z_a)

        zb = self._normalize(z_b)

        c = self._corr(za, zb)



        # Normalize each term by its element count to keep loss scales comparable.

        on_diag = torch.diagonal(c).add(-1.0).pow(2).sum() / c.size(0)

        off_group = c[self.group_mask].pow(2).sum() / self.group_mask.sum()



        temp = torch.zeros((), dtype=c.dtype, device=c.device)



        if z_prev is not None:

            if valid_prev is not None:

                keep_prev = valid_prev.bool()

                if keep_prev.any():

                    za_prev = self._normalize(z_a[keep_prev])

                    zp = self._normalize(z_prev[keep_prev])

                    c_prev = self._corr(za_prev, zp)

                    temp = temp + c_prev[self.group_mask].pow(2).sum() / self.group_mask.sum()

            else:

                zp = self._normalize(z_prev)

                c_prev = self._corr(za, zp)

                temp = temp + c_prev[self.group_mask].pow(2).sum() / self.group_mask.sum()



        if z_next is not None:

            if valid_next is not None:

                keep_next = valid_next.bool()

                if keep_next.any():

                    za_next = self._normalize(z_a[keep_next])

                    zn = self._normalize(z_next[keep_next])

                    c_next = self._corr(za_next, zn)

                    temp = temp + c_next[self.group_mask].pow(2).sum() / self.group_mask.sum()

            else:

                zn = self._normalize(z_next)

                c_next = self._corr(za, zn)

                temp = temp + c_next[self.group_mask].pow(2).sum() / self.group_mask.sum()



        return on_diag + self.lambda_group * off_group + self.lambda_temp * temp





def masked_pinball_loss(

    pred_quant: torch.Tensor,

    target_quant: torch.Tensor,

    masked_rhythm: torch.Tensor,

    taus: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),

) -> torch.Tensor:

    """Pinball loss only over masked rhythms.



    Inputs:

    - pred_quant: (B, 5, 5)

    - target_quant: (B, 5, 5)

    - masked_rhythm: (B, 5) bool or {0,1}; 1 means masked rhythm

    """

    if pred_quant.ndim != 3:

        raise ValueError("pred_quant should have shape (B, 5, 5)")



    tau = pred_quant.new_tensor(taus).view(1, 1, 5)

    u = target_quant - pred_quant

    loss = torch.maximum(tau * u, (tau - 1.0) * u)



    mask = masked_rhythm.to(loss.dtype).unsqueeze(-1)

    masked_count = mask.sum().clamp_min(1.0)

    return (loss * mask).sum() / (masked_count * 5.0)





def soft_target_cross_entropy(logits: torch.Tensor, target_prob: torch.Tensor) -> torch.Tensor:

    logp = F.log_softmax(logits, dim=-1)

    return -(target_prob * logp).sum(dim=-1).mean()





def adaptive_soft_target_loss(

    logits: torch.Tensor,

    target_prob: torch.Tensor,

    hard_labels: torch.Tensor,

    adaptive_class_weights: bool = False,

) -> torch.Tensor:

    """Soft-target cross entropy.



    The `adaptive_class_weights` flag is retained for CLI compatibility; the

    class-weighting/focal branch was removed after ablation showed no gain, so

    the loss reduces to plain soft-target CE regardless of the flag.

    """

    return soft_target_cross_entropy(logits, target_prob)



