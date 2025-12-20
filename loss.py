# loss.py
import torch
from itertools import permutations

EPS = 1e-8


def si_snr(est: torch.Tensor, ref: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    Scale-Invariant SNR (SI-SNR).
    est, ref: [B, T]
    returns:  [B]
    """
    if est.shape != ref.shape:
        raise RuntimeError(f"SI-SNR shape mismatch: {est.shape} vs {ref.shape}")

    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)

    ref_energy = (ref * ref).sum(dim=-1, keepdim=True) + eps
    proj = (est * ref).sum(dim=-1, keepdim=True) * ref / ref_energy
    noise = est - proj

    ratio = (proj * proj).sum(dim=-1) / ((noise * noise).sum(dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)


class PITSiSNRLoss:
    """
    Permutation Invariant Training loss using SI-SNR.
    Supports:
      - ests/refs as Tensor: [B, C, T]
      - ests/refs as list of length C, each [B, T]
    Returns negative mean SI-SNR (so minimizing increases SI-SNR).
    """

    def __init__(self):
        super().__init__()

    def __call__(self, ests, refs) -> torch.Tensor:
        # Normalize inputs to [B, C, T]
        if isinstance(ests, (list, tuple)):
            est = torch.stack(ests, dim=1)  # [B, C, T]
        else:
            est = ests

        if isinstance(refs, (list, tuple)):
            ref = torch.stack(refs, dim=1)  # [B, C, T]
        else:
            ref = refs

        if est.ndim != 3 or ref.ndim != 3:
            raise RuntimeError(f"Expected [B,C,T], got est={est.shape}, ref={ref.shape}")
        if est.shape != ref.shape:
            raise RuntimeError(f"Shape mismatch est={est.shape} ref={ref.shape}")

        B, C, T = est.shape

        # Fast path for C=2 (common)
        if C == 2:
            s_id = si_snr(est[:, 0, :], ref[:, 0, :]) + si_snr(est[:, 1, :], ref[:, 1, :])
            s_sw = si_snr(est[:, 0, :], ref[:, 1, :]) + si_snr(est[:, 1, :], ref[:, 0, :])
            best = torch.maximum(s_id, s_sw)  # [B]
            return -best.mean()

        # Generic path for C>2
        perms = list(permutations(range(C)))
        scores = []
        for p in perms:
            s = 0.0
            for i, j in enumerate(p):
                s = s + si_snr(est[:, i, :], ref[:, j, :])
            scores.append(s)  # [B]
        scores = torch.stack(scores, dim=0)  # [P, B]
        best, _ = scores.max(dim=0)          # [B]
        return -best.mean()