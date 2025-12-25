# metrics.py
import torch
import torch.nn as nn

from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore

from itertools import permutations

EPS = 1e-8

def si_snr(est: torch.Tensor, ref: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    # est, ref: [B,T] -> [B]
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    ref_energy = (ref * ref).sum(dim=-1, keepdim=True) + eps
    proj = (est * ref).sum(dim=-1, keepdim=True) * ref / ref_energy
    noise = est - proj
    ratio = (proj * proj).sum(dim=-1) / ((noise * noise).sum(dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)

def pit_reorder_by_sisnr(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """
    pred, tgt: [B,C,T]
    returns: pred_reordered [B,C,T] aligned so channel c matches tgt[:,c,:]
    """
    if pred.ndim != 3 or tgt.ndim != 3:
        raise RuntimeError(f"Expected [B,C,T], got pred={pred.shape}, tgt={tgt.shape}")
    if pred.shape != tgt.shape:
        raise RuntimeError(f"Shape mismatch pred={pred.shape}, tgt={tgt.shape}")

    B, C, T = pred.shape
    if C == 1:
        return pred

    # build pairwise SI-SNR matrix: pair[b, out_c, tgt_c]
    pair = torch.empty((B, C, C), device=pred.device, dtype=pred.dtype)
    for i in range(C):
        for j in range(C):
            pair[:, i, j] = si_snr(pred[:, i, :], tgt[:, j, :])

    perms = torch.tensor(list(permutations(range(C))), device=pred.device, dtype=torch.long)  # [P,C]
    P = perms.shape[0]
    ar = torch.arange(C, device=pred.device)

    # score[p,b] = sum_c pair[b, c, perms[p,c]]
    scores = []
    for p in range(P):
        scores.append(pair[:, ar, perms[p]].sum(dim=1))  # [B]
    scores = torch.stack(scores, dim=0)  # [P,B]

    best_p = scores.argmax(dim=0)              # [B]
    best_perm = perms[best_p]                  # [B,C] mapping out_c -> tgt_c

    # invert mapping to get tgt_c -> out_c, so we can reorder pred into tgt order
    inv = torch.empty_like(best_perm)
    for out_c in range(C):
        inv.scatter_(1, best_perm[:, out_c:out_c+1], out_c)

    gather_idx = inv.unsqueeze(-1).expand(-1, -1, T)  # [B,C,T]
    return pred.gather(dim=1, index=gather_idx)


class SE_metrics(nn.Module):
    """
    Computes epoch-average metrics for separation/enhancement.

    Expected shapes:
      pred_audio   : [B, C, T]
      target_audio : [B, C, T]
    Optional:
      mix_audio    : [B, T] or [B,1,T] to compute SI_SDRi.

    Notes:
      - PESQ/DNSMOS are run on CPU per utterance (slow but correct).
      - We PIT-match predictions to targets first (for fair metrics).
      - We store running sums + count (no giant Python lists).
    """
    def __init__(self, fs: int = 16000, device: str = "cpu",
                 use_dnsmos: bool = True, dnsmos_personalized: bool = False, dnsmos_threads: int = 4):
        super().__init__()
        self.fs = fs
        self.device = device
        self.use_dnsmos = use_dnsmos

        self.pesq_metric = PerceptualEvaluationSpeechQuality(fs=fs, mode="wb")
        self.stoi_metric = ShortTimeObjectiveIntelligibility(fs=fs, extended=False)
        self.sisdr_metric = ScaleInvariantSignalDistortionRatio()

        self.dnsmos_metric = None
        if use_dnsmos:
            self.dnsmos_metric = DeepNoiseSuppressionMeanOpinionScore(
                fs=fs,
                personalized=dnsmos_personalized,
                device=device,
                num_threads=dnsmos_threads,
            )

        self.reset()

    def reset(self):
        self.count = 0

        self.sum_pesq = 0.0
        self.sum_stoi = 0.0
        self.sum_sisdr = 0.0
        self.sum_sisdr_i = 0.0
        self.sum_sig = 0.0
        self.sum_bak = 0.0
        self.sum_ovrl = 0.0

        self.count_sisdr_i = 0
        self.count_dnsmos = 0

    @torch.no_grad()
    def update(self, pred_audio: torch.Tensor, target_audio: torch.Tensor, mix_audio: torch.Tensor | None = None):
        """
        pred_audio:   [B,C,T]
        target_audio: [B,C,T]
        mix_audio:    [B,T] or [B,1,T] (optional, for SI-SDRi)
        """
        if pred_audio.ndim != 3 or target_audio.ndim != 3:
            raise RuntimeError(f"Expected [B,C,T]. Got pred={pred_audio.shape}, tgt={target_audio.shape}")
        if pred_audio.shape != target_audio.shape:
            raise RuntimeError(f"pred/tgt mismatch: pred={pred_audio.shape} tgt={target_audio.shape}")

        # match lengths
        min_len = min(pred_audio.shape[-1], target_audio.shape[-1])
        pred_audio = pred_audio[..., :min_len]
        target_audio = target_audio[..., :min_len]

        if mix_audio is not None:
            if mix_audio.ndim == 3:
                mix_audio = mix_audio.squeeze(1)
            mix_audio = mix_audio[..., :min_len]

        # PIT reorder
        pred_audio = pit_reorder_by_sisnr(pred_audio, target_audio)
        
        B, C, T = pred_audio.shape

        # run per (utterance, speaker)
        for b in range(B):
            for c in range(C):
                p = pred_audio[b, c, :].detach().float().cpu().clamp(-1.0, 1.0).unsqueeze(0)  # [1,T]
                t = target_audio[b, c, :].detach().float().cpu().clamp(-1.0, 1.0).unsqueeze(0)

                # PESQ
                try:
                    self.sum_pesq += float(self.pesq_metric(p, t).item())
                except Exception:
                    pass

                # STOI
                try:
                    self.sum_stoi += float(self.stoi_metric(p, t).item())
                except Exception:
                    pass

                # SI-SDR
                try:
                    s = float(self.sisdr_metric(p, t).item())
                    self.sum_sisdr += s
                except Exception:
                    s = None

                # SI-SDRi
                if mix_audio is not None and s is not None:
                    try:
                        m = mix_audio[b].detach().float().cpu().clamp(-1.0, 1.0).unsqueeze(0)
                        s_mix = float(self.sisdr_metric(m, t).item())
                        self.sum_sisdr_i += (s - s_mix)
                        self.count_sisdr_i += 1
                    except Exception:
                        pass

                # DNSMOS (non-reference; only on predicted)
                if self.dnsmos_metric is not None:
                    try:
                        dns = self.dnsmos_metric(p)  # [p808, sig, bak, ovrl]
                        self.sum_sig += float(dns[1])
                        self.sum_bak += float(dns[2])
                        self.sum_ovrl += float(dns[3])
                        self.count_dnsmos += 1
                    except Exception:
                        pass

                self.count += 1

    def compute(self):
        denom = max(self.count, 1)
        out = {
            "PESQ": self.sum_pesq / denom,
            "STOI": self.sum_stoi / denom,
            "SI_SDR": self.sum_sisdr / denom,
        }
        if self.count_sisdr_i > 0:
            out["SI_SDRi"] = self.sum_sisdr_i / self.count_sisdr_i
        if self.count_dnsmos > 0:
            out["SIG"] = self.sum_sig / self.count_dnsmos
            out["BAK"] = self.sum_bak / self.count_dnsmos
            out["OVRL"] = self.sum_ovrl / self.count_dnsmos
        return out