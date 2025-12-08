#!/usr/bin/env python
import os
import torch
from torch.utils.data import DataLoader

from dataset import LibriMixDataModule
from metrics import SE_metrics

# ⬇️ import your Lightning system (this file – or whatever you named it)
#    If your train script file is called e.g. `train_pDCCRN_2sp.py`,
#    then change this import accordingly.
from train import E2EpSE   # <-- CHANGE MODULE NAME IF NEEDED


def cosine(a, b):
    """
    a, b: [B, D]
    returns: [B] cosine similarities
    """
    dot = (a * b).sum(dim=-1)
    an = a.norm(dim=-1) + 1e-8
    bn = b.norm(dim=-1) + 1e-8
    return dot / (an * bn)


def choose_wrong_embedding(e1, e2, emb_tgt):
    """
    e1, e2, emb_tgt: [B, D]

    For each sample, find which of (e1, e2) is MORE similar to the target,
    and deliberately return the OTHER one (the wrong embedding).
    """
    c1 = cosine(e1, emb_tgt)   # [B]
    c2 = cosine(e2, emb_tgt)   # [B]

    # mask = 1 where e1 is closer; we want the opposite speaker
    mask_e1_closer = (c1 > c2).unsqueeze(-1)   # [B, 1]
    wrong_emb = torch.where(mask_e1_closer, e2, e1)
    return wrong_emb


def main():
    # ------------------------
    # 1) Config
    # ------------------------
    data_root = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix"
    speaker_map_path = (
        "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/"
        "Libriuni_05_08/Libri2Mix_ovl50to80/wav16k/min/metadata/train360_mapping.json"
    )

    ckpt_path = (
        "/mnt/disks/data/model_ckpts/pDCCRN_2sp_tr360/best-epoch=66-val_separation=0.000.ckpt"   # <-- PUT YOUR BEST CHECKPOINT PATH HERE
    )

    batch_size = 8
    num_workers = 4
    num_speakers = 2
    sample_rate = 16000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device for model: {device}")

    # ------------------------
    # 2) DataModule & loader
    # ------------------------
    dm = LibriMixDataModule(
        data_root=data_root,
        speaker_map_path=speaker_map_path,
        batch_size=batch_size,
        num_workers=num_workers,
        num_speakers=num_speakers,
        sample_rate=sample_rate,
    )
    dm.setup()
    val_loader: DataLoader = dm.val_dataloader()

    # ------------------------
    # 3) Build model & load checkpoint
    # ------------------------
    print(f"Loading E2EpSE checkpoint from: {ckpt_path}")
    # hyperparams don't matter much here; they are overwritten by state_dict
    system = E2EpSE(
        lr=1e-4,
        finetune_encoder=False,
        emb_dim=256,
        speaker_map_path=speaker_map_path,
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    system.load_state_dict(ckpt["state_dict"], strict=False)
    system.to(device)
    system.eval()

    # Shorthand handles
    dual_model = system.dual_emb_model
    teacher = system.single_sp_model
    enh_model = system   # we'll use system.forward(...)

    dual_model.eval()
    teacher.eval()

    # ------------------------
    # 4) Metrics object
    # ------------------------
    metrics_rev = SE_metrics(fs=sample_rate)

    total_batches = len(val_loader)
    print(f"Evaluating reversed-embedding ablation on {total_batches} val batches ...")

    # ------------------------
    # 5) Iterate over validation set
    # ------------------------
    with torch.no_grad():
        for batch_idx, (mix, sources, labels) in enumerate(val_loader):
            # mix:     [B, T]
            # sources: [B, 2, T]

            mix = mix.to(device)
            sources = sources.to(device)

            B, S, T = sources.shape
            assert S == 2, "This script assumes 2-speaker Libri2Mix."

            # ----- Teacher embeddings from clean sources -----
            # emb1, emb2: [B, D]
            emb1 = teacher(sources[:, 0, :])
            emb2 = teacher(sources[:, 1, :])

            # ----- Dual embeddings from mixture -----
            # embs_dual: [B, 2, D]
            embs_dual = dual_model(mix)
            e1 = embs_dual[:, 0, :]
            e2 = embs_dual[:, 1, :]

            # ----- For each target speaker, pick the WRONG embedding -----
            # Target 1 uses emb1, but we deliberately choose the less similar of (e1, e2).
            wrong_emb1 = choose_wrong_embedding(e1, e2, emb1)  # [B, D]
            # Target 2 uses emb2, but again we pick the less similar.
            wrong_emb2 = choose_wrong_embedding(e1, e2, emb2)  # [B, D]

            # ----- Run enhancement with WRONG conditioning -----
            # Each call returns (mask, waveform); we only need waveform.
            pred1 = enh_model.forward(mix, emb=wrong_emb1)[1]  # [B, T1]
            pred2 = enh_model.forward(mix, emb=wrong_emb2)[1]  # [B, T2]

            # Match lengths across preds and sources
            min_len = min(pred1.shape[-1], pred2.shape[-1], sources.shape[-1])
            pred1 = pred1[..., :min_len]
            pred2 = pred2[..., :min_len]
            tgt = sources[..., :min_len]  # [B, 2, T']

            # Stack predictions into [B, 2, T']
            preds = torch.stack([pred1, pred2], dim=1)  # [B, 2, T']

            # We send everything to CPU for metrics (PESQ/STOI/DNSMOS)
            metrics_rev.update(preds.cpu(), tgt.cpu())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx+1}/{total_batches} batches", flush=True)

    # ------------------------
    # 6) Final results
    # ------------------------
    results = metrics_rev.compute()

    print("\n=== Reversed-Embedding Ablation on Libri2Mix (30-50  overlap noisy with wham tt) ===")
    for k, v in results.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()