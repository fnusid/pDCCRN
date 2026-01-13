#!/usr/bin/env python
import torch
from torch.utils.data import DataLoader

from dataset import LibriMixDataModule
from train import E2EpSE        # your Lightning module
from metrics import SE_metrics  # same class you use in E2EpSE

import sys
sys.path.append('/home/sidharth./codebase')
from wavlm_single_embedding.eval_metrics import compute_separation, compute_clustering_metrics, load_model
import itertools
perms = list(itertools.permutations([0, 1, 2]))

###########FINISH THIS#####################
def si_snr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
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


def main():
    # ------------------------
    # 1) Config
    # ------------------------
    data_root = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix"
    # speaker_map_path = (
    #     "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/Libriuni_05_08/Libri2Mix_ovl50to80/wav16k/min/metadata/train360_mapping.json"
    # )
    speaker_map_path = ("/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/3sp/Libri3Mix_ovl50to80/wav16k/min/metadata/train360_mapping.json")

    ckpt_path = (
        "/mnt/disks/data/model_ckpts/convtasnet_3sp_sep_/best-epoch=18-val_separation=0.000.ckpt"
    )

    emb_ckpt = "/mnt/disks/data/model_ckpts/librispeech_asp_wavlm_tr360/best-epoch=62-val_separation=0.000.ckpt"
    

    batch_size = 16
    num_workers = 0
    num_speakers = 3
    sample_rate = 16000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    emb_model = load_model(emb_ckpt, emb_dim=256, device=device)
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
    val_loader: DataLoader = dm.test_dataloader()

    # ------------------------
    # 3) Build model & load checkpoint
    # ------------------------
    print(f"Loading E2EpSE checkpoint from: {ckpt_path}")
    system = E2EpSE(
        lr=1e-4,
        finetune_encoder=False,
        emb_dim=256,
        speaker_map_path=speaker_map_path,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    # breakpoint()
    system.load_state_dict(ckpt["state_dict"], strict=True)
    system.to(device)
    system.eval()

    # *** IMPORTANT: use the SAME metrics object & validation_step logic ***
    # system.metrics = SE_metrics(fs=sample_rate)  # rebuild metrics exactly once
    # system.metrics.reset()

    total_batches = len(val_loader)
    print(f"Evaluating PSE using model.validation_step() on {total_batches} batches ...")

    embeddings = []
    lbl = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # move batch to device the same way Lightning does
            mix, src, labels = batch
            mix = mix.to(device)
            src = src.to(device)
            labels = labels.to(device)
            # breakpoint()
            # reuse the exact validation code from E2EpSE
            out, source, labels = system.validation_step((mix, src, labels), batch_idx)

            # # do PIT per-sample and swap labels to match the chosen ordering (2sp)
            # s_corr = si_snr(out[:, 0, :], source[:, 0, :]) + si_snr(out[:, 1, :], source[:, 1, :])
            # s_swap = si_snr(out[:, 0, :], source[:, 1, :]) + si_snr(out[:, 1, :], source[:, 0, :])
            # swap_mask = s_swap > s_corr  # [B]
            # out_reordered = out.clone()
            # out_reordered[swap_mask, 0, :] = out[swap_mask, 1, :]
            # out_reordered[swap_mask, 1, :] = out[swap_mask, 0, :]
            # labels_reordered = labels.clone()
            # labels_reordered[swap_mask, 0] = labels[swap_mask, 1]
            # labels_reordered[swap_mask, 1] = labels[swap_mask, 0]

            # do PIT per-sample and swap labels to match the chosen ordering (3sp)
            scores = []
            for p in perms:
                s = (
                    si_snr(out[:, p[0], :], source[:, 0, :]) +
                    si_snr(out[:, p[1], :], source[:, 1, :]) +
                    si_snr(out[:, p[2], :], source[:, 2, :])
                )
                scores.append(s)
            scores = torch.stack(scores, dim=1)  # [B, 6]

            best_idx = scores.argmax(dim=1)  # [B]
            out_reordered = out.clone()
            labels_reordered = labels.clone()

            for pi, p in enumerate(perms):
                mask = best_idx == pi
                if mask.any():
                    out_reordered[mask, 0, :] = out[mask, p[0], :]
                    out_reordered[mask, 1, :] = out[mask, p[1], :]
                    out_reordered[mask, 2, :] = out[mask, p[2], :]
                    labels_reordered[mask, 0] = labels[mask, p[0]]
                    labels_reordered[mask, 1] = labels[mask, p[1]]
                    labels_reordered[mask, 2] = labels[mask, p[2]]

            # emb = emb_model(torch.concat([out_reordered[:, 0, :], out_reordered[:, 1, :]], dim=0))  # [2B, D]
        
            emb = emb_model(torch.concat([out_reordered[:, 0, :], out_reordered[:, 1, :], out_reordered[:, 2, :]], dim=0))  # [3B, D]
            labels = labels_reordered.reshape(-1)
            embeddings.append(emb.cpu())
            lbl.append(labels.cpu())




            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx+1}/{total_batches} batches")

    # ------------------------
    # 4) Final results from system.metrics
    # ------------------------
    # results = system.metrics.compute()
    embeddings = torch.cat(embeddings)
    lbl = torch.cat(lbl)
    sep, same_mean, diff_mean = compute_separation(embeddings, lbl)
    clust = compute_clustering_metrics(embeddings, lbl)
    print("\n=== Separation Metrics (dev-clean) ===")
    print(f"same_mean_cos = {same_mean:.4f}")
    print(f"diff_mean_cos = {diff_mean:.4f}")
    print(f"separation    = {sep:.4f}")
    print("\n=== Clustering Metrics (dev-clean) ===")
    print(f"cluster_acc = {clust['cluster_acc']:.4f}")
    print(f"nmi         = {clust['nmi']:.4f}")
    print(f"ari         = {clust['ari']:.4f}")
    print(f"silhouette  = {clust['silhouette']:.4f}")

    # print("\n=== PSE (training-style, exact validation_step, speech separation) on Libri2Mix dev ===")
    # for k, v in results.items():
    #     if isinstance(v, torch.Tensor):
    #         v = v.item()
    #     print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
