#!/usr/bin/env python
import torch
from torch.utils.data import DataLoader

from dataset import LibriMixDataModule
from train import E2EpSE        # your Lightning module
from metrics import SE_metrics  # same class you use in E2EpSE


def main():
    # ------------------------
    # 1) Config
    # ------------------------
    data_root = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix"
    speaker_map_path = (
        "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/3sp/Libri3Mix_ovl50to80/wav16k/min/metadata/train360_mapping.json"
    )

    ckpt_path = (
      "/mnt/disks/data/model_ckpts/pDCCRN_3sp_tr360/best-epoch=65-val_separation=0.000.ckpt"
    )

    batch_size = 4
    num_workers = 20
    num_speakers = 3
    sample_rate = 16000

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device='cpu'
    print("Using device:", device)

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
    system.metrics = SE_metrics(fs=sample_rate)  # rebuild metrics exactly once
    system.metrics.reset()

    total_batches = len(val_loader)
    print(f"Evaluating PSE using model.validation_step() on {total_batches} batches ...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # move batch to device the same way Lightning does
            mix, src, labels = batch
            mix = mix.to(device)
            src = src.to(device)
            labels = labels.to(device)

            # reuse the exact validation code from E2EpSE
            system.validation_step((mix, src, labels), batch_idx)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx+1}/{total_batches} batches")

    # ------------------------
    # 4) Final results from system.metrics
    # ------------------------
    results = system.metrics.compute()

    print("\n=== PSE (training-style, exact validation_step, speech separation) on Libri2Mix dev ===")
    for k, v in results.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()