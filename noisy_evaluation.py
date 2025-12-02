#!/usr/bin/env python
import os
import torch
from torch.utils.data import DataLoader

# your dataset/datamodule file
from dataset import LibriMixDataModule, librimix_collate
# your metrics file
from metrics import SE_metrics


def main():
    # ------------------------
    # 1) Config
    # ------------------------
    data_root = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix"
    speaker_map_path = (
        "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/"
        "Libriuni_05_08/Libri2Mix_ovl50to80/wav16k/min/metadata/train360_mapping.json"
    )

    batch_size = 8
    num_workers = 4
    num_speakers = 2
    sample_rate = 16000
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device='cpu'
    print(f"Using device: {device}")

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
    # 3) Metrics object
    # ------------------------
    # This assumes SE_metrics is a torchmetrics-like class with
    #   - update(pred, target)
    #   - compute() -> dict
    #
    # and that it can handle [B, 2, T] tensors for both pred and target
    # (the same way you use it in training for your PSE model).
    metrics_noisy = SE_metrics(fs=sample_rate)

    # ------------------------
    # 4) Iterate over validation set
    # ------------------------
    metrics_noisy.to(device)
    metrics_noisy.eval()  # just to be safe (no dropout etc.)

    total_batches = len(val_loader)
    print(f"Evaluating noisy baseline on {total_batches} val batches ...")

    with torch.no_grad():
        for batch_idx, (mix, sources, labels) in enumerate(val_loader):
            # mix:     [B, T]
            # sources: [B, 2, T]  (clean references for both speakers)

            mix = mix.to(device)          # [B, T]
            sources = sources.to(device)  # [B, 2, T]

            # To be fair, we evaluate the mixture against **each** source.
            # Make a [B, 2, T] tensor where both channels are the same mix.
            # That way, SE_metrics will compare:
            #   noisy[:,0,:] vs sources[:,0,:]
            #   noisy[:,1,:] vs sources[:,1,:]
            noisy = mix.unsqueeze(1).repeat(1, sources.size(1), 1)  # [B, 2, T]

            # Update metrics on this batch
            metrics_noisy.update(noisy, sources)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx+1}/{total_batches} batches", flush=True)

    # ------------------------
    # 5) Final results
    # ------------------------
    results = metrics_noisy.compute()  # dict of metric_name -> value

    print("\n=== Noisy Baseline Metrics on Libri2Mix dev-clean ===")
    for k, v in results.items():
        # v may be a tensor or float
        if isinstance(v, torch.Tensor):
            v = v.item()
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()