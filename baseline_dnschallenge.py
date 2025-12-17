# eval_baseline_dnsmos_p835.py
# Compute DNSMOS P.835 (SIG/BAK/OVRL + P808) for a folder of baseline enhanced wavs.
# Optionally also score matching "noisy" wavs (same filename) and report deltas.

import os
import glob
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm
from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score


def list_audio_files(folder: str):
    exts = ("*.wav", "*.flac", "*.mp3", "*.ogg")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files)


def load_audio_mono(path: str, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)  # [C,T]
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.clamp(-1.0, 1.0).float()  # [1,T] float32


@torch.no_grad()
def dnsmos_p835_one(wav_1t: torch.Tensor, fs: int, personalized: bool,
                    device: str = None, num_threads: int = None):
    """
    wav_1t: [1,T] float tensor (CPU recommended)
    returns dict: P808, SIG, BAK, OVRL
    """
    scores = deep_noise_suppression_mean_opinion_score(
        wav_1t,
        fs=fs,
        personalized=personalized,
        device=device,
        num_threads=num_threads,
        cache_session=True,
    )  # [1,4] = [p808, sig, bak, ovrl]
    return {
        "P808": float(scores[0, 0].item()),
        "SIG":  float(scores[0, 1].item()),
        "BAK":  float(scores[0, 2].item()),
        "OVRL": float(scores[0, 3].item()),
    }


def mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enh_dir", type=str, required=True,
                    help="Folder containing enhanced baseline wavs.")
    ap.add_argument("--noisy_dir", type=str, default=None,
                    help="Optional: folder containing noisy wavs with same filenames.")
    ap.add_argument("--sr", type=int, default=16000)

    ap.add_argument("--personalized", action="store_true",
                    help="TorchMetrics DNSMOS 'personalized' flag.")
    ap.add_argument("--dnsmos_device", type=str, default=None,
                    help="Pass-through to TorchMetrics DNSMOS 'device' arg (e.g., 'cpu' or 'cuda').")
    ap.add_argument("--dnsmos_num_threads", type=int, default=None)

    ap.add_argument("--out_dir", type=str, default="./baseline_dnsmos_out")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only process first N files (debug).")
    args = ap.parse_args()

    enh_dir = Path(args.enh_dir)
    assert enh_dir.exists(), f"Missing enh_dir: {enh_dir}"

    noisy_dir = Path(args.noisy_dir) if args.noisy_dir else None
    if noisy_dir is not None:
        assert noisy_dir.exists(), f"Missing noisy_dir: {noisy_dir}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_audio_files(str(enh_dir))
    if args.limit and args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise RuntimeError(f"No audio found in {enh_dir}")

    per_file = []
    pbar = tqdm(files, desc="DNSMOS P.835 scoring", unit="file", dynamic_ncols=True)

    for i, enh_path in enumerate(pbar):
        fname = Path(enh_path).name

        enh = load_audio_mono(str(enh_path), args.sr).cpu()  # [1,T] CPU
        enh_m = dnsmos_p835_one(
            enh, fs=args.sr, personalized=args.personalized,
            device=args.dnsmos_device, num_threads=args.dnsmos_num_threads
        )

        row = {"file": fname, "enh": enh_m}

        if noisy_dir is not None:
            npath = noisy_dir / fname
            if npath.exists():
                noisy = load_audio_mono(str(npath), args.sr).cpu()
                noisy_m = dnsmos_p835_one(
                    noisy, fs=args.sr, personalized=args.personalized,
                    device=args.dnsmos_device, num_threads=args.dnsmos_num_threads
                )
                row["noisy"] = noisy_m
                row["delta"] = {k: row["enh"][k] - row["noisy"][k] for k in ["P808", "SIG", "BAK", "OVRL"]}
            else:
                row["noisy_missing"] = True

        per_file.append(row)

        # rolling postfix
        ovr = [r["enh"]["OVRL"] for r in per_file]
        bak = [r["enh"]["BAK"] for r in per_file]
        sig = [r["enh"]["SIG"] for r in per_file]
        pbar.set_postfix({
            "OVRL": f"{mean(ovr):.3f}",
            "SIG":  f"{mean(sig):.3f}",
            "BAK":  f"{mean(bak):.3f}",
        })

        if (i + 1) % 100 == 0:
            with open(out_dir / "per_file_partial.json", "w") as f:
                json.dump(per_file, f, indent=2)

    # summary
    enh_sig  = [r["enh"]["SIG"] for r in per_file]
    enh_bak  = [r["enh"]["BAK"] for r in per_file]
    enh_ovrl = [r["enh"]["OVRL"] for r in per_file]
    enh_p808 = [r["enh"]["P808"] for r in per_file]

    summary = {
        "num_files": len(per_file),
        "enh_mean_SIG": mean(enh_sig),
        "enh_mean_BAK": mean(enh_bak),
        "enh_mean_OVRL": mean(enh_ovrl),
        "enh_mean_P808": mean(enh_p808),
    }

    # deltas if available
    if noisy_dir is not None:
        deltas = [r["delta"] for r in per_file if "delta" in r]
        summary["num_files_with_noisy"] = len(deltas)
        if deltas:
            summary.update({
                "delta_mean_SIG":  mean([d["SIG"] for d in deltas]),
                "delta_mean_BAK":  mean([d["BAK"] for d in deltas]),
                "delta_mean_OVRL": mean([d["OVRL"] for d in deltas]),
                "delta_mean_P808": mean([d["P808"] for d in deltas]),
            })

    with open(out_dir / "per_file.json", "w") as f:
        json.dump(per_file, f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[DONE]", json.dumps(summary, indent=2))
    print(f"[DONE] Wrote {out_dir / 'summary.json'} and {out_dir / 'per_file.json'}")


if __name__ == "__main__":
    main()