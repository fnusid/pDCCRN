import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def safe_get(d, *keys, default=np.nan):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_path", type=str, required=False, default="/mnt/disks/data/dns_challenge/Baseline/pDCCRN/dev_testset/per_file.json")
    ap.add_argument("--out_dir", type=str, default="./dnsmos_analysis")
    ap.add_argument("--min_margin", type=float, default=0.20, help="Filter: keep only cos_margin >= this")
    ap.add_argument("--max_margin", type=float, default=1.0, help="Filter: keep only cos_margin <= this")
    ap.add_argument("--bins", type=str, default="0,0.05,0.10,0.15,0.20,0.30,0.50,1.0",
                    help="Comma-separated margin bin edges")
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()

    json_path = Path(args.json_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected JSON to be a list of per-file dicts.")

    # Flatten to dataframe
    rows = []
    for r in data:
        rows.append({
            "file": r.get("file", ""),
            "pick": r.get("pick", np.nan),
            "cos1": r.get("cos1", np.nan),
            "cos2": r.get("cos2", np.nan),
            "cos_margin": r.get("cos_margin", np.nan),

            "noisy_P808": safe_get(r, "noisy", "P808"),
            "noisy_SIG":  safe_get(r, "noisy", "SIG"),
            "noisy_BAK":  safe_get(r, "noisy", "BAK"),
            "noisy_OVRL": safe_get(r, "noisy", "OVRL"),

            "enh_P808": safe_get(r, "enh", "P808"),
            "enh_SIG":  safe_get(r, "enh", "SIG"),
            "enh_BAK":  safe_get(r, "enh", "BAK"),
            "enh_OVRL": safe_get(r, "enh", "OVRL"),

            "dP808": safe_get(r, "delta", "P808"),
            "dSIG":  safe_get(r, "delta", "SIG"),
            "dBAK":  safe_get(r, "delta", "BAK"),
            "dOVRL": safe_get(r, "delta", "OVRL"),
        })

    df = pd.DataFrame(rows)

    # Basic cleaning
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["cos_margin", "dOVRL", "dSIG", "dBAK"], how="any")

    # Optional margin filter
    if args.min_margin is not None:
        df = df[df["cos_margin"] >= args.min_margin]
    if args.max_margin is not None:
        df = df[df["cos_margin"] <= args.max_margin]

    # Save filtered dataframe
    df.to_csv(out_dir / "all_filtered.csv", index=False)

    # Overall summary
    def smean(col): return float(df[col].mean()) if len(df) else float("nan")
    def smed(col):  return float(df[col].median()) if len(df) else float("nan")

    summary = {
        "num_files": int(len(df)),
        "mean_cos_margin": smean("cos_margin"),
        "mean_delta_OVRL": smean("dOVRL"),
        "mean_delta_SIG":  smean("dSIG"),
        "mean_delta_BAK":  smean("dBAK"),
        "median_delta_OVRL": smed("dOVRL"),
        "median_delta_SIG":  smed("dSIG"),
        "median_delta_BAK":  smed("dBAK"),
        "mean_noisy_OVRL": smean("noisy_OVRL"),
        "mean_enh_OVRL":   smean("enh_OVRL"),
        "mean_noisy_SIG":  smean("noisy_SIG"),
        "mean_enh_SIG":    smean("enh_SIG"),
        "mean_noisy_BAK":  smean("noisy_BAK"),
        "mean_enh_BAK":    smean("enh_BAK"),
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Margin-binned stats
    bin_edges = [float(x) for x in args.bins.split(",")]
    df["margin_bin"] = pd.cut(df["cos_margin"], bins=bin_edges, include_lowest=True)

    grouped = df.groupby("margin_bin", observed=True).agg(
        n=("file", "count"),
        mean_margin=("cos_margin", "mean"),
        mean_dOVRL=("dOVRL", "mean"),
        mean_dSIG=("dSIG", "mean"),
        mean_dBAK=("dBAK", "mean"),
        med_dOVRL=("dOVRL", "median"),
        med_dSIG=("dSIG", "median"),
        med_dBAK=("dBAK", "median"),
        mean_noisy_OVRL=("noisy_OVRL", "mean"),
        mean_enh_OVRL=("enh_OVRL", "mean"),
    ).reset_index()

    grouped.to_csv(out_dir / "binned_by_margin.csv", index=False)

    # Top / bottom cases for listening
    topk = min(args.topk, len(df))
    best_ovrl = df.sort_values("dOVRL", ascending=False).head(topk)
    worst_ovrl = df.sort_values("dOVRL", ascending=True).head(topk)

    worst_sig = df.sort_values("dSIG", ascending=True).head(topk)
    best_bak = df.sort_values("dBAK", ascending=False).head(topk)

    best_ovrl.to_csv(out_dir / f"top_{topk}_dOVRL.csv", index=False)
    worst_ovrl.to_csv(out_dir / f"bottom_{topk}_dOVRL.csv", index=False)
    worst_sig.to_csv(out_dir / f"bottom_{topk}_dSIG.csv", index=False)
    best_bak.to_csv(out_dir / f"top_{topk}_dBAK.csv", index=False)

    # Print a quick report
    print("Wrote:")
    print(" -", out_dir / "summary.json")
    print(" -", out_dir / "binned_by_margin.csv")
    print(" -", out_dir / "all_filtered.csv")
    print(f" - top/bottom CSVs (k={topk})")
    print("\nSummary:")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()