#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot
import awkward as ak


EVENT_SUMMARY_FILES = {
    "ML": "../Scratch/Eval_and_Viz/withoutROIcut/ML_event_level_prediction_summary_STRIPGAP_MODEL.txt",
    "CURRENT": "../Scratch/Eval_and_Viz/withoutROIcut/CPP_event_level_prediction_summary.txt",
}

ROOT_FILE = (
    "/volatile/halla/sbs/bhasitha/Tracking_ML/"
    "GEM_Decoder_EVALUATION/filtered_replayed_withoutROIcut.root"
)


# ============================================================
# User-configurable branch names
# ============================================================

TREE_NAME = "T"

BRANCH_EVENT_ID = "fEvtHdr.fEvtNum"

BRANCH_STRIP_ID = "sbs.gemFT.m0.strip.istrip"
BRANCH_IS_U = "sbs.gemFT.m0.strip.IsU"
BRANCH_IS_V = "sbs.gemFT.m0.strip.IsV"

BRANCH_USTRIP_MIN = "sbs.gemFT.m0.roi.ustrip_min"
BRANCH_USTRIP_MAX = "sbs.gemFT.m0.roi.ustrip_max"
BRANCH_VSTRIP_MIN = "sbs.gemFT.m0.roi.vstrip_min"
BRANCH_VSTRIP_MAX = "sbs.gemFT.m0.roi.vstrip_max"

# Optional: if you want to count only strips with positive ADC sum,
# set this to the correct ADC-sum branch.
#
# If None, the script counts all strip entries inside the ROI.
# If set to "sbs.gemFT.m0.strip.ADCsum", the script counts only
# strip entries inside the ROI with ADCsum > 0.
BRANCH_ADC_SUM = None
# BRANCH_ADC_SUM = "sbs.gemFT.m0.strip.ADCsum"


# ============================================================
# Helper functions
# ============================================================

def read_event_summary(path, method_name):
    """
    Read event-level summary text file.

    Expected to contain one row per event.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Could not find summary file: {path}")

    df = pd.read_csv(path, sep=r"\s+", comment="#", engine="python")
    df.columns = [c.strip() for c in df.columns]

    print(f"\n[{method_name}] Loaded {path}")
    print(f"[{method_name}] Columns:")
    print(df.columns.tolist())

    event_candidates = [
        "event_id",
        "event",
        "evt",
        "evt_id",
        "fEvtNum",
        "entry",
    ]

    event_col = None
    for c in event_candidates:
        if c in df.columns:
            event_col = c
            break

    if event_col is None:
        raise RuntimeError(
            f"Could not infer event-id column in {path}. "
            f"Available columns are: {df.columns.tolist()}"
        )

    success_candidates = [
        "event_success",
        "success",
        "S_e",

        "any_good_blob",
        "any_good_blob_exact",
        "any_good_blob_range",

        "any_good_prediction_exact",
        "any_good_prediction_range",

        "hit_success",
        "correct",
        "found_true_hit",
    ]

    success_col = None
    for c in success_candidates:
        if c in df.columns:
            success_col = c
            break

    if success_col is None:
        raise RuntimeError(
            f"Could not infer success column in {path}. "
            f"Available columns are: {df.columns.tolist()}\n"
            "Edit success_candidates or explicitly rename the correct column."
        )

    print(f"[{method_name}] Using success column: {success_col}")

    out = df[[event_col, success_col]].copy()
    out = out.rename(
        columns={
            event_col: "event_id",
            success_col: f"{method_name}_success",
        }
    )

    out[f"{method_name}_success"] = out[f"{method_name}_success"].astype(bool)

    return out


def read_fired_strip_counts_from_root(root_file):
    """
    Read ROOT file and count fired U and V strips per event inside the ROI.

    By default, this counts retained strip entries inside the ROI.

    If BRANCH_ADC_SUM is set, this counts only strip entries inside the ROI
    with ADCsum > 0.
    """

    root_file = Path(root_file)

    if not root_file.exists():
        raise FileNotFoundError(f"Could not find ROOT file: {root_file}")

    branches = [
        BRANCH_EVENT_ID,
        BRANCH_STRIP_ID,
        BRANCH_IS_U,
        BRANCH_IS_V,
        BRANCH_USTRIP_MIN,
        BRANCH_USTRIP_MAX,
        BRANCH_VSTRIP_MIN,
        BRANCH_VSTRIP_MAX,
    ]

    if BRANCH_ADC_SUM is not None:
        branches.append(BRANCH_ADC_SUM)

    print(f"\nReading ROOT file:\n  {root_file}")
    print(f"Tree: {TREE_NAME}")
    print("Branches:")
    for b in branches:
        print(f"  {b}")

    with uproot.open(root_file) as f:
        tree = f[TREE_NAME]
        arr = tree.arrays(branches, library="ak")

    event_ids = ak.to_numpy(arr[BRANCH_EVENT_ID])

    strip_id = arr[BRANCH_STRIP_ID]
    is_u = arr[BRANCH_IS_U]
    is_v = arr[BRANCH_IS_V]

    roi_ustrip_min = arr[BRANCH_USTRIP_MIN]
    roi_ustrip_max = arr[BRANCH_USTRIP_MAX]
    roi_vstrip_min = arr[BRANCH_VSTRIP_MIN]
    roi_vstrip_max = arr[BRANCH_VSTRIP_MAX]

    # ============================================================
    # ROI boundary convention
    #
    # Inclusive:
    #   roi_min <= strip_id <= roi_max
    #
    # This matches the ML text-file writing logic.
    # ============================================================

    u_inside_roi = (
        (is_u == 1)
        & (strip_id >= roi_ustrip_min)
        & (strip_id <= roi_ustrip_max)
    )

    v_inside_roi = (
        (is_v == 1)
        & (strip_id >= roi_vstrip_min)
        & (strip_id <= roi_vstrip_max)
    )

    if BRANCH_ADC_SUM is not None:
        adc_sum = arr[BRANCH_ADC_SUM]

        u_count_mask = u_inside_roi & (adc_sum > 0)
        v_count_mask = v_inside_roi & (adc_sum > 0)

        count_label = "positive-ADC strip entries inside ROI"
    else:
        u_count_mask = u_inside_roi
        v_count_mask = v_inside_roi

        count_label = "strip entries inside ROI"

    n_u = ak.to_numpy(ak.sum(u_count_mask, axis=1))
    n_v = ak.to_numpy(ak.sum(v_count_mask, axis=1))

    df = pd.DataFrame({
        "event_id": event_ids,
        "n_u_fired": n_u,
        "n_v_fired": n_v,
        "roi_ustrip_min": ak.to_numpy(roi_ustrip_min),
        "roi_ustrip_max": ak.to_numpy(roi_ustrip_max),
        "roi_vstrip_min": ak.to_numpy(roi_vstrip_min),
        "roi_vstrip_max": ak.to_numpy(roi_vstrip_max),
    })

    # Keep U+V as a diagnostic, but the main plot will use U*V.
    df["n_uv_fired_sum"] = df["n_u_fired"] + df["n_v_fired"]

    # Main quantity requested: U * V fired-strip combinations.
    df["n_uv_fired_product"] = df["n_u_fired"] * df["n_v_fired"]

    df["roi_u_width"] = df["roi_ustrip_max"] - df["roi_ustrip_min"] + 1
    df["roi_v_width"] = df["roi_vstrip_max"] - df["roi_vstrip_min"] + 1
    df["roi_uv_width_product"] = df["roi_u_width"] * df["roi_v_width"]

    print(f"\nCounting definition: {count_label}")
    print("\nFired-strip count summary:")
    print(
        df[
            [
                "n_u_fired",
                "n_v_fired",
                "n_uv_fired_sum",
                "n_uv_fired_product",
                "roi_u_width",
                "roi_v_width",
                "roi_uv_width_product",
            ]
        ].describe()
    )

    return df


def make_efficiency_table(df, x_col, success_col, bins, min_events_per_bin=10):
    d = df.copy()

    d = d[np.isfinite(d[x_col])]
    d = d.dropna(subset=[success_col])

    d["bin"] = pd.cut(
        d[x_col],
        bins=bins,
        right=False,
        include_lowest=True,
    )

    rows = []

    for interval, g in d.groupby("bin", observed=True):
        n_total = len(g)

        if n_total < min_events_per_bin:
            continue

        n_success = int(g[success_col].astype(bool).sum())
        eff = n_success / n_total
        err = np.sqrt(eff * (1.0 - eff) / n_total)

        rows.append({
            "x_low": interval.left,
            "x_high": interval.right,
            "x_center": 0.5 * (interval.left + interval.right),
            "n_total": n_total,
            "n_success": n_success,
            "efficiency": eff,
            "error": err,
        })

    return pd.DataFrame(rows)


def plot_efficiency_vs_quantity(
    df,
    x_col,
    x_label,
    success_cols,
    bins,
    outpath,
    title,
    min_events_per_bin=10,
):
    plt.figure(figsize=(7.5, 5.5))

    for method_name, success_col in success_cols.items():
        eff_df = make_efficiency_table(
            df=df,
            x_col=x_col,
            success_col=success_col,
            bins=bins,
            min_events_per_bin=min_events_per_bin,
        )

        if eff_df.empty:
            print(f"[WARNING] No bins survived for {method_name}, {x_col}")
            continue

        plt.errorbar(
            eff_df["x_center"],
            eff_df["efficiency"],
            yerr=eff_df["error"],
            xerr=0.5 * (eff_df["x_high"] - eff_df["x_low"]),
            marker="o",
            linestyle="-",
            capsize=3,
            label=method_name,
        )

    plt.xlabel(x_label)
    plt.ylabel("Efficiency")
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.3)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(outpath, dpi=200)
    plt.savefig(outpath.with_suffix(".pdf"))
    plt.close()

    print(f"Saved: {outpath}")
    print(f"Saved: {outpath.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root-file",
        default=ROOT_FILE,
        help="Filtered replayed ROOT file.",
    )

    parser.add_argument(
        "--ml-summary",
        default=EVENT_SUMMARY_FILES["ML"],
        help="ML event-level prediction summary text file.",
    )

    parser.add_argument(
        "--cpp-summary",
        default=EVENT_SUMMARY_FILES["CURRENT"],
        help="Current C++ event-level prediction summary text file.",
    )

    parser.add_argument(
        "--outdir",
        default="plots/withoutROIcut/efficiency_vs_binned_fired_strip_product",
        help="Output directory.",
    )

    parser.add_argument(
        "--bin-width",
        type=int,
        default=5000,
        help="Bin width in U*V fired-strip product.",
    )

    parser.add_argument(
        "--min-events-per-bin",
        type=int,
        default=7,
        help="Minimum number of events required per bin.",
    )

    parser.add_argument(
        "--max-strip-product",
        type=int,
        default=None,
        help="Optional maximum x-axis U*V fired-strip product.",
    )

    parser.add_argument(
        "--make-uv-separate-plots",
        action="store_true",
        help="Also make separate U-only and V-only efficiency plots.",
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Read inputs
    # ------------------------------------------------------------

    fired_df = read_fired_strip_counts_from_root(args.root_file)

    ml_df = read_event_summary(args.ml_summary, "ML")
    cpp_df = read_event_summary(args.cpp_summary, "CURRENT")

    # ------------------------------------------------------------
    # Merge by event_id
    # ------------------------------------------------------------

    df = fired_df.merge(ml_df, on="event_id", how="inner")
    df = df.merge(cpp_df, on="event_id", how="inner")

    print("\nMerged event dataframe:")
    print(df.head())
    print(f"Number of merged events: {len(df)}")

    merged_csv = outdir / "merged_fired_strip_efficiency_input_withROI_UtimesV.csv"
    df.to_csv(merged_csv, index=False)
    print(f"Saved merged input table: {merged_csv}")

    # ------------------------------------------------------------
    # Binning for U*V
    # ------------------------------------------------------------

    if args.max_strip_product is None:
        max_strip_product = int(df["n_uv_fired_product"].max())
    else:
        max_strip_product = int(args.max_strip_product)

    print()
    print("MAX U*V FIRED-STRIP PRODUCT:", max_strip_product)
    print()

    bins_product = np.arange(
        0,
        max_strip_product + args.bin_width + 1,
        args.bin_width,
    )

    success_cols = {
        "Current C++": "CURRENT_success",
        "ML": "ML_success",
    }

    # ------------------------------------------------------------
    # Main plot: efficiency vs U*V fired-strip product
    # ------------------------------------------------------------

    plot_efficiency_vs_quantity(
        df=df,
        x_col="n_uv_fired_product",
        x_label="Number of fired strip combinations inside ROI, U × V",
        success_cols=success_cols,
        bins=bins_product,
        outpath=outdir / "efficiency_vs_uv_fired_strip_product_insideROI.png",
        title="Hit-finding efficiency vs fired strip product inside ROI",
        min_events_per_bin=args.min_events_per_bin,
    )

    # ------------------------------------------------------------
    # Optional diagnostic plots: U-only and V-only
    # ------------------------------------------------------------

    if args.make_uv_separate_plots:
        max_strip_count = int(
            np.nanmax([
                df["n_u_fired"].max(),
                df["n_v_fired"].max(),
            ])
        )

        bins_1d = np.arange(
            0,
            max_strip_count + 35 + 1,
            35,
        )

        plot_efficiency_vs_quantity(
            df=df,
            x_col="n_u_fired",
            x_label="Number of fired U strips inside ROI",
            success_cols=success_cols,
            bins=bins_1d,
            outpath=outdir / "efficiency_vs_u_fired_strips_insideROI.png",
            title="Hit-finding efficiency vs fired U strips inside ROI",
            min_events_per_bin=args.min_events_per_bin,
        )

        plot_efficiency_vs_quantity(
            df=df,
            x_col="n_v_fired",
            x_label="Number of fired V strips inside ROI",
            success_cols=success_cols,
            bins=bins_1d,
            outpath=outdir / "efficiency_vs_v_fired_strips_insideROI.png",
            title="Hit-finding efficiency vs fired V strips inside ROI",
            min_events_per_bin=args.min_events_per_bin,
        )


if __name__ == "__main__":
    main()