#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot
import awkward as ak


# ============================================================
# INPUT FILES
# ============================================================

EVENT_SUMMARY_FILES = {
    "ML": "../Scratch/Eval_and_Viz/withoutROIcut/ML_event_level_prediction_summary_STRIPGAP_MODEL.txt",
    "CURRENT": "../Scratch/Eval_and_Viz/withoutROIcut/CPP_event_level_prediction_summary.txt",
}

ROOT_FILE = (
    "/volatile/halla/sbs/bhasitha/Tracking_ML/"
    "GEM_Decoder_EVALUATION/filtered_replayed_withoutROIcut.root"
)


# ============================================================
# ROOT BRANCHES
# ============================================================

TREE_NAME = "T"

BRANCH_EVENT_ID = "fEvtHdr.fEvtNum"

BRANCH_USTRIP_MIN = "sbs.gemFT.m0.roi.ustrip_min"
BRANCH_USTRIP_MAX = "sbs.gemFT.m0.roi.ustrip_max"
BRANCH_VSTRIP_MIN = "sbs.gemFT.m0.roi.vstrip_min"
BRANCH_VSTRIP_MAX = "sbs.gemFT.m0.roi.vstrip_max"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_bool_column(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
            "y": True,
            "n": False,
        })
        .fillna(False)
        .astype(bool)
    )


def read_event_summary(path, method_name):
    """
    Read event-level prediction summary.

    Uses the first recognized success column from success_candidates.
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

    success_candidates = [
        "any_good_prediction_exact",
        "any_good_prediction_range",
        "any_good_blob_exact",
        "any_good_blob_range",
        "event_success",
        "success",
        "S_e",
        "hit_success",
        "correct",
        "found_true_hit",
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

    success_col = None
    for c in success_candidates:
        if c in df.columns:
            success_col = c
            break

    if success_col is None:
        raise RuntimeError(
            f"Could not infer success column in {path}. "
            f"Available columns are: {df.columns.tolist()}"
        )

    print(f"[{method_name}] Using success column: {success_col}")

    out = df[[event_col, success_col]].copy()
    out = out.rename(
        columns={
            event_col: "event_id",
            success_col: f"{method_name}_success",
        }
    )

    out["event_id"] = pd.to_numeric(
        out["event_id"],
        errors="coerce",
    ).fillna(-1).astype(int)

    out[f"{method_name}_success"] = parse_bool_column(
        out[f"{method_name}_success"]
    )

    return out


def read_roi_size_from_root(root_file):
    """
    Read ROI boundaries from the ROOT file and compute one ROI size per event.
    """

    root_file = Path(root_file)

    if not root_file.exists():
        raise FileNotFoundError(f"Could not find ROOT file: {root_file}")

    branches = [
        BRANCH_EVENT_ID,
        BRANCH_USTRIP_MIN,
        BRANCH_USTRIP_MAX,
        BRANCH_VSTRIP_MIN,
        BRANCH_VSTRIP_MAX,
    ]

    print(f"\nReading ROOT file:\n  {root_file}")
    print(f"Tree: {TREE_NAME}")
    print("Branches:")
    for b in branches:
        print(f"  {b}")

    with uproot.open(root_file) as f:
        tree = f[TREE_NAME]
        arr = tree.arrays(branches, library="ak")

    event_id = ak.to_numpy(arr[BRANCH_EVENT_ID]).astype(int)

    umin = ak.to_numpy(arr[BRANCH_USTRIP_MIN])
    umax = ak.to_numpy(arr[BRANCH_USTRIP_MAX])
    vmin = ak.to_numpy(arr[BRANCH_VSTRIP_MIN])
    vmax = ak.to_numpy(arr[BRANCH_VSTRIP_MAX])

    roi_u_size = umax - umin + 1
    roi_v_size = vmax - vmin + 1

    df = pd.DataFrame({
        "event_id": event_id,

        "roi_ustrip_min": umin,
        "roi_ustrip_max": umax,
        "roi_vstrip_min": vmin,
        "roi_vstrip_max": vmax,

        "roi_u_size": roi_u_size,
        "roi_v_size": roi_v_size,
    })

    df["roi_uv_size_sum"] = df["roi_u_size"] + df["roi_v_size"]
    df["roi_2d_size"] = df["roi_u_size"] * df["roi_v_size"]
    df["roi_max_width"] = df[["roi_u_size", "roi_v_size"]].max(axis=1)

    print("\nROI size summary:")
    print(
        df[
            [
                "roi_u_size",
                "roi_v_size",
                "roi_uv_size_sum",
                "roi_2d_size",
                "roi_max_width",
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
    plot_percent=True,
    y_max=102,
):
    plt.figure(figsize=(8.5, 6.0))

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

        x = eff_df["x_center"].to_numpy()
        y = eff_df["efficiency"].to_numpy()
        yerr = eff_df["error"].to_numpy()

        if plot_percent:
            y = 100.0 * y
            yerr = 100.0 * yerr

        plt.errorbar(
            x,
            y,
            yerr=yerr,
            xerr=0.5 * (eff_df["x_high"] - eff_df["x_low"]),
            marker="o",
            linestyle="-",
            capsize=3,
            label=method_name,
        )

    plt.xlabel(x_label)

    if plot_percent:
        plt.ylabel("Truth-hit efficiency [%]")
        plt.ylim(0, y_max)
    else:
        plt.ylabel("Truth-hit efficiency")
        plt.ylim(0, 1.02)

    plt.grid(alpha=0.3)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(outpath, dpi=300)
    plt.savefig(outpath.with_suffix(".pdf"))
    plt.close()

    print(f"Saved: {outpath}")
    print(f"Saved: {outpath.with_suffix('.pdf')}")


# ============================================================
# MAIN
# ============================================================

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
        default="plots/withoutROIcut/efficiency_vs_binned_roi_size",
        help="Output directory.",
    )

    parser.add_argument(
        "--bin-width-1d",
        type=int,
        default=25,
        help="Bin width for 1D ROI sizes such as U, V, U+V, and max width.",
    )

    parser.add_argument(
        "--bin-width-2d",
        type=int,
        default=25000,
        help="Bin width for 2D ROI size U*V.",
    )

    parser.add_argument(
        "--min-events-per-bin",
        type=int,
        default=20,
        help="Minimum number of events required per bin.",
    )

    parser.add_argument(
        "--max-roi-size-1d",
        type=float,
        default=None,
        help="Optional max x-axis value for 1D ROI-size plots.",
    )

    parser.add_argument(
        "--max-roi-size-2d",
        type=float,
        default=None,
        help="Optional max x-axis value for 2D ROI-size plot.",
    )

    parser.add_argument(
        "--y-max",
        type=float,
        default=102.0,
        help="Maximum y-axis value in percent.",
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Read inputs
    # ------------------------------------------------------------

    roi_df = read_roi_size_from_root(args.root_file)

    ml_df = read_event_summary(args.ml_summary, "ML")
    cpp_df = read_event_summary(args.cpp_summary, "CURRENT")

    # ------------------------------------------------------------
    # Merge by event_id
    # ------------------------------------------------------------

    df = roi_df.merge(ml_df, on="event_id", how="inner")
    df = df.merge(cpp_df, on="event_id", how="inner")

    print("\nMerged event dataframe:")
    print(df.head())
    print(f"Number of merged events: {len(df)}")

    merged_csv = outdir / "merged_efficiency_vs_roi_size_input.csv"
    df.to_csv(merged_csv, index=False)
    print(f"Saved merged input table: {merged_csv}")

    success_cols = {
        "ML": "ML_success",
        "Current C++": "CURRENT_success",
    }

    # ------------------------------------------------------------
    # Build bins
    # ------------------------------------------------------------

    if args.max_roi_size_1d is None:
        max_1d = int(
            np.nanmax([
                df["roi_u_size"].max(),
                df["roi_v_size"].max(),
                df["roi_uv_size_sum"].max(),
                df["roi_max_width"].max(),
            ])
        )
    else:
        max_1d = int(args.max_roi_size_1d)

    if args.max_roi_size_2d is None:
        max_2d = int(df["roi_2d_size"].max())
    else:
        max_2d = int(args.max_roi_size_2d)

    bins_1d = np.arange(
        0,
        max_1d + args.bin_width_1d + 1,
        args.bin_width_1d,
    )

    bins_2d = np.arange(
        0,
        max_2d + args.bin_width_2d + 1,
        args.bin_width_2d,
    )

    print("\nBinning:")
    print(f"1D ROI bin width: {args.bin_width_1d}")
    print(f"1D ROI max:       {max_1d}")
    print(f"2D ROI bin width: {args.bin_width_2d}")
    print(f"2D ROI max:       {max_2d}")

    # ------------------------------------------------------------
    # Make plots
    # ------------------------------------------------------------

    plot_efficiency_vs_quantity(
        df=df,
        x_col="roi_u_size",
        x_label="ROI size in U strips",
        success_cols=success_cols,
        bins=bins_1d,
        outpath=outdir / "efficiency_vs_roi_u_size.png",
        title="Hit-finding efficiency vs ROI U size",
        min_events_per_bin=args.min_events_per_bin,
        plot_percent=True,
        y_max=args.y_max,
    )

    plot_efficiency_vs_quantity(
        df=df,
        x_col="roi_v_size",
        x_label="ROI size in V strips",
        success_cols=success_cols,
        bins=bins_1d,
        outpath=outdir / "efficiency_vs_roi_v_size.png",
        title="Hit-finding efficiency vs ROI V size",
        min_events_per_bin=args.min_events_per_bin,
        plot_percent=True,
        y_max=args.y_max,
    )

    plot_efficiency_vs_quantity(
        df=df,
        x_col="roi_uv_size_sum",
        x_label="ROI size, U + V strips",
        success_cols=success_cols,
        bins=bins_1d,
        outpath=outdir / "efficiency_vs_roi_uv_size_sum.png",
        title="Hit-finding efficiency vs ROI size, U + V",
        min_events_per_bin=args.min_events_per_bin,
        plot_percent=True,
        y_max=args.y_max,
    )

    plot_efficiency_vs_quantity(
        df=df,
        x_col="roi_max_width",
        x_label=r"ROI max width, max(U size, V size)",
        success_cols=success_cols,
        bins=bins_1d,
        outpath=outdir / "efficiency_vs_roi_max_width.png",
        title="Hit-finding efficiency vs ROI maximum width",
        min_events_per_bin=args.min_events_per_bin,
        plot_percent=True,
        y_max=args.y_max,
    )

    plot_efficiency_vs_quantity(
        df=df,
        x_col="roi_2d_size",
        x_label=r"ROI 2D size, U size $\times$ V size",
        success_cols=success_cols,
        bins=bins_2d,
        outpath=outdir / "efficiency_vs_roi_2d_size.png",
        title="Hit-finding efficiency vs ROI 2D size",
        min_events_per_bin=args.min_events_per_bin,
        plot_percent=True,
        y_max=args.y_max,
    )

    print("\nAll outputs saved in:")
    print(outdir)


if __name__ == "__main__":
    main()