#!/usr/bin/env python3

import numpy as np
import awkward as ak
import uproot
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


ROOT_FILE = (
    "/volatile/halla/sbs/bhasitha/Tracking_ML/"
    "GEM_Decoder_EVALUATION/filtered_replayed_withoutROIcut.root"
)

TREE_NAME = "T"

BRANCH_EVENT_ID = "fEvtHdr.fEvtNum"

BRANCH_STRIP_ID = "sbs.gemFT.m0.strip.istrip"
BRANCH_IS_U = "sbs.gemFT.m0.strip.IsU"
BRANCH_IS_V = "sbs.gemFT.m0.strip.IsV"
BRANCH_ADC_SAMPLES = "sbs.gemFT.m0.strip.ADCsamples"

BRANCH_USTRIP_MIN = "sbs.gemFT.m0.roi.ustrip_min"
BRANCH_USTRIP_MAX = "sbs.gemFT.m0.roi.ustrip_max"
BRANCH_VSTRIP_MIN = "sbs.gemFT.m0.roi.vstrip_min"
BRANCH_VSTRIP_MAX = "sbs.gemFT.m0.roi.vstrip_max"

OUTDIR = Path("plots/withoutROIcut/fired_strip_histograms")


def count_unique_per_event(jagged_array):
    """
    Count unique values in each event of a jagged awkward array.

    This avoids using ak.unique, which is not available in some awkward versions.
    """
    out = []

    for event_values in ak.to_list(jagged_array):
        out.append(len(set(event_values)))

    return np.asarray(out, dtype=int)


def make_hist(values, xlabel, title, outname, bins=None):
    values = np.asarray(values)

    if len(values) == 0:
        print(f"[WARNING] No values for {outname}. Skipping histogram.")
        return

    if bins is None:
        max_val = int(np.nanmax(values))
        bins = np.arange(-0.5, max_val + 1.5, 1)

    plt.figure(figsize=(7.5, 5.5))
    plt.hist(values, bins=bins, histtype="step", linewidth=1.8)
    plt.xlabel(xlabel)
    plt.ylabel("Events")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    png_path = OUTDIR / f"{outname}.png"
    pdf_path = OUTDIR / f"{outname}.pdf"

    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path)
    plt.close()

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def make_overlay_hist(u_values, v_values, xlabel, title, outname, bins=None):
    u_values = np.asarray(u_values)
    v_values = np.asarray(v_values)

    if len(u_values) == 0 or len(v_values) == 0:
        print(f"[WARNING] No values for {outname}. Skipping overlay.")
        return

    if bins is None:
        max_val = int(max(np.nanmax(u_values), np.nanmax(v_values)))
        bins = np.arange(-0.5, max_val + 1.5, 1)

    plt.figure(figsize=(7.5, 5.5))
    plt.hist(
        u_values,
        bins=bins,
        histtype="step",
        linewidth=1.8,
        label="U",
    )
    plt.hist(
        v_values,
        bins=bins,
        histtype="step",
        linewidth=1.8,
        label="V",
    )

    plt.xlabel(xlabel)
    plt.ylabel("Events")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    png_path = OUTDIR / f"{outname}.png"
    pdf_path = OUTDIR / f"{outname}.pdf"

    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path)
    plt.close()

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def print_summary(name, values):
    values = np.asarray(values)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"events = {len(values)}")
    print(f"min    = {np.min(values)}")
    print(f"max    = {np.max(values)}")
    print(f"mean   = {np.mean(values):.3f}")
    print(f"median = {np.median(values):.3f}")
    print(f"p90    = {np.percentile(values, 90):.3f}")
    print(f"p95    = {np.percentile(values, 95):.3f}")
    print(f"p99    = {np.percentile(values, 99):.3f}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    branches = [
        BRANCH_EVENT_ID,
        BRANCH_STRIP_ID,
        BRANCH_IS_U,
        BRANCH_IS_V,
        BRANCH_ADC_SAMPLES,
        BRANCH_USTRIP_MIN,
        BRANCH_USTRIP_MAX,
        BRANCH_VSTRIP_MIN,
        BRANCH_VSTRIP_MAX,
    ]

    print("\nReading ROOT file:")
    print(ROOT_FILE)
    print("\nBranches:")
    for b in branches:
        print(f"  {b}")

    with uproot.open(ROOT_FILE) as f:
        tree = f[TREE_NAME]
        arr = tree.arrays(branches, library="ak")

    event_id = arr[BRANCH_EVENT_ID]
    strip_id = arr[BRANCH_STRIP_ID]
    is_u = arr[BRANCH_IS_U]
    is_v = arr[BRANCH_IS_V]
    adc_samples = arr[BRANCH_ADC_SAMPLES]

    roi_ustrip_min = arr[BRANCH_USTRIP_MIN]
    roi_ustrip_max = arr[BRANCH_USTRIP_MAX]
    roi_vstrip_min = arr[BRANCH_VSTRIP_MIN]
    roi_vstrip_max = arr[BRANCH_VSTRIP_MAX]

    # ADC sum per strip entry:
    # event -> strip -> summed over time samples
    adc_sum = ak.sum(adc_samples, axis=-1)

    # ============================================================
    # ROI masks
    #
    # Inclusive boundary convention:
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

    # ============================================================
    # Raw strip-entry counts inside ROI
    # ============================================================

    n_u_entries = ak.sum(u_inside_roi, axis=1)
    n_v_entries = ak.sum(v_inside_roi, axis=1)

    # ============================================================
    # Unique strip ID counts inside ROI
    # ============================================================

    u_strip_ids = strip_id[u_inside_roi]
    v_strip_ids = strip_id[v_inside_roi]

    n_u_unique_np = count_unique_per_event(u_strip_ids)
    n_v_unique_np = count_unique_per_event(v_strip_ids)

    # ============================================================
    # Positive-ADC strip-entry counts inside ROI
    # ============================================================

    n_u_adcpos_entries = ak.sum(
        u_inside_roi & (adc_sum > 0),
        axis=1,
    )

    n_v_adcpos_entries = ak.sum(
        v_inside_roi & (adc_sum > 0),
        axis=1,
    )

    # ============================================================
    # Unique positive-ADC strip ID counts inside ROI
    # ============================================================

    u_strip_ids_adcpos = strip_id[u_inside_roi & (adc_sum > 0)]
    v_strip_ids_adcpos = strip_id[v_inside_roi & (adc_sum > 0)]

    n_u_unique_adcpos_np = count_unique_per_event(u_strip_ids_adcpos)
    n_v_unique_adcpos_np = count_unique_per_event(v_strip_ids_adcpos)

    # ============================================================
    # Convert remaining awkward arrays to NumPy
    # ============================================================

    event_id_np = ak.to_numpy(event_id)

    n_u_entries_np = ak.to_numpy(n_u_entries)
    n_v_entries_np = ak.to_numpy(n_v_entries)

    n_u_adcpos_entries_np = ak.to_numpy(n_u_adcpos_entries)
    n_v_adcpos_entries_np = ak.to_numpy(n_v_adcpos_entries)

    roi_ustrip_min_np = ak.to_numpy(roi_ustrip_min)
    roi_ustrip_max_np = ak.to_numpy(roi_ustrip_max)
    roi_vstrip_min_np = ak.to_numpy(roi_vstrip_min)
    roi_vstrip_max_np = ak.to_numpy(roi_vstrip_max)

    roi_u_width_np = roi_ustrip_max_np - roi_ustrip_min_np + 1
    roi_v_width_np = roi_vstrip_max_np - roi_vstrip_min_np + 1

    # ============================================================
    # Print summaries
    # ============================================================

    print_summary("U strip entries inside ROI", n_u_entries_np)
    print_summary("V strip entries inside ROI", n_v_entries_np)

    print_summary("Unique U strip IDs inside ROI", n_u_unique_np)
    print_summary("Unique V strip IDs inside ROI", n_v_unique_np)

    print_summary(
        "Positive-ADC U strip entries inside ROI",
        n_u_adcpos_entries_np,
    )

    print_summary(
        "Positive-ADC V strip entries inside ROI",
        n_v_adcpos_entries_np,
    )

    print_summary(
        "Unique positive-ADC U strip IDs inside ROI",
        n_u_unique_adcpos_np,
    )

    print_summary(
        "Unique positive-ADC V strip IDs inside ROI",
        n_v_unique_adcpos_np,
    )

    print_summary("ROI U width", roi_u_width_np)
    print_summary("ROI V width", roi_v_width_np)

    # ============================================================
    # Save diagnostic CSV
    # ============================================================

    df = pd.DataFrame({
        "event_id": event_id_np,

        "roi_ustrip_min": roi_ustrip_min_np,
        "roi_ustrip_max": roi_ustrip_max_np,
        "roi_vstrip_min": roi_vstrip_min_np,
        "roi_vstrip_max": roi_vstrip_max_np,

        "roi_u_width": roi_u_width_np,
        "roi_v_width": roi_v_width_np,

        "n_u_entries_inside_roi": n_u_entries_np,
        "n_v_entries_inside_roi": n_v_entries_np,

        "n_u_unique_inside_roi": n_u_unique_np,
        "n_v_unique_inside_roi": n_v_unique_np,

        "n_u_adcpos_entries_inside_roi": n_u_adcpos_entries_np,
        "n_v_adcpos_entries_inside_roi": n_v_adcpos_entries_np,

        "n_u_unique_adcpos_inside_roi": n_u_unique_adcpos_np,
        "n_v_unique_adcpos_inside_roi": n_v_unique_adcpos_np,
    })

    df["n_uv_entries_inside_roi"] = (
        df["n_u_entries_inside_roi"] + df["n_v_entries_inside_roi"]
    )

    df["n_uv_unique_inside_roi"] = (
        df["n_u_unique_inside_roi"] + df["n_v_unique_inside_roi"]
    )

    df["n_uv_adcpos_entries_inside_roi"] = (
        df["n_u_adcpos_entries_inside_roi"]
        + df["n_v_adcpos_entries_inside_roi"]
    )

    df["n_uv_unique_adcpos_inside_roi"] = (
        df["n_u_unique_adcpos_inside_roi"]
        + df["n_v_unique_adcpos_inside_roi"]
    )

    csv_path = OUTDIR / "fired_strip_count_summary_withROI.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved summary CSV: {csv_path}")

    # ============================================================
    # Save histograms: entries inside ROI
    # ============================================================

    make_hist(
        df["n_u_entries_inside_roi"],
        xlabel="Number of U strip entries inside ROI",
        title="U strip-entry count per event inside ROI",
        outname="hist_u_strip_entries_inside_roi",
    )

    make_hist(
        df["n_v_entries_inside_roi"],
        xlabel="Number of V strip entries inside ROI",
        title="V strip-entry count per event inside ROI",
        outname="hist_v_strip_entries_inside_roi",
    )

    make_hist(
        df["n_uv_entries_inside_roi"],
        xlabel="Number of U+V strip entries inside ROI",
        title="U+V strip-entry count per event inside ROI",
        outname="hist_uv_strip_entries_inside_roi",
    )

    make_overlay_hist(
        df["n_u_entries_inside_roi"],
        df["n_v_entries_inside_roi"],
        xlabel="Number of strip entries inside ROI",
        title="U and V strip entries inside ROI",
        outname="hist_overlay_u_v_strip_entries_inside_roi",
    )

    # ============================================================
    # Save histograms: unique strip IDs inside ROI
    # ============================================================

    make_hist(
        df["n_u_unique_inside_roi"],
        xlabel="Number of unique U strip IDs inside ROI",
        title="Unique U strip count per event inside ROI",
        outname="hist_u_unique_strip_ids_inside_roi",
    )

    make_hist(
        df["n_v_unique_inside_roi"],
        xlabel="Number of unique V strip IDs inside ROI",
        title="Unique V strip count per event inside ROI",
        outname="hist_v_unique_strip_ids_inside_roi",
    )

    make_hist(
        df["n_uv_unique_inside_roi"],
        xlabel="Number of unique U+V strip IDs inside ROI",
        title="Unique U+V strip count per event inside ROI",
        outname="hist_uv_unique_strip_ids_inside_roi",
    )

    # ============================================================
    # Save histograms: positive ADC inside ROI
    # ============================================================

    make_hist(
        df["n_u_adcpos_entries_inside_roi"],
        xlabel="Number of positive-ADC U strip entries inside ROI",
        title="Positive-ADC U strip-entry count per event inside ROI",
        outname="hist_u_adcpos_entries_inside_roi",
    )

    make_hist(
        df["n_v_adcpos_entries_inside_roi"],
        xlabel="Number of positive-ADC V strip entries inside ROI",
        title="Positive-ADC V strip-entry count per event inside ROI",
        outname="hist_v_adcpos_entries_inside_roi",
    )

    make_hist(
        df["n_uv_adcpos_entries_inside_roi"],
        xlabel="Number of positive-ADC U+V strip entries inside ROI",
        title="Positive-ADC U+V strip-entry count per event inside ROI",
        outname="hist_uv_adcpos_entries_inside_roi",
    )

    make_hist(
        df["n_u_unique_adcpos_inside_roi"],
        xlabel="Number of unique positive-ADC U strip IDs inside ROI",
        title="Unique positive-ADC U strip count per event inside ROI",
        outname="hist_u_unique_adcpos_strip_ids_inside_roi",
    )

    make_hist(
        df["n_v_unique_adcpos_inside_roi"],
        xlabel="Number of unique positive-ADC V strip IDs inside ROI",
        title="Unique positive-ADC V strip count per event inside ROI",
        outname="hist_v_unique_adcpos_strip_ids_inside_roi",
    )

    make_hist(
        df["n_uv_unique_adcpos_inside_roi"],
        xlabel="Number of unique positive-ADC U+V strip IDs inside ROI",
        title="Unique positive-ADC U+V strip count per event inside ROI",
        outname="hist_uv_unique_adcpos_strip_ids_inside_roi",
    )

    # ============================================================
    # Save ROI width histograms
    # ============================================================

    make_hist(
        df["roi_u_width"],
        xlabel="ROI width in U strips",
        title="ROI U-strip width",
        outname="hist_roi_u_width",
    )

    make_hist(
        df["roi_v_width"],
        xlabel="ROI width in V strips",
        title="ROI V-strip width",
        outname="hist_roi_v_width",
    )

    # ============================================================
    # Print largest events for debugging
    # ============================================================

    print_cols = [
        "event_id",
        "n_u_entries_inside_roi",
        "n_v_entries_inside_roi",
        "n_uv_entries_inside_roi",
        "n_u_unique_inside_roi",
        "n_v_unique_inside_roi",
        "n_uv_unique_inside_roi",
        "n_u_adcpos_entries_inside_roi",
        "n_v_adcpos_entries_inside_roi",
        "n_uv_adcpos_entries_inside_roi",
        "n_u_unique_adcpos_inside_roi",
        "n_v_unique_adcpos_inside_roi",
        "n_uv_unique_adcpos_inside_roi",
        "roi_ustrip_min",
        "roi_ustrip_max",
        "roi_vstrip_min",
        "roi_vstrip_max",
        "roi_u_width",
        "roi_v_width",
    ]

    print("\nLargest events by U strip entries inside ROI:")
    print(
        df.sort_values("n_u_entries_inside_roi", ascending=False)
        .head(20)[print_cols]
        .to_string(index=False)
    )

    print("\nLargest events by unique positive-ADC U strip IDs inside ROI:")
    print(
        df.sort_values("n_u_unique_adcpos_inside_roi", ascending=False)
        .head(20)[print_cols]
        .to_string(index=False)
    )

    print("\nLargest events by unique positive-ADC V strip IDs inside ROI:")
    print(
        df.sort_values("n_v_unique_adcpos_inside_roi", ascending=False)
        .head(20)[print_cols]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()