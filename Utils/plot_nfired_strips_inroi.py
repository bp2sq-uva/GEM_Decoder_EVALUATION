import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot

# ============================================================
# INPUT
# ============================================================

ROOT_FILE = "/volatile/halla/sbs/bhasitha/Tracking_ML/GEM_Decoder_EVALUATION/filtered_replayed_withoutROIcut.root"
TREE_NAME = "T"

PREFIX = "sbs.gemFT.m0"

# ============================================================
# OUTPUT
# ============================================================

OUTDIR = "plots/withoutROIcut"

OUT_SUMMARY = os.path.join(
    OUTDIR,
    "nfired_strips_inroi_summary.txt"
)

OUT_U_PNG = os.path.join(
    OUTDIR,
    "hist_nfired_ustrips_inroi.png"
)

OUT_U_PDF = os.path.join(
    OUTDIR,
    "hist_nfired_ustrips_inroi.pdf"
)

OUT_V_PNG = os.path.join(
    OUTDIR,
    "hist_nfired_vstrips_inroi.png"
)

OUT_V_PDF = os.path.join(
    OUTDIR,
    "hist_nfired_vstrips_inroi.pdf"
)

OUT_UV_PNG = os.path.join(
    OUTDIR,
    "hist_nfired_ustrips_and_vstrips_inroi.png"
)

OUT_UV_PDF = os.path.join(
    OUTDIR,
    "hist_nfired_ustrips_and_vstrips_inroi.pdf"
)

OUT_SCATTER_PNG = os.path.join(
    OUTDIR,
    "scatter_nfired_ustrips_vs_vstrips_inroi.png"
)

OUT_SCATTER_PDF = os.path.join(
    OUTDIR,
    "scatter_nfired_ustrips_vs_vstrips_inroi.pdf"
)

# ============================================================
# SETTINGS
# ============================================================

MAX_STRIPS_TO_PLOT = 500

BIN_WIDTH = 2

LOG_Y = False

SCATTER_ALPHA = 0.25
SCATTER_SIZE = 8

# ============================================================
# BRANCHES
# ============================================================

BRANCHES = [
    f"{PREFIX}.strip.nstripsfired",
    f"{PREFIX}.strip.istrip",
    f"{PREFIX}.strip.IsU",
    f"{PREFIX}.strip.IsV",
    f"{PREFIX}.roi.ustrip_min",
    f"{PREFIX}.roi.ustrip_max",
    f"{PREFIX}.roi.vstrip_min",
    f"{PREFIX}.roi.vstrip_max",
]

# ============================================================
# LOAD AND COMPUTE
# ============================================================

def compute_nfired_strips_inroi(root_file: str) -> pd.DataFrame:
    if not os.path.exists(root_file):
        raise FileNotFoundError(f"Could not find ROOT file: {root_file}")

    with uproot.open(root_file) as f:
        tree = f[TREE_NAME]

        arrays = tree.arrays(
            BRANCHES,
            library="ak",
        )

    nstrip_arr = arrays[f"{PREFIX}.strip.nstripsfired"]
    strip_id_arr = arrays[f"{PREFIX}.strip.istrip"]
    is_u_arr = arrays[f"{PREFIX}.strip.IsU"]
    is_v_arr = arrays[f"{PREFIX}.strip.IsV"]

    umin_arr = arrays[f"{PREFIX}.roi.ustrip_min"]
    umax_arr = arrays[f"{PREFIX}.roi.ustrip_max"]
    vmin_arr = arrays[f"{PREFIX}.roi.vstrip_min"]
    vmax_arr = arrays[f"{PREFIX}.roi.vstrip_max"]

    rows = []

    n_events = len(nstrip_arr)

    for ev in range(n_events):

        nstrip = int(nstrip_arr[ev])

        strip_ids = np.asarray(strip_id_arr[ev][:nstrip], dtype=float)
        is_u = np.asarray(is_u_arr[ev][:nstrip], dtype=float)
        is_v = np.asarray(is_v_arr[ev][:nstrip], dtype=float)

        umin = float(umin_arr[ev])
        umax = float(umax_arr[ev])
        vmin = float(vmin_arr[ev])
        vmax = float(vmax_arr[ev])

        nfired_ustrips_inroi = int(
            np.sum(
                (is_u == 1) &
                (strip_ids > umin) &
                (strip_ids < umax)
            )
        )

        nfired_vstrips_inroi = int(
            np.sum(
                (is_v == 1) &
                (strip_ids > vmin) &
                (strip_ids < vmax)
            )
        )

        rows.append({
            "event_id": ev,
            "nfired_ustrips_inroi": nfired_ustrips_inroi,
            "nfired_vstrips_inroi": nfired_vstrips_inroi,
            "roi_ustrip_min": umin,
            "roi_ustrip_max": umax,
            "roi_vstrip_min": vmin,
            "roi_vstrip_max": vmax,
            "roi_u_width": umax - umin,
            "roi_v_width": vmax - vmin,
        })

    return pd.DataFrame(rows)


# ============================================================
# PLOTTING HELPERS
# ============================================================

def print_stats(df: pd.DataFrame, column: str):
    values = df[column]

    print()
    print("=" * 70)
    print(column)
    print("=" * 70)
    print(values.describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.95, 0.99]))

    print()
    print("Number of events > 160:", int((values > 160).sum()))
    print("Fraction of events > 160:", float((values > 160).mean()))


def plot_single_hist(
    df: pd.DataFrame,
    column: str,
    xlabel: str,
    title: str,
    out_png: str,
    out_pdf: str,
):
    values = df[column].clip(upper=MAX_STRIPS_TO_PLOT)

    bins = np.arange(
        0,
        MAX_STRIPS_TO_PLOT + BIN_WIDTH,
        BIN_WIDTH,
    )

    plt.figure(figsize=(9, 6))

    plt.hist(
        values,
        bins=bins,
        histtype="step",
        linewidth=2,
    )

    plt.xlabel(xlabel)
    plt.ylabel("Number of events")

    if LOG_Y:
        plt.yscale("log")

    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)

    print()
    print("Saved:")
    print("PNG:", out_png)
    print("PDF:", out_pdf)

    plt.show()


def plot_overlay_hist(df: pd.DataFrame):
    bins = np.arange(
        0,
        MAX_STRIPS_TO_PLOT + BIN_WIDTH,
        BIN_WIDTH,
    )

    u_values = df["nfired_ustrips_inroi"].clip(upper=MAX_STRIPS_TO_PLOT)
    v_values = df["nfired_vstrips_inroi"].clip(upper=MAX_STRIPS_TO_PLOT)

    plt.figure(figsize=(9, 6))

    plt.hist(
        u_values,
        bins=bins,
        histtype="step",
        linewidth=2,
        label="U strips in ROI",
    )

    plt.hist(
        v_values,
        bins=bins,
        histtype="step",
        linewidth=2,
        label="V strips in ROI",
    )

    plt.xlabel("Number of fired strips in ROI")
    plt.ylabel("Number of events")

    if LOG_Y:
        plt.yscale("log")

    plt.title("Fired U and V strips in ROI")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUT_UV_PNG, dpi=300)
    plt.savefig(OUT_UV_PDF)

    print()
    print("Saved:")
    print("PNG:", OUT_UV_PNG)
    print("PDF:", OUT_UV_PDF)

    plt.show()


def plot_scatter(df: pd.DataFrame):
    plt.figure(figsize=(7, 7))

    plt.scatter(
        df["nfired_ustrips_inroi"],
        df["nfired_vstrips_inroi"],
        s=SCATTER_SIZE,
        alpha=SCATTER_ALPHA,
    )

    plt.axvline(160, linestyle="--", linewidth=1)
    plt.axhline(160, linestyle="--", linewidth=1)

    plt.xlabel("Number of fired U strips in ROI")
    plt.ylabel("Number of fired V strips in ROI")
    plt.title("Fired U vs V strips in ROI")

    plt.xlim(0, MAX_STRIPS_TO_PLOT)
    plt.ylim(0, MAX_STRIPS_TO_PLOT)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUT_SCATTER_PNG, dpi=300)
    plt.savefig(OUT_SCATTER_PDF)

    print()
    print("Saved:")
    print("PNG:", OUT_SCATTER_PNG)
    print("PDF:", OUT_SCATTER_PDF)

    plt.show()


# ============================================================
# MAIN
# ============================================================

os.makedirs(OUTDIR, exist_ok=True)

df = compute_nfired_strips_inroi(ROOT_FILE)

print()
print("Computed fired-strip counts in ROI:")
print(df.head())

print_stats(df, "nfired_ustrips_inroi")
print_stats(df, "nfired_vstrips_inroi")

df.to_csv(
    OUT_SUMMARY,
    sep=" ",
    index=False,
    float_format="%.8f",
)

print()
print("Saved summary table:")
print("TXT:", OUT_SUMMARY)

plot_single_hist(
    df,
    column="nfired_ustrips_inroi",
    xlabel="Number of fired U strips in ROI",
    title="Distribution of fired U strips in ROI",
    out_png=OUT_U_PNG,
    out_pdf=OUT_U_PDF,
)

plot_single_hist(
    df,
    column="nfired_vstrips_inroi",
    xlabel="Number of fired V strips in ROI",
    title="Distribution of fired V strips in ROI",
    out_png=OUT_V_PNG,
    out_pdf=OUT_V_PDF,
)

plot_overlay_hist(df)

plot_scatter(df)

print()
print("Finished plotting fired-strip ROI histograms.")