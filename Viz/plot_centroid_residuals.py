import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# INPUT FILES
# ============================================================

TRUTH_FILE = "../Scratch/Truth_info/groundtruth_withoutROIcut.txt"

ML_PRED_FILE = "../Scratch/ML/textfile_outputs/hit_centers_ML_withoutROIcut.txt"

CPP_PRED_FILE = "../Scratch/CPP/textfile_outputs/hit_centers_CPP_withoutROIcut.txt"

# ============================================================
# OUTPUT
# ============================================================

OUTDIR = "plots/withoutROIcut/centroid_residuals"

OUT_RESIDUAL_TABLE = os.path.join(
    OUTDIR,
    "centroid_residuals_best_candidate_ML_vs_CURRENT.txt"
)

OUT_U_PNG = os.path.join(
    OUTDIR,
    "centroid_residual_U_ML_vs_CURRENT.png"
)

OUT_U_PDF = os.path.join(
    OUTDIR,
    "centroid_residual_U_ML_vs_CURRENT.pdf"
)

OUT_V_PNG = os.path.join(
    OUTDIR,
    "centroid_residual_V_ML_vs_CURRENT.png"
)

OUT_V_PDF = os.path.join(
    OUTDIR,
    "centroid_residual_V_ML_vs_CURRENT.pdf"
)

OUT_R_PNG = os.path.join(
    OUTDIR,
    "centroid_residual_R_ML_vs_CURRENT.png"
)

OUT_R_PDF = os.path.join(
    OUTDIR,
    "centroid_residual_R_ML_vs_CURRENT.pdf"
)

OUT_SCATTER_PNG = os.path.join(
    OUTDIR,
    "centroid_residual_scatter_ML_vs_CURRENT.png"
)

OUT_SCATTER_PDF = os.path.join(
    OUTDIR,
    "centroid_residual_scatter_ML_vs_CURRENT.pdf"
)

OUT_STATS = os.path.join(
    OUTDIR,
    "centroid_residual_stats_ML_vs_CURRENT.txt"
)

# ============================================================
# SETTINGS
# ============================================================

# Residual histogram range in strip units.
RESIDUAL_RANGE = (-20, 20)
RESIDUAL_BIN_WIDTH = 0.5

# Radial residual histogram range.
R_RANGE = (0, 40)
R_BIN_WIDTH = 0.5

# Scatter plot axis limit.
SCATTER_ABS_LIMIT = 20

# If True, only keep events where the method has at least one prediction.
# If False, events with no predictions are naturally absent from residual table.
REQUIRE_PREDICTION = True

# ============================================================
# TRUTH FILE FORMAT
# ============================================================

TRUTH_COLS = [
    "event_id",
    "module_id",
    "strip_id",
    "adc0",
    "adc1",
    "adc2",
    "adc3",
    "adc4",
    "adc5",
]

# ============================================================
# LOAD TRUTH CENTROIDS
# ============================================================

def build_truth_centroids(truth_file: str) -> pd.DataFrame:
    """
    Builds one truth centroid per event.

    groundtruth.txt format:
        event_id module_id strip_id adc0 adc1 adc2 adc3 adc4 adc5

    module_id:
        0 = U strip
        1 = V strip

    Truth centroid:
        truth_U_centroid = mean(U truth strip IDs)
        truth_V_centroid = mean(V truth strip IDs)
    """

    if not os.path.exists(truth_file):
        raise FileNotFoundError(f"Could not find truth file: {truth_file}")

    truth_df = pd.read_csv(
        truth_file,
        sep=r"\s+",
        header=None,
        names=TRUTH_COLS,
        usecols=list(range(len(TRUTH_COLS))),
    )

    for c in ["event_id", "module_id", "strip_id"]:
        truth_df[c] = pd.to_numeric(
            truth_df[c],
            errors="coerce"
        ).fillna(-1).astype(int)

    rows = []

    for event_id, sub in truth_df.groupby("event_id"):

        u_strips = np.sort(
            sub.loc[sub["module_id"] == 0, "strip_id"].astype(float).unique()
        )

        v_strips = np.sort(
            sub.loc[sub["module_id"] == 1, "strip_id"].astype(float).unique()
        )

        if len(u_strips) == 0 or len(v_strips) == 0:
            continue

        rows.append({
            "event_id": int(event_id),

            "truth_U_centroid": float(np.mean(u_strips)),
            "truth_V_centroid": float(np.mean(v_strips)),

            "truth_U_min": float(np.min(u_strips)),
            "truth_U_max": float(np.max(u_strips)),
            "truth_V_min": float(np.min(v_strips)),
            "truth_V_max": float(np.max(v_strips)),

            "n_truth_U": int(len(u_strips)),
            "n_truth_V": int(len(v_strips)),
            "truth_hit_size_2d": int(len(u_strips) * len(v_strips)),
            "truth_hit_size_1d": int(len(u_strips) + len(v_strips)),
        })

    out = pd.DataFrame(rows)

    print("Built truth centroids:")
    print("  Number of truth events:", len(out))
    print(out.head())

    return out


# ============================================================
# LOAD ML PREDICTIONS
# ============================================================

def load_ml_predictions(filename: str) -> pd.DataFrame:
    """
    Expected ML file format:

        event_id blob_id cy cx iy ix y_strip x_strip area [real_area optional]

    Here:
        x_strip = U centroid
        y_strip = V centroid
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Could not find ML prediction file: {filename}")

    raw = pd.read_csv(
        filename,
        sep=r"\s+",
        header=None,
        comment="#",
    )

    if raw.shape[1] < 9:
        raise RuntimeError(
            f"ML file must have at least 9 columns, got {raw.shape[1]} columns."
        )

    base_cols = [
        "event_id",
        "object_id",
        "cy",
        "cx",
        "iy",
        "ix",
        "y_strip",
        "x_strip",
        "area",
    ]

    extra_cols = [f"extra_{i}" for i in range(raw.shape[1] - len(base_cols))]
    raw.columns = base_cols + extra_cols

    df = raw[["event_id", "object_id", "x_strip", "y_strip"]].copy()

    df["method"] = "ML"

    df["event_id"] = pd.to_numeric(
        df["event_id"],
        errors="coerce"
    ).fillna(-1).astype(int)

    df["object_id"] = pd.to_numeric(
        df["object_id"],
        errors="coerce"
    ).fillna(-1).astype(int)

    df["pred_U"] = pd.to_numeric(
        df["x_strip"],
        errors="coerce"
    )

    df["pred_V"] = pd.to_numeric(
        df["y_strip"],
        errors="coerce"
    )

    df = df.dropna(subset=["pred_U", "pred_V"])

    # Drop negative/fake predictions if present.
    df = df[(df["pred_U"] >= 0) & (df["pred_V"] >= 0)].copy()

    return df[["method", "event_id", "object_id", "pred_U", "pred_V"]]


# ============================================================
# LOAD CURRENT / CPP PREDICTIONS
# ============================================================

def load_cpp_predictions(filename: str) -> pd.DataFrame:
    """
    Supports either headered CPP file with columns like:

        Event_ID 2D_hit_ID Hit_center_U_strip_ID Hit_center_V_strip_ID

    or headerless four-column file:

        event_id hit_id u_center v_center

    Here:
        U center -> pred_U
        V center -> pred_V
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Could not find CPP prediction file: {filename}")

    # First try headered.
    try:
        df_header = pd.read_csv(filename, sep=r"\s+")

        cols = list(df_header.columns)

        expected_header_sets = [
            {
                "Event_ID",
                "2D_hit_ID",
                "Hit_center_U_strip_ID",
                "Hit_center_V_strip_ID",
            },
            {
                "event_id",
                "hit_id",
                "x_strip",
                "y_strip",
            },
        ]

        if expected_header_sets[0].issubset(set(cols)):
            df = df_header.rename(
                columns={
                    "Event_ID": "event_id",
                    "2D_hit_ID": "object_id",
                    "Hit_center_U_strip_ID": "pred_U",
                    "Hit_center_V_strip_ID": "pred_V",
                }
            )

        elif expected_header_sets[1].issubset(set(cols)):
            df = df_header.rename(
                columns={
                    "hit_id": "object_id",
                    "x_strip": "pred_U",
                    "y_strip": "pred_V",
                }
            )

        else:
            raise ValueError("Header format not recognized.")

    except Exception:
        # Fallback: headerless four-column file.
        df = pd.read_csv(
            filename,
            sep=r"\s+",
            header=None,
            names=[
                "event_id",
                "object_id",
                "pred_U",
                "pred_V",
            ],
            usecols=[0, 1, 2, 3],
        )

    df = df[["event_id", "object_id", "pred_U", "pred_V"]].copy()

    df["method"] = "CURRENT"

    df["event_id"] = pd.to_numeric(
        df["event_id"],
        errors="coerce"
    ).fillna(-1).astype(int)

    df["object_id"] = pd.to_numeric(
        df["object_id"],
        errors="coerce"
    ).fillna(-1).astype(int)

    df["pred_U"] = pd.to_numeric(
        df["pred_U"],
        errors="coerce"
    )

    df["pred_V"] = pd.to_numeric(
        df["pred_V"],
        errors="coerce"
    )

    df = df.dropna(subset=["pred_U", "pred_V"])

    # Drop negative/fake predictions if present.
    df = df[(df["pred_U"] >= 0) & (df["pred_V"] >= 0)].copy()

    return df[["method", "event_id", "object_id", "pred_U", "pred_V"]]


# ============================================================
# BUILD RESIDUALS
# ============================================================

def build_best_candidate_residuals(
    truth_df: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each event and method, select the predicted centroid closest
    to the truth centroid, then compute residuals.

    This gives one residual per truth event per method, if that method
    produced at least one prediction in the event.
    """

    merged = pred_df.merge(
        truth_df,
        on="event_id",
        how="inner",
    )

    merged["residual_U"] = merged["pred_U"] - merged["truth_U_centroid"]
    merged["residual_V"] = merged["pred_V"] - merged["truth_V_centroid"]

    merged["residual_R"] = np.sqrt(
        merged["residual_U"] ** 2 + merged["residual_V"] ** 2
    )

    merged["abs_residual_U"] = np.abs(merged["residual_U"])
    merged["abs_residual_V"] = np.abs(merged["residual_V"])

    # Select closest candidate per event and method.
    merged = merged.sort_values(
        ["method", "event_id", "residual_R", "object_id"]
    )

    best = (
        merged
        .groupby(["method", "event_id"], as_index=False)
        .first()
    )

    return best


# ============================================================
# STATS
# ============================================================

def compute_residual_stats(res_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for method, sub in res_df.groupby("method"):

        rows.append({
            "method": method,
            "n_events_with_prediction": int(len(sub)),

            "mean_residual_U": float(sub["residual_U"].mean()),
            "median_residual_U": float(sub["residual_U"].median()),
            "std_residual_U": float(sub["residual_U"].std(ddof=1)),
            "rms_residual_U": float(np.sqrt(np.mean(sub["residual_U"] ** 2))),

            "mean_residual_V": float(sub["residual_V"].mean()),
            "median_residual_V": float(sub["residual_V"].median()),
            "std_residual_V": float(sub["residual_V"].std(ddof=1)),
            "rms_residual_V": float(np.sqrt(np.mean(sub["residual_V"] ** 2))),

            "mean_residual_R": float(sub["residual_R"].mean()),
            "median_residual_R": float(sub["residual_R"].median()),
            "std_residual_R": float(sub["residual_R"].std(ddof=1)),
            "rms_residual_R": float(np.sqrt(np.mean(sub["residual_R"] ** 2))),

            "q68_residual_R": float(sub["residual_R"].quantile(0.68)),
            "q90_residual_R": float(sub["residual_R"].quantile(0.90)),
            "q95_residual_R": float(sub["residual_R"].quantile(0.95)),

            "fraction_within_1_strip_R": float((sub["residual_R"] <= 1.0).mean()),
            "fraction_within_2_strip_R": float((sub["residual_R"] <= 2.0).mean()),
            "fraction_within_5_strip_R": float((sub["residual_R"] <= 5.0).mean()),
        })

    return pd.DataFrame(rows)


# ============================================================
# PLOTTING HELPERS
# ============================================================

def plot_residual_histogram(
    res_df: pd.DataFrame,
    column: str,
    xlabel: str,
    title: str,
    out_png: str,
    out_pdf: str,
    value_range,
    bin_width: float,
):
    bins = np.arange(
        value_range[0],
        value_range[1] + bin_width,
        bin_width,
    )

    plt.figure(figsize=(9, 6))

    for method, sub in res_df.groupby("method"):

        values = sub[column].dropna().to_numpy()

        plt.hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=2,
            alpha=0.6,
            label=f"{method}   N={len(values)}",
        )

    plt.xlabel(xlabel)
    plt.ylabel("Number of events")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)

    print("Saved:")
    print("  PNG:", out_png)
    print("  PDF:", out_pdf)

    plt.show()


def plot_residual_scatter(
    res_df: pd.DataFrame,
    out_png: str,
    out_pdf: str,
    abs_limit: float,
):
    plt.figure(figsize=(7, 7))

    for method, sub in res_df.groupby("method"):

        plt.scatter(
            sub["residual_U"],
            sub["residual_V"],
            s=10,
            alpha=0.35,
            label=f"{method}   N={len(sub)}",
        )

    plt.axhline(0.0, linewidth=1)
    plt.axvline(0.0, linewidth=1)

    plt.xlim(-abs_limit, abs_limit)
    plt.ylim(-abs_limit, abs_limit)

    plt.xlabel(r"$U_{\mathrm{pred}} - U_{\mathrm{truth\ centroid}}$ [strips]")
    plt.ylabel(r"$V_{\mathrm{pred}} - V_{\mathrm{truth\ centroid}}$ [strips]")
    plt.title("Best-candidate centroid residuals")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)

    print("Saved:")
    print("  PNG:", out_png)
    print("  PDF:", out_pdf)

    plt.show()


# ============================================================
# MAIN
# ============================================================

os.makedirs(OUTDIR, exist_ok=True)

truth_centroids = build_truth_centroids(TRUTH_FILE)

ml_pred = load_ml_predictions(ML_PRED_FILE)
cpp_pred = load_cpp_predictions(CPP_PRED_FILE)

all_pred = pd.concat(
    [
        ml_pred,
        cpp_pred,
    ],
    ignore_index=True,
)

print()
print("Loaded predictions:")
print(all_pred.groupby("method").size())

residuals = build_best_candidate_residuals(
    truth_df=truth_centroids,
    pred_df=all_pred,
)

print()
print("Best-candidate residuals:")
print(residuals.head())

residuals.to_csv(
    OUT_RESIDUAL_TABLE,
    sep=" ",
    index=False,
    float_format="%.8f",
)

print()
print("Saved residual table:")
print("TXT:", OUT_RESIDUAL_TABLE)

stats_df = compute_residual_stats(residuals)

print()
print("Centroid residual stats:")
print(stats_df.to_string(index=False))

stats_df.to_csv(
    OUT_STATS,
    sep=" ",
    index=False,
    float_format="%.8f",
)

print()
print("Saved residual stats:")
print("TXT:", OUT_STATS)

# U residual
plot_residual_histogram(
    residuals,
    column="residual_U",
    xlabel=r"$U_{\mathrm{pred}} - U_{\mathrm{truth\ centroid}}$ [strips]",
    title="Centroid residual in U",
    out_png=OUT_U_PNG,
    out_pdf=OUT_U_PDF,
    value_range=RESIDUAL_RANGE,
    bin_width=RESIDUAL_BIN_WIDTH,
)

# V residual
plot_residual_histogram(
    residuals,
    column="residual_V",
    xlabel=r"$V_{\mathrm{pred}} - V_{\mathrm{truth\ centroid}}$ [strips]",
    title="Centroid residual in V",
    out_png=OUT_V_PNG,
    out_pdf=OUT_V_PDF,
    value_range=RESIDUAL_RANGE,
    bin_width=RESIDUAL_BIN_WIDTH,
)

# Radial residual
plot_residual_histogram(
    residuals,
    column="residual_R",
    xlabel=r"$\sqrt{\Delta U^2 + \Delta V^2}$ [strips]",
    title="2D centroid residual magnitude",
    out_png=OUT_R_PNG,
    out_pdf=OUT_R_PDF,
    value_range=R_RANGE,
    bin_width=R_BIN_WIDTH,
)

# 2D scatter
plot_residual_scatter(
    residuals,
    out_png=OUT_SCATTER_PNG,
    out_pdf=OUT_SCATTER_PDF,
    abs_limit=SCATTER_ABS_LIMIT,
)

print()
print("Finished centroid residual plots.")






