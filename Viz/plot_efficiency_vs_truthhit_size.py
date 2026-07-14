import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# INPUT FILES
# ============================================================

TRUTH_FILE = "../Scratch/Truth_info/groundtruth_withoutROIcut.txt"

EVENT_SUMMARY_FILES = {
    "ML": "../Scratch/Eval_and_Viz/withoutROIcut/ML_event_level_prediction_summary_STRIPGAP_MODEL.txt",
    "CURRENT": "../Scratch/Eval_and_Viz/withoutROIcut/CPP_event_level_prediction_summary.txt",
}

# ============================================================
# OUTPUT FILES
# ============================================================

OUTDIR = "plots/withoutROIcut/efficiency_vs_binned_truth_hit_size"

OUT_PNG = os.path.join(
    OUTDIR,
    "efficiency_vs_binned_truth_hit_size_ML_vs_CURRENT.png"
)

OUT_PDF = os.path.join(
    OUTDIR,
    "efficiency_vs_binned_truth_hit_size_ML_vs_CURRENT.pdf"
)

OUT_TXT = os.path.join(
    OUTDIR,
    "efficiency_vs_binned_truth_hit_size_ML_vs_CURRENT.txt"
)

# ============================================================
# SETTINGS
# ============================================================

# Truth hit size definition:
#   "2d_area"      -> N_U * N_V
#   "total_strips" -> N_U + N_V
#   "max_width"    -> max(N_U, N_V)
TRUTH_SIZE_DEFINITION = "2d_area"

# Exact event-level success.
SUCCESS_COLUMN = "any_good_prediction_exact"

# Truth-hit-size bins.
# For 2d_area = N_U * N_V, these are good starting bins.
TRUTH_SIZE_BIN_EDGES = [
    0,
    1,
    2,
    4,
    6,
    9,
    12,
    16,
    25,
    36,
    49,
    64,
    100,
    200,
    320,
    500,
    1000,
]

# Hide bins with too few truth events from the plot.
# They will still be saved in the output text file.
MIN_EVENTS_PER_BIN_FOR_PLOT = 1

PLOT_PERCENT = True

# If True, draw binomial uncertainty bars:
#   sqrt(eff * (1 - eff) / N)
SHOW_ERROR_BARS = True

# ============================================================
# COLUMNS
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
# HELPERS
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


def size_column_from_definition(definition: str) -> str:
    if definition == "2d_area":
        return "truth_hit_size_2d"
    if definition == "total_strips":
        return "truth_hit_size_1d"
    if definition == "max_width":
        return "truth_hit_size_max_width"

    raise ValueError(
        "TRUTH_SIZE_DEFINITION must be one of: "
        "'2d_area', 'total_strips', 'max_width'"
    )


def x_label_from_definition(definition: str) -> str:
    if definition == "2d_area":
        return r"Truth hit size bin, $N_U \times N_V$"
    if definition == "total_strips":
        return r"Truth hit size bin, $N_U + N_V$"
    if definition == "max_width":
        return r"Truth hit size bin, $\max(N_U, N_V)$"

    return "Truth hit size bin"


def build_truth_size_df(truth_file: str) -> pd.DataFrame:
    """
    Reads groundtruth.txt and computes one truth-hit-size value per event.

    module_id = 0 -> U truth strip
    module_id = 1 -> V truth strip

    Since you have one good truth hit per event:
        N_U = number of truth U strips in event
        N_V = number of truth V strips in event
        truth_hit_size_2d = N_U * N_V
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

    for ev, sub in truth_df.groupby("event_id"):

        u_strips = np.sort(
            sub.loc[sub["module_id"] == 0, "strip_id"].astype(int).unique()
        )

        v_strips = np.sort(
            sub.loc[sub["module_id"] == 1, "strip_id"].astype(int).unique()
        )

        if len(u_strips) == 0 or len(v_strips) == 0:
            continue

        n_u = int(len(u_strips))
        n_v = int(len(v_strips))

        rows.append({
            "event_id": int(ev),
            "n_truth_U": n_u,
            "n_truth_V": n_v,
            "truth_hit_size_2d": int(n_u * n_v),
            "truth_hit_size_1d": int(n_u + n_v),
            "truth_hit_size_max_width": int(max(n_u, n_v)),
        })

    truth_size_df = pd.DataFrame(rows)

    size_col = size_column_from_definition(TRUTH_SIZE_DEFINITION)

    print("Maximum truth size:", truth_size_df[size_col].max())
    print(truth_size_df[size_col].describe())

    print("Valid truth-hit events:", len(truth_size_df))
    print(truth_size_df.head())

    return truth_size_df


def load_event_summary(method_name: str, filename: str) -> pd.DataFrame:
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Could not find event summary for {method_name}: {filename}"
        )

    df = pd.read_csv(filename, sep=r"\s+")

    required_cols = [
        "event_id",
        SUCCESS_COLUMN,
    ]

    for c in required_cols:
        if c not in df.columns:
            raise RuntimeError(
                f"{filename} is missing required column: {c}\n"
                f"Available columns: {list(df.columns)}"
            )

    df = df.copy()
    df["method"] = method_name

    df["event_id"] = pd.to_numeric(
        df["event_id"],
        errors="coerce"
    ).fillna(-1).astype(int)

    df[SUCCESS_COLUMN] = parse_bool_column(df[SUCCESS_COLUMN])

    if "n_predictions" not in df.columns:
        df["n_predictions"] = 0
    else:
        df["n_predictions"] = pd.to_numeric(
            df["n_predictions"],
            errors="coerce"
        ).fillna(0).astype(int)

    return df


def build_binned_efficiency(
    truth_size_df: pd.DataFrame,
    event_summary_files: dict,
    bin_edges: list,
    truth_size_definition: str,
) -> pd.DataFrame:

    size_col = size_column_from_definition(truth_size_definition)

    all_rows = []

    for method_name, filename in event_summary_files.items():

        event_df = load_event_summary(method_name, filename)

        merged = truth_size_df.merge(
            event_df[["event_id", SUCCESS_COLUMN, "n_predictions", "method"]],
            on="event_id",
            how="inner",
        )

        merged["truth_size_bin"] = pd.cut(
            merged[size_col],
            bins=bin_edges,
            include_lowest=True,
            right=True,
        )

        grouped = merged.groupby("truth_size_bin", observed=True)

        for interval, sub in grouped:

            n_events = int(len(sub))
            n_success = int(sub[SUCCESS_COLUMN].sum())

            if n_events > 0:
                efficiency = n_success / n_events
                efficiency_err = np.sqrt(efficiency * (1.0 - efficiency) / n_events)
            else:
                efficiency = np.nan
                efficiency_err = np.nan

            bin_left = float(interval.left)
            bin_right = float(interval.right)
            bin_center = 0.5 * (bin_left + bin_right)

            all_rows.append({
                "method": method_name,
                "truth_size_definition": truth_size_definition,
                "truth_size_bin": str(interval),
                "truth_size_bin_label": f"{bin_left:g}-{bin_right:g}",
                "truth_size_bin_left": bin_left,
                "truth_size_bin_right": bin_right,
                "truth_size_bin_center": bin_center,
                "n_truth_events": n_events,
                "n_success": n_success,
                "efficiency": efficiency,
                "efficiency_err": efficiency_err,
                "mean_truth_size": float(sub[size_col].mean()),
                "median_truth_size": float(sub[size_col].median()),
                "mean_n_truth_U": float(sub["n_truth_U"].mean()),
                "mean_n_truth_V": float(sub["n_truth_V"].mean()),
                "mean_predictions_per_event": float(sub["n_predictions"].mean()),
                "median_predictions_per_event": float(sub["n_predictions"].median()),
            })

    return pd.DataFrame(all_rows)


# ============================================================
# MAIN
# ============================================================

os.makedirs(OUTDIR, exist_ok=True)

truth_size_df = build_truth_size_df(TRUTH_FILE)

binned_eff_df = build_binned_efficiency(
    truth_size_df=truth_size_df,
    event_summary_files=EVENT_SUMMARY_FILES,
    bin_edges=TRUTH_SIZE_BIN_EDGES,
    truth_size_definition=TRUTH_SIZE_DEFINITION,
)

print(
    binned_eff_df[
        ["truth_size_bin_label", "n_truth_events", "method"]
    ]
)



binned_eff_df = binned_eff_df.sort_values(
    ["method", "truth_size_bin_left"]
).reset_index(drop=True)

print()
print("Binned efficiency vs truth hit size:")
print(binned_eff_df.to_string(index=False))

binned_eff_df.to_csv(
    OUT_TXT,
    sep=" ",
    index=False,
    float_format="%.8f",
)

print()
print("Saved binned efficiency table:")
print("TXT:", OUT_TXT)

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(10, 6))

for method_name, sub in binned_eff_df.groupby("method"):

    sub = sub.sort_values("truth_size_bin_left")
    sub_plot = sub[sub["n_truth_events"] >= MIN_EVENTS_PER_BIN_FOR_PLOT].copy()

    x = sub_plot["truth_size_bin_center"].to_numpy()
    y = sub_plot["efficiency"].to_numpy()
    yerr = sub_plot["efficiency_err"].to_numpy()

    if PLOT_PERCENT:
        y = 100.0 * y
        yerr = 100.0 * yerr

    if SHOW_ERROR_BARS:
        plt.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=2,
            markersize=5,
            capsize=3,
            label=method_name,
        )
    else:
        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            markersize=5,
            label=method_name,
        )

plt.xlabel(x_label_from_definition(TRUTH_SIZE_DEFINITION))

if PLOT_PERCENT:
    plt.ylabel("Efficiency [%]")
else:
    plt.ylabel("Efficiency")

plt.title(
    "Efficiency vs binned truth-hit size\n"
    f"Only bins with at least {MIN_EVENTS_PER_BIN_FOR_PLOT} truth events are shown"
)

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(OUT_PNG, dpi=300)
plt.savefig(OUT_PDF)

print()
print("Saved plots:")
print("PNG:", OUT_PNG)
print("PDF:", OUT_PDF)

plt.show()