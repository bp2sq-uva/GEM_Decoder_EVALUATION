import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# INPUT FILES
# ============================================================

METHOD_FILES = {
    "ML": "../Scratch/Eval_and_Viz/ML_blob_budget_summary_upto100_STRIPGAP_MODEL.txt",
    "CURRENT": "../Scratch/Eval_and_Viz/CPP_hit_budget_summary_upto100.txt",
}

# ============================================================
# OUTPUT
# ============================================================

OUTDIR = "plots"

OUT_PNG = os.path.join(
    OUTDIR,
    "efficiency_vs_blob_count_ML_vs_CURRENT.png"
)

OUT_PDF = os.path.join(
    OUTDIR,
    "efficiency_vs_blob_count_ML_vs_CURRENT.pdf"
)

# ============================================================
# SETTINGS
# ============================================================

# Use exact efficiency as the main metric.
EFFICIENCY_COLUMN = "success_exact_rate"

# Optional diagnostic alternative:
# EFFICIENCY_COLUMN = "success_range_rate"

BLOB_COUNT_COLUMN = "max_predictions_allowed"

# If True, multiply efficiency by 100 and plot percent.
PLOT_PERCENT = True

# Set to None to plot full available range for each method.
# Example: 100 means show up to 100 blobs/hits.
MAX_BLOB_COUNT_TO_PLOT = 100

# ============================================================
# LOAD
# ============================================================

def load_budget_file(method_name: str, filename: str) -> pd.DataFrame:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Could not find file for {method_name}: {filename}")

    df = pd.read_csv(filename, sep=r"\s+")

    required_cols = [
        BLOB_COUNT_COLUMN,
        EFFICIENCY_COLUMN,
        "success_exact_count",
        "total_truth_hit_events",
    ]

    for c in required_cols:
        if c not in df.columns:
            raise RuntimeError(
                f"File {filename} is missing required column: {c}\n"
                f"Available columns: {list(df.columns)}"
            )

    df = df.copy()
    df["method"] = method_name

    df[BLOB_COUNT_COLUMN] = pd.to_numeric(
        df[BLOB_COUNT_COLUMN],
        errors="coerce"
    )

    df[EFFICIENCY_COLUMN] = pd.to_numeric(
        df[EFFICIENCY_COLUMN],
        errors="coerce"
    )

    df = df.dropna(subset=[BLOB_COUNT_COLUMN, EFFICIENCY_COLUMN])

    df[BLOB_COUNT_COLUMN] = df[BLOB_COUNT_COLUMN].astype(int)

    if MAX_BLOB_COUNT_TO_PLOT is not None:
        df = df[df[BLOB_COUNT_COLUMN] <= int(MAX_BLOB_COUNT_TO_PLOT)]

    return df


all_dfs = []

for method_name, filename in METHOD_FILES.items():
    df_method = load_budget_file(method_name, filename)
    all_dfs.append(df_method)

eff_df = pd.concat(all_dfs, ignore_index=True)

print("Loaded efficiency points:")
print(
    eff_df[
        [
            "method",
            BLOB_COUNT_COLUMN,
            EFFICIENCY_COLUMN,
            "success_exact_count",
            "total_truth_hit_events",
        ]
    ].head(20)
)

print("\nAvailable methods:")
print(eff_df["method"].unique())

# ============================================================
# PLOT
# ============================================================

os.makedirs(OUTDIR, exist_ok=True)

plt.figure(figsize=(9, 6))

for method_name, sub in eff_df.groupby("method"):
    sub = sub.sort_values(BLOB_COUNT_COLUMN)

    x = sub[BLOB_COUNT_COLUMN].to_numpy()
    y = sub[EFFICIENCY_COLUMN].to_numpy()

    if PLOT_PERCENT:
        y = 100.0 * y

    plt.plot(
        x,
        y,
        marker="o",
        linewidth=2,
        markersize=4,
        label=method_name,
    )

plt.xlabel("Maximum predicted blobs / hits allowed")

if PLOT_PERCENT:
    plt.ylabel("Efficiency [%]")
else:
    plt.ylabel("Efficiency")

metric_label = (
    "Exact truth-strip membership"
    if EFFICIENCY_COLUMN == "success_exact_rate"
    else "Truth-strip range membership"
)

plt.title(f"Efficiency vs blob/hit count\n{metric_label}")

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(OUT_PNG, dpi=300)
plt.savefig(OUT_PDF)

print("\nSaved plots:")
print("PNG:", OUT_PNG)
print("PDF:", OUT_PDF)

plt.show()

















# ============================================================
# HISTOGRAMS: NUMBER OF PREDICTED OBJECTS PER EVENT
# ============================================================

EVENT_SUMMARY_FILES = {
    "ML": "../Scratch/Eval_and_Viz/ML_event_level_prediction_summary_STRIPGAP_MODEL.txt",
    "CURRENT": "../Scratch/Eval_and_Viz/CPP_event_level_prediction_summary.txt",
}

HIST_OUT_PNG = os.path.join(
    OUTDIR,
    "hist_number_of_predictions_per_event_ML_vs_CURRENT.png"
)

HIST_OUT_PDF = os.path.join(
    OUTDIR,
    "hist_number_of_predictions_per_event_ML_vs_CURRENT.pdf"
)

# You can increase this if CPP has many hits per event.
MAX_HITS_IN_HIST = 100

def load_event_summary_file(method_name: str, filename: str) -> pd.DataFrame:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Could not find event summary for {method_name}: {filename}")

    df = pd.read_csv(filename, sep=r"\s+")

    if "n_predictions" not in df.columns:
        raise RuntimeError(
            f"File {filename} is missing required column: n_predictions\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df["method"] = method_name

    df["n_predictions"] = pd.to_numeric(
        df["n_predictions"],
        errors="coerce"
    ).fillna(0).astype(int)

    return df


event_summary_dfs = []

for method_name, filename in EVENT_SUMMARY_FILES.items():
    df_method = load_event_summary_file(method_name, filename)
    event_summary_dfs.append(df_method)

event_hist_df = pd.concat(event_summary_dfs, ignore_index=True)

print("\nPrediction-count summary per event:")
print(
    event_hist_df
    .groupby("method")["n_predictions"]
    .describe()
)

# ------------------------------------------------------------
# Overlay histogram
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

bins = range(0, MAX_HITS_IN_HIST + 2)

for method_name, sub in event_hist_df.groupby("method"):
    values = sub["n_predictions"].clip(upper=MAX_HITS_IN_HIST)

    plt.hist(
        values,
        bins=bins,
        histtype="step",
        linewidth=2,
        label=method_name,
    )

plt.xlabel("Number of predicted blobs / 2D hits per truth event")
plt.ylabel("Number of events")
plt.title("Distribution of predicted objects per event")

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(HIST_OUT_PNG, dpi=300)
plt.savefig(HIST_OUT_PDF)

print("\nSaved prediction-count histograms:")
print("PNG:", HIST_OUT_PNG)
print("PDF:", HIST_OUT_PDF)

plt.show()