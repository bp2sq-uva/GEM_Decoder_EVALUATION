import os
import numpy as np
import pandas as pd

# ============================================================
# INPUT FILES
# ============================================================

ML_PRED_FILE = "Scratch/ML/textfile_outputs/hit_centers_ML.txt"

CPP_PRED_FILE = "Scratch/CPP/textfile_outputs/hit_centers_CPP.txt"
# Change this path to your actual CPP output file.
# Expected columns:
# Event_ID  2D_hit_ID  Hit_center_U_strip_ID  Hit_center_V_strip_ID

TRUTH_FILE = "Scratch/Truth_info/groundtruth.txt"

# ============================================================
# OUTPUT FILES
# ============================================================

OUTDIR = "Scratch/Eval_and_Viz"

ML_BLOB_COMPARE_OUTFILE = os.path.join(
    OUTDIR,
    "ML_predicted_blobs_compared_to_truth_STRIPGAP_MODEL.txt"
)

ML_EVENT_SUMMARY_OUTFILE = os.path.join(
    OUTDIR,
    "ML_event_level_prediction_summary_STRIPGAP_MODEL.txt"
)

ML_BLOB_BUDGET_SUMMARY_OUTFILE = os.path.join(
    OUTDIR,
    "ML_blob_budget_summary_upto100_STRIPGAP_MODEL.txt"
)

CPP_HIT_COMPARE_OUTFILE = os.path.join(
    OUTDIR,
    "CPP_hits_compared_to_truth.txt"
)

CPP_EVENT_SUMMARY_OUTFILE = os.path.join(
    OUTDIR,
    "CPP_event_level_prediction_summary.txt"
)

CPP_HIT_BUDGET_SUMMARY_OUTFILE = os.path.join(
    OUTDIR,
    "CPP_hit_budget_summary_upto100.txt"
)

COMBINED_BUDGET_SUMMARY_OUTFILE = os.path.join(
    OUTDIR,
    "ML_vs_CPP_budget_summary_upto100.txt"
)

# ============================================================
# COLUMNS
# ============================================================

ML_PRED_COLS_REQUIRED = [
    "event_id",
    "blob_id",
    "cy",
    "cx",
    "iy",
    "ix",
    "y_strip",
    "x_strip",
    "area",
]

ML_PRED_COLS_OPTIONAL = [
    "real_area",
]

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

CPP_COLS_CANONICAL = [
    "event_id",
    "hit_id",
    "x_strip",
    "y_strip",
]

# ============================================================
# LOAD TRUTH
# ============================================================

truth_df = pd.read_csv(
    TRUTH_FILE,
    sep=r"\s+",
    header=None,
    names=TRUTH_COLS,
    usecols=list(range(len(TRUTH_COLS)))
)

for c in ["event_id", "module_id", "strip_id"]:
    truth_df[c] = pd.to_numeric(
        truth_df[c],
        errors="coerce"
    ).fillna(-1).astype(int)

print("Truth rows:", len(truth_df))
print("\nTruth file head:")
print(truth_df.head())

# ============================================================
# TRUTH HELPERS
# ============================================================

def build_truth_info(truth_df: pd.DataFrame):
    """
    One row per truth event.

    module_id == 0 -> truth X/U strips
    module_id == 1 -> truth Y/V strips

    Since you said each event has only one good hit, all truth U strips
    and truth V strips in the event belong to the same physical hit.
    """

    rows = []

    for ev, sub in truth_df.groupby("event_id"):

        x_truth = np.sort(
            sub.loc[sub["module_id"] == 0, "strip_id"].astype(int).unique()
        )

        y_truth = np.sort(
            sub.loc[sub["module_id"] == 1, "strip_id"].astype(int).unique()
        )

        if len(x_truth) == 0 or len(y_truth) == 0:
            continue

        rows.append({
            "event_id": int(ev),
            "x_truth_min": int(x_truth.min()),
            "x_truth_max": int(x_truth.max()),
            "y_truth_min": int(y_truth.min()),
            "y_truth_max": int(y_truth.max()),
            "x_truth_strips": tuple(int(v) for v in x_truth),
            "y_truth_strips": tuple(int(v) for v in y_truth),
            "n_truth_x": int(len(x_truth)),
            "n_truth_y": int(len(y_truth)),
            "truth_area": int(len(x_truth) * len(y_truth)),
        })

    truth_info = pd.DataFrame(rows)

    all_truth_events = set(truth_df["event_id"].unique())
    valid_truth_events = set(truth_info["event_id"].unique())
    invalid_truth_events = sorted(all_truth_events - valid_truth_events)

    if len(invalid_truth_events) > 0:
        print(
            f"[warning] {len(invalid_truth_events)} truth events were dropped "
            "because they did not have both X and Y truth strips."
        )

    return truth_info


truth_info_df = build_truth_info(truth_df)

print("\nNumber of valid truth events:", len(truth_info_df))
print(truth_info_df.head())

# ============================================================
# GENERIC PREDICTION COMPARISON
# ============================================================

def compare_centers_to_truth(
    pred_df: pd.DataFrame,
    truth_info_df: pd.DataFrame,
    object_id_col: str,
    source_name: str,
):
    """
    Generic comparison for either ML blobs or CPP 2D hits.

    Required prediction columns:
      event_id
      object_id_col, e.g. blob_id or hit_id
      x_strip
      y_strip

    Correctness definitions:
      exact:
        x_strip is exactly one of the truth U/X strips
        and y_strip is exactly one of the truth V/Y strips.

      range:
        x_strip and y_strip are inside truth min/max ranges.
        This is printed for diagnostics, but exact is the main metric.
    """

    pred_df = pred_df.copy()

    pred_df["event_id"] = pred_df["event_id"].astype(int)
    pred_df[object_id_col] = pred_df[object_id_col].astype(int)

    pred_df["x_strip"] = pd.to_numeric(
        pred_df["x_strip"],
        errors="coerce"
    )

    pred_df["y_strip"] = pd.to_numeric(
        pred_df["y_strip"],
        errors="coerce"
    )

    pred_df = pred_df.dropna(subset=["x_strip", "y_strip"]).copy()

    # Round CPP centers like 142.5 to nearest integer only for membership checks.
    # The original floating centers are kept separately.
    pred_df["x_strip_raw"] = pred_df["x_strip"]
    pred_df["y_strip_raw"] = pred_df["y_strip"]

    pred_df["x_strip"] = np.rint(pred_df["x_strip"]).astype(int)
    pred_df["y_strip"] = np.rint(pred_df["y_strip"]).astype(int)

    fake_mask = (
        (pred_df["x_strip"] < 0) |
        (pred_df["y_strip"] < 0)
    )

    n_fake = int(fake_mask.sum())

    if n_fake > 0:
        print(
            f"[warning] {source_name}: {n_fake} prediction(s) had fake/negative "
            "strip IDs and will be removed."
        )

        print(
            pred_df.loc[
                fake_mask,
                ["event_id", object_id_col, "x_strip", "y_strip"]
            ].to_string(index=False)
        )

    pred_df = pred_df.loc[~fake_mask].copy()

    compared = pred_df.merge(
        truth_info_df,
        on="event_id",
        how="inner"
    )

    compared["x_center_inside_truth_range"] = (
        (compared["x_strip"] >= compared["x_truth_min"]) &
        (compared["x_strip"] <= compared["x_truth_max"])
    )

    compared["y_center_inside_truth_range"] = (
        (compared["y_strip"] >= compared["y_truth_min"]) &
        (compared["y_strip"] <= compared["y_truth_max"])
    )

    compared["prediction_correct_range"] = (
        compared["x_center_inside_truth_range"] &
        compared["y_center_inside_truth_range"]
    )

    compared["x_center_exact_truth"] = [
        int(x) in set(xs)
        for x, xs in zip(compared["x_strip"], compared["x_truth_strips"])
    ]

    compared["y_center_exact_truth"] = [
        int(y) in set(ys)
        for y, ys in zip(compared["y_strip"], compared["y_truth_strips"])
    ]

    compared["prediction_correct_exact"] = (
        compared["x_center_exact_truth"] &
        compared["y_center_exact_truth"]
    )

    compared["source"] = source_name

    return compared


def build_event_level_summary(
    compared_df: pd.DataFrame,
    truth_info_df: pd.DataFrame,
    object_id_col: str,
    source_name: str,
):
    """
    One row per truth-hit event.

    Since you do not care about no-hit events, denominator is all valid
    events in groundtruth.txt.
    """

    pred_event_summary = (
        compared_df
        .groupby("event_id")
        .agg(
            n_predictions=(object_id_col, "count"),
            n_good_predictions_exact=("prediction_correct_exact", "sum"),
            n_good_predictions_range=("prediction_correct_range", "sum"),
            any_good_prediction_exact=("prediction_correct_exact", "any"),
            any_good_prediction_range=("prediction_correct_range", "any"),
        )
        .reset_index()
    )

    event_summary = truth_info_df[["event_id"]].merge(
        pred_event_summary,
        on="event_id",
        how="left"
    )

    fill_int_cols = [
        "n_predictions",
        "n_good_predictions_exact",
        "n_good_predictions_range",
    ]

    fill_bool_cols = [
        "any_good_prediction_exact",
        "any_good_prediction_range",
    ]

    for c in fill_int_cols:
        event_summary[c] = event_summary[c].fillna(0).astype(int)

    for c in fill_bool_cols:
        event_summary[c] = event_summary[c].fillna(False).astype(bool)

    event_summary["source"] = source_name

    return event_summary


def compute_budget_summary(
    event_summary_df: pd.DataFrame,
    source_name: str,
    max_predictions: int = 16,
):
    """
    For N = 1..max_predictions:

      success_exact_upto_N =
          n_predictions <= N and any_good_prediction_exact

      success_range_upto_N =
          n_predictions <= N and any_good_prediction_range

    Denominator = all valid truth-hit events.
    """

    total_hit_events = len(event_summary_df)

    rows = []

    for n in range(1, max_predictions + 1):

        success_exact_mask = (
            (event_summary_df["n_predictions"] <= n) &
            (event_summary_df["any_good_prediction_exact"])
        )

        success_range_mask = (
            (event_summary_df["n_predictions"] <= n) &
            (event_summary_df["any_good_prediction_range"])
        )

        success_exact_count = int(success_exact_mask.sum())
        success_range_count = int(success_range_mask.sum())

        success_exact_rate = (
            success_exact_count / total_hit_events
            if total_hit_events > 0 else np.nan
        )

        success_range_rate = (
            success_range_count / total_hit_events
            if total_hit_events > 0 else np.nan
        )

        rows.append({
            "source": source_name,
            "max_predictions_allowed": n,
            "success_exact_count": success_exact_count,
            "success_range_count": success_range_count,
            "total_truth_hit_events": total_hit_events,
            "success_exact_rate": success_exact_rate,
            "success_range_rate": success_range_rate,
        })

    return pd.DataFrame(rows)


def print_budget_summary(budget_df: pd.DataFrame, source_name: str):
    print()
    print("=" * 70)
    print(f"{source_name} centroid-strip hit accuracy under prediction budget")
    print("Using EXACT truth-strip membership as the main number.")
    print("=" * 70)

    for _, row in budget_df.iterrows():

        n = int(row["max_predictions_allowed"])

        exact_rate = float(row["success_exact_rate"])
        exact_count = int(row["success_exact_count"])

        range_rate = float(row["success_range_rate"])
        range_count = int(row["success_range_count"])

        total = int(row["total_truth_hit_events"])

        print(
            f"<= {n:2d} predictions : "
            f"exact={exact_rate:.4f} ({exact_count}/{total})   "
            f"range={range_rate:.4f} ({range_count}/{total})"
        )

# ============================================================
# LOAD ML PREDICTIONS
# ============================================================

ml_pred_df = pd.read_csv(
    ML_PRED_FILE,
    sep=r"\s+"
)

for c in ML_PRED_COLS_REQUIRED:
    if c not in ml_pred_df.columns:
        raise RuntimeError(f"Missing required ML prediction column: {c}")

for c in ML_PRED_COLS_OPTIONAL:
    if c not in ml_pred_df.columns:
        ml_pred_df[c] = np.nan

ml_pred_df = ml_pred_df[
    ML_PRED_COLS_REQUIRED + ML_PRED_COLS_OPTIONAL
].copy()

print("\nML prediction rows:", len(ml_pred_df))
print("\nML prediction file head:")
print(ml_pred_df.head())

# ============================================================
# LOAD CPP PREDICTIONS
# ============================================================

def load_cpp_predictions(cpp_file: str):
    """
    Loads CPP 2D hit-center output.

    Supports either:
      1. Headered file with columns:
         Event_ID 2D_hit_ID Hit_center_U_strip_ID Hit_center_V_strip_ID

      2. Headerless file with four columns:
         event_id hit_id u_center v_center

    Converts to canonical columns:
      event_id hit_id x_strip y_strip

    Here:
      x_strip = U strip center
      y_strip = V strip center
    """

    # Try headered read first.
    df_try = pd.read_csv(
        cpp_file,
        sep=r"\s+"
    )

    expected_header_cols = {
        "Event_ID",
        "2D_hit_ID",
        "Hit_center_U_strip_ID",
        "Hit_center_V_strip_ID",
    }

    if expected_header_cols.issubset(set(df_try.columns)):

        cpp_df = df_try.rename(
            columns={
                "Event_ID": "event_id",
                "2D_hit_ID": "hit_id",
                "Hit_center_U_strip_ID": "x_strip",
                "Hit_center_V_strip_ID": "y_strip",
            }
        )[CPP_COLS_CANONICAL].copy()

    else:
        # Fall back to headerless 4-column format.
        cpp_df = pd.read_csv(
            cpp_file,
            sep=r"\s+",
            header=None,
            names=CPP_COLS_CANONICAL,
            usecols=list(range(len(CPP_COLS_CANONICAL)))
        )

    for c in CPP_COLS_CANONICAL:
        cpp_df[c] = pd.to_numeric(cpp_df[c], errors="coerce")

    cpp_df = cpp_df.dropna(subset=CPP_COLS_CANONICAL).copy()

    cpp_df["event_id"] = cpp_df["event_id"].astype(int)
    cpp_df["hit_id"] = cpp_df["hit_id"].astype(int)

    return cpp_df


cpp_pred_df = load_cpp_predictions(CPP_PRED_FILE)

print("\nCPP prediction rows:", len(cpp_pred_df))
print("\nCPP prediction file head:")
print(cpp_pred_df.head())

# ============================================================
# COMPARE ML TO TRUTH
# ============================================================

ml_compare_df = compare_centers_to_truth(
    pred_df=ml_pred_df,
    truth_info_df=truth_info_df,
    object_id_col="blob_id",
    source_name="ML",
)

ml_event_summary_df = build_event_level_summary(
    compared_df=ml_compare_df,
    truth_info_df=truth_info_df,
    object_id_col="blob_id",
    source_name="ML",
)

ml_budget_summary_df = compute_budget_summary(
    event_summary_df=ml_event_summary_df,
    source_name="ML",
    max_predictions=100,
)

print("\nML predicted blobs with matching truth events:", len(ml_compare_df))
print(
    "ML correct predicted blobs, exact:",
    int(ml_compare_df["prediction_correct_exact"].sum())
)
print(
    "ML correct predicted blobs, range:",
    int(ml_compare_df["prediction_correct_range"].sum())
)
print(
    "ML wrong predicted blobs, exact:",
    int((~ml_compare_df["prediction_correct_exact"]).sum())
)

print(
    "ML truth events with at least one good predicted blob, exact:",
    int(ml_event_summary_df["any_good_prediction_exact"].sum())
)
print(
    "ML truth events with zero predicted blobs:",
    int((ml_event_summary_df["n_predictions"] == 0).sum())
)

print_budget_summary(ml_budget_summary_df, "ML")

# ============================================================
# COMPARE CPP TO TRUTH
# ============================================================

cpp_compare_df = compare_centers_to_truth(
    pred_df=cpp_pred_df,
    truth_info_df=truth_info_df,
    object_id_col="hit_id",
    source_name="CPP",
)

cpp_event_summary_df = build_event_level_summary(
    compared_df=cpp_compare_df,
    truth_info_df=truth_info_df,
    object_id_col="hit_id",
    source_name="CPP",
)

cpp_budget_summary_df = compute_budget_summary(
    event_summary_df=cpp_event_summary_df,
    source_name="CPP",
    max_predictions=100,
)

print("\nCPP 2D hits with matching truth events:", len(cpp_compare_df))
print(
    "CPP correct 2D hits, exact:",
    int(cpp_compare_df["prediction_correct_exact"].sum())
)
print(
    "CPP correct 2D hits, range:",
    int(cpp_compare_df["prediction_correct_range"].sum())
)
print(
    "CPP wrong 2D hits, exact:",
    int((~cpp_compare_df["prediction_correct_exact"]).sum())
)

print(
    "CPP truth events with at least one good 2D hit, exact:",
    int(cpp_event_summary_df["any_good_prediction_exact"].sum())
)
print(
    "CPP truth events with zero 2D hits:",
    int((cpp_event_summary_df["n_predictions"] == 0).sum())
)

print_budget_summary(cpp_budget_summary_df, "CPP")

# ============================================================
# COMBINED SUMMARY
# ============================================================

combined_budget_summary_df = pd.concat(
    [
        ml_budget_summary_df,
        cpp_budget_summary_df,
    ],
    ignore_index=True
)

print()
print("=" * 70)
print("ML vs CPP exact success-rate comparison")
print("=" * 70)

combined_pivot = combined_budget_summary_df.pivot(
    index="max_predictions_allowed",
    columns="source",
    values="success_exact_rate"
).reset_index()

print(combined_pivot.to_string(index=False))

# ============================================================
# SAVE OUTPUTS
# ============================================================

os.makedirs(OUTDIR, exist_ok=True)

ml_compare_df.to_csv(
    ML_BLOB_COMPARE_OUTFILE,
    sep=" ",
    index=False,
    float_format="%.6f"
)

ml_event_summary_df.to_csv(
    ML_EVENT_SUMMARY_OUTFILE,
    sep=" ",
    index=False,
    float_format="%.6f"
)

ml_budget_summary_df.to_csv(
    ML_BLOB_BUDGET_SUMMARY_OUTFILE,
    sep=" ",
    index=False,
    float_format="%.6f"
)

cpp_compare_df.to_csv(
    CPP_HIT_COMPARE_OUTFILE,
    sep=" ",
    index=False,
    float_format="%.6f"
)

cpp_event_summary_df.to_csv(
    CPP_EVENT_SUMMARY_OUTFILE,
    sep=" ",
    index=False,
    float_format="%.6f"
)

cpp_budget_summary_df.to_csv(
    CPP_HIT_BUDGET_SUMMARY_OUTFILE,
    sep=" ",
    index=False,
    float_format="%.6f"
)

combined_budget_summary_df.to_csv(
    COMBINED_BUDGET_SUMMARY_OUTFILE,
    sep=" ",
    index=False,
    float_format="%.6f"
)

print()
print("Saved outputs:")
print("ML blob comparison:        ", ML_BLOB_COMPARE_OUTFILE)
print("ML event summary:          ", ML_EVENT_SUMMARY_OUTFILE)
print("ML budget summary:         ", ML_BLOB_BUDGET_SUMMARY_OUTFILE)
print("CPP hit comparison:        ", CPP_HIT_COMPARE_OUTFILE)
print("CPP event summary:         ", CPP_EVENT_SUMMARY_OUTFILE)
print("CPP budget summary:        ", CPP_HIT_BUDGET_SUMMARY_OUTFILE)
print("Combined ML vs CPP summary:", COMBINED_BUDGET_SUMMARY_OUTFILE)