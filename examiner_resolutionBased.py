import os
import numpy as np
import pandas as pd

# ============================================================
# INPUT FILES
# ============================================================

ML_PRED_FILE = "Scratch/ML/textfile_outputs/hit_centers_ML.txt"

CPP_PRED_FILE = "Scratch/CPP/textfile_outputs/hit_centers_CPP.txt"

TRUTH_FILE = "Scratch/Truth_info/groundtruth.txt"

# ============================================================
# OUTPUT FILES
# ============================================================

OUTDIR = "/work/halla/sbs/bhasitha/Tracking_ML/GEMDecoder_ML/GEM_Decoder_EVALUATION/Scratch/Eval_and_Viz"

MARGIN_STRIPS = 3

ML_COMPARE_OUTFILE = os.path.join(
    OUTDIR,
    f"ML_predicted_blobs_compared_to_truth_boundary_margin_{MARGIN_STRIPS}strips.txt"
)

ML_EVENT_SUMMARY_OUTFILE = os.path.join(
    OUTDIR,
    f"ML_event_level_prediction_summary_boundary_margin_{MARGIN_STRIPS}strips.txt"
)

ML_BUDGET_SUMMARY_OUTFILE = os.path.join(
    OUTDIR,
    f"ML_blob_budget_summary_upto100_boundary_margin_{MARGIN_STRIPS}strips.txt"
)

CPP_COMPARE_OUTFILE = os.path.join(
    OUTDIR,
    f"CPP_hits_compared_to_truth_boundary_margin_{MARGIN_STRIPS}strips.txt"
)

CPP_EVENT_SUMMARY_OUTFILE = os.path.join(
    OUTDIR,
    f"CPP_event_level_prediction_summary_boundary_margin_{MARGIN_STRIPS}strips.txt"
)

CPP_BUDGET_SUMMARY_OUTFILE = os.path.join(
    OUTDIR,
    f"CPP_hit_budget_summary_upto100_boundary_margin_{MARGIN_STRIPS}strips.txt"
)

COMBINED_BUDGET_SUMMARY_OUTFILE = os.path.join(
    OUTDIR,
    f"ML_vs_CPP_budget_summary_upto100_boundary_margin_{MARGIN_STRIPS}strips.txt"
)

UNLIMITED_EFFICIENCY_OUTFILE = os.path.join(
    OUTDIR,
    f"ML_vs_CPP_unlimited_efficiency_boundary_margin_{MARGIN_STRIPS}strips.txt"
)

PAIRED_OUTCOME_OUTFILE = os.path.join(
    OUTDIR,
    f"ML_vs_CPP_paired_event_outcomes_boundary_margin_{MARGIN_STRIPS}strips.txt"
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
    usecols=list(range(len(TRUTH_COLS))),
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

def build_truth_info(truth_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per truth event.

    module_id == 0 -> truth U strips
    module_id == 1 -> truth V strips

    Since each event has one good hit, all truth U strips and truth V strips
    in the event belong to the same physical truth hit.
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

            "x_truth_min_margin": int(x_truth.min()) - MARGIN_STRIPS,
            "x_truth_max_margin": int(x_truth.max()) + MARGIN_STRIPS,
            "y_truth_min_margin": int(y_truth.min()) - MARGIN_STRIPS,
            "y_truth_max_margin": int(y_truth.max()) + MARGIN_STRIPS,

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
            "because they did not have both U and V truth strips."
        )

    return truth_info


truth_info_df = build_truth_info(truth_df)

print("\nNumber of valid truth events:", len(truth_info_df))
print(truth_info_df.head())

# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_ml_predictions(filename: str) -> pd.DataFrame:
    """
    Expected ML format:

        event_id blob_id cy cx iy ix y_strip x_strip area [real_area optional]

    Here:
        x_strip = U center
        y_strip = V center
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Could not find ML prediction file: {filename}")

    df = pd.read_csv(filename, sep=r"\s+")

    for c in ML_PRED_COLS_REQUIRED:
        if c not in df.columns:
            raise RuntimeError(
                f"Missing required ML prediction column: {c}\n"
                f"Available columns: {list(df.columns)}"
            )

    for c in ML_PRED_COLS_OPTIONAL:
        if c not in df.columns:
            df[c] = np.nan

    df = df[ML_PRED_COLS_REQUIRED + ML_PRED_COLS_OPTIONAL].copy()

    print("\nML prediction rows:", len(df))
    print("\nML prediction file head:")
    print(df.head())

    return df


def load_cpp_predictions(cpp_file: str) -> pd.DataFrame:
    """
    Loads CPP 2D hit-center output.

    Supports either:

      1. Headered file:
         Event_ID 2D_hit_ID Hit_center_U_strip_ID Hit_center_V_strip_ID

      2. Headerless file:
         event_id hit_id u_center v_center

    Converts to canonical columns:
        event_id hit_id x_strip y_strip

    Here:
        x_strip = U center
        y_strip = V center
    """

    if not os.path.exists(cpp_file):
        raise FileNotFoundError(f"Could not find CPP prediction file: {cpp_file}")

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

        cpp_df = pd.read_csv(
            cpp_file,
            sep=r"\s+",
            header=None,
            names=CPP_COLS_CANONICAL,
            usecols=list(range(len(CPP_COLS_CANONICAL))),
        )

    for c in CPP_COLS_CANONICAL:
        cpp_df[c] = pd.to_numeric(cpp_df[c], errors="coerce")

    cpp_df = cpp_df.dropna(subset=CPP_COLS_CANONICAL).copy()

    cpp_df["event_id"] = cpp_df["event_id"].astype(int)
    cpp_df["hit_id"] = cpp_df["hit_id"].astype(int)

    print("\nCPP prediction rows:", len(cpp_df))
    print("\nCPP prediction file head:")
    print(cpp_df.head())

    return cpp_df


ml_pred_df = load_ml_predictions(ML_PRED_FILE)
cpp_pred_df = load_cpp_predictions(CPP_PRED_FILE)

# ============================================================
# COMPARISON
# ============================================================

def compare_centers_to_truth_with_boundary_margin(
    pred_df: pd.DataFrame,
    truth_info_df: pd.DataFrame,
    object_id_col: str,
    source_name: str,
    margin: int,
) -> pd.DataFrame:
    """
    Compares predicted centers to goodADC truth hit.

    Main new success definition:

        prediction_within_margin_boundary =
            x_truth_min - margin <= x_pred <= x_truth_max + margin
            and
            y_truth_min - margin <= y_pred <= y_truth_max + margin

    This includes predictions inside the original goodADC truth box.
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

    pred_df["x_strip_raw"] = pred_df["x_strip"]
    pred_df["y_strip_raw"] = pred_df["y_strip"]

    # Use rounded strip centers for strip-level boundary check.
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

    pred_df = pred_df.loc[~fake_mask].copy()

    compared = pred_df.merge(
        truth_info_df,
        on="event_id",
        how="inner",
    )

    # ------------------------------------------------------------
    # Original exact truth-strip membership
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Original inside truth min/max box
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # New 3-strip padded boundary-box check
    # ------------------------------------------------------------

    compared["x_center_inside_margin_boundary"] = (
        (compared["x_strip"] >= compared["x_truth_min"] - margin) &
        (compared["x_strip"] <= compared["x_truth_max"] + margin)
    )

    compared["y_center_inside_margin_boundary"] = (
        (compared["y_strip"] >= compared["y_truth_min"] - margin) &
        (compared["y_strip"] <= compared["y_truth_max"] + margin)
    )

    compared["prediction_within_margin_boundary"] = (
        compared["x_center_inside_margin_boundary"] &
        compared["y_center_inside_margin_boundary"]
    )

    # ------------------------------------------------------------
    # Diagnostic: distance outside original truth box
    # ------------------------------------------------------------
    # This is zero for predictions inside the original goodADC box.
    # If outside, it tells how many strips away the prediction is.

    compared["x_distance_outside_truth_box"] = np.where(
        compared["x_strip"] < compared["x_truth_min"],
        compared["x_truth_min"] - compared["x_strip"],
        np.where(
            compared["x_strip"] > compared["x_truth_max"],
            compared["x_strip"] - compared["x_truth_max"],
            0,
        )
    )

    compared["y_distance_outside_truth_box"] = np.where(
        compared["y_strip"] < compared["y_truth_min"],
        compared["y_truth_min"] - compared["y_strip"],
        np.where(
            compared["y_strip"] > compared["y_truth_max"],
            compared["y_strip"] - compared["y_truth_max"],
            0,
        )
    )

    compared["max_distance_outside_truth_box"] = np.maximum(
        compared["x_distance_outside_truth_box"],
        compared["y_distance_outside_truth_box"],
    )

    compared["euclidean_distance_outside_truth_box"] = np.sqrt(
        compared["x_distance_outside_truth_box"] ** 2 +
        compared["y_distance_outside_truth_box"] ** 2
    )

    compared["source"] = source_name
    compared["boundary_margin_strips"] = margin

    return compared


def build_event_level_summary(
    compared_df: pd.DataFrame,
    truth_info_df: pd.DataFrame,
    object_id_col: str,
    source_name: str,
) -> pd.DataFrame:
    """
    One row per truth-hit event.

    Denominator = all valid truth-hit events from groundtruth.txt.
    """

    pred_event_summary = (
        compared_df
        .groupby("event_id")
        .agg(
            n_predictions=(object_id_col, "count"),

            n_good_predictions_exact=("prediction_correct_exact", "sum"),
            n_good_predictions_range=("prediction_correct_range", "sum"),
            n_good_predictions_margin=("prediction_within_margin_boundary", "sum"),

            any_good_prediction_exact=("prediction_correct_exact", "any"),
            any_good_prediction_range=("prediction_correct_range", "any"),
            any_prediction_within_margin_boundary=(
                "prediction_within_margin_boundary",
                "any",
            ),

            min_max_distance_outside_truth_box=(
                "max_distance_outside_truth_box",
                "min",
            ),
            min_euclidean_distance_outside_truth_box=(
                "euclidean_distance_outside_truth_box",
                "min",
            ),
        )
        .reset_index()
    )

    event_summary = truth_info_df[[
        "event_id",
        "x_truth_min",
        "x_truth_max",
        "y_truth_min",
        "y_truth_max",
        "n_truth_x",
        "n_truth_y",
        "truth_area",
    ]].merge(
        pred_event_summary,
        on="event_id",
        how="left",
    )

    fill_int_cols = [
        "n_predictions",
        "n_good_predictions_exact",
        "n_good_predictions_range",
        "n_good_predictions_margin",
    ]

    fill_bool_cols = [
        "any_good_prediction_exact",
        "any_good_prediction_range",
        "any_prediction_within_margin_boundary",
    ]

    for c in fill_int_cols:
        event_summary[c] = event_summary[c].fillna(0).astype(int)

    for c in fill_bool_cols:
        event_summary[c] = event_summary[c].fillna(False).astype(bool)

    # If no prediction exists, distance is undefined.
    event_summary["min_max_distance_outside_truth_box"] = (
        event_summary["min_max_distance_outside_truth_box"].fillna(np.nan)
    )

    event_summary["min_euclidean_distance_outside_truth_box"] = (
        event_summary["min_euclidean_distance_outside_truth_box"].fillna(np.nan)
    )

    event_summary["source"] = source_name
    event_summary["boundary_margin_strips"] = MARGIN_STRIPS

    return event_summary


def compute_budget_summary(
    event_summary_df: pd.DataFrame,
    source_name: str,
    max_predictions: int = 100,
) -> pd.DataFrame:
    """
    For N = 1..max_predictions:

      success_margin_upto_N =
          n_predictions <= N and any_prediction_within_margin_boundary

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

        success_margin_mask = (
            (event_summary_df["n_predictions"] <= n) &
            (event_summary_df["any_prediction_within_margin_boundary"])
        )

        success_exact_count = int(success_exact_mask.sum())
        success_range_count = int(success_range_mask.sum())
        success_margin_count = int(success_margin_mask.sum())

        success_exact_rate = (
            success_exact_count / total_hit_events
            if total_hit_events > 0 else np.nan
        )

        success_range_rate = (
            success_range_count / total_hit_events
            if total_hit_events > 0 else np.nan
        )

        success_margin_rate = (
            success_margin_count / total_hit_events
            if total_hit_events > 0 else np.nan
        )

        rows.append({
            "source": source_name,
            "boundary_margin_strips": MARGIN_STRIPS,
            "max_predictions_allowed": n,

            "success_exact_count": success_exact_count,
            "success_range_count": success_range_count,
            "success_margin_count": success_margin_count,

            "total_truth_hit_events": total_hit_events,

            "success_exact_rate": success_exact_rate,
            "success_range_rate": success_range_rate,
            "success_margin_rate": success_margin_rate,
        })

    return pd.DataFrame(rows)


def compute_unlimited_efficiency(
    event_summary_df: pd.DataFrame,
    source_name: str,
) -> dict:
    """
    Unlimited-prediction efficiency.

    Main number here:

      success_margin =
          any_prediction_within_margin_boundary
    """

    total_hit_events = len(event_summary_df)

    exact_count = int(event_summary_df["any_good_prediction_exact"].sum())
    range_count = int(event_summary_df["any_good_prediction_range"].sum())
    margin_count = int(
        event_summary_df["any_prediction_within_margin_boundary"].sum()
    )

    exact_rate = (
        exact_count / total_hit_events
        if total_hit_events > 0 else np.nan
    )

    range_rate = (
        range_count / total_hit_events
        if total_hit_events > 0 else np.nan
    )

    margin_rate = (
        margin_count / total_hit_events
        if total_hit_events > 0 else np.nan
    )

    return {
        "source": source_name,
        "boundary_margin_strips": MARGIN_STRIPS,

        "success_exact_count": exact_count,
        "success_range_count": range_count,
        "success_margin_count": margin_count,

        "total_truth_hit_events": total_hit_events,

        "success_exact_rate": exact_rate,
        "success_range_rate": range_rate,
        "success_margin_rate": margin_rate,
    }


def print_budget_summary(
    budget_df: pd.DataFrame,
    source_name: str,
):
    print()
    print("=" * 80)
    print(f"{source_name} hit accuracy under prediction budget")
    print(f"Main number: prediction within {MARGIN_STRIPS} strips of good-hit boundary")
    print("=" * 80)

    for _, row in budget_df.iterrows():

        n = int(row["max_predictions_allowed"])

        margin_rate = float(row["success_margin_rate"])
        margin_count = int(row["success_margin_count"])

        exact_rate = float(row["success_exact_rate"])
        exact_count = int(row["success_exact_count"])

        total = int(row["total_truth_hit_events"])

        print(
            f"<= {n:3d} predictions : "
            f"margin={margin_rate:.6f} ({margin_count}/{total})   "
            f"exact={exact_rate:.6f} ({exact_count}/{total})"
        )


def compute_paired_outcomes(
    ml_event_summary_df: pd.DataFrame,
    cpp_event_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Paired event-level outcomes using the margin-boundary success definition.
    """

    ml = ml_event_summary_df[[
        "event_id",
        "any_prediction_within_margin_boundary",
    ]].rename(
        columns={
            "any_prediction_within_margin_boundary": "ML_success_margin"
        }
    )

    cpp = cpp_event_summary_df[[
        "event_id",
        "any_prediction_within_margin_boundary",
    ]].rename(
        columns={
            "any_prediction_within_margin_boundary": "CPP_success_margin"
        }
    )

    paired = ml.merge(cpp, on="event_id", how="inner")

    paired["ML_success_margin"] = paired["ML_success_margin"].astype(bool)
    paired["CPP_success_margin"] = paired["CPP_success_margin"].astype(bool)

    n_both = int(
        (paired["ML_success_margin"] & paired["CPP_success_margin"]).sum()
    )

    n_only_ml = int(
        (paired["ML_success_margin"] & ~paired["CPP_success_margin"]).sum()
    )

    n_only_cpp = int(
        (~paired["ML_success_margin"] & paired["CPP_success_margin"]).sum()
    )

    n_neither = int(
        (~paired["ML_success_margin"] & ~paired["CPP_success_margin"]).sum()
    )

    rows = [
        {
            "event_outcome": "Both methods successful within margin",
            "number_of_events": n_both,
        },
        {
            "event_outcome": "Only ML successful within margin",
            "number_of_events": n_only_ml,
        },
        {
            "event_outcome": "Only CPP successful within margin",
            "number_of_events": n_only_cpp,
        },
        {
            "event_outcome": "Both methods unsuccessful within margin",
            "number_of_events": n_neither,
        },
    ]

    out = pd.DataFrame(rows)

    total = int(out["number_of_events"].sum())
    out["fraction"] = (
        out["number_of_events"] / total
        if total > 0 else np.nan
    )

    return out

# ============================================================
# RUN ML
# ============================================================

ml_compare_df = compare_centers_to_truth_with_boundary_margin(
    pred_df=ml_pred_df,
    truth_info_df=truth_info_df,
    object_id_col="blob_id",
    source_name="ML",
    margin=MARGIN_STRIPS,
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

ml_unlimited = compute_unlimited_efficiency(
    event_summary_df=ml_event_summary_df,
    source_name="ML",
)

print("\nML predicted blobs with matching truth events:", len(ml_compare_df))
print(
    f"ML predictions within {MARGIN_STRIPS}-strip boundary margin:",
    int(ml_compare_df["prediction_within_margin_boundary"].sum())
)
print(
    f"ML truth events with at least one prediction within {MARGIN_STRIPS}-strip boundary margin:",
    int(ml_event_summary_df["any_prediction_within_margin_boundary"].sum())
)
print(
    "ML truth events with zero predicted blobs:",
    int((ml_event_summary_df["n_predictions"] == 0).sum())
)

print_budget_summary(ml_budget_summary_df, "ML")

# ============================================================
# RUN CPP
# ============================================================

cpp_compare_df = compare_centers_to_truth_with_boundary_margin(
    pred_df=cpp_pred_df,
    truth_info_df=truth_info_df,
    object_id_col="hit_id",
    source_name="CPP",
    margin=MARGIN_STRIPS,
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

cpp_unlimited = compute_unlimited_efficiency(
    event_summary_df=cpp_event_summary_df,
    source_name="CPP",
)

print("\nCPP 2D hits with matching truth events:", len(cpp_compare_df))
print(
    f"CPP predictions within {MARGIN_STRIPS}-strip boundary margin:",
    int(cpp_compare_df["prediction_within_margin_boundary"].sum())
)
print(
    f"CPP truth events with at least one prediction within {MARGIN_STRIPS}-strip boundary margin:",
    int(cpp_event_summary_df["any_prediction_within_margin_boundary"].sum())
)
print(
    "CPP truth events with zero 2D hits:",
    int((cpp_event_summary_df["n_predictions"] == 0).sum())
)

print_budget_summary(cpp_budget_summary_df, "CPP")

# ============================================================
# UNLIMITED EFFICIENCY
# ============================================================

unlimited_efficiency_df = pd.DataFrame([
    ml_unlimited,
    cpp_unlimited,
])

print()
print("=" * 80)
print("Unlimited-prediction efficiency")
print(f"Main success: at least one prediction within {MARGIN_STRIPS} strips of good-hit boundary")
print("=" * 80)

for _, row in unlimited_efficiency_df.iterrows():

    source = row["source"]

    margin_count = int(row["success_margin_count"])
    exact_count = int(row["success_exact_count"])
    total = int(row["total_truth_hit_events"])

    margin_rate = float(row["success_margin_rate"])
    exact_rate = float(row["success_exact_rate"])

    print(
        f"{source}: "
        f"margin={margin_rate:.8f} ({margin_count}/{total})   "
        f"exact={exact_rate:.8f} ({exact_count}/{total})"
    )

# ============================================================
# COMBINED BUDGET SUMMARY
# ============================================================

combined_budget_summary_df = pd.concat(
    [
        ml_budget_summary_df,
        cpp_budget_summary_df,
    ],
    ignore_index=True,
)

print()
print("=" * 80)
print(f"ML vs CPP margin success-rate comparison, margin = {MARGIN_STRIPS} strips")
print("=" * 80)

combined_pivot = combined_budget_summary_df.pivot(
    index="max_predictions_allowed",
    columns="source",
    values="success_margin_rate",
).reset_index()

print(combined_pivot.to_string(index=False))

# ============================================================
# PAIRED OUTCOMES
# ============================================================

paired_outcome_df = compute_paired_outcomes(
    ml_event_summary_df=ml_event_summary_df,
    cpp_event_summary_df=cpp_event_summary_df,
)

print()
print("=" * 80)
print(f"Paired event outcomes using {MARGIN_STRIPS}-strip boundary margin")
print("=" * 80)
print(paired_outcome_df.to_string(index=False))

# ============================================================
# SAVE OUTPUTS
# ============================================================

os.makedirs(OUTDIR, exist_ok=True)

ml_compare_df.to_csv(
    ML_COMPARE_OUTFILE,
    sep=" ",
    index=False,
    float_format="%.6f",
)

ml_event_summary_df.to_csv(
    ML_EVENT_SUMMARY_OUTFILE,
    sep=" ",
    index=False,
    float_format="%.6f",
)

ml_budget_summary_df.to_csv(
    ML_BUDGET_SUMMARY_OUTFILE,
    sep=" ",
    index=False,
    float_format="%.8f",
)

cpp_compare_df.to_csv(
    CPP_COMPARE_OUTFILE,
    sep=" ",
    index=False,
    float_format="%.6f",
)

cpp_event_summary_df.to_csv(
    CPP_EVENT_SUMMARY_OUTFILE,
    sep=" ",
    index=False,
    float_format="%.6f",
)

cpp_budget_summary_df.to_csv(
    CPP_BUDGET_SUMMARY_OUTFILE,
    sep=" ",
    index=False,
    float_format="%.8f",
)

combined_budget_summary_df.to_csv(
    COMBINED_BUDGET_SUMMARY_OUTFILE,
    sep=" ",
    index=False,
    float_format="%.8f",
)

unlimited_efficiency_df.to_csv(
    UNLIMITED_EFFICIENCY_OUTFILE,
    sep=" ",
    index=False,
    float_format="%.8f",
)

paired_outcome_df.to_csv(
    PAIRED_OUTCOME_OUTFILE,
    sep=" ",
    index=False,
    float_format="%.8f",
)

print()
print("Saved outputs:")
print("ML margin comparison:       ", ML_COMPARE_OUTFILE)
print("ML event summary:           ", ML_EVENT_SUMMARY_OUTFILE)
print("ML margin budget summary:   ", ML_BUDGET_SUMMARY_OUTFILE)
print("CPP margin comparison:      ", CPP_COMPARE_OUTFILE)
print("CPP event summary:          ", CPP_EVENT_SUMMARY_OUTFILE)
print("CPP margin budget summary:  ", CPP_BUDGET_SUMMARY_OUTFILE)
print("Combined ML vs CPP summary: ", COMBINED_BUDGET_SUMMARY_OUTFILE)
print("Unlimited efficiency:       ", UNLIMITED_EFFICIENCY_OUTFILE)
print("Paired outcomes:            ", PAIRED_OUTCOME_OUTFILE)