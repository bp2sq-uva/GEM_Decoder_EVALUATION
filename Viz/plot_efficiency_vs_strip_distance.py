#!/usr/bin/env python3
"""
Plot ML and C++ event-level efficiencies as a function of allowed strip
separation from the goodADC ground-truth center.

Default success rule for tolerance N:
    abs(predicted_U - truth_center_U) <= N
    abs(predicted_V - truth_center_V) <= N

This is equivalent to Chebyshev distance <= N. An event succeeds when at
least one prediction satisfies the rule. Events with zero predictions fail.
The denominator is all valid truth events containing both U and V truth strips.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_ML_FILE = (
    "../Scratch/ML/textfile_outputs/"
    "hit_centers_ML_withoutROIcut.txt"
)

DEFAULT_CPP_FILE = (
    "../Scratch/CPP/textfile_outputs/"
    "hit_centers_CPP_withoutROIcut.txt"
)

DEFAULT_TRUTH_FILE = (
    "../Scratch/Truth_info/"
    "groundtruth_withoutROIcut.txt"
)

DEFAULT_OUTDIR = (
    "plots/withoutROIcut/"
    "efficiency_vs_strip_distance"
)

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

CPP_COLS = [
    "event_id",
    "hit_id",
    "x_strip",
    "y_strip",
]

ML_REQUIRED_COLS = [
    "event_id",
    "blob_id",
    "x_strip",
    "y_strip",
]


# ============================================================
# LOAD FILES
# ============================================================

def load_truth(path: str | Path) -> pd.DataFrame:
    truth_df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=TRUTH_COLS,
        usecols=list(range(len(TRUTH_COLS))),
    )

    for column in ["event_id", "module_id", "strip_id"]:
        truth_df[column] = pd.to_numeric(
            truth_df[column],
            errors="coerce",
        )

    truth_df = truth_df.dropna(
        subset=["event_id", "module_id", "strip_id"]
    ).copy()

    truth_df["event_id"] = truth_df["event_id"].astype(int)
    truth_df["module_id"] = truth_df["module_id"].astype(int)
    truth_df["strip_id"] = truth_df["strip_id"].astype(int)

    return truth_df


def load_ml_predictions(path: str | Path) -> pd.DataFrame:
    ml_df = pd.read_csv(path, sep=r"\s+")

    missing = set(ML_REQUIRED_COLS) - set(ml_df.columns)
    if missing:
        raise RuntimeError(
            "ML prediction file is missing required columns: "
            + ", ".join(sorted(missing))
        )

    ml_df = ml_df[ML_REQUIRED_COLS].copy()

    for column in ML_REQUIRED_COLS:
        ml_df[column] = pd.to_numeric(
            ml_df[column],
            errors="coerce",
        )

    ml_df = ml_df.dropna(subset=ML_REQUIRED_COLS).copy()
    ml_df["event_id"] = ml_df["event_id"].astype(int)
    ml_df["blob_id"] = ml_df["blob_id"].astype(int)

    return ml_df


def load_cpp_predictions(path: str | Path) -> pd.DataFrame:
    """Load either the headered or headerless four-column C++ file."""

    header_try = pd.read_csv(path, sep=r"\s+")

    expected_header = {
        "Event_ID",
        "2D_hit_ID",
        "Hit_center_U_strip_ID",
        "Hit_center_V_strip_ID",
    }

    if expected_header.issubset(header_try.columns):
        cpp_df = header_try.rename(
            columns={
                "Event_ID": "event_id",
                "2D_hit_ID": "hit_id",
                "Hit_center_U_strip_ID": "x_strip",
                "Hit_center_V_strip_ID": "y_strip",
            }
        )[CPP_COLS].copy()
    else:
        cpp_df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=CPP_COLS,
            usecols=list(range(4)),
        )

    for column in CPP_COLS:
        cpp_df[column] = pd.to_numeric(
            cpp_df[column],
            errors="coerce",
        )

    cpp_df = cpp_df.dropna(subset=CPP_COLS).copy()
    cpp_df["event_id"] = cpp_df["event_id"].astype(int)
    cpp_df["hit_id"] = cpp_df["hit_id"].astype(int)

    return cpp_df


# ============================================================
# BUILD GOODADC TRUTH CENTERS
# ============================================================

def build_truth_centers(truth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per valid truth event.

    module_id == 0 -> U/X strips
    module_id == 1 -> V/Y strips

    The center is the mean of the unique strip IDs in each view.
    """

    rows = []

    for event_id, event_df in truth_df.groupby("event_id"):
        u_truth = np.sort(
            event_df.loc[
                event_df["module_id"] == 0,
                "strip_id",
            ].astype(int).unique()
        )

        v_truth = np.sort(
            event_df.loc[
                event_df["module_id"] == 1,
                "strip_id",
            ].astype(int).unique()
        )

        if len(u_truth) == 0 or len(v_truth) == 0:
            continue

        rows.append(
            {
                "event_id": int(event_id),
                "u_truth_center": float(np.mean(u_truth)),
                "v_truth_center": float(np.mean(v_truth)),
                "u_truth_min": int(u_truth.min()),
                "u_truth_max": int(u_truth.max()),
                "v_truth_min": int(v_truth.min()),
                "v_truth_max": int(v_truth.max()),
                "n_truth_u": int(len(u_truth)),
                "n_truth_v": int(len(v_truth)),
                "u_truth_strips": tuple(int(v) for v in u_truth),
                "v_truth_strips": tuple(int(v) for v in v_truth),
            }
        )

    truth_center_df = pd.DataFrame(rows)

    all_events = set(truth_df["event_id"].unique())
    valid_events = set(truth_center_df["event_id"].unique())
    invalid_events = sorted(all_events - valid_events)

    if invalid_events:
        print(
            f"[warning] Dropped {len(invalid_events)} truth event(s) "
            "without both U and V truth strips."
        )

    return truth_center_df


# ============================================================
# PREDICTION DISTANCES
# ============================================================

def compare_predictions_to_truth_center(
    pred_df: pd.DataFrame,
    truth_center_df: pd.DataFrame,
    object_id_col: str,
    source_name: str,
    distance_mode: str,
) -> pd.DataFrame:
    """Calculate prediction-to-truth-center distances in strip units."""

    pred_df = pred_df.copy()

    for column in [
        "event_id",
        object_id_col,
        "x_strip",
        "y_strip",
    ]:
        pred_df[column] = pd.to_numeric(
            pred_df[column],
            errors="coerce",
        )

    pred_df = pred_df.dropna(
        subset=[
            "event_id",
            object_id_col,
            "x_strip",
            "y_strip",
        ]
    ).copy()

    pred_df["event_id"] = pred_df["event_id"].astype(int)
    pred_df[object_id_col] = pred_df[object_id_col].astype(int)

    fake_mask = (
        (pred_df["x_strip"] < 0)
        | (pred_df["y_strip"] < 0)
    )

    if fake_mask.any():
        print(
            f"[warning] {source_name}: removed "
            f"{int(fake_mask.sum())} negative/fake prediction(s)."
        )

    pred_df = pred_df.loc[~fake_mask].copy()

    compared_df = pred_df.merge(
        truth_center_df,
        on="event_id",
        how="inner",
    )

    compared_df["delta_u_signed"] = (
        compared_df["x_strip"]
        - compared_df["u_truth_center"]
    )

    compared_df["delta_v_signed"] = (
        compared_df["y_strip"]
        - compared_df["v_truth_center"]
    )

    compared_df["delta_u_abs"] = np.abs(
        compared_df["delta_u_signed"]
    )

    compared_df["delta_v_abs"] = np.abs(
        compared_df["delta_v_signed"]
    )

    if distance_mode == "chebyshev":
        # Passes tolerance N only when both |delta_U| and |delta_V| <= N.
        compared_df["center_distance_strips"] = np.maximum(
            compared_df["delta_u_abs"],
            compared_df["delta_v_abs"],
        )
    elif distance_mode == "euclidean":
        compared_df["center_distance_strips"] = np.sqrt(
            compared_df["delta_u_abs"] ** 2
            + compared_df["delta_v_abs"] ** 2
        )
    elif distance_mode == "manhattan":
        compared_df["center_distance_strips"] = (
            compared_df["delta_u_abs"]
            + compared_df["delta_v_abs"]
        )
    else:
        raise ValueError(
            f"Unsupported distance mode: {distance_mode}"
        )

    compared_df["source"] = source_name
    return compared_df


def build_event_nearest_distance(
    compared_df: pd.DataFrame,
    truth_center_df: pd.DataFrame,
    object_id_col: str,
    source_name: str,
) -> pd.DataFrame:
    """
    Build one row per truth event using the closest prediction.

    Events with no predictions get nearest_distance_strips = infinity,
    so they fail at every finite tolerance.
    """

    base_df = truth_center_df[
        [
            "event_id",
            "u_truth_center",
            "v_truth_center",
            "n_truth_u",
            "n_truth_v",
        ]
    ].copy()

    if compared_df.empty:
        base_df["n_predictions"] = 0
        base_df["nearest_prediction_id"] = np.nan
        base_df["nearest_prediction_u"] = np.nan
        base_df["nearest_prediction_v"] = np.nan
        base_df["nearest_delta_u_abs"] = np.nan
        base_df["nearest_delta_v_abs"] = np.nan
        base_df["nearest_distance_strips"] = np.inf
        base_df["source"] = source_name
        return base_df

    count_df = (
        compared_df.groupby("event_id")
        .size()
        .rename("n_predictions")
        .reset_index()
    )

    nearest_indices = (
        compared_df.groupby("event_id")[
            "center_distance_strips"
        ].idxmin()
    )

    nearest_df = compared_df.loc[
        nearest_indices,
        [
            "event_id",
            object_id_col,
            "x_strip",
            "y_strip",
            "delta_u_abs",
            "delta_v_abs",
            "center_distance_strips",
        ],
    ].copy()

    nearest_df = nearest_df.rename(
        columns={
            object_id_col: "nearest_prediction_id",
            "x_strip": "nearest_prediction_u",
            "y_strip": "nearest_prediction_v",
            "delta_u_abs": "nearest_delta_u_abs",
            "delta_v_abs": "nearest_delta_v_abs",
            "center_distance_strips": "nearest_distance_strips",
        }
    )

    event_df = base_df.merge(
        count_df,
        on="event_id",
        how="left",
    ).merge(
        nearest_df,
        on="event_id",
        how="left",
    )

    event_df["n_predictions"] = (
        event_df["n_predictions"]
        .fillna(0)
        .astype(int)
    )

    event_df["nearest_distance_strips"] = (
        event_df["nearest_distance_strips"]
        .fillna(np.inf)
    )

    event_df["source"] = source_name
    return event_df


# ============================================================
# EFFICIENCY AS A FUNCTION OF STRIP TOLERANCE
# ============================================================

def compute_efficiency_curve(
    event_distance_df: pd.DataFrame,
    source_name: str,
    max_strips: int,
) -> pd.DataFrame:
    """Calculate cumulative event efficiency for tolerance 0..max_strips."""

    total_truth_events = len(event_distance_df)
    rows = []

    for tolerance in range(max_strips + 1):
        success_mask = (
            event_distance_df["nearest_distance_strips"]
            <= float(tolerance)
        )

        success_count = int(success_mask.sum())
        efficiency = (
            success_count / total_truth_events
            if total_truth_events > 0
            else np.nan
        )

        rows.append(
            {
                "source": source_name,
                "strip_tolerance": int(tolerance),
                "success_count": success_count,
                "total_truth_events": int(total_truth_events),
                "efficiency": float(efficiency),
            }
        )

    return pd.DataFrame(rows)


def print_efficiency_curve(
    efficiency_df: pd.DataFrame,
    source_name: str,
) -> None:
    print()
    print("=" * 72)
    print(f"{source_name} efficiency versus strip tolerance")
    print("=" * 72)

    for row in efficiency_df.itertuples(index=False):
        print(
            f"within {int(row.strip_tolerance):3d} strip(s): "
            f"{float(row.efficiency):.6f} "
            f"({int(row.success_count)}/"
            f"{int(row.total_truth_events)})"
        )


# ============================================================
# PLOT
# ============================================================

def plot_efficiency_curves(
    ml_efficiency_df: pd.DataFrame,
    cpp_efficiency_df: pd.DataFrame,
    distance_mode: str,
    output_base: Path,
    show: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))

    # Matplotlib default colors produce C++ blue and ML orange.
    axis.plot(
        cpp_efficiency_df["strip_tolerance"],
        cpp_efficiency_df["efficiency"],
        marker="o",
        linewidth=2,
        label="C++",
    )

    axis.plot(
        ml_efficiency_df["strip_tolerance"],
        ml_efficiency_df["efficiency"],
        marker="s",
        linewidth=2,
        label="ML",
    )

    if distance_mode == "chebyshev":
        subtitle = "Within N strips independently in both U and V"
    elif distance_mode == "euclidean":
        subtitle = "Euclidean distance in the U-V strip plane"
    else:
        subtitle = "Manhattan distance in the U-V strip plane"

    axis.set_title(
        "ML versus C++ hit-finding efficiency\n"
        + subtitle
    )
    axis.set_xlabel(
        "Allowed distance from goodADC center [strips]"
    )
    axis.set_ylabel("Efficiency")
    axis.set_ylim(0.0, 1.02)
    axis.set_xlim(
        0,
        int(ml_efficiency_df["strip_tolerance"].max()),
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    figure.savefig(
        output_base.with_suffix(".png"),
        dpi=200,
        bbox_inches="tight",
    )

    figure.savefig(
        output_base.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    if show:
        plt.show()
    else:
        plt.close(figure)


# ============================================================
# MAIN ANALYSIS
# ============================================================

def run_analysis(
    ml_file: str | Path,
    cpp_file: str | Path,
    truth_file: str | Path,
    outdir: str | Path,
    max_strips: int,
    distance_mode: str,
    show: bool,
) -> None:
    if max_strips < 0:
        raise ValueError("--max-strips must be nonnegative.")

    truth_df = load_truth(truth_file)
    ml_pred_df = load_ml_predictions(ml_file)
    cpp_pred_df = load_cpp_predictions(cpp_file)

    truth_center_df = build_truth_centers(truth_df)

    if truth_center_df.empty:
        raise RuntimeError(
            "No valid truth events containing both U and V strips."
        )

    print(f"Valid truth events: {len(truth_center_df)}")
    print(f"ML prediction rows: {len(ml_pred_df)}")
    print(f"C++ prediction rows: {len(cpp_pred_df)}")

    ml_compare_df = compare_predictions_to_truth_center(
        pred_df=ml_pred_df,
        truth_center_df=truth_center_df,
        object_id_col="blob_id",
        source_name="ML",
        distance_mode=distance_mode,
    )

    cpp_compare_df = compare_predictions_to_truth_center(
        pred_df=cpp_pred_df,
        truth_center_df=truth_center_df,
        object_id_col="hit_id",
        source_name="CPP",
        distance_mode=distance_mode,
    )

    ml_event_distance_df = build_event_nearest_distance(
        compared_df=ml_compare_df,
        truth_center_df=truth_center_df,
        object_id_col="blob_id",
        source_name="ML",
    )

    cpp_event_distance_df = build_event_nearest_distance(
        compared_df=cpp_compare_df,
        truth_center_df=truth_center_df,
        object_id_col="hit_id",
        source_name="CPP",
    )

    ml_efficiency_df = compute_efficiency_curve(
        event_distance_df=ml_event_distance_df,
        source_name="ML",
        max_strips=max_strips,
    )

    cpp_efficiency_df = compute_efficiency_curve(
        event_distance_df=cpp_event_distance_df,
        source_name="CPP",
        max_strips=max_strips,
    )

    print_efficiency_curve(ml_efficiency_df, "ML")
    print_efficiency_curve(cpp_efficiency_df, "C++")

    combined_df = pd.concat(
        [ml_efficiency_df, cpp_efficiency_df],
        ignore_index=True,
    )

    comparison_df = combined_df.pivot(
        index="strip_tolerance",
        columns="source",
        values=["success_count", "efficiency"],
    )

    comparison_df.columns = [
        f"{metric}_{source}"
        for metric, source in comparison_df.columns
    ]

    comparison_df = comparison_df.reset_index()

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    output_base = (
        outdir
        / "ML_vs_CPP_efficiency_vs_strip_tolerance"
    )

    plot_efficiency_curves(
        ml_efficiency_df=ml_efficiency_df,
        cpp_efficiency_df=cpp_efficiency_df,
        distance_mode=distance_mode,
        output_base=output_base,
        show=show,
    )

    summary_file = output_base.with_suffix(".txt")
    comparison_df.to_csv(
        summary_file,
        sep=" ",
        index=False,
        float_format="%.8f",
    )

    ml_event_file = (
        outdir
        / "ML_nearest_prediction_distance_per_event.txt"
    )

    cpp_event_file = (
        outdir
        / "CPP_nearest_prediction_distance_per_event.txt"
    )

    ml_event_distance_df.to_csv(
        ml_event_file,
        sep=" ",
        index=False,
        float_format="%.6f",
    )

    cpp_event_distance_df.to_csv(
        cpp_event_file,
        sep=" ",
        index=False,
        float_format="%.6f",
    )

    print()
    print("Saved outputs:")
    print(output_base.with_suffix(".png"))
    print(output_base.with_suffix(".pdf"))
    print(summary_file)
    print(ml_event_file)
    print(cpp_event_file)


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot ML and C++ efficiencies versus allowed strip "
            "distance from the goodADC truth center."
        )
    )

    parser.add_argument(
        "--ml-file",
        default=DEFAULT_ML_FILE,
        help="ML prediction text file.",
    )

    parser.add_argument(
        "--cpp-file",
        default=DEFAULT_CPP_FILE,
        help="C++ prediction text file.",
    )

    parser.add_argument(
        "--truth-file",
        default=DEFAULT_TRUTH_FILE,
        help="goodADC truth text file.",
    )

    parser.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
        help="Output directory.",
    )

    parser.add_argument(
        "--max-strips",
        type=int,
        default=20,
        help="Maximum strip tolerance to plot. Default: 20.",
    )

    parser.add_argument(
        "--distance-mode",
        choices=["chebyshev", "euclidean", "manhattan"],
        default="chebyshev",
        help=(
            "Distance definition. The default chebyshev mode means "
            "within N strips in both U and V."
        ),
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_analysis(
        ml_file=args.ml_file,
        cpp_file=args.cpp_file,
        truth_file=args.truth_file,
        outdir=args.outdir,
        max_strips=args.max_strips,
        distance_mode=args.distance_mode,
        show=args.show,
    )


if __name__ == "__main__":
    main()
