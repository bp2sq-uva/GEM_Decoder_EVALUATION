#!/usr/bin/env python3
"""
Plot one GEM event in one figure containing three panels:

  1. goodADC truth hit only
  2. goodADC truth hit + C++ predictions
  3. goodADC truth hit + ML predictions

Prediction colors:
  C++ incorrect prediction -> blue cross
  ML incorrect prediction  -> orange cross
  Correct prediction       -> red cross with "Correct" annotation

The correctness rule matches the evaluation script:
the predicted U and V centers are rounded to the nearest strip ID, and the
prediction is correct when both rounded strip IDs belong to the goodADC truth
strips for the event.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# DEFAULT FILES
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
    "example_predictions"
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


# ============================================================
# LOAD INPUT FILES
# ============================================================

def load_truth(path: str | Path) -> pd.DataFrame:
    """Load the headerless goodADC truth file."""

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=TRUTH_COLS,
        usecols=list(range(len(TRUTH_COLS))),
    )

    for column in ["event_id", "module_id", "strip_id"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=["event_id", "module_id", "strip_id"]
    ).copy()

    df["event_id"] = df["event_id"].astype(int)
    df["module_id"] = df["module_id"].astype(int)
    df["strip_id"] = df["strip_id"].astype(int)

    return df


def load_ml_predictions(path: str | Path) -> pd.DataFrame:
    """Load the headered ML blob-center file."""

    df = pd.read_csv(path, sep=r"\s+")

    required = {
        "event_id",
        "blob_id",
        "x_strip",
        "y_strip",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            "ML file is missing required columns: "
            + ", ".join(sorted(missing))
        )

    df = df[
        ["event_id", "blob_id", "x_strip", "y_strip"]
    ].copy()

    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna().copy()

    df["event_id"] = df["event_id"].astype(int)
    df["blob_id"] = df["blob_id"].astype(int)

    return df


def load_cpp_predictions(path: str | Path) -> pd.DataFrame:
    """
    Load either of these C++ formats:

    Headered:
      Event_ID 2D_hit_ID Hit_center_U_strip_ID Hit_center_V_strip_ID

    Headerless:
      event_id hit_id u_center v_center
    """

    first_try = pd.read_csv(path, sep=r"\s+")

    expected_header = {
        "Event_ID",
        "2D_hit_ID",
        "Hit_center_U_strip_ID",
        "Hit_center_V_strip_ID",
    }

    if expected_header.issubset(first_try.columns):
        df = first_try.rename(
            columns={
                "Event_ID": "event_id",
                "2D_hit_ID": "hit_id",
                "Hit_center_U_strip_ID": "x_strip",
                "Hit_center_V_strip_ID": "y_strip",
            }
        )[CPP_COLS].copy()
    else:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=CPP_COLS,
            usecols=list(range(4)),
        )

    for column in CPP_COLS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=CPP_COLS).copy()

    df["event_id"] = df["event_id"].astype(int)
    df["hit_id"] = df["hit_id"].astype(int)

    return df


# ============================================================
# TRUTH AND CORRECTNESS
# ============================================================

def get_truth_strips(
    truth_df: pd.DataFrame,
    event_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    module_id == 0 -> U/X truth strips
    module_id == 1 -> V/Y truth strips
    """

    event_df = truth_df.loc[
        truth_df["event_id"] == event_id
    ]

    if event_df.empty:
        raise ValueError(
            f"Event {event_id} does not exist in the truth file."
        )

    u_truth = np.sort(
        event_df.loc[
            event_df["module_id"] == 0,
            "strip_id",
        ].unique()
    ).astype(int)

    v_truth = np.sort(
        event_df.loc[
            event_df["module_id"] == 1,
            "strip_id",
        ].unique()
    ).astype(int)

    if len(u_truth) == 0 or len(v_truth) == 0:
        raise ValueError(
            f"Event {event_id} does not contain both U and V truth strips."
        )

    return u_truth, v_truth


def classify_predictions(
    predictions: pd.DataFrame,
    u_truth: np.ndarray,
    v_truth: np.ndarray,
) -> pd.DataFrame:
    """Add rounded centers and exact-correctness status."""

    result = predictions.copy()

    result["x_strip"] = pd.to_numeric(
        result["x_strip"],
        errors="coerce",
    )

    result["y_strip"] = pd.to_numeric(
        result["y_strip"],
        errors="coerce",
    )

    result = result.dropna(
        subset=["x_strip", "y_strip"]
    ).copy()

    # Remove fake negative centers, matching the evaluator.
    result = result.loc[
        (result["x_strip"] >= 0)
        & (result["y_strip"] >= 0)
    ].copy()

    result["x_rounded"] = np.rint(
        result["x_strip"]
    ).astype(int)

    result["y_rounded"] = np.rint(
        result["y_strip"]
    ).astype(int)

    result["correct"] = (
        result["x_rounded"].isin(set(u_truth))
        & result["y_rounded"].isin(set(v_truth))
    )

    return result


# ============================================================
# AVAILABLE EVENT STATUS
# ============================================================

def build_available_event_status(
    truth_df: pd.DataFrame,
    ml_df: pd.DataFrame,
    cpp_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return all valid truth events with PASS/FAIL for both methods.

    PASS means at least one exact correct prediction.
    """

    rows = []

    for event_id in sorted(truth_df["event_id"].unique()):
        try:
            u_truth, v_truth = get_truth_strips(
                truth_df,
                int(event_id),
            )
        except ValueError:
            continue

        ml_event = classify_predictions(
            ml_df.loc[ml_df["event_id"] == event_id],
            u_truth,
            v_truth,
        )

        cpp_event = classify_predictions(
            cpp_df.loc[cpp_df["event_id"] == event_id],
            u_truth,
            v_truth,
        )

        ml_pass = bool(
            not ml_event.empty and ml_event["correct"].any()
        )

        cpp_pass = bool(
            not cpp_event.empty and cpp_event["correct"].any()
        )

        rows.append(
            {
                "event_id": int(event_id),
                "ML": (
                    "PASS"
                    if ml_pass
                    else (
                        "FAIL (no predictions)"
                        if ml_event.empty
                        else "FAIL"
                    )
                ),
                "CPP": (
                    "PASS"
                    if cpp_pass
                    else (
                        "FAIL (no predictions)"
                        if cpp_event.empty
                        else "FAIL"
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# PLOTTING HELPERS
# ============================================================

def draw_goodadc_hit(
    axis: plt.Axes,
    u_truth: np.ndarray,
    v_truth: np.ndarray,
) -> None:
    """
    Draw all U-V combinations belonging to the goodADC truth hit.
    """

    u_grid, v_grid = np.meshgrid(
        u_truth,
        v_truth,
    )

    axis.scatter(
        u_grid.ravel(),
        v_grid.ravel(),
        marker="s",
        s=150,
        color="green",
        alpha=0.45,
        label="goodADC hit",
    )


def draw_predictions(
    axis: plt.Axes,
    predictions: pd.DataFrame,
    method: str,
) -> None:
    """Draw incorrect and correct predictions."""

    if predictions.empty:
        return

    incorrect = predictions.loc[
        ~predictions["correct"]
    ]

    correct = predictions.loc[
        predictions["correct"]
    ]

    if method == "CPP":
        incorrect_color = "blue"
        incorrect_label = "C++ prediction"
    elif method == "ML":
        incorrect_color = "orange"
        incorrect_label = "ML prediction"
    else:
        raise ValueError(f"Unknown method: {method}")

    if not incorrect.empty:
        axis.scatter(
            incorrect["x_strip"],
            incorrect["y_strip"],
            marker="x",
            s=110,
            linewidths=2.2,
            color=incorrect_color,
            label=incorrect_label,
        )

    if not correct.empty:
        axis.scatter(
            correct["x_strip"],
            correct["y_strip"],
            marker="x",
            s=130,
            linewidths=2.8,
            color="red",
            label="Correct prediction",
        )

        for _, row in correct.iterrows():
            axis.annotate(
                "Correct",
                xy=(
                    float(row["x_strip"]),
                    float(row["y_strip"]),
                ),
                xytext=(7, 7),
                textcoords="offset points",
                color="red",
                fontsize=11,
                fontweight="bold",
            )


def calculate_axis_limits(
    u_truth: np.ndarray,
    v_truth: np.ndarray,
    ml_event: pd.DataFrame,
    cpp_event: pd.DataFrame,
    padding: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Use identical limits in all three figures."""

    all_u = list(u_truth.astype(float))
    all_v = list(v_truth.astype(float))

    for predictions in [ml_event, cpp_event]:
        if not predictions.empty:
            all_u.extend(
                predictions["x_strip"].astype(float).tolist()
            )
            all_v.extend(
                predictions["y_strip"].astype(float).tolist()
            )

    u_min = min(all_u) - padding
    u_max = max(all_u) + padding
    v_min = min(all_v) - padding
    v_max = max(all_v) + padding

    if u_max - u_min < 2:
        center = 0.5 * (u_min + u_max)
        u_min = center - 1
        u_max = center + 1

    if v_max - v_min < 2:
        center = 0.5 * (v_min + v_max)
        v_min = center - 1
        v_max = center + 1

    return (u_min, u_max), (v_min, v_max)


def format_axis(
    axis: plt.Axes,
    title: str,
    u_limits: tuple[float, float],
    v_limits: tuple[float, float],
) -> None:
    """Apply shared formatting."""

    axis.set_title(title)
    axis.set_xlabel("U strip ID")
    axis.set_ylabel("V strip ID")
    axis.set_xlim(*u_limits)
    axis.set_ylim(*v_limits)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")


def save_figure(
    figure: plt.Figure,
    output_base: Path,
) -> None:
    """Save each figure as PNG and PDF."""

    figure.savefig(
        output_base.with_suffix(".png"),
        dpi=200,
        bbox_inches="tight",
    )

    figure.savefig(
        output_base.with_suffix(".pdf"),
        bbox_inches="tight",
    )


# ============================================================
# MAIN EVENT PLOTTING
# ============================================================

def plot_event(
    event_id: int,
    ml_file: str | Path,
    cpp_file: str | Path,
    truth_file: str | Path,
    outdir: str | Path,
    padding: float = 5.0,
    show: bool = False,
) -> None:
    """Create one combined figure containing the three requested panels."""

    truth_df = load_truth(truth_file)
    ml_df = load_ml_predictions(ml_file)
    cpp_df = load_cpp_predictions(cpp_file)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        u_truth, v_truth = get_truth_strips(
            truth_df,
            event_id,
        )
    except ValueError as error:
        print(f"\n{error}\n")

        status_df = build_available_event_status(
            truth_df=truth_df,
            ml_df=ml_df,
            cpp_df=cpp_df,
        )

        status_file = outdir / "available_event_status.txt"

        status_df.to_csv(
            status_file,
            sep=" ",
            index=False,
        )

        print("Available valid truth events:")
        print(status_df.to_string(index=False))
        print(f"\nSaved event list: {status_file}")
        return

    ml_event = classify_predictions(
        ml_df.loc[ml_df["event_id"] == event_id],
        u_truth,
        v_truth,
    )

    cpp_event = classify_predictions(
        cpp_df.loc[cpp_df["event_id"] == event_id],
        u_truth,
        v_truth,
    )

    u_limits, v_limits = calculate_axis_limits(
        u_truth=u_truth,
        v_truth=v_truth,
        ml_event=ml_event,
        cpp_event=cpp_event,
        padding=padding,
    )

    # --------------------------------------------------------
    # SINGLE FIGURE WITH THREE PANELS
    # --------------------------------------------------------

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(21, 7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    # Panel 1: goodADC truth only
    draw_goodadc_hit(
        axis=axes[0],
        u_truth=u_truth,
        v_truth=v_truth,
    )

    format_axis(
        axis=axes[0],
        title="goodADC hit",
        u_limits=u_limits,
        v_limits=v_limits,
    )

    # Panel 2: C++ predictions
    draw_goodadc_hit(
        axis=axes[1],
        u_truth=u_truth,
        v_truth=v_truth,
    )

    draw_predictions(
        axis=axes[1],
        predictions=cpp_event,
        method="CPP",
    )

    format_axis(
        axis=axes[1],
        title="C++ predictions",
        u_limits=u_limits,
        v_limits=v_limits,
    )

    # Panel 3: ML predictions
    draw_goodadc_hit(
        axis=axes[2],
        u_truth=u_truth,
        v_truth=v_truth,
    )

    draw_predictions(
        axis=axes[2],
        predictions=ml_event,
        method="ML",
    )

    format_axis(
        axis=axes[2],
        title="ML predictions",
        u_limits=u_limits,
        v_limits=v_limits,
    )

    figure.suptitle(
        f"Event {event_id}: goodADC truth and prediction comparison",
        fontsize=16,
    )

    output_base = outdir / f"event_{event_id}_goodADC_CPP_ML"

    save_figure(
        figure=figure,
        output_base=output_base,
    )

    print(f"\nEvent {event_id}")
    print(
        f"C++: {int(cpp_event['correct'].sum())} correct "
        f"out of {len(cpp_event)} predictions"
    )
    print(
        f"ML:  {int(ml_event['correct'].sum())} correct "
        f"out of {len(ml_event)} predictions"
    )

    print("\nSaved combined figure:")
    print(output_base.with_suffix(".png"))
    print(output_base.with_suffix(".pdf"))

    if show:
        plt.show()
    else:
        plt.close(figure)


# ============================================================
# COMMAND LINE
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot goodADC truth, C++ predictions, and ML predictions "
            "in three separate figures."
        )
    )

    parser.add_argument(
        "--event-id",
        type=int,
        required=True,
        help="Event ID to plot.",
    )

    parser.add_argument(
        "--ml-file",
        default=DEFAULT_ML_FILE,
    )

    parser.add_argument(
        "--cpp-file",
        default=DEFAULT_CPP_FILE,
    )

    parser.add_argument(
        "--truth-file",
        default=DEFAULT_TRUTH_FILE,
    )

    parser.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
    )

    parser.add_argument(
        "--padding",
        type=float,
        default=5.0,
        help="Plot padding in strip units.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display all three figures interactively.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    plot_event(
        event_id=args.event_id,
        ml_file=args.ml_file,
        cpp_file=args.cpp_file,
        truth_file=args.truth_file,
        outdir=args.outdir,
        padding=args.padding,
        show=args.show,
    )


if __name__ == "__main__":
    main()
