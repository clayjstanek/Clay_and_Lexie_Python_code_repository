"""Analyze the noise sweep produced by kalman_filter_haos_flight_01(2).py.

REQUIRED INPUT: kf_samples.csv (the LARGE per-sample CSV).
Neither kf_summary.csv nor the original flight CSV is required. This script
does not rerun or change the Kalman filter. References are the observation_*
columns already exported in kf_samples.csv, NOT independent ground truth.

QUICK START (Python 3.10+, numpy, pandas, matplotlib):
    pip install numpy pandas matplotlib
    python analyze_kf_sweep.py --samples "path/to/kf_samples.csv"

SPYDER: edit CONFIG below and press Run. With input_csv=None, the script looks
beside itself and in its haos_kalman_output or results subdirectory. If there
is more than one candidate, set input_csv explicitly. Relative input/output
paths are resolved beside THIS SCRIPT, not against Spyder's working directory.

ITEMS IMPLEMENTED:
1. Prior/prediction residual RMS, median, p95, maximum/time, exceedance, bias.
2. Common maneuver/lower-acceleration/onset/recovery subsets and event plots.
3. Open-loop constant-velocity forecasts at 1, 5, 10 seconds; no updates.
4. Prior/posterior position/velocity covariance summaries, principal-axis
   uncertainty, and full mean covariance matrices (never a mixed-unit trace).
5. Posterior residuals, descriptive NIS, and innovation autocorrelation.

Read output/README_results.txt for definitions and interpretation. CSV metric
tables contain every setting; plots overlay only the four sweep corners to
remain readable. Heatmaps show every combination. All metrics are sample-
weighted; on this HAOS file samples are uniformly spaced at one second.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import re
import sys
import warnings

import matplotlib
matplotlib.use("Agg")  # Save figures without opening hundreds of GUI windows.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AnalysisConfig:
    input_csv: str | None = "kf_samples.csv"  # e.g. r"C:\Users\Clay\KF\haos_kalman_output\kf_samples.csv"
    output_directory: str = "kf_analysis_output"
    burn_in_seconds: float = 0.0  # Initialization is ALWAYS excluded; try 30.0 separately.
    forecast_horizons_s: tuple[float, ...] = (1.0, 5.0, 10.0)
    position_tolerance_m: float = 10.0  # Illustrative, editable; applies to each position norm.
    velocity_tolerance_mps: float = 2.0  # Illustrative, editable.
    reference_smoothing_seconds: float = 5.0
    maneuver_acceleration_threshold_mps2: float | None = None
    maneuver_percentile: float = 75.0  # Used only when absolute threshold is None.
    event_window_seconds: float = 10.0
    max_event_plots: int = 3
    acf_max_lag_samples: int = 20
    exact_time_tolerance_s: float = 1e-6
    save_figures: bool = True


CONFIG = AnalysisConfig()
#CONFIG.input_csv = "kf_samples.csv"
BASE_DIR = Path(__file__).resolve().parent
AXES = ("east", "north", "up", "ve", "vn", "vu")
KEYS = ["segment_id", "model_acceleration_std_g", "position_noise_std_m"]
QUANTITIES = {
    "position_3d": ([0, 1, 2], "m"),
    "position_horizontal": ([0, 1], "m"),
    "position_vertical": ([2], "m"),
    "velocity_3d": ([3, 4, 5], "m/s"),
}


def resolve_input(config):
    if config.input_csv is not None:
        p = Path(config.input_csv).expanduser()
        p = p if p.is_absolute() else BASE_DIR / p
        if not p.is_file():
            raise FileNotFoundError(f"Input does not exist: {p}")
        return p.resolve()
    candidates = [BASE_DIR / "kf_samples.csv",
                  BASE_DIR / "haos_kalman_output" / "kf_samples.csv",
                  BASE_DIR / "results" / "kf_samples.csv"]
    found = [p for p in candidates if p.is_file()]
    if len(found) != 1:
        raise ValueError("Set CONFIG.input_csv or use --samples with the LARGE kf_samples.csv. "
                         f"Automatic search found {len(found)} candidates beside this script.")
    return found[0].resolve()


def validate_config(c):
    positive = (c.position_tolerance_m, c.velocity_tolerance_mps,
                c.reference_smoothing_seconds, c.event_window_seconds,
                c.exact_time_tolerance_s, *c.forecast_horizons_s)
    if not positive or not np.isfinite(positive).all() or min(positive) <= 0:
        raise ValueError("Horizons, tolerances, smoothing and event windows must be finite and positive.")
    if not c.forecast_horizons_s or len(set(c.forecast_horizons_s)) != len(c.forecast_horizons_s):
        raise ValueError("Provide at least one unique forecast horizon.")
    if not np.isfinite(c.burn_in_seconds) or c.burn_in_seconds < 0:
        raise ValueError("burn_in_seconds must be finite and nonnegative.")
    if not 0 < c.maneuver_percentile < 100:
        raise ValueError("maneuver_percentile must be between 0 and 100.")
    if c.maneuver_acceleration_threshold_mps2 is not None:
        if not np.isfinite(c.maneuver_acceleration_threshold_mps2) or c.maneuver_acceleration_threshold_mps2 < 0:
            raise ValueError("Maneuver acceleration threshold must be finite and nonnegative.")
    if c.acf_max_lag_samples < 1 or c.max_event_plots < 0:
        raise ValueError("ACF lag must be positive and event plot count nonnegative.")


def load_samples(path):
    required = KEYS + ["sample_index", "time_s", "nis", "measure_velocity",
                       "velocity_noise_std_mps", "initial_velocity_std_mps"]
    required += [f"observation_{a}" for a in AXES]
    for phase in ("prior", "posterior"):
        required += [f"x_{phase}_{a}" for a in AXES]
        required += [f"P_{phase}_{a}_{b}" for a in AXES for b in AXES]
    header = pd.read_csv(path, nrows=0).columns
    missing = set(required) - set(header)
    if missing:
        raise ValueError("Wrong input or incompatible export. Use the LARGE kf_samples.csv. "
                         f"Missing columns: {sorted(missing)}")
    data = pd.read_csv(path, dtype={"segment_id": str})
    if data.empty or data[KEYS].isna().any().any():
        raise ValueError("Empty input or missing run identifiers.")
    for col in ("model_acceleration_std_g", "position_noise_std_m",
                "velocity_noise_std_mps", "initial_velocity_std_mps"):
        if not np.isfinite(data[col]).all() or (data[col] <= 0).any():
            raise ValueError(f"Invalid noise metadata: {col}")
    for col in ("measure_velocity", "velocity_noise_std_mps", "initial_velocity_std_mps"):
        if data[col].nunique(dropna=False) != 1 or data[col].isna().any():
            raise ValueError(f"This analysis expects one fixed {col} across the sweep.")
    mode = str(data.measure_velocity.iloc[0]).lower()
    if mode not in ("true", "false"):
        raise ValueError("measure_velocity must contain True or False.")
    dimension = 6 if mode == "true" else 3
    missing = {f"innovation_{a}" for a in AXES[:dimension]} - set(header)
    if missing:
        raise ValueError(f"Missing innovation columns: {sorted(missing)}")
    if data.duplicated(KEYS + ["sample_index"]).any():
        raise ValueError("Duplicate samples within a parameter pair/segment; do not concatenate duplicate runs.")
    return data, dimension


def vector(group, prefix):
    return group[[f"{prefix}_{a}" for a in AXES]].to_numpy(float)


def covariance(group, phase):
    return group[[f"P_{phase}_{a}_{b}" for a in AXES for b in AXES]].to_numpy(float).reshape(-1, 6, 6)


def boolean_intervals(mask):
    """Inclusive sample-index endpoints of contiguous True intervals."""
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def reference_context(group, c):
    t = group.time_s.to_numpy(float)
    obs = vector(group, "observation")
    idx = group.sample_index.to_numpy()
    if len(t) < 3 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Each segment requires at least three strictly increasing finite times.")
    if not np.array_equal(idx, np.arange(len(t))) or not np.isclose(t[0], 0.0):
        raise ValueError("Expected sample_index=0..N-1 and segment time starting at zero.")
    if not np.isfinite(obs).all():
        raise ValueError("Reference observations contain missing/nonfinite values.")
    eligible = (np.arange(len(t)) > 0) & (t >= c.burn_in_seconds)
    if eligible.sum() < 2:
        raise ValueError("At least two evaluation samples must remain after initialization and burn-in.")
    # OFFLINE reference-only segmentation: centered smoothing is not used by the KF
    # or its forecasts. This mask is constructed once and reused for every setting.
    ref_v = pd.DataFrame(obs[:, 3:], index=pd.to_timedelta(t, unit="s"))
    smooth_v = ref_v.rolling(pd.Timedelta(seconds=c.reference_smoothing_seconds),
                            center=True, min_periods=1).mean().to_numpy()
    acceleration = np.linalg.norm(np.gradient(smooth_v, t, axis=0), axis=1)
    threshold = c.maneuver_acceleration_threshold_mps2
    if threshold is None:
        threshold = float(np.percentile(acceleration[eligible], c.maneuver_percentile))
    maneuver = acceleration > threshold
    onset = np.zeros(len(t), bool)
    recovery = np.zeros(len(t), bool)
    events = []
    for event_id, (start, end) in enumerate(boolean_intervals(maneuver)):
        left_censored, right_censored = start == 0, end == len(t) - 1
        # A censored start/end is not an observed onset/recovery.
        recovery_t = np.nan if right_censored else t[end + 1]
        if not left_censored:
            onset |= (t >= t[start] - c.event_window_seconds) & (t <= t[start] + c.event_window_seconds)
        if not right_censored:
            recovery |= (t >= recovery_t) & (t <= recovery_t + c.event_window_seconds)
        events.append(dict(event_id=event_id, start_s=t[start], last_maneuver_sample_s=t[end],
                           recovery_start_s=recovery_t, duration_between_samples_s=t[end] - t[start],
                           n_samples=end-start+1, left_censored=left_censored,
                           right_censored=right_censored, peak_acceleration_mps2=float(acceleration[start:end+1].max())))
    masks = {"all": eligible, "maneuver": eligible & maneuver,
             "lower_acceleration": eligible & ~maneuver,
             "onset_window": eligible & onset, "recovery_window": eligible & recovery}
    # Exact timestamps only. All horizons share the SAME origins, trimmed to the
    # longest horizon and valid target matches; no future interpolation or updates.
    targets = {}
    common_origins = eligible.copy()
    for horizon in c.forecast_horizons_s:
        requested = t + horizon
        right = np.clip(np.searchsorted(t, requested), 0, len(t)-1)
        left = np.maximum(right-1, 0)
        nearest = np.where(abs(t[left]-requested) < abs(t[right]-requested), left, right)
        valid = (abs(t[nearest]-requested) <= c.exact_time_tolerance_s) & (nearest > np.arange(len(t)))
        common_origins &= valid
        targets[horizon] = nearest
    if not common_origins.any():
        warnings.warn("No common exact forecast targets for this segment/horizon list; forecast scores will be empty.")
    return dict(t=t, observations=obs, eligible=eligible, masks=masks,
                acceleration=acceleration, threshold=float(threshold), maneuver=maneuver,
                events=events, targets=targets, common_origins=common_origins)


def residual_stats(residual, t, mask, columns, tolerance):
    r = residual[mask][:, columns]
    n = len(r)
    empty = dict(n_samples=0, rms=np.nan, median=np.nan, p95=np.nan, maximum=np.nan,
                 maximum_time_s=np.nan, fraction_exceeding_tolerance=np.nan,
                 tolerance=tolerance, bias_east=np.nan, bias_north=np.nan, bias_up=np.nan)
    if not n:
        return empty
    if not np.isfinite(r).all():
        raise ValueError("Nonfinite residual in an eligible evaluation sample.")
    distances = np.linalg.norm(r, axis=1)
    result = dict(n_samples=n, rms=float(np.sqrt(np.mean(distances**2))),
                  median=float(np.median(distances)), p95=float(np.percentile(distances, 95)),
                  maximum=float(distances.max()), maximum_time_s=float(t[mask][np.argmax(distances)]),
                  fraction_exceeding_tolerance=float(np.mean(distances > tolerance)), tolerance=tolerance,
                  bias_east=np.nan, bias_north=np.nan, bias_up=np.nan)
    for j, col in enumerate(columns):
        result[f"bias_{('east', 'north', 'up')[col % 3]}"] = float(r[:, j].mean())
    return result


def add_residual_rows(rows, meta, residual, t, masks, c, **extra):
    for regime, mask in masks.items():
        for quantity, (columns, unit) in QUANTITIES.items():
            tolerance = c.position_tolerance_m if unit == "m" else c.velocity_tolerance_mps
            rows.append(dict(meta, **extra, regime=regime, quantity=quantity, unit=unit,
                             **residual_stats(residual, t, mask, columns, tolerance)))


def covariance_summaries(rows, mean_rows, meta, p, phase, context):
    for regime, mask in context["masks"].items():
        selected = p[mask]
        mean = selected.mean(axis=0) if len(selected) else np.full((6, 6), np.nan)
        for i, a in enumerate(AXES):
            for j, b in enumerate(AXES):
                mean_rows.append(dict(meta, phase=phase, regime=regime, n_samples=len(selected),
                                      row_state=a, column_state=b, mean_covariance=mean[i, j],
                                      unit="m^2" if max(i, j) < 3 else "m^2/s^2" if min(i, j) >= 3 else "m^2/s"))
        for name, sl, unit in (("position", slice(0, 3), "m"), ("velocity", slice(3, 6), "m/s")):
            block = selected[:, sl, sl]
            if len(block):
                radius = np.sqrt(np.maximum(np.trace(block, axis1=1, axis2=2), 0))
                principal = np.sqrt(np.maximum(np.linalg.eigvalsh(block)[:, -1], 0))
                values = dict(rms_uncertainty=float(np.sqrt(np.mean(radius**2))),
                              median_radius=float(np.median(radius)), p95_radius=float(np.percentile(radius, 95)),
                              maximum_radius=float(radius.max()),
                              mean_largest_axis_std=float(principal.mean()),
                              p95_largest_axis_std=float(np.percentile(principal, 95)))
            else:
                values = dict.fromkeys(("rms_uncertainty", "median_radius", "p95_radius", "maximum_radius",
                                        "mean_largest_axis_std", "p95_largest_axis_std"), np.nan)
            rows.append(dict(meta, phase=phase, regime=regime, quantity=name, unit=unit,
                             n_samples=len(block), **values))


def autocorrelation(values, t, max_lag):
    """Conventional demeaned, biased sample ACF. Lags are in samples, not seconds."""
    centered = values - values.mean()
    denominator = float(centered @ centered)
    for lag in range(1, min(max_lag, len(values)-1) + 1):
        yield dict(lag_samples=lag, n_pairs=len(values)-lag,
                   mean_lag_seconds=float(np.mean(t[lag:] - t[:-lag])),
                   acf=float(centered[lag:] @ centered[:-lag] / denominator) if denominator > 0 else np.nan)


def validate_run(group, context, dimension):
    t = group.time_s.to_numpy(float)
    if not np.array_equal(group.sample_index.to_numpy(), np.arange(len(context["t"]))):
        raise ValueError("Runs have unequal/missing sample indices.")
    if not np.array_equal(t, context["t"]):
        raise ValueError("Runs must have identical timestamps for paired comparisons.")
    if not np.allclose(vector(group, "observation"), context["observations"], rtol=0, atol=1e-10):
        raise ValueError("Runs do not share identical reference observations.")
    for phase in ("prior", "posterior"):
        states = vector(group, f"x_{phase}")
        p = covariance(group, phase)
        start = 1 if phase == "prior" else 0
        if not np.isfinite(states[start:]).all() or not np.isfinite(p[start:]).all():
            raise ValueError(f"Nonfinite {phase} states/covariances outside initialization.")
        if not np.allclose(p[start:], p[start:].transpose(0, 2, 1), rtol=1e-7, atol=1e-8):
            raise ValueError("Covariance is not symmetric.")
        if np.linalg.eigvalsh(p[start:]).min() < -1e-7:
            raise ValueError("Covariance is not positive semidefinite.")
    needed = ["nis"] + [f"innovation_{a}" for a in AXES[:dimension]]
    if not np.isfinite(group[needed].iloc[1:].to_numpy(float)).all():
        raise ValueError("Missing NIS/innovation after initialization.")


def heatmap(ax, table, value, title, unit):
    grid = table.pivot(index="model_acceleration_std_g", columns="position_noise_std_m", values=value).sort_index()
    grid = grid.reindex(sorted(grid.columns), axis=1)
    values = grid.to_numpy(float)
    im = ax.imshow(np.ma.masked_invalid(values), origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(grid.columns)), [f"{v:g}" for v in grid.columns])
    ax.set_yticks(np.arange(len(grid.index)), [f"{v:.3g}" for v in grid.index])
    ax.set(xlabel="Position std (m)", ylabel="Acceleration std (g)", title=title)
    for i in range(len(grid.index)):
        for j in range(len(grid.columns)):
            v = values[i, j]
            color = "white" if np.isfinite(v) and im.norm(v) < .55 else "black"
            ax.text(j, i, f"{v:.3g}" if np.isfinite(v) else "—", ha="center", va="center", fontsize=7, color=color)
    ax.figure.colorbar(im, ax=ax, label=unit, shrink=.85)


def save_figure(fig, path):
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_plots(segment, context, corners, metrics, forecasts, uncertainty, acf, output, c, dimension):
    t, obs, eligible = context["t"], context["observations"], context["eligible"]
    title_tail = f"{segment} | observation residuals; initialization excluded"
    subset = lambda frame, **terms: frame.loc[np.logical_and.reduce([frame[k].eq(v) for k, v in terms.items()])]
    fig, axs = plt.subplots(2, 3, figsize=(17, 10), layout="constrained")
    for row, quantity, unit in ((0, "position_3d", "m"), (1, "velocity_3d", "m/s")):
        for col, (regime, score) in enumerate((("all", "rms"), ("all", "p95"), ("maneuver", "rms"))):
            heatmap(axs[row, col], subset(metrics, phase="prior", regime=regime, quantity=quantity), score,
                    f"{quantity.replace('_', ' ')}: {regime}, {score.upper()}", unit)
    fig.suptitle("One-step prediction comparison\n" + title_tail, fontsize=14)
    save_figure(fig, output / "01_prediction_heatmaps.png")

    horizons = sorted(c.forecast_horizons_s)
    fig, axs = plt.subplots(len(horizons), 2, figsize=(12, 4.5*len(horizons)), layout="constrained", squeeze=False)
    for i, horizon in enumerate(horizons):
        table = subset(forecasts, horizon_s=horizon, regime="all", quantity="position_3d")
        for j, score in enumerate(("rms", "p95")):
            heatmap(axs[i, j], table, score, f"{horizon:g} s open-loop position: {score.upper()}", "m")
    fig.suptitle("Common forecast origins for every horizon\n" + title_tail, fontsize=14)
    save_figure(fig, output / "02_forecast_heatmaps.png")

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), layout="constrained")
    for i, phase in enumerate(("prior", "posterior")):
        for j, quantity, unit in ((0, "position", "m"), (1, "velocity", "m/s")):
            heatmap(axs[i, j], subset(uncertainty, phase=phase, regime="all", quantity=quantity),
                    "rms_uncertainty", f"{phase.capitalize()} {quantity}: sqrt(mean trace(P))", unit)
    fig.suptitle(f"Reported uncertainty, not measured accuracy\n{segment}", fontsize=14)
    save_figure(fig, output / "03_covariance_heatmaps.png")

    fig, axs = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    for i, phase in enumerate(("prior", "posterior")):
        for j, quantity, unit in ((0, "position_3d", "m"), (1, "velocity_3d", "m/s")):
            heatmap(axs[i, j], subset(metrics, phase=phase, regime="all", quantity=quantity), "rms",
                    f"{phase.capitalize()} {quantity.replace('_', ' ')} residual RMS", unit)
    fig.suptitle("Prediction versus measurement fitting\n" + title_tail, fontsize=14)
    save_figure(fig, output / "04_prior_posterior_heatmaps.png")

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for (acc, pos), run in corners.items():
        residual = vector(run, "x_prior") - obs
        for ax, sl, unit in ((axs[0], slice(0, 3), "m"), (axs[1], slice(3, 6), "m/s")):
            distances = np.sort(np.linalg.norm(residual[eligible, sl], axis=1))
            ax.step(distances, np.arange(1, len(distances)+1)/len(distances), where="post",
                    label=f"{acc:.3g} g, {pos:g} m")
            ax.set(xlabel=f"Prior residual norm ({unit})", ylabel="Empirical cumulative fraction")
            ax.grid(alpha=.2)
    axs[0].set_title("Position residual distribution")
    axs[1].set_title("Velocity residual distribution")
    axs[1].legend(fontsize=8)
    fig.suptitle(f"Four sweep corners | {segment}")
    save_figure(fig, output / "05_residual_distributions.png")

    def temporal_plot(filename, start, end, caption):
        fig, axs = plt.subplots(3, 1, figsize=(13, 9), sharex=True, layout="constrained")
        view = eligible & (t >= start) & (t <= end)
        for (acc, pos), run in corners.items():
            residual = vector(run, "x_prior") - obs
            for ax, sl in ((axs[0], slice(0, 3)), (axs[1], slice(3, 6))):
                ax.plot(t[view], np.linalg.norm(residual[view, sl], axis=1), lw=1,
                        label=f"{acc:.3g} g, {pos:g} m")
        axs[0].set_ylabel("Prior position residual (m)")
        axs[1].set_ylabel("Prior velocity residual (m/s)")
        axs[2].plot(t[view], context["acceleration"][view], color="black", lw=1)
        axs[2].axhline(context["threshold"], color="darkorange", ls="--", label="Maneuver threshold")
        axs[2].set(xlabel="Time within segment (s)", ylabel="Reference acceleration (m/s²)")
        for ax in axs:
            ax.fill_between(t, 0, 1, where=context["maneuver"], transform=ax.get_xaxis_transform(),
                            step="mid", color="orange", alpha=.10)
            ax.set_xlim(start, end)
            ax.grid(alpha=.2)
        axs[0].legend(fontsize=8, ncol=4)
        axs[2].legend(fontsize=8)
        fig.suptitle(f"{caption} | {segment}\nOrange shading: common maneuver mask", fontsize=13)
        save_figure(fig, output / filename)

    temporal_plot("06_residuals_over_time.png", t[eligible][0], t[-1], "Prediction residuals")
    events = [e for e in context["events"] if e["last_maneuver_sample_s"] >= t[eligible][0]]
    events.sort(key=lambda e: e["peak_acceleration_mps2"], reverse=True)
    for rank, event in enumerate(events[:c.max_event_plots], 1):
        start = max(t[eligible][0], event["start_s"]-c.event_window_seconds)
        end = min(t[-1], event["last_maneuver_sample_s"]+c.event_window_seconds)
        if end > start:
            temporal_plot(f"07_event_{rank:02d}.png", start, end,
                          f"Reference event {event['event_id']} (ranked by peak acceleration)")

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for i, a in enumerate(AXES):
        ax = axs.flat[i]
        if i >= dimension:
            ax.set_axis_off()
            continue
        for (acc, pos) in corners:
            table = subset(acf, model_acceleration_std_g=acc, position_noise_std_m=pos, component=a)
            ax.plot(table.lag_samples, table.acf, marker=".", label=f"{acc:.3g} g, {pos:g} m")
        ax.axhline(0, color="gray", lw=.7)
        ax.set(title=f"Innovation: {a}", xlabel="Lag (samples)", ylabel="Demeaned ACF", ylim=(-1.05, 1.05))
        ax.grid(alpha=.2)
    axs.flat[0].legend(fontsize=8)
    fig.suptitle(f"Innovation serial dependence (descriptive) | {segment}")
    save_figure(fig, output / "08_innovation_autocorrelation.png")

    fig, ax = plt.subplots(figsize=(13, 5), layout="constrained")
    for (acc, pos), run in corners.items():
        ax.plot(t[eligible], run.nis.to_numpy()[eligible], lw=.8, alpha=.8, label=f"{acc:.3g} g, {pos:g} m")
    ax.axhline(dimension, color="black", ls="--", label=f"Ideal-model mean: {dimension}")
    ax.set(xlabel="Time within segment (s)", ylabel="NIS", title=f"NIS diagnostic; not an accuracy ranking | {segment}")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=.2)
    save_figure(fig, output / "09_nis_diagnostic.png")


RESULTS_GUIDE = """KF SWEEP ANALYSIS — READING THE OUTPUTS

INPUT: the LARGE kf_samples.csv, exported by the sweep filter script.
No need for kf_summary.csv or the original flight CSV. No filter is rerun.
The input is read-only. Output files with matching names are replaced on rerun;
use a different output directory when comparing analysis settings.

START HERE
  figures/<segment>/01_prediction_heatmaps.png
  figures/<segment>/02_forecast_heatmaps.png
  figures/<segment>/03_covariance_heatmaps.png
  figures/<segment>/07_event_*.png

TABLES
  residual_metrics.csv: item 1 and item 5. Each row identifies a noise pair,
    segment, prior/posterior phase, regime, and quantity. RMS, median, p95,
    maximum, time of maximum, fraction exceeding the stated tolerance, and
    signed component means. Vertical distance is absolute vertical residual.
    Position units: m; velocity: m/s. Bias columns use the same units.
    Fraction exceeding tolerance lies in [0,1], and uses strict >.
  forecast_metrics.csv: item 3. Same residual statistics for open-loop
    forecasts launched from posterior states. x(t+tau|t) = F(tau) x_posterior(t).
    Position advances as p+tau*v; velocity stays constant. No measurement
    updates, acceleration extrapolation, or future data enter these forecasts.
    EXACT reference target timestamps only; no interpolation. Every horizon
    uses the intersection of eligible origins with valid targets at ALL
    requested horizons. Thus no horizon benefits from different origin times.
    Regime is the regime at forecast ORIGIN; it does not mean the entire
    forecast interval stays in that regime. maximum_time_s is TARGET time.
    maximum_origin_time_s is the corresponding forecast origin.
    One-step prior metrics use all eligible updates, so their window differs
    slightly from the common-origin 1-second forecast metric.
  covariance_metrics.csv: item 4. Separate position and velocity blocks, prior
    and posterior. rms_uncertainty = sqrt(mean(trace(P_block))). It is a
    model-implied RMS magnitude, not an observed error. radius at each time is
    sqrt(trace(P_block)); median/p95/maximum summarize that time series.
    Largest-axis std = sqrt(max eigenvalue(P_block)). No full 6-state trace is
    calculated because position and velocity have different units.
  mean_covariance.csv: all 36 entries of each mean covariance, with state
    labels, units, phase, regime, and sample count. Cross-covariances retained.
  nis_metrics.csv: item 5. Mean, median, p95, maximum and maximum timestamp.
    ideal_model_expected_mean is measurement dimension, not a tuning target
    guaranteed valid for these inputs. Do not rank prediction accuracy by NIS.
  innovation_acf.csv: item 5. Demeaned sample ACF, sum(d[k]*d[k-lag])/sum(d^2).
    All eligible samples; lags in SAMPLES. mean_lag_seconds gives their mean
    time separation. If sampling is irregular this is not a fixed-time-lag ACF.
    Constant innovation gives undefined ACF (empty cell). No independence
    significance bands or validated consistency-test claims are made.
  reference_regimes.csv: item 2. Common reference acceleration, threshold,
    maneuver mask, evaluation masks, and common forecast-origin mask.
  maneuver_events.csv: contiguous above-threshold reference events. Recovery
    starts at the first following below/equal-threshold sample. End-censored
    events have no observed recovery. Events are not clinical/physical labels.
  analysis_settings.json: exact settings, input location, run counts, units,
    and thresholds. Record this with the metrics for reproducibility.

EVALUATION RULES
  Residual = estimated/predicted state MINUS reference observation. RMS is
    sqrt(mean(sum(component residual^2))), not mean distance or mean signed error.
  Reference = original CSV observation_* values; NOT independent ground truth.
    Derived CSV velocities may be correlated with positions and across time.
  Initialization (sample 0) is always excluded. Optional burn-in excludes
    time_s < burn_in_seconds, identically for every configuration in a segment.
    Default burn-in=0 reproduces the earlier summary's evaluation interval.
  Metrics are sample-weighted. No pooling of segments, invented statistical
    significance, overall weighted score, or automatic "best" setting.
  Maneuver mask is computed ONCE per segment using the derivative of centered
    rolling-mean reference velocity (5 seconds by default). This is offline
    segmentation only. Default threshold is its 75th percentile over eligible
    samples. "Lower acceleration" does not assert constant-velocity truth.
    Set maneuver_acceleration_threshold_mps2 to use an absolute threshold.
  Onset windows span +/- event_window_seconds around observed onset. Recovery
    windows span from the first following lower-acceleration sample through
    event_window_seconds later. These subsets can overlap each other and the
    maneuver/lower-acceleration subsets; they are NOT additive partitions.
    Events touching segment boundaries are marked censored.
  Empty subsets produce n_samples=0 and empty metric cells, never false zeros.
  Position/velocity tolerances are illustrative defaults, not requirements.
    For separate horizontal/vertical requirements, edit add_residual_rows.
  Current input contains direct velocity observations with fixed uncertainty.
    Changes in posterior fit and reported P are not independent proof of accuracy.
  For a prescribed linear F,Q,H,R and initialization, P depends on the matrices
    and timestamps, not observed maneuver values. Maneuver covariance summaries
    simply summarize the same deterministic covariance sequence in those windows.

FIGURES
  Heatmaps include all settings with ACTUAL numeric tick labels. Cells are
    equally spaced for readability, not a continuous parameter surface.
    Each panel has its own color scale; use the numbers/colorbars for comparisons.
  Time plots, ECDFs, ACFs, and NIS overlay four sweep corners; CSV tables contain
    all settings. Event plots select up to three events with largest reference
    acceleration peaks, independently of the filter results.
  No changes to the input CSV or filter implementation are made.
"""


def main(config=CONFIG):
    validate_config(config)
    source = resolve_input(config)
    output = Path(config.output_directory).expanduser()
    output = output if output.is_absolute() else BASE_DIR / output
    output.mkdir(parents=True, exist_ok=True)
    print(f"Reading LARGE per-sample file: {source}")
    data, dimension = load_samples(source)
    residual_rows, forecast_rows, cov_rows, mean_rows = [], [], [], []
    nis_rows, acf_rows, mask_frames, event_rows = [], [], [], []
    segment_records = []
    for segment_number, (segment, segdata) in enumerate(data.groupby("segment_id", sort=False)):
        first = next(iter(segdata.groupby(KEYS[1:], sort=True)))[1].sort_values("sample_index")
        context = reference_context(first, config)
        t, obs = context["t"], context["observations"]
        mask_frames.append(pd.DataFrame(dict(segment_id=segment, sample_index=np.arange(len(t)), time_s=t,
                                            reference_acceleration_mps2=context["acceleration"],
                                            maneuver_threshold_mps2=context["threshold"],
                                            raw_maneuver=context["maneuver"],
                                            common_forecast_origin=context["common_origins"], **context["masks"])))
        event_rows.extend(dict(segment_id=segment, **e) for e in context["events"])
        accelerations = np.sort(segdata.model_acceleration_std_g.unique())
        positions = np.sort(segdata.position_noise_std_m.unique())
        wanted = {(float(a), float(p)) for a in (accelerations[0], accelerations[-1]) for p in (positions[0], positions[-1])}
        corners = {}
        start_counts = [len(x) for x in (residual_rows, forecast_rows, cov_rows, acf_rows)]
        nruns = 0
        for (acc, pos), run in segdata.groupby(KEYS[1:], sort=True):
            run = run.sort_values("sample_index")
            validate_run(run, context, dimension)
            meta = dict(segment_id=segment, model_acceleration_std_g=float(acc), position_noise_std_m=float(pos))
            if (acc, pos) in wanted:
                corners[(acc, pos)] = run
            for phase in ("prior", "posterior"):
                states = vector(run, f"x_{phase}")
                add_residual_rows(residual_rows, meta, states-obs, t, context["masks"], config, phase=phase)
                covariance_summaries(cov_rows, mean_rows, meta, covariance(run, phase), phase, context)
            posterior = vector(run, "x_posterior")
            for horizon in config.forecast_horizons_s:
                predicted = posterior.copy()
                predicted[:, :3] += horizon * posterior[:, 3:]
                targets = context["targets"][horizon]
                masks = {regime: mask & context["common_origins"] for regime, mask in context["masks"].items()}
                before = len(forecast_rows)
                add_residual_rows(forecast_rows, meta, predicted-obs[targets], t[targets], masks,
                                  config, horizon_s=horizon)
                for row in forecast_rows[before:]:
                    mask = masks[row["regime"]]
                    columns = QUANTITIES[row["quantity"]][0]
                    distances = np.linalg.norm((predicted-obs[targets])[mask][:, columns], axis=1)
                    row["maximum_origin_time_s"] = (float(t[mask][np.argmax(distances)])
                                                       if len(distances) else np.nan)
            for regime, mask in context["masks"].items():
                values = run.nis.to_numpy(float)[mask]
                stats = dict(mean=np.nan, median=np.nan, p95=np.nan, maximum=np.nan, maximum_time_s=np.nan)
                if len(values):
                    stats = dict(mean=float(values.mean()), median=float(np.median(values)),
                                 p95=float(np.percentile(values, 95)), maximum=float(values.max()),
                                 maximum_time_s=float(t[mask][np.argmax(values)]))
                nis_rows.append(dict(meta, regime=regime, n_samples=len(values),
                                     ideal_model_expected_mean=dimension, **stats))
            mask = context["eligible"]
            for a in AXES[:dimension]:
                values = run[f"innovation_{a}"].to_numpy(float)[mask]
                acf_rows.extend(dict(meta, component=a, **row)
                                for row in autocorrelation(values, t[mask], config.acf_max_lag_samples))
            nruns += 1
        if nruns != len(accelerations)*len(positions):
            warnings.warn(f"{segment}: incomplete rectangular sweep; missing heatmap cells will be blank.")
        segment_records.append(dict(segment_id=segment, runs=nruns, samples_per_run=len(t),
                                    eligible_samples=int(context["eligible"].sum()),
                                    common_forecast_origins=int(context["common_origins"].sum()),
                                    maneuver_threshold_mps2=context["threshold"],
                                    min_dt_s=float(np.diff(t).min()), max_dt_s=float(np.diff(t).max())))
        print(f"{segment}: {nruns} settings, {int(context['masks']['maneuver'].sum())} maneuver samples, "
              f"threshold {context['threshold']:.4g} m/s²")
        if config.save_figures:
            safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(segment))
            folder = output / "figures" / f"{segment_number:02d}_{safe}"
            folder.mkdir(parents=True, exist_ok=True)
            draw_plots(segment, context, corners,
                       pd.DataFrame(residual_rows[start_counts[0]:]),
                       pd.DataFrame(forecast_rows[start_counts[1]:]),
                       pd.DataFrame(cov_rows[start_counts[2]:]),
                       pd.DataFrame(acf_rows[start_counts[3]:]), folder, config, dimension)
    for name, rows in (("residual_metrics", residual_rows), ("forecast_metrics", forecast_rows),
                       ("covariance_metrics", cov_rows), ("mean_covariance", mean_rows),
                       ("nis_metrics", nis_rows), ("innovation_acf", acf_rows)):
        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
    pd.concat(mask_frames, ignore_index=True).to_csv(output / "reference_regimes.csv", index=False)
    event_columns = ["segment_id", "event_id", "start_s", "last_maneuver_sample_s", "recovery_start_s",
                     "duration_between_samples_s", "n_samples", "left_censored", "right_censored", "peak_acceleration_mps2"]
    pd.DataFrame(event_rows, columns=event_columns).to_csv(output / "maneuver_events.csv", index=False)
    settings = dict(config=asdict(config), input_file=str(source), input_bytes=source.stat().st_size,
                    reference="observation_* columns; not independent ground truth", measurement_dimension=dimension,
                    fixed_velocity_noise_std_mps=float(data.velocity_noise_std_mps.iloc[0]),
                    fixed_initial_velocity_std_mps=float(data.initial_velocity_std_mps.iloc[0]),
                    segments=segment_records, numpy_version=np.__version__, pandas_version=pd.__version__,
                    matplotlib_version=matplotlib.__version__)
    (output / "analysis_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (output / "README_results.txt").write_text(RESULTS_GUIDE, encoding="utf-8")
    print(f"Done. Tables, settings, interpretation guide and figures: {output.resolve()}")
    return output


def cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", help="Path to the LARGE kf_samples.csv; not kf_summary.csv")
    parser.add_argument("--output", help="Output directory (relative paths are beside this script)")
    parser.add_argument("--burn-in-seconds", type=float)
    parser.add_argument("--position-tolerance-m", type=float)
    parser.add_argument("--velocity-tolerance-mps", type=float)
    parser.add_argument("--maneuver-threshold-mps2", type=float, help="Absolute threshold instead of reference percentile")
    parser.add_argument("--maneuver-percentile", type=float)
    parser.add_argument("--horizons", type=float, nargs="+", help="Forecast horizons in seconds; default 1 5 10")
    parser.add_argument("--no-figures", action="store_true")
    # Spyder can inject unrelated kernel arguments. There, use editable CONFIG.
    args = parser.parse_args([] if "spyder_kernels" in sys.modules else None)
    updates = {}
    for argument, field_name in (("samples", "input_csv"), ("output", "output_directory"),
                                 ("burn_in_seconds", "burn_in_seconds"),
                                 ("position_tolerance_m", "position_tolerance_m"),
                                 ("velocity_tolerance_mps", "velocity_tolerance_mps"),
                                 ("maneuver_threshold_mps2", "maneuver_acceleration_threshold_mps2"),
                                 ("maneuver_percentile", "maneuver_percentile")):
        if getattr(args, argument) is not None:
            updates[field_name] = getattr(args, argument)
    if args.horizons is not None:
        updates["forecast_horizons_s"] = tuple(args.horizons)
    if args.no_figures:
        updates["save_figures"] = False
    main(replace(CONFIG, **updates))


if __name__ == "__main__":
    cli()
