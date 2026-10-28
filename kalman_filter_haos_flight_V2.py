"""HAOS drone-data adaptation of the Cartesian Kalman-filter demonstration.

Install: pip install numpy pandas matplotlib
Place HAOS_Flight_403.csv beside this script, then run it (also works in Spyder).

Uses O/P/Q = x_east_m/y_north_m/z_up_m and U/V/W = vx_mps/vy_mps/vz_mps.
State: [east, north, up, east_velocity, north_velocity, up_velocity].
Retains the original constant-velocity prediction and Joseph-form update,
expanded to 3-D. Acceleration is process noise, NOT an estimated state.
No simulated measurements or additional noise are generated.

Each segment is filtered independently for all 49 noise combinations and gets
four figures, so gaps are not joined. Acceleration std is in g, position std
in meters. kf_samples.csv records prior/posterior states and full covariance
matrices (position units m, velocity units m/s); covariance entries have the
product of their row/column units. Prior quantities and NIS are undefined at
initialization and exported as empty cells. kf_summary.csv excludes that row.
R uses adjustable illustrative uncertainties, NOT calibrated sensor specs.
CSV velocities may be derived from positions; the diagonal R approximation
ignores any such correlations. NIS is therefore a diagnostic, not a validated
consistency test. Input residuals are NOT ground-truth estimation errors.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STANDARD_GRAVITY_MPS2 = 9.80665
BASE_DIR = Path(__file__).resolve().parent

@dataclass(frozen=True)
class Config:
    csv_file: str = "HAOS_Flight_403.csv"  # Relative to this script, or absolute.
    output_directory: str = "haos_kalman_output"
    segment_id: str | None = None  # None processes all; or "HAOS_Flight_01_seg00".
    measure_velocity: bool = True
    position_noise_std_m: list[float] = field(default_factory=lambda: [5, 10, 20, 35, 50, 65, 100])
    velocity_noise_std_mps: float = 1.0  # Illustrative tuning parameter.
    initial_velocity_std_mps: float = 20.0  # Position-only initialization.
    model_acceleration_std_g: list[float] = field(default_factory=lambda: np.linspace(0.05, 0.75, 7).tolist())
    show_figures: bool = False
    save_figures: bool = True
    trajectory_figure: str = "haos_01_cartesian_trajectory"
    speed_figure: str = "haos_02_speed_response"
    position_error_figure: str = "haos_03_position_residuals"
    nis_figure: str = "haos_04_normalized_innovation_squared"


def load_flight_data(config: Config) -> pd.DataFrame:
    path = Path(config.csv_file).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    df = pd.read_csv(path)
    columns = ["elapsed_s", "x_east_m", "y_north_m", "z_up_m",
               "vx_mps", "vy_mps", "vz_mps"]
    missing = set(columns + ["segment_id"]) - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    df[columns] = df[columns].apply(pd.to_numeric, errors="raise")
    if df.empty or df.segment_id.isna().any() or not np.isfinite(df[columns].to_numpy()).all():
        raise ValueError("CSV is empty or contains missing/nonfinite required values.")
    if config.segment_id is not None:
        df = df.loc[df.segment_id == config.segment_id].copy()
        if df.empty:
            raise ValueError(f"Segment not found: {config.segment_id}")
    return df


def make_filter_matrices(config: Config, dt: float, position_noise_std_m: float,
                         model_acceleration_std_g: float) -> tuple[np.ndarray, ...]:
    """Same F, G, Qa, Q, H, R construction as original, extended to 3 axes."""
    f = np.eye(6)
    f[:3, 3:] = dt * np.eye(3)
    g = np.vstack((0.5 * dt**2 * np.eye(3), dt * np.eye(3)))
    qa = (model_acceleration_std_g * STANDARD_GRAVITY_MPS2)**2 * np.eye(3)
    q = g @ qa @ g.T
    if config.measure_velocity:
        h = np.eye(6)
        r = np.diag([position_noise_std_m**2] * 3
                    + [config.velocity_noise_std_mps**2] * 3)
    else:
        h = np.eye(6)[:3]
        r = position_noise_std_m**2 * np.eye(3)
    return f, g, qa, q, h, r


def run_kalman_filter(
    config: Config,
    measurements: np.ndarray,
    f: np.ndarray,
    q: np.ndarray,
    h: np.ndarray,
    r: np.ndarray,
    time_s: np.ndarray,
    position_noise_std_m: float,
    model_acceleration_std_g: float,
) -> tuple[np.ndarray, ...]:
    """Run the predict/update recursion using the Joseph covariance update."""
    sample_count = measurements.shape[0]
    estimates = np.zeros((sample_count, 6))
    covariances = np.zeros((sample_count, 6, 6))
    innovations = np.full((sample_count, h.shape[0]), np.nan)
    predictions = np.full((sample_count, 6), np.nan)
    prior_covariances = np.full((sample_count, 6, 6), np.nan)
    nis = np.full(sample_count, np.nan)

    # Initialize from the first observation; do not update it twice.
    estimates[0, :h.shape[0]] = measurements[0]
    # For position-only measurements, start with zero velocity and broad uncertainty.
    covariances[0] = np.diag(
        [4.0 * position_noise_std_m**2] * 3
        + [4.0 * (config.velocity_noise_std_mps if config.measure_velocity
                  else config.initial_velocity_std_mps)**2] * 3
    )
    identity = np.eye(6)

    for k in range(1, sample_count):
        f, _g, _qa, q, _h, _r = make_filter_matrices(
            config, time_s[k] - time_s[k - 1], position_noise_std_m,
            model_acceleration_std_g)

        # Predict: p(x_k | z_1:k-1)
        x_pred = f @ estimates[k - 1]
        p_pred = f @ covariances[k - 1] @ f.T + q

        predictions[k] = x_pred
        prior_covariances[k] = p_pred

        # Update: p(x_k | z_1:k)
        innovation = measurements[k] - h @ x_pred
        innovation_covariance = h @ p_pred @ h.T + r

        # K = P_pred H^T S^-1.  solve() is preferable to an explicit inverse.
        pht = p_pred @ h.T
        kalman_gain = np.linalg.solve(innovation_covariance.T, pht.T).T

        estimates[k] = x_pred + kalman_gain @ innovation

        # Joseph form preserves symmetry and positive semidefiniteness better.
        a = identity - kalman_gain @ h
        p_updated = a @ p_pred @ a.T + kalman_gain @ r @ kalman_gain.T
        covariances[k] = 0.5 * (p_updated + p_updated.T)

        innovations[k] = innovation
        nis[k] = innovation @ np.linalg.solve(
            innovation_covariance, innovation
        )

    return estimates, covariances, innovations, nis, predictions, prior_covariances


def plot_results(config, segment, time_s, observations, estimates, covariances, nis, acc_noise, pos_noise):
    """Four original diagnostic types, with real-data labels and all three axes."""
    output = Path(config.output_directory).expanduser()
    if not output.is_absolute():
        output = BASE_DIR / output
    output.mkdir(parents=True, exist_ok=True)
    safe_segment = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(segment))

    def save(fig, name):
        fig.suptitle(f"Acceleration std: {acc_noise:.6g} g | Position std: {pos_noise:g} m")
        path = output / f"{safe_segment}_a{acc_noise:.8g}g_p{pos_noise:g}m_{name}.jpeg"
        fig.savefig(path, dpi=180)
        print(f"  Saved {path}")
        if not config.show_figures:
            plt.close(fig)

    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*observations[:, :3].T, color="#d95f02", alpha=.6, label="CSV position")
    ax.plot(*estimates[:, :3].T, color="#1f77b4", label="Kalman estimate")
    ax.scatter(*observations[0, :3], marker="*", s=100, color="green", label="Start")
    ax.set(xlabel="East (m)", ylabel="North (m)", zlabel="Up (m)",
#           title=f"Six-state Kalman filter: {segment} Acc Noise: {acc_noise}, Pos Noise: {pos_noise}")
           title=f"Six-state Kalman filter: {segment}")
    ax.legend()
    save(fig, config.trajectory_figure)

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.plot(time_s, np.linalg.norm(observations[:, 3:], axis=1),
            color="#d95f02", alpha=.65, label="CSV speed (from U/V/W)")
    ax.plot(time_s, np.linalg.norm(estimates[:, 3:], axis=1),
            color="#1f77b4", label="Kalman speed")
    ax.set(xlabel="Time within segment (s)", ylabel="3-D speed (m/s)",
           title=f"Speed response: {segment}")
    ax.grid(alpha=.25)
    ax.legend()
    save(fig, config.speed_figure)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    for j, (ax, label) in enumerate(zip(axes, ["East", "North", "Up"])):
        ax.plot(time_s, estimates[:, j] - observations[:, j], label="Estimate minus CSV position")
        sigma = np.sqrt(np.maximum(covariances[:, j, j], 0))
        ax.plot(time_s, 2*sigma, "--", color="gray", label="Filter state +/-2 sigma")
        ax.plot(time_s, -2*sigma, "--", color="gray")
        ax.set_ylabel(f"{label} (m)")
        ax.grid(alpha=.25)
    axes[0].set_title(f"Position residuals (not truth errors): {segment}")
    axes[0].legend()
    axes[-1].set_xlabel("Time within segment (s)")
    # State covariance bounds are shown for scale, not residual confidence bounds.
    save(fig, config.position_error_figure)

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    dimension = 6 if config.measure_velocity else 3
    ax.plot(time_s, nis, color="#8c564b")
    ax.axhline(dimension, color="black", linestyle="--",
               label=f"Ideal-model expected mean = {dimension}")
    ax.set(xlabel="Time within segment (s)", ylabel="NIS",
           title=f"Normalized innovation squared: {segment}")
    ax.grid(alpha=.25)
    ax.legend()
    save(fig, config.nis_figure)


def main(config: Config | None = None) -> None:
    config = Config() if config is None else config
    for name in ("position_noise_std_m", "velocity_noise_std_mps",
                 "initial_velocity_std_mps", "model_acceleration_std_g"):
        values = np.atleast_1d(getattr(config, name)).astype(float)
        if values.size == 0 or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"{name} must contain finite positive values")
    df = load_flight_data(config)
    output = Path(config.output_directory).expanduser()
    if not output.is_absolute():
        output = BASE_DIR / output
    output.mkdir(parents=True, exist_ok=True)
    labels = ["east", "north", "up", "ve", "vn", "vu"]
    summaries = []
    print("Residuals are comparisons with CSV observations, not truth errors.")
    # One fresh recursion per combination and segment. Time is always innermost.
    with (output / "kf_samples.csv").open("w", newline="") as sample_file:
        first = True
        for ii, acc_noise in enumerate(config.model_acceleration_std_g):
            for jj, pos_noise in enumerate(config.position_noise_std_m):
                for segment, group in df.groupby("segment_id", sort=False):
                    time_s = group.elapsed_s.to_numpy(dtype=float)
                    time_s = time_s - time_s[0]
                    if len(time_s) < 2 or np.any(np.diff(time_s) <= 0):
                        raise ValueError(f"{segment}: need at least two strictly increasing times")
                    observations = group[["x_east_m", "y_north_m", "z_up_m",
                                          "vx_mps", "vy_mps", "vz_mps"]].to_numpy(dtype=float)
                    measurements = observations if config.measure_velocity else observations[:, :3]
                    f, _g, _qa, q, h, r = make_filter_matrices(
                        config, time_s[1] - time_s[0], pos_noise, acc_noise)
                    estimates, covariances, innovations, nis, predictions, priors = run_kalman_filter(
                        config, measurements, f, q, h, r, time_s, pos_noise, acc_noise)
                    meta = dict(acceleration_index=ii, position_index=jj,
                                model_acceleration_std_g=acc_noise, position_noise_std_m=pos_noise,
                                velocity_noise_std_mps=config.velocity_noise_std_mps,
                                initial_velocity_std_mps=config.initial_velocity_std_mps,
                                measure_velocity=config.measure_velocity, segment_id=segment)
                    columns = dict(meta, sample_index=np.arange(len(time_s)), time_s=time_s,
                                   is_initialization=np.arange(len(time_s)) == 0, nis=nis)
                    for j, label in enumerate(labels):
                        columns[f"observation_{label}"] = observations[:, j]
                        columns[f"x_prior_{label}"] = predictions[:, j]
                        columns[f"x_posterior_{label}"] = estimates[:, j]
                        if j < h.shape[0]:
                            columns[f"innovation_{label}"] = innovations[:, j]
                        for k, other in enumerate(labels):
                            columns[f"P_prior_{label}_{other}"] = priors[:, j, k]
                            columns[f"P_posterior_{label}_{other}"] = covariances[:, j, k]
                    pd.DataFrame(columns).to_csv(sample_file, index=False, header=first)
                    first = False
                    summary = dict(meta, samples=len(time_s), duration_s=time_s[-1],
                                   mean_nis=float(np.mean(nis[1:])), measurement_dimension=h.shape[0])
                    # Exclude initialization for every metric. Units: position m, velocity m/s.
                    for phase, states in (("prior", predictions), ("posterior", estimates)):
                        residual = states[1:] - observations[1:]
                        for j, label in enumerate(labels):
                            summary[f"{phase}_residual_rms_{label}"] = np.sqrt(np.mean(residual[:, j]**2))
                        summary[f"{phase}_position_residual_rms_3d_m"] = np.sqrt(np.mean(np.sum(residual[:, :3]**2, axis=1)))
                        summary[f"{phase}_velocity_residual_rms_3d_mps"] = np.sqrt(np.mean(np.sum(residual[:, 3:]**2, axis=1)))
                    for j, label in enumerate(labels):
                        summary[f"mean_posterior_variance_{label}"] = np.mean(covariances[1:, j, j])
                    summaries.append(summary)
                    print(f"{segment}: acceleration={acc_noise:.6g} g, position={pos_noise:g} m; mean NIS={summary['mean_nis']:.3f}")
                    if config.save_figures:
                        plot_results(config, segment, time_s, observations, estimates,
                                     covariances, nis, acc_noise, pos_noise)
                        if config.show_figures:
                            plt.show()
    pd.DataFrame(summaries).to_csv(output / "kf_summary.csv", index=False)
    print(f"Saved {len(summaries)} runs to {output}")


if __name__ == "__main__":
    main()
