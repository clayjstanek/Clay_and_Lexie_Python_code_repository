"""HAOS drone-data adaptation of the Cartesian Kalman-filter demonstration.

Install: pip install numpy pandas matplotlib
Place HAOS_Flight_01.csv beside this script, then run it (also works in Spyder).

Uses O/P/Q = x_east_m/y_north_m/z_up_m and U/V/W = vx_mps/vy_mps/vz_mps.
State: [east, north, up, east_velocity, north_velocity, up_velocity].
Retains the original constant-velocity prediction and Joseph-form update,
expanded to 3-D. Acceleration is process noise, NOT an estimated state.
No simulated measurements or additional noise are generated.

The file contains three segments with elapsed_s restarting at zero. Each is
filtered independently and gets four figures, so gaps are not joined.
R uses adjustable illustrative uncertainties, NOT calibrated sensor specs.
CSV velocities may be derived from positions; the diagonal R approximation
ignores any such correlations. NIS is therefore a diagnostic, not a validated
consistency test. Input residuals are NOT ground-truth estimation errors.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STANDARD_GRAVITY_MPS2 = 9.80665
BASE_DIR = Path(__file__).resolve().parent

@dataclass(frozen=True)
class Config:
    csv_file: str = "HAOS_Flight_01.csv"  # Relative to this script, or absolute.
    output_directory: str = "haos_kalman_output"
    segment_id: str | None = None  # None processes all; or "HAOS_Flight_01_seg00".
    measure_velocity: bool = True
    position_noise_std_m: float = 5.0  # Illustrative tuning parameter.
    velocity_noise_std_mps: float = 1.0  # Illustrative tuning parameter.
    initial_velocity_std_mps: float = 20.0  # Position-only initialization.
    model_acceleration_std_g: float = 0.10
    show_figures: bool = True
    trajectory_figure: str = "haos_01_cartesian_trajectory.png"
    speed_figure: str = "haos_02_speed_response.png"
    position_error_figure: str = "haos_03_position_residuals.png"
    nis_figure: str = "haos_04_normalized_innovation_squared.png"


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


def make_filter_matrices(config: Config, dt: float) -> tuple[np.ndarray, ...]:
    """Same F, G, Qa, Q, H, R construction as original, extended to 3 axes."""
    f = np.eye(6)
    f[:3, 3:] = dt * np.eye(3)
    g = np.vstack((0.5 * dt**2 * np.eye(3), dt * np.eye(3)))
    qa = (config.model_acceleration_std_g * STANDARD_GRAVITY_MPS2)**2 * np.eye(3)
    q = g @ qa @ g.T
    if config.measure_velocity:
        h = np.eye(6)
        r = np.diag([config.position_noise_std_m**2] * 3
                    + [config.velocity_noise_std_mps**2] * 3)
    else:
        h = np.eye(6)[:3]
        r = config.position_noise_std_m**2 * np.eye(3)
    return f, g, qa, q, h, r


def run_kalman_filter(
    config: Config,
    measurements: np.ndarray,
    f: np.ndarray,
    q: np.ndarray,
    h: np.ndarray,
    r: np.ndarray,
    time_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the predict/update recursion using the Joseph covariance update."""
    sample_count = measurements.shape[0]
    estimates = np.zeros((sample_count, 6))
    covariances = np.zeros((sample_count, 6, 6))
    innovations = np.zeros((sample_count, h.shape[0]))
    nis = np.full(sample_count, np.nan)

    # Initialize from the first observation; do not update it twice.
    estimates[0, :h.shape[0]] = measurements[0]
    # For position-only measurements, start with zero velocity and broad uncertainty.
    covariances[0] = np.diag(
        [4.0 * config.position_noise_std_m**2] * 3
        + [4.0 * (config.velocity_noise_std_mps if config.measure_velocity
                  else config.initial_velocity_std_mps)**2] * 3
    )
    identity = np.eye(6)

    for k in range(1, sample_count):
        f, _g, _qa, q, _h, _r = make_filter_matrices(config, time_s[k] - time_s[k - 1])

        # Predict: p(x_k | z_1:k-1)
        x_pred = f @ estimates[k - 1]
        p_pred = f @ covariances[k - 1] @ f.T + q

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

    return estimates, covariances, innovations, nis


def plot_results(config, segment, time_s, observations, estimates, covariances, nis):
    """Four original diagnostic types, with real-data labels and all three axes."""
    output = Path(config.output_directory).expanduser()
    if not output.is_absolute():
        output = BASE_DIR / output
    output.mkdir(parents=True, exist_ok=True)
    safe_segment = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(segment))

    def save(fig, name):
        path = output / f"{safe_segment}_{name}"
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


def main() -> None:
    config = Config()
    for name in ("position_noise_std_m", "velocity_noise_std_mps",
                 "initial_velocity_std_mps", "model_acceleration_std_g"):
        value = getattr(config, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    df = load_flight_data(config)
    print("3-D HAOS Kalman filter: CSV measurements, no synthetic noise")
    print("R uncertainties are illustrative; residuals are not truth errors.")
    for segment, group in df.groupby("segment_id", sort=False):
        time_s = group.elapsed_s.to_numpy(dtype=float)
        time_s = time_s - time_s[0]
        if len(time_s) < 2 or np.any(np.diff(time_s) <= 0):
            raise ValueError(f"{segment}: need at least two strictly increasing times")
        observations = group[["x_east_m", "y_north_m", "z_up_m",
                              "vx_mps", "vy_mps", "vz_mps"]].to_numpy(dtype=float)
        measurements = observations if config.measure_velocity else observations[:, :3]
        f, _g, _qa, q, h, r = make_filter_matrices(config, time_s[1] - time_s[0])
        estimates, covariances, _innovations, nis = run_kalman_filter(
            config, measurements, f, q, h, r, time_s)
        print(f"\n{segment}: {len(time_s)} samples; {time_s[-1]:.1f} seconds")
        print("Position residual RMS [E,N,U] (m):",
              np.sqrt(np.mean((estimates[:, :3] - observations[:, :3])**2, axis=0)))
        print(f"Mean NIS after initialization: {np.nanmean(nis):.3f}")
        plot_results(config, segment, time_s, observations, estimates, covariances, nis)
    if config.show_figures:
        plt.show()


if __name__ == "__main__":
    main()
