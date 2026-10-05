"""HAOS six-state sliding-window MLP -- sklearn teaching edition.

Python 3.10+. Install numpy pandas matplotlib scikit-learn joblib.
Place this standalone script beside the 17 HAOS_Flight CSV files and run it.
In Spyder: edit Config below and press F5. No GPU is required.
CLI example: python haos_mlp_sklearn.py --data-dir "C:/drone_data" --no-show

Ordinary feedforward NN: 60 -> 64 ReLU -> 32 ReLU -> 6 linear outputs.
Forecasts the six recorded position/velocity components one second ahead.
No smoothed columns, Kalman updates, simulated truth, LSTM, or transformer.
Position outputs use displacement from the current position; velocity outputs
are future Cartesian velocity. This is NOT the physics-residual extension.

CAUTION FOR INTERPRETATION: O/P/Q and U/V/W are observations, not truth.
The generation of supplied velocity and interpolation is not established here.
If those columns used future samples, this is an offline processed-data benchmark,
not evidence of leakage-free online forecasting. Whole-flight separation prevents
train/test window overlap but cannot undo upstream noncausal preprocessing.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

# ---------------------- SETTINGS TO EDIT IN SPYDER ----------------------
BASE_DIR = Path(__file__).resolve().parent

@dataclass
class Config:
    # Put all 17 HAOS CSVs beside this script, or change data_directory.
    data_directory: str = str(BASE_DIR)
    output_directory: str = str(BASE_DIR / "haos_mlp_sklearn_output")
    history_steps: int = 10       # Ten samples at 1 Hz span nine seconds.
    horizon_steps: int = 1        # Direct prediction 1 second ahead; try 5 later.
    hidden_layers: tuple = (64, 32)
    learning_rate: float = 0.001
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 25            # Stop after this many unimproved epochs.
    seed: int = 42
    show_figures: bool = True
    device: str = "cpu"           # PyTorch only; change to "cuda" if configured.

# Fixed BEFORE training; both programs use exactly the same whole-flight split.
# This overrides the CSV's pre-existing 'split' column intentionally.
# Flight 301 is the sole 300-series flight, so retain it in training here.
TRAIN_IDS = ("01", "03", "101", "102", "104", "105", "201", "202",
             "204", "205", "206", "208", "301")
VALIDATION_IDS = ("02", "203")
TEST_IDS = ("103", "207")
STATE_COLUMNS = ["x_east_m", "y_north_m", "z_up_m",  # O, P, Q
                 "vx_mps", "vy_mps", "vz_mps"]      # U, V, W
LABELS = ["East", "North", "Up", "East velocity", "North velocity", "Up velocity"]


def load_flights(config):
    """Read by column NAME, so reordering CSV columns cannot change the inputs."""
    directory = Path(config.data_directory).expanduser()
    flights = {}
    wanted = set(TRAIN_IDS + VALIDATION_IDS + TEST_IDS)
    for path in sorted(directory.glob("HAOS_Flight_*.csv")):
        df = pd.read_csv(path)
        required = STATE_COLUMNS + ["trajectory_id", "segment_id", "elapsed_s"]
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        ids = df.trajectory_id.dropna().unique()
        if len(ids) != 1 or df.trajectory_id.isna().any():
            raise ValueError(f"{path.name}: expected exactly one trajectory_id")
        flight_id = str(ids[0]).removeprefix("HAOS_Flight_")
        if flight_id not in wanted:
            continue
        if flight_id in flights:
            raise ValueError(f"Duplicate flight {flight_id}; keep only one copy of its CSV")
        numeric = STATE_COLUMNS + ["elapsed_s"]
        df[numeric] = df[numeric].apply(pd.to_numeric, errors="raise")
        if df.empty or df.segment_id.isna().any() or not np.isfinite(df[numeric]).all().all():
            raise ValueError(f"{path.name}: empty data or missing/nonfinite values")
        # Do not silently sort, interpolate, or bridge a discontinuity.
        for sid, segment in df.groupby("segment_id", sort=False):
            dt = np.diff(segment.elapsed_s.to_numpy())
            if not np.allclose(dt, 1.0, rtol=0, atol=1e-6):
                raise ValueError(f"{sid}: expected 1-second steps; split/resample explicitly")
        flights[flight_id] = (path.name, df)
    missing = wanted - set(flights)
    if missing:
        raise FileNotFoundError(f"Missing flights {sorted(missing)} in {directory}")
    return flights


def build_windows(flights, ids, config):
    """Convert each segment into ordinary (input vector, target vector) pairs.

    For t = current observation, inputs contain ONLY t-L+1,...,t.
    Target = [position(t+h)-position(t), velocity(t+h)].
    Historical positions are translated relative to position(t); velocities
    remain in the original East/North/Up coordinate frame. Translating removes
    irrelevant absolute location but retains motion and axis orientation.

    This is a direct forecast with refreshed observed history at every t,
    NOT an autonomous rollout feeding earlier predictions back as inputs.
    """
    xs, ys, origins, references, baselines, metadata = [], [], [], [], [], []
    L, h = config.history_steps, config.horizon_steps
    for fid in ids:
        _, df = flights[fid]
        count_before = len(xs)
        for sid, seg in df.groupby("segment_id", sort=False):
            states = seg[STATE_COLUMNS].to_numpy(dtype=np.float64)
            times = seg.elapsed_s.to_numpy(dtype=np.float64)
            for t in range(L - 1, len(states) - h):
                # copy() prevents in-place subtraction from changing source data.
                history = states[t-L+1:t+1].copy()
                origin = states[t, :3].copy()
                history[:, :3] -= origin
                target = states[t+h].copy()
                target[:3] -= origin
                xs.append(history.reshape(-1))  # oldest six features first
                ys.append(target)
                origins.append(origin)
                references.append(states[t+h].copy())
                dt = times[t+h] - times[t]
                baselines.append(np.r_[origin + dt*states[t, 3:], states[t, 3:]])
                metadata.append((fid, str(sid), times[t], times[t+h]))
        if len(xs) == count_before:
            raise ValueError(f"Flight {fid}: no windows; shorten history/horizon")
    return {"X": np.asarray(xs), "y": np.asarray(ys), "origin": np.asarray(origins),
            "reference": np.asarray(references), "cv": np.asarray(baselines),
            "meta": pd.DataFrame(metadata, columns=["flight_id", "segment_id",
                                                     "origin_time_s", "target_time_s"])}


def recover_states(scaled_outputs, target_scaler, origins):
    """Undo target scaling, then undo the local position translation."""
    states = target_scaler.inverse_transform(scaled_outputs)
    states[:, :3] += origins
    return states


def predict_history(history, checkpoint):
    """Public inference helper: an L x 6 history -> a six-component forecast.

    Rows must be consecutive 1-second observations, oldest first, in
    STATE_COLUMNS order. Forecast horizon is stored in the checkpoint.
    For example: predict_history(last_ten_states, load_checkpoint(path)).
    """
    history = np.asarray(history, dtype=float)
    L = checkpoint["config"]["history_steps"]
    if history.shape != (L, 6) or not np.isfinite(history).all():
        raise ValueError(f"Expected finite history with shape ({L}, 6)")
    local = history.copy()
    origin = local[-1, :3].copy()
    local[:, :3] -= origin
    scaled = checkpoint["input_scaler"].transform(local.reshape(1, -1))
    prediction = predict_scaled(checkpoint["model"], scaled)
    return recover_states(prediction, checkpoint["target_scaler"], origin[None, :])[0]


def metric_row(split, flight, method, reference, prediction):
    error = prediction - reference
    # Vector RMSE = sqrt(mean(||error_vector||^2)), not component-averaged RMSE.
    row = {"split": split, "flight": flight, "method": method, "samples": len(error),
           "position_vector_RMSE_m": float(np.sqrt(np.mean(np.sum(error[:, :3]**2, axis=1)))),
           "velocity_vector_RMSE_mps": float(np.sqrt(np.mean(np.sum(error[:, 3:]**2, axis=1))))}
    for i, name in enumerate(STATE_COLUMNS):
        row[name + "_RMSE"] = float(np.sqrt(np.mean(error[:, i]**2)))
    return row


def evaluate(split, data, prediction, output, config):
    """Save full numeric predictions and plot all six states for each segment."""
    table = data["meta"].copy()
    for j, name in enumerate(STATE_COLUMNS):
        table["observed_" + name] = data["reference"][:, j]
        table["mlp_" + name] = prediction[:, j]
        table["cv_" + name] = data["cv"][:, j]
    table.to_csv(output / f"{split}_predictions.csv", index=False)
    rows = []
    groups = [("ALL", np.ones(len(table), dtype=bool))]
    groups += [(fid, table.flight_id.to_numpy() == fid) for fid in table.flight_id.unique()]
    for fid, mask in groups:
        for name, values in [("MLP", prediction), ("Constant velocity", data["cv"])]:
            rows.append(metric_row(split, fid, name, data["reference"][mask], values[mask]))
    # Plot test only; avoid dozens of windows during a live lesson.
    if split == "test":
        for sid in table.segment_id.unique():
            mask = table.segment_id.to_numpy() == sid
            time = table.loc[mask, "target_time_s"].to_numpy()
            fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True, constrained_layout=True)
            for j in range(6):
                ax = axes[j % 3, j // 3]
                ax.plot(time, data["reference"][mask, j], color="black", lw=1.4, label="Recorded observation")
                ax.plot(time, prediction[mask, j], color="tab:blue", lw=1, label="MLP forecast")
                ax.plot(time, data["cv"][mask, j], color="tab:orange", lw=.9, alpha=.7,
                        linestyle="--", label="Constant-velocity forecast")
                ax.set_ylabel(LABELS[j] + (" (m)" if j < 3 else " (m/s)"))
                ax.grid(alpha=.25)
            axes[0, 0].legend(fontsize=8)
            axes[2, 0].set_xlabel("Target time within segment (s)")
            axes[2, 1].set_xlabel("Target time within segment (s)")
            fig.suptitle(f"{sid}: {config.horizon_steps}-second forecasts vs recorded observations")
            fig.savefig(output / f"{sid}_six_states.png", dpi=160)
            if not config.show_figures:
                plt.close(fig)
            fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
            for label, values, color in [("MLP", prediction, "tab:blue"),
                                          ("Constant velocity", data["cv"], "tab:orange")]:
                err = values[mask] - data["reference"][mask]
                axes[0].plot(time, np.linalg.norm(err[:, :3], axis=1), label=label, color=color)
                axes[1].plot(time, np.linalg.norm(err[:, 3:], axis=1), label=label, color=color)
            axes[0].set_ylabel("Position error norm (m)")
            axes[1].set_ylabel("Velocity error norm (m/s)")
            axes[1].set_xlabel("Target time within segment (s)")
            axes[0].set_title(f"{sid}: forecast differences from observations (not truth errors)")
            axes[0].legend()
            for ax in axes:
                ax.grid(alpha=.25)
            fig.savefig(output / f"{sid}_forecast_errors.png", dpi=160)
            if not config.show_figures:
                plt.close(fig)
    return rows


def main(config=None):
    config = config or Config()
    if min(config.history_steps, config.horizon_steps, config.max_epochs,
           config.patience, config.batch_size) < 1:
        raise ValueError("History, horizon, epochs, patience and batch size must be positive")
    np.random.seed(config.seed)
    output = Path(config.output_directory).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    print("1. Loading Cartesian O/P/Q and velocity U/V/W; no smoothed columns used.")
    print("Targets are recorded observations, not ground truth. Velocity preprocessing provenance is unverified.")
    flights = load_flights(config)
    splits = {"train": TRAIN_IDS, "validation": VALIDATION_IDS, "test": TEST_IDS}
    manifest = []
    for split, ids in splits.items():
        print(f"   {split}: {', '.join(ids)}")
        for fid in ids:
            name, df = flights[fid]
            manifest.append({"flight": fid, "split": split, "filename": name,
                             "rows": len(df), "segments": df.segment_id.nunique()})
    pd.DataFrame(manifest).to_csv(output / "flight_split.csv", index=False)
    print("2. Building windows independently within each segment.")
    datasets = {name: build_windows(flights, ids, config) for name, ids in splits.items()}
    for name, data in datasets.items():
        print(f"   {name}: X {data['X'].shape}, y {data['y'].shape}")
    print("3. Fitting input AND output scalers on training flights only.")
    input_scaler = StandardScaler().fit(datasets["train"]["X"])
    target_scaler = StandardScaler().fit(datasets["train"]["y"])
    X = {name: input_scaler.transform(d["X"]).astype(np.float32) for name, d in datasets.items()}
    y = {name: target_scaler.transform(d["y"]).astype(np.float32) for name, d in datasets.items()}
    print(f"4. Training {6*config.history_steps} -> {config.hidden_layers} -> 6 MLP.")
    print("   Validation selects the best epoch; test data never enters training or selection.")
    with threadpool_limits(limits=1):
        model, history, best_epoch = train_model(X["train"], y["train"],
                                                X["validation"], y["validation"], config)
        predictions = {name: recover_states(predict_scaled(model, X[name]), target_scaler,
                                             datasets[name]["origin"])
                       for name in ("validation", "test")}
    history.to_csv(output / "learning_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(history.epoch, history.train_mse, label="Training")
    ax.plot(history.epoch, history.validation_mse, label="Validation (whole held-out flights)")
    ax.axvline(best_epoch, color="gray", ls="--", label=f"Selected epoch {best_epoch}")
    ax.set(xlabel="Epoch", ylabel="MSE in standardized target coordinates", title="MLP learning curve")
    ax.set_yscale("log")
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(output / "learning_curve.png", dpi=160)
    if not config.show_figures:
        plt.close(fig)
    print("5. Evaluating direct forecasts against recorded observations.")
    metrics = []
    for name in ("validation", "test"):
        metrics += evaluate(name, datasets[name], predictions[name], output, config)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(output / "metrics.csv", index=False)
    print(metrics.loc[metrics.flight == "ALL", ["split", "method", "samples",
          "position_vector_RMSE_m", "velocity_vector_RMSE_mps"]].to_string(index=False))
    checkpoint = {"model": model, "input_scaler": input_scaler, "target_scaler": target_scaler,
                  "config": asdict(config), "state_columns": STATE_COLUMNS, "best_epoch": best_epoch}
    save_checkpoint(checkpoint, output)
    summary = {"config": asdict(config), "best_epoch": best_epoch,
               "split": splits, "python": platform.python_version(), "versions": versions(),
               "interpretation": "Forecast errors against recorded observations, not ground truth; "
               "CSV velocity causality and interpolation provenance not established."}
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"6. Saved model, scalers, metrics, predictions and figures to {output}")
    if config.show_figures:
        plt.show()
    return checkpoint, metrics


def command_line_config():
    """Spyder users can edit Config above; command-line users can override it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--history", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args()
    config = Config()
    for source, destination in [("data_dir", "data_directory"), ("output_dir", "output_directory"),
        ("history", "history_steps"), ("horizon", "horizon_steps"), ("epochs", "max_epochs"),
        ("device", "device")]:
        value = getattr(args, source)
        if value is not None:
            setattr(config, destination, value)
    config.show_figures = not args.no_show
    return config

# ------------------------ SCIKIT-LEARN TRAINING -------------------------
import joblib
import sklearn
from sklearn.neural_network import MLPRegressor


def train_model(X_train, y_train, X_val, y_val, config):
    # partial_fit performs ONE epoch and retains Adam's optimizer state.
    # Built-in early_stopping would randomly carve overlapping training windows
    # into a validation set. Instead, explicitly use our two validation flights.
    model = MLPRegressor(hidden_layer_sizes=config.hidden_layers, activation="relu",
                         solver="adam", alpha=0.0001, batch_size=config.batch_size,
                         learning_rate_init=config.learning_rate, shuffle=True,
                         random_state=config.seed, early_stopping=False)
    best_model, best_loss, best_epoch, stale = None, np.inf, 0, 0
    records = []
    for epoch in range(1, config.max_epochs + 1):
        model.partial_fit(X_train, y_train)
        train_loss = float(np.mean((model.predict(X_train) - y_train)**2))
        val_loss = float(np.mean((model.predict(X_val) - y_val)**2))
        if not np.isfinite(train_loss + val_loss):
            raise FloatingPointError("Nonfinite loss; check data and learning rate")
        records.append((epoch, train_loss, val_loss))
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_model = copy.deepcopy(model)
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"   Epoch {epoch:3d}: training MSE={train_loss:.6f}; validation MSE={val_loss:.6f}")
        if stale >= config.patience:
            break
    print(f"   Restoring epoch {best_epoch}, validation MSE={best_loss:.6f}")
    return best_model, pd.DataFrame(records, columns=["epoch", "train_mse", "validation_mse"]), best_epoch


def predict_scaled(model, X):
    return model.predict(X)


def save_checkpoint(checkpoint, output):
    joblib.dump(checkpoint, output / "mlp_sklearn.joblib")


def load_checkpoint(path):
    # Only load model files you trust; joblib uses Python pickle internally.
    return joblib.load(path)


def versions():
    return {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}


if __name__ == "__main__":
    main(command_line_config())
