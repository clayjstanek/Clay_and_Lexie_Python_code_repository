"""
Chapter 8 companion example:
The SAME 2 -> 4 -> 1 neural-network classification problem, now using scikit-learn.

Purpose
-------
This script is intentionally short and heavily commented so a student can compare:

    1. the earlier "from-scratch" NumPy implementation, and
    2. the same model built with scikit-learn's MLPClassifier.

The data, Gaussian parameters, random seed, train/hold-out split, hidden-layer
size, sigmoid hidden activation, mini-batch size, learning rate, and plot
limits all match the earlier Chapter 8 example as closely as practical.

Important note
--------------
scikit-learn controls the low-level weight initialization and optimization
details internally, so matching the same RANDOM_SEED does NOT guarantee that
the initial weights are numerically identical to the hand-built NumPy network.

The point of the comparison is architectural and procedural equivalence:
same data -> same 2-4-1 architecture -> same classification objective ->
far less user-written training code.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)

# -------------------------------------------------------------------------
# Configuration -- deliberately mirrors the NumPy-from-scratch script
# -------------------------------------------------------------------------

RANDOM_SEED = 42
N_PER_CLASS = 1000
TRAIN_FRACTION = 0.80
BATCH_SIZE = 16
LEARNING_RATE = 0.05
MAX_EPOCHS = 1000

CSV_FILENAME = "chapter8_2d_gaussian_classification_v2.csv"

MU0 = np.array([3.0, 2.0])
COV0 = np.array([[3.0, 0.0],
                 [0.0, 1.0]])

MU1 = np.array([-2.0, 0.0])
COV1 = np.array([[1.5, 1.0],
                 [1.0, 3.0]])

PLOT_X1_LIMITS = (-12.0, 12.0)
PLOT_X2_LIMITS = (-12.0, 12.0)


def generate_dataset(seed=RANDOM_SEED):
    """Generate the exact same two Gaussian classes used in the NumPy example."""
    rng = np.random.default_rng(seed)

    X0 = rng.multivariate_normal(MU0, COV0, size=N_PER_CLASS)
    X1 = rng.multivariate_normal(MU1, COV1, size=N_PER_CLASS)

    df0 = pd.DataFrame(X0, columns=["x1", "x2"])
    df0["class"] = 0

    df1 = pd.DataFrame(X1, columns=["x1", "x2"])
    df1["class"] = 1

    df = pd.concat([df0, df1], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # We deliberately do not overwrite the shared CSV here.  The exact same
    # samples are recreated deterministically from RANDOM_SEED, MU0/COV0,
    # and MU1/COV1, so the scikit-learn comparison remains reproducible.
    return df


def stratified_train_holdout(df, train_fraction=TRAIN_FRACTION, seed=RANDOM_SEED):
    """
    Reproduce the same manual 80/20 stratified split used in the original script.
    """
    rng = np.random.default_rng(seed)

    train_indices = []
    holdout_indices = []

    for cls in sorted(df["class"].unique()):
        idx = df.index[df["class"] == cls].to_numpy()
        rng.shuffle(idx)

        n_train = int(round(train_fraction * len(idx)))
        train_indices.extend(idx[:n_train])
        holdout_indices.extend(idx[n_train:])

    train_df = df.loc[train_indices].sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)

    holdout_df = df.loc[holdout_indices].sample(
        frac=1.0, random_state=seed + 1
    ).reset_index(drop=True)

    return train_df, holdout_df


def build_network():
    """
    Create a 2-input -> 4-hidden-neuron -> 1-output classifier.

    hidden_layer_sizes=(4,)
        One hidden layer containing exactly 4 neurons.

    activation="logistic"
        scikit-learn's name for sigmoid activation in the hidden layer.

    solver="sgd"
        Use stochastic/mini-batch gradient descent so the training philosophy
        stays close to the hand-built NumPy example.

    batch_size=16
        Same mini-batch size as before.

    learning_rate_init=0.05
        Same learning rate as before.

    momentum=0.0
        Keep the update rule conceptually close to plain gradient descent.

    alpha=0.0
        Turn off L2 regularization to match the hand-built model more closely.
    """
    return MLPClassifier(
        hidden_layer_sizes=(4,),
        activation="logistic",
        solver="sgd",
        batch_size=BATCH_SIZE,
        learning_rate="constant",
        learning_rate_init=LEARNING_RATE,
        momentum=0.0,
        nesterovs_momentum=False,
        alpha=0.0,
        shuffle=True,
        random_state=RANDOM_SEED,
        max_iter=1,
        tol=0.0,
    )


def train_and_record_history(model, X_train, y_train, X_hold, y_hold):
    """
    Train one epoch at a time with partial_fit() so we can reproduce the
    epoch-vs-loss figure for both training and hold-out data.
    """
    history = {
        "epoch": [],
        "train_loss": [],
        "holdout_loss": [],
    }

    classes = np.array([0, 1])

    for epoch in range(1, MAX_EPOCHS + 1):

        if epoch == 1:
            model.partial_fit(X_train, y_train, classes=classes)
        else:
            model.partial_fit(X_train, y_train)

        p_train = model.predict_proba(X_train)
        p_hold = model.predict_proba(X_hold)

        train_bce = log_loss(y_train, p_train, labels=[0, 1])
        hold_bce = log_loss(y_hold, p_hold, labels=[0, 1])

        history["epoch"].append(epoch)
        history["train_loss"].append(train_bce)
        history["holdout_loss"].append(hold_bce)

        if epoch == 1 or epoch % 100 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"Training BCE = {train_bce:.5f} | "
                f"Hold-out BCE = {hold_bce:.5f}"
            )

    return pd.DataFrame(history)


def print_confusion_matrix_and_metrics(y_true, y_pred):
    """
    Print the confusion matrix and four common classification metrics:
    accuracy, precision, recall, and F1 score.

    We also print specificity because it is often useful in engineering work.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp)

    print("\nCONFUSION MATRIX -- HOLD-OUT SET")
    print("--------------------------------")
    print("                 Predicted 0   Predicted 1")
    print(f"Actual 0          {tn:10d}   {fp:10d}")
    print(f"Actual 1          {fn:10d}   {tp:10d}")

    print("\nFOUR KEY CLASSIFICATION METRICS")
    print("--------------------------------")
    print(f"Accuracy : {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall   : {recall:.6f}")
    print(f"F1 score : {f1:.6f}")

    print("\nAdditional engineering metric:")
    print(f"Specificity: {specificity:.6f}")


def plot_training_convergence(history):
    """Same epoch-vs-BCE figure as the from-scratch implementation."""
    plt.figure(figsize=(10, 6))
    plt.plot(history["epoch"], history["train_loss"], label="Training BCE")
    plt.plot(history["epoch"], history["holdout_loss"], label="Hold-out BCE")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.title("Scikit-learn Neural Network Training Convergence")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_decision_surface(train_df, holdout_df, scaler, model):
    """
    Plot P(class=1 | x1, x2) over the same feature-space window as the
    NumPy implementation.
    """
    x1_min, x1_max = PLOT_X1_LIMITS
    x2_min, x2_max = PLOT_X2_LIMITS

    xx, yy = np.meshgrid(
        np.linspace(x1_min, x1_max, 250),
        np.linspace(x2_min, x2_max, 250)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid)

    p_class1 = model.predict_proba(grid_scaled)[:, 1]
    P = p_class1.reshape(xx.shape)

    X_train = train_df[["x1", "x2"]].to_numpy()
    y_train = train_df["class"].to_numpy()

    X_hold = holdout_df[["x1", "x2"]].to_numpy()
    y_hold = holdout_df["class"].to_numpy()

    plt.figure(figsize=(11, 8))

    contour = plt.contourf(
        xx, yy, P,
        levels=np.linspace(0, 1, 11),
        alpha=0.35
    )
    plt.colorbar(contour, label="scikit-learn estimate P(class=1 | x)")

    # P=0.5 is the classification boundary.
    plt.contour(xx, yy, P, levels=[0.5], linewidths=2.5)

    plt.scatter(
        X_train[y_train == 0, 0],
        X_train[y_train == 0, 1],
        s=12,
        alpha=0.22,
        label="Training class 0"
    )

    plt.scatter(
        X_train[y_train == 1, 0],
        X_train[y_train == 1, 1],
        s=12,
        alpha=0.22,
        label="Training class 1"
    )

    plt.scatter(
        X_hold[y_hold == 0, 0],
        X_hold[y_hold == 0, 1],
        s=30,
        marker="o",
        facecolors="none",
        label="Hold-out class 0"
    )

    plt.scatter(
        X_hold[y_hold == 1, 0],
        X_hold[y_hold == 1, 1],
        s=30,
        marker="s",
        facecolors="none",
        label="Hold-out class 1"
    )

    plt.xlim(PLOT_X1_LIMITS)
    plt.ylim(PLOT_X2_LIMITS)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Scikit-learn 2-4-1 Neural Network Decision Surface")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():

    print("\n" + "=" * 78)
    print("CHAPTER 8: THE SAME 2 -> 4 -> 1 NETWORK USING SCIKIT-LEARN")
    print("=" * 78)

    # Generate the same data and same split as the from-scratch example.
    df = generate_dataset()
    train_df, holdout_df = stratified_train_holdout(df)

    X_train = train_df[["x1", "x2"]].to_numpy()
    y_train = train_df["class"].to_numpy()

    X_hold = holdout_df[["x1", "x2"]].to_numpy()
    y_hold = holdout_df["class"].to_numpy()

    print(f"\nTotal samples   : {len(df)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Hold-out samples: {len(holdout_df)}")
    print(f"Random seed     : {RANDOM_SEED}")

    # StandardScaler replaces the hand-written standardization function.
    # Fit ONLY on training data to avoid information leakage.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_hold_scaled = scaler.transform(X_hold)

    # This constructor replaces almost all of the network-definition boilerplate.
    model = build_network()

    print("\nArchitecture:")
    print("  Input features       : 2")
    print("  Hidden layers        : 1")
    print("  Hidden neurons       : 4")
    print("  Hidden activation    : sigmoid/logistic")
    print("  Output               : binary class probability")
    print("  Mini-batch size      :", BATCH_SIZE)
    print("  Learning rate        :", LEARNING_RATE)

    history = train_and_record_history(
        model,
        X_train_scaled,
        y_train,
        X_hold_scaled,
        y_hold
    )

    y_hold_pred = model.predict(X_hold_scaled)

    p_train = model.predict_proba(X_train_scaled)
    p_hold = model.predict_proba(X_hold_scaled)

    print("\nFinal losses:")
    print(f"Training BCE = {log_loss(y_train, p_train, labels=[0, 1]):.6f}")
    print(f"Hold-out BCE = {log_loss(y_hold, p_hold, labels=[0, 1]):.6f}")

    print_confusion_matrix_and_metrics(y_hold, y_hold_pred)

    # scikit-learn still lets us inspect the learned weights.
    print("\nLEARNED WEIGHT ARRAY SHAPES")
    print("---------------------------")
    print("Input -> hidden weights :", model.coefs_[0].shape)
    print("Hidden -> output weights:", model.coefs_[1].shape)
    print("Hidden biases           :", model.intercepts_[0].shape)
    print("Output bias             :", model.intercepts_[1].shape)

    plot_training_convergence(history)
    plot_decision_surface(train_df, holdout_df, scaler, model)

    print("\n" + "=" * 78)
    print("THE TEACHING POINT")
    print("=" * 78)
    print("""
The from-scratch NumPy program explicitly implemented:

    weight initialization
    forward propagation
    sigmoid activation
    binary cross-entropy
    backpropagation
    gradient calculation
    mini-batch updates
    parameter updates
    prediction

In scikit-learn, almost all of that machinery is contained inside:

    MLPClassifier(...)
    model.partial_fit(...)
    model.predict(...)
    model.predict_proba(...)

The mathematics has not disappeared.
The package is simply performing it for us.
""")


if __name__ == "__main__":
    main()
