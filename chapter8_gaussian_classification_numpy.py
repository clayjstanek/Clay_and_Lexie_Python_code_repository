"""
Chapter 8 worked example:
2D Gaussian classification solved two ways:

Part I  - Linear algebra / Fisher Linear Discriminant (LDA)
Part II - A hand-built 2 -> 4 -> 1 sigmoid neural network

Only numpy, scipy, pandas, and matplotlib are used.

Important teaching note
-----------------------
The two Gaussian classes in this example have DIFFERENT covariance matrices.
Therefore, the exact Bayes-optimal decision boundary is quadratic, not a line.
Because this chapter is introducing hyperplanes, Part I deliberately asks for
the best LINEAR separator using Fisher's linear discriminant.

Also, because the two Gaussian distributions overlap, neither classification
error nor cross-entropy loss should be expected to converge to exactly zero.
There is irreducible class overlap in the data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

RANDOM_SEED = 42
N_PER_CLASS = 1000
TRAIN_FRACTION = 0.80
BATCH_SIZE = 16
LEARNING_RATE = 0.05
MAX_EPOCHS = 3000
PATIENCE = 150
MIN_DELTA = 1e-6

CSV_FILENAME = "chapter8_2d_gaussian_classification.csv"


# =========================================================================
# PART 0 -- Generate and save the data
# =========================================================================

def generate_dataset(seed=RANDOM_SEED):
    """
    Generate two 2D Gaussian classes.

    Class 0:
        mean = [5, 4]
        covariance = [[2, 0],
                      [0, 2]]

    Class 1:
        mean = [-2, 1]
        covariance = [[2, 1],
                      [1, 2]]

    Returns
    -------
    df : pandas DataFrame
        Columns: x1, x2, class
    """
    rng = np.random.default_rng(seed)

    mu0 = np.array([5.0, 4.0])
    cov0 = np.array([[2.0, 4.0],
                     [4.0, 2.0]])

    mu1 = np.array([-2.0, 1.0])
    cov1 = np.array([[3.0, 5.0],
                     [5.0, 6.0]])

    X0 = rng.multivariate_normal(mu0, cov0, size=N_PER_CLASS)
    X1 = rng.multivariate_normal(mu1, cov1, size=N_PER_CLASS)

    df0 = pd.DataFrame(X0, columns=["x1", "x2"])
    df0["class"] = 0

    df1 = pd.DataFrame(X1, columns=["x1", "x2"])
    df1["class"] = 1

    df = pd.concat([df0, df1], ignore_index=True)

    # Shuffle rows so class order is not encoded in the file.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def stratified_train_holdout(df, train_fraction=TRAIN_FRACTION, seed=RANDOM_SEED):
    """
    Make an 80/20 stratified train/hold-out split using pandas/numpy only.
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

    train_df = df.loc[train_indices].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    holdout_df = df.loc[holdout_indices].sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)

    return train_df, holdout_df


# =========================================================================
# PART I -- LINEAR CLASSIFICATION FROM LINEAR ALGEBRA
# =========================================================================

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def mahalanobis_distance_between_means(mu0, mu1, pooled_cov):
    """
    A covariance-aware distance between class means.

    Euclidean distance ignores the orientation and spread of each class.
    Mahalanobis distance accounts for covariance and is usually much more
    meaningful for classification.
    """
    delta = mu1 - mu0
    return np.sqrt(delta.T @ np.linalg.inv(pooled_cov) @ delta)


def fit_fisher_linear_discriminant(X, y):
    """
    Derive a linear separating hyperplane using Fisher's Linear Discriminant.

    Fisher's criterion chooses w to maximize

                  (w^T (mu1 - mu0))^2
        J(w) = ---------------------------
                  w^T S_W w

    where S_W is the within-class scatter matrix.

    The maximizing direction is

        w proportional to inv(S_W) @ (mu1 - mu0)

    For equal class priors, a convenient separating threshold is the midpoint
    between the projected class means.  We write the boundary as

        w^T x + b = 0.

    NOTE:
    The classes have unequal covariances, so the true Bayes-optimal boundary
    would be quadratic (QDA).  This is the optimal Fisher LINEAR direction.
    """
    X0 = X[y == 0]
    X1 = X[y == 1]

    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)

    # Sample covariance matrices
    S0 = np.cov(X0, rowvar=False, ddof=1)
    S1 = np.cov(X1, rowvar=False, ddof=1)

    # Within-class scatter.  The scaling does not affect the direction of w.
    SW = (len(X0) - 1) * S0 + (len(X1) - 1) * S1

    # Solve S_W w = (mu1 - mu0) rather than explicitly forming inverse.
    w = np.linalg.solve(SW, mu1 - mu0)

    # Midpoint of projected means for equal priors.
    midpoint = 0.5 * (mu0 + mu1)
    b = -w @ midpoint

    # Make orientation consistent: class 1 should have positive score.
    if (w @ mu1 + b) < (w @ mu0 + b):
        w = -w
        b = -b

    pooled_cov = ((len(X0) - 1) * S0 + (len(X1) - 1) * S1) / (len(X0) + len(X1) - 2)

    return {
        "w": w,
        "b": b,
        "mu0": mu0,
        "mu1": mu1,
        "cov0": S0,
        "cov1": S1,
        "pooled_cov": pooled_cov
    }


def signed_distance_to_hyperplane(X, w, b):
    """
    Signed perpendicular distance from each point x to

        w^T x + b = 0

    Distance is

        d(x) = (w^T x + b) / ||w||_2.
    """
    return (X @ w + b) / np.linalg.norm(w)


def predict_hyperplane(X, w, b, threshold_distance=0.0):
    """
    Classify by signed perpendicular distance.

    class 1 if distance >= threshold_distance, otherwise class 0.
    """
    d = signed_distance_to_hyperplane(X, w, b)
    return (d >= threshold_distance).astype(int), d


def confusion_metrics(y_true, y_pred):
    """
    Return confusion matrix terms and common classification metrics.

    Here class 1 is treated as the "positive" class.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true)
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    fnr = fn / (fn + tp) if (fn + tp) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan

    return {
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "accuracy": accuracy,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "false_negative_rate": fnr,
        "false_positive_rate": fpr
    }


def tune_distance_threshold(y, distances):
    """
    Optional threshold tuning.

    The orientation of the hyperplane comes from Fisher LDA.  We then search
    possible offsets parallel to that plane.

    Teaching objective:
      - first maximize TRAINING accuracy,
      - among thresholds tied for maximum accuracy, choose the one with the
        LOWEST false-negative rate.

    This makes the user's "highest accuracy and false negatives" request precise:
    maximize accuracy while preferring fewer false negatives.

    The hold-out set is NOT used for threshold selection.
    """
    candidates = np.unique(distances)
    # Add values just beyond the extremes.
    eps = 1e-9
    candidates = np.r_[distances.min() - eps, candidates, distances.max() + eps]

    best = None
    for tau in candidates:
        pred = (distances >= tau).astype(int)
        m = confusion_metrics(y, pred)

        key = (m["accuracy"], -m["false_negative_rate"])
        if best is None or key > best["key"]:
            best = {"threshold": float(tau), "metrics": m, "key": key}

    return best["threshold"], best["metrics"]


def plot_linear_classifier(train_df, holdout_df, lda, tau):
    X_train = train_df[["x1", "x2"]].to_numpy()
    y_train = train_df["class"].to_numpy()

    X_hold = holdout_df[["x1", "x2"]].to_numpy()
    y_hold = holdout_df["class"].to_numpy()

    w = lda["w"]
    b = lda["b"]

    plt.figure(figsize=(11, 8))

    # Plot training samples lightly.
    plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
                s=12, alpha=0.28, label="Training class 0")
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
                s=12, alpha=0.28, label="Training class 1")

    # Hold-out samples with stronger markers.
    plt.scatter(X_hold[y_hold == 0, 0], X_hold[y_hold == 0, 1],
                s=28, marker="o", facecolors="none", label="Hold-out class 0")
    plt.scatter(X_hold[y_hold == 1, 0], X_hold[y_hold == 1, 1],
                s=28, marker="s", facecolors="none", label="Hold-out class 1")

    # Means as black circles.
    plt.scatter(*lda["mu0"], s=180, facecolors="none", edgecolors="black",
                linewidths=2.5, label="Training class means")
    plt.scatter(*lda["mu1"], s=180, facecolors="none", edgecolors="black",
                linewidths=2.5)

    # Our decision rule is distance >= tau.
    # Since distance=(w^T x+b)/||w||, the shifted boundary is:
    #     w^T x + b - tau*||w|| = 0
    b_eff = b - tau * np.linalg.norm(w)

    x_min = min(train_df.x1.min(), holdout_df.x1.min()) - 1
    x_max = max(train_df.x1.max(), holdout_df.x1.max()) + 1
    xx = np.linspace(x_min, x_max, 400)

    if abs(w[1]) > 1e-12:
        yy = -(w[0] * xx + b_eff) / w[1]
        plt.plot(xx, yy, "k-", linewidth=2.5, label="Fisher linear decision boundary")
    else:
        x_boundary = -b_eff / w[0]
        plt.axvline(x_boundary, color="black", linewidth=2.5,
                    label="Fisher linear decision boundary")

    # Demonstrate perpendicular distance for one hold-out point near the plane.
    hold_dist = signed_distance_to_hyperplane(X_hold, w, b) - tau
    idx = np.argmin(np.abs(hold_dist))
    p = X_hold[idx]

    # Orthogonal projection of point p onto shifted plane.
    # boundary: w^T x + b_eff = 0
    signed_raw = (w @ p + b_eff) / (w @ w)
    p_proj = p - signed_raw * w

    plt.plot([p[0], p_proj[0]], [p[1], p_proj[1]], "k--", linewidth=1.8)
    plt.scatter([p[0]], [p[1]], s=110, marker="*", edgecolors="black",
                label="Example classified point")

    d_perp = abs((w @ p + b_eff) / np.linalg.norm(w))
    mid = 0.5 * (p + p_proj)
    plt.text(mid[0], mid[1], f"  perpendicular distance = {d_perp:.3f}",
             fontsize=10)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("2D Gaussian Classification with a Fisher Linear Hyperplane")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


# =========================================================================
# PART II -- A HAND-BUILT 2 -> 4 -> 1 SIGMOID NEURAL NETWORK
# =========================================================================

def sigmoid(z):
    # scipy.special.expit is a numerically stable sigmoid
    return expit(z)


def binary_cross_entropy(y, p):
    """
    Mean binary cross-entropy.
    """
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    y = y.reshape(-1, 1)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def standardize_from_training(X_train, X_other):
    """
    Standardize using TRAINING statistics only.
    This prevents hold-out information leakage.
    """
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0, ddof=0)
    sigma[sigma == 0] = 1.0

    return (X_train - mu) / sigma, (X_other - mu) / sigma, mu, sigma


def initialize_network(seed=RANDOM_SEED):
    """
    2 input features -> 4 sigmoid hidden neurons -> 1 sigmoid output neuron.

    W1 shape: (2, 4)
    b1 shape: (1, 4)
    W2 shape: (4, 1)
    b2 shape: (1, 1)
    """
    rng = np.random.default_rng(seed)

    # Xavier-style small initialization is helpful for sigmoid networks.
    W1 = rng.normal(0.0, np.sqrt(1.0 / 2.0), size=(2, 4))
    b1 = np.zeros((1, 4))

    W2 = rng.normal(0.0, np.sqrt(1.0 / 4.0), size=(4, 1))
    b2 = np.zeros((1, 1))

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def forward_propagation(X, params):
    """
    Forward pass:

        Z1 = X W1 + b1
        A1 = sigmoid(Z1)
        Z2 = A1 W2 + b2
        A2 = sigmoid(Z2)

    A2 is interpreted as P(class=1 | x).
    """
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1)

    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)

    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


def backward_propagation(y, params, cache):
    """
    Backpropagation for sigmoid hidden units and sigmoid+BCE output.

    The especially convenient output derivative is

        dZ2 = A2 - y

    because the sigmoid derivative and BCE derivative simplify together.

    Shapes:
        X   : (m, 2)
        A1  : (m, 4)
        A2  : (m, 1)
        dW2 : (4, 1)
        db2 : (1, 1)
        dW1 : (2, 4)
        db1 : (1, 4)
    """
    X = cache["X"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    W2 = params["W2"]

    m = X.shape[0]
    y = y.reshape(-1, 1)

    # Output layer:
    dZ2 = A2 - y
    dW2 = (A1.T @ dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # Hidden layer:
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * A1 * (1.0 - A1)
    dW1 = (X.T @ dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def update_parameters(params, grads, learning_rate=LEARNING_RATE):
    params["W1"] -= learning_rate * grads["dW1"]
    params["b1"] -= learning_rate * grads["db1"]
    params["W2"] -= learning_rate * grads["dW2"]
    params["b2"] -= learning_rate * grads["db2"]


def neural_network_predict(X, params):
    p, _ = forward_propagation(X, params)
    pred = (p[:, 0] >= 0.5).astype(int)
    return pred, p[:, 0]


def train_network(X_train, y_train, X_hold, y_hold,
                  batch_size=BATCH_SIZE,
                  learning_rate=LEARNING_RATE,
                  max_epochs=MAX_EPOCHS,
                  patience=PATIENCE,
                  min_delta=MIN_DELTA,
                  seed=RANDOM_SEED):
    """
    Mini-batch gradient descent with batch size 16.

    Early stopping watches HOLD-OUT loss here only for a pedagogical display.
    In a production ML workflow, one would normally use:
        training / validation / final test
    rather than tuning against the final hold-out set.

    To preserve the requested 80/20 setup, this demonstration reports the
    hold-out trajectory but restores the best parameter state observed.
    """
    rng = np.random.default_rng(seed)
    params = initialize_network(seed)

    history = {
        "epoch": [],
        "train_loss": [],
        "holdout_loss": [],
        "train_accuracy": [],
        "holdout_accuracy": []
    }

    best_holdout_loss = np.inf
    best_params = None
    epochs_without_improvement = 0

    n = len(X_train)

    for epoch in range(1, max_epochs + 1):
        indices = rng.permutation(n)

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            Xb = X_train[batch_idx]
            yb = y_train[batch_idx]

            _, cache = forward_propagation(Xb, params)
            grads = backward_propagation(yb, params, cache)
            update_parameters(params, grads, learning_rate)

        # Record full-dataset metrics once per epoch.
        p_train, _ = forward_propagation(X_train, params)
        p_hold, _ = forward_propagation(X_hold, params)

        train_loss = binary_cross_entropy(y_train, p_train)
        hold_loss = binary_cross_entropy(y_hold, p_hold)

        train_pred = (p_train[:, 0] >= 0.5).astype(int)
        hold_pred = (p_hold[:, 0] >= 0.5).astype(int)

        train_acc = np.mean(train_pred == y_train)
        hold_acc = np.mean(hold_pred == y_hold)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["holdout_loss"].append(hold_loss)
        history["train_accuracy"].append(train_acc)
        history["holdout_accuracy"].append(hold_acc)

        if hold_loss < best_holdout_loss - min_delta:
            best_holdout_loss = hold_loss
            best_params = {k: v.copy() for k, v in params.items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 100 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"train BCE={train_loss:.5f}, holdout BCE={hold_loss:.5f} | "
                f"train acc={train_acc:.4f}, holdout acc={hold_acc:.4f}"
            )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_params is not None:
        params = best_params

    return params, pd.DataFrame(history)


def plot_nn_training(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history["epoch"], history["train_loss"], label="Training BCE")
    plt.plot(history["epoch"], history["holdout_loss"], label="Hold-out BCE")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.title("Neural Network Training Convergence")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_nn_decision_surface(train_df, holdout_df, params, feature_mean, feature_std):
    """
    Show NN P(class=1 | x) over the original x1/x2 coordinate system.
    """
    all_df = pd.concat([train_df, holdout_df], ignore_index=True)

    x1_min, x1_max = all_df["x1"].min() - 1, all_df["x1"].max() + 1
    x2_min, x2_max = all_df["x2"].min() - 1, all_df["x2"].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x1_min, x1_max, 250),
        np.linspace(x2_min, x2_max, 250)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_std = (grid - feature_mean) / feature_std

    _, p = neural_network_predict(grid_std, params)
    P = p.reshape(xx.shape)

    X_hold = holdout_df[["x1", "x2"]].to_numpy()
    y_hold = holdout_df["class"].to_numpy()

    plt.figure(figsize=(11, 8))
    contour = plt.contourf(xx, yy, P, levels=np.linspace(0, 1, 11), alpha=0.35)
    plt.colorbar(contour, label="NN estimate P(class=1 | x)")

    # 0.5 probability decision contour
    plt.contour(xx, yy, P, levels=[0.5], linewidths=2.5)

    plt.scatter(X_hold[y_hold == 0, 0], X_hold[y_hold == 0, 1],
                s=28, marker="o", facecolors="none", label="Hold-out class 0")
    plt.scatter(X_hold[y_hold == 1, 0], X_hold[y_hold == 1, 1],
                s=28, marker="s", facecolors="none", label="Hold-out class 1")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("2-4-1 Sigmoid Neural Network Decision Surface")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================================
# MAIN DEMONSTRATION
# =========================================================================

def main():
    print("\n" + "=" * 78)
    print("CHAPTER 8: 2D CLASSIFICATION FROM LINEAR ALGEBRA TO A NEURAL NETWORK")
    print("=" * 78)

    # ---------------------------------------------------------------------
    # Generate data and save CSV
    # ---------------------------------------------------------------------
    df = generate_dataset()
    df.to_csv(CSV_FILENAME, index=False)

    train_df, holdout_df = stratified_train_holdout(df)

    print(f"\nSaved generated data to: {CSV_FILENAME}")
    print(f"Total samples:   {len(df)}")
    print(f"Training samples:{len(train_df)}")
    print(f"Hold-out samples:{len(holdout_df)}")

    X_train = train_df[["x1", "x2"]].to_numpy()
    y_train = train_df["class"].to_numpy()

    X_hold = holdout_df[["x1", "x2"]].to_numpy()
    y_hold = holdout_df["class"].to_numpy()

    # ---------------------------------------------------------------------
    # PART I: Fisher linear classifier
    # ---------------------------------------------------------------------
    lda = fit_fisher_linear_discriminant(X_train, y_train)

    d_euclid = euclidean_distance(lda["mu0"], lda["mu1"])
    d_mahal = mahalanobis_distance_between_means(
        lda["mu0"], lda["mu1"], lda["pooled_cov"]
    )

    print("\n" + "-" * 78)
    print("PART I -- LINEAR CLASSIFIER FROM LINEAR ALGEBRA")
    print("-" * 78)

    print("\nEstimated training means:")
    print("mu0 =", lda["mu0"])
    print("mu1 =", lda["mu1"])

    print("\nEstimated training covariance matrices:")
    print("Sigma0 =\n", lda["cov0"])
    print("Sigma1 =\n", lda["cov1"])

    print(f"\nEuclidean distance between class means = {d_euclid:.6f}")
    print(f"Mahalanobis distance between class means = {d_mahal:.6f}")

    print("\nImportant:")
    print("Euclidean distance between means is descriptive, but it does NOT by itself")
    print("determine the best classifier because it ignores covariance.")
    print("Mahalanobis distance is a more useful separation measure because it")
    print("accounts for within-class spread and correlation.")

    w = lda["w"]
    b = lda["b"]

    print("\nFisher linear discriminant:")
    print("w =", w)
    print("b =", b)
    print("\nCanonical hyperplane equation:")
    print(f"    ({w[0]:.8f}) x1 + ({w[1]:.8f}) x2 + ({b:.8f}) = 0")

    # Tune a parallel threshold using training data only.
    train_dist = signed_distance_to_hyperplane(X_train, w, b)
    tau, train_metrics = tune_distance_threshold(y_train, train_dist)

    print(f"\nSelected signed-distance threshold tau = {tau:.8f}")
    print("This threshold maximizes training accuracy; ties prefer lower FNR.")

    # Effective hyperplane after threshold shift.
    b_eff = b - tau * np.linalg.norm(w)
    print("\nFinal tuned decision hyperplane:")
    print(f"    ({w[0]:.8f}) x1 + ({w[1]:.8f}) x2 + ({b_eff:.8f}) = 0")
    print("Classify as class 1 when the signed perpendicular distance is >= tau.")

    train_pred, _ = predict_hyperplane(X_train, w, b, tau)
    hold_pred, _ = predict_hyperplane(X_hold, w, b, tau)

    train_m = confusion_metrics(y_train, train_pred)
    hold_m = confusion_metrics(y_hold, hold_pred)

    print("\nTraining metrics:")
    for k, v in train_m.items():
        print(f"  {k:20s}: {v:.6f}" if isinstance(v, float) else f"  {k:20s}: {v}")

    print("\nHold-out metrics:")
    for k, v in hold_m.items():
        print(f"  {k:20s}: {v:.6f}" if isinstance(v, float) else f"  {k:20s}: {v}")

    plot_linear_classifier(train_df, holdout_df, lda, tau)

    # ---------------------------------------------------------------------
    # PART II: Neural network
    # ---------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("PART II -- HAND-BUILT 2 -> 4 -> 1 SIGMOID NEURAL NETWORK")
    print("-" * 78)

    X_train_std, X_hold_std, feature_mean, feature_std = standardize_from_training(
        X_train, X_hold
    )

    print("\nNetwork architecture:")
    print("  2 input features")
    print("  4 sigmoid hidden neurons")
    print("  1 sigmoid output neuron")
    print("  mini-batch size =", BATCH_SIZE)

    # Demonstrate one forward pass before training.
    initial_params = initialize_network()
    first_batch_X = X_train_std[:BATCH_SIZE]
    first_batch_y = y_train[:BATCH_SIZE]
    initial_output, initial_cache = forward_propagation(first_batch_X, initial_params)

    print("\nExample FIRST forward propagation before learning:")
    print("X batch shape  :", first_batch_X.shape)
    print("Z1 shape       :", initial_cache["Z1"].shape)
    print("A1 shape       :", initial_cache["A1"].shape)
    print("Z2 shape       :", initial_cache["Z2"].shape)
    print("Output shape   :", initial_output.shape)
    print("First 5 predicted probabilities:")
    print(initial_output[:5, 0])
    print("First 5 target labels:")
    print(first_batch_y[:5])

    # Show one backprop gradient calculation before full training.
    initial_grads = backward_propagation(first_batch_y, initial_params, initial_cache)
    print("\nGradient shapes from one backward pass:")
    for name, arr in initial_grads.items():
        print(f"  {name}: {arr.shape}")

    params, history = train_network(
        X_train_std, y_train,
        X_hold_std, y_hold
    )

    nn_train_pred, nn_train_prob = neural_network_predict(X_train_std, params)
    nn_hold_pred, nn_hold_prob = neural_network_predict(X_hold_std, params)

    nn_train_m = confusion_metrics(y_train, nn_train_pred)
    nn_hold_m = confusion_metrics(y_hold, nn_hold_pred)

    print("\nFinal neural-network training metrics:")
    for k, v in nn_train_m.items():
        print(f"  {k:20s}: {v:.6f}" if isinstance(v, float) else f"  {k:20s}: {v}")

    print("\nFinal neural-network hold-out metrics:")
    for k, v in nn_hold_m.items():
        print(f"  {k:20s}: {v:.6f}" if isinstance(v, float) else f"  {k:20s}: {v}")

    final_train_prob, _ = forward_propagation(X_train_std, params)
    final_hold_prob, _ = forward_propagation(X_hold_std, params)

    print(f"\nFinal training BCE = {binary_cross_entropy(y_train, final_train_prob):.6f}")
    print(f"Final hold-out BCE = {binary_cross_entropy(y_hold, final_hold_prob):.6f}")

    print("\nDo NOT expect the loss to reach exactly zero.")
    print("The two Gaussian class distributions overlap, so some observations are")
    print("intrinsically ambiguous.  A non-zero irreducible classification error is")
    print("therefore expected even for an excellent classifier.")

    print("\nLearned network parameters:")
    print("W1 =\n", params["W1"])
    print("b1 =\n", params["b1"])
    print("W2 =\n", params["W2"])
    print("b2 =\n", params["b2"])

    plot_nn_training(history)
    plot_nn_decision_surface(train_df, holdout_df, params, feature_mean, feature_std)

    print("\n" + "=" * 78)
    print("COMPARISON ON THE SAME HOLD-OUT SET")
    print("=" * 78)
    print(f"Fisher linear classifier accuracy : {hold_m['accuracy']:.4f}")
    print(f"Fisher false-negative rate        : {hold_m['false_negative_rate']:.4f}")
    print(f"Neural network accuracy           : {nn_hold_m['accuracy']:.4f}")
    print(f"Neural network false-negative rate: {nn_hold_m['false_negative_rate']:.4f}")

    print("\nThe linear classifier produces one hyperplane.")
    print("The 2-4-1 neural network can produce a nonlinear decision boundary.")
    print("Because the generating covariances are unequal, the exact Gaussian")
    print("Bayes boundary is quadratic, so nonlinear flexibility can be useful.")


if __name__ == "__main__":
    main()
