"""
Chapter 8 companion example:
Exact Gaussian Bayes / Quadratic Discriminant Analysis (QDA)

This script uses the same Gaussian classification problem as the earlier
linear-algebra / neural-network example, but now derives the exact Gaussian
Bayes classifier when the two classes have different covariance matrices.

Because Sigma_0 != Sigma_1, the log-posterior difference contains quadratic
terms in x1 and x2, so the Bayes decision boundary is curved.

Only numpy, pandas, and matplotlib are used.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RANDOM_SEED = 42
N_PER_CLASS = 1000
TRAIN_FRACTION = 0.80
CSV_FILENAME = "chapter8_2d_gaussian_classification_v2.csv"

# Teaching-friendly Gaussian parameters.
#
# Both covariance matrices are symmetric positive definite, so they are
# legitimate covariance matrices.  They are intentionally different so
# Gaussian Bayes/QDA has a genuinely quadratic decision boundary.
MU0 = np.array([3.0, 2.0])
COV0 = np.array([[3.0, 0.0],
                 [0.0, 1.0]])

MU1 = np.array([-2.0, 0.0])
COV1 = np.array([[1.5, 1.0],
                 [1.0, 3.0]])

# Use the same viewing window in the LDA, QDA, and neural-network figures.
# The wider range makes both branches of the quadratic QDA boundary easier
# to recognize while still keeping the data clouds clearly visible.
PLOT_X1_LIMITS = (-12.0, 12.0)
PLOT_X2_LIMITS = (-12.0, 12.0)




def validate_covariances():
    """Print eigenvalues to verify the covariance matrices are positive definite."""
    e0 = np.linalg.eigvalsh(COV0)
    e1 = np.linalg.eigvalsh(COV1)
    print("Covariance eigenvalues:")
    print("  COV0:", e0)
    print("  COV1:", e1)
    if np.any(e0 <= 0) or np.any(e1 <= 0):
        raise ValueError("A covariance matrix is not positive definite.")


# =========================================================================
# DATA
# =========================================================================

def generate_dataset(seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    mu0 = MU0.copy()
    cov0 = COV0.copy()

    mu1 = MU1.copy()
    cov1 = COV1.copy()

    X0 = rng.multivariate_normal(mu0, cov0, size=N_PER_CLASS)
    X1 = rng.multivariate_normal(mu1, cov1, size=N_PER_CLASS)

    df0 = pd.DataFrame(X0, columns=["x1", "x2"])
    df0["class"] = 0

    df1 = pd.DataFrame(X1, columns=["x1", "x2"])
    df1["class"] = 1

    df = pd.concat([df0, df1], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return df


def generate_and_save_dataset():
    """
    Generate the canonical Chapter 8 dataset every time.

    Because RANDOM_SEED and the Gaussian parameters are identical in the
    companion LDA/NN script, both scripts create the exact same 2,000 samples.
    This avoids accidentally loading a stale CSV generated with older parameters.
    """
    df = generate_dataset()
    df.to_csv(CSV_FILENAME, index=False)
    print(f"Generated and saved canonical dataset: {CSV_FILENAME}")
    return df


def stratified_train_holdout(df, train_fraction=TRAIN_FRACTION, seed=RANDOM_SEED):
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


# =========================================================================
# GAUSSIAN BAYES / QDA
# =========================================================================

def fit_qda(X, y):
    """
    Estimate class means, class covariance matrices, and class priors
    from the TRAINING set.

    For class k:

        x | C_k ~ N(mu_k, Sigma_k)

    and

        P(C_k) = pi_k.
    """

    model = {}

    classes = np.unique(y)

    for cls in classes:
        Xk = X[y == cls]

        model[int(cls)] = {
            "mu": Xk.mean(axis=0),
            "cov": np.cov(Xk, rowvar=False, ddof=1),
            "prior": len(Xk) / len(X)
        }

    return model


def gaussian_log_discriminant(X, mu, cov, prior):
    """
    Compute the Gaussian log discriminant:

        g_k(x)
          = -1/2 log|Sigma_k|
            -1/2 (x-mu_k)^T Sigma_k^{-1} (x-mu_k)
            + log(pi_k)

    The common term -d/2 log(2*pi) is omitted because it is identical
    for both classes and cancels when discriminants are compared.

    Larger g_k(x) means class k is more probable under the Gaussian model.
    """

    inv_cov = np.linalg.inv(cov)
    sign, logdet = np.linalg.slogdet(cov)

    if sign <= 0:
        raise ValueError("Covariance matrix must be positive definite.")

    delta = X - mu

    # Efficiently compute each Mahalanobis quadratic form:
    # (x-mu)^T inv(Sigma) (x-mu)
    quadratic = np.einsum("ni,ij,nj->n", delta, inv_cov, delta)

    return -0.5 * logdet - 0.5 * quadratic + np.log(prior)


def qda_scores(X, model):
    g0 = gaussian_log_discriminant(
        X,
        model[0]["mu"],
        model[0]["cov"],
        model[0]["prior"]
    )

    g1 = gaussian_log_discriminant(
        X,
        model[1]["mu"],
        model[1]["cov"],
        model[1]["prior"]
    )

    return g0, g1


def qda_predict(X, model):
    g0, g1 = qda_scores(X, model)
    return (g1 > g0).astype(int)


def qda_probability_class1(X, model):
    """
    Convert the two log discriminants into posterior class probabilities.

    P(C1 | x)
       = exp(g1) / [exp(g0) + exp(g1)]

    We compute it stably from the difference:

       P(C1 | x) = 1 / [1 + exp(g0-g1)].
    """
    g0, g1 = qda_scores(X, model)

    d = np.clip(g0 - g1, -700, 700)
    return 1.0 / (1.0 + np.exp(d))


# =========================================================================
# EXPLICIT QUADRATIC BOUNDARY FROM LINEAR ALGEBRA
# =========================================================================

def qda_boundary_coefficients(model):
    """
    Derive coefficients for the exact quadratic decision boundary

        g_1(x) - g_0(x) = 0

    written as

        x^T A x + q^T x + c = 0.

    Expanding the Gaussian discriminants gives

      A = 1/2 (Sigma_0^{-1} - Sigma_1^{-1})

      q = Sigma_1^{-1} mu_1 - Sigma_0^{-1} mu_0

      c = -1/2 mu_1^T Sigma_1^{-1} mu_1
          +1/2 mu_0^T Sigma_0^{-1} mu_0
          -1/2 log|Sigma_1|
          +1/2 log|Sigma_0|
          +log(pi_1/pi_0)

    In two dimensions this becomes

        a*x1^2 + b*x1*x2 + c2*x2^2
        + d*x1 + e*x2 + f = 0.
    """

    mu0 = model[0]["mu"]
    mu1 = model[1]["mu"]

    S0 = model[0]["cov"]
    S1 = model[1]["cov"]

    p0 = model[0]["prior"]
    p1 = model[1]["prior"]

    S0_inv = np.linalg.inv(S0)
    S1_inv = np.linalg.inv(S1)

    A = 0.5 * (S0_inv - S1_inv)
    q = S1_inv @ mu1 - S0_inv @ mu0

    _, logdet0 = np.linalg.slogdet(S0)
    _, logdet1 = np.linalg.slogdet(S1)

    c = (
        -0.5 * (mu1.T @ S1_inv @ mu1)
        +0.5 * (mu0.T @ S0_inv @ mu0)
        -0.5 * logdet1
        +0.5 * logdet0
        +np.log(p1 / p0)
    )

    # x^T A x in 2D:
    #
    # A00*x1^2 + 2*A01*x1*x2 + A11*x2^2
    coeffs = {
        "x1^2": A[0, 0],
        "x1*x2": 2.0 * A[0, 1],
        "x2^2": A[1, 1],
        "x1": q[0],
        "x2": q[1],
        "constant": c
    }

    return A, q, c, coeffs


# =========================================================================
# METRICS
# =========================================================================

def confusion_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true)
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    fnr = fn / (fn + tp) if (fn + tp) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan

    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "accuracy": accuracy,
        "recall": recall,
        "specificity": specificity,
        "false_negative_rate": fnr,
        "false_positive_rate": fpr
    }


# =========================================================================
# PLOTTING
# =========================================================================

def plot_qda(train_df, holdout_df, model):
    all_df = pd.concat([train_df, holdout_df], ignore_index=True)

    x1_min, x1_max = PLOT_X1_LIMITS
    x2_min, x2_max = PLOT_X2_LIMITS

    xx, yy = np.meshgrid(
        np.linspace(x1_min, x1_max, 350),
        np.linspace(x2_min, x2_max, 350)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    p1 = qda_probability_class1(grid, model).reshape(xx.shape)

    X_train = train_df[["x1", "x2"]].to_numpy()
    y_train = train_df["class"].to_numpy()

    X_hold = holdout_df[["x1", "x2"]].to_numpy()
    y_hold = holdout_df["class"].to_numpy()

    plt.figure(figsize=(11, 8))

    # Probability surface
    contour = plt.contourf(
        xx, yy, p1,
        levels=np.linspace(0, 1, 11),
        alpha=0.35
    )
    plt.colorbar(contour, label="Gaussian Bayes P(class=1 | x)")

    # Exact Bayes decision boundary P=0.5 <=> g1=g0
    plt.contour(
        xx, yy, p1,
        levels=[0.5],
        colors="black",
        linewidths=2.6
    )

    # Training samples
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

    # Hold-out samples
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

    # Estimated class means as black circles
    plt.scatter(
        *model[0]["mu"],
        s=180,
        facecolors="none",
        edgecolors="black",
        linewidths=2.5,
        label="Estimated class means"
    )

    plt.scatter(
        *model[1]["mu"],
        s=180,
        facecolors="none",
        edgecolors="black",
        linewidths=2.5
    )

    plt.xlim(PLOT_X1_LIMITS)
    plt.ylim(PLOT_X2_LIMITS)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Exact Gaussian Bayes / QDA Decision Surface")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_qda_boundary_only(train_df, holdout_df, model):
    """
    Cleaner figure emphasizing only the curved QDA boundary.
    """
    all_df = pd.concat([train_df, holdout_df], ignore_index=True)

    x1_min, x1_max = PLOT_X1_LIMITS
    x2_min, x2_max = PLOT_X2_LIMITS

    xx, yy = np.meshgrid(
        np.linspace(x1_min, x1_max, 400),
        np.linspace(x2_min, x2_max, 400)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    g0, g1 = qda_scores(grid, model)

    # Boundary occurs where g1-g0 = 0
    D = (g1 - g0).reshape(xx.shape)

    X_train = train_df[["x1", "x2"]].to_numpy()
    y_train = train_df["class"].to_numpy()
    X_hold = holdout_df[["x1", "x2"]].to_numpy()
    y_hold = holdout_df["class"].to_numpy()

    plt.figure(figsize=(11, 8))

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
        s=35,
        marker="o",
        label="Hold-out class 0"
    )

    plt.scatter(
        X_hold[y_hold == 1, 0],
        X_hold[y_hold == 1, 1],
        s=35,
        marker="s",
        label="Hold-out class 1"
    )

    plt.contour(
        xx, yy, D,
        levels=[0],
        colors="black",
        linewidths=3.0
    )

    plt.scatter(
        *model[0]["mu"],
        s=180,
        facecolors="none",
        edgecolors="black",
        linewidths=2.5,
        label="Estimated class means"
    )

    plt.scatter(
        *model[1]["mu"],
        s=180,
        facecolors="none",
        edgecolors="black",
        linewidths=2.5
    )

    plt.xlim(PLOT_X1_LIMITS)
    plt.ylim(PLOT_X2_LIMITS)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Curved QDA Decision Boundary from Unequal Covariances")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("=" * 78)
    print("GAUSSIAN BAYES / QUADRATIC DISCRIMINANT ANALYSIS")
    print("=" * 78)

    validate_covariances()

    df = generate_and_save_dataset()
    train_df, holdout_df = stratified_train_holdout(df)

    X_train = train_df[["x1", "x2"]].to_numpy()
    y_train = train_df["class"].to_numpy()

    X_hold = holdout_df[["x1", "x2"]].to_numpy()
    y_hold = holdout_df["class"].to_numpy()

    model = fit_qda(X_train, y_train)

    print("\nEstimated parameters from the TRAINING set:")

    for cls in [0, 1]:
        print(f"\nClass {cls}:")
        print("mean =")
        print(model[cls]["mu"])
        print("covariance =")
        print(model[cls]["cov"])
        print(f"prior = {model[cls]['prior']:.6f}")

    print("\n" + "-" * 78)
    print("WHY THE BOUNDARY IS QUADRATIC")
    print("-" * 78)

    print("""
For each class k, Gaussian Bayes uses

  g_k(x)
    = -1/2 log|Sigma_k|
      -1/2 (x-mu_k)^T Sigma_k^(-1) (x-mu_k)
      + log P(C_k).

The decision boundary is where

  g_1(x) = g_0(x).

If Sigma_0 = Sigma_1, the x^T Sigma^(-1) x quadratic terms cancel and
the boundary becomes a line (LDA).

Here Sigma_0 != Sigma_1, so those quadratic terms do NOT cancel.
Therefore the exact Gaussian Bayes boundary is a quadratic curve (QDA).
""")

    A, q, c, coeffs = qda_boundary_coefficients(model)

    print("Matrix form of boundary:")
    print("\n    x^T A x + q^T x + c = 0")

    print("\nA =")
    print(A)

    print("\nq =")
    print(q)

    print(f"\nc = {c:.10f}")

    print("\nExpanded 2D quadratic equation:")
    print(
        f"({coeffs['x1^2']:.10f}) x1^2 "
        f"+ ({coeffs['x1*x2']:.10f}) x1*x2 "
        f"+ ({coeffs['x2^2']:.10f}) x2^2 "
        f"+ ({coeffs['x1']:.10f}) x1 "
        f"+ ({coeffs['x2']:.10f}) x2 "
        f"+ ({coeffs['constant']:.10f}) = 0"
    )

    train_pred = qda_predict(X_train, model)
    hold_pred = qda_predict(X_hold, model)

    train_m = confusion_metrics(y_train, train_pred)
    hold_m = confusion_metrics(y_hold, hold_pred)

    print("\nTraining metrics:")
    for k, v in train_m.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.6f}")
        else:
            print(f"  {k:20s}: {v}")

    print("\nHold-out metrics:")
    for k, v in hold_m.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.6f}")
        else:
            print(f"  {k:20s}: {v}")

    print("\nInterpretation:")
    print("QDA knows nothing about a neural network.")
    print("It uses only the Gaussian class model, estimated means, covariances,")
    print("and priors.  Curvature appears analytically because the covariance")
    print("matrices differ.")

    plot_qda(train_df, holdout_df, model)
    plot_qda_boundary_only(train_df, holdout_df, model)


if __name__ == "__main__":
    main()
