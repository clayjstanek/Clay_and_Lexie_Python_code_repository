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
# PART 0 -- Generate and save the data
# =========================================================================

def generate_dataset(seed=RANDOM_SEED):
    """
    Generate two 2D Gaussian classes.

    Class 0:
        mean = [3, 2]
        covariance = [[3, 0],
                      [0, 1]]

    Class 1:
        mean = [-2, 0]
        covariance = [[1.5, 1],
                      [1, 3]]

    Returns
    -------
    df : pandas DataFrame
        Columns: x1, x2, class
    """
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

    x_min, x_max = PLOT_X1_LIMITS
    y_min, y_max = PLOT_X2_LIMITS
    xx = np.linspace(x_min, x_max, 500)

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

    plt.xlim(PLOT_X1_LIMITS)
    plt.ylim(PLOT_X2_LIMITS)
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
#
# This section deliberately implements the neural network "the long way."
#
# Nothing here is hidden inside TensorFlow, PyTorch, Keras, or scikit-learn.
# Every important mathematical operation is written explicitly so that we can
# see exactly how a neural network learns.
#
# Architecture:
#
#       x1 ----\
#               \
#                >---- [ h1 ] ----\
#               /                  \
#       x2 ----/                    \
#                                     \
#       x1 ----\                       \
#               \                       >---- [ output ] ---> P(class = 1 | x)
#                >---- [ h2 ] --------/
#               /
#       x2 ----/
#
#       ...and similarly for hidden neurons h3 and h4.
#
# More compactly:
#
#       2 input features  ->  4 hidden sigmoid neurons  ->  1 sigmoid output
#
# Each neuron performs the same two basic calculations:
#
#   1. weighted sum:
#
#          z = sum_i(w_i x_i) + b
#
#   2. nonlinear activation:
#
#          a = sigmoid(z)
#
# The hidden neurons create nonlinear intermediate features.  The output
# neuron combines those learned features and converts its final score into a
# number between 0 and 1.
#
# During training we repeat:
#
#   forward propagation
#          ↓
#   compute binary cross-entropy loss
#          ↓
#   backpropagate derivatives
#          ↓
#   compute gradients dL/dW and dL/db
#          ↓
#   update weights and biases with gradient descent
#
# The functions below correspond almost one-for-one with those steps.
# =========================================================================


def sigmoid(z):
    """
    Logistic sigmoid activation function.

    Mathematical definition:

        sigma(z) = 1 / (1 + exp(-z))

    Why use it here?
    ----------------
    The weighted sum z can be any real number.  The sigmoid maps that number
    smoothly into the interval (0, 1).  This makes it convenient for binary
    classification.

    In the hidden layer:
        sigmoid introduces NONLINEARITY.

    In the output layer:
        sigmoid allows us to interpret the output as an estimate of

            P(class = 1 | x)

    provided the model is trained with an appropriate probabilistic loss and
    is reasonably calibrated.

    Why scipy.special.expit instead of coding the formula directly?
    ---------------------------------------------------------------
    The naive formula

        1 / (1 + np.exp(-z))

    can overflow numerically for very large positive or negative values of z.
    scipy.special.expit computes the same mathematical function in a more
    numerically stable way.
    """
    return expit(z)


def binary_cross_entropy(y, p):
    """
    Compute the mean binary cross-entropy (BCE) loss.

    For one training example with true class y in {0,1} and predicted
    probability p = P(class=1 | x),

        L = -[ y log(p) + (1-y) log(1-p) ].

    For a batch of m examples we average the individual losses.

    Interpretation:
    ---------------
    If y = 1:
        L = -log(p)

        The model is rewarded when p is near 1 and strongly penalized when
        it assigns a small probability to the true positive class.

    If y = 0:
        L = -log(1-p)

        The model is rewarded when p is near 0 and strongly penalized when
        it incorrectly assigns high probability to class 1.

    This is not an arbitrary loss function.  BCE is the negative
    log-likelihood that follows naturally from a Bernoulli observation model.

    Why clip p?
    -----------
    log(0) is undefined.  A floating-point network can occasionally produce
    a probability extremely close to exactly 0 or 1, so we clip the values
    very slightly away from those endpoints before taking logarithms.
    """
    eps = 1e-12

    # Keep probabilities in a numerically safe interval.
    p = np.clip(p, eps, 1.0 - eps)

    # Convert y from shape (m,) to shape (m,1) so it matches p.
    y = y.reshape(-1, 1)

    # Mean BCE across all examples in the batch.
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def standardize_from_training(X_train, X_other):
    """
    Standardize features using statistics computed ONLY from the training set.

    For each feature:

        x_standardized = (x - training_mean) / training_std

    Why standardize?
    ----------------
    Neural-network optimization usually behaves better when input features
    live on comparable numerical scales.  If one feature is around 0.001 and
    another is around 100000, the gradient geometry can become unnecessarily
    difficult.

    Why use training statistics only?
    ---------------------------------
    The hold-out data are supposed to represent information the model has not
    seen during training.  If we used hold-out means or standard deviations to
    preprocess the training set, we would leak information from the hold-out
    set into the training process.

    Returns
    -------
    standardized training data
    standardized second dataset
    training mean
    training standard deviation
    """
    # Mean of each input feature across training samples.
    mu = X_train.mean(axis=0)

    # Standard deviation of each feature across training samples.
    sigma = X_train.std(axis=0, ddof=0)

    # Defensive protection against a constant-valued feature.
    sigma[sigma == 0] = 1.0

    X_train_std = (X_train - mu) / sigma
    X_other_std = (X_other - mu) / sigma

    return X_train_std, X_other_std, mu, sigma


def initialize_network(seed=RANDOM_SEED):
    """
    Initialize the trainable parameters of the 2 -> 4 -> 1 network.

    Network dimensions
    ------------------
    Input layer:
        2 values: x1, x2

    Hidden layer:
        4 neurons

    Output layer:
        1 neuron

    Weight matrix W1:
        shape (2, 4)

        Rows correspond to the 2 input features.
        Columns correspond to the 4 hidden neurons.

        Element W1[i,j] is the weight connecting input feature i
        to hidden neuron j.

    Bias vector b1:
        shape (1, 4)

        One independent bias for each hidden neuron.

    Weight matrix W2:
        shape (4, 1)

        Four hidden activations feed the one output neuron.

    Bias b2:
        shape (1, 1)

        One bias for the output neuron.

    Why not initialize every weight to zero?
    ----------------------------------------
    If all hidden neurons started with exactly the same weights, they would
    receive identical gradients and remain identical forever.  Random
    initialization breaks that symmetry so different hidden neurons can learn
    different useful nonlinear features.

    Why keep the random values relatively small?
    ---------------------------------------------
    Sigmoid functions become nearly flat when |z| is very large.  In those
    saturated regions the derivative approaches zero, which can make learning
    extremely slow.  Small initial weights help keep early activations in a
    useful region.

    The scaling below is Xavier-like:

        std(W1) ~ sqrt(1 / number_of_inputs)
        std(W2) ~ sqrt(1 / number_of_hidden_units)

    This is a simple way to keep the variance of signals from exploding or
    collapsing as they pass between layers.
    """
    rng = np.random.default_rng(seed)

    # Connections from 2 inputs to 4 hidden neurons.
    W1 = rng.normal(
        loc=0.0,
        scale=np.sqrt(1.0 / 2.0),
        size=(2, 4)
    )

    # Start hidden biases at zero.
    b1 = np.zeros((1, 4))

    # Connections from 4 hidden neurons to 1 output neuron.
    W2 = rng.normal(
        loc=0.0,
        scale=np.sqrt(1.0 / 4.0),
        size=(4, 1)
    )

    # Start output bias at zero.
    b2 = np.zeros((1, 1))

    return {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }


def forward_propagation(X, params):
    """
    Perform the complete FORWARD PASS through the network.

    The forward pass answers:

        "Given the current weights and biases, what does the network predict?"

    For a batch containing m samples:

        X shape  = (m, 2)

    -----------------------------------------------------------------------
    HIDDEN LAYER
    -----------------------------------------------------------------------

    First compute the linear pre-activation:

        Z1 = X W1 + b1

    Shapes:

        (m,2) @ (2,4) -> (m,4)

    so Z1 contains one pre-activation value for every sample and every hidden
    neuron.

    Then apply the sigmoid element-by-element:

        A1 = sigmoid(Z1)

    A1 therefore has shape (m,4).

    -----------------------------------------------------------------------
    OUTPUT LAYER
    -----------------------------------------------------------------------

    The four hidden activations become inputs to the output neuron:

        Z2 = A1 W2 + b2

    Shapes:

        (m,4) @ (4,1) -> (m,1)

    Then apply sigmoid again:

        A2 = sigmoid(Z2)

    A2 contains one probability-like output for every input sample.

    We interpret:

        A2[n,0] ~= P(class = 1 | x_n)

    -----------------------------------------------------------------------
    WHY CACHE INTERMEDIATE VALUES?
    -----------------------------------------------------------------------

    Backpropagation needs the intermediate quantities generated during the
    forward pass.  In particular, derivatives through the hidden sigmoid
    require A1, and gradients for W1 require X.

    Instead of recomputing them, we store them in "cache" and reuse them
    during the backward pass.

    This is the same basic idea used by modern automatic-differentiation
    systems: values generated during forward computation are retained because
    they will be needed when derivatives are propagated backward.
    """
    # Unpack parameters for readability.
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    # ---------------------------------------------------------------------
    # Hidden layer forward pass
    # ---------------------------------------------------------------------

    # Each row of X is one sample [x1, x2].
    # Matrix multiplication simultaneously computes all 4 weighted sums.
    Z1 = X @ W1 + b1

    # Apply sigmoid separately to every element of Z1.
    A1 = sigmoid(Z1)

    # ---------------------------------------------------------------------
    # Output layer forward pass
    # ---------------------------------------------------------------------

    # Combine the four hidden activations into one output score per sample.
    Z2 = A1 @ W2 + b2

    # Convert the score into a value in (0,1).
    A2 = sigmoid(Z2)

    # Save values needed later by backpropagation.
    cache = {
        "X": X,
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    return A2, cache


def backward_propagation(y, params, cache):
    """
    Perform BACKPROPAGATION through the network.

    The backward pass answers:

        "How should every weight and bias change in order to reduce the loss?"

    We already know the final prediction A2 from the forward pass.
    Backpropagation now applies the chain rule from the output back toward
    the input.

    -----------------------------------------------------------------------
    STEP 1: OUTPUT LAYER ERROR
    -----------------------------------------------------------------------

    With a sigmoid output neuron and binary cross-entropy loss, a remarkable
    simplification occurs:

        dL/dZ2 = A2 - y

    This result already combines:
        derivative of BCE with respect to the output probability
    and
        derivative of sigmoid with respect to its input.

    It is one reason sigmoid + BCE is such a convenient pairing.

    -----------------------------------------------------------------------
    STEP 2: GRADIENTS FOR OUTPUT-LAYER PARAMETERS
    -----------------------------------------------------------------------

    Since:

        Z2 = A1 W2 + b2

    the weight gradient is:

        dW2 = A1^T dZ2 / m

    and the bias gradient is:

        db2 = sum(dZ2) / m

    -----------------------------------------------------------------------
    STEP 3: PROPAGATE ERROR BACK INTO HIDDEN LAYER
    -----------------------------------------------------------------------

    The hidden activations influenced the output through W2.

    Therefore:

        dA1 = dZ2 W2^T

    This tells us how much the loss changes with each hidden activation.

    -----------------------------------------------------------------------
    STEP 4: PASS THROUGH THE HIDDEN SIGMOIDS
    -----------------------------------------------------------------------

    For sigmoid:

        sigma'(z) = sigma(z)[1 - sigma(z)]

    Since A1 = sigma(Z1),

        dZ1 = dA1 * A1 * (1 - A1)

    The "*" here is elementwise multiplication.

    -----------------------------------------------------------------------
    STEP 5: GRADIENTS FOR INPUT-TO-HIDDEN PARAMETERS
    -----------------------------------------------------------------------

    Since:

        Z1 = X W1 + b1

    the gradients are:

        dW1 = X^T dZ1 / m

        db1 = sum(dZ1) / m

    -----------------------------------------------------------------------
    RESULT
    -----------------------------------------------------------------------

    After this function finishes, we have the gradient of the loss with
    respect to EVERY trainable parameter:

        dW1, db1, dW2, db2

    These gradients do not change the network by themselves.
    update_parameters() will use them to perform gradient descent.

    Shapes for batch size m:
        X    : (m, 2)
        A1   : (m, 4)
        A2   : (m, 1)
        dZ2  : (m, 1)
        dW2  : (4, 1)
        db2  : (1, 1)
        dA1  : (m, 4)
        dZ1  : (m, 4)
        dW1  : (2, 4)
        db1  : (1, 4)
    """
    # Retrieve values stored during forward propagation.
    X = cache["X"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    # We need W2 to propagate derivatives from the output back to hidden units.
    W2 = params["W2"]

    # Number of samples in this mini-batch.
    m = X.shape[0]

    # Make target shape compatible with the output matrix.
    y = y.reshape(-1, 1)

    # ---------------------------------------------------------------------
    # OUTPUT LAYER
    # ---------------------------------------------------------------------

    # Sigmoid + BCE simplification:
    # prediction error measured at the output pre-activation Z2.
    dZ2 = A2 - y

    # Each hidden activation A1 contributed to Z2 through W2.
    # A1.T accumulates those contributions across the batch.
    dW2 = (A1.T @ dZ2) / m

    # Bias contributes equally to every sample, so sum the output errors.
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # ---------------------------------------------------------------------
    # PROPAGATE ERROR INTO HIDDEN LAYER
    # ---------------------------------------------------------------------

    # Move the output error backward through W2.
    dA1 = dZ2 @ W2.T

    # Differentiate the hidden sigmoid activations.
    # Since A1 = sigmoid(Z1):
    #
    #     dA1/dZ1 = A1 * (1 - A1)
    #
    # Chain rule:
    #
    #     dL/dZ1 = dL/dA1 * dA1/dZ1
    dZ1 = dA1 * A1 * (1.0 - A1)

    # ---------------------------------------------------------------------
    # INPUT -> HIDDEN WEIGHT AND BIAS GRADIENTS
    # ---------------------------------------------------------------------

    # X.T connects each input feature with the error signal at each hidden unit.
    dW1 = (X.T @ dZ1) / m

    # Sum hidden pre-activation errors across examples for each hidden bias.
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }


def update_parameters(params, grads, learning_rate=LEARNING_RATE):
    """
    Apply one gradient-descent update to every trainable parameter.

    Generic gradient-descent rule:

        parameter_new
            =
        parameter_old
            -
        learning_rate * gradient

    The gradient points in the direction of greatest INCREASE of the loss.
    Therefore, subtracting the gradient moves the parameter in the direction
    of local DECREASE.

    The learning rate controls how far we move on each update.

    This function mutates the arrays stored in params in place.
    """

    # Input -> hidden weights
    params["W1"] -= learning_rate * grads["dW1"]

    # Hidden-layer biases
    params["b1"] -= learning_rate * grads["db1"]

    # Hidden -> output weights
    params["W2"] -= learning_rate * grads["dW2"]

    # Output bias
    params["b2"] -= learning_rate * grads["db2"]


def neural_network_predict(X, params):
    """
    Use the trained network for inference.

    Prediction requires only a FORWARD PASS.
    No loss is calculated.
    No gradients are computed.
    No parameters are changed.

    The network first produces probabilities p.

    We then impose the conventional threshold:

        p >= 0.5  -> class 1
        p <  0.5  -> class 0

    The threshold could be changed in an operational application if the
    relative cost of false positives and false negatives were different.
    """
    p, _ = forward_propagation(X, params)

    # p[:,0] converts shape (m,1) to shape (m,)
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
    Train the neural network with mini-batch gradient descent.

    This function is the full learning loop.

    Vocabulary
    ----------
    SAMPLE:
        One input example and its target.

    MINI-BATCH:
        A small group of samples used to compute one gradient update.

    EPOCH:
        One complete pass through the entire training dataset.

    With 1600 training samples and batch_size=16, each epoch performs about:

        1600 / 16 = 100 parameter updates.

    Therefore 500 epochs correspond to roughly 50,000 mini-batch gradient
    updates.

    -----------------------------------------------------------------------
    THE CORE TRAINING CYCLE
    -----------------------------------------------------------------------

    For every mini-batch:

        1. forward_propagation()
        2. backward_propagation()
        3. update_parameters()

    That three-line sequence is the essence of neural-network training.

    -----------------------------------------------------------------------
    WHY SHUFFLE EACH EPOCH?
    -----------------------------------------------------------------------

    Mini-batch gradients depend on which samples happen to appear together.
    Re-randomizing the sample order each epoch prevents the optimizer from
    repeatedly seeing the exact same batches in the exact same order.

    -----------------------------------------------------------------------
    EARLY STOPPING
    -----------------------------------------------------------------------

    We monitor hold-out loss here for pedagogical simplicity.

    If the hold-out BCE does not improve by at least min_delta for "patience"
    consecutive epochs, training stops.

    IMPORTANT:
    In a production ML workflow, the proper split would normally be:

        training set
        validation set
        final untouched test set

    The validation set would drive early stopping, while the final test set
    would remain completely untouched until the model is finalized.

    Here we intentionally retain the simpler 80/20 training/hold-out design
    because the purpose is to expose the mechanics of learning.
    """

    # Reproducible RNG used for shuffling training samples every epoch.
    rng = np.random.default_rng(seed)

    # Start with random weights and zero biases.
    params = initialize_network(seed)

    # Store metrics after every epoch so we can plot convergence later.
    history = {
        "epoch": [],
        "train_loss": [],
        "holdout_loss": [],
        "train_accuracy": [],
        "holdout_accuracy": []
    }

    # Early-stopping bookkeeping.
    best_holdout_loss = np.inf
    best_params = None
    epochs_without_improvement = 0

    # Number of training examples.
    n = len(X_train)

    # =====================================================================
    # OUTER LOOP: EPOCHS
    # =====================================================================
    for epoch in range(1, max_epochs + 1):

        # Randomly reorder sample indices at the beginning of every epoch.
        indices = rng.permutation(n)

        # =================================================================
        # INNER LOOP: MINI-BATCHES
        # =================================================================
        for start in range(0, n, batch_size):

            # Select the next group of sample indices.
            batch_idx = indices[start:start + batch_size]

            # Extract mini-batch inputs and labels.
            Xb = X_train[batch_idx]
            yb = y_train[batch_idx]

            # -------------------------------------------------------------
            # STEP 1 -- FORWARD PROPAGATION
            # -------------------------------------------------------------
            #
            # Compute current predictions for this mini-batch and retain the
            # intermediate values required by backpropagation.
            #
            # We do not need the returned probability array directly here,
            # hence the underscore.  A2 is already stored inside cache.
            _, cache = forward_propagation(Xb, params)

            # -------------------------------------------------------------
            # STEP 2 -- BACKPROPAGATION
            # -------------------------------------------------------------
            #
            # Use the known labels and cached forward-pass values to compute
            # dL/dW and dL/db for every layer.
            grads = backward_propagation(yb, params, cache)

            # -------------------------------------------------------------
            # STEP 3 -- GRADIENT-DESCENT PARAMETER UPDATE
            # -------------------------------------------------------------
            #
            # Move every weight and bias slightly in the direction that
            # reduces the mini-batch loss.
            update_parameters(params, grads, learning_rate)

        # =================================================================
        # END OF ONE EPOCH
        # =================================================================
        #
        # At this point every training sample has participated in one complete
        # pass through the network.
        #
        # Now perform full-dataset FORWARD PASSES to measure progress.
        # These calls are for evaluation only and do not change parameters.
        p_train, _ = forward_propagation(X_train, params)
        p_hold, _ = forward_propagation(X_hold, params)

        # Compute BCE across the entire training and hold-out sets.
        train_loss = binary_cross_entropy(y_train, p_train)
        hold_loss = binary_cross_entropy(y_hold, p_hold)

        # Convert probabilities to class labels using threshold 0.5.
        train_pred = (p_train[:, 0] >= 0.5).astype(int)
        hold_pred = (p_hold[:, 0] >= 0.5).astype(int)

        # Fraction correctly classified.
        train_acc = np.mean(train_pred == y_train)
        hold_acc = np.mean(hold_pred == y_hold)

        # Save metrics for plotting and inspection.
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["holdout_loss"].append(hold_loss)
        history["train_accuracy"].append(train_acc)
        history["holdout_accuracy"].append(hold_acc)

        # =================================================================
        # EARLY STOPPING LOGIC
        # =================================================================

        # Did the hold-out loss improve enough to count as meaningful progress?
        if hold_loss < best_holdout_loss - min_delta:

            # Remember the best loss seen so far.
            best_holdout_loss = hold_loss

            # IMPORTANT: use .copy().
            #
            # Without copying, best_params would point at the same mutable
            # arrays that continue changing during later epochs.
            best_params = {
                k: v.copy()
                for k, v in params.items()
            }

            # Improvement resets the patience counter.
            epochs_without_improvement = 0

        else:
            # No meaningful improvement this epoch.
            epochs_without_improvement += 1

        # Print occasional progress rather than flooding the console.
        if epoch == 1 or epoch % 100 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"train BCE={train_loss:.5f}, holdout BCE={hold_loss:.5f} | "
                f"train acc={train_acc:.4f}, holdout acc={hold_acc:.4f}"
            )

        # Stop if the hold-out objective has stalled for long enough.
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    # Restore the parameter values that achieved the best observed hold-out BCE.
    if best_params is not None:
        params = best_params

    # Convert the history dictionary to a DataFrame for convenient plotting.
    return params, pd.DataFrame(history)


def plot_nn_training(history):
    """
    Plot training and hold-out BCE versus epoch.

    Desired behavior:
    -----------------
    Both curves should fall substantially from their initial values.

    Training loss will usually be lower than hold-out loss because the model
    directly optimized the training examples.

    A large and growing gap between training and hold-out losses can indicate
    overfitting.

    A flat hold-out curve means additional epochs are no longer providing
    meaningful generalization improvement.
    """
    plt.figure(figsize=(10, 6))

    plt.plot(
        history["epoch"],
        history["train_loss"],
        label="Training BCE"
    )

    plt.plot(
        history["epoch"],
        history["holdout_loss"],
        label="Hold-out BCE"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.title("Neural Network Training Convergence")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_nn_decision_surface(train_df, holdout_df, params, feature_mean, feature_std):
    """
    Visualize what the trained network has learned across the 2D input plane.

    The network was trained on standardized inputs, but the plot is shown in
    the ORIGINAL x1/x2 coordinate system so the result remains easy to read.

    Procedure:
    ----------
    1. create a dense rectangular grid of hypothetical (x1,x2) locations
    2. standardize every grid point using TRAINING statistics
    3. pass every grid point through the trained neural network
    4. obtain P(class=1 | x) at every location
    5. display those probabilities as a filled contour map
    6. draw the P=0.5 contour as the classification boundary

    Why can this boundary be nonlinear?
    -----------------------------------
    Each hidden neuron computes a different sigmoid of a different linear
    combination of x1 and x2.  The output neuron then combines those nonlinear
    hidden activations.

    Therefore, unlike a single logistic-regression neuron, the 2-4-1 network
    can construct a curved decision surface.
    """

    # Use the same viewing limits as the LDA and QDA figures.
    x1_min, x1_max = PLOT_X1_LIMITS
    x2_min, x2_max = PLOT_X2_LIMITS

    # Create a dense grid covering the entire plotting region.
    xx, yy = np.meshgrid(
        np.linspace(x1_min, x1_max, 250),
        np.linspace(x2_min, x2_max, 250)
    )

    # Flatten the mesh into rows of [x1, x2].
    grid = np.c_[xx.ravel(), yy.ravel()]

    # The network was trained on standardized features, so apply exactly the
    # same transformation to the visualization grid.
    grid_std = (grid - feature_mean) / feature_std

    # Forward pass only: obtain one class-1 probability for every grid point.
    _, p = neural_network_predict(grid_std, params)

    # Reshape probabilities back to the same 2D shape as the plotting mesh.
    P = p.reshape(xx.shape)

    # Extract actual training and hold-out points for display.
    X_train = train_df[["x1", "x2"]].to_numpy()
    y_train = train_df["class"].to_numpy()

    X_hold = holdout_df[["x1", "x2"]].to_numpy()
    y_hold = holdout_df["class"].to_numpy()

    plt.figure(figsize=(11, 8))

    # Filled contours show how the predicted class-1 probability changes
    # throughout feature space.
    contour = plt.contourf(
        xx,
        yy,
        P,
        levels=np.linspace(0, 1, 11),
        alpha=0.35
    )

    plt.colorbar(
        contour,
        label="NN estimate P(class=1 | x)"
    )

    # The conventional classification boundary occurs where
    #
    #     P(class=1 | x) = 0.5.
    #
    # On one side of this contour the network predicts class 0;
    # on the other side it predicts class 1.
    plt.contour(
        xx,
        yy,
        P,
        levels=[0.5],
        linewidths=2.5
    )

    # Show training examples lightly so the probability surface remains visible.
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

    # Emphasize hold-out observations because they test generalization.
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

    validate_covariances()

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
    #
    # We now solve the SAME classification problem with a small neural network.
    #
    # This is useful pedagogically because the data have not changed.
    # What changes is the model class:
    #
    #   Fisher LDA:
    #       one linear separating hyperplane
    #
    #   2-4-1 neural network:
    #       four nonlinear hidden features whose combination can produce
    #       a curved decision boundary.
    #
    # Everything below is written explicitly rather than delegated to an ML
    # framework so the connection among forward propagation, loss,
    # backpropagation, gradients, and parameter updates remains visible.
    # ---------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("PART II -- HAND-BUILT 2 -> 4 -> 1 SIGMOID NEURAL NETWORK")
    print("-" * 78)

    # Standardize using TRAINING statistics only.
    #
    # We save feature_mean and feature_std because they must also be used later
    # to standardize the dense x1/x2 grid used for the decision-surface plot.
    X_train_std, X_hold_std, feature_mean, feature_std = standardize_from_training(
        X_train, X_hold
    )

    print("\nNetwork architecture:")
    print("  2 input features")
    print("  4 sigmoid hidden neurons")
    print("  1 sigmoid output neuron")
    print("  mini-batch size =", BATCH_SIZE)

    # =====================================================================
    # BEFORE TRAINING: inspect one forward pass
    # =====================================================================
    #
    # This is deliberately shown before the optimizer starts so we can see
    # the raw mechanics of the network while its parameters are still random.
    initial_params = initialize_network()

    # Use the first mini-batch only for demonstration.
    first_batch_X = X_train_std[:BATCH_SIZE]
    first_batch_y = y_train[:BATCH_SIZE]

    # Forward propagation produces predictions plus a cache of intermediate
    # values required for backpropagation.
    initial_output, initial_cache = forward_propagation(
        first_batch_X,
        initial_params
    )

    print("\nExample FIRST forward propagation before learning:")
    print("X batch shape  :", first_batch_X.shape)

    # Z1: four hidden pre-activations for every sample.
    print("Z1 shape       :", initial_cache["Z1"].shape)

    # A1: four hidden sigmoid outputs for every sample.
    print("A1 shape       :", initial_cache["A1"].shape)

    # Z2: one output pre-activation per sample.
    print("Z2 shape       :", initial_cache["Z2"].shape)

    # Final sigmoid probability: one number per sample.
    print("Output shape   :", initial_output.shape)

    print("First 5 predicted probabilities:")
    print(initial_output[:5, 0])

    print("First 5 target labels:")
    print(first_batch_y[:5])

    # =====================================================================
    # BEFORE TRAINING: inspect one backward pass
    # =====================================================================
    #
    # We now propagate error derivatives backward through the exact same
    # mini-batch.  This computes gradients but does NOT yet update parameters.
    initial_grads = backward_propagation(
        first_batch_y,
        initial_params,
        initial_cache
    )

    print("\nGradient shapes from one backward pass:")
    for name, arr in initial_grads.items():
        print(f"  {name}: {arr.shape}")

    # =====================================================================
    # FULL TRAINING
    # =====================================================================
    #
    # train_network() repeatedly executes:
    #
    #     forward pass
    #         ↓
    #     backpropagation
    #         ↓
    #     gradient-descent parameter update
    #
    # across mini-batches until convergence/early stopping.
    params, history = train_network(
        X_train_std,
        y_train,
        X_hold_std,
        y_hold
    )

    # =====================================================================
    # INFERENCE WITH THE TRAINED NETWORK
    # =====================================================================
    #
    # At this stage learning is over.
    # Prediction requires only forward propagation.
    nn_train_pred, nn_train_prob = neural_network_predict(
        X_train_std,
        params
    )

    nn_hold_pred, nn_hold_prob = neural_network_predict(
        X_hold_std,
        params
    )

    # Convert predicted labels into confusion-matrix-derived metrics.
    nn_train_m = confusion_metrics(y_train, nn_train_pred)
    nn_hold_m = confusion_metrics(y_hold, nn_hold_pred)

    print("\nFinal neural-network training metrics:")
    for k, v in nn_train_m.items():
        print(
            f"  {k:20s}: {v:.6f}"
            if isinstance(v, float)
            else f"  {k:20s}: {v}"
        )

    print("\nFinal neural-network hold-out metrics:")
    for k, v in nn_hold_m.items():
        print(
            f"  {k:20s}: {v:.6f}"
            if isinstance(v, float)
            else f"  {k:20s}: {v}"
        )

    # Perform one final forward pass over each entire dataset to report BCE.
    final_train_prob, _ = forward_propagation(X_train_std, params)
    final_hold_prob, _ = forward_propagation(X_hold_std, params)

    print(
        f"\nFinal training BCE = "
        f"{binary_cross_entropy(y_train, final_train_prob):.6f}"
    )

    print(
        f"Final hold-out BCE = "
        f"{binary_cross_entropy(y_hold, final_hold_prob):.6f}"
    )

    # The distributions overlap, so perfect classification is not generally
    # attainable.  Some observations are intrinsically ambiguous even for an
    # optimal classifier.
    print("\nDo NOT expect the loss to reach exactly zero.")
    print("The two Gaussian class distributions overlap, so some observations are")
    print("intrinsically ambiguous.  A non-zero irreducible classification error is")
    print("therefore expected even for an excellent classifier.")

    # Expose the learned numerical parameters.
    #
    # These matrices are the quantities gradient descent has been adjusting
    # throughout training.
    print("\nLearned network parameters:")
    print("W1 =\n", params["W1"])
    print("b1 =\n", params["b1"])
    print("W2 =\n", params["W2"])
    print("b2 =\n", params["b2"])

    # Visualize:
    #   1. optimization convergence
    #   2. the nonlinear probability/decision surface learned by the network
    plot_nn_training(history)
    plot_nn_decision_surface(
        train_df,
        holdout_df,
        params,
        feature_mean,
        feature_std
    )

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
