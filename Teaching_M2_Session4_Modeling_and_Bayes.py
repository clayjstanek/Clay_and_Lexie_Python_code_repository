# -*- coding: utf-8 -*-
"""
Teaching_M2_Session4_Modeling_and_Bayes.py

Teaching script for Month 2, Session 4.

Topics
------
1) Best-fit line via least squares
2) Residuals
3) Logistic curve as probability mapping
4) Bayesian updating with a Beta prior
"""

import numpy as np
import matplotlib.pyplot as plt
from math import gamma


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# -------------------------------------------------------------------
# Part 1: Best-fit line
# -------------------------------------------------------------------
def best_fit_line_demo():
    section("PART 1 — Best-fit line")

    np.random.seed(42)

    x = np.linspace(-5, 5, 25).reshape(-1, 1)
    m_true = 1.6
    b_true = -0.8
    noise = np.random.normal(0, 1.2, size=(25, 1))
    y = m_true * x + b_true + noise

    # Design matrix with x and intercept column
    A = np.hstack([x, np.ones_like(x)])

    # Least squares estimate
    beta_hat, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
    m_hat = beta_hat[0, 0]
    b_hat = beta_hat[1, 0]
    m_and_b = np.linalg.inv(A.T@A)@A.T@y

    print(f"True slope = {m_true:.3f}, estimated slope = {m_hat:.3f}")
    print(f"True intercept = {b_true:.3f}, estimated intercept = {b_hat:.3f}")
    print(f"Rank of design matrix = {rank}")

    x_line = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
    y_line = m_hat * x_line + b_hat

    plt.figure()
    plt.scatter(x, y, label="noisy data")
    plt.plot(x_line, y_line, label="best-fit line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Least squares line fit")
    plt.legend()
    plt.show()

    return x, y, A, beta_hat


# -------------------------------------------------------------------
# Part 2: Residuals
# -------------------------------------------------------------------
def residual_demo(x, y, A, beta_hat):
    section("PART 2 — Residuals")

    y_hat = A @ beta_hat
    residuals = y - y_hat

    print(f"Residual mean ≈ {residuals.mean():.4f}")
    print(f"Residual std  ≈ {residuals.std():.4f}")
    print(f"Residual norm = {np.linalg.norm(residuals):.4f}")

    plt.figure()
    plt.scatter(x, residuals)
    plt.axhline(0, linestyle="--")
    plt.xlabel("x")
    plt.ylabel("residual")
    plt.title("Residuals vs x")
    plt.show()

    plt.figure()
    plt.hist(residuals, bins=12)
    plt.xlabel("residual")
    plt.ylabel("count")
    plt.title("Residual histogram")
    plt.show()

    print("\nInterpretation:")
    print("Residuals are the errors left over after fitting the model.")
    print("They tell us what structure the model failed to explain.")


# -------------------------------------------------------------------
# Part 3: Logistic curve
# -------------------------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_demo():
    section("PART 3 — Logistic curve")

    z = np.linspace(-8, 8, 400)

    plt.figure()
    plt.plot(z, sigmoid(z), label="sigma(z)")
    plt.plot(z, sigmoid(2*z), label="sigma(2z)")
    plt.plot(z, sigmoid(0.5*z), label="sigma(0.5z)")
    plt.xlabel("z")
    plt.ylabel("probability-like output")
    plt.title("Logistic function and slope changes")
    plt.legend()
    plt.show()

    print("Interpretation:")
    print("The logistic function maps any real-valued input to the interval (0,1),")
    print("which is why it is useful for probability-style modeling.")


# -------------------------------------------------------------------
# Part 4: Bayesian update with Beta prior
# -------------------------------------------------------------------
def beta_pdf(x, a, b):
    B = gamma(a) * gamma(b) / gamma(a + b)
    return (x**(a - 1) * (1 - x)**(b - 1)) / B


def bayes_demo():
    section("PART 4 — Beta prior and posterior")

    np.random.seed(42)

    true_p = 0.7
    n_flips = 30
    flips = np.random.binomial(1, true_p, size=n_flips)

    H = flips.sum()
    T = n_flips - H

    print(f"Observed heads = {H}, tails = {T}")

    priors = [(1, 1), (5, 2)]
    x = np.linspace(0.001, 0.999, 400)

    plt.figure()

    for a, b in priors:
        post_a = a + H
        post_b = b + T

        plt.plot(x, beta_pdf(x, a, b), linestyle="--", label=f"Prior Beta({a},{b})")
        plt.plot(x, beta_pdf(x, post_a, post_b), label=f"Posterior Beta({post_a},{post_b})")

        prior_mean = a / (a + b)
        post_mean = post_a / (post_a + post_b)
        print(f"\nPrior Beta({a},{b}): mean = {prior_mean:.4f}")
        print(f"Posterior Beta({post_a},{post_b}): mean = {post_mean:.4f}")

    plt.xlabel("p")
    plt.ylabel("density")
    plt.title("Bayesian updating for coin probability")
    plt.legend()
    plt.show()

    print("\nInterpretation:")
    print("The prior expresses beliefs before data.")
    print("The posterior updates those beliefs using observed evidence.")


def main():
    x, y, A, beta_hat = best_fit_line_demo()
    residual_demo(x, y, A, beta_hat)
    logistic_demo()
    bayes_demo()

    section("FINAL SUMMARY")
    print("1) Least squares finds parameters that best fit noisy data.")
    print("2) Residuals show what the model missed.")
    print("3) The logistic curve turns scores into probabilities.")
    print("4) Bayesian updating combines prior beliefs with observed data.")


if __name__ == "__main__":
    main()
