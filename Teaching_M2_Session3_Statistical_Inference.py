# -*- coding: utf-8 -*-
"""
Teaching_M2_Session3_Statistical_Inference.py

Teaching script for Month 2, Session 3.

Topics
------
1) Sampling variation
2) Bootstrap confidence intervals
3) Permutation tests
4) Chi-square intuition for categorical tables

This file is designed to be walked through live in Spyder.
"""

import numpy as np
import matplotlib.pyplot as plt

# Optional SciPy import for comparison to built-in calculations
try:
    from scipy.stats import chi2_contingency
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# -------------------------------------------------------------------
# Part 1: Sampling variation
# -------------------------------------------------------------------
def sampling_variation_demo():
    section("PART 1 — Sampling variation")

    np.random.seed(42)

    pop_A = np.random.normal(loc=0.0, scale=1.0, size=50000)
    pop_B = np.random.normal(loc=0.5, scale=1.0, size=50000)

    plt.figure()
    plt.hist(pop_A, bins=60, density=True, alpha=0.5, label="Population A ~ N(0,1)")
    plt.hist(pop_B, bins=60, density=True, alpha=0.5, label="Population B ~ N(0.5,1)")
    plt.title("Two populations with slightly different means")
    plt.xlabel("value")
    plt.ylabel("density")
    plt.legend()
    plt.show()

    n = 40
    reps = 500
    diffs = []

    for _ in range(reps):
        a = np.random.choice(pop_A, size=n, replace=False)
        b = np.random.choice(pop_B, size=n, replace=False)
        diffs.append(b.mean() - a.mean())

    diffs = np.array(diffs)
    print(f"Mean of sampled mean differences over {reps} reps: {diffs.mean():.4f}")
    print(f"Std dev of sampled mean differences: {diffs.std():.4f}")

    plt.figure()
    plt.hist(diffs, bins=30)
    plt.title("Sampling distribution of mean differences")
    plt.xlabel("sample mean(B) - sample mean(A)")
    plt.ylabel("count")
    plt.show()


# -------------------------------------------------------------------
# Part 2: Bootstrap confidence interval
# -------------------------------------------------------------------
def bootstrap_means(sample, n_boot=2000):
    means = []
    n = len(sample)
    for _ in range(n_boot):
        boot = np.random.choice(sample, size=n, replace=True)
        means.append(np.mean(boot))
    return np.array(means)


def bootstrap_demo():
    section("PART 2 — Bootstrap confidence interval")

    np.random.seed(42)
    sample = np.random.normal(loc=2.0, scale=1.5, size=40)

    print(f"Sample mean = {sample.mean():.4f}")

    boot_means = bootstrap_means(sample, n_boot=2000)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    print(f"Approximate 95% bootstrap CI = [{ci_low:.4f}, {ci_high:.4f}]")

    plt.figure()
    plt.hist(boot_means, bins=30)
    plt.axvline(ci_low, linestyle="--", label="2.5th percentile")
    plt.axvline(ci_high, linestyle="--", label="97.5th percentile")
    plt.title("Bootstrap distribution of the sample mean")
    plt.xlabel("bootstrap mean")
    plt.ylabel("count")
    plt.legend()
    plt.show()


# -------------------------------------------------------------------
# Part 3: Permutation test
# -------------------------------------------------------------------
def permutation_test_mean_diff(a, b, n_perm=5000):
    observed = np.mean(b) - np.mean(a)
    pooled = np.concatenate([a, b])

    null_diffs = []
    n_a = len(a)

    for _ in range(n_perm):
        shuffled = np.random.permutation(pooled)
        a_star = shuffled[:n_a]
        b_star = shuffled[n_a:]
        null_diffs.append(np.mean(b_star) - np.mean(a_star))

    null_diffs = np.array(null_diffs)
    p_value = np.mean(np.abs(null_diffs) >= abs(observed))
    return observed, null_diffs, p_value


def permutation_demo():
    section("PART 3 — Permutation test")

    np.random.seed(42)
    a = np.random.normal(loc=0.0, scale=1.0, size=40)
    b = np.random.normal(loc=0.4, scale=1.0, size=40)

    observed, null_diffs, p_value = permutation_test_mean_diff(a, b, n_perm=5000)

    print(f"Observed difference in means = {observed:.4f}")
    print(f"Approximate permutation p-value = {p_value:.4f}")

    plt.figure()
    plt.hist(null_diffs, bins=40)
    plt.axvline(observed, linestyle="--", label="observed diff")
    plt.axvline(-observed, linestyle="--")
    plt.title("Permutation null distribution")
    plt.xlabel("difference in means under shuffled labels")
    plt.ylabel("count")
    plt.legend()
    plt.show()

    print("\nInterpretation:")
    print("If the observed difference lies far in the tails of the null distribution,")
    print("the groups are unlikely to be interchangeable.")


# -------------------------------------------------------------------
# Part 4: Chi-square intuition
# -------------------------------------------------------------------
def expected_counts(observed):
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    grand_total = observed.sum()
    return row_totals @ col_totals / grand_total


def chi_square_stat(observed):
    exp = expected_counts(observed)
    stat = np.sum((observed - exp)**2 / exp)
    return stat, exp


def chi_square_demo():
    section("PART 4 — Chi-square for a 2x2 table")

    observed = np.array([[30, 20],
                         [10, 40]], dtype=float)

    stat, exp = chi_square_stat(observed)

    print("Observed table:")
    print(observed)
    print("\nExpected counts under independence:")
    print(exp)
    print(f"\nChi-square statistic = {stat:.4f}")

    if SCIPY_AVAILABLE:
        scipy_stat, scipy_p, _, scipy_exp = chi2_contingency(observed, correction=False)
        print(f"SciPy chi-square statistic = {scipy_stat:.4f}")
        print(f"SciPy p-value = {scipy_p:.4f}")

    plt.figure(figsize=(7, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(observed, cmap="Blues")
    plt.title("Observed")
    plt.colorbar(fraction=0.046)
    plt.subplot(1, 2, 2)
    plt.imshow(exp, cmap="Greens")
    plt.title("Expected under independence")
    plt.colorbar(fraction=0.046)
    plt.tight_layout()
    plt.show()

    print("\nInterpretation:")
    print("Chi-square measures how far the observed counts are from the counts")
    print("we would expect if the two variables were independent.")


def main():
    sampling_variation_demo()
    bootstrap_demo()
    permutation_demo()
    chi_square_demo()

    section("FINAL SUMMARY")
    print("1) Samples vary even when drawn from the same population.")
    print("2) Bootstrap resampling estimates uncertainty from the data itself.")
    print("3) Permutation tests build a null world by shuffling labels.")
    print("4) Chi-square compares observed categorical counts to independence-based expectations.")


if __name__ == "__main__":
    main()
