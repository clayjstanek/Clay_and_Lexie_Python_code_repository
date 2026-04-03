# -*- coding: utf-8 -*-
"""
Teaching_M2_Session3_Statistical_Inference_revised.py

Teaching script for Month 2, Session 3.

Topics
------
0) Gaussian pdf, mean, variance, and standard deviation
1) Law of large numbers and variance of the sample mean
2) Sampling variation
3) Bootstrap confidence intervals
4) Permutation tests
5) Chi-square intuition for categorical tables

This file is designed to be walked through live in Spyder.
"""

import numpy as np
import matplotlib.pyplot as plt

# Optional SciPy import for comparison to built-in calculations
try:
    from scipy.stats import chi2_contingency, ttest_ind, t as student_t
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def gaussian_pdf(x, mu=0.0, sigma=1.0):
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def welch_t_test(a, b):
    """
    Return Welch's two-sample t statistic, approximate degrees of freedom,
    and two-sided p-value.

    We use Welch's test because in practice the two groups may not have
    exactly equal variances, and it is a standard default for comparing means.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    n_a = len(a)
    n_b = len(b)
    mean_a = a.mean()
    mean_b = b.mean()
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)

    se = np.sqrt(var_a / n_a + var_b / n_b)
    t_stat = (mean_b - mean_a) / se

    # Welch-Satterthwaite effective degrees of freedom
    numerator = (var_a / n_a + var_b / n_b) ** 2
    denominator = ((var_a / n_a) ** 2) / (n_a - 1) + ((var_b / n_b) ** 2) / (n_b - 1)
    df = numerator / denominator

    if SCIPY_AVAILABLE:
        p_value = 2 * (1 - student_t.cdf(abs(t_stat), df))
    else:
        # For n around 40 per group, the normal approximation is quite good.
        p_value = 2 * (1 - 0.5 * (1 + np.math.erf(abs(t_stat) / np.sqrt(2))))

    return t_stat, df, p_value


# -------------------------------------------------------------------
# Part 0: Gaussian, mean, variance, standard deviation
# -------------------------------------------------------------------
def gaussian_intro_demo():
    section("PART 0 — Gaussian pdf, mean, variance, and standard deviation")

    np.random.seed(42)
    mu = 2.0
    sigma = 1.5
    sample = np.random.normal(loc=mu, scale=sigma, size=4000)

    x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 500)
    pdf = gaussian_pdf(x, mu=mu, sigma=sigma)

    print("A Gaussian (normal) random variable with mean mu and standard deviation sigma")
    print("has density")
    print("f(x) = (1 / (sigma * sqrt(2*pi))) * exp(-(x - mu)^2 / (2*sigma^2))")
    print(f"\nHere we use mu = {mu:.1f} and sigma = {sigma:.1f}.")
    print("The mean is the center of the distribution.")
    print("The variance measures spread and equals sigma^2.")
    print("The standard deviation is sqrt(variance), so it is in the same units as x.")

    print(f"\nFor the generated sample:")
    print(f"sample mean     = {sample.mean():.4f}")
    print(f"sample variance = {sample.var(ddof=1):.4f}")
    print(f"sample std dev  = {sample.std(ddof=1):.4f}")
    print(f"theoretical variance = sigma^2 = {sigma**2:.4f}")

    plt.figure()
    plt.hist(sample, bins=50, density=True, alpha=0.6, label="sample histogram")
    plt.plot(x, pdf, linewidth=2.5, label="analytic Gaussian pdf")
    plt.axvline(mu, linestyle="--", label="true mean mu")
    plt.title("Histogram of samples with analytic Gaussian pdf overlay")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend()
    plt.show()



# -------------------------------------------------------------------
# Part 0.5: Sum / convolution of two Gaussians
# -------------------------------------------------------------------
def gaussian_sum_convolution_demo():
    section("PART 0.5 — Adding two Gaussians and discrete convolution")

    np.random.seed(42)
    mu = 0.0
    sigma = 1.0
    n = 200000

    x1 = np.random.normal(loc=mu, scale=sigma, size=n)
    x2 = np.random.normal(loc=mu, scale=sigma, size=n)
    summed = x1 + x2

    print("If X ~ N(0, 1) and Y ~ N(0, 1) are independent, then")
    print("X + Y ~ N(0, 2).")
    print("So the mean stays 0, the variance doubles from 1 to 2,")
    print("and the standard deviation becomes sqrt(2).")

    print(f"Empirical mean of X      = {x1.mean():.4f}")
    print(f"Empirical variance of X  = {x1.var(ddof=1):.4f}")
    print(f"Empirical mean of Y      = {x2.mean():.4f}")
    print(f"Empirical variance of Y  = {x2.var(ddof=1):.4f}")
    print(f"Empirical mean of X + Y  = {summed.mean():.4f}")
    print(f"Empirical variance X + Y = {summed.var(ddof=1):.4f}")
    print(f"Theoretical variance     = {sigma**2 + sigma**2:.4f}")
    print(f"Theoretical std dev      = {np.sqrt(2) * sigma:.4f}")

    x = np.linspace(-6, 6, 600)
    pdf_single = gaussian_pdf(x, mu=0.0, sigma=1.0)
    pdf_sum = gaussian_pdf(x, mu=0.0, sigma=np.sqrt(2.0))

    plt.figure()
    plt.hist(x1, bins=60, density=True, alpha=0.45, label="samples from N(0,1)")
    plt.hist(summed, bins=60, density=True, alpha=0.45, label="samples from X + Y")
    plt.plot(x, pdf_single, linewidth=2, label="analytic N(0,1) pdf")
    plt.plot(x, pdf_sum, linewidth=2.5, label="analytic N(0,2) pdf")
    plt.title("Adding two independent Gaussians widens the distribution")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend()
    plt.show()

    # Discrete convolution using histogram bin probabilities
    edges = np.linspace(-6, 6, 241)
    dx = edges[1] - edges[0]
    centers = 0.5 * (edges[:-1] + edges[1:])
    pmf = gaussian_pdf(centers, mu=0.0, sigma=1.0) * dx
    pmf = pmf / pmf.sum()

    conv_pmf = np.convolve(pmf, pmf, mode="full")
    conv_centers = np.arange(len(conv_pmf)) * dx + (2 * centers[0])
    conv_pdf_approx = conv_pmf / dx

    discrete_mean = np.sum(conv_centers * conv_pmf)
    discrete_var = np.sum((conv_centers - discrete_mean) ** 2 * conv_pmf)

    print("Using discrete convolution on a gridded approximation to the Gaussian:")
    print(f"Approximate convolved mean     = {discrete_mean:.4f}")
    print(f"Approximate convolved variance = {discrete_var:.4f}")
    print("This is close to the exact answer N(0,2).")

    x_conv = np.linspace(-8, 8, 800)
    plt.figure()
    plt.plot(centers, pmf / dx, linewidth=2, label="single Gaussian on grid")
    plt.plot(conv_centers, conv_pdf_approx, linewidth=2.5, label="discrete convolution")
    plt.plot(x_conv, gaussian_pdf(x_conv, mu=0.0, sigma=np.sqrt(2.0)),
             linestyle="--", linewidth=2, label="analytic N(0,2) pdf")
    plt.title("Discrete convolution of two Gaussian-shaped grids")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend()
    plt.show()


# -------------------------------------------------------------------
# Part 1: LLN and variance of the sample mean
# -------------------------------------------------------------------
def lln_and_sem_demo():
    section("PART 1 — Law of large numbers and variance of the sample mean")

    np.random.seed(42)
    mu = 2.0
    sigma = 1.5
    long_sample = np.random.normal(loc=mu, scale=sigma, size=2000)

    running_mean = np.cumsum(long_sample) / np.arange(1, len(long_sample) + 1)

    plt.figure()
    plt.plot(running_mean, label="running sample mean")
    plt.axhline(mu, linestyle="--", label="true mean mu")
    plt.title("Law of large numbers: the running mean settles near the true mean")
    plt.xlabel("number of samples N")
    plt.ylabel("running mean")
    plt.legend()
    plt.show()

    print("The law of large numbers says that as N grows, the sample mean tends to")
    print("settle down near the true population mean.")
    print("That does NOT mean individual points get less noisy.")
    print("It means the average becomes more stable.")

    n_values = np.array([5, 10, 20, 40, 80, 160])
    reps = 3000
    empirical_var_means = []
    theoretical_var_means = []

    for n in n_values:
        means = []
        for _ in range(reps):
            draw = np.random.normal(loc=mu, scale=sigma, size=n)
            means.append(draw.mean())
        means = np.array(means)
        empirical_var_means.append(means.var(ddof=1))
        theoretical_var_means.append(sigma**2 / n)

    empirical_var_means = np.array(empirical_var_means)
    theoretical_var_means = np.array(theoretical_var_means)

    print("\nVariance of the sample mean:")
    print("If X has variance sigma^2, then mean(X_1,...,X_N) has variance sigma^2 / N.")
    print("So the sample mean gets more stable like 1/N, and its standard deviation")
    print("(often called the standard error) shrinks like 1/sqrt(N).\n")

    for n, emp, theo in zip(n_values, empirical_var_means, theoretical_var_means):
        print(f"N={n:3d} | empirical Var(sample mean)={emp:.5f} | theory sigma^2/N={theo:.5f}")

    plt.figure()
    plt.plot(n_values, empirical_var_means, marker="o", label="empirical variance of sample mean")
    plt.plot(n_values, theoretical_var_means, marker="o", label="theory: sigma^2 / N")
    plt.title("Variance of the sample mean decreases like 1/N")
    plt.xlabel("sample size N")
    plt.ylabel("variance")
    plt.legend()
    plt.show()


# -------------------------------------------------------------------
# Part 2: Sampling variation
# -------------------------------------------------------------------
def sampling_variation_demo():
    section("PART 2 — Sampling variation")

    np.random.seed(42)

    pop_A = np.random.normal(loc=0.0, scale=1.0, size=50000)
    pop_B = np.random.normal(loc=0.5, scale=1.0, size=50000)

    x = np.linspace(-4, 5, 500)

    plt.figure()
    plt.hist(pop_A, bins=60, density=True, alpha=0.45, label="Population A samples")
    plt.hist(pop_B, bins=60, density=True, alpha=0.45, label="Population B samples")
    plt.plot(x, gaussian_pdf(x, mu=0.0, sigma=1.0), linewidth=2, label="Analytic pdf A")
    plt.plot(x, gaussian_pdf(x, mu=0.5, sigma=1.0), linewidth=2, label="Analytic pdf B")
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
    print(f"Std dev of sampled mean differences: {diffs.std(ddof=1):.4f}")

    plt.figure()
    plt.hist(diffs, bins=30)
    plt.title("Sampling distribution of mean differences")
    plt.xlabel("sample mean(B) - sample mean(A)")
    plt.ylabel("count")
    plt.show()


# -------------------------------------------------------------------
# Part 3: Bootstrap confidence interval
# -------------------------------------------------------------------
def bootstrap_means(sample, n_boot=2000):
    means = []
    n = len(sample)
    for _ in range(n_boot):
        boot = np.random.choice(sample, size=n, replace=True)
        means.append(np.mean(boot))
    return np.array(means)

"""
The simplest comparison

A Gaussian 95% CI for the mean is usually

𝑥 -  ± 1.96𝑠/sqrt(n)
	​
It comes from the idea that the sample mean is approximately normally distributed.

A bootstrap 95% CI for the mean is built by:

resampling your data with replacement many times,

recomputing the mean each time,

taking the middle 95% of those bootstrap means.

So the difference is:

Gaussian CI: uses a formula

Bootstrap CI: uses simulated resampling from the data

What they are both trying to estimate

Both are trying to capture uncertainty in the population mean μ, based on one sample.

They are both answers to: “Given my sample, what range of values is plausible for the true mean?”
"""

def bootstrap_demo():
    section("PART 3 — Bootstrap confidence interval")

    np.random.seed(42)
    mu = 2.0
    sigma = 1.5
    sample = np.random.normal(loc=mu, scale=sigma, size=40)

    print(f"Sample mean = {sample.mean():.4f}")
    print(f"Sample std dev = {sample.std(ddof=1):.4f}")
    print("\nBootstrap idea:")
    print("We only have one sample. So we resample from that sample, with replacement,")
    print("to estimate how much the sample mean would vary across repeated samples.")

    boot_means = bootstrap_means(sample, n_boot=2000)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    print(f"Approximate 95% bootstrap CI = [{ci_low:.4f}, {ci_high:.4f}]")

    plt.figure()
    plt.hist(boot_means, bins=30)
    plt.axvline(ci_low, linestyle="--", label="2.5th percentile")
    plt.axvline(ci_high, linestyle="--", label="97.5th percentile")
    plt.axvline(sample.mean(), linestyle="-", label="original sample mean")
    plt.title("Bootstrap distribution of the sample mean")
    plt.xlabel("bootstrap mean")
    plt.ylabel("count")
    plt.legend()
    plt.show()


# -------------------------------------------------------------------
# Part 4: Permutation test
# -------------------------------------------------------------------
"""
If there were really no difference between the two groups, how often would we see 
a difference in means at least this large just by random relabeling?

Null hypothesis

The permutation test starts with the null hypothesis:

𝐻0 : the two groups come from the same distribution	​

:the two groups come from the same distribution

Under that null, the labels “A” and “B” are arbitrary. If there is no true group effect, 
then all the data values could have been assigned to either group.

That is the key logic.

How the test works

Combine all data into one pooled set.

Shuffle the labels randomly.

Split the shuffled data into two groups of the original sizes.

Compute the difference in means for this shuffled split.

Repeat many times.

Compare the observed difference to this permutation distribution.

If the observed difference is far out in the tail of that distribution, the p-value is small.

Why it makes sense

If the null hypothesis is true, then the group labels should not matter.

So by repeatedly shuffling labels, you generate the distribution of mean differences 
you would expect just from chance, assuming no real group distinction.

Then you ask:

Is my observed difference unusually large compared with that null distribution?
A permutation test:

is more data-driven
uses shuffling instead of a theoretical formula
is especially attractive when sample sizes are small or normality is questionable
"""

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
    section("PART 4 — Permutation test and comparison to a t-test")

    np.random.seed(42)
    a = np.random.normal(loc=0.0, scale=1.0, size=40)
    b = np.random.normal(loc=0.4, scale=1.0, size=40)

    observed, null_diffs, p_value = permutation_test_mean_diff(a, b, n_perm=5000)
    t_stat, df, t_p_value = welch_t_test(a, b)

    print("We now compare two ways to produce a p-value for a difference in means.")
    print("1) Permutation test: build a null distribution by shuffling labels.")
    print("2) Welch t-test: assume the sampling distribution of the mean difference")
    print("   is approximately t-shaped under the null hypothesis.")
    print("\nObserved difference in means = {:.4f}".format(observed))
    print(f"Approximate permutation p-value = {p_value:.4f}")
    print(f"Welch t statistic = {t_stat:.4f}")
    print(f"Approximate degrees of freedom = {df:.2f}")
    print(f"Welch two-sided p-value = {t_p_value:.4f}")

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
    print("Both tests ask the same high-level question:")
    print("'If there were really no difference between the groups, how unusual is")
    print(" the observed difference in means?'")
    print("The permutation test answers that by shuffling labels.")
    print("The t-test answers that by using a theoretical t distribution.")
    print("When sample sizes are decent and the distributions are fairly well-behaved,")
    print("the two p-values are often similar.")

# -------------------------------------------------------------------
# Part 5: Chi-square intuition
# -------------------------------------------------------------------
"""
Chi-square is a very common test for categorical data arranged in a contingency
table. It is most useful when the data are counts, not continuous measurements.

The core question is usually:

    "If the two categorical variables were independent, would counts this far
    from the expected pattern be surprising?"

Examples:
- Does treatment choice appear independent of recovery category?
- Is political party independent of survey response?
- Is sensor type independent of classification outcome?

The null hypothesis is independence. Under that null hypothesis, the pattern in
one direction of the table should not systematically depend on the category in
the other direction.

Why expected counts look like (row total * column total) / grand total
----------------------------------------------------------------------
Suppose 50% of all observations fall in row 1, and 30% of all observations fall
in column 1. If rows and columns are independent, then we would expect about
0.50 * 0.30 = 0.15 of the full table to land in the cell (row 1, col 1).
Multiplying that fraction by the grand total gives the expected count:

    expected_ij = (row_total_i * col_total_j) / grand_total

This gives the full table of expected counts under independence.

How the chi-square statistic is built
-------------------------------------
For each cell, compare:

    observed count - expected count

If that difference is large, that cell contributes evidence against independence.
To make cells with different expected sizes comparable, divide by the expected
count and square the difference:

    (observed - expected)^2 / expected

Then add those contributions over all cells:

    chi_square = sum over cells of (observed - expected)^2 / expected

Large chi-square values mean the observed table is far from what independence
would predict.

How that becomes a p-value
--------------------------
If the null hypothesis is true and the expected counts are not too small, the
chi-square statistic is approximately distributed like a chi-square random
variable with

    degrees of freedom = (number of rows - 1) * (number of columns - 1)

The p-value is the right-tail probability:

    p_value = P(Chi-square random variable >= observed statistic)

So:
- small p-value -> the observed table would be unusual under independence
- large p-value -> the deviations could easily be due to chance

Conditions where the test is most useful
----------------------------------------
- Data are counts in categories
- Observations are reasonably independent
- Expected counts are not too small
- A common classroom rule of thumb is that expected counts should usually be at
  least about 5 in each cell for the chi-square approximation to work well

For very small counts, an exact test (such as Fisher's exact test in a 2x2
table) is often better.

For this lesson, the key intuition is:
chi-square measures how far the observed counts are from the counts predicted by
independence, and then converts that discrepancy into a p-value.
"""
def expected_counts(observed):
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    grand_total = observed.sum()
    return row_totals @ col_totals / grand_total



def chi_square_stat(observed):
    exp = expected_counts(observed)
    stat = np.sum((observed - exp) ** 2 / exp)
    return stat, exp



def chi_square_demo():
    section("PART 5 — Chi-square for a 2x2 table")

    observed = np.array([[30, 20],
                         [10, 40]], dtype=float)

    stat, exp = chi_square_stat(observed)
    dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)

    print("Observed table:")
    print(observed)
    print("\nStep 1: Compute row totals, column totals, and the grand total.")
    print("Step 2: Use independence to compute expected counts.")
    print("Step 3: For each cell, compute (observed - expected)^2 / expected.")
    print("Step 4: Add all those terms to get the chi-square statistic.")
    print("\nExpected counts under independence:")
    print(exp)
    print(f"\nChi-square statistic = {stat:.4f}")
    print(f"Degrees of freedom = {dof}")

    if SCIPY_AVAILABLE:
        scipy_stat, scipy_p, _, scipy_exp = chi2_contingency(observed, correction=False)
        print(f"SciPy chi-square statistic = {scipy_stat:.4f}")
        print(f"SciPy p-value = {scipy_p:.4f}")
    else:
        print("SciPy not available, so no chi-square p-value lookup is shown here.")

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
    gaussian_intro_demo()
    gaussian_sum_convolution_demo()
    lln_and_sem_demo()
    sampling_variation_demo()
    bootstrap_demo()
    permutation_demo()
    chi_square_demo()

    section("FINAL SUMMARY")
    print("1) A Gaussian has an analytic pdf with parameters mu and sigma.")
    print("2) Variance measures spread; standard deviation is its square root.")
    print("3) Adding two independent N(0,1) variables gives N(0,2), so variances add.")
    print("4) The sample mean stabilizes as N grows, and Var(sample mean) shrinks like 1/N.")
    print("5) Samples vary even when drawn from the same population.")
    print("6) Bootstrap resampling estimates uncertainty from the data itself.")
    print("7) Permutation tests build a null world by shuffling labels.")
    print("8) Welch t-tests use a theoretical t distribution for a p-value on mean differences.")
    print("9) Chi-square compares observed categorical counts to independence-based expectations.")


if __name__ == "__main__":
    main()
