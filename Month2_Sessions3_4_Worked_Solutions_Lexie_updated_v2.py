
# -*- coding: utf-8 -*-
"""
Month 2 Sessions 3 and 4 -- Detailed Worked Solutions for Lexie

This file is intentionally LONG and heavily commented.
It is designed to teach, not just to give answers.

What is inside?
---------------
SESSION 3: Statistical Inference
    A1) Simulate two Gaussian populations and bootstrap a 95% CI
    A2) Perform a permutation test for a difference in means
    A3) Compute a chi-square test by hand and compare to SciPy
    Project) Titanic mini-project using a 2x2 contingency table

SESSION 4: Modeling and Bayesian Methods
    A1) Linear regression using the normal equation
    A2) Residual analysis
    A3) Logistic curve interpretation
    Project) Logistic regression on Titanic WITHOUT sklearn

Important teaching note for Lexie
---------------------------------
For the Session 3 Titanic project, the original assignment says to test
independence of survival and passenger class. If we use the passenger class
exactly as-is, then class has 3 categories (1st, 2nd, 3rd), so the natural
contingency table is 2x3 and the degrees of freedom would be:

    (2 - 1) * (3 - 1) = 2

But Clay specifically wants a 2x2 example so the arithmetic is easier to learn.
So in this worked solution, we create a NEW binary column:

    UpperClass = 1 if Pclass is 1 or 2
    UpperClass = 0 if Pclass is 3

That turns the problem into a 2x2 table:
    rows    = Survived (0/1)
    columns = UpperClass (0/1)

Then the degrees of freedom become:

    (2 - 1) * (2 - 1) = 1

This is perfectly fine as a teaching example, as long as we clearly state what
we changed and why.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# We try to import SciPy so we can compare "our code" to a standard library result.
# The script still teaches the core ideas even if SciPy is missing.
try:
    from scipy.stats import chi2, chi2_contingency
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# =============================================================================
# General helper functions
# =============================================================================
def section(title):
    """Pretty print a section header."""
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def check_file_exists(path):
    """Raise a clear error if the expected file is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find file: {path}")


# =============================================================================
# SESSION 3 -- Statistical Inference
# =============================================================================

# -----------------------------------------------------------------------------
# Session 3, Exercise 1
# Simulate two Gaussian populations and bootstrap a 95% confidence interval
# -----------------------------------------------------------------------------
def bootstrap_means(sample, n_boot=3000, seed=123):
    """
    Create many bootstrap resamples of 'sample' and record their means.

    Bootstrap idea:
    We only have one sample in hand. To estimate how variable the sample mean is,
    we resample FROM THAT SAMPLE with replacement many times.

    Each bootstrap sample has the same size as the original sample.
    """
    rng = np.random.default_rng(seed)
    sample = np.asarray(sample)
    n = len(sample)
    means = []

    for _ in range(n_boot):
        boot = rng.choice(sample, size=n, replace=True)
        means.append(np.mean(boot))

    return np.array(means)


def session3_ex1_bootstrap():
    section("SESSION 3 - EXERCISE 1: Gaussian populations and bootstrap 95% CI")

    rng = np.random.default_rng(42)

    # Simulate two populations
    # Population A has mean 0.0, Population B has mean 0.6
    # Both have standard deviation 1.0
    pop_A = rng.normal(loc=0.0, scale=1.0, size=100000)
    pop_B = rng.normal(loc=0.6, scale=1.0, size=100000)

    # Take one sample from each population
    sample_size = 40
    sample_A = rng.choice(pop_A, size=sample_size, replace=False)
    sample_B = rng.choice(pop_B, size=sample_size, replace=False)

    mean_A = np.mean(sample_A)
    mean_B = np.mean(sample_B)

    print("Population A was simulated from N(0.0, 1.0^2)")
    print("Population B was simulated from N(0.6, 1.0^2)")
    print(f"Sample size from each population = {sample_size}")
    print(f"Sample mean A = {mean_A:.4f}")
    print(f"Sample mean B = {mean_B:.4f}")

    # Bootstrap a CI for the mean of sample_A
    boot_A = bootstrap_means(sample_A, n_boot=3000, seed=123)
    ci_A = np.percentile(boot_A, [2.5, 97.5])

    # Bootstrap a CI for the mean of sample_B
    boot_B = bootstrap_means(sample_B, n_boot=3000, seed=456)
    ci_B = np.percentile(boot_B, [2.5, 97.5])

    print("\nBootstrap 95% confidence intervals:")
    print(f"Mean of A: [{ci_A[0]:.4f}, {ci_A[1]:.4f}]")
    print(f"Mean of B: [{ci_B[0]:.4f}, {ci_B[1]:.4f}]")

    print("\nInterpretation:")
    print("A bootstrap CI tries to give a plausible range for the TRUE population mean")
    print("based only on the sample we have in hand.")

    plt.figure()
    plt.hist(boot_A, bins=30, alpha=0.6, label="Bootstrap means for A")
    plt.axvline(ci_A[0], linestyle="--", label="A CI lower")
    plt.axvline(ci_A[1], linestyle="--", label="A CI upper")
    plt.axvline(mean_A, linestyle="-", label="Original mean A")
    plt.title("Bootstrap distribution of the mean for Sample A")
    plt.xlabel("Bootstrap sample mean")
    plt.ylabel("Count")
    plt.legend()
    plt.show()

    plt.figure()
    plt.hist(boot_B, bins=30, alpha=0.6, label="Bootstrap means for B")
    plt.axvline(ci_B[0], linestyle="--", label="B CI lower")
    plt.axvline(ci_B[1], linestyle="--", label="B CI upper")
    plt.axvline(mean_B, linestyle="-", label="Original mean B")
    plt.title("Bootstrap distribution of the mean for Sample B")
    plt.xlabel("Bootstrap sample mean")
    plt.ylabel("Count")
    plt.legend()
    plt.show()


# -----------------------------------------------------------------------------
# Session 3, Exercise 2
# Permutation test for a difference in means
# -----------------------------------------------------------------------------
def permutation_test_mean_diff(a, b, n_perm=5000, seed=1234):
    """
    Two-sided permutation test for the difference in means.

    Null hypothesis:
        The two groups come from the same distribution.

    If that is true, then the labels "group A" and "group B" are exchangeable.
    So we can shuffle the pooled data and rebuild fake groups many times.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a)
    b = np.asarray(b)

    observed = np.mean(b) - np.mean(a)
    pooled = np.concatenate([a, b])
    n_a = len(a)

    null_diffs = []
    for _ in range(n_perm):
        shuffled = rng.permutation(pooled)
        a_star = shuffled[:n_a]
        b_star = shuffled[n_a:]
        null_diffs.append(np.mean(b_star) - np.mean(a_star))

    null_diffs = np.array(null_diffs)
    p_value = np.mean(np.abs(null_diffs) >= abs(observed))
    return observed, null_diffs, p_value


def session3_ex2_permutation():
    section("SESSION 3 - EXERCISE 2: Permutation test for a difference in means")

    rng = np.random.default_rng(100)

    # Build two samples with a modest mean difference
    group_A = rng.normal(loc=0.0, scale=1.0, size=35)
    group_B = rng.normal(loc=0.5, scale=1.0, size=35)

    observed, null_diffs, p_value = permutation_test_mean_diff(group_A, group_B)

    print(f"Observed difference in means (B - A) = {observed:.4f}")
    print(f"Approximate two-sided permutation p-value = {p_value:.4f}")

    print("\nInterpretation:")
    print("This p-value asks:")
    print("'If there were really no group difference, how often would random shuffling")
    print(" produce a difference in means at least this large?'")

    plt.figure()
    plt.hist(null_diffs, bins=40)
    plt.axvline(observed, linestyle="--", label="Observed diff")
    plt.axvline(-observed, linestyle="--")
    plt.title("Permutation null distribution of the difference in means")
    plt.xlabel("Shuffled mean difference")
    plt.ylabel("Count")
    plt.legend()
    plt.show()


# -----------------------------------------------------------------------------
# Session 3, Exercise 3
# Chi-square test by hand, then compare to scipy
# -----------------------------------------------------------------------------
def expected_counts(observed):
    """
    Compute expected counts under independence.

    Formula:
        expected[i,j] = (row_total_i * col_total_j) / grand_total
    """
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    grand_total = observed.sum()
    return row_totals @ col_totals / grand_total


def chi_square_components(observed):
    """
    Return:
        expected counts
        cell-by-cell contributions (O-E)^2 / E
        total chi-square statistic
    """
    exp = expected_counts(observed)
    components = (observed - exp) ** 2 / exp
    chi_sq = components.sum()
    return exp, components, chi_sq


def session3_ex3_chisquare():
    section("SESSION 3 - EXERCISE 3: Chi-square by hand and compare to SciPy")

    # Example 2x2 table
    observed = np.array([
        [30, 20],
        [10, 40]
    ], dtype=float)

    exp, components, chi_sq = chi_square_components(observed)
    dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)

    print("Observed table:")
    print(observed)

    print("\nExpected table under independence:")
    print(np.round(exp, 4))

    print("\nCell-by-cell chi-square contributions = (O - E)^2 / E")
    print(np.round(components, 4))

    print(f"\nTotal chi-square statistic = {chi_sq:.4f}")
    print(f"Degrees of freedom = {dof}")

    if SCIPY_AVAILABLE:
        scipy_stat, scipy_p, _, scipy_exp = chi2_contingency(observed, correction=False)
        print(f"SciPy chi-square statistic (no Yates correction) = {scipy_stat:.4f}")
        print(f"SciPy p-value = {scipy_p:.6f}")

    # Print a small critical value table for dof = 1
    # Here p means RIGHT-TAIL probability.
    if SCIPY_AVAILABLE and dof == 1:
        print("\nChi-square critical values for 1 degree of freedom:")
        for p in [0.2, 0.1, 0.05, 0.01, 0.001]:
            crit = chi2.isf(p, 1)
            print(f"Right-tail p = {p:>5}: critical chi-square = {crit:.4f}")


# -----------------------------------------------------------------------------
# Session 3 Project
# Titanic: create a 2x2 contingency table and do the chi-square arithmetic
# -----------------------------------------------------------------------------
def load_titanic_dataframe():
    """
    Load the Titanic CSV.
    We first try the exact uploaded filename. If that is not found, we try the
    cleaned file. This makes the script a little more forgiving.
    """
    preferred_paths = [
        "/mnt/data/titanic(1).csv",
        "/mnt/data/titanic_cleaned(1).csv",
        "titanic(1).csv",
        "titanic_cleaned(1).csv",
        "Titanic.csv",
        "titanic.csv",
    ]

    for path in preferred_paths:
        if os.path.exists(path):
            print(f"Loaded Titanic data from: {path}")
            return pd.read_csv(path)

    raise FileNotFoundError("Could not find a Titanic CSV file in expected locations.")


def session3_project_titanic_chisquare():
    section("SESSION 3 - PROJECT: Titanic 2x2 contingency table, step by step")

    df = load_titanic_dataframe()

    # -------------------------------------------------------------------------
    # STEP 1: Remind Lexie how a DataFrame assignment works.
    # -------------------------------------------------------------------------
    print("Step 1: Create a NEW column using DataFrame assignment.")
    print("We want a 2x2 table, so we convert Pclass into a binary variable.")
    print("UpperClass = 1 if Pclass is 1 or 2, otherwise 0 if Pclass is 3.")

    # This is the part Clay suspected she may be forgetting:
    # DataFrame assignment means:
    #     df['NewColumn'] = some_expression_using_existing_columns
    df["UpperClass"] = np.where(df["Pclass"].isin([1, 2]), 1, 0)

    print("\nPreview of the columns we care about:")
    print(df[["Survived", "Pclass", "UpperClass"]].head(10))

    # -------------------------------------------------------------------------
    # STEP 2: Build the contingency table.
    # -------------------------------------------------------------------------
    print("\nStep 2: Build a 2x2 contingency table using pd.crosstab.")
    print("Rows = Survived (0=no, 1=yes)")
    print("Cols = UpperClass (0=lower class, 1=upper class)")

    contingency_df = pd.crosstab(df["Survived"], df["UpperClass"])

    # Reorder columns so the printed table is easier to read:
    # column 1 = UpperClass=1, column 0 = UpperClass=0
    contingency_df = contingency_df[[1, 0]]
    contingency_df.columns = ["UpperClass(1or2)", "LowerClass(3)"]
    contingency_df.index = ["Died (Survived=0)", "Survived (Survived=1)"]

    print("\n2x2 contingency table:")
    print(contingency_df)

    # Convert to numpy for arithmetic
    observed = contingency_df.to_numpy(dtype=float)

    # -------------------------------------------------------------------------
    # STEP 3: Compute row totals, column totals, grand total.
    # -------------------------------------------------------------------------
    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)
    grand_total = observed.sum()

    print("\nStep 3: Totals")
    print(f"Row totals    = {row_totals}")
    print(f"Column totals = {col_totals}")
    print(f"Grand total   = {grand_total}")

    # -------------------------------------------------------------------------
    # STEP 4: Compute expected counts.
    # -------------------------------------------------------------------------
    # IMPORTANT TEACHING NOTE:
    # These are EXPECTED COUNTS, not "expected means".
    # The chi-square test for a contingency table works with counts.
    exp = expected_counts(observed)

    print("\nStep 4: Expected counts under independence")
    print("Formula for each cell:")
    print("Expected = (row total * column total) / grand total")
    print("\nExpected table:")
    print(np.round(exp, 6))

    # Show each expected count individually for maximum clarity
    print("\nExpected count calculations, one cell at a time:")
    print(f"E[0,0] = ({row_totals[0]:.0f} * {col_totals[0]:.0f}) / {grand_total:.0f} = {exp[0,0]:.6f}")
    print(f"E[0,1] = ({row_totals[0]:.0f} * {col_totals[1]:.0f}) / {grand_total:.0f} = {exp[0,1]:.6f}")
    print(f"E[1,0] = ({row_totals[1]:.0f} * {col_totals[0]:.0f}) / {grand_total:.0f} = {exp[1,0]:.6f}")
    print(f"E[1,1] = ({row_totals[1]:.0f} * {col_totals[1]:.0f}) / {grand_total:.0f} = {exp[1,1]:.6f}")

    # -------------------------------------------------------------------------
    # STEP 5: Compute each cell contribution (O-E)^2 / E
    # -------------------------------------------------------------------------
    diff = observed - exp
    squared_diff = diff ** 2
    contributions = squared_diff / exp

    print("\nStep 5: Chi-square arithmetic cell by cell")
    print("For each cell:")
    print("  difference      = observed - expected")
    print("  squared diff    = (observed - expected)^2")
    print("  contribution    = (observed - expected)^2 / expected")

    # Print a detailed walkthrough for each cell
    cell_names = [
        ("Died, UpperClass", 0, 0),
        ("Died, LowerClass", 0, 1),
        ("Survived, UpperClass", 1, 0),
        ("Survived, LowerClass", 1, 1),
    ]

    for label, i, j in cell_names:
        print(f"\n{label}")
        print(f"  Observed O = {observed[i,j]:.6f}")
        print(f"  Expected E = {exp[i,j]:.6f}")
        print(f"  O - E      = {diff[i,j]:.6f}")
        print(f"  (O - E)^2  = {squared_diff[i,j]:.6f}")
        print(f"  (O - E)^2 / E = {contributions[i,j]:.6f}")

    # -------------------------------------------------------------------------
    # STEP 6: Add contributions.
    # -------------------------------------------------------------------------
    # Clay asked to "add them up columnwise to produce two numbers."
    # We do that first, then also sum everything to get the final chi-square.
    column_sums = contributions.sum(axis=0)
    row_sums = contributions.sum(axis=1)
    chi_sq = contributions.sum()

    print("\nStep 6: Add the contributions")
    print(f"Column sums of chi-square contributions = {np.round(column_sums, 6)}")
    print(f"Row sums of chi-square contributions    = {np.round(row_sums, 6)}")
    print(f"Total chi-square statistic              = {chi_sq:.6f}")

    # -------------------------------------------------------------------------
    # STEP 7: Degrees of freedom
    # -------------------------------------------------------------------------
    dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    print("\nStep 7: Degrees of freedom")
    print(f"dof = (rows - 1) * (cols - 1) = (2 - 1) * (2 - 1) = {dof}")

    # -------------------------------------------------------------------------
    # STEP 8: Critical values at the requested p values
    # -------------------------------------------------------------------------
    print("\nStep 8: Chi-square critical values for 1 degree of freedom")
    critical_ps = [0.2, 0.1, 0.05, 0.01, 0.001]

    if SCIPY_AVAILABLE:
        for p in critical_ps:
            crit = chi2.isf(p, 1)
            print(f"Right-tail p = {p:>5}: critical value = {crit:.6f}")
    else:
        # Hard-code the key values requested in case SciPy is not installed
        # These are standard chi-square critical values for 1 degree of freedom.
        hardcoded = {
            0.2: 1.642374,
            0.1: 2.705543,
            0.05: 3.841459,
            0.01: 6.634897,
            0.001: 10.827566,
        }
        for p in critical_ps:
            print(f"Right-tail p = {p:>5}: critical value = {hardcoded[p]:.6f}")

    # -------------------------------------------------------------------------
    # STEP 9: p-value from SciPy (comparison)
    # -------------------------------------------------------------------------
    if SCIPY_AVAILABLE:
        scipy_stat, scipy_p, _, scipy_expected = chi2_contingency(observed, correction=False)
        print("\nStep 9: Compare our manual work to SciPy")
        print(f"Our chi-square statistic = {chi_sq:.6f}")
        print(f"SciPy chi-square statistic = {scipy_stat:.6f}")
        print(f"SciPy p-value = {scipy_p:.12f}")

    # -------------------------------------------------------------------------
    # STEP 10: Interpretation
    # -------------------------------------------------------------------------
    print("\nInterpretation:")
    print("The chi-square statistic is very large relative to the common 1-dof")
    print("critical values, so survival and this binary class grouping are NOT")
    print("independent in the Titanic data.")
    print("In plain English: being in upper class versus lower class appears to be")
    print("strongly associated with survival in this dataset.")

    # Simple visual
    plt.figure()
    plt.imshow(observed)
    plt.xticks([0, 1], ["UpperClass(1or2)", "LowerClass(3)"])
    plt.yticks([0, 1], ["Died", "Survived"])
    plt.title("Observed 2x2 Titanic contingency table")
    plt.colorbar()
    plt.show()


def session3_project_titanic_chisquare_2():
    section("SESSION 3 - PROJECT: Titanic 2x2 contingency table, step by step (Sex vs Survived)")

    df = load_titanic_dataframe()

    # -------------------------------------------------------------------------
    # STEP 1: Remind Lexie how a DataFrame assignment works.
    # -------------------------------------------------------------------------
    print("Step 1: Create a NEW column using DataFrame assignment.")
    print("We want a 2x2 table, so we use Sex as the binary variable.")
    print("Gender = 1 if Sex is female, otherwise 0 if Sex is male.")

    # This is the part Clay suspected she may be forgetting:
    # DataFrame assignment means:
    #     df['NewColumn'] = some_expression_using_existing_columns
    df["Gender"] = np.where(df["Sex"].str.lower() == "female", 1, 0)

    print("\nPreview of the columns we care about:")
    print(df[["Survived", "Sex", "Gender"]].head(10))

    # -------------------------------------------------------------------------
    # STEP 2: Build the contingency table.
    # -------------------------------------------------------------------------
    print("\nStep 2: Build a 2x2 contingency table using pd.crosstab.")
    print("Rows = Survived (0=no, 1=yes)")
    print("Cols = Gender (0=male, 1=female)")

    contingency_df = pd.crosstab(df["Survived"], df["Gender"])

    # Reorder columns so the printed table is easier to read:
    # column 1 = Gender=1, column 0 = Gender=0
    contingency_df = contingency_df[[1, 0]]
    contingency_df.columns = ["Female", "Male"]
    contingency_df.index = ["Died (Survived=0)", "Survived (Survived=1)"]

    print("\n2x2 contingency table:")
    print(contingency_df)

    # Convert to numpy for arithmetic
    observed = contingency_df.to_numpy(dtype=float)

    # -------------------------------------------------------------------------
    # STEP 3: Compute row totals, column totals, grand total.
    # -------------------------------------------------------------------------
    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)
    grand_total = observed.sum()

    print("\nStep 3: Totals")
    print(f"Row totals    = {row_totals}")
    print(f"Column totals = {col_totals}")
    print(f"Grand total   = {grand_total}")

    # -------------------------------------------------------------------------
    # STEP 4: Compute expected counts.
    # -------------------------------------------------------------------------
    # IMPORTANT TEACHING NOTE:
    # These are EXPECTED COUNTS, not "expected means".
    # The chi-square test for a contingency table works with counts.
    exp = expected_counts(observed)

    print("\nStep 4: Expected counts under independence")
    print("Formula for each cell:")
    print("Expected = (row total * column total) / grand total")
    print("\nExpected table:")
    print(np.round(exp, 6))

    # Show each expected count individually for maximum clarity
    print("\nExpected count calculations, one cell at a time:")
    print(f"E[0,0] = ({row_totals[0]:.0f} * {col_totals[0]:.0f}) / {grand_total:.0f} = {exp[0,0]:.6f}")
    print(f"E[0,1] = ({row_totals[0]:.0f} * {col_totals[1]:.0f}) / {grand_total:.0f} = {exp[0,1]:.6f}")
    print(f"E[1,0] = ({row_totals[1]:.0f} * {col_totals[0]:.0f}) / {grand_total:.0f} = {exp[1,0]:.6f}")
    print(f"E[1,1] = ({row_totals[1]:.0f} * {col_totals[1]:.0f}) / {grand_total:.0f} = {exp[1,1]:.6f}")

    # -------------------------------------------------------------------------
    # STEP 5: Compute each cell contribution (O-E)^2 / E
    # -------------------------------------------------------------------------
    diff = observed - exp
    squared_diff = diff ** 2
    contributions = squared_diff / exp

    print("\nStep 5: Chi-square arithmetic cell by cell")
    print("For each cell:")
    print("  difference      = observed - expected")
    print("  squared diff    = (observed - expected)^2")
    print("  contribution    = (observed - expected)^2 / expected")

    # Print a detailed walkthrough for each cell
    cell_names = [
        ("Died, Female", 0, 0),
        ("Died, Male", 0, 1),
        ("Survived, Female", 1, 0),
        ("Survived, Male", 1, 1),
    ]

    for label, i, j in cell_names:
        print(f"\n{label}")
        print(f"  Observed O = {observed[i,j]:.6f}")
        print(f"  Expected E = {exp[i,j]:.6f}")
        print(f"  O - E      = {diff[i,j]:.6f}")
        print(f"  (O - E)^2  = {squared_diff[i,j]:.6f}")
        print(f"  (O - E)^2 / E = {contributions[i,j]:.6f}")

    # -------------------------------------------------------------------------
    # STEP 6: Add contributions.
    # -------------------------------------------------------------------------
    # Clay asked to "add them up columnwise to produce two numbers."
    # We do that first, then also sum everything to get the final chi-square.
    column_sums = contributions.sum(axis=0)
    row_sums = contributions.sum(axis=1)
    chi_sq = contributions.sum()

    print("\nStep 6: Add the contributions")
    print(f"Column sums of chi-square contributions = {np.round(column_sums, 6)}")
    print(f"Row sums of chi-square contributions    = {np.round(row_sums, 6)}")
    print(f"Total chi-square statistic              = {chi_sq:.6f}")

    # -------------------------------------------------------------------------
    # STEP 7: Degrees of freedom
    # -------------------------------------------------------------------------
    dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    print("\nStep 7: Degrees of freedom")
    print(f"dof = (rows - 1) * (cols - 1) = (2 - 1) * (2 - 1) = {dof}")

    # -------------------------------------------------------------------------
    # STEP 8: Critical values at the requested p values
    # -------------------------------------------------------------------------
    print("\nStep 8: Chi-square critical values for 1 degree of freedom")
    critical_ps = [0.2, 0.1, 0.05, 0.01, 0.001]

    if SCIPY_AVAILABLE:
        for p in critical_ps:
            crit = chi2.isf(p, 1)
            print(f"Right-tail p = {p:>5}: critical value = {crit:.6f}")
    else:
        # Hard-code the key values requested in case SciPy is not installed
        # These are standard chi-square critical values for 1 degree of freedom.
        hardcoded = {
            0.2: 1.642374,
            0.1: 2.705543,
            0.05: 3.841459,
            0.01: 6.634897,
            0.001: 10.827566,
        }
        for p in critical_ps:
            print(f"Right-tail p = {p:>5}: critical value = {hardcoded[p]:.6f}")

    # -------------------------------------------------------------------------
    # STEP 9: p-value from SciPy (comparison)
    # -------------------------------------------------------------------------
    if SCIPY_AVAILABLE:
        scipy_stat, scipy_p, _, scipy_expected = chi2_contingency(observed, correction=False)
        print("\nStep 9: Compare our manual work to SciPy")
        print(f"Our chi-square statistic = {chi_sq:.6f}")
        print(f"SciPy chi-square statistic = {scipy_stat:.6f}")
        print(f"SciPy p-value = {scipy_p:.12f}")

    # -------------------------------------------------------------------------
    # STEP 10: Interpretation
    # -------------------------------------------------------------------------
    print("\nInterpretation:")
    print("The chi-square statistic is very large relative to the common 1-dof")
    print("critical values, so survival and gender are NOT")
    print("independent in the Titanic data.")
    print("In plain English: sex/gender appears to be")
    print("strongly associated with survival in this dataset.")

    # Contingency-style bar chart (this matches the homework request more directly
    # than an image-style heat map).
    contingency_plot_df = pd.crosstab(df["Sex"], df["Survived"])
    contingency_plot_df = contingency_plot_df[[0, 1]]
    contingency_plot_df.columns = ["Died", "Survived"]

    plt.figure()
    contingency_plot_df.plot(kind="bar")
    plt.title("Titanic counts by Sex and Survival")
    plt.xlabel("Sex")
    plt.ylabel("Count")
    plt.legend(title="Outcome")
    plt.tight_layout()
    plt.show()

    # A second visual: the raw 2x2 count table shown as an image.
    plt.figure()
    plt.imshow(observed)
    plt.xticks([0, 1], ["Female", "Male"])
    plt.yticks([0, 1], ["Died", "Survived"])
    plt.title("Observed 2x2 Titanic contingency table")
    plt.colorbar()
    plt.show()


def session3_project_titanic_permutation_numeric():
    section("SESSION 3 - PROJECT: Titanic permutation test for mean Fare")

    df = load_titanic_dataframe().copy()

    # -------------------------------------------------------------------------
    # STEP 1: Choose a numeric comparison for the permutation test.
    # -------------------------------------------------------------------------
    # The homework asks for a numeric comparison such as:
    #   mean Fare for survivors vs non-survivors
    # or
    #   mean Age for survivors vs non-survivors after cleaning missing values
    #
    # We use Fare here because it is naturally numeric and usually has little or
    # no missing data in the Titanic dataset.
    print("Step 1: Choose a numeric variable and two groups to compare.")
    print("We will compare mean Fare for survivors versus non-survivors.")

    df["Fare_filled"] = df["Fare"].fillna(df["Fare"].median())

    fare_non_survivors = df.loc[df["Survived"] == 0, "Fare_filled"].to_numpy(dtype=float)
    fare_survivors = df.loc[df["Survived"] == 1, "Fare_filled"].to_numpy(dtype=float)

    mean_non_survivors = np.mean(fare_non_survivors)
    mean_survivors = np.mean(fare_survivors)
    observed_diff = mean_survivors - mean_non_survivors

    print(f"Number of non-survivors = {len(fare_non_survivors)}")
    print(f"Number of survivors     = {len(fare_survivors)}")
    print(f"Mean Fare for non-survivors = {mean_non_survivors:.6f}")
    print(f"Mean Fare for survivors     = {mean_survivors:.6f}")
    print(f"Observed difference (survivors - non-survivors) = {observed_diff:.6f}")

    # -------------------------------------------------------------------------
    # STEP 2: Show the observed fare distributions.
    # -------------------------------------------------------------------------
    plt.figure()
    plt.hist(fare_non_survivors, bins=30, alpha=0.6, label="Non-survivors")
    plt.hist(fare_survivors, bins=30, alpha=0.6, label="Survivors")
    plt.xlabel("Fare")
    plt.ylabel("Count")
    plt.title("Titanic Fare distributions for survivors vs non-survivors")
    plt.legend()
    plt.show()

    # -------------------------------------------------------------------------
    # STEP 3: Run the permutation test.
    # -------------------------------------------------------------------------
    print("\nStep 2: Build the permutation null distribution.")
    print("Null hypothesis:")
    print("Survival status and Fare are unrelated with respect to the mean,")
    print("so the survivor / non-survivor labels can be shuffled.")

    observed, null_diffs, p_value = permutation_test_mean_diff(
        fare_non_survivors,
        fare_survivors,
        n_perm=5000,
        seed=2026
    )

    print(f"Permutation-test observed difference = {observed:.6f}")
    print(f"Approximate two-sided permutation p-value = {p_value:.12f}")

    # -------------------------------------------------------------------------
    # STEP 4: Plot the null distribution and mark the observed statistic.
    # -------------------------------------------------------------------------
    plt.figure()
    plt.hist(null_diffs, bins=40, alpha=0.8)
    plt.axvline(observed, linestyle="--", linewidth=2, label="Observed difference")
    plt.axvline(-observed, linestyle="--", linewidth=2)
    plt.xlabel("Difference in mean Fare under shuffled labels")
    plt.ylabel("Count")
    plt.title("Permutation null distribution for Titanic mean Fare difference")
    plt.legend()
    plt.show()

    # -------------------------------------------------------------------------
    # STEP 5: Plain-language interpretation.
    # -------------------------------------------------------------------------
    print("\nInterpretation:")
    print("The permutation test asks whether a mean Fare difference this large")
    print("would be common if survival labels were exchangeable.")
    print("A very small p-value means the observed Fare difference would be")
    print("hard to explain by random label shuffling alone.")


def session3_project_titanic_summary():
    section("SESSION 3 - PROJECT: Titanic Part B summary in plain language")

    print("Plain-language summary:")
    print("1) The Survived vs Sex contingency table shows a very strong relationship.")
    print("   The observed counts are far from the counts we would expect if")
    print("   survival and sex were unrelated.")
    print("2) The permutation test on Fare also shows evidence of a real difference:")
    print("   survivors and non-survivors do not appear to have the same mean Fare.")
    print("3) Among the relationships examined here, Sex vs Survived appears to be")
    print("   the strongest and clearest pattern in the Titanic data.")
    print("4) The evidence comes from both the large chi-square result for categorical")
    print("   association and the very small permutation p-value for the numeric")
    print("   comparison of Fare.")


# =============================================================================
# SESSION 4 -- Modeling and Bayesian Methods
# =============================================================================

# -----------------------------------------------------------------------------
# Session 4, Exercise 1
# Linear regression via the normal equation
# -----------------------------------------------------------------------------
def session4_ex1_linear_regression():
    section("SESSION 4 - EXERCISE 1: Linear regression via the normal equation")

    rng = np.random.default_rng(5)

    # Build synthetic linear data
    x = np.linspace(-5, 5, 25).reshape(-1, 1)
    true_slope = 1.8
    true_intercept = -0.7
    noise = rng.normal(0, 1.0, size=(25, 1))
    y = true_slope * x + true_intercept + noise

    # Design matrix A:
    # first column  = x values
    # second column = ones for the intercept
    A = np.hstack([x, np.ones_like(x)])

    # Normal equation:
    # beta_hat = (A^T A)^(-1) A^T y
    beta_hat = np.linalg.inv(A.T @ A) @ (A.T @ y)

    slope_hat = beta_hat[0, 0]
    intercept_hat = beta_hat[1, 0]

    print("Normal equation:")
    print("beta_hat = (A^T A)^(-1) A^T y")
    print(f"Estimated slope     = {slope_hat:.6f}")
    print(f"Estimated intercept = {intercept_hat:.6f}")
    print(f"True slope          = {true_slope:.6f}")
    print(f"True intercept      = {true_intercept:.6f}")

    x_line = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
    y_line = slope_hat * x_line + intercept_hat

    plt.figure()
    plt.scatter(x, y, label="Data")
    plt.plot(x_line, y_line, label="Best-fit line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear regression via the normal equation")
    plt.legend()
    plt.show()

    return x, y, A, beta_hat


# -----------------------------------------------------------------------------
# Session 4, Exercise 2
# Residual analysis
# -----------------------------------------------------------------------------
def session4_ex2_residuals(x, y, A, beta_hat):
    section("SESSION 4 - EXERCISE 2: Residual analysis")

    y_hat = A @ beta_hat
    residuals = y - y_hat

    print(f"Residual mean = {residuals.mean():.6f}")
    print(f"Residual std  = {residuals.std(ddof=1):.6f}")
    print(f"Residual norm = {np.linalg.norm(residuals):.6f}")

    print("\nInterpretation:")
    print("Residuals are the parts of y that the line did NOT explain.")
    print("If the model is reasonable, the residuals should usually look like")
    print("noise rather than a strong pattern.")

    plt.figure()
    plt.scatter(x, residuals)
    plt.axhline(0, linestyle="--")
    plt.xlabel("x")
    plt.ylabel("Residual")
    plt.title("Residuals vs x")
    plt.show()

    plt.figure()
    plt.hist(residuals, bins=10)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residual histogram")
    plt.show()


# -----------------------------------------------------------------------------
# Session 4, Exercise 3
# Logistic curve
# -----------------------------------------------------------------------------
def sigmoid(z):
    """Standard logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-z))


def session4_ex3_logistic_curve():
    section("SESSION 4 - EXERCISE 3: Logistic curve and interpretation")

    z = np.linspace(-8, 8, 400)

    plt.figure()
    plt.plot(z, sigmoid(z), label="sigmoid(z)")
    plt.plot(z, sigmoid(2 * z), label="sigmoid(2z)")
    plt.plot(z, sigmoid(0.5 * z), label="sigmoid(0.5z)")
    plt.xlabel("z")
    plt.ylabel("Output")
    plt.title("Logistic / sigmoid curve")
    plt.legend()
    plt.show()

    print("Key ideas:")
    print("1) The sigmoid maps any real number to a value between 0 and 1.")
    print("2) That is why it is useful for probability-style predictions.")
    print("3) A larger slope in z makes the transition from 0 to 1 steeper.")


# -----------------------------------------------------------------------------
# Session 4 Project
# Logistic regression on Titanic WITHOUT sklearn
# -----------------------------------------------------------------------------
def standardize_columns(X):
    """
    Standardize each feature column so gradient descent behaves better.

    Returns:
        X_scaled, means, stds
    """
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=0)
    stds[stds == 0] = 1.0
    X_scaled = (X - means) / stds
    return X_scaled, means, stds


def add_intercept_column(X):
    """Add a leading column of ones for the intercept term."""
    return np.column_stack([np.ones(X.shape[0]), X])


def logistic_loss_and_gradient(X, y, beta):
    """
    Compute average logistic loss and its gradient.

    Model:
        p_i = sigmoid(X_i beta)

    Loss:
        -(1/n) sum [ y log p + (1-y) log(1-p) ]
    """
    z = X @ beta
    p = sigmoid(z)

    # Numerical safety to avoid log(0)
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)

    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    grad = (X.T @ (p - y)) / len(y)
    return loss, grad


def fit_logistic_regression_gradient_descent(X, y, learning_rate=0.1, n_iter=5000):
    """
    Fit logistic regression from scratch using gradient descent.

    We use gradient descent because the assignment asked for logistic regression
    without sklearn. This lets Lexie see the mechanics directly.
    """
    beta = np.zeros(X.shape[1])
    losses = []

    for i in range(n_iter):
        loss, grad = logistic_loss_and_gradient(X, y, beta)
        beta = beta - learning_rate * grad
        losses.append(loss)

        # Print occasional progress so the learner can see optimization happening
        if i % 1000 == 0:
            print(f"Iteration {i:4d} | loss = {loss:.6f}")

    # Final loss
    final_loss, _ = logistic_loss_and_gradient(X, y, beta)
    print(f"Final loss after {n_iter} iterations = {final_loss:.6f}")
    return beta, np.array(losses)


def confusion_matrix_manual(y_true, y_pred):
    """
    Compute a simple 2x2 confusion matrix manually.

    Returns:
        tn, fp, fn, tp
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    return tn, fp, fn, tp


def session4_project_titanic_logistic():
    section("SESSION 4 - PROJECT: Logistic regression on Titanic WITHOUT sklearn")

    df = load_titanic_dataframe().copy()

    # -------------------------------------------------------------------------
    # STEP 1: Choose a small set of features that are easy to explain.
    # -------------------------------------------------------------------------
    # We use:
    #   Pclass   -> passenger class (1,2,3)
    #   Sex      -> female/male converted to 1/0
    #   Age      -> fill missing values with median
    #   Fare     -> ticket fare
    #
    # This is not the fanciest model. It is a TEACHING model.
    print("Step 1: Choose explainable features from the Titanic table.")

    df["Sex_Female"] = (df["Sex"] == "female").astype(int)

    # Age has missing values in Titanic, so we fill them with the median age.
    age_median = df["Age"].median()
    df["Age_filled"] = df["Age"].fillna(age_median)

    # Fare usually has very few missing values, but fill anyway for safety.
    fare_median = df["Fare"].median()
    df["Fare_filled"] = df["Fare"].fillna(fare_median)

    feature_cols = ["Pclass", "Sex_Female", "Age_filled", "Fare_filled"]
    target_col = "Survived"

    X_raw = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)

    print("\nPreview of the modeling table:")
    print(df[feature_cols + [target_col]].head(10))

    # -------------------------------------------------------------------------
    # STEP 2: Standardize numeric features.
    # -------------------------------------------------------------------------
    # Logistic regression often trains more smoothly when features are on similar scales.
    print("\nStep 2: Standardize features so gradient descent behaves better.")
    X_scaled, means, stds = standardize_columns(X_raw)
    X = add_intercept_column(X_scaled)

    print("Feature means used for standardization:")
    for name, m in zip(feature_cols, means):
        print(f"  {name:12s}: {m:.6f}")

    print("Feature std devs used for standardization:")
    for name, s in zip(feature_cols, stds):
        print(f"  {name:12s}: {s:.6f}")

    # -------------------------------------------------------------------------
    # STEP 3: Fit logistic regression by gradient descent.
    # -------------------------------------------------------------------------
    print("\nStep 3: Fit logistic regression from scratch.")
    beta, losses = fit_logistic_regression_gradient_descent(
        X, y, learning_rate=0.1, n_iter=5000
    )

    print("\nEstimated coefficients (including intercept):")
    print(f"  Intercept   : {beta[0]: .6f}")
    for name, b in zip(feature_cols, beta[1:]):
        print(f"  {name:12s}: {b: .6f}")

    print("\nSign interpretation:")
    print("A positive coefficient means that as that standardized feature increases,")
    print("the log-odds of survival increase, holding the other features fixed.")
    print("A negative coefficient means the log-odds of survival decrease.")

    # -------------------------------------------------------------------------
    # STEP 4: Convert scores to probabilities and classify.
    # -------------------------------------------------------------------------
    print("\nStep 4: Turn linear scores into survival probabilities.")
    probabilities = sigmoid(X @ beta)
    predictions = (probabilities >= 0.5).astype(int)

    accuracy = np.mean(predictions == y)
    tn, fp, fn, tp = confusion_matrix_manual(y, predictions)

    print(f"Training-set accuracy (for teaching only) = {accuracy:.4f}")
    print("Confusion matrix at threshold 0.5:")
    print(f"  TN = {tn}")
    print(f"  FP = {fp}")
    print(f"  FN = {fn}")
    print(f"  TP = {tp}")

    # -------------------------------------------------------------------------
    # STEP 5: Show a few example predicted probabilities.
    # -------------------------------------------------------------------------
    print("\nStep 5: Show example predicted probabilities.")
    preview = df[["Name", "Survived"]].copy()
    preview["Predicted_Prob_Survival"] = probabilities
    preview["Predicted_Class"] = predictions
    print(preview.head(10))

    # -------------------------------------------------------------------------
    # STEP 6: Plot loss curve.
    # -------------------------------------------------------------------------
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Logistic loss")
    plt.title("Gradient descent learning curve for Titanic logistic regression")
    plt.show()

    # -------------------------------------------------------------------------
    # STEP 7: Teaching summary
    # -------------------------------------------------------------------------
    print("\nTeaching summary:")
    print("1) Logistic regression does NOT predict only 0 or 1 at first.")
    print("   It predicts a probability between 0 and 1.")
    print("2) The sigmoid converts a linear score into a probability.")
    print("3) Gradient descent adjusts the coefficients to reduce the logistic loss.")
    print("4) We then choose a threshold, here 0.5, to convert probabilities to")
    print("   class predictions.")


# =============================================================================
# Main runner
# =============================================================================
def main():
    # ------------------------------
    # Session 3
    # ------------------------------
    session3_ex1_bootstrap()
    session3_ex2_permutation()
    session3_ex3_chisquare()
    session3_project_titanic_chisquare()
    session3_project_titanic_chisquare_2()
    session3_project_titanic_permutation_numeric()
    session3_project_titanic_summary()

    # ------------------------------
    # Session 4
    # ------------------------------
    x, y, A, beta_hat = session4_ex1_linear_regression()
    session4_ex2_residuals(x, y, A, beta_hat)
    session4_ex3_logistic_curve()
    session4_project_titanic_logistic()

    section("FINAL NOTE FOR LEXIE")
    print("The most important ideas to practice again are:")
    print("1) DataFrame assignment, for example:")
    print("      df['NewColumn'] = some expression")
    print("2) Crosstabs for categorical counts:")
    print("      pd.crosstab(df['row_variable'], df['col_variable'])")
    print("3) Expected counts in a contingency table:")
    print("      E = (row total * column total) / grand total")
    print("4) Chi-square contributions in each cell:")
    print("      (O - E)^2 / E")
    print("5) Logistic regression workflow:")
    print("      features -> linear score -> sigmoid -> probability -> class decision")


if __name__ == "__main__":
    main()
