# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 12:13:29 2026

@author: annil
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency

    
# ------------------------------------
# Part 0 -- Foundations
# ------------------------------------

# A0.1 Normal dsitribution basics

def gaussian_pdf(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def plot_analytic_gaussian_pdf():
    mu = 2.0
    sigma= 1.5
    sample = np.random.normal(loc=mu, scale=sigma, size=5000)
    
    x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 500)
    pdf = gaussian_pdf(x, mu=mu, sigma=sigma)
    
    plt.figure()
    plt.hist(sample, bins=50, density=True, alpha=0.6, label="sample histogram")
    plt.plot(x, pdf, linewidth=2.5, label="analytic Gaussian pdf")
    plt.axvline(mu, linestyle="--", label="true mean mu")
    plt.axvline(mu - sigma, linestyle="--")
    plt.axvline(mu + sigma, linestyle="--")
    plt.title("Histogram of samples with analytic Gaussian pdf overlay")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend()
    plt.show()
    

"""
Mu determines where the center of the curve is because it is the mean. Since everything averages to mu and the curve is symmetrical, mu is in the middle. The value of sigma determines how spread out the curve is. 
About 2/3 of the mass appears to lie roughly within one standard deviation.
"""


# A0.2 Mean, variance, and standard deviation

def mean_variance_and_standard_deviation_of_sample():

    x = np.random.normal(size=100)

    print(f"Mean of sample               = {x.mean():.4f}")
    print(f"Variance of sample           = {x.var():.4f}")
    print(f"Standard deviation of sample = {np.sqrt(x.var()):.4f}")

"""
Variance is the average distance of each point from the mean and standard deviation is the square root of variance and it measures how far apart the data is spread.
"""

def effect_of_change_in_standard_deviation():

    N1 = np.random.normal(loc=0, scale=1, size=5000)
    N2 = np.random.normal(loc=0, scale=3, size=5000)

    plt.figure()
    plt.hist(N1, bins=50, density=True, label="N(0,1)", alpha=0.6)
    plt.hist(N2, bins=50, density=True, label="N(0,3)", alpha=0.6)
    plt.title("Plots of normal distributions with different standard deviaitons")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend()
    plt.show
    
def effect_of_change_in_mean():

    N1 = np.random.normal(loc=0, scale=1, size=5000)
    N2 = np.random.normal(loc=3, scale=1, size=5000)

    plt.figure()
    plt.hist(N1, bins=50, density=True, label="N(0,1)", alpha=0.6)
    plt.hist(N2, bins=50, density=True, label="N(3,1)", alpha=0.6)
    plt.title("Plots of normal distributions with different standard deviaitons")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend()
    plt.show
   
"""
Changing location shifts the graph on the x-axis, but keeps the shape of the graph the same. Changing the spread of the graph affects the height of the graph and distance of points from the center.
"""


# A0.3 Law of large numbers and the sample mean

def law_of_large_numbers_and_sample_mean():
    
    mu = 2.0
    sigma = 1.5
    sample = np.random.normal(loc=mu, scale=sigma, size=2000)
    
    running_mean = np.cumsum(sample) / np.arange(1, len(sample) + 1)
    
    plt.figure()
    plt.plot(running_mean, label="running sample mean")
    plt.axhline(mu, linestyle="--", label="true mean mu")
    plt.title("Law of large numbers")
    plt.xlabel("number of samples N")
    plt.ylabel("running mean")
    plt.legend()
    plt.show()
    
"""
The running mean fluctuates less as the number of samples increases and tends towards mu.
"""
  
# A0.4 Variance of the sample mean scales like 1/N

def variance_of_sample_means():
    
    mu = 0
    sigma = 1
    n_values = np.array([5, 10, 20, 50, 100, 200])
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
    
    """
    To be honest, I know I didn't really do this part correctly and and I'm kind of confused what is even going on with this part.
    """
        
def foundations():
    plot_analytic_gaussian_pdf() 
    mean_variance_and_standard_deviation_of_sample()
    effect_of_change_in_standard_deviation()
    effect_of_change_in_mean()
    law_of_large_numbers_and_sample_mean()
    variance_of_sample_means()
    
    
# ------------------------------------
# Part A -- Warm-up Exercises
# ------------------------------------

# A1. Simulating two populations

def simulating_two_populations():
    population_A = np.random.normal(loc=0, scale=1, size=1000)
    population_B = np.random.normal(loc=0.5, scale=1, size=1000)
    
    plt.figure()
    plt.hist(population_A, bins=10, density=True, alpha=0.6, label="Population A")
    plt.hist(population_B, bins=10, density=True, alpha=0.6, label="Population B")
    plt.title("Histogram of populations A and B")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend()
    plt.show()
    
    n = 40
    reps = 500
    diffs = []
    
    sample_a = np.random.choice(population_A, size=n, replace=False)
    sample_b = np.random.choice(population_B, size=n, replace=False)
    print(f"Mean of sample of A: {sample_a.mean():.4f}")
    print(f"Mean of sample of B: {sample_b.mean():.4f}")
    print(f"Difference of sample means: {(sample_a.mean() - sample_b.mean()):.4f}")

    for _ in range(reps):
        a = np.random.choice(population_A, size=n, replace=False)
        b = np.random.choice(population_B, size=n, replace=False)
        diffs.append(b.mean() - a.mean())

    diffs = np.array(diffs)

    plt.figure()
    plt.hist(diffs, bins=30)
    plt.title("Sampling distribution of mean differences")
    plt.xlabel("sample mean(B) - sample mean(A)")
    plt.ylabel("count")
    plt.show()
    
    
# A2. Bootstrap confidence interval

def bootstrap_means(sample, n_boot=2000):
        means = []
        n = len(sample)
        for _ in range(n_boot):
            boot = np.random.choice(sample, size=n, replace=True)
            means.append(np.mean(boot))
        return np.array(means)
    
def bootstrap_confidence_interval():
    mu = 2.0
    sigma = 1.5
    sample = np.random.normal(loc=mu, scale=sigma, size=40)

    boot_means = bootstrap_means(sample, n_boot=2000)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    plt.figure()
    plt.hist(boot_means, bins=30)
    plt.axvline(ci_low, linestyle="--", label="2.5th percentile")
    plt.axvline(ci_high, linestyle="--", label="97.5th percentile")
    plt.title("Bootstrap distribution of the sample mean")
    plt.xlabel("bootstrap mean")
    plt.ylabel("count")
    plt.legend()
    plt.show()
    
    print(f"Approximate 95% bootstrap CI = [{ci_low:.4f}, {ci_high:.4f}]")
    
"""
Bootstrapping makes more sense now that we have already studied the sampling distribution of the mean because we're basically just doing the same thing, but varring the means by using different samples obtained from the same sample.
"""

# A3. Permutation test for difference in means    

def permutation_test():
    a = np.random.normal(loc=0.0, scale=1.0, size=40)
    b = np.random.normal(loc=0.5, scale=1.0, size=40)
    
    observed = np.mean(a) - np.mean(b)
    n_a = len(a)
    n_perm = 5000
    null_diffs = []
    
    for _ in range(n_perm):
        shuffled = np.random.permutation(np.concatenate([a, b]))
        new_a = shuffled[:n_a]
        new_b = shuffled[n_a:]
        null_diffs.append(np.mean(new_a) - np.mean(new_b))
        
    null_diffs = np.array(null_diffs)
    p_value = np.mean(np.abs(null_diffs) >= abs(observed))
        
    print(f"Observed difference in means = {observed:.4f}")
    print(f"Approximate permutation p-value = {p_value:.4f}")

    plt.figure()
    plt.hist(null_diffs, bins=50)
    plt.axvline(observed, linestyle="--", label="observed diff")
    plt.axvline(-observed, linestyle="--")
    plt.title("Permutation null distribution")
    plt.xlabel("difference in means under shuffled labels")
    plt.ylabel("count")
    plt.legend()
    plt.show()
        
# A4. Chi-square test for categorical data

def expected_counts(observed):
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    grand_total = observed.sum()
    return row_totals @ col_totals / grand_total

def chi_square_test():
    observed = np.array([[30, 20], [10, 40]])

    
    exp = expected_counts(observed)
    stat = np.sum((observed - exp) ** 2 / exp)
    
    print("Observed table:")
    print(observed)
    print("\nExpected counts under independence:")
    print(exp)
    print(f"\nChi-square statistic = {stat:.4f}")
    
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
    
def warm_up_exercises():
    simulating_two_populations()
    bootstrap_confidence_interval()
    permutation_test()
    chi_square_test()
    
    
# ------------------------------------
# Part B -- Titanic inference lab
# ------------------------------------

def load_titanic():
    
    return pd.read_csv("titanic.csv")

def titanic_table_survived_vs_sex():
    
    df = load_titanic()
    
    df["UpperClass"] = np.where(df["Pclass"].isin([1, 2]), 1, 0)
    contingency_df = pd.crosstab(df["Survived"], df["UpperClass"])

    contingency_df = contingency_df[[1, 0]]
    contingency_df.columns = ["UpperClass(1or2)", "LowerClass(3)"]
    contingency_df.index = ["Died (Survived=0)", "Survived (Survived=1)"]
    
    print("\n2x2 contingency table:")
    print(contingency_df)
    
    observed = contingency_df.to_numpy(dtype=float)
    
    exp = expected_counts(observed)
    
    diff = observed - exp
    squared_diff = diff ** 2
    contributions = squared_diff / exp
    
    cell_names = [
        ("Died, UpperClass", 0, 0),
        ("Died, LowerClass", 0, 1),
        ("Survived, UpperClass", 1, 0),
        ("Survived, LowerClass", 1, 1)
    ]
    
    for label, i, j in cell_names:
        print(f"\n{label}")
        print(f"  Observed O = {observed[i,j]:.4f}")
        print(f"  Expected E = {exp[i,j]:.4f}")
        print(f"  O - E      = {diff[i,j]:.4f}")
        print(f"  (O - E)^2  = {squared_diff[i,j]:.4f}")
        print(f"  (O - E)^2 / E = {contributions[i,j]:.4f}")
 
    chi_sq = contributions.sum()
    
    plt.figure()
    plt.imshow(observed)
    plt.xticks([0, 1], ["UpperClass(1or2)", "LowerClass(3)"])
    plt.yticks([0, 1], ["Died", "Survived"])
    plt.title("Observed 2x2 Titanic contingency table")
    plt.colorbar()
    plt.show()
    
    print(f"chi-square statistic: {chi_sq:.4f}")
    
"""    
The chi-square statistic is pretty large for a contingency table with 1 degree of freedom, so we reject the null hypothesis. This means that class and survival are probably related.  
"""  
    

    
foundations()
warm_up_exercises()
titanic_table_survived_vs_sex()