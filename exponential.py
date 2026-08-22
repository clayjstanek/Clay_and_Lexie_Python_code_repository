# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 21:50:32 2026

@author: cstan
"""

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Generate exponential observations
# ------------------------------------------------------------

rng = np.random.default_rng(2026)

true_lambda = 0.50
n = 80

# NumPy uses the scale parameter, which is 1/lambda.
observations = rng.exponential(
    scale=1.0 / true_lambda,
    size=n
)


# ------------------------------------------------------------
# 2. Calculate the maximum-likelihood estimate
#
# lambda_MLE = n / sum(x_i) = 1 / sample mean
# ------------------------------------------------------------

sample_mean = np.mean(observations)
lambda_mle = n / np.sum(observations)


# ------------------------------------------------------------
# 3. Evaluate the likelihood over candidate lambda values
# ------------------------------------------------------------

lambda_grid = np.linspace(0.10, 1.10, 1000)

# For independent exponential observations:
#
# log L(lambda)
#     = n log(lambda) - lambda sum(x_i)
#
log_likelihood = (
    n * np.log(lambda_grid)
    - lambda_grid * np.sum(observations)
)

# The raw likelihood may contain extremely small numbers.
# Normalize it so that its maximum is 1.
relative_likelihood = np.exp(
    log_likelihood - np.max(log_likelihood)
)


# ------------------------------------------------------------
# 4. Plot the likelihood curve
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(
    lambda_grid,
    relative_likelihood,
    linewidth=2.5,
    label="Relative likelihood"
)

plt.axvline(
    true_lambda,
    linestyle="--",
    linewidth=2,
    label=rf"True $\lambda={true_lambda:.3f}$"
)

plt.axvline(
    lambda_mle,
    linestyle="-",
    linewidth=2,
    label=rf"MLE $\hat{{\lambda}}={lambda_mle:.3f}$"
)

plt.scatter(
    lambda_mle,
    1.0,
    s=80,
    zorder=3
)

plt.xlabel(r"Candidate rate parameter, $\lambda$")
plt.ylabel("Relative likelihood")

plt.title(
    "Maximum-Likelihood Estimate of an Exponential Rate\n"
    f"{n} Simulated Observations"
)

plt.ylim(bottom=0)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(
    "exponential_lambda_mle.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 5. Print the numerical results
# ------------------------------------------------------------

print(f"Number of observations: {n}")
print(f"True lambda:            {true_lambda:.6f}")
print(f"True mean:              {1 / true_lambda:.6f}")
print(f"Sample mean:            {sample_mean:.6f}")
print(f"MLE lambda:             {lambda_mle:.6f}")
print(f"Estimated mean:         {1 / lambda_mle:.6f}")