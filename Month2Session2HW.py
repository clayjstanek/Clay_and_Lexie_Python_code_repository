# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 18:52:00 2026

@author: annil
"""

import random
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# Part A - Exercises
# --------------------------------------------------

# A1. Sampling and summary statistics
random_samples = np.array([random.random() for _ in range(10000)], dtype=float)
print(f"Sample mean: {random_samples.mean():.4f}")
print(f"Sample variance: {random_samples.var():.4f}")

plt.figure()
plt.hist(random_samples, bins=40, density=True)
plt.title("Uniform(0,1) via random.random()")
plt.xlabel("x")
plt.ylabel("density")
plt.show()


# A2. From uniform to 'more Gaussian': a CLT experiment
def U(n_samples: int) -> np.array:
    return np.array([random.random() - 0.5 for _ in range(n_samples)], dtype=float)

N = 50000
S2 = np.zeros(N, dtype=float)
for _ in range(2):
    S2 += U(N)
plt.figure()
plt.hist(S2, bins=80, density=True)
plt.title("S2 = U1 + U2")
plt.xlabel("s")
plt.ylabel("density")
plt.show()

S4 = np.zeros(N, dtype=float)
for _ in range(4):
    S4 += U(N)
plt.figure()
plt.hist(S4, bins=80, density=True)
plt.title("S4 = U1 + U2 + U3 + U8")
plt.xlabel("s")
plt.ylabel("density")
plt.show

"""
The plot of S2 and S4 are kind of similar since the values in the middle are much more likely that the ones on the outsides. However, the graph of S2 is a triangle with a sharp point and flat sides while S4 is more curved. This is because as we sum more independent uniforms, the middle becomes much more likely.
"""


# A3. Theoretical check: triangle PDF for the sum of two uniforms
def triangle_pdf(z: np.ndarray) -> np.ndarray:
    out = np.zeros_like(z, dtype=float)
    mask = np.abs(z) <= 1.0
    out[mask] = 1.0 - np.abs(z[mask])
    return out

z = np.linspace(-1, 1, 1000)
tri = triangle_pdf(z)

plt.figure()
plt.hist(S2, bins=80, density=True, label="Monte Carlo S2")
plt.plot(z, tri, linewidth=2, label="Analytic triangle PDF")
plt.title("Monte Carlo S2 vs Analytic Triangle")
plt.xlabel("z")
plt.ylabel("density")
plt.legend()
plt.show()


# A4. A discrete convolution warm-up (numerical, not calculus)
def make_discrete_uniform(dx: float = 0.002):
    x = np.arrange(-0.5, 0.5 + dx, dx)
    unif = np.ones_like(x, dtype=float)
    unif = unif / (unif.sum() * dx)
    return x, unif

dx = 0.002 
x_u, unif = make_discrete_uniform(dx=dx)

tri_disc = np.convolve(unif, unif, mode="full") * dx
x_tri = np.linspace(x_u[0] + x_u[0], x_u[-1] + x_u[-1], len(x_u) + len(x_u) - 1)

z2 = np.linspace(-1, 1, 1000)
tri_analytic = triangle_pdf(z2)

plt.figure()
plt.plot(x_tri, tri_disc, label="Discrete conv (unif * unif)")
plt.plot(z2, tri_analytic, "--", label="Analytic triangle")
plt.title("A4: Triangle PDF via discrete convolution vs analytic formula")
plt.xlabel("z")
plt.ylabel("density")
plt.legend()
plt.show()


# ----------------------------------------------------------------------------
# Part B - Mini Project: Convolution as the Mechanism Behind the CLT
# ----------------------------------------------------------------------------
def convolve_pdf(f: np.ndarray, g: np.ndarray, dx: float) -> np.ndarray:
    return np.convolve(f, g, mode='full') * dx

def convolved_support(x: np.ndarray, y: np.ndarray, dx: float) -> np.ndarray:
    return np.linspace(x[0] + y[0], x[-1] + y[-1], len(x) + len(y) - 1)

def plot_hist_with_curve(samples: np.ndarray, curve_x: np.ndarray, curve_y: np.ndarray, title: str):
    plt.figure()
    plt.hist(samples, bins=80, density=True, alpha=0.7, label="Monte Carlo (density=True)")
    plt.plot(curve_x, curve_y, linewidth=2.0, label="Theory")
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density / Probability")
    plt.legend()
    plt.show()


four_disc = convolve_pdf(tri_disc, tri_disc, dx)
x_four = convolved_support(x_tri, x_tri, dx)

plot_hist_with_curve(S4, x_four, four_disc, "Part B: Monte Carlo S4 vs discrete convolution theory")

N2 = 100000
S8 = np.zeros(N2, dtype=float)
for _ in range(8):
    S8 += U(N)
eight_disc = convolve_pdf(four_disc, four_disc, dx)
x_eight = convolved_support(x_four, x_four, dx)
plot_hist_with_curve(S8, x_eight, eight_disc, "Monte Carlo S8 vs discrete convolution theory")

# Reflection questions
"""
Convolution is the correct operation for adding independent random variables because it calculates the probability of all possible pairs that sum to a specific value.

The S4 distribution looks "more Gaussian" than S2 because more independent variables means that the extremes become more rare and the middle values are more common, and each convolution smooths the graph more and more.

The graph has less noise when N is increased becuase of the law of large numbers. The graph is smoother and less blocky when dx is decreased.

This idea might matter in chemistry for trying to represent electron orbitals in molecules. 
"""