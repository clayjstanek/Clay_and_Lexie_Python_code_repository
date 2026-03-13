# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 08:48:06 2026

@author: cstan

Teaching_M2_Session2_Probability_Convolution.py

Purpose
-------
Teaching script for Lexie: Month 2, Session 2

Topics:
1) Optional teaser: cosine similarity on sentence vectors
2) Uniform distributions and summary statistics
3) Monte Carlo sums of centered uniforms
4) Triangle distribution from sum of two uniforms
5) Discrete convolution as the mechanism behind the sum
6) CLT intuition: repeated sums look more Gaussian

This script is designed for live teaching in Spyder.
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------
# Toggle this if you want the sentence-vector teaser at start
# ------------------------------------------------------------
RUN_TEXT_VECTOR_TEASER = False


# ------------------------------------------------------------
# Helper printing
# ------------------------------------------------------------
def section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


# ------------------------------------------------------------
# Part 0: Optional teaser — language to vectors
# ------------------------------------------------------------
def text_vector_teaser():
    section("PART 0 — Optional teaser: cosine similarity on sentences")

    sentences = [
        "The cat sat on the mat.",
        "A kitten rested on the rug.",
        "The stock market fell sharply today.",
        "Investors reacted to a sudden market decline.",
        "I love eating pizza with mushrooms.",
        "She ordered a pepperoni pizza for dinner."
    ]

    for i, s in enumerate(sentences):
        print(f"S{i}: {s}")

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(sentences).toarray()

    sim = cosine_similarity(X)

    print("\nCosine similarity matrix:")
    np.set_printoptions(precision=3, suppress=True)
    print(sim)

    plt.figure(figsize=(6, 5))
    plt.imshow(sim, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="Cosine Similarity")
    plt.xticks(range(len(sentences)), [f"S{i}" for i in range(len(sentences))], rotation=45)
    plt.yticks(range(len(sentences)), [f"S{i}" for i in range(len(sentences))])
    plt.title("Sentence Similarity Heatmap")
    plt.tight_layout()
    plt.show()

    print("\nKey takeaway:")
    print("Even language can be mapped into vectors.")
    print("Once that happens, linear algebra tools like cosine similarity still work.")


# ------------------------------------------------------------
# Part 1: Uniform(0,1) sampling
# ------------------------------------------------------------
def part1_uniform_sampling():
    section("PART 1 — Uniform(0,1) sampling")

    N = 10000
    u01 = np.array([random.random() for _ in range(N)], dtype=float)

    print(f"Number of samples: {N}")
    print(f"Sample mean     ≈ {u01.mean():.4f}")
    print(f"Sample variance ≈ {u01.var():.4f}")
    print("Expected mean   = 0.5")
    print("Expected var    = 1/12 ≈ 0.0833")

    plt.figure()
    plt.hist(u01, bins=40, density=True)
    plt.title("Uniform(0,1) via random.random()")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.show()

    return u01


# ------------------------------------------------------------
# Centered uniform sampling
# ------------------------------------------------------------
def sample_uniform_centered(n_samples: int) -> np.ndarray:
    """Samples Uniform(-0.5, 0.5) using random.random()."""
    return np.array([random.random() - 0.5 for _ in range(n_samples)], dtype=float)


def monte_carlo_sums(n_samples: int, n_terms: int) -> np.ndarray:
    """Monte Carlo sum of n_terms independent Uniform(-0.5,0.5) variables."""
    s = np.zeros(n_samples, dtype=float)
    for _ in range(n_terms):
        s += sample_uniform_centered(n_samples)
    return s


# ------------------------------------------------------------
# Part 2: CLT intuition from sums of uniforms
# ------------------------------------------------------------
def part2_monte_carlo_sums():
    section("PART 2 — Sums of centered uniforms: S2, S4, S8")

    N = 50000
    S2 = monte_carlo_sums(N, 2)
    S4 = monte_carlo_sums(N, 4)
    S8 = monte_carlo_sums(N, 8)

    print("Generated Monte Carlo samples for:")
    print("S2 = U1 + U2")
    print("S4 = U1 + U2 + U3 + U4")
    print("S8 = sum of 8 centered uniforms")

    plt.figure()
    plt.hist(S2, bins=80, density=True)
    plt.title("S2 = U1 + U2,  U ~ Uniform(-0.5,0.5)")
    plt.xlabel("s")
    plt.ylabel("density")
    plt.show()

    plt.figure()
    plt.hist(S4, bins=80, density=True)
    plt.title("S4 = U1 + U2 + U3 + U4")
    plt.xlabel("s")
    plt.ylabel("density")
    plt.show()

    plt.figure()
    plt.hist(S8, bins=80, density=True)
    plt.title("S8 = Sum of 8 Uniform(-0.5,0.5) Variables")
    plt.xlabel("s")
    plt.ylabel("density")
    plt.show()

    print("\nTeaching point:")
    print("As we add more independent centered random variables,")
    print("the distribution becomes smoother and more Gaussian-like.")

    return S2, S4, S8


# ------------------------------------------------------------
# Part 3: Analytic triangle PDF for sum of two uniforms
# ------------------------------------------------------------
def triangle_pdf(z: np.ndarray) -> np.ndarray:
    """
    Analytic PDF for S2 = U1 + U2, where Ui ~ Uniform(-0.5,0.5):
        f(z) = 1 - |z|,  for |z| <= 1
             = 0,        otherwise
    """
    out = np.zeros_like(z, dtype=float)
    mask = np.abs(z) <= 1.0
    out[mask] = 1.0 - np.abs(z[mask])
    return out


def part3_triangle_overlay(S2):
    section("PART 3 — Analytic triangle PDF for sum of two uniforms")

    z = np.linspace(-1, 1, 1000)
    tri = triangle_pdf(z)

    plt.figure()
    plt.hist(S2, bins=80, density=True, alpha=0.7, label="Monte Carlo S2")
    plt.plot(z, tri, linewidth=2, label="Analytic triangle PDF")
    plt.title("Monte Carlo S2 vs Analytic Triangle")
    plt.xlabel("z")
    plt.ylabel("density")
    plt.legend()
    plt.show()

    print("Key idea:")
    print("Uniform * Uniform  ->  Triangle")
    print("This is the first exact example of convolution at work.")


# ------------------------------------------------------------
# Part 4: Discrete convolution
# ------------------------------------------------------------
def make_discrete_uniform(dx: float = 0.002):
    """
    Create a discrete approximation of Uniform(-0.5,0.5).
    """
    x = np.arange(-0.5, 0.5 + dx, dx)
    unif = np.ones_like(x, dtype=float)

    # Normalize so integral is approximately 1
    unif = unif / (unif.sum() * dx)

    return x, unif


def convolve_pdf(f: np.ndarray, g: np.ndarray, dx: float) -> np.ndarray:
    """
    Discrete convolution approximating continuous convolution.
    Multiply by dx to preserve normalization.
    """
    return np.convolve(f, g, mode="full") * dx


def convolved_support(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Build support grid for convolution result.
    """
    return np.linspace(x[0] + y[0], x[-1] + y[-1], len(x) + len(y) - 1)


def part4_discrete_convolution(S4):
    section("PART 4 — Discrete convolution as computational theory")

    dx = 0.002
    x_u, unif = make_discrete_uniform(dx=dx)

    # uniform * uniform -> triangle
    tri_disc = convolve_pdf(unif, unif, dx)
    x_tri = convolved_support(x_u, x_u)

    # compare discrete triangle to analytic triangle
    z = np.linspace(-1, 1, 1000)
    tri_analytic = triangle_pdf(z)

    plt.figure()
    plt.plot(x_tri, tri_disc, label="Discrete convolution: unif * unif")
    plt.plot(z, tri_analytic, "--", label="Analytic triangle")
    plt.title("Triangle PDF: Discrete Convolution vs Analytic Formula")
    plt.xlabel("z")
    plt.ylabel("density")
    plt.legend()
    plt.show()

    # triangle * triangle -> 4-uniform sum
    four_disc = convolve_pdf(tri_disc, tri_disc, dx)
    x_four = convolved_support(x_tri, x_tri)

    plt.figure()
    plt.hist(S4, bins=80, density=True, alpha=0.7, label="Monte Carlo S4")
    plt.plot(x_four, four_disc, linewidth=2, label="Discrete convolution theory")
    plt.title("Monte Carlo S4 vs Discrete Convolution Theory")
    plt.xlabel("z")
    plt.ylabel("density")
    plt.legend()
    plt.show()

    print("Key idea:")
    print("Convolution is the correct operation for adding independent random variables.")
    print("uniform * uniform gives the S2 PDF")
    print("triangle * triangle gives the S4 PDF")


# ------------------------------------------------------------
# Part 5: Optional comparison plot
# ------------------------------------------------------------
def part5_compare_shapes(S2, S4, S8):
    section("PART 5 — Shape comparison: S2, S4, S8")

    plt.figure()
    plt.hist(S2, bins=80, density=True, alpha=0.4, label="S2")
    plt.hist(S4, bins=80, density=True, alpha=0.4, label="S4")
    plt.hist(S8, bins=80, density=True, alpha=0.4, label="S8")
    plt.title("How the Shape Changes as We Add More Uniforms")
    plt.xlabel("z")
    plt.ylabel("density")
    plt.legend()
    plt.show()

    print("Teaching point:")
    print("The CLT does not say the result becomes exactly Gaussian immediately.")
    print("It says the normalized sum approaches a Gaussian as the number of terms grows.")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    random.seed(42)
    np.random.seed(42)

    if RUN_TEXT_VECTOR_TEASER:
        text_vector_teaser()

    part1_uniform_sampling()
    S2, S4, S8 = part2_monte_carlo_sums()
    part3_triangle_overlay(S2)
    part4_discrete_convolution(S4)
    part5_compare_shapes(S2, S4, S8)

    section("FINAL SUMMARY")
    print("1) Histograms let us see empirical distributions from simulation.")
    print("2) Summing two centered uniforms gives a triangle distribution.")
    print("3) Convolution is the mechanism behind addition of independent random variables.")
    print("4) Repeated sums tend toward Gaussian-like shapes (CLT intuition).")
    print("5) These same ideas show up in science, engineering, and data science.")


if __name__ == "__main__":
    main()