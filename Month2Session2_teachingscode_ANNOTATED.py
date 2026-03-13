# -*- coding: utf-8 -*-
"""
Month2Session2_teachingscode_ANNOTATED.py

Created for Lexie as a heavily documented teaching companion.

PURPOSE
-------
This file teaches several core ideas from probability, statistics, and linear algebra:

1) OPTIONAL teaser: language can be converted into vectors, and cosine similarity
   can then be used to compare meaning geometrically.
2) Uniform random sampling and empirical distributions.
3) Centered uniform random variables and why centering matters.
4) Monte Carlo simulation of sums of random variables.
5) The exact PDF for the sum of two centered uniforms: a triangle.
6) Convolution as the correct operation for adding independent random variables.
7) Central Limit Theorem (CLT) intuition: repeated sums start to look Gaussian.

This version is intentionally verbose. The goal is that Lexie can re-read the file
later and understand not only WHAT each line does, but WHY it is there.

RUNNING THE FILE
----------------
Run from Spyder or any Python environment with:
    - numpy
    - matplotlib
    - scikit-learn

OPTIONAL PACKAGE:
    - sentence-transformers (only needed if the optional modern text embedding demo
      is turned on)

KEY BIG IDEAS
-------------
A) Histograms let us approximate unknown distributions from simulated samples.
B) If X and Y are independent random variables, the PDF of X+Y is the convolution
   of the PDFs of X and Y.
C) The sum of two centered Uniform(-0.5, 0.5) random variables has a triangular PDF.
D) Repeated sums of centered, independent random variables tend toward Gaussian-like
   shapes (Central Limit Theorem intuition).
"""

# Standard library imports
import random

# Third-party scientific computing imports
import numpy as np
import matplotlib.pyplot as plt

# Optional text-vector teaser imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------------------------------------------------
# TOGGLES
# -----------------------------------------------------------------------------
# If True, we begin with a short teaser showing that even language can be mapped
# into vector space, so cosine similarity still applies.
RUN_TEXT_VECTOR_TEASER = True

# If True, we run a tiny "sliding window" toy convolution example that prints
# the elementwise multiply-and-sum process step by step.
RUN_SLIDING_WINDOW_DEMO = True


# -----------------------------------------------------------------------------
# HELPER PRINTING
# -----------------------------------------------------------------------------
def section(title: str) -> None:
    """
    Print a visual section divider in the console so the lesson is easy to follow.
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# -----------------------------------------------------------------------------
# PART 0: OPTIONAL TEASER — LANGUAGE TO VECTORS
# -----------------------------------------------------------------------------
def text_vector_teaser() -> None:
    """
    Show that sentences can be converted into vectors using TF-IDF, and then
    cosine similarity can be used to compare them.

    IMPORTANT TEACHING NOTE:
    ------------------------
    We are NOT teaching natural language processing in depth here.
    The point is simply to show that once a non-numeric object (like a sentence)
    is mapped into a vector space, linear algebra still applies.
    """
    section("PART 0 — Optional teaser: cosine similarity on sentences")

    # A small set of related and unrelated sentences.
    # Some are about animals, some about finance, some about food.
    sentences = [
        "The cat sat on the mat.",
        "A kitten rested on the rug.",
        "The stock market fell sharply today.",
        "Investors reacted to a sudden market decline.",
        "I love eating pizza with mushrooms.",
        "She ordered a pepperoni pizza for dinner."
    ]

    # Print the sentence list with short IDs so the matrix is easier to read.
    for i, s in enumerate(sentences):
        print(f"S{i}: {s}")

    # TfidfVectorizer turns text into a matrix of numbers.
    # Roughly speaking:
    #   - each ROW is one sentence
    #   - each COLUMN is one important vocabulary term
    #   - entries tell us how important a word is in a sentence
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(sentences).toarray()

    # Cosine similarity compares the angle between two vectors.
    # If two sentence vectors point in similar directions, their cosine similarity
    # is close to 1.
    sim = cosine_similarity(X)

    print("\nCosine similarity matrix:")
    np.set_printoptions(precision=3, suppress=True)
    print(sim)

    # Plot the similarity matrix as a heatmap.
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


# -----------------------------------------------------------------------------
# PART 1: UNIFORM(0,1) SAMPLING
# -----------------------------------------------------------------------------
def part1_uniform_sampling() -> np.ndarray:
    """
    Generate many samples from Uniform(0,1) using random.random() and show:
      - empirical mean
      - empirical variance
      - histogram

    Returns
    -------
    u01 : np.ndarray
        Array of sampled values in [0,1).
    """
    section("PART 1 — Uniform(0,1) sampling")

    # Number of Monte Carlo samples.
    N = 10000

    # random.random() returns one float in [0, 1).
    # The list comprehension repeats that N times, and np.array converts the list
    # into a NumPy array for convenient numerical work.
    u01 = np.array([random.random() for _ in range(N)], dtype=float)

    # Compare empirical estimates to known theory for Uniform(0,1):
    # mean = 0.5
    # variance = 1/12
    print(f"Number of samples: {N}")
    print(f"Sample mean     ≈ {u01.mean():.4f}")
    print(f"Sample variance ≈ {u01.var():.4f}")
    print("Expected mean   = 0.5")
    print("Expected var    = 1/12 ≈ 0.0833")

    # density=True makes the histogram scale as a probability density, not raw count.
    plt.figure()
    plt.hist(u01, bins=40, density=True)
    plt.title("Uniform(0,1) via random.random()")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.show()

    return u01


# -----------------------------------------------------------------------------
# CENTERED UNIFORM SAMPLING
# -----------------------------------------------------------------------------
def sample_uniform_centered(n_samples: int) -> np.ndarray:
    """
    Sample Uniform(-0.5, 0.5) using random.random().

    WHY CENTER AT ZERO?
    -------------------
    Centering makes the mean equal to zero. That way, when we add many such
    variables, the resulting sums stay centered, which makes the symmetry easier
    to see and supports the CLT intuition more clearly.
    """
    return np.array([random.random() - 0.5 for _ in range(n_samples)], dtype=float)


def monte_carlo_sums(n_samples: int, n_terms: int) -> np.ndarray:
    """
    Return the Monte Carlo sum of 'n_terms' independent Uniform(-0.5,0.5) variables.

    Example:
        n_terms = 2 gives S2 = U1 + U2
        n_terms = 4 gives S4 = U1 + U2 + U3 + U4
    """
    # Start with all-zero sums, one for each Monte Carlo trial.
    s = np.zeros(n_samples, dtype=float)

    # Each iteration adds one more independent centered-uniform draw to every trial.
    for _ in range(n_terms):
        s += sample_uniform_centered(n_samples)

    return s


# -----------------------------------------------------------------------------
# PART 2: CLT INTUITION FROM SUMS OF UNIFORMS
# -----------------------------------------------------------------------------
def part2_monte_carlo_sums():
    """
    Generate Monte Carlo samples for S2, S4, and S8 and plot histograms.

    Returns
    -------
    S2, S4, S8 : np.ndarray
        Arrays containing the simulated sums.
    """
    section("PART 2 — Sums of centered uniforms: S2, S4, S8")

    N = 50000

    # S2 is the sum of 2 centered uniforms.
    S2 = monte_carlo_sums(N, 2)

    # S4 is the sum of 4 centered uniforms.
    S4 = monte_carlo_sums(N, 4)

    # S8 is the sum of 8 centered uniforms.
    S8 = monte_carlo_sums(N, 8)

    print("Generated Monte Carlo samples for:")
    print("S2 = U1 + U2")
    print("S4 = U1 + U2 + U3 + U4")
    print("S8 = sum of 8 centered uniforms")

    # Histogram for S2: should look triangular.
    plt.figure()
    plt.hist(S2, bins=80, density=True)
    plt.title("S2 = U1 + U2,  U ~ Uniform(-0.5,0.5)")
    plt.xlabel("s")
    plt.ylabel("density")
    plt.show()

    # Histogram for S4: smoother and more bell-like.
    plt.figure()
    plt.hist(S4, bins=80, density=True)
    plt.title("S4 = U1 + U2 + U3 + U4")
    plt.xlabel("s")
    plt.ylabel("density")
    plt.show()

    # Histogram for S8: even more Gaussian-looking.
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


# -----------------------------------------------------------------------------
# PART 3: ANALYTIC TRIANGLE PDF FOR SUM OF TWO UNIFORMS
# -----------------------------------------------------------------------------
def triangle_pdf(z: np.ndarray) -> np.ndarray:
    """
    Analytic PDF for S2 = U1 + U2, where Ui ~ Uniform(-0.5,0.5).

    Exact formula:
        f(z) = 1 - |z|,  for |z| <= 1
             = 0,        otherwise

    WHY A TRIANGLE?
    ---------------
    If you add two centered uniforms, the middle values can be formed in many ways,
    while extreme values near -1 or +1 can only be formed in very few ways.
    That is why the density is largest in the middle and smallest at the edges.
    """
    out = np.zeros_like(z, dtype=float)
    mask = np.abs(z) <= 1.0
    out[mask] = 1.0 - np.abs(z[mask])
    return out


def part3_triangle_overlay(S2: np.ndarray) -> None:
    """
    Overlay the analytic triangle PDF on top of the Monte Carlo histogram for S2.
    """
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


# -----------------------------------------------------------------------------
# PART 4: DISCRETE CONVOLUTION
# -----------------------------------------------------------------------------
def make_discrete_uniform(dx: float = 0.002):
    """
    Create a discrete approximation of Uniform(-0.5,0.5).

    We build a grid x and then define a PDF-like array 'unif' that is constant
    across the interval.

    IMPORTANT:
    ----------
    In the continuous world, a PDF must integrate to 1.
    In the discrete grid world, we therefore normalize so that:
        sum(unif) * dx ≈ 1
    """
    x = np.arange(-0.5, 0.5 + dx, dx)

    # Start with all 1's across the support interval.
    unif = np.ones_like(x, dtype=float)

    # Normalize so discrete area is approximately 1.
    unif = unif / (unif.sum() * dx)

    return x, unif


def convolve_pdf(f: np.ndarray, g: np.ndarray, dx: float) -> np.ndarray:
    """
    Discrete convolution approximating continuous convolution.

    In continuous probability:
        (f * g)(z) = integral f(t) g(z - t) dt

    In discrete form, numpy.convolve performs the multiply-and-sum accumulation
    over shifted overlaps. We multiply by dx so that the discrete result still
    approximates a properly normalized PDF.

    This is one of the most important lines in the whole lesson:
        np.convolve(f, g, mode="full") * dx
    """
    return np.convolve(f, g, mode="full") * dx


def convolved_support(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Build the x-axis for the result of convolving arrays supported on grids x and y.

    If x spans [a,b] and y spans [c,d], then the sum spans [a+c, b+d].
    """
    return np.linspace(x[0] + y[0], x[-1] + y[-1], len(x) + len(y) - 1)


def sliding_window_convolution_demo() -> None:
    """
    Tiny printed demo of convolution as a sliding window operation.

    This uses SHORT arrays so the multiply-and-sum process can be seen directly.

    CONCEPTUAL PICTURE
    ------------------
    Imagine one function sits still while the other is:
      1) flipped
      2) shifted across
      3) multiplied elementwise where they overlap
      4) summed

    That sum becomes one output value of the convolution.
    Then the window moves one step, and the process repeats.
    """
    section("SLIDING WINDOW DEMO — what convolution is doing")

    # Two toy arrays for demonstration.
    f = np.array([1.0, 2.0, 1.0])
    g = np.array([1.0, 1.0, 1.0])

    print("Toy array f =", f)
    print("Toy array g =", g)
    print("\nFor each output position, convolution:")
    print("  - slides one array across the other,")
    print("  - multiplies overlapping entries elementwise,")
    print("  - sums those products.\n")

    # Do a manual full convolution using zero padding logic.
    full_len = len(f) + len(g) - 1
    result = np.zeros(full_len)

    # Reverse g to mimic the classical convolution picture.
    g_rev = g[::-1]

    # Pad f on both sides so the sliding window can move across.
    padded = np.pad(f, (len(g) - 1, len(g) - 1), mode="constant")

    print("Reversed g =", g_rev)
    print("Padded f   =", padded)
    print()

    for k in range(full_len):
        window = padded[k:k + len(g_rev)]
        products = window * g_rev
        result[k] = products.sum()

        print(f"Output index {k}:")
        print("  window   =", window)
        print("  g_rev    =", g_rev)
        print("  products =", products)
        print("  sum      =", result[k])
        print()

    print("Manual convolution result =", result)
    print("NumPy convolution result  =", np.convolve(f, g, mode='full'))


def part4_discrete_convolution(S4: np.ndarray) -> None:
    """
    Build the theoretical S4 curve numerically using repeated discrete convolution,
    then compare it to Monte Carlo S4.
    """
    section("PART 4 — Discrete convolution as computational theory")

    dx = 0.002
    x_u, unif = make_discrete_uniform(dx=dx)

    # First convolution:
    # uniform * uniform -> triangle
    tri_disc = convolve_pdf(unif, unif, dx)
    x_tri = convolved_support(x_u, x_u)

    # Compare discrete triangle to exact analytic triangle.
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

    # Second convolution:
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


# -----------------------------------------------------------------------------
# PART 5: OPTIONAL SHAPE COMPARISON
# -----------------------------------------------------------------------------
def part5_compare_shapes(S2: np.ndarray, S4: np.ndarray, S8: np.ndarray) -> None:
    """
    Overlay histograms for S2, S4, and S8 to visually compare how the shape evolves.
    """
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


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Main teaching flow for the lesson.
    """
    # Fix seeds so the lesson is reproducible.
    random.seed(42)
    np.random.seed(42)

    # Optional vector-space teaser from language.
    if RUN_TEXT_VECTOR_TEASER:
        text_vector_teaser()

    # Uniform sampling and summary statistics.
    part1_uniform_sampling()

    # Monte Carlo sums.
    S2, S4, S8 = part2_monte_carlo_sums()

    # Exact theory for S2.
    part3_triangle_overlay(S2)

    # Sliding-window explanation of convolution.
    if RUN_SLIDING_WINDOW_DEMO:
        sliding_window_convolution_demo()

    # Numerical theory using discrete convolution.
    part4_discrete_convolution(S4)

    # Visual comparison of shapes.
    part5_compare_shapes(S2, S4, S8)

    section("FINAL SUMMARY")
    print("1) Histograms let us see empirical distributions from simulation.")
    print("2) Summing two centered uniforms gives a triangle distribution.")
    print("3) Convolution is the mechanism behind addition of independent random variables.")
    print("4) Repeated sums tend toward Gaussian-like shapes (CLT intuition).")
    print("5) These same ideas show up in science, engineering, and data science.")


if __name__ == "__main__":
    main()
