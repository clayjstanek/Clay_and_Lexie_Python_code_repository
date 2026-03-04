# Session 2 Solution (Probability, CLT, and Convolution)
# Works as a single script: Part A exercises + Part B mini-project
#
# Dependencies: numpy, matplotlib. (scipy not required)
#
# Key ideas:
# - Monte Carlo sampling of sums of uniforms (S2, S4, optional S8)
# - Analytic triangle PDF for S2
# - Discrete convolution on a grid to get theoretical PDFs for S2 and S4

import numpy as np
import random
import matplotlib.pyplot as plt


def sample_uniform_centered(n_samples: int) -> np.ndarray:
    \"\"\"Samples Uniform(-0.5, 0.5) using random.random().\"\"\"
    # random.random() gives Uniform(0,1); shift to (-0.5,0.5)
    return np.array([random.random() - 0.5 for _ in range(n_samples)], dtype=float)


def monte_carlo_sums(n_samples: int, n_terms: int) -> np.ndarray:
    \"\"\"Monte Carlo: sum of n_terms independent Uniform(-0.5,0.5).\"\"\"
    s = np.zeros(n_samples, dtype=float)
    for _ in range(n_terms):
        s += sample_uniform_centered(n_samples)
    return s


def triangle_pdf(z: np.ndarray) -> np.ndarray:
    \"\"\"Analytic PDF for sum of two Uniform(-0.5,0.5): f(z)=1-|z| on [-1,1].\"\"\"
    out = np.zeros_like(z, dtype=float)
    mask = np.abs(z) <= 1.0
    out[mask] = 1.0 - np.abs(z[mask])
    return out


def make_discrete_uniform(dx: float = 0.002):
    \"\"\"Discrete grid + discrete uniform PDF on [-0.5,0.5].\"\"\"
    x = np.arange(-0.5, 0.5 + dx, dx)  # include endpoint
    unif = np.ones_like(x, dtype=float)  # constant on the interval
    # Normalize so integral ~ 1: sum(unif)*dx should be 1.
    unif = unif / (unif.sum() * dx)
    return x, unif


def convolve_pdf(f: np.ndarray, g: np.ndarray, dx: float) -> np.ndarray:
    \"\"\"Discrete convolution approximating integral convolution; multiply by dx.\"\"\"
    return np.convolve(f, g, mode='full') * dx


def convolved_support(x: np.ndarray, y: np.ndarray, dx: float) -> np.ndarray:
    \"\"\"If x and y are equally spaced grids with same dx, support grid for convolution.\"\"\"
    # For two grids: min adds, max adds. Length = len(x)+len(y)-1
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


def main():
    random.seed(42)
    np.random.seed(42)

    # ---------------- Part A1 ----------------
    N = 10000
    u01 = np.array([random.random() for _ in range(N)], dtype=float)
    print("A1: Uniform(0,1) samples")
    print("  mean ~", u01.mean())
    print("  var  ~", u01.var())

    plt.figure()
    plt.hist(u01, bins=40, density=True)
    plt.title("A1: Uniform(0,1) via random.random()")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.show()

    # ---------------- Part A2 ----------------
    N2 = 50000
    S2 = monte_carlo_sums(N2, 2)
    S4 = monte_carlo_sums(N2, 4)

    plt.figure()
    plt.hist(S2, bins=80, density=True)
    plt.title("A2: S2 = U1+U2, U~Unif(-0.5,0.5)")
    plt.xlabel("s")
    plt.ylabel("density")
    plt.show()

    plt.figure()
    plt.hist(S4, bins=80, density=True)
    plt.title("A2: S4 = U1+U2+U3+U4, U~Unif(-0.5,0.5)")
    plt.xlabel("s")
    plt.ylabel("density")
    plt.show()

    # ---------------- Part A3 ----------------
    z = np.linspace(-1, 1, 1000)
    tri = triangle_pdf(z)
    plot_hist_with_curve(S2, z, tri, "A3: Monte Carlo S2 vs analytic triangle PDF")

    # ---------------- Part A4 + Part B ----------------
    dx = 0.002
    x_u, unif = make_discrete_uniform(dx=dx)

    # Convolve uniform with itself -> triangle (discrete)
    tri_disc = convolve_pdf(unif, unif, dx)
    x_tri = convolved_support(x_u, x_u, dx)

    # Compare discrete triangle to analytic triangle
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

    # Convolve triangle with triangle -> 4-uniform sum PDF
    four_disc = convolve_pdf(tri_disc, tri_disc, dx)
    x_four = convolved_support(x_tri, x_tri, dx)

    # Compare Monte Carlo S4 with discrete convolution theory
    plot_hist_with_curve(S4, x_four, four_disc, "Part B: Monte Carlo S4 vs discrete convolution theory")

    # Optional: S8 for deeper CLT feel (convolve four with four)
    do_s8 = True
    if do_s8:
        S8 = monte_carlo_sums(N2, 8)
        eight_disc = convolve_pdf(four_disc, four_disc, dx)
        x_eight = convolved_support(x_four, x_four, dx)
        plot_hist_with_curve(S8, x_eight, eight_disc, "Optional: Monte Carlo S8 vs discrete convolution theory")

    print("\nDone. Key checks:")
    print("- S2 matches triangle shape; S4 is smoother; S8 smoother still.")
    print("- Discrete convolution requires multiplying by dx to preserve normalization.")


if __name__ == "__main__":
    main()
