import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import xlogy, xlog1py
from scipy.stats import beta


def determine_beta_prior(
    target_mean: float = 0.25,
    threshold: float = 0.40,
    target_cdf: float = 0.85,
) -> tuple[float, float]:
    """
    Determine alpha and beta such that:

        E(theta) = target_mean
        P(theta < threshold) = target_cdf
    """

    beta_alpha_ratio = (1.0 - target_mean) / target_mean

    def objective(alpha: float) -> float:
        beta_parameter = beta_alpha_ratio * alpha
        return beta.cdf(
            threshold,
            alpha,
            beta_parameter,
        ) - target_cdf

    alpha = brentq(objective, 1.0e-6, 1.0e4)
    beta_parameter = beta_alpha_ratio * alpha

    return alpha, beta_parameter


def create_binary_sample(
    n: int = 15,
    mle_theta: float = 0.75,
    seed: int = 12345,
) -> np.ndarray:
    """
    Create a randomly ordered binary sample whose sample proportion
    is exactly mle_theta.

    Here, 1 means survival and 0 means nonsurvival.
    """

    survivors = int(round(n * mle_theta))

    if not np.isclose(survivors / n, mle_theta):
        raise ValueError(
            "The requested sample size does not permit the desired "
            "sample proportion exactly."
        )

    sample = np.concatenate(
        [
            np.ones(survivors, dtype=int),
            np.zeros(n - survivors, dtype=int),
        ]
    )

    rng = np.random.default_rng(seed)
    rng.shuffle(sample)

    return sample


def normalized_likelihood(
    theta: np.ndarray,
    survivors: int,
    failures: int,
) -> np.ndarray:
    """
    Compute the binomial likelihood shape, normalized so its
    maximum value is 1.
    """

    log_likelihood = (
        xlogy(survivors, theta)
        + xlog1py(failures, -theta)
    )

    log_likelihood -= np.max(log_likelihood)

    return np.exp(log_likelihood)


def main() -> None:
    # -------------------------------------------------------------
    # 1. Construct the elicited beta prior.
    # -------------------------------------------------------------
    alpha_prior, beta_prior = determine_beta_prior()

    # -------------------------------------------------------------
    # 2. Generate a random ordering of 25 survivors and 75 failures.
    # -------------------------------------------------------------
    cancer_sample = create_binary_sample(
        n=20,
        mle_theta=0.30,
        seed=2026,
    )

    n = cancer_sample.size
    survivors = int(cancer_sample.sum())
    failures = n - survivors

    theta_mle = survivors / n

    # -------------------------------------------------------------
    # 3. Conjugate beta-binomial update.
    # -------------------------------------------------------------
    alpha_posterior = alpha_prior + survivors
    beta_posterior = beta_prior + failures

    posterior_mean = (
        alpha_posterior
        / (alpha_posterior + beta_posterior)
    )

    prior_probability_below_04 = beta.cdf(
        0.40,
        alpha_prior,
        beta_prior,
    )

    posterior_probability_below_04 = beta.cdf(
        0.40,
        alpha_posterior,
        beta_posterior,
    )

    # -------------------------------------------------------------
    # 4. Print results.
    # -------------------------------------------------------------
    print("Simulated CANCER dataset")
    print("------------------------")
    print(f"Number of patients:              {n}")
    print(f"Six-month survivors:             {survivors}")
    print(f"Six-month nonsurvivors:          {failures}")
    print(f"MLE of theta:                    {theta_mle:.6f}")
    print()

    print("Prior")
    print("-----")
    print(
        f"theta ~ Beta({alpha_prior:.6f}, "
        f"{beta_prior:.6f})"
    )
    print(
        f"Prior effective sample size:     "
        f"{alpha_prior + beta_prior:.6f}"
    )
    print(
        f"Prior mean:                      "
        f"{alpha_prior / (alpha_prior + beta_prior):.6f}"
    )
    print(
        f"Prior P(theta < 0.4):            "
        f"{prior_probability_below_04:.6f}"
    )
    print()

    print("Posterior")
    print("---------")
    print(
        f"theta | data ~ Beta({alpha_posterior:.6f}, "
        f"{beta_posterior:.6f})"
    )
    print(
        f"Posterior mean:                  "
        f"{posterior_mean:.6f}"
    )
    print(
        f"Posterior P(theta < 0.4):        "
        f"{posterior_probability_below_04:.6f}"
    )

    # -------------------------------------------------------------
    # 5. Calculate curves.
    # -------------------------------------------------------------
    theta = np.linspace(0.0001, 0.9999, 3000)

    prior_density = beta.pdf(
        theta,
        alpha_prior,
        beta_prior,
    )

    posterior_density = beta.pdf(
        theta,
        alpha_posterior,
        beta_posterior,
    )

    likelihood = normalized_likelihood(
        theta,
        survivors,
        failures,
    )

    # Scale the likelihood vertically so it can be compared visually
    # with the prior and posterior densities.
    likelihood_scaled = (
        likelihood
        * max(prior_density.max(), posterior_density.max())
    )

    # -------------------------------------------------------------
    # 6. Plot all three curves.
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    plt.plot(
        theta,
        prior_density,
        color="royalblue",
        linewidth=2.5,
        linestyle="-",
        label=(
            rf"Prior: Beta({alpha_prior:.2f}, "
            rf"{beta_prior:.2f})"
        ),
    )

    plt.plot(
        theta,
        likelihood_scaled,
        color="darkorange",
        linewidth=2.5,
        linestyle="-",
        label=(
            rf"Likelihood: $s={survivors}$, "
            rf"$f={failures}$, scaled"
        ),
    )

    plt.plot(
        theta,
        posterior_density,
        color="forestgreen",
        linewidth=2.5,
        linestyle="-",
        label=(
            rf"Posterior: Beta({alpha_posterior:.2f}, "
            rf"{beta_posterior:.2f})"
        ),
    )

    plt.axvline(
        theta_mle,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=rf"MLE: $\hat{{\theta}}={theta_mle:.2f}$",
    )

    plt.axvline(
        0.40,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label=r"Threshold: $\theta=0.40$",
    )

    plt.xlabel(r"Six-month survival probability, $\theta$")
    plt.ylabel("Density / scaled likelihood")
    plt.title(
        "Beta-Binomial Conjugate Analysis\n"
        "Simulated CANCER Data: 25 Survivors out of 100"
    )
    plt.xlim(0, 0.65)
    plt.ylim(bottom=0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "simulated_cancer_beta_binomial.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()