import math
import warnings

import numpy as np
import pandas as pd
from autodp.mechanism_zoo import LaplaceMechanism
from autodp.transformer_zoo import AmplificationBySampling
from omegaconf import OmegaConf

from alibaba_privacy_workload.opacus_rdp import compute_rdp, get_privacy_spent

warnings.filterwarnings("ignore")
ALPHAS = [
    1.5,
    1.75,
    2,
    2.5,
    3,
    4,
    5,
    6,
    8,
    16,
    32,
    64,
]


def sample_from_frequencies_dict(d):
    frequencies = np.array(list(d.values()))
    frequencies = frequencies / np.sum(frequencies)
    choice = np.random.choice(list(d.keys()), p=frequencies)
    return choice


def ceil_decimal(x):
    return math.ceil(x * 10) / 10


def floor_decimal(x):
    return math.floor(x * 10) / 10


def map_to_range(value, min_input, max_input, min_output, max_output):
    normalized_value = (value - min_input) / (max_input - min_input)
    return min_output + normalized_value * (max_output - min_output)


def get_df(file, header=None):
    df = pd.read_csv(file, header=None)
    df.columns = (
        pd.read_csv(file.with_suffix(".header")).columns if header is None else header
    )
    return df


def gaussian_dp2sigma(epsilon, delta, sensitivity=1):
    return (sensitivity / epsilon) * math.sqrt(2 * math.log(1.25 / delta))


def compute_rdp_epsilons_gaussian(sigma, steps=1, alphas=ALPHAS):
    orders = []
    for alpha in alphas:
        orders.append(alpha * steps / (2 * (sigma**2)))
    return orders


def compute_gaussian_demands(epsilon, delta, steps=1, alphas=ALPHAS):
    sigma = (1 / epsilon) * math.sqrt(2 * math.log(1.25 / delta))
    orders = compute_rdp_epsilons_gaussian(sigma, steps)
    return orders


def compute_laplace_demands(laplace_noise, steps=1, alphas=ALPHAS):
    orders = []
    λ = laplace_noise
    for α in alphas:
        ε = (1 / (α - 1)) * np.log(
            (α / (2 * α - 1)) * np.exp((α - 1) / λ)
            + ((α - 1) / (2 * α - 1)) * np.exp(-α / λ)
        )
        orders.append(float(ε) * steps)
    return orders


def compute_subsampled_gaussian_demands(
    sigma, dataset_size=50000, batch_size=100, epochs=100, alphas=ALPHAS
):
    sampling_probability = batch_size / dataset_size
    steps = (dataset_size * epochs) // batch_size
    rdp = compute_rdp(
        q=sampling_probability,
        noise_multiplier=sigma,
        steps=steps,
        orders=alphas,
    )
    return rdp


def compute_subsampled_laplace_rdp(q, noise_multiplier, steps, orders, alphas=ALPHAS):
    # Same interface as compute_rdp for subsampled Gaussian
    curve = AmplificationBySampling(PoissonSampling=True)(
        LaplaceMechanism(b=noise_multiplier),
        q,
        improved_bound_flag=True,
    )
    rdp = np.array([curve.get_RDP(alpha) * steps for alpha in alphas])
    return rdp


def compute_noise_and_rdp_from_target_epsilon(
    target_epsilon,
    target_delta,
    epochs,
    batch_size,
    dataset_size,
    alphas=None,
    approx_ratio=0.01,
    min_noise=0.001,
    max_noise=1000,
    gaussian=True,  # Computes subsampled Laplace if set to false
):
    """
    Takes a target epsilon (eps) and some hyperparameters.
    Returns a noise scale that gives an epsilon in [0.99 eps, eps].
    The approximation ratio can be tuned.
    If alphas is None, we'll explore orders.
    """
    steps = epochs * dataset_size // batch_size
    sampling_rate = batch_size / dataset_size
    if alphas is None:
        alphas = ALPHAS

    def get_eps_rdp(noise, gaussian=True):
        if gaussian:
            rdp = compute_rdp(
                q=sampling_rate, noise_multiplier=noise, steps=steps, orders=alphas
            )
        else:
            rdp = compute_subsampled_laplace_rdp(
                q=sampling_rate, noise_multiplier=noise, steps=steps, orders=alphas
            )
        epsilon, _ = get_privacy_spent(orders=alphas, rdp=rdp, delta=target_delta)
        return epsilon, rdp

    # Binary search bounds
    noise_min = min_noise
    noise_max = max_noise

    # Start with the smallest epsilon possible with reasonable noise
    candidate_noise = noise_max
    candidate_eps, candidate_rdp = get_eps_rdp(candidate_noise, gaussian)

    if candidate_eps > target_epsilon:
        # raise ("Cannot reach target eps. Try to increase MAX_NOISE.")
        # We just output the maximum noise instead of failing.
        return candidate_eps, candidate_rdp.tolist()

    # Search up to approx ratio
    while (
        candidate_eps < (1 - approx_ratio) * target_epsilon
        or candidate_eps > target_epsilon
    ):
        if candidate_eps < (1 - approx_ratio) * target_epsilon:
            noise_max = candidate_noise
        else:
            noise_min = candidate_noise
        candidate_noise = (noise_max + noise_min) / 2
        candidate_eps, candidate_rdp = get_eps_rdp(candidate_noise, gaussian)

    return candidate_noise, candidate_rdp.tolist()
