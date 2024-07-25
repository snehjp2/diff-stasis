import os
import pickle
import pprint

import jax
import jax.numpy as jnp
import yaml
from jax import jit, vmap
from stasis_simulation_differentiable import StasisSolver


def load_config(yaml_file):
    """load configuration from a yaml file."""
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def simulator_eval(omegas, gammas, H_0):
    """Simulator to compute stasis value and asymptote value."""
    sim = StasisSolver(
        Omega_0=omegas,
        Gamma_0=gammas,
        H_0=H_0,
        log_transform=False,
        epsilon=0.02,
        band=0.09,
    )

    stasis_val, asymptote_val = sim.return_stasis(eval=True)
    return stasis_val, asymptote_val


def simulator_vmap_eval(omegas, gammas, H_0):
    """A vectorized map of the simulator."""
    return jit(vmap(simulator_eval, in_axes=(0, 0, None), out_axes=0))(
        omegas, gammas, H_0
    )


def create_directory(path):
    """Utility to create a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def save_to_pickle(filename, data):
    """Utility to save data to a pickle file."""
    with open(filename, "wb") as file:
        pickle.dump(data, file)


def save_to_text(filename, data):
    """Utility to save data to a pickle file."""
    with open(filename, "w") as file:
        pprint.pprint(data, stream=file)


def save_yaml(filename, data):
    """Utility to save data as YAML text file."""
    with open(filename, "w") as file:
        yaml.dump(data, file)


@jit
def ordered_transform(raw_params):
    softplus_diffs = jax.nn.softplus(raw_params)
    sorted_params = jnp.cumsum(softplus_diffs, axis=-1)
    return sorted_params


@jit
def differentiable_min(x):
    return -jax.scipy.special.logsumexp(-x)


@jit
def differentiable_max(x):
    return jax.scipy.special.logsumexp(x)


@jit
def sort_transform_log_uniform(raw_samples, lower_bound, upper_bound):
    sorted_samples = jnp.array(
        [
            ordered_transform(sample)
            * (upper_bound - lower_bound)
            / jnp.sum(jax.nn.softplus(sample))
            + lower_bound
            for sample in raw_samples
        ]
    )
    return sorted_samples


@jit
def shuffle_samples(array, key):
    def shuffle_row(row_key, row):
        return row[jax.random.permutation(row_key, row.shape[0])]

    keys = jax.random.split(key, array.shape[0])
    shuffled_array = jax.vmap(shuffle_row)(keys, array)

    return shuffled_array, keys


@jit
def shuffle_log_probs(array, keys):
    def apply_permutation(row_key, log_prob):
        return log_prob[jax.random.permutation(row_key, 1)][
            0
        ]  # No actual shuffling needed for 1-element array

    shuffled_log_probs = jax.vmap(apply_permutation)(keys, array[:, None])
    return jnp.squeeze(shuffled_log_probs)  # Remove the singleton dimension


# Implement a simple differentiable sorting network
@jit
def bitonic_sort(x):
    n = x.shape[0]
    if n <= 1:
        return x

    # Split the array into two halves
    half_n = n // 2
    lower = x[:half_n]
    upper = x[half_n:]

    # Recursively sort both halves
    lower_sorted = bitonic_sort(lower)
    upper_sorted = bitonic_sort(upper)

    # Perform a bitonic merge
    return bitonic_merge(lower_sorted, upper_sorted)


@jit
def bitonic_merge(lower, upper):
    n = lower.shape[0] + upper.shape[0]
    sorted_array = jnp.concatenate([lower, upper])

    def compare_and_swap(arr, _):
        def swap(i, arr):
            a = arr[i]
            b = arr[i + 1]
            swapped_a = jnp.minimum(a, b)
            swapped_b = jnp.maximum(a, b)
            arr = arr.at[i].set(swapped_a)
            arr = arr.at[i + 1].set(swapped_b)
            return arr

        indices = jnp.arange(n - 1)
        arr = jax.lax.scan(lambda arr, i: (swap(i, arr), None), arr, indices)[0]
        return arr, None

    indices = jnp.arange(n)
    sorted_array, _ = jax.lax.scan(compare_and_swap, sorted_array, indices)
    return sorted_array


@jit
def sort_transform_pareto(raw_samples, lower_bound, upper_bound):
    def ordered_transform(sample):
        sorted_sample = bitonic_sort(sample)
        return sorted_sample

    sorted_samples = jnp.array([ordered_transform(sample) for sample in raw_samples])
    return sorted_samples
