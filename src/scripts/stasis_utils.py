import pickle
import pprint

import jax
import jax.numpy as jnp
import yaml
from jax import jit


def load_config(yaml_file):
    """load configuration from a yaml file."""
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def save_to_pickle(filename, data):
    """Utility to save data to a pickle file."""
    with open(filename, "wb") as file:
        pickle.dump(data, file)

def save_yaml(filename, data):
    """Utility to save data as YAML text file."""
    with open(filename, "w") as file:
        yaml.dump(data, file)


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
