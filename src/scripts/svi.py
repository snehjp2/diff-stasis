# For memory issues
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import pickle
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from jax import jit, vmap
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoBNAFNormal
from numpyro.optim import Adam
from stasis_simulation_differentiable import StasisSolver
from stasis_utils import (
    bitonic_sort,
    load_config,
    save_to_pickle,
    save_yaml,
)
from tqdm import tqdm

print(f"Number of devices: {jax.device_count()}")


def simulator(omegas, gammas, H_0):
    """
    Wrapper function for the Stasis simulation.
    Returns stasis value and asymptote value.
    """

    sim = StasisSolver(
        Omega_0=omegas,
        Gamma_0=gammas,
        H_0=H_0,
        log_transform=config["log_transform"],
        epsilon=0.02,
        band=0.09,
    )

    stasis_val, asymptote_val = sim.return_stasis()

    return stasis_val, asymptote_val


def simulator_vmap(omegas, gammas, H_0):
    """
    A vectorized map of the simulator. Makes things faster.
    """

    return jit(vmap(simulator, in_axes=(0, 0, None), out_axes=0))(omegas, gammas, H_0)


def main(config):
    """
    Conducts stochastic variational inference.

    - defines the variational distribution, which is a Block Neural Autoregressive Flow (arxiv:1904.04676) in this case.
            - optimized the ELBO using Adam optimizer.
            - draws 1000 posterior samples and saves them; also conducts a rejection sampling if the ratio for Gamma_{N-1} / H^0 is not satisfied.
    - saves:
            - losses during training
            - posterior samples
            - stasis values
            - matter abundances
            - plots of stasis values vs matter abundances
            - plots of Gamma and Omega distributions
            - plots of stasis configurations for 10 randomly selected samples
    """

    N_SPECIES = config["num_species"]
    BATCH_SIZE = config["batch_size"]
    GPU_BATCH = jax.device_count()
    EFFECTIVE_BATCH = BATCH_SIZE // GPU_BATCH
    pmap_flag = GPU_BATCH > 1
    timestr = time.strftime("%Y%m%d-%H%M%S")

    job_name = config["job_name"]
    save_path = config["save_path"]

    PATH = f"./{save_path}/N={N_SPECIES}_{job_name}_{timestr}"
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    save_yaml(f"{PATH}/config_{job_name}.text", config)

    def get_shapes():
        if pmap_flag:  ### if using multiple gpus
            device_shape = (GPU_BATCH,)
            batch_shape = (EFFECTIVE_BATCH,)
            return device_shape + batch_shape + (N_SPECIES,)

        else:
            return (1,) + (N_SPECIES,)

    @jit
    def abundance_penalty(asymptote_val, lower_bound, upper_bound):
        penalty = jnp.where(
            (asymptote_val < lower_bound) | (asymptote_val > upper_bound),
            (asymptote_val - lower_bound) ** 2 + (asymptote_val - upper_bound) ** 2,
            0,
        )
        return penalty

    def model(prior=config["prior"]):
        sample_shape = get_shapes()

        if prior == "log_uniform":
            raw_omegas = numpyro.sample(
                "raw_omegas",
                dist.LogUniform(config["omega_min"], config["omega_max"]).expand(
                    sample_shape
                ),
            )
            raw_gammas = numpyro.sample(
                "raw_gammas",
                dist.LogUniform(config["gamma_min"], config["gamma_max"]).expand(
                    sample_shape
                ),
            )

            omegas_sorted = jnp.log10(
                jnp.apply_along_axis(bitonic_sort, -1, raw_omegas)
            )
            gammas_sorted = jnp.log10(
                jnp.apply_along_axis(bitonic_sort, -1, raw_gammas)
            )

            gammas_sorted = gammas_sorted.at[:, -1].set(jnp.log10(config['edge_effect_ratio']))
            gammas_sorted = jnp.clip(gammas_sorted, -jnp.inf, 0)

            stasis_val, asymptote_val = simulator_vmap(
                omegas=omegas_sorted, gammas=gammas_sorted, H_0=1
            )
            log_likelihood = stasis_val.sum()
            numpyro.factor("stasis", config["kappa"] * log_likelihood)

        if prior == "uniform":
            raw_omegas = numpyro.sample(
                "raw_omegas", dist.Uniform(0, 1).expand(sample_shape)
            )
            raw_gammas = numpyro.sample(
                "raw_gammas", dist.Uniform(0, 1).expand(sample_shape)
            )

            omegas_sorted = jnp.log10(
                jnp.apply_along_axis(bitonic_sort, -1, raw_omegas)
            )
            gammas_sorted = jnp.log10(
                jnp.apply_along_axis(bitonic_sort, -1, raw_gammas)
            )

            gammas_sorted = gammas_sorted.at[:, -1].set(jnp.log10(config['edge_effect_ratio']))
            gammas_sorted = jnp.clip(gammas_sorted, -jnp.inf, 0)


            stasis_val, asymptote_val = simulator_vmap(
                omegas=omegas_sorted, gammas=gammas_sorted, H_0=1
            )
            log_likelihood = stasis_val.sum()
            numpyro.factor("stasis", config["kappa"] * log_likelihood)

        if prior == "pareto":
            raw_omegas = numpyro.sample(
                "raw_omegas",
                dist.Pareto(10, (1 / config["alpha_p_omega"])).expand(sample_shape),
            )
            raw_gammas = numpyro.sample(
                "raw_gammas",
                dist.Pareto(10, (1 / config["alpha_p_gamma"])).expand(sample_shape),
            )

            omegas = jnp.log10(1.0 / raw_omegas)
            gammas = jnp.log10(1.0 / raw_gammas)

            omegas_sorted = jnp.apply_along_axis(bitonic_sort, -1, omegas)
            gammas_sorted = jnp.apply_along_axis(bitonic_sort, -1, gammas)
            
            gammas_sorted = gammas_sorted.at[:, -1].set(jnp.log10(config['edge_effect_ratio']))
            gammas_sorted = jnp.clip(gammas_sorted, -jnp.inf, 0)

            stasis_val, asymptote_val = simulator_vmap(
                omegas=omegas_sorted, gammas=gammas_sorted, H_0=1
            )
            penalty_val = abundance_penalty(asymptote_val, 0.2, 0.8)
            log_likelihood = stasis_val.sum()
            numpyro.factor(
                "stasis", config["kappa"] * log_likelihood - 100 * penalty_val
            )

    ############################################################################################################

    guide = AutoBNAFNormal(
        model,
        num_flows=config["num_flows"],
        hidden_factors=[config["hidden_dim"], config["hidden_dim"]],
    )

    rng_key = jax.random.PRNGKey(0)
    optimizer = Adam(step_size=config["lr"])

    svi = SVI(
        model,
        guide,
        optimizer,
        loss=Trace_ELBO(num_particles=config["batch_size"] if not pmap_flag else 1),
    )

    state = svi.init(rng_key)
    best_loss_counter, best_loss = 0, jnp.inf
    losses = []

    for i in tqdm(range(config["num_epochs"])):
        rng_key, _ = jax.random.split(rng_key)
        state, loss = svi.update(state)
        losses.append(loss)
        best_loss_counter += 1

        if jnp.isnan(loss):
            print("Nan loss!")
            break

        if loss < best_loss:
            best_loss = loss
            save_to_pickle(f"{PATH}/best_model_params.pkl", svi.get_params(state))
            save_to_pickle(f"{PATH}/best_model_state.pkl", state)
            best_loss_counter = 0

        if i > 200 and best_loss_counter > config["patience"]:
            print("Early stopping!")
            break

    save_to_pickle(f"{PATH}/final_model_params.pkl", svi.get_params(state))

    plt.figure(figsize=(10, 10))
    plt.plot(losses)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.savefig(f"{PATH}/losses.png")
    plt.close()

    desired_num_samples = 1000

    params = pickle.load(open(f"{PATH}/best_model_params.pkl", "rb"))
    best_samples = guide.sample_posterior(
        jax.random.PRNGKey(1), svi.get_params(state), (desired_num_samples,)
    )

    gammas = jnp.squeeze(best_samples["raw_gammas"])
    omegas = jnp.squeeze(best_samples["raw_omegas"])

    if config["prior"] == "log_uniform":
        sort_transformed_gammas = jnp.sort(gammas, axis=-1)
        sort_transformed_omegas = jnp.sort(omegas, axis=-1)

        best_sorted_gammas = 10 ** (
            sort_transformed_gammas.reshape(desired_num_samples, config["num_species"])
        )
        best_sorted_omegas = 10 ** (
            sort_transformed_omegas.reshape(desired_num_samples, config["num_species"])
        )

    elif config["prior"] == "pareto":
        omegas = jnp.log10(1.0 / omegas)
        gammas = jnp.log10(1.0 / gammas)

        transformed_omegas = jnp.sort(omegas, axis=-1)
        transformed_gammas = jnp.sort(gammas, axis=-1)

        best_sorted_gammas = 10 ** (
            transformed_gammas.reshape(desired_num_samples, config["num_species"])
        )
        best_sorted_omegas = 10 ** (
            transformed_omegas.reshape(desired_num_samples, config["num_species"])
        )

    best_sorted_gammas = best_sorted_gammas.at[best_sorted_gammas > config['edge_effect_ratio']].set(config['edge_effect_ratio'])
    best_sorted_gammas = best_sorted_gammas.at[-1].set(config['edge_effect_ratio'])

    best_sorted_samples = {"gammas": best_sorted_gammas, "omegas": best_sorted_omegas}

    save_to_pickle(f"{PATH}/sorted_samples_best_model.pkl", best_sorted_samples)

    params = pickle.load(open(f"{PATH}/final_model_params.pkl", "rb"))
    final_samples = guide.sample_posterior(
        jax.random.PRNGKey(1), params, (desired_num_samples,)
    )

    gammas = jnp.squeeze(final_samples["raw_gammas"])
    omegas = jnp.squeeze(final_samples["raw_omegas"])

    if config["prior"] == "log_uniform":
        sort_transformed_gammas = jnp.sort(gammas, axis=-1)
        sort_transformed_omegas = jnp.sort(omegas, axis=-1)

        final_sorted_gammas = 10 ** (
            sort_transformed_gammas.reshape(desired_num_samples, config["num_species"])
        )
        final_sorted_omegas = 10 ** (
            sort_transformed_omegas.reshape(desired_num_samples, config["num_species"])
        )

    if config["prior"] == "pareto":
        omegas = jnp.log10(1.0 / omegas)
        gammas = jnp.log10(1.0 / gammas)

        transformed_omegas = jnp.sort(omegas, axis=-1)
        transformed_gammas = jnp.sort(gammas, axis=-1)

        final_sorted_gammas = 10 ** (
            transformed_gammas.reshape(desired_num_samples, config["num_species"])
        )
        final_sorted_omegas = 10 ** (
            transformed_omegas.reshape(desired_num_samples, config["num_species"])
        )


    final_sorted_gammas = final_sorted_gammas.at[final_sorted_gammas > config['edge_effect_ratio']].set(config['edge_effect_ratio'])
    final_sorted_gammas = final_sorted_gammas.at[-1].set(config['edge_effect_ratio'])

    final_sorted_samples = {
        "gammas": final_sorted_gammas,
        "omegas": final_sorted_omegas,
    }

    save_to_pickle(f"{PATH}/sorted_samples_final_model.pkl", final_sorted_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="experiment_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config_file)
    main(config)
