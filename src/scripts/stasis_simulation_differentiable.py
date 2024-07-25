import warnings

import jax
import jax.lax as lax
import numpy as np
from diffrax import (
    DiscreteTerminatingEvent,
    Kvaerno5,
    ODETerm,
    PIDController,
    RecursiveCheckpointAdjoint,
    SaveAt,
    diffeqsolve,
)
from jax import jit, value_and_grad
from jax import numpy as jnp
from typing import Union
from jax import random

# from stasis_simulation_non_diff import Simulator as StasisSolver

np.random.seed(0)

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore")


class StasisSolver:
    """
    Differentiable simulation to solve the Boltzmann Equation for a given set of initial conditions.
    Solving is vmap-compatible and further is pmap-compatible if using multiple devices.

    INPUTS:

        - Omega_0: Initial Abundances of the species
        - Gamma_0: Decay rates of the species
        - H_0: Hubble constant
        - max_time: max (FRW) time to run simualtion for. This value gets exponentiated.
        - n_points: Number of points to evaluate the solution (simulation resolution)
        - max_steps: Maximum number of steps to solve the ODE
        - epsilon: Epsilon value for the flatness score
        - band: Band value for the flatness score
        - log_transform: Whether to log-transform the input values (for stable optimization)

    OUPTUTS:

        - stasis_val: Number of e-folds of stasis
        - abundance: Asymptotic value of the total abundance

    NOTES:

        - The number of species is set by the shape of Omega_0 and Gamma_0 that are passed as inputs.
        - Default epsilon and band values roughly correspond to a 10% stasis tolerance; e.g. deviations in abundance of 0.1 from the asymptote during stasis.
        - Both `stasis_val` and `abundance_val` are differentiable w.r.t. the initial conditions Omega_0 and Gamma_0.
        - Simulation is terminated when a solution is found, alternatively when the total abundance is indicative of radiation domination (i.e. is very small).
        - If the solution is not found, an error is not thrown and both the stasis value and stasis abundance are set to 0.
    """

    def __init__(
        self,
        Omega_0: Union[jnp.ndarray, float],
        Gamma_0: Union[jnp.ndarray, float],
        H_0: float = 1.0,
        max_time: int = 400,
        n_points: int = 4000,
        max_steps: int = 4096,  # changing max steps can change expense dramatically! e.g. see https://github.com/patrick-kidger/diffrax/issues/161
        epsilon: float = 0.02,
        band: float = 0.09,
        log_transform: bool = False,
        use_adjoint = True
    ):
        self.H_0 = H_0
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.band = band
        self.max_time = max_time
        self.n_points = n_points
        self.log_transform = log_transform
        self.use_adjoint = use_adjoint

        assert (
            Omega_0.shape == Gamma_0.shape
        ), "Must have same number of species and decay rates."

        if self.log_transform:
            Omega_0 = 10 ** (Omega_0)
            Gamma_0 = 10 ** (Gamma_0)

        self.Omega_0 = jnp.array(Omega_0).squeeze()
        self.Gamma_0 = jnp.array(Gamma_0).squeeze()
        self.Omega_0 = self.Omega_0 / jnp.sum(self.Omega_0)
        self.N = self.Omega_0.shape[0]

        self.t_span = (0, jnp.exp(self.max_time))
        self.t_eval = jnp.exp(jnp.linspace(0, max_time, self.n_points))

        if len(self.Omega_0.shape) == 1:
            self.y0 = jnp.concatenate([jnp.array([self.H_0]), self.Omega_0])

        else:
            batch_size = self.Omega_0.shape[0]
            expanded_H_0 = jnp.tile(jnp.array([self.H_0]), (batch_size, 1))
            self.y0 = jnp.concatenate([expanded_H_0, self.Omega_0], axis=-1)
            
    @staticmethod
    @jit
    def replace_infs_with_random_uniform(x, 
                                         bottom=10000., 
                                         top=20000.):
        
        ''' 
        ODE solution, when returned, is padded with jnp.nans due to static array sizes in Jax. 
        This replaces the nans with random uniform values for differentiability reasons. 
        These values get clipped away.
        '''
        
        key = random.PRNGKey(0)
        rand_nums = random.uniform(key, x.shape, minval=bottom, maxval=top)
        mask = jnp.isinf(x)
        return jnp.where(mask, rand_nums, x)

    @staticmethod
    @jit
    def return_stasis_e_folds(
        total_abundance, target_value=0.0, epsilon=0.02, band=0.09
    ):
        """
        Computes the flatness score (stasis value) for the total abundance.
        Takes in the abundance computed from (non-differentiable) stasis-finder.
        The stasis value is returned.
        """

        target_value = jnp.clip(target_value, 0.00, 1.00)

        differences = jnp.abs(jnp.diff(total_abundance))
        weights = jnp.exp(-jnp.abs(total_abundance[:-1] - target_value) / band)
        flatness_scores = jnp.exp(-differences / epsilon) * weights
        flatness_scores = jnp.where(flatness_scores < 0.2, 0.0, flatness_scores)

        return jnp.sum(flatness_scores)

    @staticmethod
    @jit
    def return_stasis_abundance(total_abundance):
        """
        Differentiable stasis-finder to compute the stasis abundance value. This is used primarily to compute the abundance.
        The mean of the non-differentiable stasis value and the differentiable stasis value is returned.

        """
        total_abundance = jnp.asarray(total_abundance)

        in_range = (total_abundance > 0.01) & (total_abundance < (0.99))

        differences = jnp.abs(total_abundance[:, None] - total_abundance[None, :])
        close_enough = differences < 0.1

        interval_lengths = jnp.triu(close_enough, k=1).sum(axis=1)
        longest_stasis_length = jnp.max(interval_lengths)
        longest_stasis_start = jnp.argmax(interval_lengths)
        longest_stasis_end = longest_stasis_start + longest_stasis_length

        index_range = jnp.arange(total_abundance.size)
        stasis_mask = (index_range >= longest_stasis_start) & (
            index_range <= longest_stasis_end
        )

        full_mask = in_range & stasis_mask
        non_zero_masked_values = jnp.where(full_mask, total_abundance, jnp.nan)
        abundance = jnp.nanmedian(non_zero_masked_values)

        ## this algorithm can compute the stasis configuration as well, but it doesn't work as well as 'return_stasis_e_folds'.
        stasis_beginning = longest_stasis_start
        stasis_ending = longest_stasis_end
        stasis_length = longest_stasis_length

        return abundance

    def solve_boltzmann_eq(self):
        """
        Solves the Boltzmann Equations using the diffrax library.
        Solution is terminated when the total abundance is radiation-dominated (e.g. below a threshold (1e-4)).
        """

        @jit
        def boltzmann_eq(t, y, args):
            H, Omega = y[0], y[1:]
            Omega_m = jnp.sum(Omega)
            dH_dt = (-1 / 2) * H**2 * (4 - Omega_m)
            dOmega_dt = H * Omega - self.Gamma_0 * Omega - H * Omega * Omega_m

            return jnp.concatenate((jnp.array([dH_dt]), jnp.array(dOmega_dt)))

        def condition_fn(state, **kwargs):
            del kwargs
            OMEGA_M_MIN = 1e-4
            Omega_m = jnp.sum(state.y[1:])
            condition = Omega_m < OMEGA_M_MIN

            return lax.cond(condition, lambda _: True, lambda _: False, None)

        def configure_solver(t_eval, condition_fn):
            equation = ODETerm(boltzmann_eq)  ## solve the Boltzmann equations
            solver = Kvaerno5()  ## use the Kvaerno5 solver
            saveat = SaveAt(ts=t_eval)  ## checkpointing the solution
            stepsize_controller = PIDController(
                rtol=1e-8, atol=1e-8
            )  ## adaptive stepsize controller
            event = DiscreteTerminatingEvent(
                cond_fn=condition_fn
            )  ## terminate the simulation when the condition is met
            adjoint = RecursiveCheckpointAdjoint(
                checkpoints=int(100)
            )  ## for cheaper backpropagation

            return equation, solver, saveat, stepsize_controller, event, adjoint

        (equation, solver, saveat, stepsize_controller, event, adjoint) = (
            configure_solver(self.t_eval, condition_fn)
        )

        @jit
        def return_sol():
            """
            Wrapper to return the ODE solution.
            """
            sol = diffeqsolve(
                equation,
                solver,
                t0=self.t_eval[0],
                t1=self.t_eval[-1],
                dt0=0.01,
                y0=self.y0,
                args=(self.Gamma_0,),
                max_steps=int(self.max_steps),
                saveat=saveat,
                stepsize_controller=stepsize_controller,
                adjoint=adjoint,
                discrete_terminating_event=event,
                throw=False
            )
            
                
            return sol

        self.sol = return_sol()

        return self.sol

    def return_stasis(self):
        """
        Returns the stasis value and stasis abundance.
        If a solution is not found, a stasis value and abundance of 0 is returned.
        """

        @jit
        def scale_factor_from_H(H_sol, ts, a_initial=1.0):
            """
            Compute e-folds from the time evolution of the scale factor.
            Numerical integral of the solution for the Hubble parameter.
            """

            ln_a_initial = jnp.log(a_initial)  ## initial scale factor
            H_sol = jnp.where(
                jnp.isfinite(H_sol), H_sol, 0.0
            )  ## replaces nans and infs with 0
            finite_mask = jnp.isfinite(ts)  ## mask for finite values
            non_inf_ts = jnp.sum(finite_mask)
            ts = jnp.where(jnp.isfinite(ts), ts, 0.0)

            total_ln_a_change = jnp.trapezoid(
                H_sol, ts
            )  ## trapezoidal rule for numerical integration
            final_ln_a = ln_a_initial + total_ln_a_change
            final_a = jnp.exp(final_ln_a)

            N_efolds = jnp.log(final_a / a_initial)

            ## the ratio of the number of e-folds to the number of non-infinite time steps is used to scale the stasis value. Return both.
            return N_efolds, non_inf_ts

        self.solve_boltzmann_eq()  ## solve the Boltzmann equations

        if self.sol is None:
            self.stasis_val, self.stasis_abundance = (
                0.0,
                0.0,
            )  ## if no solution is found, return 0

        else:
            self.species_sol = jnp.array(self.sol.ys[:, 1:]).T
            self.H_sol = jnp.array(self.sol.ys[:, 0])
            self.t_sol = jnp.abs(self.sol.ts)
            self.scale_factor_e_folds, self.non_inf_ts = scale_factor_from_H(
                self.H_sol, self.t_sol
            )
            self.total_abundance = self.replace_infs_with_random_uniform(
                jnp.sum(self.species_sol, axis=0)
            )
            self.stasis_abundance = self.return_stasis_abundance(
                total_abundance=self.total_abundance
            )

            self.stasis_val = self.return_stasis_e_folds(
                total_abundance=self.total_abundance,
                target_value=self.stasis_abundance,
                epsilon=self.epsilon,
                band=self.band,
            )  ## computes stasis duration in t

            self.stasis_val = self.stasis_val * (
                self.scale_factor_e_folds / self.non_inf_ts
            )  ## convert stasis duration to e-folds

            self.stasis_abundance = lax.cond(
                (self.stasis_abundance >= 0.01) & (self.stasis_abundance <= 0.999),
                lambda _: self.stasis_abundance,
                lambda _: 0.0,
                None,
            )  ## if the stasis abundance is not in the range (0.01, 0.99), set it to 0

            self.stasis_val = lax.cond(
                (self.stasis_abundance > 0.01) & (self.stasis_abundance < 0.99),
                lambda _: self.stasis_val,
                lambda _: 0.0,
                None,
            )  ## if the stasis value is not in the range (0.01, 0.99), set it to 0

            return self.stasis_val, self.stasis_abundance

    def contributing_species(self):
        stasis_end_index = jnp.searchsorted(
            self.scale_factor_e_folds, self.stasis_end, side="right"
        )

        clean_species_sol = jnp.where(
            jnp.isinf(self.species_sol), 0.0, self.species_sol
        )
        max_indices = jnp.argmax(clean_species_sol, axis=1)

        max_before_stasis_end = max_indices < stasis_end_index

        self.contributing_species = lax.cond(
            self.stasis_val < 2,
            lambda _: 0,
            lambda _: jnp.sum(max_before_stasis_end),
            None,
        )

        return self.contributing_species


############################################################################################################
### Wrapper functions to check differentiability.
############################################################################################################
@jit
def stasis_wrapper(Omega_0, Gamma_0, H_0=1.0, log_transform=False):
    sim = StasisSolver(Omega_0, Gamma_0, H_0, log_transform=log_transform)
    stasis_val, _ = sim.return_stasis()
    return stasis_val


@jit
def abundance_wrapper(Omega_0, Gamma_0, H_0=1.0, log_transform=False):
    sim = StasisSolver(Omega_0, Gamma_0, H_0, log_transform=log_transform)
    _, asymptote_val = sim.return_stasis()
    return asymptote_val


def main():
    # test_omega = jnp.sort(10 ** (np.random.uniform(-2, 0, 30)))  ## test abundances
    # test_gamma = jnp.sort(10 ** (np.random.uniform(-62, 0, 30)))  ## test decay rates
    
    np.random.seed(0)
    test_omega = 10**(np.sort(np.random.uniform(-2, 0 , 30)))
    test_gamma = 10**(np.sort(np.random.uniform(-2, 0 , 30)))

    stasis_fn = value_and_grad(stasis_wrapper, argnums=(0, 1))
    stasis_val, (stasis_grad_omega, stasis_grad_gamma) = stasis_fn(
        test_omega, test_gamma
    )

    abundance_fn = value_and_grad(abundance_wrapper, argnums=(0, 1))
    asymptote_val, (abundance_grad_omega, abundance_grad_gamma) = abundance_fn(
        test_omega, test_gamma
    )

    print("Stasis e-folds", stasis_val)
    print("Abundance value", asymptote_val)
    print("Stasis gradient w.r.t. Omega_0", stasis_grad_omega)
    print("Stasis gradient w.r.t. Gamma_0", stasis_grad_gamma)
    print("Abundance gradient w.r.t. Omega_0", abundance_grad_omega)
    print("Abundance gradient w.r.t. Gamma_0", abundance_grad_gamma)

    ### if gradients are non-zero, things are differentiable :)


if __name__ == "__main__":
    main()
