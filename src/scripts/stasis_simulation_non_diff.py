from typing import Any, List, Optional, Union

import jax
import jax.lax as lax
import matplotlib.pyplot as plt
import numpy as np
from diffrax import (
    DiscreteTerminatingEvent,
    Kvaerno5,
    ODETerm,
    PIDController,
    SaveAt,
    diffeqsolve,
)
from jax import jit
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)  ## Very important for numerical stability


class NonDiffStasisSolver:
    """
    Non-differentiable stasis simulation. This is compatible with both power-law and exponential models of stasis.
    For power-law, specify the following parameters:
        alpha, gamma, delta, delta_m_m0, N, Gamma_N_1_H_0 (see: https://arxiv.org/abs/2111.04753).
    For exponential, specify the following parameters:
        alpha, gamma, Gamma_N, N, H_0.

    You can also specify the initial conditions Omega_0 and Gamma_0 by passing them directly as parameters. Omega_0 will get properly normalized.
    The Boltzmann equations will be solved upon calling the return_stasis() method.

    You can view the potential stasis period by calling the plot_abundance() method, from which the number of stasis e-folds and the asymptotic value of Omega_M can be obtained.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        H_0: float = 1,
        t_max: float = 4000,
        max_steps: int = 4096,
        n_points: int = 40000,
        epsilon: float = 0.10,
        Omega_0: Optional[Union[jnp.ndarray, List[float]]] = None,
        Gamma_0: Optional[Union[jnp.ndarray, List[float]]] = None,
        **kwargs: Any,
    ):
        if model not in ["power-law", "exponential", None]:
            raise ValueError("Invalid model type. Choose 'power-law', 'exponential', or None.")

        self.Omega_0 = Omega_0
        self.Gamma_0 = Gamma_0
        self.H_0 = H_0
        self.model = model

        self.t_max = t_max
        self.max_steps = max_steps
        self.n_points = n_points
        self.epsilon = epsilon

        if model is not None:
            if model == "power-law":
                required_params = [
                    "alpha",
                    "gamma",
                    "delta",
                    "delta_m_m0",
                    "N",
                    "Gamma_N_1_H_0",
                ]
            elif model == "exponential":
                required_params = ["alpha", "gamma", "Gamma_N", "N", "H_0"]

            for key, value in kwargs.items():
                if key not in required_params:
                    raise ValueError(f"Invalid parameter '{key}' for model '{model}'")
                setattr(self, key, value)  # Dynamically set the parameter as an attribute

            for param in required_params:
                if not hasattr(self, param):
                    setattr(self, param, None)

            missing_params = [
                param for param in required_params if getattr(self, param, None) is None
            ]
            if missing_params:
                raise ValueError(
                    f"Missing required parameters for model '{model}': {', '.join(missing_params)}"
                )

        self.t_span = (0, jnp.exp(self.t_max))
        self.t_eval = jnp.exp(jnp.linspace(0, self.t_max, self.n_points))

        if Gamma_0 is not None:
            self.N = len(Gamma_0)
            assert Gamma_0.shape == (self.N,), "Gamma_0 must be an array of length N"
        if Omega_0 is not None:
            self.N = len(Omega_0)
            assert Omega_0.shape == (self.N,), "Omega_0 must be an array of length N"
            self.Omega_0 = Omega_0 / jnp.sum(Omega_0)

        if Omega_0 is None and Gamma_0 is None:
            self.Gamma_0, self.Omega_0, self.y0 = self.initial_conditions()
        else:
            self.y0 = jnp.concatenate([jnp.array([self.H_0]), self.Omega_0])

    def __getattribute__(self, __name: str):
        return super().__getattribute__(__name)

    def __len__(self):
        return self.N

    def initial_conditions(self):
        if self.model == "power-law":
            species_list = jnp.linspace(0, self.N - 1, self.N)
            ml_m0 = 1 + self.delta_m_m0 * species_list**self.delta
            ml_m0 = jnp.array(ml_m0)
            self.Omega_0 = ml_m0**self.alpha / jnp.sum(ml_m0**self.alpha)

            num = (self.H_0) * (ml_m0) ** self.gamma
            denom = (
                self.Gamma_N_1_H_0**-1 * (1 + (self.N - 1) ** self.delta) ** self.gamma
            )
            self.Gamma_0 = jnp.float64(num) / jnp.float64(denom)

        if self.model == "exponential":
            species_list = jnp.linspace(1, self.N, self.N)
            self.Omega_0 = jnp.exp(self.alpha * (species_list - self.N)) / jnp.sum(
                jnp.exp(self.alpha * (species_list - self.N))
            )
            self.Gamma_0 = self.Gamma_N * jnp.exp(self.gamma * (species_list - self.N))

        self.y0 = jnp.concatenate([jnp.array([self.H_0]), self.Omega_0])

        return self.Gamma_0, self.Omega_0, self.y0

    def solve_boltzmann_eq(self):
        @jit
        def boltzmann_eq(t, y, args):
            H, Omega = y[0], y[1:]
            Omega_m = jnp.sum(Omega)
            dH_dt = (-1 / 2) * H**2 * (4 - Omega_m)
            dOmega_dt = H * Omega - self.Gamma_0 * Omega - H * Omega * Omega_m

            return jnp.concatenate((jnp.array([dH_dt]), jnp.array(dOmega_dt)))

        def condition_fn(state, **kwargs):
            del kwargs
            OMEGA_M_MIN = 1e-8
            Omega_m = jnp.sum(state.y[1:])
            condition = Omega_m < OMEGA_M_MIN
            return lax.cond(condition, lambda _: True, lambda _: False, None)

        def configure_solver(t_eval, condition_fn):
            equation = ODETerm(boltzmann_eq)
            solver = Kvaerno5()
            saveat = SaveAt(ts=t_eval)
            stepsize_controller = PIDController(rtol=1e-8, atol=1e-8)
            event = DiscreteTerminatingEvent(cond_fn=condition_fn)
            return equation, solver, saveat, stepsize_controller, event

        (equation, solver, saveat, stepsize_controller, event) = configure_solver(
            self.t_eval, condition_fn
        )

        @jit
        def return_sol():
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
                discrete_terminating_event=event,
                throw=False,
            )
            return sol

        self.sol = return_sol()
        return self.sol

    def return_stasis(self):
        @jit
        def get_total_e_folds(H_sol, ts, a_initial=1.0):
            ln_a_initial = jnp.log(a_initial)
            H_sol = jnp.where(jnp.isfinite(H_sol), H_sol, 0.0)
            ts = jnp.where(jnp.isfinite(ts), ts, 0.0)

            total_ln_a_change = jnp.trapezoid(H_sol, ts)
            final_ln_a = ln_a_initial + total_ln_a_change
            final_a = jnp.exp(final_ln_a)
            N_efolds = jnp.log(final_a / a_initial)

            return N_efolds

        @jit
        def get_scale_factor(H_sol, ts, a_initial=1.0):
            ln_a_initial = jnp.log(a_initial)

            dt = jnp.diff(ts)
            H_midpoints = (H_sol[:-1] + H_sol[1:]) / 2.0
            ln_a_changes = jnp.concatenate(
                [jnp.array([0.0]), jnp.cumsum(H_midpoints * dt)]
            )
            ln_a_t = ln_a_initial + ln_a_changes
            a_t = jnp.exp(ln_a_t)

            return a_t

        self.solve_boltzmann_eq()

        try:
            valid_indices = ~jnp.isnan(jnp.abs(jnp.log(self.sol.ts))) & ~jnp.isinf(
                jnp.abs(jnp.log(self.sol.ts))
            )
            self.ts = jnp.abs(self.sol.ts)[valid_indices]
            self.H_sol = jnp.array(self.sol.ys[:, 0])[valid_indices]
            self.species_sol = jnp.array(self.sol.ys[:, 1:])[valid_indices].T
            self.total_abundance = jnp.sum(self.species_sol, axis=0)
            self.max_e_folds = get_total_e_folds(self.H_sol, self.ts)
            self.scale_factor = get_scale_factor(self.H_sol, self.ts)
            self.original_t_efolds = jnp.abs(jnp.log(self.sol.ts))[valid_indices]
            self.t_efolds = self.max_e_folds * (
                self.original_t_efolds / self.original_t_efolds[-1]
            )

        except Exception as e:
            self.species_sol = jnp.zeros((self.N - 1, self.n_points))
            self.t_efolds = jnp.zeros(self.n_points)
            self.total_abundance = jnp.zeros(self.n_points)
            self.H_sol = jnp.zeros(self.n_points)
            print(f"An exception occurred: {e}")

        longest_stasis_length = 0
        longest_stasis_start = None
        longest_stasis_end = None
        self.asymptote = None

        current_interval_start = None
        for i in range(len(self.total_abundance)):
            if current_interval_start is None:
                if self.total_abundance[i] <= 0.01 or self.total_abundance[i] >= (
                    1 - 0.01
                ):
                    continue
                else:
                    current_interval_start = i
            else:
                if (
                    abs(
                        self.total_abundance[i]
                        - self.total_abundance[current_interval_start]
                    )
                    <= self.epsilon
                ):
                    if (i - current_interval_start) > longest_stasis_length:
                        longest_stasis_length = i - current_interval_start
                        longest_stasis_start = current_interval_start
                        longest_stasis_end = i
                else:
                    current_interval_start = None

        if longest_stasis_start is not None and longest_stasis_end is not None:
            self.asymptote = jnp.mean(
                self.total_abundance[longest_stasis_start : longest_stasis_end + 1]
            )

            self.stasis_begin = self.t_efolds[longest_stasis_start]
            self.stasis_end = self.t_efolds[longest_stasis_end]
            self.len_stasis = self.stasis_end - self.stasis_begin
        else:
            self.stasis_begin, self.stasis_end, self.len_stasis = 0, 0, 0

        if self.asymptote > 0.99:
            self.stasis_begin, self.stasis_end = 0, 0
            self.len_stasis = 0
        if self.asymptote < 0.01:
            self.stasis_begin, self.stasis_end = 0, 0
            self.len_stasis = 0

        return self.len_stasis, self.asymptote

    def plot_scale_factor_efolds(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(
            self.t_efolds, self.scale_factor, color="black", label="Scale Factor $a(t)$"
        )

        N = np.array(self.t_efolds)
        a_rad = np.exp(2 * N)  # Radiation-dominated
        a_matter = np.exp(1.5 * N)  # Matter-dominated

        ax.plot(
            N,
            a_rad,
            color="blue",
            linestyle="--",
            label="Radiation-Dominated ($a \propto e^{2\mathcal{N}}$)",
        )
        ax.plot(
            N,
            a_matter,
            color="green",
            linestyle="--",
            label="Matter-Dominated ($a \propto e^{3/2\mathcal{N}}$)",
        )

        omega_m_bar = self.asymptote  # Assuming omega_m_bar is available
        kappa = 2 / (4 - omega_m_bar)
        a_stasis = 1 * np.exp(kappa**-1 * N) ** (kappa)

        stasis_begin = self.stasis_begin  # Replace with actual value
        stasis_end = self.stasis_end  # Replace with actual value
        ax.axvspan(
            stasis_begin,
            stasis_end,
            color="lightgray",
            alpha=0.5,
            label="Stasis Period",
        )

        ax.plot(
            N,
            a_stasis,
            color="purple",
            linestyle="--",
            label="Theoretical ($a \propto e^{\\frac{1}{\\kappa} \\mathcal{N} \\kappa}$)",
        )
        ax.axhline(y=1, color="red", linestyle="--", label="$a(t_0) = 1$")
        ax.set_xticks(np.arange(0, self.t_efolds[-1], 5))
        ax.set_xlim(self.t_efolds[0], self.t_efolds[-1])
        ax.set_yscale("log")
        ax.set_xlabel("$\\mathcal{N}$", fontsize=12)
        ax.set_ylabel("$a$", fontsize=12)
        ax.legend()
        ax.set_title(r"Scale Factor $a$, $a(t_0) = 1$", fontsize=12)

    def compute_species_contribution(self):
        # Determine indices corresponding to stasis_begin and stasis_end in the t_efolds array
        stasis_begin_index = jnp.searchsorted(
            self.t_efolds, self.stasis_begin, side="left"
        )
        stasis_end_index = jnp.searchsorted(
            self.t_efolds, self.stasis_end, side="right"
        )

        # Find the index of the maximum value for each species
        max_indices = jnp.argmax(self.species_sol, axis=1)

        # Check if the max occurs before stasis or after stasis for each species
        max_before_stasis_end = max_indices < stasis_end_index

        # Species that contribute to stasis are those that have their max after stasis_begin and before stasis_end
        # contributing_species = jnp.sum(~max_before_stasis & ~max_after_stasis)
        self.contributing_species = jnp.sum(max_before_stasis_end)

        if self.len_stasis < 2:
            self.contributing_species = 0

        return self.contributing_species

    def plot_abundance(self, ax=None, logy=False, taus=False):
        if ax is None:
            fig, ax = plt.subplots()
        if logy:
            ax.set_yscale("log")

        cmap = plt.colormaps.get_cmap("plasma")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=self.N))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)  # Pass the ax object here
        cbar.set_label(r"$\ell$", fontsize=14)
        # cbar.ax.tick_params(labelsize=50)  # Increase cbar tick label size here

        for i in range(self.N - 1):
            color = cmap(i / self.N)
            ax.plot(
                self.t_efolds, self.species_sol[i, :], label="_nolegend_", color=color
            )

            if i < 5 and taus:
                ax.axvline(
                    x=self.t_efolds[np.argmax(self.species_sol[i, :])],
                    color=color,
                    linestyle="--",
                    alpha=0.2,
                    label="_nolegend_",
                )

        ax.plot(self.t_efolds, self.total_abundance, color="red", label="$\\Omega_M$")

        if np.round(self.asymptote, 2) > 0:
            ax.axhline(
                y=self.asymptote,
                color="red",
                linestyle="--",
                label="$\\overline{\\Omega}_{M}$",
            )

        if self.len_stasis > 0:
            ax.scatter(
                self.stasis_begin,
                self.asymptote,
                marker="*",
                color="black",
                s=50,
                zorder=10,
            )
            ax.scatter(
                self.stasis_end,
                self.asymptote,
                marker="*",
                color="black",
                s=50,
                zorder=10,
            )

        ax.set_xlim(self.t_efolds[0], self.t_efolds[-1])
        ax.set_ylim(1e-4, 1.1 * self.total_abundance[0])
        ax.set_xticks(np.arange(0, self.t_efolds[-1], 5))
        ax.set_xlabel("$\\mathcal{N}$", fontsize=14)
        ax.set_ylabel("$\\Omega$", fontsize=14)

        if self.Omega_0 is not None:
            ax.set_title(
                f"{self.N} Species, $H_0$ = {self.H_0}, $\\overline{{\\Omega}}_M$ = {self.asymptote:.2f}, {self.len_stasis:.2f} e-folds of stasis",
                fontsize=15,
            )
        else:
            ax.set_title(
                f"{self.N} Species, $H_0$ = {self.H_0}, $\\overline{{\\Omega_m}}$ = {self.asymptote:.2f}, {self.len_stasis:.2f} $e$-folds of stasis, $\\alpha$ = {self.alpha}, $\\gamma$ = {self.gamma}, $\\delta$ = {self.delta}",
                fontsize=15,
            )

        ax.legend()

        return cbar

    def plot_H_efolds(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.t_efolds, self.H_sol, color="black", label="Hubble Parameter")
        ax.axhline(y=self.H_0, color="red", linestyle="--", label="$H_0$")

        N = np.array(self.t_efolds)
        H_rad = self.H_0 * np.exp(-2 * (N - N[0]))  # Radiation-dominated
        H_matter = self.H_0 * np.exp(-1.5 * (N - N[0]))  # Matter-dominated

        ax.plot(
            N,
            H_rad,
            color="blue",
            linestyle="--",
            label="Radiation-Dominated ($H \propto e^{-2\mathcal{N}}$)",
        )
        ax.plot(
            N,
            H_matter,
            color="green",
            linestyle="--",
            label="Matter-Dominated ($H \propto e^{-3/2\mathcal{N}}$)",
        )

        stasis_begin = self.stasis_begin  # Replace with actual value
        stasis_end = self.stasis_end  # Replace with actual value
        ax.axvspan(
            stasis_begin,
            stasis_end,
            color="lightgray",
            alpha=0.5,
            label="Stasis Period",
        )

        Omega_M_mean = self.asymptote  # example value
        H_theoretical = (
            2 / (4 - Omega_M_mean) * np.exp(-(4 - Omega_M_mean) / 2 * (N - N[0]))
        )

        ax.plot(
            N,
            H_theoretical,
            color="purple",
            linestyle="--",
            label="Theoretical ($H \propto \\frac{\kappa}{3 t_0} e^{-\\frac{3}{\kappa} \\mathcal{N}}$)",
        )
        ax.set_yscale("log")
        # ax.set_xticks(np.arange(0, self.t_efolds[-1], 5))
        ax.set_xlim(self.t_efolds[0], self.t_efolds[-1])
        ax.set_xlabel("$\\mathcal{N}$", fontsize=12)
        ax.set_ylabel("$H$", fontsize=12)
        ax.set_title(f"Hubble Parameter, $H^{(0)}$={self.H_0}", fontsize=12)
        ax.legend()

    def plot_scale_factor_t(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        H_sol = self.H_sol
        dt = np.diff(self.t_efolds) / H_sol[:-1]  # Differential time steps
        t = np.concatenate([[0], np.cumsum(dt)])  # Integrate to get time t

        ax.plot(t, self.scale_factor, color="black", label="Scale Factor $a(t)$")

        t_rad = t
        a_rad = (t_rad - t_rad[0]) ** (1 / 2)  # Radiation-dominated
        t_matter = t
        a_matter = (t_matter - t_matter[0]) ** (2 / 3)  # Matter-dominated

        ax.plot(
            t_rad,
            a_rad,
            color="blue",
            linestyle="--",
            label="Radiation-Dominated ($a \propto t^{1/2}$)",
        )
        ax.plot(
            t_matter,
            a_matter,
            color="green",
            linestyle="--",
            label="Matter-Dominated ($a \propto t^{2/3}$)",
        )

        omega_m_bar = self.asymptote  # Assuming omega_m_bar is available
        kappa = 2 / (4 - omega_m_bar)
        a_stasis = 1 * np.exp(kappa**-1 * self.t_efolds) ** (kappa)
        t_stasis = np.concatenate(
            [[0], np.cumsum(np.diff(self.t_efolds) / (kappa * H_sol[:-1]))]
        )

        ax.plot(
            t_stasis,
            a_stasis,
            color="purple",
            linestyle="--",
            label="Theoretical ($a \propto t^{\\kappa}$)",
        )

        ax.axhline(y=1, color="red", linestyle="--", label="$a(t_0) = 1$")

        stasis_begin = self.stasis_begin  # Replace with actual value
        stasis_end = self.stasis_end  # Replace with actual value
        t_stasis_begin = np.interp(stasis_begin, self.t_efolds, t)
        t_stasis_end = np.interp(stasis_end, self.t_efolds, t)
        ax.axvspan(
            t_stasis_begin,
            t_stasis_end,
            color="lightgray",
            alpha=0.5,
            label="Stasis Period",
        )

        ax.set_xticks(np.linspace(0, t[-1], num=10))
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("$t$", fontsize=12)
        ax.set_ylabel("$a$", fontsize=12)
        ax.legend()
        ax.set_title(r"Scale Factor $a$, $a(t_0) = 1$", fontsize=12)

    def plot_H_t(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        H_sol = self.H_sol
        dt = np.diff(self.t_efolds) / H_sol[:-1]  # Differential time steps
        t = np.concatenate([[0], np.cumsum(dt)])

        Omega_M_mean = self.asymptote  # example value
        kappa = 6 / (4 - Omega_M_mean)

        t_stasis = np.exp(3 * kappa**-1 * (self.t_efolds - self.t_efolds[0]))

        # Plot the computed Hubble parameter
        ax.plot(t_stasis, self.H_sol, color="black", label="Hubble Parameter")
        ax.axhline(y=self.H_0, color="red", linestyle="--", label="$H_0$")

        t_rad = t
        H_rad = 1 / (2 * t + 1e-2)  # Radiation-dominated
        t_matter = t
        H_matter = 2 / (3 * t + 1e-2)  # Matter-dominated

        ax.plot(
            t_rad,
            H_rad,
            color="blue",
            linestyle="--",
            label="Radiation-Dominated ($H \propto \\frac{1}{2t}$)",
        )
        ax.plot(
            t_matter,
            H_matter,
            color="green",
            linestyle="--",
            label="Matter-Dominated ($H \propto \\frac{2}{3t}$)",
        )

        stasis_begin = self.stasis_begin
        stasis_end = self.stasis_end  # Replace with actual value
        t_stasis_begin = np.interp(stasis_begin, self.t_efolds, t)
        t_stasis_end = np.interp(stasis_end, self.t_efolds, t)
        ax.axvspan(
            t_stasis_begin,
            t_stasis_end,
            color="lightgray",
            alpha=0.5,
            label="Stasis Period",
        )

        Omega_M_mean = self.asymptote
        H_theoretical = 2 / (4 - Omega_M_mean) * (t) ** -1

        ax.plot(
            t,
            H_theoretical,
            color="purple",
            linestyle="--",
            label="Theoretical ($H \propto \\frac{\kappa}{3 t} $)",
        )
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("$t$", fontsize=12)
        ax.set_ylabel("$H$", fontsize=12)
        ax.set_title(f"Hubble Parameter, $H^{(0)}$={self.H_0}", fontsize=12)
        ax.legend()


if __name__ == "__main__":
    ### test case for power-law model
    power_law_stasis = NonDiffStasisSolver(
        model="power-law",
        N=100,
        H_0=1,
        alpha=1,
        gamma=7,
        delta=1,
        delta_m_m0=1,
        Gamma_N_1_H_0=0.01,
    )
    stasis_val_power_law, asymptote_val_power_law = power_law_stasis.return_stasis()
    print(stasis_val_power_law, asymptote_val_power_law)
    ### results in MRE with 13 e-folds

    ### test case for exponential model
    exponential_stasis = NonDiffStasisSolver(
        model="exponential", N=100, H_0=1, alpha=(2/7), gamma=1, Gamma_N=0.01
    )
    stasis_val_exponential, asymptote_val_exponential = exponential_stasis.return_stasis()
    print(stasis_val_exponential, asymptote_val_exponential)
    ## results in MRE with 50 e-folds
