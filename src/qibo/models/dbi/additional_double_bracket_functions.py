from copy import deepcopy
from enum import Enum, auto
from functools import partial
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hyperopt import hp, tpe

from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian
from qibo.models.dbi.double_bracket import (
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
)
from qibo.symbols import I, X, Z


class DoubleBracketStrategyType(Enum):
    """Determines how to variationally choose the diagonal operator"""

    local_Z_search = auto()
    """Search through Z permutations"""
    gradient_descent_search = auto()
    """Search by gradient descent (magnetic field)"""
    # TODO: how to input list
    prescribed_search = auto()
    """Search within a prescribed list of operators"""


class DoubleBracketIterationStrategies(DoubleBracketIteration):
    def __init__(
        self,
        hamiltonian: Hamiltonian,
        NSTEPS: int = 5,
        please_be_verbose: bool = True,
        please_use_hyperopt: bool = True,
        mode: DoubleBracketGeneratorType = DoubleBracketGeneratorType.canonical,
        variation_strategy: DoubleBracketStrategyType = DoubleBracketStrategyType.local_Z_search,
    ):
        super().__init__(hamiltonian, mode)
        self.NSTEPS = NSTEPS
        self.please_be_verbose = please_be_verbose
        self.please_use_hyperopt = please_use_hyperopt
        self.variation_strategy = variation_strategy

        self.DBI_outputs = {
            "iterated_h": [],
            "iteration_steps": [],
            "off_diagonal_norm_histories": [],
            "energy_fluctuations": [],
        }

    @staticmethod
    def visualize_matrix(matrix, title="", ax=None):
        """Visualize hamiltonian in a heatmap form."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        else:
            fig = ax.figure
        ax.set_title(title)
        try:
            im = ax.imshow(np.absolute(matrix), cmap="inferno")
        except TypeError:
            im = ax.imshow(np.absolute(matrix.get()), cmap="inferno")
        fig.colorbar(im, ax=ax)

    @staticmethod
    def visualize_drift(h0, h):
        """Visualize drift (absolute difference) of the evolved hamiltonian w.r.t. h0."""
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title(r"Drift: $|\hat{H}_0 - \hat{H}_{\ell}|$")
        try:
            im = ax.imshow(np.absolute(h0 - h), cmap="inferno")
        except TypeError:
            im = ax.imshow(np.absolute((h0 - h).get()), cmap="inferno")

        fig.colorbar(im, ax=ax)

    @staticmethod
    def plot_histories(histories, labels):
        """Plot off-diagonal norm histories over a sequential evolution."""
        colors = sns.color_palette("inferno", n_colors=len(histories)).as_hex()
        plt.figure(figsize=(5, 5 * 6 / 8))
        for i, (h, l) in enumerate(zip(histories, labels)):
            plt.plot(h, lw=2, color=colors[i], label=l, marker=".")
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel(r"$\| \sigma(\hat{H}) \|^2$")
        plt.title("Loss function histories")
        plt.grid(True)
        plt.show()

    def double_bracket_rotation(
        self,
        step: float,
        mode: DoubleBracketGeneratorType = None,
        d: np.array = None,
        update_h: bool = False,
    ):
        """ "Computes the iterated hamiltonian after one double bracket iteration (and updates)"""
        if mode is None:
            mode = self.mode

        if mode is DoubleBracketGeneratorType.canonical:
            operator = self.backend.calculate_matrix_exp(
                1.0j * step,
                self.commutator(self.diagonal_h_matrix, self.h.matrix),
            )
        elif mode is DoubleBracketGeneratorType.single_commutator:
            if d is None:
                raise_error(ValueError, f"Cannot use group_commutator with matrix {d}")
            operator = self.backend.calculate_matrix_exp(
                1.0j * step,
                self.commutator(d, self.h.matrix),
            )
        elif mode is DoubleBracketGeneratorType.group_commutator:
            if d is None:
                raise_error(ValueError, f"Cannot use group_commutator with matrix {d}")
            operator = (
                self.h.exp(-step)
                @ self.backend.calculate_matrix_exp(-step, d)
                @ self.h.exp(step)
                @ self.backend.calculate_matrix_exp(step, d)
            )
        operator_dagger = self.backend.cast(
            np.matrix(self.backend.to_numpy(operator)).getH()
        )
        if update_h is True:
            self.h.matrix = operator @ self.h.matrix @ operator_dagger
        return operator @ self.h.matrix @ operator_dagger

    def iterate_forwards_fixed_generator(self, step=0.1):
        """Execute multiple Double Bracket iterations with fixed flow generator"""
        self.store_initial_inputs()
        for s in range(self.NSTEPS):
            if self.please_use_hyperopt is True:
                step = self.hyperopt_step(
                    step_min=1e-5,
                    step_max=1,
                    space=hp.uniform,
                    optimizer=tpe,
                    max_evals=100,
                    verbose=True,
                )
            self.double_bracket_rotation(step, update_h=True)
            self.store_iteration_outputs(step)

            if self.please_be_verbose is True:
                print("try")

    def generate_local_Z_operators(self, L: int = None):
        if L is None:
            L = self.h.nqubits
        combination_strings = product("ZI", repeat=L)
        operator_map = {"Z": Z, "I": I}
        operators = []

        for op_string in combination_strings:
            tensor_op = 1
            # except for the identity
            if "Z" in op_string:
                for qubit, char in enumerate(op_string):
                    if char in operator_map:
                        tensor_op *= operator_map[char](qubit)
                operators.append(tensor_op)
        return operators

    def iterate_forwards_via_local_Z_search(
        self, step: float = 0.1, Z_list: list = [], check_canonical: bool = False
    ):
        """Execute double bracket iterations with the optimal operator form a prescribed list"""
        # prepare iteration
        self.store_initial_inputs()

        # generate local Z operators
        L = self.h.nqubits
        if Z_list is []:
            Z_list = self.generate_local_Z_operators(L)

        # search for best flow generator
        for Z_operator in Z_list:
            # stash values -> min -> store
            iterated_h = self.double_bracket_rotation(
                self.h, step, d=Z_operator, update_h=False
            )
            if check_canonical is True:
                # compare
                print("Check")

    def store_outputs(self, **outputs):
        """Stores ('key', item) or (key = item) as a dictionary"""
        for output_key in outputs:
            if output_key in self.DBI_outputs:
                self.DBI_outputs[output_key].append(outputs[output_key])
            else:
                self.DBI_outputs[output_key] = [outputs[output_key]]

    def store_initial_inputs(self):
        self.store_outputs(
            iterated_h=self.h,
            iteration_steps=0.0,
            off_diagonal_norms=self.off_diagonal_norm,
            energy_fluctuations=self.energy_fluctuation(self.h.ground_state()),
        )

    def store_iteration_outputs(
        self,
        iteration_step: float,
        off_diagonal_norm: float = None,
        energy_fluctuation=None,
    ):
        if off_diagonal_norm is None:
            off_diagonal_norm = self.off_diagonal_norm
        self.store_outputs(
            iterated_h=self.h,
            iteration_steps=iteration_step,
            off_diagonal_norms=off_diagonal_norm,
            energy_fluctuations=energy_fluctuation,
        )

    def visualize_iteration_results(
        self, DBI_outputs=None, cost_function_type="off_diagonal_norm"
    ):
        """a. Plot the cost function wrt to iterations time, default being off diagoanl cost_function_histories
        b. Visualize the initial matrix
        c. Visualize the final iterated matrix
        """
        if DBI_outputs is None:
            DBI_outputs = self.DBI_outputs

        # limit options for cost functions
        cost_function_type_options = ["off_diagonal_norm", "energy_fluctuation"]
        if cost_function_type not in cost_function_type_options:
            raise ValueError(
                f"cost_function_type must be in {cost_function_type_options}"
            )

        f = plt.figure(figsize=(15, 4))
        # a
        ax_a = f.add_subplot(1, 3, 1)
        if cost_function_type == "off_diagonal_norm":
            cost_function_histories = DBI_outputs["off_diagonal_norms"]
            title_a = r"Off-diagonal Norm $\vert\vert\sigma(H_k)\vert\vert$"
        elif cost_function_type == "energy_fluctuation":
            cost_function_histories = DBI_outputs["energy_fluctuations"]
            title_a = r"Energy Fluctuation $\Xi(\mu)$"
            # = \sqrt{\langle \mu | \hat{H}_k^2 | \mu \rangle - \langle \mu | \hat{H}_k | \mu \rangle^2}
        x_axis = [
            sum(DBI_outputs["iteration_steps"][:k])
            for k in range(1, len(DBI_outputs["iteration_steps"]) + 1)
        ]

        plt.plot(x_axis, cost_function_histories, "-o")
        x_labels_rounded = [round(x, 2) for x in x_axis]
        x_labels_rounded = [0] + x_labels_rounded[0:5] + [max(x_labels_rounded)]
        x_labels_rounded.pop(3)
        plt.xticks(x_labels_rounded)

        y_labels_rounded = [round(y, 1) for y in cost_function_histories]
        y_labels_rounded = y_labels_rounded[0:5] + [min(y_labels_rounded)]
        plt.yticks(y_labels_rounded)

        plt.grid()
        plt.xlabel(r"Flow duration $s$")
        plt.title(title_a)

        # panel label
        a = -0.1
        b = 1.05
        plt.annotate("a)", xy=(a, b), xycoords="axes fraction")

        # b
        ax_b = f.add_subplot(1, 3, 2)
        plt.annotate("b)", xy=(a, b), xycoords="axes fraction")
        self.visualize_matrix(self.h0.matrix, "Initial Matrix", ax_b)

        # c
        ax_c = f.add_subplot(1, 3, 3)
        plt.annotate("b)", xy=(a, b), xycoords="axes fraction")
        self.visualize_matrix(self.h.matrix, "Final Matrix", ax_c)

    def hyperopt_iteration_step(
        self,
        step_min: float = 1e-5,
        step_max: float = 1,
        max_evals: int = 1000,
        space: callable = None,
        optimizer: callable = None,
        look_ahead: int = 1,
        verbose: bool = False,
        d: np.array = None,
    ):
        """
        Optimize iteration step.

        Args:
            step_min: lower bound of the search grid;
            step_max: upper bound of the search grid;
            max_evals: maximum number of iterations done by the hyperoptimizer;
            space: see hyperopt.hp possibilities;
            optimizer: see hyperopt algorithms;
            look_ahead: number of iteration steps to compute the loss function;
            verbose: level of verbosity.
            d: diagonal operator for iteration generator.

        Returns:
            (float): optimized best iteration step.
        """
        try:
            import hyperopt
        except:  # pragma: no cover
            raise_error(
                ImportError, "hyperopt_step function requires hyperopt to be installed."
            )

        if space is None:
            space = hyperopt.hp.uniform
        if optimizer is None:
            optimizer = hyperopt.tpe

        space = space("step", step_min, step_max)
        best = hyperopt.fmin(
            fn=partial(self.iteration_loss(d), look_ahead=look_ahead),
            space=space,
            algo=optimizer.suggest,
            max_evals=max_evals,
            verbose=verbose,
        )

        return best["step"]

    def iteration_loss(self, step: float, look_ahead: int = 1, d: np.array = None):
        """
        Compute loss function distance between `look_ahead` steps.

        Args:
            step: iteration step.
            look_ahead: number of iteration steps to compute the loss function;
        """
        # copy initial hamiltonian
        h_copy = deepcopy(self.h)

        for _ in range(look_ahead):
            self.__call__(mode=self.mode, step=step, d=d)

        # off_diagonal_norm's value after the steps
        loss = self.off_diagonal_norm

        # set back the initial configuration
        self.h = h_copy

        return loss
