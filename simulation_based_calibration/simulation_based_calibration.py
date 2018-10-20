"""Simulation based calibration (Talts et. al. 2018) in PyMC3."""
import itertools
import logging

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
from tqdm import tqdm


class quiet_logging:
    """Turn off logging for certain libraries.

    PyMC3 and theano compile locks are a little noisy when running a bunch of loops.
    """

    def __init__(self, *libraries):
        self.loggers = [logging.getLogger(library) for library in libraries]

    def __call__(self, func):
        def wrapped(cls, *args, **kwargs):
            levels = []
            for logger in self.loggers:
                levels.append(logger.level)
                logger.setLevel(logging.CRITICAL)
            res = func(cls, *args, **kwargs)
            for logger, level in zip(self.loggers, levels):
                logger.setLevel(level)
            return res

        return wrapped


class SBC:
    def __init__(
        self,
        model_func,
        observed_vars,
        num_simulations=1000,
        sample_kwargs=None,
        seed=None,
    ):
        """Set up class for doing SBC.

        Note that you must define your model using a function so the observations
        can change on each run, and the keyword arguments of the function must
        match the observed variables.

        You should also specify the shape of the actual observations! See the example.

        Example
        -------
        def my_model(y=None):
            with pm.Model() as model:
                x = pm.Normal('x')
                obs_y = pm.Normal('y', mu=2 * x, observed=y, shape=2)
            return model

        sbc = SBC(my_model, 'y', num_simulations=1000)
        sbc.run_simulations()
        sbc.plot_sbc()

        Parameters
        ----------
        model_func : function
            A function whose keyword arguments are `observed_vars` and which
            returns a pymc3.Model instance
        observed_vars : list[str]
            A list of the observed variables in the model
        num_simulations : int
            How many simulations to run
        sample_kwargs : dict[str] -> Any
            Arguments passed to pymc3.sample
        seed : int (optional)
            Random seed. This persists even if running the simulations is
            paused for whatever reason.
        """
        self.model_func = model_func

        if isinstance(observed_vars, str):
            observed_vars = [observed_vars]
        self.observed_vars = observed_vars
        self.num_simulations = num_simulations

        test_point = self._get_prior_predictive_samples()
        self.var_names = [v for v in test_point if v not in self.observed_vars]

        if sample_kwargs is None:
            sample_kwargs = {}
        sample_kwargs.setdefault("progressbar", False)
        sample_kwargs.setdefault("compute_convergence_checks", False)
        self.sample_kwargs = sample_kwargs

        self.simulations = {name: [] for name in self.var_names}
        self._simulations_complete = 0
        self._seed = seed
        self._warnings = {}

    def _get_seeds(self):
        """Set the random seed, and generate seeds for all the simulations."""
        if self._seed is not None:
            np.random.seed(self._seed)
        return np.random.randint(2 ** 30, size=self.num_simulations)

    def _get_prior_predictive_samples(self):
        """Generate samples to use for the simulations."""
        with self.model_func(**{v: None for v in self.observed_vars}):
            prior = pm.sample_prior_predictive(self.num_simulations)
        return prior

    @quiet_logging("pymc3", "theano.gof.compilelock")
    def run_simulations(self):
        """Run all the simulations.

        This function can be stopped and restarted on the same instance, so you can
        keyboard interrupt part way through, look at the plot, and then resume. If a
        seed was passed initially, it will still be respected (that is, the resulting
        simulations will be identical to running without pausing in the middle).
        """
        seeds = self._get_seeds()
        prior = self._get_prior_predictive_samples()

        progress = tqdm(
            initial=self._simulations_complete,
            total=self.num_simulations,
            postfix=self._warnings,
        )
        try:
            while self._simulations_complete < self.num_simulations:
                idx = self._simulations_complete
                prior_predictive_draw = {v: prior[v][idx] for v in self.observed_vars}
                np.random.seed(seeds[idx])
                with self.model_func(**prior_predictive_draw):
                    check = pm.sample(**self.sample_kwargs)
                for name in self.var_names:
                    self.simulations[name].append(
                        (check[name] < prior[name][idx]).sum(axis=0)
                    )
                self._simulations_complete += 1
                self._update_progress_bar(check, progress)
        finally:
            self.simulations = {
                k: v[: self._simulations_complete] for k, v in self.simulations.items()
            }
            progress.close()

    def _update_progress_bar(self, check, progress):
        """Helper to pipe PyMC3 warnings into the progress bar."""
        for w in check._report._warnings:
            if w.level != "debug":
                name = str(w.kind)
                if name not in self._warnings:
                    self._warnings[name] = 0
                self._warnings[name] += 1
        progress.set_postfix(self._warnings, refresh=False)
        progress.update()

    def plot_sbc(self, var_names=None, plot_kwargs=None):
        """Produce plots similar to those in the SBC paper."""
        return plot_sbc(self.simulations, var_names=var_names, plot_kwargs=plot_kwargs)


def plot_sbc(simulations, var_names=None, plot_kwargs=None):
    """Produce plots similar to those in the SBC paper.

    The data is pretty easy to serialize, and this function makes it
    easier to do that and still produce plots.

    Parameters
    ----------
    simulations : dict[str] -> listlike
        The SBC.simulations dictionary.
    var_names : list[str]
        Variables to plot (defaults to all)
    plot_kwargs : dict[str] -> Any
        Keyword arguments passed to plt.bar

    Returns
    -------
    fig, axes
        matplotlib figure and axes
    """
    if plot_kwargs is None:
        plot_kwargs = {}
    plot_kwargs.setdefault("bins", "auto")
    plot_kwargs.setdefault("color", "#B00A22")  # stan red
    plot_kwargs.setdefault("edgecolor", "black")

    if var_names is None:
        var_names = list(simulations.keys())

    sims = {}
    for k in var_names:
        ary = np.array(simulations[k])
        while ary.ndim < 2:
            ary = np.expand_dims(ary, -1)
        sims[k] = ary

    n_plots = sum(np.prod(v.shape[1:]) for v in sims.values())

    fig, axes = plt.subplots(nrows=n_plots, figsize=(12, 4 * n_plots))

    idx = 0
    for var_name, var_data in sims.items():
        plot_idxs = list(itertools.product(*(np.arange(s) for s in var_data.shape[1:])))
        if len(plot_idxs) > 1:
            has_dims = True
        else:
            has_dims = False

        for indices in plot_idxs:
            if has_dims:
                dim_label = f'{var_name}[{"][".join(map(str, indices))}]'
            else:
                dim_label = var_name
            ax = axes[idx]
            ary = var_data[(...,) + indices]
            ax.hist(ary, **plot_kwargs)
            ax.set_title(dim_label)
            idx += 1
    return fig, axes
