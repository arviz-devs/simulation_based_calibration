"""Simulation based calibration (Talts et. al. 2018) in PyMC."""
from copy import copy
import logging

import arviz as az
import numpy as np
import pymc as pm

try:
    import bambi as bmb
except ImportError:
    pass

from tqdm import tqdm

from plots import plot_results


class quiet_logging:
    """Turn off logging for PyMC, Bambi and PyTensor."""

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
        model,
        num_simulations=1000,
        sample_kwargs=None,
        seed=None,
    ):
        """Set up class for doing SBC.

        Parameters
        ----------
        model : function
            A PyMC or Bambi model. If a PyMC model the data needs to be defined as
            mutable data.
        num_simulations : int
            How many simulations to run
        sample_kwargs : dict[str] -> Any
            Arguments passed to pymc.sample or bambi.Model.fit
        seed : int (optional)
            Random seed. This persists even if running the simulations is
            paused for whatever reason.

        Example
        -------

        with pm.Model() as model:
            obs = pm.MutableData('obs', data)
            x = pm.Normal('x')
            y = pm.Normal('y', mu=2 * x, observed=obs)
        return model

        sbc = SBC(my_model, ("obs", "y"), num_simulations=1000)
        sbc.run_simulations()
        sbc.plot_results()
        """
        if isinstance(model, pm.Model):
            self.engine = "pymc"
            self.model = model
            self.observed_vars = {model.rvs_to_values[rv].name : rv.name for rv in model.observed_RVs}
        else:
            self.engine = "bambi"
            model.build()
            self.bambi_model = model
            self.model = model.backend.model
            self.formula = model.formula
            self.new_data = copy(model.data)
            self.observed_vars = {
                model.response_component.term.name: model.response_component.term.name
            }

        self.num_simulations = num_simulations

        self.var_names = [v.name for v in self.model.free_RVs]

        if sample_kwargs is None:
            sample_kwargs = {}
        sample_kwargs.setdefault("progressbar", False)
        sample_kwargs.setdefault("compute_convergence_checks", False)
        self.sample_kwargs = sample_kwargs

        self.simulations = {name: [] for name in self.var_names}
        self._simulations_complete = 0
        self._seed = seed

    def _get_seeds(self):
        """Set the random seed, and generate seeds for all the simulations."""
        if self._seed is not None:
            np.random.seed(self._seed)
        return np.random.randint(2**30, size=self.num_simulations)

    def _get_prior_predictive_samples(self):
        """Generate samples to use for the simulations."""
        with self.model:
            idata = pm.sample_prior_predictive(samples=self.num_simulations)
            prior_pred = az.extract(idata, group="prior_predictive")
            prior = az.extract(idata, group="prior")
        return prior, prior_pred

    def _get_posterior_samples(self, prior_predictive_draw):
        """Generate posterior samples conditioned to a prior predictive sample."""
        if self.engine == "pymc":
            with self.model:
                pm.set_data(prior_predictive_draw)
                check = pm.sample(**self.sample_kwargs)
        else:
            for k, v in prior_predictive_draw.items():
                self.new_data[k] = v
            check = bmb.Model(self.formula, self.new_data).fit(**self.sample_kwargs).fit()

        posterior = az.extract(check, group="posterior")
        return posterior

    @quiet_logging("pymc", "pytensor.gof.compilelock", "bambi")
    def run_simulations(self):
        """Run all the simulations.

        This function can be stopped and restarted on the same instance, so you can
        keyboard interrupt part way through, look at the plot, and then resume. If a
        seed was passed initially, it will still be respected (that is, the resulting
        simulations will be identical to running without pausing in the middle).
        """
        seeds = self._get_seeds()
        prior, prior_pred = self._get_prior_predictive_samples()

        progress = tqdm(
            initial=self._simulations_complete,
            total=self.num_simulations,
        )
        try:
            while self._simulations_complete < self.num_simulations:
                idx = self._simulations_complete
                prior_predictive_draw = {
                    k: prior_pred[v].sel(chain=0, draw=idx).values
                    for k, v in self.observed_vars.items()
                }
                np.random.seed(seeds[idx])

                posterior = self._get_posterior_samples(prior_predictive_draw)
                for name in self.var_names:
                    self.simulations[name].append(
                        (posterior[name] < prior[name].sel(chain=0, draw=idx)).sum("sample").values
                    )
                self._simulations_complete += 1
                progress.update()
        finally:
            self.simulations = {
                k: v[: self._simulations_complete] for k, v in self.simulations.items()
            }
            progress.close()

    def plot_results(self, kind="ecdf", var_names=None, color="C0"):
        """Visual diagnostic for SBC.

        Currently it support two options: `ecdf` for the empirical CDF plots
        of the difference between prior and posterior. `hist` for the rank
        histogram.


        Parameters
        ----------
        simulations : dict[str] -> listlike
            The SBC.simulations dictionary.
        kind : str
            What kind of plot to make. Supported values are 'ecdf' (default) and 'hist'
        var_names : list[str]
            Variables to plot (defaults to all)
        figsize : tuple
            Figure size for the plot. If None, it will be defined automatically.
        color : str
            Color to use for the eCDF or histogram

        Returns
        -------
        fig, axes
            matplotlib figure and axes
        """
        return plot_results(self.simulations, kind=kind, var_names=var_names, color=color)