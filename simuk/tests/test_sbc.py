import numpy as np
import pymc as pm

import simuk

data = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

with pm.Model() as centered_eight:
    obs = pm.MutableData("obs", data)
    mu = pm.Normal("mu", mu=0, sigma=5)
    tau = pm.HalfCauchy("tau", beta=5)
    theta = pm.Normal("theta", mu=mu, sigma=tau, shape=8)
    y_obs = pm.Normal("y", mu=theta, sigma=sigma, observed=obs)


def test_sbc():
    sbc = simuk.SBC(
        centered_eight,
        num_simulations=100,
        sample_kwargs={"draws": 25, "tune": 50},
    )
    sbc.run_simulations()
    sbc.plot_results()
