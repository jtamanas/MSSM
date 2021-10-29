import jax
import jax.numpy as np
import numpy as onp

import sys

sys.path.append("/media/jt/data/Projects/MSSM/micromegas_5.2.7.a/MSSM/")
from pymicromegas import MicromegasSettings, SugraParameters
from spheno import spheno
from softsusy import softsusy

micromegas = {"spheno": spheno, "softsusy": softsusy}


settings = MicromegasSettings(
    relic_density=True,
    masses=True,
    gmuon=True,
    bsg=True,
    bsmumu=True,
    btaunu=True,
    delta_rho=True,
    sort_odd=True,
    fast=True,
    beps=0.0001,
    cut=0.01,
)



def theta_addunits(unitless_theta):
    theta = onp.zeros_like(unitless_theta)
    # WARNING: changes unitless_theta in place
    theta[:, 0] = 1000 * unitless_theta[:, 0] * 10.0
    theta[:, 1] = 1000 * unitless_theta[:, 1] * 10.0
    theta[:, 2] = 12 * unitless_theta[:, 0] - 6 * unitless_theta[:, 0]
    theta[:, 3] = 48.5 * unitless_theta[:, 3] + 1.5
    theta = onp.hstack((theta, onp.ones((len(theta), 1)))) # positive sign mu
    return theta



def get_simulator(micromegas_simulator=None, **kwargs):
    """
    Parameters:
    -----------
    micromegas_simulator: str
        Name of the micromegas simulator to use.
        One of ["spheno", "softsusy"]

    Returns:
        simulator: a function that takes a state and returns a new state
        obs_dim: the dimension of the observation space
        theta_dim: the dimension of the theta space
    """

    def simulator(rng, theta, num_samples_per_theta=1):
        """
        Parameters:
        -----------
        rng: jax rng object
        theta: cMSSM parameters
        num_samples_per_theta: int  # number of samples to take per theta
        
        Returns:
            observables
        """
        theta = theta_addunits(theta)
        params = [SugraParameters(*th) for th in theta]
        results = _simulator(params=params, settings=settings)
        out = onp.array([results.omega(), results.gmuon(), results.mhsm()])
        out = out.T
        return out

    if micromegas_simulator is None:
        micromegas_simulator = "spheno"

    obs_dim = 3
    theta_dim = 4
    _simulator = micromegas[micromegas_simulator]

    return simulator, obs_dim, theta_dim


