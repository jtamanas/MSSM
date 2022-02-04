import jax
import jax.numpy as np
import numpy as onp

import sys

from utils.distributed import apply_distributed
from pymicromegas import MicromegasSettings, SugraParameters, spheno, softsusy

micromegas = {"spheno": spheno, "softsusy": softsusy}


def theta_addunits(unitless_theta):
    unitless_theta = onp.atleast_2d(unitless_theta)
    theta = onp.zeros_like(unitless_theta)
    # WARNING: changes unitless_theta in place
    theta[:, 0] = 1000 * unitless_theta[:, 0] * 10.0
    theta[:, 1] = 1000 * unitless_theta[:, 1] * 10.0
    theta[:, 2] = 12 * unitless_theta[:, 0] - 6 * unitless_theta[:, 0]
    theta[:, 3] = 48.5 * unitless_theta[:, 3] + 1.5
    theta = onp.hstack((theta, onp.ones((len(theta), 1))))  # positive sign mu
    return theta


def get_simulator(micromegas_simulator=None, preprocess=None, **kwargs):
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
    global simulator

    if preprocess is None:
        preprocess = lambda x: x

    def simulator(unitless_theta):
        """
        Parameters:
        -----------
        rng: jax rng object
        theta: cMSSM parameters

        Returns:
            observables
        """
        theta = theta_addunits(unitless_theta)
        params = [SugraParameters(*th) for th in theta]
        results = _simulator(params=params)
        out = onp.array([results.omega, results.gmuon, results.mhsm])
        out = out.T
        out = preprocess(out)
        return out

    if micromegas_simulator is None:
        micromegas_simulator = "spheno"

    obs_dim = 3
    theta_dim = 4
    _simulator = micromegas[micromegas_simulator]

    distributed_simulator = (
        lambda rng, args, num_samples_per_theta=1: apply_distributed(
            simulator, args, nprocs=10
        )
    )

    return distributed_simulator, obs_dim, theta_dim

if __name__ == "__main__":
    
    test_theta = np.array([0.5] * 4)
    test_theta = np.stack([test_theta, np.array([0.25] * 4)])
    theta = theta_addunits(test_theta)
    print(theta)
    params = [SugraParameters(*th) for th in theta]
    results = micromegas['spheno'](params=params)#, settings=settings)
    out = onp.array([results.omega, results.gmuon, results.mhsm])
    out = out.T
    print(out)