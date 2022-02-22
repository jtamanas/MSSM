import jax
import jax.numpy as np
import numpy as onp

import sys
from utils.distributed import apply_distributed

sys.path.append("/media/jt/data/Projects/MSSM/micromegas/MSSM/")
from pymicromegas import EwsbParameters, SugraParameters, spheno, softsusy, suspect

"""
Ranges taken from Hollingsworth et al. All in TeV
mu : +- (0.1, 4) 
mg1 : +- (0.05, 4)
mg2 : +- (0.1, 4)
mg3 : (0.4, 4)
ml1 : (0.1, 4)
ml2 : (0.1, 4)  # They did not look at 2nd gen slepton masses
ml3 : (0.1, 4)
mr1 : (0.1, 4)
mr2 : (0.1, 4)
mr3 : (0.1, 4)
mq1 : (0.4, 4)
mq2 : (0.4, 4)  # They did not look at 2nd gen squark masses
mq3 : (0.2, 4)
mu1 : (0.4, 4)
mu2 : (0.4, 4)
mu3 : (0.2, 4)
md1 : (0.4, 4)
md2 : (0.4, 4)
md3 : (0.2, 4)
mh3 : (0.1, 4)  # mass of cp odd higgs
tb : (1, 60)
at : +- (0, 4)
ab : +- (0, 4)
al : +- (0, 4)

To handle the plus/minus sign, we'll have the prior go from -1 to 1
and then take the absolute value of everything that is not plus/minus. 

"""


micromegas = {"spheno": spheno, "softsusy": softsusy}


# settings = MicromegasSettings(
#     relic_density=True,
#     masses=True,
#     gmuon=True,
#     bsg=True,
#     bsmumu=True,
#     btaunu=True,
#     delta_rho=True,
#     sort_odd=True,
#     fast=True,
#     beps=0.0001,
#     cut=0.01,
# )



# theta_ranges = [(take_abs_value_in_micromegas, low_in_TeV, high_in_TeV)]
theta_ranges = [
    (False, 0.1, 4),
    (False, 0.05, 4),
    (False, 0.1, 4),
    (True, 0.4, 4),
    (True, 0.1, 4),
    (True, 0.1, 4),  
    (True, 0.1, 4),
    (True, 0.1, 4),
    (True, 0.1, 4),
    (True, 0.1, 4),
    (True, 0.4, 4),
    (True, 0.4, 4),  
    (True, 0.2, 4),
    (True, 0.4, 4),
    (True, 0.4, 4),
    (True, 0.2, 4),
    (True, 0.4, 4),
    (True, 0.4, 4),
    (True, 0.2, 4),
    (True, 0.1, 4), 
    (True, 0.001, 0.060),  # these get multiplied by 1000
    (False, 0, 4),
    (False, 0, 4),
    (False, 0, 4),
]


def theta_addunits(unitless_theta):
    unitless_theta = onp.atleast_2d(unitless_theta)
    theta = onp.zeros_like(unitless_theta)
    for i, (take_abs, low, high) in enumerate(theta_ranges):
        if take_abs:
            theta[:, i] = onp.abs(unitless_theta[:, i]) 
        else:
            theta[:, i] = unitless_theta[:, i]
        #? Is there a sleek way to do this?
        sgn = onp.sign(theta[:, i])  
        theta[:, i] = (theta[:, i] * (high - low) + sgn*low) * 1000  # TeV
        
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
        params = [EwsbParameters(*th) for th in theta]
        results = _simulator(params=params)#, settings=settings)
        out = onp.array([results.omega, results.gmuon, results.mhsm])
        out = out.T
        out = preprocess(out)
        return out

    if micromegas_simulator is None:
        micromegas_simulator = "spheno"
        # micromegas_simulator = "softsusy"

    obs_dim = 3
    theta_dim = 24  
    _simulator = micromegas[micromegas_simulator]

    distributed_simulator = (
        lambda rng, args, num_samples_per_theta=1: apply_distributed(
            simulator, args, nprocs=12
        )
    )

    return distributed_simulator, obs_dim, theta_dim
    # return simulator, obs_dim, theta_dim


if __name__ == "__main__":
    import IPython; IPython.embed()
    
    test_theta = np.array([0.5] * 24)
    test_theta = np.stack([test_theta, np.array([0.25] * 24)])
    theta = theta_addunits(test_theta)
    print(theta)
    params = [EwsbParameters(*th) for th in theta]
    results = micromegas['spheno'](params=params)#, settings=settings)
    out = onp.array([results.omega, results.gmuon, results.mhsm])
    out = out.T
    print(out)