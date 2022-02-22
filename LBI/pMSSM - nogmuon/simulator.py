import jax
import jax.numpy as np
import numpy as onp
from tqdm.auto import tqdm

from utils.distributed import apply_distributed

from pymicromegas import EwsbParameters, spheno, MicromegasSettings

"""
Ranges taken from Hollingsworth et al. All in TeV
mu : +- (0.1, 4) 
mg1 : +- (0.05, 4)->  delta
mg2 : +- (0.1, 4)
mg3 : (0.4, 4)
ml1 : (0.1, 4)
ml2 : (0.1, 4)  # They did not look at 2nd gen slepton masses
ml3 : (0.1, 4)
mr1 : (0.1, 4)
mr2 : (0.1, 4)
mr3 : (0.1, 4)
mq1 : (0.4, 4) -> 4
mq2 : (0.4, 4) -> 4 # They did not look at 2nd gen squark masses
mq3 : (0.2, 4) -> 4
mu1 : (0.4, 4) -> 4
mu2 : (0.4, 4) -> 4
mu3 : (0.2, 4) -> 4
md1 : (0.4, 4) -> 4
md2 : (0.4, 4) -> 4
md3 : (0.2, 4) -> 4
mh3 : (0.1, 4)  # mass of cp odd higgs
tb : (1, 60)
at : +- (0, 4)
ab : +- (0, 4)
al : +- (0, 4)

To handle the plus/minus sign, we'll have the prior go from -1 to 1
and then take the absolute value of everything that is not plus/minus. 

"""


micromegas = {"spheno": spheno}


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
    (False, 0.1, 4.0),
    # (False, 0.05, 4),
    (False, 0.0, 0.001),  # this is R. R = M1/(|mu| + |M2|) and we want < 1
    (False, 0.1, 4.0),
    (True, 0.4, 4.0),
    (True, 0.1, 4.0),
    (True, 0.1, 4.0),
    (True, 0.1, 4.0),
    (True, 0.1, 4.0),
    (True, 0.1, 4.0),
    (True, 0.1, 4.0),
    (True, 0.4, 4.0),
    # skip squark masses
    # (True, 0.4, 4.0),
    # (True, 0.2, 4.0),
    # (True, 0.4, 4.0),
    # (True, 0.4, 4.0),
    # (True, 0.2, 4.0),
    # (True, 0.4, 4.0),
    # (True, 0.4, 4.0),
    # (True, 0.2, 4.0),
    # (True, 0.1, 4.0),
    (True, 0.001, 0.060),  # these get multiplied by 1000
    (False, 0.0, 4.0),
    (False, 0.0, 4.0),
    (False, 0.0, 4.0),
]

M1_idx = 1
squark_idx = 11

def theta_addunits(unitless_theta):
    unitless_theta = onp.atleast_2d(unitless_theta)
    theta = onp.zeros_like(unitless_theta)
    for i, (take_abs, low, high) in enumerate(theta_ranges):
        if take_abs:
            theta[:, i] = onp.abs(unitless_theta[:, i])
        else:
            theta[:, i] = unitless_theta[:, i]
        # ? Is there a sleek way to do this?
        sgn = onp.sign(theta[:, i])
        theta[:, i] = (theta[:, i] * (high - low) + sgn * low) * 1000  # TeV
    
    # invert R to get M1 (R = M1*(1/|mu| + 1/|M2|))
    sgn = onp.sign(theta[:, M1_idx])
    theta[:, M1_idx] = sgn*50 + theta[:, M1_idx] / (1/onp.abs(theta[:, 0]) + 1/onp.abs(theta[:, 2]))
    # theta[:, M1_idx] = 50 + theta[:, M1_idx] * (onp.abs(theta[:, 0] + onp.abs(theta[:, 2]))) 
    
    
    # insert 9 squark masses
    theta = onp.insert(theta, [squark_idx], [4000 for i in range(9)], axis=1)


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


    # settings = MicromegasSettings(
    #     relic_density=True,
    #     masses=True,
    #     # gmuon=True,
    #     # bsg=True,
    #     # bsmumu=True,
    #     # btaunu=True,
    #     # delta_rho=True,
    #     # sort_odd=True,
    #     fast=True,
    #     # beps=0.0001,
    #     # cut=0.01,
    # )
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
        # import IPython; IPython.embed()
        out = onp.array([results.omega, results.mhsm])
        out = out.T
        out = preprocess(out)
        return out

    if micromegas_simulator is None:
        micromegas_simulator = "spheno"
        # micromegas_simulator = "softsusy"

    obs_dim = 2
    theta_dim = 15
    _simulator = micromegas[micromegas_simulator]

    distributed_simulator = (
        lambda rng, args, num_samples_per_theta=1: apply_distributed(
            simulator, args, nprocs=10
        )
    )

    def batched_distributed_simulator(
        rng, args, num_samples_per_theta=1, batch_size=2500
    ):
        batches = np.array_split(args, 1 + len(args) // batch_size)
        results = []
        for batch in tqdm(batches):
            # print(batch)
            results.append(
                distributed_simulator(
                    rng, batch, num_samples_per_theta=num_samples_per_theta
                )
            )
        return np.concatenate(results)

    return batched_distributed_simulator, obs_dim, theta_dim
    # return distributed_simulator, obs_dim, theta_dim
    # test_simulator = lambda rng, args: simulator(args)
    # return test_simulator, obs_dim, theta_dim


if __name__ == "__main__":
    import IPython

    IPython.embed()

    test_theta = np.array([0.5] * 24)
    test_theta = np.stack([test_theta, np.array([0.25] * 24)])
    theta = theta_addunits(test_theta)
    print(theta)
    params = [EwsbParameters(*th) for th in theta]
    results = micromegas["spheno"](params=params)  # , settings=settings)
    out = onp.array([results.omega, results.gmuon, results.mhsm])
    out = out.T
    print(out)
