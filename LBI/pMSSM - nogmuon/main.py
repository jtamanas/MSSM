import torch
import jax
import jax.numpy as np
import numpy as onp
from trax.jaxboard import SummaryWriter
from lbi.pipeline.base import pipeline
from lbi.prior import SmoothedBoxPrior

# from lbi.diagnostics import MMD, ROC_AUC, LR_ROC_AUC
from lbi.sampler import hmc
from simulator import get_simulator, theta_addunits, theta_ranges

import corner
import matplotlib.pyplot as plt

from pipeline_kwargs import pipeline_kwargs


# from jax.config import config
# config.update('jax_disable_jit', True)


def scale_X(x):
    """
    x: jax.numpy array
    """
    # omega = (backend.log(x[0]) - 1.6271843) / 2.257083
    # gmuon = (1e10 * x[1] - 0.686347) / (3.0785 * 3)
    # mh = (x[2] - 124.86377) / 2.2839
    # print("x shape", x.shape)
    omega = np.atleast_2d(np.log(x[..., 0]) - 2).T
    gmuon = np.atleast_2d(1e10 * x[..., 1]).T
    mh = np.atleast_2d((x[..., 2] - 124.86377) / 2.2839).T
    # print("omega shape", omega.shape)
    out = np.hstack([omega, gmuon, mh])
    return x
    return out


def scale_Theta(theta):
    """
    theta : jax.numpy array

    theta is uniform between 0 and 1
    """
    std = 0.70710678
    # return theta
    return (theta) / std


# --------------------------

seed = 1234
rng, model_rng, hmc_rng = jax.random.split(jax.random.PRNGKey(seed), num=3)


# --------------------------
# Create logger
# experiment_name = datetime.datetime.now().strftime("%s")
# experiment_name = f"{model_type}_{experiment_name}"
# logger = SummaryWriter("runs/" + experiment_name)
logger = None


# set up true model for posterior inference test
simulator_kwargs = {}
simulate, obs_dim, theta_dim = get_simulator(**simulator_kwargs)
X_true = np.array([[0.12, 125.0]])
X_sigma = np.array([0.1, 2.0]) * 1.0

print("X_true", X_true)

# --------------------------
# set up prior
log_prior, sample_prior = SmoothedBoxPrior(
    theta_dim=theta_dim, lower=-1.0, upper=1.0, sigma=0.01
)

model, log_prob, params, Theta_post = pipeline(
    rng,
    X_true,
    get_simulator,
    # Prior
    log_prior,
    sample_prior,
    # Simulator
    simulator_kwargs=simulator_kwargs,
    # scaling
    sigma=X_sigma,
    scale_X=scale_X,
    scale_Theta=scale_Theta,
    **pipeline_kwargs,
)

parallel_log_prob = jax.vmap(log_prob, in_axes=(0, None, None))


def potential_fn(theta):
    if len(theta.shape) == 1:
        theta = theta[None, :]

    log_P = log_prior(theta)

    log_L = parallel_log_prob(params, X_true, theta)
    log_L = log_L.mean(axis=0)

    log_post = -(log_L + log_P)
    return log_post.sum()


num_chains = 10
init_theta = sample_prior(rng, num_samples=num_chains)

mcmc = hmc(
    rng,
    potential_fn,
    init_theta,
    adapt_step_size=True,
    adapt_mass_matrix=True,
    dense_mass=True,
    step_size=1e0,
    max_tree_depth=8,
    num_warmup=2000,
    num_samples=500,
    num_chains=num_chains,
    extra_fields=("potential_energy",),
    chain_method="vectorized",
)
mcmc.print_summary()

samples = mcmc.get_samples(group_by_chain=False).squeeze()

# samples are unitless. let's scale them to the true values
unitful_samples = theta_addunits(samples)

unitful_samples = onp.hstack([unitful_samples[:, :11], unitful_samples[:, 20:]])

posterior_log_probs = -mcmc.get_extra_fields()["potential_energy"]
unitful_samples = np.hstack([unitful_samples, posterior_log_probs[:, None]])


labels = [
    r"$\mu$",
    r"$M_1$",
    r"$M_2$",
    r"$M_3$",
    r"$M_{L_1}$",
    r"$M_{L_2}$",
    r"$M_{L_3}$",
    r"$M_{r_1}$",
    r"$M_{r_2}$",
    r"$M_{r_3}$",
    # r"$M_{Q_1}$",
    # r"$M_{Q_2}$",
    # r"$M_{Q_3}$",
    # r"$M_{u_1}$",
    # r"$M_{u_2}$",
    # r"$M_{u_3}$",
    # r"$M_{d_1}$",
    # r"$M_{d_2}$",
    # r"$M_{d_3}$",
    r"$M_A$",
    r"$\tan\beta$",
    r"$A_t$",
    r"$A_b$",
    r"$A_{\tau}$",
]
labels += ["log prob"]
theta_ranges = np.array(theta_ranges)
ranges = np.stack([theta_ranges[:, 0] - 1, np.ones_like(theta_ranges[:, 0])]).T
ranges = theta_addunits(ranges.T + 1e-10).T  # add a small number to avoid 0
ranges = onp.hstack([ranges.T[:, :11], ranges.T[:, 20:]]).T

ranges = np.vstack([ranges, np.array([(posterior_log_probs.min(), posterior_log_probs.max())])])

# ignore squarks

corner.corner(onp.array(unitful_samples), range=ranges, labels=labels)

if hasattr(logger, "plot"):
    logger.plot(f"Final Corner Plot", plt, close_plot=True)
else:
    plt.savefig("posterior_corner.png")

# Run simulations on the samples
# --------------------------
# First shuffle samples to get a random order
_, _, observables_rng = jax.random.split(rng, num=3)
samples = jax.random.shuffle(observables_rng, samples, axis=0)
# Generate the observables from 2000 samples of the posterior
observables = simulate(rng, samples[:2000])
# take out nan values
observables = observables[~np.any(np.isnan(observables), axis=1)]
observables = onp.array(observables)

observables[:, 0] = onp.log10(observables[:, 0]) # log of omega h^2

X_true = onp.array(X_true)
X_true[:, 0] = onp.log10(X_true[:, 0]) # log of omega h^2

ranges = [(-6, 3), (110, 130)]
labels = [
    r"$\log\Omega$" + r"$h^2$",  # not sure why i need two strings here for latex
    r"$M_h$",
]
corner.corner(onp.array(observables), range=ranges, truths=onp.array(X_true[0]), labels=labels)
if hasattr(logger, "plot"):
    logger.plot(f"Final Corner Plot of Observables", plt, close_plot=True)
else:
    plt.savefig("observables_corner.png")

