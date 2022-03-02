import torch
import jax
import jax.numpy as np
import numpy as onp
from trax.jaxboard import SummaryWriter
from lbi.pipeline.base import pipeline
from lbi.prior import SmoothedBoxPrior
import h5py
# from lbi.diagnostics import MMD, ROC_AUC, LR_ROC_AUC
from lbi.sampler import hmc
from simulator import get_simulator, theta_addunits, theta_ranges


from simulator import get_simulator_with_more_observables
import limits_plots as lp


import corner
import matplotlib.pyplot as plt

from pipeline_kwargs import pipeline_kwargs


from jax.config import config
config.update('jax_disable_jit', True)


def scale_X(x):
    """
    x: jax.numpy array
    """
    omega = np.atleast_2d(np.log10(np.clip(x[..., 0], a_min=1e-5, a_max=None))*2).T
    
    gmuon = np.clip(x[..., 1], a_min=1e-11, a_max=1e-8)
    gmuon = np.atleast_2d(np.log10(gmuon) + 9.5).T
    
    mh = np.clip(x[..., 2], a_min=120)
    mh = np.atleast_2d((mh - 123.86377) / 2.2839).T
    
    out = [omega, gmuon, mh]
    
    if pipeline_kwargs["simulator_kwargs"]["use_direct_detection"]:
        # if exists, it'll always be 3rd
        xenon_pval = np.atleast_2d(np.clip(x[..., 3], a_min=None, a_max=X_true[:, 3])).T
        out += [xenon_pval]
        
    if pipeline_kwargs["simulator_kwargs"]["use_atlas_constraints"]:
        # this will always be last
        atlas_pval = np.atleast_2d(np.clip(x[..., -1], a_min=None, a_max=X_true[:, -1])).T
        out += [atlas_pval]
        
    out = np.hstack(out)
    
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
simulate, obs_dim, theta_dim = get_simulator(**pipeline_kwargs["simulator_kwargs"])
# --------------------------
# The true values correspond to: omega, gmuon, mh, xenon pvals
X_true = [0.12, 251e-11, 125.0]
X_sigma = [0.005, 59e-11, 1.0]

if pipeline_kwargs["simulator_kwargs"]["use_direct_detection"]:
    X_true += [[0.5]]
    X_sigma += [1e-5]
if pipeline_kwargs["simulator_kwargs"]["use_atlas_constraints"]:
    X_true += [0.5]
    X_sigma += [1e-5]

X_true = np.array([X_true])
X_sigma = np.array(X_sigma)

print("X_true", X_true)

# --------------------------
# set up prior
log_prior, sample_prior = SmoothedBoxPrior(
    theta_dim=theta_dim, lower=-1.0, upper=1.0, sigma=0.01
)

# Plot prior
lp.prior_plotter(sample_prior)

model, log_prob, params, Theta_post = pipeline(
    rng,
    X_true,
    get_simulator,
    # Prior
    log_prior,
    sample_prior,
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


num_chains = 16
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
    num_samples=750,
    num_chains=num_chains,
    extra_fields=("potential_energy",),
    chain_method="vectorized",
)
mcmc.print_summary()

samples = mcmc.get_samples(group_by_chain=False).squeeze()

# samples are unitless. let's scale them to the true values
unitful_samples = theta_addunits(samples)
# ignore squarks
unitful_samples = np.delete(unitful_samples, slice(11, 20), 1)

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
ranges = [
    [-2000, 2000],
    [-1200, 1200],
    [-2000, 2000],
    [400, 4000],
    [100, 4000],
    [100, 1000],
    [100, 4000],
    [100, 4000],
    [100, 1000],
    [100, 4000],
    [400, 4000],
    [1, 50],
    [-4000, 4000],
    [-4000, 4000],
    [-4000, 4000],
]
ranges = onp.array(ranges)

ranges = np.vstack(
    [ranges, np.array([(posterior_log_probs.min(), posterior_log_probs.max())])]
)

try:
    corner.corner(onp.array(unitful_samples), range=ranges, labels=labels)

    if hasattr(logger, "plot"):
        logger.plot(f"Final Corner Plot", plt, close_plot=True)
    else:
        plt.savefig("posterior_corner.png")
except KeyboardInterrupt:
    print("Could not plot corner plot")
plt.clf()


# --------------------------
# --------------------------
# Run full simulations on the samples
# --------------------------


# First shuffle samples to get a random order
_, _, observables_rng = jax.random.split(rng, num=3)
samples = jax.random.shuffle(observables_rng, samples, axis=0)
# Generate the observables from 2000 samples of the posterior
big_simulator = get_simulator_with_more_observables()

num_samples = -1  # -1 means all
results = big_simulator(rng, samples[:num_samples])


# --------------------------
# Save the samples and micromegas results
# --------------------------
samples_and_results = results
samples_and_results["samples"] = unitful_samples[:num_samples]

hf = h5py.File('samples_and_results.h5', 'w')
for key, arr in samples_and_results.items():
    hf.create_dataset(key, data=arr)
hf.close()

# --------------------------
# Plot the samples and results
# --------------------------

lower_limits = onp.array(X_true - 3 * X_sigma)
# lower limit for direct detection is 3 sigma excluded
lower_limits[:, 3] = 2 * 0.02275
upper_limits = onp.array(X_true + 3 * X_sigma)
limits = onp.stack([lower_limits, upper_limits]).squeeze()


lp.M1_vs_mchi(samples[:num_samples], results)
lp.plot_observable_corner(X_true, X_sigma, results, logger=logger)
lp.plot_direct_detection_limits(
    results, logger=logger, filename="unfiltered_direct_detection.png"
)
lp.plot_mass_splitting(
    results, logger=logger, filename="unfiltered_LHC_constraints.png"
)
lp.plot_masses_corner(results, logger=logger, filename="unfiltered_masses_corner.png")

filtered_unitful_samples, filtered_results = lp.filter_observables(
    unitful_samples[:num_samples], results, limits
)

print(filtered_unitful_samples.shape[0], "samples after filtering")

lp.plot_direct_detection_limits(filtered_results, logger=logger)
lp.plot_mass_splitting(filtered_results, logger=logger)
try:
    lp.plot_masses_corner(filtered_results, logger=logger)
except:
    print("Could not plot masses corner")

try:
    corner.corner(onp.array(filtered_unitful_samples), range=ranges, labels=labels)

    if hasattr(logger, "plot"):
        logger.plot(f"Final Corner Plot", plt, close_plot=True)
    else:
        plt.savefig("filtered_posterior_corner.png")
    plt.clf()
except:
    print("Could not plot filtered posterior corner")
