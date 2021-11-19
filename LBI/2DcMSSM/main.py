import jax
import jax.numpy as np
import numpy as onp
from trax.jaxboard import SummaryWriter
from lbi.pipeline.base import pipeline
from lbi.prior import SmoothedBoxPrior

# from lbi.diagnostics import MMD, ROC_AUC, LR_ROC_AUC
from lbi.sampler import hmc
from simulator import get_simulator

import corner
import matplotlib.pyplot as plt

from pipeline_kwargs import pipeline_kwargs


@jax.jit
def scale_X(x):
    omega = (np.log(x[:, 0]) - 1.6271843) / 2.257083
    gmuon = (1e10 * x[:, 1] - 0.686347) / (3.0785 * 3)
    mh = (x[:, 2] - 124.86377) / 2.2839
    out = np.stack([omega, gmuon, mh], axis=1)
    return out

@jax.jit
def inverse_scale_X(x):
    omega = np.exp(x[:, 0] * 2.257083 + 1.6271843)
    gmuon = (x[:, 1] * (3.0785 * 3) + 0.686347) / 1e10
    mh = x[:, 2] * 2.2839 + 124.8637
    out = np.stack([omega, gmuon, mh], axis=1)
    return out


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
simulator_kwargs = {"preprocess": scale_X}
simulate, obs_dim, theta_dim = get_simulator(**simulator_kwargs)
X_true = np.array([[0.12, 251e-11, 125.0]])
X_true = scale_X(X_true)
X_sigma = np.array([0.03, 59e-11, 2.0])
print("X_true", X_true)

# --------------------------
# set up prior
log_prior, sample_prior = SmoothedBoxPrior(
    theta_dim=theta_dim, lower=0, upper=1, sigma=0.01
)

model, params, Theta_post = pipeline(
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
    inverse_scale_X=inverse_scale_X,
    **pipeline_kwargs,
)

parallel_log_prob = jax.vmap(model.apply, in_axes=(0, None, None))


def potential_fn(theta):
    if len(theta.shape) == 1:
        theta = theta[None, :]

    log_L = parallel_log_prob(params, X_true, theta)
    log_L = log_L.mean(axis=0)

    log_post = -log_L - log_prior(theta)
    return log_post.sum()


num_chains = 32
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
posterior_log_probs = -mcmc.get_extra_fields()["potential_energy"]
samples = np.hstack([samples, posterior_log_probs[:, None]])


labels = list(map(r"$\theta_{{{0}}}$".format, range(1, theta_dim + 1)))
labels = [r"$M_0$", r"$M_{1/2}$"]
labels += ["log prob"]
ranges = [
    (0.0, 1.0),
    (0.0, 1.0),
    (posterior_log_probs.min(), posterior_log_probs.max()),
]

corner.corner(onp.array(samples), range=ranges, labels=labels)

if hasattr(logger, "plot"):
    logger.plot(f"Final Corner Plot", plt, close_plot=True)
else:
    plt.savefig("temp.png")
