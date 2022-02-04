
import jax
import jax.numpy as np
import numpy as onp
import corner
import emcee
import matplotlib.pyplot as plt
from mcmc_settings import mcmc_settings
import numpy as np

import sys 
sys.path.append('/media/jt/data/Projects/MSSM/LBI/cMSSM/')
from simulator import get_simulator

# ------------------------------------
# Simulate observables
# ------------------------------------

# set up true model for posterior inference test
# simulator_kwargs = {"preprocess": scale_X}
simulator_kwargs = {}
simulate, obs_dim, theta_dim = get_simulator(**simulator_kwargs)

seed = 1234
rng, model_rng, hmc_rng = jax.random.split(jax.random.PRNGKey(seed), num=3)
filename = f"runs/fixedObs_{mcmc_settings['nwalkers']}walkers_{mcmc_settings['max_n']}sampleseach_{mcmc_settings['simulator']}.h5"


reader = emcee.backends.HDFBackend(filename)

tau = reader.get_autocorr_time(tol=0)
burnin = int(2 * np.max(tau))
thin = max(1, int(0.5 * np.min(tau))) 
theta = reader.get_chain(discard=burnin, flat=True, thin=thin)

import IPython; IPython.embed()


observables = simulate(rng, theta, num_samples_per_theta=1)

theta_and_log_probs = np.save(
    "Posterior Samples/mcmc_posterior_observables.npy", onp.array(observables)
)

# numpy delete rows with nan values
observables = observables[~np.isnan(observables).any(axis=1)]

# ------------------------------------
# Plotting
# ------------------------------------

X_true = np.array([0.12, 251e-11, 125.0])
X_sigma = np.array([0.03, 59e-11, 2.0])
n_sigma = 4

ranges = onp.vstack([(X_true - n_sigma * X_sigma), (X_true + n_sigma * X_sigma)]).T


labels = [r"$\Omega$", r"$a_\mu$", r"$m_h$"]
fig = corner.corner(
    onp.array(observables), labels=labels, range=ranges, hist_kwargs={"density": True}
)
axes = onp.array(fig.axes).reshape((3, 3))

# add likelihoods to corner plot

xs = []
pdfs = []
for _mean, _sigma in zip(X_true, X_sigma):
    x = np.linspace(_mean - n_sigma * _sigma, _mean + n_sigma * _sigma)
    pdf = jax.scipy.stats.norm.pdf(x, _mean, _sigma)
    xs.append(x)
    pdfs.append(pdf)

xs = onp.stack(xs)
pdfs = onp.stack(pdfs)

for i in range(len(X_true)):
    axes[i, i].plot(xs[i], pdfs[i], color="black", linewidth=0.5, linestyle="--")

plt.savefig("Posterior Samples/mcmc_posterior_obvervables_corner.png")
