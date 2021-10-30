import emcee
from emcee.autocorr import AutocorrError
import numpy as np
from lbi.prior import SmoothedBoxPrior
import corner
import matplotlib.pyplot as plt


def log_likelihood(theta):
    return -0.5 * np.sum(theta ** 2)


def posterior(theta):
    pdf = log_likelihood(theta) + prior(theta).sum()
    return pdf


ndim, nwalkers = 5, 10
prior, _ = SmoothedBoxPrior(theta_dim=ndim, lower=-1, upper=1, sigma=0.01)

p0 = (np.random.rand(nwalkers, ndim) - 0.5) * 2

sampler = emcee.EnsembleSampler(nwalkers, ndim, posterior)
sampler.run_mcmc(p0, 10000, progress=True)


try:
    tau = sampler.get_autocorr_time()
    burnin = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))
except AutocorrError:
    burnin = 0
    thin = 1

samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)

corner.corner(samples, range=((-1.2, 1.2) for i in range(ndim)))
plt.show()


# 10 walkers, 5 dimensions, 7500 steps
# 50 walkers, 4 dimensions, 5000 steps
# 50 walkers, 3 dimensions, 2500 steps
# 50 walkers, 2 dimensions, 2000 steps
# 50 walkers, 1 dimensions, 1000 steps