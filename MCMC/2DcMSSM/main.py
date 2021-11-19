import emcee
from emcee.autocorr import AutocorrError
import numpy as np
from lbi.prior import SmoothedBoxPrior
import corner
import matplotlib.pyplot as plt
from mcmc_settings import mcmc_settings
import sys
from tqdm.auto import tqdm


sys.path.append("/media/jt/data/Projects/MSSM/micromegas_5.2.7.a/MSSM/")
from pymicromegas import MicromegasSettings, SugraParameters
from spheno import spheno
from softsusy import softsusy
micromegas = {"spheno": spheno, "softsusy": softsusy}


def sim_wrapper(theta):
    params = [SugraParameters(*th) for th in theta]
    results = micromegas[mcmc_settings["simulator"]](params=params, settings=settings)
    out = np.array([results.omega(), results.gmuon(), results.mhsm()])
    out = out.T
    # print(out)
    return out


def log_likelihood(unitless_theta):
    theta = theta_addunits(unitless_theta)
    th_obs = sim_wrapper(theta)
    lnp = -0.5 * np.sum(((th_obs - exp_obs) / sigma_obs) ** 2, axis=1)
    return np.nan_to_num(lnp, nan=-np.inf)


def log_posterior(unitless_theta):
    unitless_theta = np.atleast_2d(unitless_theta)
    log_prior_pdf = unitless_prior(unitless_theta).sum(axis=1)
    log_prob = log_likelihood(unitless_theta) + log_prior_pdf
    return log_prob, log_prior_pdf


def theta_addunits(unitless_theta):
    theta = np.zeros_like(unitless_theta)
    theta = np.hstack((theta, np.zeros((len(theta), 1))))
    theta = np.hstack((theta, np.ones((len(theta), 1))))
    theta = np.hstack((theta, np.ones((len(theta), 1))))
    # WARNING: changes unitless_theta in place
    theta[:, 0] = 1000 * unitless_theta[:, 0] * 10.0
    theta[:, 1] = 1000 * unitless_theta[:, 1] * 10.0
    theta[:, 2] = 0
    theta[:, 3] = 48.5
    return theta


# default settings
theta_dim = 2
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


unitless_prior, _ = SmoothedBoxPrior(theta_dim=theta_dim, lower=0, upper=1, sigma=0.01)

exp_obs = np.array([0.12, 251e-11, 125.0])
sigma_obs = np.array([0.03, 59e-11, 2.0])

coords = np.random.rand(mcmc_settings["nwalkers"], theta_dim)


# Set up the backend
# Don't forget to clear it in case the file already exists
filename = f"runs/{mcmc_settings['nwalkers']}walkers_{mcmc_settings['max_n']}sampleseach_{mcmc_settings['simulator']}.h5"
backend = emcee.backends.HDFBackend(filename)
# backend.reset(mcmc_settings["nwalkers"], theta_dim)

# Initialize the sampler
sampler = emcee.EnsembleSampler(
    mcmc_settings["nwalkers"],
    theta_dim,
    log_posterior,
    backend=backend,
)


# We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(mcmc_settings["max_n"])

# This will be useful to testing convergence
old_tau = np.inf

# Now we'll sample for up to max_n steps
for sample in sampler.sample(coords, iterations=mcmc_settings["max_n"], progress=True):
    # Only check convergence every 100 steps
    if sampler.iteration % 100:
        continue

    # Compute the autocorrelation time so far
    # Using tol=0 means that we'll always get an estimate even
    # if it isn't trustworthy
    tau = sampler.get_autocorr_time(tol=0)
    autocorr[index] = np.mean(tau)
    index += 1

    # Check convergence
    converged = np.all(tau * 100 < sampler.iteration)
    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged:
        break
    old_tau = tau
