import corner
import emcee
import numpy as np
import matplotlib.pyplot as plt
from mcmc_settings import mcmc_settings
import arviz as az

theta_dim = 4

filename = f"runs/fixedObs_{mcmc_settings['nwalkers']}walkers_{mcmc_settings['max_n']}sampleseach_{mcmc_settings['simulator']}.h5"
reader = emcee.backends.HDFBackend(filename)

tau = reader.get_autocorr_time(tol=0)
burnin = int(2 * np.max(tau))
thin = max(1, int(0.5 * np.min(tau))) 

samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
log_prior_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)

print("tau: {0}".format(np.mean(tau)))
print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
print("flat log prob shape: {0}".format(log_prob_samples.shape))
print("flat log prior shape: {0}".format(log_prior_samples.shape))

all_samples = np.concatenate(
    (samples, log_prob_samples[:, None]), axis=1
)

labels = list(map(r"$\theta_{{{0}}}$".format, range(1, theta_dim + 1)))
labels = [r"$M_0$", r"$M_{1/2}$", r"$A_0$", r"$\tan\beta$"]
labels += ["log prob"]

ranges = [(0.0, 1.), (0.0, 1.), (0.0, 1.), (0.0, 1.), (0., 7.5)]

az_samp = az.from_emcee(reader)
print(az.summary(az_samp))


corner.corner(all_samples, labels=labels, range=ranges)
plt.show()
