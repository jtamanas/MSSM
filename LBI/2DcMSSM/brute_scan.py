import jax.numpy as np
from simulator import get_simulator
import matplotlib.pyplot as plt


def log_likelihood(x):
    true = np.array([0.12, 251e-11, 125.0])
    sigma = np.array([0.03, 59e-11, 2.0]) * 0.1
    z = (x - true) / sigma
    out = -0.5 * z ** 2 # - np.log(sigma * np.sqrt(2 * np.pi))
    return np.sum(out, axis=1)


sim, obs_dim, theta_dim = get_simulator()

simulator = lambda theta: sim(None, theta)

res = 50
M0, M12 = np.meshgrid(
    *[np.linspace(0, 1, res) for _ in range(theta_dim)], indexing="ij"
)
grid = np.stack([M0.ravel(), M12.ravel()]).T

obs = simulator(grid)

logL = log_likelihood(obs)


plt.scatter(*grid.T, c=-np.log(logL*np.nanmin(logL)), marker='o', alpha=0.4)
plt.xlabel("M0")
plt.ylabel("M12")
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.colorbar()
plt.show()
