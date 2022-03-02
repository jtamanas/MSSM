from re import L
import numpy as onp
import matplotlib.pyplot as plt
import limits_plots as lp
import h5py
import corner 


plot_dir = "/media/jt/data/Projects/MSSM/LBI/pMSSM/plots/"


# Load in array
logger = None
hf = h5py.File('/media/jt/data/Projects/MSSM/LBI/pMSSM/results/samples_and_results.h5', 'r')
samples = hf['samples'][:]
results = {}
for key in hf.keys():
    if key != 'samples':
        results[key] = hf[key][:]
hf.close()
# what are negative chi masses?
results['mneut1'] = onp.abs(results['mneut1'])

# Plot arguments 
lower_limits = onp.array([[0.01, 74e-11, 122.0, 2 * 0.02275]])
upper_limits = onp.array([[0.30, 428e-11, 128.0, 1.1]])
limits = onp.stack([lower_limits, upper_limits]).squeeze()
ranges = [
    [-2000, 2000],
    [-1000, 1000],
    [-2000, 2000],
    [400, 4000],
    [100, 4000],
    [100, 1000],
    [100, 4000],
    [100, 4000],
    [100, 1000],
    [100, 4000],
    [400, 4000],
    [1, 60],
    [-4000, 4000],
    [-4000, 4000],
    [-4000, 4000],
]
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
levels = [0.6827, 0.9545, 0.9973]

X_true = onp.array([[0.12, 251e-11, 125.0, 0.05]])
X_sigma = onp.array([0.01, 59e-11, 1.0, 1e-5])


# Plot the stuff 

lp.M1_vs_mchi(samples, results)
lp.plot_observable_corner(X_true, X_sigma, results, logger=logger, levels=levels)
lp.plot_gaugino_corner(samples, logger=logger, levels=levels)
lp.plot_direct_detection_limits(
    results, logger=logger, filename=plot_dir + "unfiltered_direct_detection.png"
)
lp.plot_mass_splitting(
    results, logger=logger, filename=plot_dir + "unfiltered_LHC_constraints.png"
)
lp.plot_masses_corner(
    results, logger=logger, filename=plot_dir + "unfiltered_masses_corner.png", levels=levels
)

filtered_unitful_samples, filtered_results = lp.filter_observables(
    samples, results, limits
)

print(filtered_unitful_samples.shape[0], "samples after filtering")


lp.plot_direct_detection_limits(filtered_results, logger=logger)
lp.plot_mass_splitting(filtered_results, logger=logger)
try:
    lp.plot_masses_corner(filtered_results, logger=logger, levels=levels)
except:
    print("Could not plot masses corner")

try:
    corner.corner(
        onp.array(filtered_unitful_samples),
        range=ranges,
        labels=labels,
        levels=levels,
    )

    if hasattr(logger, "plot"):
        logger.plot(f"Final Corner Plot", plt, close_plot=True)
    else:
        plt.savefig(plot_dir + "filtered_posterior_corner.png")
    plt.clf()
except:
    print("Could not plot filtered posterior corner")
