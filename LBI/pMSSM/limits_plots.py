from simulator import get_simulator_with_more_observables, theta_addunits
import jax
import numpy as onp
import matplotlib.pyplot as plt
import corner
import os
from cycler import cycler
from digitized_constraints import (
    atlas_mean_limits,
    atlas_upper_1sigma, 
    atlas_lower_1sigma,
    xenont_1T_SI_MAP,
    xenon_1T_SI_MAP_upper_1sig,
    xenon_1T_SI_MAP_lower_1sig,
)


pyplot_params = {
    "backend": "TkAgg",
    "font.family": "serif",
    "font.serif": ["CMU serif"],
    "axes.prop_cycle": cycler(
        "color",
        [
            "steelblue",
            "firebrick",
            "goldenrod",
            "mediumorchid",
            "darkslateblue",
            "DarkSalmon",
            "LightSkyBlue",
            "Navy",
            "Peru",
        ],
    ),
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{physics} \usepackage{amsmath} \usepackage{bm}",
}

plt.rcParams.update(pyplot_params)


def filter_observables(params, results, limits):
    """
    Filter params and results to only consider the points which
    fall within the limits of the observables.
    """
    observables = onp.array(
        [results["omega"], results["gmuon"], results["mhsm"], results["pval_xenon1T"]]
    ).T

    mask_idx = onp.all(observables > limits[0], axis=1)
    mask_idx &= onp.all(observables < limits[1], axis=1)

    # filter every key of results
    filtered_results = {key: value[mask_idx] for key, value in results.items()}

    return params[mask_idx], filtered_results


def plot_direct_detection_limits(
    results, logger=None, spin_dependent=True, filename=None, plot_dir=None
):
    if plot_dir is None:
        plot_dir = "/media/jt/data/Projects/MSSM/LBI/pMSSM/plots/"
    if filename is None:
        filename = "direct_detection.png"
    filename = os.path.join(plot_dir, filename)

    def amp_to_xsec(M_chi, amp, proton=True, spin_dependent=True):
        M_prot = 0.938
        M_neut = 0.939
        if proton:
            M_nuc = M_prot
        else:
            M_nuc = M_neut

        if spin_dependent:
            amp_coeff = 3
        else:
            amp_coeff = 1

        xsec = (
            4
            * (M_chi ** 2)
            * (M_nuc ** 2)
            / (onp.pi * (M_chi + M_nuc) ** 2)
            * amp_coeff
            * onp.abs(amp) ** 2
        )
        return xsec

    chi_masses = onp.abs(results["mneut1"])
    proton_si_amps = results["proton_si_amp"]
    proton_sd_amps = results["proton_sd_amp"]
    neutron_si_amps = results["neutron_si_amp"]
    neutron_sd_amps = results["neutron_sd_amp"]

    proton_si_xsec = amp_to_xsec(
        chi_masses, proton_si_amps, proton=True, spin_dependent=False
    )
    proton_sd_xsec = amp_to_xsec(
        chi_masses,
        proton_sd_amps,
        proton=True,
        spin_dependent=True,
    )
    neutron_si_xsec = amp_to_xsec(
        chi_masses,
        neutron_si_amps,
        proton=False,
        spin_dependent=False,
    )
    neutron_sd_xsec = amp_to_xsec(
        chi_masses,
        neutron_sd_amps,
        proton=False,
        spin_dependent=True,
    )

    inv_gev_to_cm = 1.0 / (1.98e-14)

    if spin_dependent:
        plt.plot(*xenont_1T_SI_MAP.T, color="black", label="Xenon 1T")
        plt.plot(
            *xenon_1T_SI_MAP_upper_1sig.T,
            color="black",
            linestyle="--",
            alpha=0.3,
        )
        plt.plot(
            *xenon_1T_SI_MAP_lower_1sig.T,
            color="black",
            label="Xenon 1T (1 sigma)",
            linestyle="--",
            alpha=0.3,
        )
        plt.scatter(
            chi_masses,
            neutron_si_xsec / inv_gev_to_cm ** 2,
            marker=".",
            c=results["pval_xenon1T"],
        )
        plt.yscale("log")
        plt.xlim(20, 900)
        plt.ylim(1e-50, 1e-38)

        plt.xlabel(r"WIMP mass (GeV)")
        plt.ylabel(r"WIMP-neutron $\sigma^{SI}$ (cm$^{2}$)")

        plt.colorbar()
        plt.legend()

    if hasattr(logger, "plot"):
        logger.plot(f"Final Corner Plot of Observables", plt, close_plot=True)
    else:
        plt.savefig(filename)

    plt.clf()
    pass


def plot_observable_corner(
    X_true, X_sigma, results, logger=None, filename=None, levels=None, plot_dir=None
):
    if plot_dir is None:
        plot_dir = "/media/jt/data/Projects/MSSM/LBI/pMSSM/plots/"
    if levels is None:
        levels = onp.array([0.68, 0.95])
    if filename is None:
        filename = "observable_corner.png"
    filename = os.path.join(plot_dir, filename)

    # Sample a gaussian distribution for each observable
    true_likelihood_samples = onp.random.normal(
        loc=onp.array(X_true[0, :3]), scale=onp.array(X_sigma[:3]), size=(30000, 3)
    )
    observables_with_nans = onp.array(
        [results["omega"], results["gmuon"], results["mhsm"]]
    ).T

    # take out nan values
    nan_mask = ~onp.isnan(observables_with_nans).any(axis=1)
    observables = observables_with_nans[nan_mask]
    observables = onp.array(observables)

    true_likelihood_samples[:, 0] = onp.log10(
        true_likelihood_samples[:, 0]
    )  # log of omega h^2
    observables[:, 0] = onp.log10(observables[:, 0])  # log of omega h^2

    X_true = onp.array(X_true)
    X_true[:, 0] = onp.log10(X_true[:, 0])  # log of omega h^2

    ranges = [(-4, 2), (7e-12, 551e-11), (121, 126)]  # , (0, 1000)]
    # ranges=None
    labels = [
        r"$\log\Omega$" + r"$h^2$",  # not sure why i need two strings here for latex
        r"$g_\mu$",
        r"$M_h$",
        # r"$M_\chi$",
    ]

    fig = corner.corner(
        onp.array(true_likelihood_samples),
        range=ranges,
        labels=labels,
        levels=levels,
        color="#4682b4",
        hist_kwargs={"density": True, "alpha": 0.5},
        plot_datapoints=False,
        # contourf_kwargs={"alpha": 0.3},
        contour_kwargs={"alpha": 0.3},
        plot_density=False,
    )

    fig = corner.corner(
        onp.array(observables),
        range=ranges,
        truths=onp.array(X_true[0, :3]),
        labels=labels,
        levels=levels,
        hist_kwargs={"density": True, "color": "#000000"},
        fig=fig,
    )

    if hasattr(logger, "plot"):
        logger.plot(f"Final Corner Plot of Observables", plt, close_plot=True)
    else:
        plt.savefig(filename)

    plt.clf()
    pass


def plot_masses_corner(results, logger=None, filename=None, levels=None, plot_dir=None):
    if plot_dir is None:
        plot_dir = "/media/jt/data/Projects/MSSM/LBI/pMSSM/plots/"
    if levels is None:
        levels = onp.array([0.68, 0.95])

    if filename is None:
        filename = "masses_corner.png"
    filename = os.path.join(plot_dir, filename)

    all_masses = onp.array(
        [
            results["mneut1"],
            results["mser"],
            results["msel"],
            results["msmr"],
            results["msml"],
        ]
    ).T

    # ? What's a negative neutralino mass?
    all_masses = onp.abs(all_masses)

    ranges = [(0, 1000), (0, 4000), (0, 4000), (0, 1000), (0, 1000)]
    # ranges = None
    labels = [
        r"$M_\chi$",
        r"$M_{\tilde{e}_R}$",
        r"$M_{\tilde{e}_L}$",
        r"$M_{\tilde{\mu}_R}$",
        r"$M_{\tilde{\mu}_L}$",
    ]

    corner.corner(onp.array(all_masses), range=ranges, labels=labels, levels=levels)
    if hasattr(logger, "plot"):
        logger.plot(f"Final Corner Plot of Observables", plt, close_plot=True)
    else:
        plt.savefig(filename)

    plt.clf()
    pass


def plot_gaugino_corner(
    samples, logger=None, filename=None, levels=None, plot_dir=None
):
    if plot_dir is None:
        plot_dir = "/media/jt/data/Projects/MSSM/LBI/pMSSM/plots/"
    if levels is None:
        levels = onp.array([0.68, 0.95])

    if filename is None:
        filename = "gaugino_corner.png"
    filename = os.path.join(plot_dir, filename)

    all_masses = onp.array(
        [
            samples[:, 0],
            samples[:, 1],
            samples[:, 2],
            samples[:, 3],
        ]
    ).T

    # ? What's a negative neutralino mass?
    all_masses = onp.abs(all_masses)

    ranges = [(0, 2000), (0, 600), (0, 2000), (0, 4000)]
    # ranges = None
    labels = [
        r"$\mu$",
        r"$M_1$",
        r"$M_2$",
        r"$M_3$",
    ]

    corner.corner(onp.array(all_masses), range=ranges, labels=labels, levels=levels)
    if hasattr(logger, "plot"):
        logger.plot(f"Final Corner Plot of Observables", plt, close_plot=True)
    else:
        plt.savefig(filename)

    plt.clf()
    pass


def plot_mass_splitting(results, logger=None, filename=None, plot_dir=None):
    if plot_dir is None:
        plot_dir = "/media/jt/data/Projects/MSSM/LBI/pMSSM/plots/"
    if filename is None:
        filename = os.path.join(
            plot_dir,
            "LHC_mass_splitting_constraints.png",
        )
    
    slepton_masses = onp.array(
        [results["msel"], results["msml"], results["mser"], results["msmr"]]
    ).T

    # ? What's a negative neutralino mass?
    slepton_masses = onp.abs(slepton_masses)
    chi_masses = onp.abs(results["mneut1"])

    lightest_sleptons = onp.min(slepton_masses, axis=1)
    mass_splitting = lightest_sleptons - chi_masses

    plt.scatter(
        chi_masses,
        mass_splitting,
        marker=".",
        c=results["pval_xenon1T"],
    )

    plt.plot(*atlas_mean_limits.T, color="maroon", label="ATLAS")
    plt.plot(*atlas_upper_1sigma.T, color="maroon", ls="--")
    plt.plot(*atlas_lower_1sigma.T, color="maroon", ls="--")

    plt.ylabel(r"$M_{\ell} - M_{\chi}$ (GeV)")
    plt.xlabel(r"$M_\chi$ (GeV)")
    plt.xlim(85, 1000)
    plt.ylim(0.1, 500)
    plt.yscale("log")

    plt.legend()
    plt.colorbar()
    if hasattr(logger, "plot"):
        logger.plot(f"Final Corner Plot of Observables", plt, close_plot=True)
    else:
        plt.savefig(filename)

    plt.clf()
    pass


def prior_plotter(
    sample_prior, key=345, n_samples=10000, logger=None, filename=None, plot_dir=None
):
    """
    A corner plot of all model parameters in the prior.
    This is useful for checking that the prior is reasonable and when
    comparing with the posterior to see how much has been learned.
    """
    if plot_dir is None:
        plot_dir = "/media/jt/data/Projects/MSSM/LBI/pMSSM/plots/"
    if filename is None:
        filename = "prior_plot.png"
    filename = os.path.join(plot_dir, filename)
    prior_rng_key = jax.random.PRNGKey(key)
    prior_samples = sample_prior(prior_rng_key, n_samples)

    # samples are unitless. let's scale them to the true values
    unitful_samples = theta_addunits(prior_samples)
    # ignore squarks
    unitful_samples = onp.delete(unitful_samples, slice(11, 20), 1)

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
        r"$M_A$",
        r"$\tan\beta$",
        r"$A_t$",
        r"$A_b$",
        r"$A_{\tau}$",
    ]
    corner.corner(unitful_samples, labels=labels)

    if hasattr(logger, "plot"):
        logger.plot(f"Corner plot of Prior", plt, close_plot=True)
    else:
        plt.savefig(filename)

    plt.clf()
    pass


def M1_vs_mchi(unitful_samples, results, logger=None, filename=None, plot_dir=None):
    if plot_dir is None:
        plot_dir = "/media/jt/data/Projects/MSSM/LBI/pMSSM/plots/"
    if filename is None:
        filename = "M1_vs_mchi.png"
    filename = os.path.join(plot_dir, filename)

    M1 = unitful_samples[:, 1]
    chi_masses = onp.abs(results["mneut1"])
    plt.scatter(M1, chi_masses, marker=".", alpha=0.2)

    if hasattr(logger, "plot"):
        logger.plot(f"Final Corner Plot of Observables", plt, close_plot=True)
    else:
        plt.savefig(filename)

    plt.clf()
    pass


if __name__ == "__main__":
    from lbi.prior import SmoothedBoxPrior

    theta_dim = 15

    log_prior, sample_prior = SmoothedBoxPrior(
        theta_dim=theta_dim, lower=-1.0, upper=1.0, sigma=0.01
    )

    prior_plotter(sample_prior)
