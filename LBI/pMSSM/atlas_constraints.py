from statistics import NormalDist
import jax
# import jax.numpy as np
import numpy as onp
from digitized_constraints import atlas_mean_limits, atlas_upper_1sigma, atlas_lower_1sigma


def calc_atlas_pvals(mass_splitting, m_chi):
    """
    Given a mass_splitting, calculate the m_chi p-value, i.e. 
            p(m_chi | mass_splitting)
    
    To do so, we 
        - digitize the ATLAS constraints and obtain their
                means and standard deviations. 
        - interpolate these onto a specified range of 
                mass splitting. Outside of the range means
                the points are not constrained (pval = 1)
        - For data, plug in mass_splitting to obtain mean
                and std, then use to calculate p-value of 
                specified m_chi
                
    Parameters:
    -----------
    mass_splitting: float or np.array
        mass splitting between neutralino and lightest of the 
                selectrons and smuons (both L and R)
        
    m_chi: float
        mass of the neutralino
        
    Returns:
    --------
    pval: float or np.array
        p-value of m_chi given mass_splitting
    """
    def vectorized_normal(x, mean, std):
        """
        We need to vectorize the normal distribution to be able to
        evaluate it on an array of points.
        
        We're using NormalDist from statistics instead of scipy.stats
        because scipy.stats.norm can return nans for some inputs.
        """
        dists = onp.vectorize(NormalDist)(mean, std)
        def eval_dist(x, dist):
            return dist.cdf(x)
        return onp.vectorize(eval_dist)(x, dists)
        
    
    # don't interpolate outside of the range of the ATLAS constraints    
    oob_mask = (mass_splitting > onp.max(atlas_mean_limits[:, 1]))
    oob_mask = oob_mask | (mass_splitting < onp.min(atlas_mean_limits[:, 1]))
    
    # interpolate onto the specified mass splitting
    means = onp.interp(mass_splitting, atlas_mean_limits[:, 1], atlas_mean_limits[:, 0])
    lower_1sigma = onp.interp(mass_splitting, atlas_lower_1sigma[:, 1], atlas_lower_1sigma[:, 0])
    stds = means - lower_1sigma
    stds = onp.clip(stds, 1e-5, None)
    # calculate the p-value
    pvals = vectorized_normal(m_chi, means, stds)
    pvals[oob_mask] = 1.0  # no constraints outside of the ATLAS range
    return pvals
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # scan over m_chi
    mass_split = onp.linspace(0.5, 20, 300)
    m_chi = onp.linspace(100, 300, 300)
    
    grid = onp.meshgrid(mass_split, m_chi)
    grid = onp.stack(grid, axis=-1).reshape(-1, 2)
    
    pvals = calc_atlas_pvals(grid[:, 0], grid[:, 1])
    plt.scatter(grid[:, 1], grid[:, 0], c=pvals, marker='.', s=1)
    
    plt.plot(*atlas_mean_limits.T, color="maroon", label="ATLAS")
    plt.plot(*atlas_upper_1sigma.T, color="maroon", ls="--")
    plt.plot(*atlas_lower_1sigma.T, color="maroon", ls="--")
    plt.colorbar()
    plt.xlim(100, 300)
    plt.ylim(0.5, 20)
    
    plt.xlabel("m_chi")
    plt.ylabel("mass splitting")
    
    plt.show()
    