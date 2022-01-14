import jax.numpy as np
import numpy as onp 


def calculate_higgs_mass(theta):
    pass

def calculate_gmuon(theta):
    pass

def calculate_colider_constraints(theta):
    pass

def get_simulator(**kwargs):
    """
    Parameters:
    -----------
    Returns:
        simulator: a function that takes a theta vector and returns an array of observables
        obs_dim: the dimension of the observation space
        theta_dim: the dimension of the theta space
    """
    def simulator(rng, theta, num_samples_per_theta=1):
        higgs_mass = calculate_higgs_mass(theta)
        gmuon = calculate_gmuon(theta)
        colider_constraints = calculate_colider_constraints(theta)
        
        return onp.array([higgs_mass, gmuon, colider_constraints]).T
    
    
    obs_dim = 3 #! Make sure this is correct
    theta_dim = 4 #! Make sure this is correct

    return simulator, obs_dim, theta_dim