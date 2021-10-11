import pandas as pd
import torch


dataset_dir = "../../datasets/Hollingsworth"

import sys
import os
sys.path.append(os.path.abspath(dataset_dir))
from read_dataset import import_data

def normalize(df, eps=1e-9):
    return (df - df.min() + eps)/(df - df.min()).max()

def logit(p, eps=1e-9):
    """added epsilon to prevent infinities"""
    return np.log(eps + p/(1.- p + eps))

def sigmoid(p, eps=1e-9):
    """added epsilon to invert regularized logit above"""
    return ((1 + eps)*(-eps + np.exp(p)))/(1 - eps + np.exp(p))


pmssm_dir = os.path.join(dataset_dir, 'pMSSM/')
pMSSM = import_data(pmssm_dir, 'pmssm')

valid_pMSSM = pMSSM[pMSSM.omega != -1.]

valid_pMSSM = valid_pMSSM.sample(frac=1) # shuffle

obs_cols = ['omega', 'mh']
param_cols  = ['m1g', 'm2g', 'm3g', 'mmug', 
               'mAg', 'Abg', 'Atg', 'Ataug', 
               'meLg','mtaLg', 'meRg', 'mtaRg', 
               'muLg', 'mtLg', 'muRg', 'mtRg', 
               'mdLg','mbLg', 'tanb'] 


obs = torch.tensor(valid_pMSSM[obs_cols].values)
params = torch.tensor(valid_pMSSM[param_cols].values)


# add noise to obs (need to do bc its a likelihood)
delta_omega = 0.001
delta_mh = 1.

noise_obs = torch.randn(obs.shape)*torch.tensor([delta_omega, delta_mh])

obs = obs + noise_obs

# take log omega to make the distribution more manageable
obs[:, 0] = obs[:, 0].log()

# some omega go negative so need to drop them as they are nan
nan_idx = torch.any(obs.isnan(),dim=1)
params = params[~nan_idx]
obs = obs[~nan_idx]
