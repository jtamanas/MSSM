import pandas as pd
import torch
import jax
import jax.numpy as np
import numpy as onp
dataset_dir = "../../datasets/Hollingsworth"

import sys
import os
sys.path.append(os.path.abspath(dataset_dir))
from read_dataset import import_data


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
        

class AddNoiseDataset(torch.utils.data.TensorDataset):
    def __init__(self, 
                 obs: torch.Tensor, 
                 thetas: torch.Tensor, 
                 sigma_obs: torch.Tensor, 
                 use_logit: bool = True,
                 eps_logit: float = 1e-5,
                 rand_key=None,
                ) -> None:
        assert obs.shape[0] == thetas.shape[0], "Size mismatch between tensors"
        self.obs = obs
        self.thetas = thetas
        self.sigma_obs = sigma_obs
        self.use_logit = use_logit
        self.eps_logit = eps_logit
        self.rand_key = rand_key

        # log omega
        _obs = obs.clone()
        _obs[:, 0] = torch.log(obs[:, 0])
        self.obs_mean = torch.mean(_obs, dim=0) 
        self.obs_std = torch.std(_obs, dim=0)
        
        if use_logit:
            self.thetas_min, self.thetas_max = thetas.min(axis=0).values, thetas.max(axis=0).values
        
    def transform(self, noisy_obs, thetas):
        # log omega
        noisy_obs[..., 0] = torch.log(torch.abs(noisy_obs[..., 0]))
        noisy_obs = (noisy_obs - self.obs_mean)/self.obs_std
        # min max norm
        thetas = (thetas - self.thetas_min)/(self.thetas_max - self.thetas_min)
        thetas = torch_logit(thetas, eps=self.eps_logit)
        return noisy_obs, thetas
    
    def inverse_transform(self, noisy_obs, thetas):
        noisy_obs = noisy_obs*self.obs_std + self.obs_mean
        noisy_obs[..., 0] = torch.exp(noisy_obs[..., 0])
        # invert min max norm
        thetas = torch_sigmoid(thetas, eps=self.eps_logit)
        thetas = thetas*(self.thetas_max - self.thetas_min) + self.thetas_min
        return noisy_obs, thetas
    
    def __len__(self):
        return self.obs.shape[0]
            
    def __getitem__(self, idx):
        noisy_obs = self.obs[idx] + torch.rand_like(self.obs[idx])*self.sigma_obs
        thetas = self.thetas[idx]
        
        if self.use_logit:
            noisy_obs, thetas = self.transform(noisy_obs, thetas)
            
        return noisy_obs, thetas
    

def torch_logit(p, eps=1e-9):
    """added epsilon to prevent infinities"""
    return torch.log(eps + p/(1.- p + eps))

def torch_sigmoid(p, eps=1e-9):
    """added epsilon to invert regularized logit above"""
    return ((1 + eps)*(-eps + torch.exp(p)))/(1 - eps + torch.exp(p))

def logit(p, eps=1e-9):
    """added epsilon to prevent infinities"""
    return np.log(eps + p/(1.- p + eps))

def sigmoid(p, eps=1e-9):
    """added epsilon to invert regularized logit above"""
    return ((1 + eps)*(-eps + np.exp(p)))/(1 - eps + np.exp(p))


def get_dataset(N=100000, sigma_obs=None, rand_key=None, use_logit=True):
    if sigma_obs is None:
        sigma_omega = 0.001
        sigma_mh = 2.
        sigma_obs = np.array([sigma_omega, sigma_mh])
    cmssm_dir = os.path.join(dataset_dir, 'pMSSM/')
    cMSSM = import_data(cmssm_dir, 'pmssm')

    valid_cMSSM = cMSSM[cMSSM.omega != -1.]


    obs_cols = ['omega', 'mh']
    theta_cols  = ['m1g', 'm2g', 'm3g', 'mmug', 
                   'mAg', 'Abg', 'Atg', 'Ataug', 
                   'meLg','mtaLg', 'meRg', 'mtaRg', 
                   'muLg', 'mtLg', 'muRg', 'mtRg', 
                   'mdLg','mbLg', 'tanb'] 
    obs_dim = len(obs_cols)
    valid_cMSSM = valid_cMSSM[obs_cols + theta_cols]
    
    valid_cMSSM = jax.random.permutation(rand_key, valid_cMSSM.values)
    
    obs = valid_cMSSM[:, :obs_dim]
    thetas = valid_cMSSM[:, obs_dim:]

    Dataset = AddNoiseDataset(obs=torch.tensor(onp.array(obs[:N])), 
                                   thetas=torch.tensor(onp.array(thetas[:N])), 
                                   sigma_obs=torch.tensor(onp.array(sigma_obs)), 
                                   use_logit=use_logit,
                                   eps_logit=1e-2,
                                   rand_key=rand_key, 
                                   )
    
    return Dataset
