# Simulation Based Inference for Efficient Theory Space Sampling: an Application to Supersymmetric Explanations of the Anomalous Muon (g-2)
**Authors:** Logan Morrison, Stefano Profumo, and John Tamanas
<!---[\[arxiv\]](arXiv)[\[bibtex\]](bibtex)-->
[![arXiv](http://img.shields.io/badge/arXiv-2006.00615-cd5c5c.svg)](https://arxiv.org/abs/.)
[![BibTex](http://img.shields.io/badge/BibTex-4682b4.svg)](https://ui.adsabs.harvard.edu/abs//exportcitation)


In [our paper](https://arxiv.org/abs/2006.00615), we introduce the simulation-based inference (SBI) framework to the problem of sampling from experimentally-constrained theory spaces. 


In this repository, you'll find SBI applications to cMSSM and pMSSM parameter space samplings. The main 


# Dependencies

Required python packages are listed in [environment.yml](environment.yml). To create a Conda environment with these dependencies use the following command:

```
conda env create -f environment.yml
```

Additionally, this package relies on:
*  jax-based sbi package available here: https://github.com/jtamanas/LBI
* python-wrapped micromegas package available here: https://github.com/LoganAMorrison/pymicromegas
