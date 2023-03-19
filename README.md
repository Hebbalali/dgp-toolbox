# dgp-toolbox

This repository accompanies the doctoral thesis [**"Deep gaussian process package for the analysis and optimization of complex systems"**](https://hal.science/tel-03276426/document).

The code is based on GPflow 2.0 and the [Doubly-Stochastic-DGP by Salimbeni *et al*](https://github.com/UCL-SML/Doubly-Stochastic-DGP) implementation of DGP proposed. For multi-fidelity DGP, the base model is based on [multi-fidelity-DGP by Cutajar *et al*](https://github.com/EmuKit/emukit/tree/main/emukit/examples/multi_fidelity_dgp)

## Bayesian Optimization
A Bayesian optimization (BO) class is implemented for coupling DGPs and BO.
SO_BO is a Module class for single objective Unconstrained/constrained Bayesian Optimization using Gaussian processes and deep Gaussian processes. 

## Models
### DGP Model
This model implements the base DGP model. 
This implementation is heavily based on the doubly stochastic implementation DGP implementation of Hugh Salimbeni: 
    @inproceedings{salimbeni2018natural, title={Natural Gradients in Practice: Non-Conjugate Variational Inference in Gaussian Process Models}, 
                   author={Salimbeni, Hugh and Eleftheriadis, Stefanos and Hensman, James}, booktitle={Artificial Intelligence and Statistics}, 
                   year={2018} }
                  
The training of the DGP takes into account the differential geometry of the parameter space using natural gradient
@article{hebbal2020bayesian,
title={Bayesian optimization using deep Gaussian processes with applications to aerospace system design},
author={Hebbal, Ali and Brevault, Loic and Balesdent, Mathieu and Talbi, El-Ghazali and Melab, Nouredine},
journal={Optimization and Engineering},
pages={1--41},
year={2020},
publisher={Springer}
}
### Multi-fidelity deep GP Embedded Mapping 
This model implements the multi-fidelity Embedded mapping for multi-fidelity with variant input space proposed in [**Multi-fidelity modeling with different input domain definitions using deep Gaussian processes.** by A. Hebbal *et al*](https://arxiv.org/pdf/2006.15924.pdf)

### Multi-objective deep GP
This model implements the multi-objective DGP model proposed in [**Deep Gaussian process for multi-objective Bayesian optimization.** by A. Hebbal *et al*] (https://link.springer.com/article/10.1007/s11081-022-09753-0)

## Examples
To show how to use the implemented code five differents notebooks are accessible in the notebooks folder:
-  *nb_DGP_regression*: Regression using deep Gaussian processes.
-  *nb_dgp_BO*: Bayesian optimization using deep Gaussian processes.
-  *nb_mfdgp_improved*: Improved multi-fidelity deep Gaussian process (optimization of the induced points)
-  *nb_mfdgpem*: Multi-fidelity deep GP Embedded Mapping for multi-fidelity with variant input space.
-  *nb_modgp*:  Multi-objective deep Gaussian processes.
