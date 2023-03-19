# dgp-toolbox

This repository accompanies the doctoral thesis [**"Deep gaussian process package for the analysis and optimization of complex systems"**](https://hal.science/tel-03276426/document).

The code is based on GPflow 2.0 and the [Doubly-Stochastic-DGP](https://github.com/UCL-SML/Doubly-Stochastic-DGP) implementation of DGP proposed by Salimbeni *et al*.

## Bayesian Optimization
A Bayesian optimization (BO) class is implemented for coupling DGPs and BO.
SO_BO is a Module class for single objective Unconstrained/constrained Bayesian Optimization using Gaussian processes and deep Gaussian processes. 
The following optimization problem is considered:

Min     f(x)
s.t.    x $\in$ [0,1]^d
        g(x) $\leq$ 0

:param problem: The problem to minimize.
:param X: The input data [n,d] (It is considered between 0 and 1). 
:param Y: The objective function evaluation [n,d].
:param Y: The constrain functions evaluations [n,n_c].
:param DoE_size: The size of initial data (corresponding to n). If X,Y are given, DoE_size is not taken into account.
:Dic model_Y_dic: A dictionary which defines the architecture of the objective function model. It has the following form:
{'layers':l, 'num_units':[q_1,q_2,\ldots,q_l],kernels:['rbf','matern32','matern52',...],num_samples:S}    
    l=0 comes back to a regular GP.
    if num_units is an integer, the same number of units will be applied to the different layers.
    if kernels is a string, the same kernel will be applied to the different layers. (only the 'rbf','matern32', and 'matern52' are implemented) 
:Dic model_C_dic: A list of n_c dictionaries defining the architecture of the constraint functions. It has the following form:
{'layers':l, 'num_units':[q_1,q_2,\ldots,q_l],kernels:['rbf','matern32','matern52',...],num_samples:S}   
    if only one dictionary instead of a list is given, the same model archtiecture will be applied to the different constraint models.  
Three outputs are obtained Fs, Fmeans, Fvars
:normalize_input=True: A boolean parameter. If true X,Y,C are normalized and standarized.
