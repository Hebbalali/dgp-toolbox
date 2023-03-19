"""
This file contains the multi-fidelity deep Gaussian process model from:
Deep Gaussian Processes for Multi-fidelity Modeling (Kurt Cutajar, Mark Pullin, Andreas Damianou, Neil Lawrence, Javier González)

The class intended for public consumption is MultiFidelityDeepGP, which is an emukit model class.

This file requires the following packages:
- tensorflow
- gpflow
- doubly_stochastic_dgp https://github.com/ICL-SML/Doubly-Stochastic-DGP/tree/master/doubly_stochastic_dgp
"""
import logging
from typing import List, Tuple

import numpy as np
import gpflow
import tensorflow as tf
from gpflow.base import Module, Parameter
from gpflow.mean_functions import Zero
from gpflow.models.model import GPModel, MeanAndVariance
from gpflow.kernels import *
from ..utils.layer_initializations import init_layers_linear
from ..utils.utils import BroadcastingLikelihood
from ..utils.layers_red import SVGP_Layer
from gpflow.likelihoods import Gaussian
from gpflow import set_trainable
from gpflow.optimizers import NaturalGradient

float_type = gpflow.default_float()
_log = logging.getLogger(__name__)


def sample(layer,Z,num_samples=50):
    Z = tf.tile(tf.expand_dims(Z, 0), [num_samples, 1, 1])
    return tf.reduce_mean(layer.sample_from_conditional(Z)[0],axis=0)


def sample_Z_right(layers,layers_red,Z):
    H=Z
    Hs=[H]
    if layers_red is not None:    
        for i, layer_red in enumerate(layers_red):
            H = sample(layer_red, H)
            Hs.append(H)
        for i, layer in enumerate(layers):
            if i == 0:
                Z_right= sample(layer,Hs[-1])
            else:
                Z_aug = tf.concat([Hs[-(i+1)], Z_right], 1)
                Z_right = sample(layer,Z_aug)
    else:
        for i,layer in enumerate(layers):
            if i==0:
                Z_right = sample(layer,Z)
            else:
                Z_aug = tf.concat([Z, Z_right], 1)
                Z_right = sample(layer,Z_aug)
    return Z_right

def init_layers_mf(X,Z, W, kernels,kernels_red, num_outputs=None, Layer=SVGP_Layer):
    """
    Creates layer objects from initial data

    :param Y: Numpy array of training targets
    :param Z: List of numpy arrays of inducing point locations for each layer
    :param kernels: List of kernels for each layer
    :param num_outputs: Number of outputs (same for each layer)
    :param Layer: The layer object to use
    :return: List of layer objects with which to build a multi-fidelity deep Gaussian process model
    """
    num_outputs = num_outputs or 1
    
    
    layers = []
    layers_red = []
    num_layers = len(Z)
    for i in range(1, len(X)):
        num_outputs_red = X[-(1+i)].shape[1]
        layers_red.append(Layer(kernels_red[i-1], W[i-1], num_outputs_red,Zero()))
        
    L=len(layers_red)
    
    layers.append(Layer(kernels[0], Z[0], num_outputs, Zero()))
    for i in range(1, num_layers):
        layers.append(Layer(kernels[i], Parameter(Z[i], dtype=float_type), num_outputs, Zero(),augmented=True,layers=layers[:i],layers_red=layers_red[L-i:]))
    return layers, layers_red


class DGP_Base(Module):
    """
    The base class for Deep Gaussian process models.

    Implements a Monte-Carlo variational bound and convenience functions.
    """

    def __init__(self, likelihood, layers,layers_red, minibatch_size=None, num_samples=1, **kwargs):
        """

        :param X: List of training inputs where each element of the list is a numpy array corresponding to the inputs of one fidelity.
        :param Y: List of training targets where each element of the list is a numpy array corresponding to the inputs of one fidelity.
        :param likelihood: gpflow likelihood object for use at the final layer
        :param layers: List of doubly_stochastic_dgp.layers.Layer objects
        :param minibatch_size: Minibatch size if using minibatch trainingz
        :param num_samples: Number of samples when propagating predictions through layers
        :param kwargs: kwarg inputs to gpflow.models.Model
        """

        Module.__init__(self, **kwargs)

        self.minibatch_size = minibatch_size

        self.num_samples = num_samples

        self._train_upto_fidelity = -1
        self.num_layers = len(layers)
        self.layers = layers
        self.layers_red = layers_red

        self.likelihood = BroadcastingLikelihood(likelihood)
        likelihood_projection = gpflow.likelihoods.Gaussian()
        self.likelihood_projection = BroadcastingLikelihood(likelihood_projection)

    def propagate(self, X, full_cov=False, S=1, zs=None, ws=None,fidelity_dim=None,project=False):
        """
        Propagate some prediction to the final layer and return predictions at each intermediate layer

        :param X: Input(s) at which to predict at
        :param full_cov: Whether the predict with the full covariance matrix
        :param S: Number of samples to use for sampling at intermediate layers
        :param zs: ??
        :return:
        """
        sX = tf.tile(tf.expand_dims(X, 0), [S, 1, 1])
        Fs, Fmeans, Fvars = [], [], []
        Hs = []
        F = sX
        H = sX
        Hs.append(H)
        Hmeans=[]
        Hvars=[]
        L = len(self.layers_red)
        if fidelity_dim==None:
            fidelity_dim = L
        zs = zs or [None, ] * (L+1)
        ws = ws or [None, ] * L
        for i, (layer_red, w) in enumerate(zip(self.layers_red[L-fidelity_dim:], ws[L-fidelity_dim:])):
            H, Hmean , Hvar = layer_red.sample_from_conditional(H, z=w, full_cov=full_cov)
            Hs.append(H)
            Hmeans.append(Hmean)
            Hvars.append(Hvar)
            
        if project:
            return Hs,Hmeans,Hvars
        else:
            for i, (layer, z) in enumerate(zip(self.layers[:fidelity_dim+1], zs[:fidelity_dim+1])):
                if i == 0:
                    F, Fmean, Fvar = layer.sample_from_conditional(Hs[-1], z=z, full_cov=full_cov)
                else:
                    '''
                    KC - At all layers 1..L, the input to the next layer is original input augmented with 
                    the realisation of the function at the previous layer at that input.
                    '''
                    F_aug = tf.concat([Hs[-(i+1)], F], 2)
                    F, Fmean, Fvar = layer.sample_from_conditional(F_aug, z=z, full_cov=full_cov)
                Fs.append(F)
                Fmeans.append(Fmean)
                Fvars.append(Fvar)
            return Fs, Fmeans, Fvars
    def predict_f(self, X, full_cov=False, S=1, fidelity=None,fidelity_dim=None):
        """
        Predicts from the fidelity level specified. If fidelity is not specified, return prediction at highest fidelity.

        :param X: Location at which to predict
        :param full_cov: Whether to predict full covariance matrix
        :param S: Number of samples to use for MC sampling between layers
        :param fidelity: zero based fidelity index at which to predict
        :return: (mean, variance) where each is [S, N, 1] where S is number of samples and N is number of predicted points.
        """

        if fidelity is None:
            fidelity = -1

        _, Fmeans, Fvars = self.propagate(X, full_cov=full_cov, S=S,fidelity_dim=fidelity_dim)
        return Fmeans[fidelity], Fvars[fidelity]


    def project(self, X, full_cov=False, S=1, fidelity=None,fidelity_dim=None):
        """
        Predicts from the fidelity level specified. If fidelity is not specified, return prediction at highest fidelity.

        :param X: Location at which to predict
        :param full_cov: Whether to predict full covariance matrix
        :param S: Number of samples to use for MC sampling between layers
        :param fidelity: zero based fidelity index at which to predict
        :return: (mean, variance) where each is [S, N, 1] where S is number of samples and N is number of predicted points.
        """

        if fidelity is None:
            fidelity = -1

        _, Hmeans, Hvars = self.propagate(X, full_cov=full_cov, S=S,fidelity_dim=fidelity_dim,project=True)

        return  Hmeans[fidelity], Hvars[fidelity]
    
    def _likelihood_at_fidelity(self, Fmu, Fvar, Y, variance):
        """
        Calculate likelihood term for observations corresponding to one fidelity

        :param Fmu: Posterior mean
        :param Fvar: Posterior variance
        :param Y: training observations
        :param variance: likelihood variance
        :return:
        """
        return -0.5 * np.log(2 * np.pi) - 0.5 * tf.math.log(variance) \
               - 0.5 * (tf.square(Y - Fmu) + Fvar) / variance

    def E_log_p_Y(self, X_f, Y_f, fidelity=None,fidelity_dim=None,project=False):
        """
        Calculate the expectation of the data log likelihood under the variational distribution with MC samples

        :param X_f: Training inputs for a given
        :param Y_f:
        :param fidelity:
        :return:
        """
        if project:
            Hmean, Hvar = self.project(X_f, full_cov=False, S=self.num_samples, fidelity=fidelity,fidelity_dim=fidelity_dim)
            # variance = self.layers_red[0].kern.kernels[-1].variance
            # variance = self.layers_red[-(1+fidelity)].kern.kernels[-1].variance
            # f = lambda vars_SND, vars_ND, vars_N: self._likelihood_at_fidelity(vars_SND[0],
            #                                                                     vars_SND[1],
            #                                                                     vars_ND[0],
            #                                                                     vars_N)
            # var_exp = f([Hmean, Hvar], [tf.expand_dims(Y_f, 0)], variance)
            var_exp = self.likelihood_projection.variational_expectations(Hmean, Hvar, Y_f)
        else:
            Fmean, Fvar = self.predict_f(X_f, full_cov=False, S=self.num_samples, fidelity=fidelity,fidelity_dim=fidelity_dim)
            if fidelity == (self.num_layers - 1):
                """
                KC - The likelihood of the observations at the last layer is computed using the model's 'likelihood' object
                """
                var_exp = self.likelihood.variational_expectations(Fmean, Fvar, Y_f)  # S, N, D
            else:
                """
                KC - The Gaussian likelihood of the observations at the intermediate layers is computed using the noise 
                parameter pertaining to the White noise kernel.

                This assumes that a White kernel should be added to all layers except for the last!
                If no noise is desired, the variance parameter in the White kernel should be set to zero and fixed.
                """
                variance = self.layers[fidelity].kern.kernels[-1].variance

                f = lambda vars_SND, vars_ND, vars_N: self._likelihood_at_fidelity(vars_SND[0],
                                                                                vars_SND[1],
                                                                                vars_ND[0],
                                                                                vars_N)

                var_exp = f([Fmean, Fvar], [tf.expand_dims(Y_f, 0)], variance)
        return tf.reduce_mean(var_exp, 0)  # N, D

    def ELBO(self,data):
        """
        ELBO calculation
        :return: MC estimate of lower bound
        """
        X,Y,X_red =data
        L = len(self.layers_red)
        for i in range(1,len(self.layers)):
                    self.layers[i].feature.Z_right = sample_Z_right(self.layers[0:i],self.layers_red[L-i:],(self.layers[i].feature.Z_left))
                    self.layers[i].feature.Z=tf.concat([self.layers[i].feature.Z_left,self.layers[i].feature.Z_right],1)
        L = 0.
        KL = 0.
        L_red=0.
        KL_red = 0.
        for fidelity in range(self.num_layers):

            if (self._train_upto_fidelity != -1) and (fidelity > self._train_upto_fidelity):
                continue

            X_l = X[fidelity]
            Y_l = Y[fidelity]

            n_data = X_l.shape[0]
            scale = tf.cast(n_data, float_type)/tf.cast(tf.shape(X_l)[0], float_type)

            L += (tf.reduce_sum(self.E_log_p_Y(X_l, Y_l, fidelity,fidelity_dim=fidelity)) * scale)
            KL += tf.reduce_sum(self.layers[fidelity].KL())
            if fidelity <self.num_layers-1:
                X_next = X[fidelity+1]
                Y_next = X_red[fidelity]
                n_data = X[fidelity+1].shape[0]
                scale = tf.cast(n_data, float_type)/tf.cast(tf.shape(X_l)[0], float_type)
                L_red += (tf.reduce_sum(self.E_log_p_Y(X_next, Y_next, fidelity,fidelity_dim=fidelity+1,project=True)) * scale)
                KL_red += tf.reduce_sum(self.layers_red[fidelity].KL())
                
        self.L = L
        self.KL = KL
        self.L_red = L_red
        self.KL_red = KL_red
        return self.L + self.L_red - self.KL - self.KL_red

    @tf.function
    def ELBO_closure(self,data):
        def closure():
            return self.ELBO(data)
        return closure()

    def predict_all_layers(self, Xnew, num_samples):
        return self.propagate(Xnew, full_cov=False, S=num_samples)


    def predict_y(self, Xnew, num_samples,full_cov=False):
        Fmean, Fvar = self.predict_f(Xnew, full_cov=full_cov, S=num_samples)
        return self.likelihood.predict_mean_and_var(Fmean, Fvar)


    def predict_density(self, Xnew, Ynew, num_samples):
        Fmean, Fvar = self.predict_f(Xnew, full_cov=False, S=num_samples)
        l = self.likelihood.predict_density(Fmean, Fvar, Ynew)
        log_num_samples = tf.math.log(tf.cast(num_samples, float_type))
        return tf.reduce_logsumexp(l - log_num_samples, axis=0)

    @classmethod
    def make_mf_dgp(cls, X,Z,W, add_linear=True, minibatch_size=None):
        """
        Constructor for convenience. Constructs a mf-dgp model from training data and inducing point locations

        :param X: List of target
        :param Y:
        :param Z:
        :param add_linear:
        :return:
        """

        n_fidelities = len(Z)

        Din = X[0].shape[1]
        Dout = 1

        kernels = [RBF(active_dims=list(range(Din)), variance=1., lengthscales=[1.]*Din)]
        for l in range(1, n_fidelities):
            Din = X[l].shape[1]
            D = Din + Dout
            D_range = list(range(D))
            k_corr = RBF(active_dims=D_range[:Din], variance=1.0)
            k_prev = RBF(active_dims=D_range[Din:], variance=1.)
            k_in = RBF(active_dims=D_range[:Din], variance=1.)
            if add_linear:
                k_l = k_corr * (k_prev + Linear(active_dims=D_range[Din:], variance=1.)) + k_in
            else:
                k_l = k_corr * k_prev + k_in
            kernels.append(k_l)
        kernels_red = [RBF(variance=1., lengthscales=[1.]*X[-(l+1)].shape[1]) for l in range(n_fidelities-1)]
        # kernels_red = [RBF(variance=1., lengthscales=1.) + White( variance=1e-6) for l in range(n_fidelities-1)]
        """
        A White noise kernel is currently expected by Mf-DGP at all layers except the last.
        In cases where no noise is desired, this should be set to 0 and fixed, as follows:

            white = White(1, variance=0.)
            white.variance.trainable = False
            kernels[i] += white
        """
        for i, kernel in enumerate(kernels[:-1]):
            # white = White(1, variance=1e-6)
            # white.variance.trainable = True
            kernels[i] +=  White( variance=1e-6)
        num_data = 0

        layers,layers_red = init_layers_mf(X, Z,W, kernels,kernels_red, num_outputs=Dout)

        model = DGP_Base(Gaussian(), layers,layers_red, num_samples=100, minibatch_size=minibatch_size)

        return model

    def fix_inducing_point_locations(self):
        """
        Fix all inducing point locations
        """
        for layer in self.layers:
            layer.feature.Z.trainable = False

class MultiFidelityDeepGP_EM(Module):
    """
    Inducing points are fixed for first part of optimization then freed.
    Both sets of inducing points are initialized at low fidelity training data locations.
    """

    def __init__(self, X, Y, X_red,Z=None, W=None,n_iter=5000, fix_inducing=True, minibatch_size=None):
        super().__init__(name="mf_dgp_EM")
        self._Y = Y
        self._X = X
        self._X_red = X_red
        self.minibatch_size = minibatch_size

        if Z is None:
            self.Z = self._make_inducing_points(X, Y)
        else:
            self.Z = Z
        if W is None:
            self.W = [X[-1].copy()]
            for i in range(1,len(X)-1):
                self.W.append(X[-(1+i)])
        else:
            self.W = W
        self.model = self._get_model(X, self.Z,self.W)
        self.n_fidelities = len(X)
        self.n_iter = n_iter
        self.fix_inducing = fix_inducing
        for i in range(len(X)):
            _log.info('\nData at Fidelity {}'.format(i + 1))
            _log.info('X - {}'.format(X[i].shape))
            _log.info('Y - {}'.format(Y[i].shape))
            _log.info('Z - {}'.format(self.Z[i].shape))

    def _get_model(self, X, Z, W):
        return DGP_Base.make_mf_dgp( X, Z, W, minibatch_size=self.minibatch_size)

    def predict(self, X_test,full_cov=False):
        # assume high fidelity only!!!!
        y_m, y_v = self.model.predict_y(X_test, 250,full_cov=full_cov)
        y_mean_high = np.mean(y_m, axis=0).flatten()
        y_var_high = np.mean(y_v, axis=0).flatten() + np.var(y_m, axis=0).flatten()
        return y_mean_high[:, None], y_var_high[:, None]
    def objective(self):
        return self.model.ELBO((self._X,self._Y,self._X_red))


    def optimize_adam(self,lr = 0.01,iterations1=2000,iterations2=5000 ,iterations3=7500, beta_1=0.9, beta_2=0.999, epsilon=1e-07, messages=500):

        L=len(self.model.layers_red)
        optimizer = tf.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2,epsilon=epsilon) ### Define the Adam optimizer
        ### Initialization of the variational parameters.
        for i, layer in enumerate(self.model.layers[:-1]):
            layer.q_mu.assign(self._Y[i])
            set_trainable(layer.q_mu,False)
            layer.q_sqrt.assign(layer.q_sqrt.numpy()  *1e-2* self._Y[i].var())
            set_trainable(layer.q_sqrt,False)
        self.model.layers[-1].q_sqrt.assign( self.model.layers[-1].q_sqrt.numpy() * self._Y[-1].var()*1e-2)
        set_trainable(self.model.layers[-1].q_sqrt, False)
        set_trainable(self.model.layers[-1].q_mu, False)
        self.model.layers[-1].q_mu.assign( self._Y[-1])
        for i, layer in enumerate(self.model.layers_red):
            layer.q_sqrt.assign(layer.q_sqrt.numpy() *  1e-2)
            set_trainable(layer.q_sqrt, False)
        for i, layer in enumerate(self.model.layers_red):
            layer.q_mu.assign(self._X_red[-(i+1)])
            set_trainable(layer.q_mu, False)  
        
        ### Initialization of the likelihood variance.
        self.model.likelihood.likelihood.variance.assign(self._Y[-1].var() * 1e-2)
        set_trainable(self.model.likelihood.likelihood.variance,False)
        ### Fixing the inducing inputs
        set_trainable(self.model.layers[0].feature.Z, False)
        for layer in self.model.layers[1:]:
            set_trainable(layer.feature.Z_left,False)
        
        ### Part 1 of the training
        for _ in range(iterations1):
            with tf.GradientTape(watch_accessed_variables=True) as tape:
                tape.watch(self.model.trainable_variables)
                objective = -self.model.ELBO_closure((self._X,self._Y,self._X_red))
                gradients = tape.gradient(objective, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            if _%messages==0:
                print(f"ELBO: {-objective.numpy()}")
            
        ### Part 2 of the training
        set_trainable(self.model.layers[0].feature.Z, True)
        for layer in self.model.layers[1:]:
            set_trainable(layer.feature.Z_left, True)
        for _ in range(iterations2):
            with tf.GradientTape(watch_accessed_variables=True) as tape:
                tape.watch(self.model.trainable_variables)
                objective = -self.model.ELBO_closure((self._X,self._Y,self._X_red))
                gradients = tape.gradient(objective, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            if _%messages==0:
                print(f"ELBO: {-objective.numpy()}")
        for i in range(1,len(self.model.layers)):
            self.model.layers[i].feature.Z_right = sample_Z_right(self.model.layers[0:i],self.model.layers_red[L-i:],(self.model.layers[i].feature.Z_left))
            self.model.layers[i].feature.Z=tf.concat([self.model.layers[i].feature.Z_left,self.model.layers[i].feature.Z_right],1)
            
        ### Part 3 of the training
        set_trainable(self.model.likelihood.likelihood.variance, True)
        for i, layer in enumerate(self.model.layers):
            set_trainable(layer.q_mu,True)
            set_trainable(layer.q_sqrt,True)
        for _ in range(iterations3):
            with tf.GradientTape(watch_accessed_variables=True) as tape:
                tape.watch(self.model.trainable_variables)
                objective = -self.model.ELBO_closure((self._X,self._Y,self._X_red))
                gradients = tape.gradient(objective, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            if _%messages==0:
                print(f"ELBO: {-objective.numpy()}")
        for i in range(1,len(self.model.layers)):
            self.model.layers[i].feature.Z_right = sample_Z_right(self.model.layers[0:i],self.model.layers_red[L-i:],(self.model.layers[i].feature.Z_left))
            self.model.layers[i].feature.Z=tf.concat([self.model.layers[i].feature.Z_left,self.model.layers[i].feature.Z_right],1)

    def optimize_nat_adam(self,lr_adam = 0.01,lr_gamma = 0.01,iterations1=2000,iterations2=5000 ,iterations3=7500,beta_1=0.9, beta_2=0.999, epsilon=1e-07, messages= 500):
        
        L=len(self.model.layers_red)
        optimizer = tf.optimizers.Adam(learning_rate=lr_adam, beta_1=beta_1, beta_2=beta_2,epsilon=epsilon) ### Define the Adam optimizer
        optimizer_nat = NaturalGradient(gamma=lr_gamma) ### Define the Natural optimizer
        ### Initialization of the variational parameters.
        for i, layer in enumerate(self.model.layers[:-1]):
            layer.q_mu.assign(self._Y[i])
            set_trainable(layer.q_mu,False)
            layer.q_sqrt.assign(layer.q_sqrt.numpy()  *1e-3* self._Y[i].var())
            set_trainable(layer.q_sqrt,False)
        self.model.layers[-1].q_sqrt.assign( self.model.layers[-1].q_sqrt.numpy() * self._Y[-1].var()*1e-3)
        set_trainable(self.model.layers[-1].q_sqrt, False)
        set_trainable(self.model.layers[-1].q_mu, False)
        self.model.layers[-1].q_mu.assign( self._Y[-1])
        for i, layer in enumerate(self.model.layers_red):
            layer.q_sqrt.assign(layer.q_sqrt.numpy() *  1e-5)
            set_trainable(layer.q_sqrt, False)
        for i, layer in enumerate(self.model.layers_red):
            layer.q_mu.assign(self._X_red[-(i+1)])
            set_trainable(layer.q_mu, False)        
        self.model.likelihood_projection.likelihood.variance.assign(self._X_red[-1].var() * 1e-3)
        set_trainable(self.model.likelihood_projection.likelihood.variance,False)
        ### Initialization of the likelihood variance.
        self.model.likelihood.likelihood.variance.assign(self._Y[-1].var() * 1e-3)
        set_trainable(self.model.likelihood.likelihood.variance,False)
        ### Fixing the inducing inputs
        set_trainable(self.model.layers[0].feature.Z, False)
        for layer in self.model.layers[1:]:
            set_trainable(layer.feature.Z_left,False)
        
        ### Part 1 of the training
        for _ in range(iterations1):
            with tf.GradientTape(watch_accessed_variables=True) as tape:
                tape.watch(self.model.trainable_variables)
                objective = -self.model.ELBO_closure((self._X,self._Y,self._X_red))
                gradients = tape.gradient(objective, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            if _%messages==0:
                print(f"ELBO: {-objective.numpy()}")
                # print(f"KL: {self.model.KL}")
                # print(f"KL_red: {self.model.KL_red}")
                # print(f"L: {self.model.L}")
                # print(f"L_red: {self.model.L_red}")
            
        ### Part 2 of the training
        set_trainable(self.model.layers[0].feature.Z, True)
        for layer in self.model.layers[1:]:
            set_trainable(layer.feature.Z_left, True)
        for _ in range(iterations2):
            with tf.GradientTape(watch_accessed_variables=True) as tape:
                tape.watch(self.model.trainable_variables)
                objective = -self.model.ELBO_closure((self._X,self._Y,self._X_red))
                gradients = tape.gradient(objective, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            if _%messages==0:
                print(f"ELBO: {-objective.numpy()}")
        for i in range(1,len(self.model.layers)):
            self.model.layers[i].feature.Z_right = sample_Z_right(self.model.layers[0:i],self.model.layers_red[L-i:],(self.model.layers[i].feature.Z_left))
            self.model.layers[i].feature.Z=tf.concat([self.model.layers[i].feature.Z_left,self.model.layers[i].feature.Z_right],1)       
        ### Part 3 of the training
        set_trainable(self.model.likelihood.likelihood.variance, False)
        variational_params= [[layer.q_mu, layer.q_sqrt ] for layer in self.model.layers]
        variational_params += [[l.q_mu, l.q_sqrt] for l in self.model.layers_red]
        def objective_nat():
            return -self.model.ELBO_closure((self._X, self._Y,self._X_red))
        for _ in range(iterations3):
            with tf.GradientTape(watch_accessed_variables=True) as tape:
                tape.watch(self.model.trainable_variables)
                objective = -self.model.ELBO_closure((self._X,self._Y,self._X_red))
                gradients = tape.gradient(objective, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            optimizer_nat.minimize(objective_nat, var_list=variational_params)
            if _%messages==0:
                print(f"ELBO: {-objective.numpy()}")
        for i in range(1,len(self.model.layers)):
            self.model.layers[i].feature.Z_right = sample_Z_right(self.model.layers[0:i],self.model.layers_red[L-i:],(self.model.layers[i].feature.Z_left))
            self.model.layers[i].feature.Z=tf.concat([self.model.layers[i].feature.Z_left,self.model.layers[i].feature.Z_right],1)
    @staticmethod
    def _make_inducing_points(X: List, Y: List) -> List:
        """
        Makes inducing points at every other training point location which is deafult behaviour if no inducing point
        locations are passed

        :param X: training locations
        :param Y: training targets
        :return: List of numpy arrays containing inducing point locations
        """
        # Z = [X[0].copy()]
        # for x, y in zip(X[:-1], Y[:-1]):
        #     Z.append(np.concatenate((x.copy(), y.copy()), axis=1))
        # return Z
        Z = [X[0].copy()]
        for x, y in zip(X[1:], Y[1:]):
            Z.append(x.copy())
        return Z
