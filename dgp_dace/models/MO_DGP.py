
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
from ..utils.layers import SVGP_Layer
from gpflow.likelihoods import Gaussian
from gpflow import set_trainable
from gpflow.optimizers import NaturalGradient

float_type = gpflow.default_float()
_log = logging.getLogger(__name__)


def sample(layer,Z,num_samples=50):
    Z = tf.tile(tf.expand_dims(Z, 0), [num_samples, 1, 1])
    return tf.reduce_mean(layer.sample_from_conditional(Z)[0],axis=0)


def sample_Z_right(layers,Z):
    for i,layer in enumerate(layers):
        if i==0:
            Z_right = sample(layer,Z)
        Z_aug = tf.concat([Z, Z_right], 1)
        Z_right = sample(layer,Z_aug)
    return Z_right

def init_layers_mf(Z, kernels, num_outputs=None, Layer=SVGP_Layer):
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
    num_layers = len(Z)
    layers.append(Layer(kernels[0], Z[0], num_outputs, Zero()))
    for i in range(1, num_layers):
        layers.append(Layer(kernels[i], Parameter(Z[i], dtype=float_type), num_outputs, Zero(),augmented=True,layers=layers[:i]))
    return layers


class DGP_Base(Module):
    """
    The base class for Deep Gaussian process models.

    Implements a Monte-Carlo variational bound and convenience functions.
    """

    def __init__(self, likelihood, layers, minibatch_size=None, num_samples=1,loop=2, **kwargs):
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

        self._train_upto_objective = -1
        self.num_layers = len(layers)
        self.layers = layers
        self.loop = loop
        self.likelihood = BroadcastingLikelihood(likelihood)

    def propagate(self, X, full_cov=False, S=1, zs=None):
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
        F = sX
        zs = zs or [None, ] * len(self.layers)
        F = tf.random.normal([tf.shape(X)[0], 1],mean =0.0,stddev=1.0,dtype= tf.float64)
        F = tf.tile(tf.expand_dims(F, 0), [S, 1, 1])
        F_aug = tf.concat([sX, F], 2)
        F, Fmean, Fvar = self.layers[0].sample_from_conditional(F_aug, z=zs[0], full_cov=full_cov)
        if self.loop ==0:
            F_aug = tf.concat([sX, F], 2)
            F, Fmean, Fvar = self.layers[1].sample_from_conditional(F_aug, z=zs[1], full_cov=full_cov)
        else:
            for j in range(2*self.loop):
                k=(j+1)%2 
                F_aug = tf.concat([sX, F], 2)
                F, Fmean, Fvar = self.layers[k].sample_from_conditional(F_aug, z=zs[k], full_cov=full_cov)
        Fs.append(F)
        Fmeans.append(Fmean)
        Fvars.append(Fvar)
        F_aug = tf.concat([sX, F], 2)
        F, Fmean, Fvar = self.layers[1].sample_from_conditional(F_aug, z=zs[1], full_cov=full_cov)
        Fs.append(F)
        Fmeans.append(Fmean)
        Fvars.append(Fvar)
        return Fs, Fmeans, Fvars
    
    def predict_f(self, X, full_cov=False, S=1, objective=None):
        """
        Predicts from the fidelity level specified. If fidelity is not specified, return prediction at highest fidelity.

        :param X: Location at which to predict
        :param full_cov: Whether to predict full covariance matrix
        :param S: Number of samples to use for MC sampling between layers
        :param fidelity: zero based fidelity index at which to predict
        :return: (mean, variance) where each is [S, N, 1] where S is number of samples and N is number of predicted points.
        """

        if objective is None:
            objective = -1

        _, Fmeans, Fvars = self.propagate(X, full_cov=full_cov, S=S)
        return Fmeans[objective], Fvars[objective]

    def _likelihood_at_objective(self, Fmu, Fvar, Y, variance):
        """
        Calculate likelihood term for observations corresponding to one objective

        :param Fmu: Posterior mean
        :param Fvar: Posterior variance
        :param Y: training observations
        :param variance: likelihood variance
        :return:
        """
        return -0.5 * np.log(2 * np.pi) - 0.5 * tf.math.log(variance) \
               - 0.5 * (tf.square(Y - Fmu) + Fvar) / variance

    def E_log_p_Y(self, X_o, Y_o, objective=None):
        """
        Calculate the expectation of the data log likelihood under the variational distribution with MC samples

        :param X_o: Training inputs for a given
        :param Y_o:
        :param objective:
        :return:
        """

        Fmean, Fvar = self.predict_f(X_o, full_cov=False, S=self.num_samples, objective=objective)

        if objective == (self.num_layers - 1):
            """
            KC - The likelihood of the observations at the last layer is computed using the model's 'likelihood' object
            """
            var_exp = self.likelihood.variational_expectations(Fmean, Fvar, Y_o)  # S, N, D
        else:
            """
            KC - The Gaussian likelihood of the observations at the intermediate layers is computed using the noise 
            parameter pertaining to the White noise kernel.

            This assumes that a White kernel should be added to all layers except for the last!
            If no noise is desired, the variance parameter in the White kernel should be set to zero and fixed.
            """
            variance = self.layers[objective].kern.kernels[-1].variance

            f = lambda vars_SND, vars_ND, vars_N: self._likelihood_at_objective(vars_SND[0],
                                                                               vars_SND[1],
                                                                               vars_ND[0],
                                                                               vars_N)

            var_exp = f([Fmean, Fvar], [tf.expand_dims(Y_o, 0)], variance)

        return tf.reduce_mean(var_exp, 0)  # N, D

    def ELBO(self,data,tf_sample_Z_right=True):
        """
        ELBO calculation
        :return: MC estimate of lower bound
        """
        if tf_sample_Z_right:
            for i in range(1,len(self.layers)):
                self.layers[i].feature.Z_right = sample_Z_right(self.layers[0:i],(self.layers[i].feature.Z_left))
                self.layers[i].feature.Z=tf.concat([self.layers[i].feature.Z_left,self.layers[i].feature.Z_right],1)
        X,Y =data
        L = 0.
        KL = 0.
        for objective in range(self.num_layers):

            if (self._train_upto_objective != -1) and (objective > self._train_upto_fidelity):
                continue

            X_l = X[objective]
            Y_l = Y[objective]

            n_data = X_l.shape[0]
            scale = tf.cast(n_data, float_type)/tf.cast(tf.shape(X_l)[0], float_type)

            L += (tf.reduce_sum(self.E_log_p_Y(X_l, Y_l, objective)) * scale)
            KL += tf.reduce_sum(self.layers[objective].KL())
        self.L = L
        self.KL = KL
        return self.L - self.KL

    @tf.function
    def ELBO_closure(self,data,tf_sample_Z_right=True):
        def closure():
            return self.ELBO(data,tf_sample_Z_right=tf_sample_Z_right)
        return closure()

    @tf.function
    def propagate_closure(self,X, full_cov=False, S=1, zs=None):
        def closure():
            return self.propagate(X,full_cov=False, S=1, zs=None)
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
    def make_mf_dgp(cls, Z,loop=2, add_linear=True, minibatch_size=None):
        """
        Constructor for convenience. Constructs a mo-dgp model from training data and inducing point locations

        :param X: List of target
        :param Y:
        :param Z:
        :param add_linear:
        :return:
        """

        n_fidelities = len(Z)

        Din = Z[0].shape[1]
        Dout = 1
        print(Din)
        # kernels = [RBF(active_dims=list(range(Din)), variance=1., lengthscales=[1]*Din)]
        kernels=[]
        for l in range(0, n_fidelities):
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

        layers = init_layers_mf( Z, kernels, num_outputs=Dout)

        model = DGP_Base(Gaussian(), layers, loop = loop, num_samples=10, minibatch_size=minibatch_size)

        return model

    def fix_inducing_point_locations(self):
        """
        Fix all inducing point locations
        """
        for layer in self.layers:
            layer.feature.Z.trainable = False

class MultiObjDeepGP(Module):
    """
    Inducing points are fixed for first part of optimization then freed.
    Both sets of inducing points are initialized at low fidelity training data locations.
    """

    def __init__(self, X, Y, Z=None, n_iter=5000, loop=2,fix_inducing=True, training=True, minibatch_size=None):
        super().__init__(name="mo_dgp")
        self._Y = Y
        self._X = X
        self.minibatch_size = minibatch_size
        self.loop = loop
        
        if Z is None:
            self.Z = self._make_inducing_points(X, Y)
        else:
            self.Z = Z

        self.model = self._get_model(self.Z)
        self.n_fidelities = len(X)
        self.n_iter = n_iter
        self.fix_inducing = fix_inducing
        for i in range(len(X)):
            _log.info('\nData at Objective {}'.format(i + 1))
            _log.info('X - {}'.format(X[i].shape))
            _log.info('Y - {}'.format(Y[i].shape))
            _log.info('Z - {}'.format(self.Z[i].shape))

    def _get_model(self, Z):
        return DGP_Base.make_mf_dgp(Z, loop=self.loop,minibatch_size=self.minibatch_size)

    def predict(self, X_test,full_cov=False):
        # assume high fidelity only!!!!
        y_m, y_v = self.model.predict_y(X_test, 250,full_cov=full_cov)
        y_mean_high = np.mean(y_m, axis=0).flatten()
        y_var_high = np.mean(y_v, axis=0).flatten() + np.var(y_m, axis=0).flatten()
        return y_mean_high[:, None], y_var_high[:, None]
    def objective(self):
        return self.model.ELBO((self._X,self._Y))
    
    def optimize_adam(self,lr = 0.01,iterations1=2000,iterations2=5000 ,iterations3=7500,messages=500):
        """
        An Adam optimization method for the training of DGPs. The approach is devided into three parts:
            -In the first part  only the kernel parameters are optimized 
            while the inducing inputs and variational parameters are fixed.
            - In the second part the inducing inputs are unfixed.
            - In the third part the variational parameters and the likelihood variance are unfixed.

        :param iterations i: the number of iterations of part i.
        :param lr: the learning rate for the adam optimizer. Defaults to 0.01
        :param beta_1: The exponential decay rate for the 1st moment estimates. Defaults to 0.9
        :param beta_2: The exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
        :param  epsilon: 	A small constant for numerical stability. Defaults to 1e-8
        :param messages: int to print the evaluation of the ELBO. Default to 100 iterations
        """
        optimizer = tf.optimizers.Adam(learning_rate=lr,epsilon=1e-8)
        self.model.layers[0].q_mu.assign(self._Y[0]) 
        set_trainable(self.model.layers[0].q_mu,False)
        for i, layer in enumerate(self.model.layers[1:]):
            layer.q_mu.assign(self._Y[i+1])
            set_trainable(layer.q_mu,False)
            layer.q_sqrt.assign(layer.q_sqrt.numpy()  *1e-5* self._Y[i].var())
            set_trainable(layer.q_sqrt,False)
        self.model.layers[-1].q_sqrt.assign( self.model.layers[-1].q_sqrt.numpy() * self._Y[-1].var())
        set_trainable(self.model.layers[-1].q_sqrt, False)
        self.model.likelihood.likelihood.variance.assign(self._Y[-1].var() * 1e-2)
        set_trainable(self.model.likelihood.likelihood.variance,False)
        set_trainable(self.model.layers[0].feature.Z, False)
        for layer in self.model.layers[1:]:
            set_trainable(layer.feature.Z_left,False)
        print('Training part 1')
        for _ in range(iterations1):
            with tf.GradientTape(watch_accessed_variables=True) as tape:
                tape.watch(self.model.trainable_variables)
                objective = -self.model.ELBO_closure((self._X, self._Y),tf_sample_Z_right=True)
                gradients = tape.gradient(objective, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            if _%messages==0:
                print(f"ELBO: {-objective.numpy()}")
        set_trainable(self.model.layers[0].feature.Z, True)
        for layer in self.model.layers[1:]:
            set_trainable(layer.feature.Z_left, True)
        for i in range(1,len(self.model.layers)):
            self.model.layers[i].feature.Z_right = sample_Z_right(self.model.layers[0:i],(self.model.layers[i].feature.Z_left))
            self.model.layers[i].feature.Z=tf.concat([self.model.layers[i].feature.Z_left,self.model.layers[i].feature.Z_right],1)
        print('Training part 2')
        for _ in range(iterations2):
            with tf.GradientTape(watch_accessed_variables=True) as tape:
                tape.watch(self.model.trainable_variables)
                objective = -self.model.ELBO_closure((self._X, self._Y),tf_sample_Z_right=True)
                gradients = tape.gradient(objective, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            if _%messages==0:
                print(f"ELBO: {-objective.numpy()}")
        for i in range(1,len(self.model.layers)):
                self.model.layers[i].feature.Z_right = sample_Z_right(self.model.layers[0:i],(self.model.layers[i].feature.Z_left))
                self.model.layers[i].feature.Z=tf.concat([self.model.layers[i].feature.Z_left,self.model.layers[i].feature.Z_right],1)
        set_trainable(self.model.likelihood.likelihood.variance, True)
        for i, layer in enumerate(self.model.layers):
            set_trainable(layer.q_mu,True)
            set_trainable(layer.q_sqrt,True)
        print('Training part 3')
        for _ in range(iterations3):
            with tf.GradientTape(watch_accessed_variables=True) as tape:
                tape.watch(self.model.trainable_variables)
                objective = -self.model.ELBO_closure((self._X, self._Y),tf_sample_Z_right=True)
                gradients = tape.gradient(objective, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            if _%messages==0:
                print(f"ELBO: {-objective.numpy()}")
        for i in range(1,len(self.model.layers)):
            self.model.layers[i].feature.Z_right = sample_Z_right(self.model.layers[0:i],(self.model.layers[i].feature.Z_left))
            self.model.layers[i].feature.Z=tf.concat([self.model.layers[i].feature.Z_left,self.model.layers[i].feature.Z_right],1)
    def optimize_nat_adam(self,lr_adam = 0.01,lr_gamma = 0.01,iterations1=2000,iterations2=5000 ,iterations3=7500,messages=500):
        """
        A training approach for MO-DGP taking into account the differential geometry of the parameter space using natural gradient
        The training approach is divided into three parts. 
            -in the first part only the kernel parameters are optimized using an Adam optimizer
            while the variational parameters and the inducing inputs are fixed.
            - in the second part the inducing inputs are unfixed.
            - in the third part a loop procedure is used where iteratively an Adam step in the Euclidean space and a natural 
            gradient step in the variational parameter space are performed.

        :param iterations i: the number of iterations for part 2.
        :param lr_adam: the learning rate for the adam optimizer. Defaults to 0.01
        :param lr_gamma: the learning rate for the natural gradient. Defaults to 0.01
        :param beta_1: Adam parameter, the exponential decay rate for the 1st moment estimates. Defaults to 0.9
        :param beta_2: Adam parameter, the exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
        :param  epsilon: Adam parameter, a small constant for numerical stability. Defaults to 1e-8
        :param  ng_all: Boolean parameter. True natural gradient  for the variational parameters at each layer. False natural
        gradient only at the last layer.
        :param messages: int to print the evaluation of the ELBO. Default to 100 iterations.
            """
        optimizer = tf.optimizers.Adam(learning_rate=lr_adam,epsilon=1e-8)
        optimizer_nat = NaturalGradient(gamma=lr_gamma)
        self.model.layers[0].q_mu.assign(self._Y[0]) 
        set_trainable(self.model.layers[0].q_mu,False)
        for i, layer in enumerate(self.model.layers[1:]):
            layer.q_mu.assign(self._Y[i+1])
            set_trainable(layer.q_mu,False)
            layer.q_sqrt.assign(layer.q_sqrt.numpy()  *1e-2* self._Y[i].var())
            set_trainable(layer.q_sqrt,False)
        self.model.layers[-1].q_sqrt.assign( self.model.layers[-1].q_sqrt.numpy() * self._Y[-1].var()*1e-2)
        set_trainable(self.model.layers[-1].q_sqrt, False)
        self.model.likelihood.likelihood.variance.assign(self._Y[-1].var() * 1e-2)
        set_trainable(self.model.likelihood.likelihood.variance,False)
        set_trainable(self.model.layers[0].feature.Z, False)
        for layer in self.model.layers[1:]:
            set_trainable(layer.feature.Z_left,False)
        print('Training part 1')
        for _ in range(iterations1):
            with tf.GradientTape(watch_accessed_variables=True) as tape:
                tape.watch(self.model.trainable_variables)
                objective = -self.model.ELBO_closure((self._X, self._Y),tf_sample_Z_right=False)
                gradients = tape.gradient(objective, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            if _%messages==0:
                print(f"ELBO: {-objective.numpy()}")
        set_trainable(self.model.layers[0].feature.Z, True)
        for layer in self.model.layers[1:]:
            set_trainable(layer.feature.Z_left, True)
        print('Training part 2')
        for _ in range(iterations2):
            with tf.GradientTape(watch_accessed_variables=True) as tape:
                tape.watch(self.model.trainable_variables)
                objective = -self.model.ELBO_closure((self._X, self._Y),tf_sample_Z_right=True)
                gradients = tape.gradient(objective, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            if _%messages==0:
                print(f"ELBO: {-objective.numpy()}")
        for i in range(1,len(self.model.layers)):
            self.model.layers[i].feature.Z_right = sample_Z_right(self.model.layers[0:i],(self.model.layers[i].feature.Z_left))
            self.model.layers[i].feature.Z=tf.concat([self.model.layers[i].feature.Z_left,self.model.layers[i].feature.Z_right],1)
        set_trainable(self.model.likelihood.likelihood.variance, True)
        variational_params= [[layer.q_mu, layer.q_sqrt ] for layer in self.model.layers]
        def objective_nat():
            return -self.model.ELBO_closure((self._X, self._Y))
        print('Training part 3')
        for _ in range(iterations3):
            with tf.GradientTape(watch_accessed_variables=True) as tape:
                tape.watch(self.model.trainable_variables)
                objective = -self.model.ELBO_closure((self._X, self._Y),tf_sample_Z_right=True)
                gradients = tape.gradient(objective, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            optimizer_nat.minimize(objective_nat, var_list=variational_params)
            if _%messages==0:
                print(f"ELBO: {-objective.numpy()}")
        for i in range(1,len(self.model.layers)):
            self.model.layers[i].feature.Z_right = sample_Z_right(self.model.layers[0:i],(self.model.layers[i].feature.Z_left))
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
        Z = [np.concatenate((X[0].copy(), Y[1].copy()), axis=1)]
        for x, y in zip(X[1:], Y[1:]):
            Z.append(x.copy())
        return Z
