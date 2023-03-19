# Copyright 2017 Hugh Salimbeni
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import tensorflow as tf
# import numpy as np

# from gpflow.params import Parameter, Parameterized
# from gpflow.conditionals import conditional
# from gpflow.features import InducingPoints
# from gpflow.kullback_leiblers import gauss_kl
# from gpflow.priors import Gaussian as Gaussian_prior
# from gpflow import transforms
# from gpflow import settings
# from gpflow.models.gplvm import BayesianGPLVM
# from gpflow.expectations import expectation
# from gpflow.probability_distributions import DiagonalGaussian
# from gpflow import params_as_tensors, autoflow
# from gpflow.logdensities import multivariate_normal



# from utils import reparameterize
# float_type = settings.float_type


import tensorflow as tf
import gpflow
from gpflow.base import Module, Parameter
import numpy as np
from gpflow.inducing_variables import InducingPoints
from gpflow.utilities import triangular
import gpflow.covariances as covs

from .utils import reparameterize

class Layer(Module):
    def __init__(self, input_prop_dim=None, **kwargs):
        """
        A base class for GP layers. Basic functionality for multisample conditional, and input propagation
        :param input_prop_dim: the first dimensions of X to propagate. If None (or zero) then no input prop
        :param kwargs:
        """
        Module.__init__(self, **kwargs)
        self.input_prop_dim = input_prop_dim

    def conditional_ND(self, X, full_cov=False):
        raise NotImplementedError

    def KL(self):
        return tf.cast(0.,  dtype=gpflow.default_float())

    def conditional_SND(self, X, full_cov=False):
        """
        A multisample conditional, where X is shape (S,N,D_out), independent over samples S

        if full_cov is True
            mean is (S,N,D_out), var is (S,N,N,D_out)

        if full_cov is False
            mean and var are both (S,N,D_out)

        :param X:  The input locations (S,N,D_in)
        :param full_cov: Whether to calculate full covariance or just diagonal
        :return: mean (S,N,D_out), var (S,N,D_out or S,N,N,D_out)
        """
        if full_cov is True:
            f = lambda a: self.conditional_ND(a, full_cov=full_cov)
            mean, var = tf.map_fn(f, X, dtype=(tf.float64, tf.float64))
            return tf.stack(mean), tf.stack(var)
        else:
            S, N, D = tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2]
            X_flat = tf.reshape(X, [S * N, D])
            mean, var = self.conditional_ND(X_flat)
            return [tf.reshape(m, [S, N, self.num_outputs]) for m in [mean, var]]

    def sample_from_conditional(self, X, z=None, full_cov=False):
        """
        Calculates self.conditional and also draws a sample, adding input propagation if necessary

        If z=None then the tensorflow random_normal function is used to generate the
        N(0, 1) samples, otherwise z are used for the whitened sample points

        :param X: Input locations (S,N,D_in)
        :param full_cov: Whether to compute correlations between outputs
        :param z: None, or the sampled points in whitened representation
        :return: mean (S,N,D), var (S,N,N,D or S,N,D), samples (S,N,D)
        """
        mean, var = self.conditional_SND(X, full_cov=full_cov)

        # set shapes
        S = tf.shape(X)[0]
        N = tf.shape(X)[1]
        D = self.num_outputs

        mean = tf.reshape(mean, (S, N, D))
        if full_cov:
            var = tf.reshape(var, (S, N, N, D))
        else:
            var = tf.reshape(var, (S, N, D))

        if z is None:
            z = tf.random.normal(tf.shape(mean), dtype=gpflow.default_float())
        samples = reparameterize(mean, var, z, full_cov=full_cov)

        if self.input_prop_dim:
            shape = [tf.shape(X)[0], tf.shape(X)[1], self.input_prop_dim]
            X_prop = tf.reshape(X[:, :, :self.input_prop_dim], shape)

            samples = tf.concat([X_prop, samples], 2)
            mean = tf.concat([X_prop, mean], 2)

            if full_cov:
                shape = (tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[1], tf.shape(var)[3])
                zeros = tf.zeros(shape, dtype=gpflow.default_float())
                var = tf.concat([zeros, var], 3)
            else:
                var = tf.concat([tf.zeros_like(X_prop), var], 2)

        return samples, mean, var


def sample(layer,Z,S):
    Z = tf.tile(tf.expand_dims(Z, 0), [S, 1, 1])
    return tf.reduce_mean(layer.sample_from_conditional(Z)[0],axis=0)

def sample_Z_right_array(layer,Z,S):
    return sample(layer,Z,S)

def sample_Z_right_array_all_layers(layers,layers_red,Z,S):
    H=Z
    Hs=[H]
    if layers_red is not None:    
        for i, layer_red in enumerate(layers_red):
            H = sample_Z_right_array(layer_red, H,S)
            Hs.append(H)
        for i, layer in enumerate(layers):
            if i == 0:
                Z_right= sample_Z_right_array(layer,Hs[-1],S)
            else:
                Z_aug = np.concatenate([Hs[-(i+1)], Z_right], 1)
                Z_right = sample_Z_right_array(layer,Z_aug,S)
    else:
        for i,layer in enumerate(layers):
            if i==0:
                Z_right = sample_Z_right_array(layer,Z,S)
            else:
                Z_aug = np.concatenate([Z, Z_right], 1)
                Z_right = sample_Z_right_array(layer,Z_aug,S)
    return Z_right

class SVGP_Layer(Layer):
    def __init__(self, kern, Z, num_outputs, mean_function,augmented=False,layers=None,layers_red=None,
                 white=False, input_prop_dim=None, **kwargs):
        """
        A sparse variational GP layer in whitened representation. This layer holds the kernel,
        variational parameters, inducing points and mean function.

        The underlying model at inputs X is
        f = Lv + mean_function(X), where v \sim N(0, I) and LL^T = kern.K(X)

        The variational distribution over the inducing points is
        q(v) = N(q_mu, q_sqrt q_sqrt^T)

        The layer holds D_out independent GPs with the same kernel and inducing points.

        :param kern: The kernel for the layer (input_dim = D_in)
        :param Z: Inducing points (M, D_in)
        :param num_outputs: The number of GP outputs (q_mu is shape (M, num_outputs))
        :param mean_function: The mean function
        :return:
        """
        Layer.__init__(self, input_prop_dim, **kwargs)
        self.num_inducing = Z.shape[0]
        q_mu = np.zeros((self.num_inducing, num_outputs))
        self.q_mu = Parameter(q_mu, name="q_mu")
        q_sqrt = np.tile(np.eye(self.num_inducing)[None, :, :], [num_outputs, 1, 1])
        # transform = transforms.LowerTriangular(self.num_inducing, num_matrices=num_outputs)
        self.q_sqrt = Parameter(q_sqrt, transform=triangular(), name="q_sqrt")
        if augmented==False:
            self.feature = InducingPoints(Z=Z)
        else:
            self.feature = InducingPoints(layers=layers,layers_red=layers_red,Z=Z)
            Z_right = sample_Z_right_array_all_layers(layers,layers_red,Z.numpy(),100)
            Z = np.concatenate((Z.numpy(),Z_right),1)
        self.kern = kern
        self.mean_function = mean_function
        
        self.num_outputs = num_outputs
        self.white = white

        if not self.white:  # initialize to prior
            Ku = self.kern.K(Z)
            Lu = np.linalg.cholesky(Ku + np.eye(Z.shape[0])*gpflow.default_jitter())
            self.q_sqrt = Parameter(np.tile(Lu[None, :, :], [num_outputs, 1, 1]),transform=triangular(), name="q_sqrt")
        self.needs_build_cholesky = True


    def build_cholesky_if_needed(self):
        # # make sure we only compute this once
        # if self.needs_build_cholesky:
        self.Ku = covs.Kuu(self.feature, self.kern, jitter=gpflow.default_jitter())
        self.Lu = tf.linalg.cholesky(self.Ku)
        self.Ku_tiled = tf.tile(self.Ku[None, :, :], [self.num_outputs, 1, 1])
        self.Lu_tiled = tf.tile(self.Lu[None, :, :], [self.num_outputs, 1, 1])
        self.needs_build_cholesky = False


    def conditional_ND(self, X, full_cov=False):
        self.build_cholesky_if_needed()

        # mmean, vvar = conditional(X, self.feature.Z, self.kern,
        #             self.q_mu, q_sqrt=self.q_sqrt,
        #             full_cov=full_cov, white=self.white)
        Kuf = covs.Kuf(self.feature,self.kern, X)

        A = tf.linalg.triangular_solve(self.Lu, Kuf, lower=True)
        if not self.white:
            A = tf.linalg.triangular_solve(tf.transpose(self.Lu), A, lower=False)

        mean = tf.matmul(A, self.q_mu, transpose_a=True)
        
        A_tiled = tf.tile(A[None, :, :], [self.num_outputs, 1, 1])
        I = tf.eye(self.num_inducing, dtype=gpflow.default_float())[None, :, :]

        if self.white:
            SK = -I
        else:
            SK = -self.Ku_tiled

        if self.q_sqrt is not None:
            SK += tf.matmul(self.q_sqrt, self.q_sqrt, transpose_b=True)


        B = tf.matmul(SK, A_tiled)

        if full_cov:
            # (num_latent, num_X, num_X)
            delta_cov = tf.matmul(A_tiled, B, transpose_a=True)
            Kff = self.kern.K(X)
        else:
            # (num_latent, num_X)
            delta_cov = tf.reduce_sum(A_tiled * B, 1)
            Kff = self.kern.K_diag(X)

        # either (1, num_X) + (num_latent, num_X) or (1, num_X, num_X) + (num_latent, num_X, num_X)
        var = tf.expand_dims(Kff, 0) + delta_cov
        var = tf.transpose(var)

        return mean+ self.mean_function(X), var

    def KL(self):
        """
        The KL divergence from the variational distribution to the prior

        :return: KL divergence from N(q_mu, q_sqrt) to N(0, I), independently for each GP
        """
        # if self.white:
        #     return gauss_kl(self.q_mu, self.q_sqrt)
        # else:
        #     return gauss_kl(self.q_mu, self.q_sqrt, self.Ku)

        self.build_cholesky_if_needed()
        # p_mu = self.mean_function(self.feature.Z)
        # mean = p_mu - self.q_mu 
        KL = -0.5 * self.num_outputs * self.num_inducing
        KL -= 0.5 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.q_sqrt) ** 2))

        if not self.white:
            KL += tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.Lu))) * self.num_outputs
            KL += 0.5 * tf.reduce_sum(tf.square(tf.linalg.triangular_solve(self.Lu_tiled, self.q_sqrt, lower=True)))
            Kinv_m = tf.linalg.cholesky_solve(self.Lu, self.q_mu)
            KL += 0.5 * tf.reduce_sum(self.q_mu * Kinv_m)
            # Kinv_m = tf.cholesky_solve(self.Lu, mean)
            # KL += 0.5 * tf.reduce_sum(mean * Kinv_m)
        else:
            KL += 0.5 * tf.reduce_sum(tf.square(self.q_sqrt))
            KL += 0.5 * tf.reduce_sum(self.q_mu**2)

        return KL
