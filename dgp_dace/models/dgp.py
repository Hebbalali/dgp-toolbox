
"""
This script implements the base DGP model. 
This implementation is heavily based on the doubly stochastic implementation DGP implementation of Hugh Salimbeni: 
    @inproceedings{salimbeni2018natural, title={Natural Gradients in Practice: Non-Conjugate Variational Inference in Gaussian Process Models}, 
                   author={Salimbeni, Hugh and Eleftheriadis, Stefanos and Hensman, James}, booktitle={Artificial Intelligence and Statistics}, 
                   year={2018} }
"""

  
import gpflow
import tensorflow as tf
from gpflow.base import Module
from gpflow.mean_functions import Zero
from gpflow.models.model import GPModel, MeanAndVariance
from gpflow.optimizers import NaturalGradient
from ..utils.layer_initializations import init_layers_linear
from ..utils.utils import BroadcastingLikelihood
from gpflow import set_trainable
import numpy as np
class DGP_Base(Module):
    """
    The base class for Deep Gaussian process models.

    Implements a Monte-Carlo variational bound and convenience functions.

    """
    def __init__(self, likelihood, layers, num_samples=1, **kwargs):
        super().__init__(name="dgp")
        self.num_samples = num_samples ### The number of samples for the estimation of the expectation term in the ELBO
        self.likelihood = BroadcastingLikelihood(likelihood) ### A wrapper for the likelihood to broadcast over the samples dimension.
        self.layers = layers ### The number of layers of the DGP

    def propagate(self, X, full_cov=False, S=1, zs=None):
        """
        Propagte the input through all the layers

        :param X: The inputs to propagate
        :param full_cov: If True, draw correlated samples over the inputs. Computes the Cholesky over the
            dense covariance matrix of size [num_data, num_data].
            If False, draw samples that are uncorrelated over the inputs.
        :param S: the number of samples to propagate
        
        Three outputs are obtained Fs, Fmeans, Fvars
        :output Fs: The list of the propagated samples at each layer
        :output Fmeans: The list of the propagated GP posterior means at each layer
        :output Fvars: The list of the propagated GP posterior variance at each layer
        """
        sX = tf.tile(tf.expand_dims(X, 0), [S, 1, 1]) ### Tiling the inputs to the number of samples

        Fs, Fmeans, Fvars = [], [], []

        F = sX
        zs = zs or [None, ] * len(self.layers)
        for layer, z in zip(self.layers, zs): ### This loop go through the different layers and uses the sample given by the previous layer as input of
            ### the actual layer to obtain the corresponding outputs and so on.
            F, Fmean, Fvar = layer.sample_from_conditional(F, z=z, full_cov=full_cov) 

            Fs.append(F)
            Fmeans.append(Fmean)
            Fvars.append(Fvar)

        return Fs, Fmeans, Fvars

    
    def predict_f(self, X, full_cov=False, S=1):
        """
        Compute the mean and variance obtained at the last layer of the DGP for inputs X

        :param X: The inputs to propagate
        :param full_cov: If True, draw correlated samples over the inputs. Computes the Cholesky over the
            dense covariance matrix of size [num_data, num_data].
            If False, draw samples that are uncorrelated over the inputs.
        :param S: the number of samples to propagate
        """
        Fs, Fmeans, Fvars = self.propagate(X, full_cov=full_cov, S=S)
        return Fmeans[-1], Fvars[-1]
    
    def E_log_p_Y(self, X, Y):
        """
        Calculate the expectation of the data log likelihood under the variational distribution
         with MC samples
        """
        Fmean, Fvar = self.predict_f(X, full_cov=False, S=self.num_samples)
        var_exp = self.likelihood.variational_expectations(Fmean, Fvar, Y)  # Given the likelihood, computes the expectation term 
        # for each sample. output tensor of size [S, N, D] (N the number of data-points, D the output dimension)
        return tf.reduce_mean(var_exp, 0)  # sum over the samples. output tensor of size [N, D] (N the number of data-points, D the output dimension)
    
    def ELBO(self,data):
        """
        Computes the Evidence Lower Bound constitued with a first term based the expectation of the data log likelihood
        under the variational distribution with MC samples and an analytical second term based on the KL divergence (eager execution)
        """
        X, Y = data
        num_data = X.shape[0]
        L = tf.reduce_sum(self.E_log_p_Y(X, Y)) #### Computation of the expectation term
        KL = tf.reduce_sum([layer.KL() for layer in self.layers]) ### Computation of the KL term
        scale = tf.cast(num_data, gpflow.default_float())
        scale /= tf.cast(tf.shape(X)[0], gpflow.default_float())  # minibatch size ( in case of minibatch size = N the scale comes back to 1)
        return L * scale - KL 

    @tf.function
    def ELBO_closure(self,data):
        """
        The Evidence Lower Bound Wrapped within a tensorflow function to create a tensorflow graph to accelerate the computation.
        """
        def closure():
                return self.ELBO(data)
        return closure()



    def predict_y(self, Xnew, num_samples):
        """
        Compute the mean and variance obtained at the last layer of the DGP for inputs X taking into account the likelihood

        :param X: The inputs to propagate
        :param full_cov: If True, draw correlated samples over the inputs. Computes the Cholesky over the
            dense covariance matrix of size [num_data, num_data].
            If False, draw samples that are uncorrelated over the inputs.
        :param S: the number of samples to propagate
        """
        Fmean, Fvar = self.predict_f(Xnew, full_cov=False, S=num_samples)
        return self.likelihood.predict_mean_and_var(Fmean, Fvar)

    def predict_density(self, Xnew, Ynew, num_samples):
        Fmean, Fvar = self.predict_f(Xnew, full_cov=False, S=num_samples)
        l = self.likelihood.predict_density(Fmean, Fvar, Ynew)
        log_num_samples = tf.log(tf.cast(num_samples, gpflow.default_float()))
        return tf.reduce_logsumexp(l - log_num_samples, axis=0)

    def optimize_adam(self,data,iterations = 5000,lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, messages=100):
        """
        An Adam optimization method for the training of DGPs

        :param data: The data (X,Y)
        :param iterations: the number of iterations
        :param lr: the learning rate for the adam optimizer. Defaults to 0.01
        :param beta_1: The exponential decay rate for the 1st moment estimates. Defaults to 0.9
        :param beta_2: The exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
        :param  epsilon: 	A small constant for numerical stability. Defaults to 1e-8
        :param messages: int to print the evaluation of the ELBO. Default to 100 iterations
        """
        X_train, Y_train = data
        optimizer = tf.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2,epsilon=epsilon) ### Define the Adam optimizer
        ### Optimization loop 
        for step in range(iterations): 
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.trainable_variables)
                objective = -self.ELBO_closure((X_train, Y_train)) ### loss function
                gradients = tape.gradient(objective, self.trainable_variables) ### computation of the gradients
            optimizer.apply_gradients(zip(gradients, self.trainable_variables)) ### update the the trainable variables
            if step%messages==0:
                print(f"ELBO: {-objective.numpy()}")
    def optimize_nat_adam(self,data,iterations1=100, iterations2=5000,lr_adam =0.01,lr_gamma=0.01,beta_1=0.9, beta_2=0.999, epsilon=1e-07, ng_all = True, messages= 100):
        """
        A training approach for DGP taking into account the differential geometry of the parameter space using natural gradient
        The training approach is divided into two parts. 
            -in the first part only the kernel parameters and the induced inputs are optimized using an Adam optimizer
            while the variational parameters are fixed.
            - in the second part a loop procedure is used where iteratively an Adam step in the Euclidean space and a natural 
            gradient step in the variational parameter space are performed.

        :param data: the data (X,Y)
        :param iterations1: the number of iterations for part 1.
        :param iterations1: the number of iterations for part 2.
        :param lr_adam: the learning rate for the adam optimizer. Defaults to 0.01
        :param lr_gamma: the learning rate for the natural gradient. Defaults to 0.01
        :param beta_1: Adam parameter, the exponential decay rate for the 1st moment estimates. Defaults to 0.9
        :param beta_2: Adam parameter, the exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
        :param  epsilon: Adam parameter, a small constant for numerical stability. Defaults to 1e-8
        :param  ng_all: Boolean parameter. True natural gradient  for the variational parameters at each layer. False natural
        gradient only at the last layer.
        :param messages: int to print the evaluation of the ELBO. Default to 100 iterations.
        The key reference is:
        
        @article{hebbal2020bayesian,
        title={Bayesian optimization using deep Gaussian processes with applications to aerospace system design},
        author={Hebbal, Ali and Brevault, Lo{\"\i}c and Balesdent, Mathieu and Talbi, El-Ghazali and Melab, Nouredine},
        journal={Optimization and Engineering},
        pages={1--41},
        year={2020},
        publisher={Springer}
        }
        """
        X_train, Y_train = data
        optimizer_adam = tf.optimizers.Adam(learning_rate=lr_adam, beta_1=beta_1, beta_2=beta_2,epsilon=epsilon) ### Define the Adam optimizer
        optimizer_nat = NaturalGradient(gamma=lr_gamma) ### Define the Natural gradient optimizer
        ##### Determination of the variables to be updated by the Natural gradient
        if ng_all: ### All the variational parameters are updated using the Natural gradient
            for layer in self.layers:
                set_trainable(layer.q_mu,False)
                set_trainable(layer.q_sqrt,False)
            variational_params= [[layer.q_mu, layer.q_sqrt ] for layer in self.layers]
        else: ### Only the last layer variational parameters are updated using the Natural gradient
            set_trainable(self.layers[-1].q_mu,False)
            set_trainable(self.layers[-1].q_sqrt,False)
            variational_params= [(self.layers[-1].q_mu, self.layers[-1].q_sqrt)]   
            
        ### Part 1 of the optimization procedure using only the Adam optimizer            
        for step in range(iterations1):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.trainable_variables)
                objective = -self.ELBO_closure(data)
                gradients = tape.gradient(objective, self.trainable_variables)
            optimizer_adam.apply_gradients(zip(gradients, self.trainable_variables))
            if step%messages==0:
                print(f"ELBO: {-objective.numpy()}")
        ### Part 2 of the optimization procedure using also the Natural gradient
        def objective_nat():
            return -self.ELBO_closure((X_train, Y_train))
        for step in range(iterations2):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.trainable_variables)
                objective = -self.ELBO_closure(data)
                gradients = tape.gradient(objective, self.trainable_variables)
            optimizer_adam.apply_gradients(zip(gradients, self.trainable_variables))
            optimizer_nat.minimize(objective_nat, var_list=variational_params)
            if step%messages==0:
                print(f"ELBO: {-objective.numpy()}")
class DGP(DGP_Base):
    """
    
    This is the Doubly-Stochastic Deep GP, with linear/identity mean functions at each layer.

    :param X: input observations
    :param Y: observed values
    :param Z: induces inputs
    :param kernel: list of GPflow kernels at each layer
    :param  num_units: list of number of units at each layer 
    :param likelihood: a GPflow likelihood
    :param mean_function: the final layer GPflow mean function
    
    The key reference is

    ::
      @inproceedings{salimbeni2017doubly,
        title={Doubly Stochastic Variational Inference for Deep Gaussian Processes},
        author={Salimbeni, Hugh and Deisenroth, Marc},
        booktitle={NIPS},
        year={2017}
      }

    """
    def __init__(self, X, Y, Z, kernels,num_units, likelihood,
                 num_outputs=None,
                 mean_function=Zero(),  # the final layer mean function,
                 white=False, **kwargs):
        layers = init_layers_linear(X, Y, Z, kernels,num_units,
                                    num_outputs=num_outputs,
                                    mean_function=mean_function,
                                    white=white)
        DGP_Base.__init__(self,likelihood, layers, **kwargs)
        self.data = ((X,Y))
    def optimize_adam(self,iterations = 5000,lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, messages=100):
        """
        An Adam optimization method for the training of DGPs

        :param iterations: the number of iterations
        :param lr: the learning rate for the adam optimizer. Defaults to 0.01
        :param beta_1: The exponential decay rate for the 1st moment estimates. Defaults to 0.9
        :param beta_2: The exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
        :param  epsilon: 	A small constant for numerical stability. Defaults to 1e-8
        :param messages: int to print the evaluation of the ELBO. Default to 100 iterations
        """
        X_train, Y_train = self.data
        optimizer = tf.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2,epsilon=epsilon) ### Define the Adam optimizer
        for layer in self.layers[:-1]:
            layer.q_sqrt.assign(layer.q_sqrt * 1e-3) ### Initialization of variational variance to low values for optimization stability
        ### Optimization loop 
        for step in range(iterations):
            with tf.GradientTape(watch_accessed_variables=True) as tape:
                tape.watch(self.trainable_variables)
                objective = -self.ELBO_closure((X_train, Y_train)) ### loss function
                gradients = tape.gradient(objective, self.trainable_variables) ### computation of the gradients
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))  ### update the the trainable variables
            if step%messages==0:
                print(f"ELBO: {-objective.numpy()}")

    def optimize_nat_adam(self,iterations1=100, iterations2=5000,lr_adam =0.01,lr_gamma=0.01,beta_1=0.9, beta_2=0.999, epsilon=1e-07, ng_all = True, messages= 100):
        """
        A training approach for DGP taking into account the differential geometry of the parameter space using natural gradient
        The training approach is divided into two parts. 
            -in the first part only the kernel parameters and the induced inputs are optimized using an Adam optimizer
            while the variational parameters are fixed.
            - in the second part a loop procedure is used where iteratively an Adam step in the Euclidean space and a natural 
            gradient step in the variational parameter space are performed.

        :param iterations 1: the number of iterations for part 1.
        :param iterations 2: the number of iterations for part 2.
        :param lr_adam: the learning rate for the adam optimizer. Defaults to 0.01
        :param lr_gamma: the learning rate for the natural gradient. Defaults to 0.01
        :param beta_1: Adam parameter, the exponential decay rate for the 1st moment estimates. Defaults to 0.9
        :param beta_2: Adam parameter, the exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
        :param  epsilon: Adam parameter, a small constant for numerical stability. Defaults to 1e-8
        :param  ng_all: Boolean parameter. True natural gradient  for the variational parameters at each layer. False natural
        gradient only at the last layer.
        :param messages: int to print the evaluation of the ELBO. Default to 100 iterations.
        The key reference is:
        
        @article{hebbal2020bayesian,
        title={Bayesian optimization using deep Gaussian processes with applications to aerospace system design},
        author={Hebbal, Ali and Brevault, Lo{\"\i}c and Balesdent, Mathieu and Talbi, El-Ghazali and Melab, Nouredine},
        journal={Optimization and Engineering},
        pages={1--41},
        year={2020},
        publisher={Springer}
        }
        """
        X_train, Y_train = self.data
        optimizer_adam = tf.optimizers.Adam(learning_rate=lr_adam, beta_1=beta_1, beta_2=beta_2,epsilon=epsilon) ### Define the Adam optimizer
        optimizer_nat = NaturalGradient(gamma=lr_gamma) ### Define the Natural gradient optimizer
        ##### Determination of the variables to be updated by the Natural gradient
        if ng_all:  ### All the variational parameters are updated using the Natural gradient
            for layer in self.layers:
                set_trainable(layer.q_mu,False)
                set_trainable(layer.q_sqrt,False)
            variational_params= [[layer.q_mu, layer.q_sqrt ] for layer in self.layers]
        else: ### Only the last layer variational parameters are updated using the Natural gradient
            set_trainable(self.layers[-1].q_mu,False)
            set_trainable(self.layers[-1].q_sqrt,False)
            variational_params= [(self.layers[-1].q_mu, self.layers[-1].q_sqrt)]    
        for layer in self.layers[:-1]:
            layer.q_sqrt.assign(layer.q_sqrt * 1e-3) ### Initialization of variational variance to low values for optimization stability
        ### Part 1 of the optimization procedure using only the Adam optimizer           
        for step in range(iterations1):
            with tf.GradientTape(watch_accessed_variables=True) as tape:
                tape.watch(self.trainable_variables)
                objective = -self.ELBO_closure((X_train, Y_train))
                gradients = tape.gradient(objective, self.trainable_variables)
            optimizer_adam.apply_gradients(zip(gradients, self.trainable_variables))
            if step%messages==0:
                print(f"ELBO: {-objective.numpy()}")
        ### Part 2 of the optimization procedure using also the Natural gradient
        def objective_nat():
            return -self.ELBO_closure((X_train, Y_train))
        for step in range(iterations2):
            with tf.GradientTape(watch_accessed_variables=True) as tape:
                tape.watch(self.trainable_variables)
                objective = -self.ELBO_closure((X_train, Y_train))
                gradients = tape.gradient(objective, self.trainable_variables)
            optimizer_adam.apply_gradients(zip(gradients, self.trainable_variables))
            optimizer_nat.minimize(objective_nat, var_list=variational_params)
            if step%messages==0:
                print(f"ELBO: {-objective.numpy()}")
                

    def number_parameters(self,trainable=True):
        '''
        Computes the total number of parameters.
        if trainable is True only the trainable parameters are considered
        '''
        size_problem = 0
        if trainable:
            for param in self.trainable_parameters:
                size_problem = size_problem + len(np.ravel(param.numpy()))
        else:
            for param in self.parameters:
                size_problem = size_problem + len(np.ravel(param.numpy()))
        return size_problem
    
    def predict(self, Xnew, num_samples):
        y_m,y_v = self.predict_y(Xnew, num_samples=num_samples)
        y_mean_prediction = np.mean(y_m, axis=0)
        y_var_prediction = np.mean(y_v + y_m**2, 0) - y_mean_prediction**2
        return y_mean_prediction, y_var_prediction