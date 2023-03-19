# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:04:15 2018

@author: ahebbal
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 16:45:58 2018

@author: ahebbal
"""
import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
import math
import time
import pyDOE
import scipy
from ..Infill_criteria import *
import gpflow
from gpflow.base import Module, Parameter
from ..models.dgp import DGP


def normalize(*args):
    normalized_data=[]
    if len (args)==1:
        return (args[0]-args[0].mean(axis=0))/args[0].std(axis=0)
    for arg in args:
        normalized_data.append((arg-arg.mean(axis=0))/arg.std(axis=0))
    return normalized_data

def normalize_X(X):
    return (X-X.mean(axis=0))/X.std(axis=0), (0-X.mean(axis=0))/X.std(axis=0), (1-X.mean(axis=0))/X.std(axis=0)

def normalize_C(X):
    return (X-X.mean(axis=0))/X.std(axis=0), (0-X.mean(axis=0))/X.std(axis=0)

def denormalize(Xstar_N,X):
        return X.std(axis=0)*Xstar_N+X.mean(axis=0)  

def denormalize_var(Xstar_N,X):
        return X.std(axis=0)**2*Xstar_N

def DoE(problem,DoE_size):
    X = pyDOE.lhs(problem.dim,DoE_size)
    if problem.constraint:
        Y,C = problem.fun(X)
        return X,Y,C
    else:
        Y = problem.fun(X)[0]
        return X,Y

class SO_BO(Module):
    def __init__(self,problem=None,X=None,Y=None,C=None,DoE_size = None, model_Y_dic=None,model_C_dic=None,normalize_input=True):
        """
        SO_BO is a Module class for single objective Unconstrained/constrained Bayesian Optimization using Gaussian processes and deep Gaussian 
        processes. 
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
        """
        super(SO_BO, self).__init__()

        ### Chek the arguments given
        if problem is None:
            raise Exception("You have to specify a problem to optimize")
        self.problem=problem
        if model_Y_dic is None or isinstance(model_Y_dic,dict)==False:
            raise Exception("You have to specify a dictionary for the architecture of the objective function model")
        if problem.constraint and model_C_dic is None:
            raise Exception("You have to specify a dictionary for the architecture of the constraint functions models")
        self.model_Y_dic = model_Y_dic
        self.model_C_dic = model_C_dic
        if DoE_size is None and X is None:
            raise Exception("You have to specify either a size to generate a DoE or specify a known DoE (X,Y)")
        
        # # # DoE generation
        if X is None:
            if problem.constraint:
                self.X,self.Y,self.C = DoE(problem,DoE_size)
            else:
                self.X,self.Y = DoE(problem,DoE_size)
        else:
            self.X=X.copy()
            self.Y=Y.copy()
            if problem.constraint:
                self.C=C.copy()
            else:
                self.C = None
        self.d=self.problem.dim
        self.n=self.X.shape[0]

        # # # Normalization of the data
        self.normalize_input = normalize_input
        if normalize_input:
            self.X_n,self.lw_n,self.up_n=normalize_X(self.X)
            self.Y_n = normalize(self.Y)
            if problem.constraint:
                self.C_n,self.feasible_0=normalize_C(self.C)

        # # # Creation of the models 
        if normalize_input:
            self.X_train = self.X_n
            self.Y_train = self.Y_n
            if problem.constraint:
                self.C_train = self.C_n
        else:
            self.X_train = self.X
            self.Y_train = self.Y
            if problem.constraint:
                self.C_train = self.C
        if isinstance(model_Y_dic,dict):
            self.model_Y = self.make_model(model_Y_dic,self.X_train,self.Y_train)
        ### In case of constraints
        if problem.constraint:
            self.model_C=[[]]*self.C.shape[1]
            if isinstance(model_C_dic,list)==False:
                self.model_C_dic = [model_C_dic]*self.C.shape[1]
            for i in range(self.C.shape[1]):
                if isinstance(self.model_C_dic[i],dict):
                    self.model_C[i] = self.make_model(self.model_C_dic[i],self.X_train,self.C_train[:,i].reshape(len(self.X_train),1))
                else:
                    raise Exception("Model_C[",i,"] has to be a dictionary")
            
        self.Xfeasible=[] ### Feasible input data
        self.Yfeasible=[] ### Feasible output data
        self.Ymin = [] ### List of minimum observed values through the BO procedure
        self.feasible() ### Determine the feasible data points
        self.added_points=[] ### Added points through the BO procedure
        self.IC = None ### Infill criteria class
        self.constrained_IC = None ### Constrained infill criteria class

    def feasible(self):
        '''
        Determine the feasible data points
        '''
        self.Xfeasible=[]
        self.Yfeasible=[]
        if self.C is not None:
            self.Cfeasible=[]
            for i in range(self.X.shape[0]):
                if self.C[i].max()<=0:
                    self.Yfeasible=np.append(self.Yfeasible,self.Y[i])
                    self.Xfeasible=np.append(self.Xfeasible,self.X[i])
                    self.Cfeasible=np.append(self.Cfeasible,self.C[i])
                if len(self.Yfeasible)==0: ### If there is no feasible data point 
                    self.Ymin=[np.max(self.Y)]
                else:
                    self.Ymin=[np.min(self.Yfeasible)]
        else:
            self.Xfeasible=self.X
            self.Yfeasible=self.Y
            self.Ymin=[np.min(self.Y)]

    def make_model(self,dic,X,Y):
        '''
        Creates a model given a dictionary and the input/output data
        '''
        try:
            num_layers = dic['num_layers']
        except ValueError:
            print("num_layers entry is not specified.")
        


        if num_layers==0:
            try:
                kern_name = dic['kernels']
            except ValueError:
                print("kernels entry is not specified")
            if kern_name == 'rbf':
                kernel = gpflow.kernels.SquaredExponential(lengthscales=[1.0]*X.shape[1],variance=1.0) 
            elif kern_name == 'matern32':
                kernel = gpflow.kernels.Matern32(lengthscales=[1.0]*X.shape[1],variance=1.0) 
            elif kern_name == 'matern52':
                kernel = gpflow.kernels.Matern52(lengthscales=[1.0]*X.shape[1],variance=1.0) 
            else:
                raise Exception("The kernel has to be a string or a list of strings: 'rbf', 'matern32', matern52'")
            model = gpflow.models.GPR((X, Y), kernel, noise_variance=1e-5)

        elif num_layers>0:
            try:
                num_samples = dic['num_samples']
            except ValueError:
                print("num_samples entry is not specified.")
            try:
                num_units = dic['num_units']
            except ValueError:
                print("num_units entry is not specified")
            if isinstance(num_units,list):
                if len(num_units) != num_layers:
                    raise Exception("The length of the list of units has to be equal to the number of layers")
            elif isinstance(num_units,int):
                num_units = [num_units]*num_layers
            else:
                raise Exception("num_units has to be an integer or a list of integer for each layer")


            try:
                kern_names = dic['kernels']
            except ValueError:
                print("kernels entry is not specified")
            if isinstance(kern_names,list):
                if len(kern_names) != num_layers+1:
                    raise Exception("The length of the list of kernels has to be equal to the number of layers")
            elif isinstance(kern_names,str):
                kern_names = [kern_names]*(num_layers+1)
            else:
                raise Exception("The kernel has to be a string or a list of strings: 'rbf', 'matern32', matern52'")
            
            kernels = []
            for l in range(num_layers+1):
                if l ==0: 
                    units = X.shape[1]
                else:
                    units = num_units[l-1]

                if kern_names[l] == 'rbf':
                    kernel = gpflow.kernels.SquaredExponential(lengthscales=[1.0]*units,variance=1.0) 
                elif kern_names[l] == 'matern32':
                    kernel = gpflow.kernels.Matern32(lengthscales=[1.0]*units,variance=1.0) 
                elif kern_names[l] == 'matern52':
                    kernel = gpflow.kernels.Matern52(lengthscales=[1.0]*units,variance=1.0) 
                else:
                    raise Exception("The kernel has to be a string or a list of strings: 'rbf', 'matern32', matern52'")
                kernels.append(kernel)
            model = DGP(X, Y, X, kernels, num_units, gpflow.likelihoods.Gaussian(), num_samples=num_samples)
        return model

    def train_model(self,model,iteration=3000):
        if model.name == 'gpr':
            training_loss = model.training_loss_closure(compile=True) 
            opt = tf.optimizers.Adam()
            for step in range(iteration):
                opt.minimize(training_loss, model.trainable_variables)
        if model.name == 'dgp':
            model.optimize_nat_adam(iterations1=500,iterations2=iteration,beta_1=0.8, beta_2=0.9,lr_gamma=0.01)

    def train_models(self,iteration_Y = 3000,iteration_C = 3000):
        print('Training of the objective function model')
        self.train_model(self.model_Y,iteration_Y)
        if self.problem.constraint:
            if isinstance(iteration_C,list)==False:
                iteration_C = [iteration_C]*self.C.shape[1]
            for i in range(self.C.shape[1]):
                print('Training of constraint model',i+1)
                self.train_model(self.model_C[i],iteration_C[i])

    def run(self,iterations,from_scratch=None,IC ='EI' ,constraint_handling='PoF',threshold=0.1,train_iterations=1000,popsize_DE=300\
        ,popstd_DE = 1.5,iterations_DE=400, init_adam=None,iterations_adam=1000,IC_method='DE+Adam',analytic=True):
        if self.model_Y_dic is None and from_scratch is not None:
            raise Exception('To create models from scratch you have to specify model_Y_dic and model_C_dic')
        for j in range(iterations):
            print ('adding the most promising data point in iteration',j)
            bounds = (self.lw_n,self.up_n)
            if from_scratch is None:
                from_scratch = iterations+1
            if j%from_scratch==0 and j!=0:
                self.make_model(self.model_Y_dic,self.X_train,self.Y_train)
                if self.problem.constraint:
                    self.model_C = [[]]*self.C.shape[1]
                    for i in range(self.C.shape[1]):
                        self.model_C[i] = self.make_model(self.model_C_dic[i],self.X_train,self.C_train[:,i].reshape(len(self.X_train),1))
            if j%from_scratch==0:
                self.train_models(iteration_Y=train_iterations,iteration_C=train_iterations)
            elif j!=0:
                self.model_Y.data = (self.X_train,self.Y_train)
                if self.problem.constraint:
                    for i in range(self.C.shape[1]):
                            self.model_C[i].data = (self.X_train,self.C_train.reshape(len(self.C),1))
                self.train_models(iteration_Y=int(train_iterations/2),iteration_C=int(train_iterations/2))
            if IC == 'EI':  
                self.IC = EI((self.Ymin[-1]-self.Y.mean(axis=0))/self.Y.std(axis=0),self.d) 
            elif IC == 'WB2':  
                self.IC = WB2((self.Ymin[-1]-self.Y.mean(axis=0))/self.Y.std(axis=0),self.d)
            elif IC == 'WB2S':   
                self.IC = WB2S((self.Ymin[-1]-self.Y.mean(axis=0))/self.Y.std(axis=0),self.d)

            if self.problem.constraint:
                if constraint_handling =='PoF':
                    self.constrained_IC = PoF(self.feasible_0,self.d) 
                    self.added_points  = self.constrained_IC.optimize_with_IC(self.IC,self.model_Y,self.model_C,bounds)
                elif constraint_handling =='EV':
                    self.constrained_IC = EV(self.feasible_0,self.d) 
                    self.added_points = self.constrained_IC.optimize_with_IC(self.IC,self.model_Y,self.model_C,bounds,threshold=threshold, popsize_DE=popsize_DE\
        ,popstd_DE = popstd_DE,iterations_DE=iterations_DE,method=IC_method,analytic=analytic)
            else:         
                self.added_points= self.IC.optimize(self.model_Y,bounds,popsize_DE=popsize_DE,popstd_DE = popstd_DE,\
                    iterations_DE=iterations_DE, init_adam=init_adam,iterations_adam=iterations_adam,method=IC_method,\
                        analytic=analytic)
            self.add_point()
            print('Actual Y min:', self.Ymin[-1])

    def add_point(self):
        if self.normalize_input==True:       
            temp=self.problem.fun(denormalize(self.added_points,self.X))
            self.X=  np.append(self.X, denormalize(self.added_points,self.X) ,axis=0)
        else:
            temp=self.problem.fun(self.added_points)
            self.X=  np.append(self.X, self.added_points,axis=0)
        self.Y = np.append(self.Y, temp[0] ,axis=0)
        self.X_n,self.lw_n,self.up_n=normalize_X(self.X)
        self.Y_n = normalize(self.Y)
        if self.problem.constraint:
            if self.C.shape[1]==1:
                self.C = np.append(self.C, temp[1],axis=0)
            else:
                self.C = np.append(self.C, temp[1],axis=0)
            if self.C[-1].max()<=0:
                    self.Yfeasible=np.append(self.Yfeasible,self.Y[-1])
                    self.Xfeasible=np.append(self.Xfeasible,self.X[-1])
                    self.Ymin=np.append(self.Ymin,np.min(self.Yfeasible))
            else:
                self.Ymin=np.append(self.Ymin,self.Ymin[-1])
            self.C_n,self.feasible_0=normalize_C(self.C)
        else:
            self.Yfeasible=self.Y
            self.Xfeasible=self.X
            self.Ymin=np.append(self.Ymin,np.min(self.Y))
        if self.normalize_input:
            self.X_train = self.X_n
            self.Y_train = self.Y_n
            if self.problem.constraint:
                self.C_train = self.C_n
        else:
            self.X_train = self.X
            self.Y_train = self.Y
            if self.problem.constraint:
                self.C_train = self.C