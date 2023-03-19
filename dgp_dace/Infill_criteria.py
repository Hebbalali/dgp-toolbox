# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 17:08:49 2018

@author: ahebbal
"""
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp


class Infill_criteria(object):
    def __init__(self):
        self.name= 'Infill criteria'
    def run(self,x):
        raise NotImplementedError("method not implemented")
    def optimize(self):
        raise NotImplementedError("method not implemented")

class EI(Infill_criteria):
    def __init__(self, y_min,d):
        super(Infill_criteria, self).__init__()
        self.name = 'Expected Improvement'
        self.y_min = y_min
        self.d = d
        self.IC_optimized = None
        self.x_opt = None
    def run(self,model,x,analytic=True,num_samples=1000):
        if model.name=='gpr':
            candidate_mean, candidate_var = model.predict_y(x)
            normal = tfp.distributions.Normal(candidate_mean, tf.sqrt(candidate_var))
            t1 = (self.y_min - candidate_mean) * normal.cdf(self.y_min)
            # t2 = tf.sqrt(candidate_var) * normal.prob(fmin)
            t2 = candidate_var * normal.prob(self.y_min)
            EI = tf.add(t1, t2)
        if model.name=='dgp':
            # Obtain predictive distributions for candidates
            if analytic==True:
                candidate_mean_, candidate_var_ = model.predict_f(x,S=num_samples)
                candidate_mean = tf.math.reduce_mean(candidate_mean_,axis=0)
                candidate_var = tf.math.reduce_mean(candidate_var_ + candidate_mean_**2,axis=0) - candidate_mean**2
                # Compute EI
                normal = tfp.distributions.Normal(candidate_mean, tf.sqrt(candidate_var))
                t1 = (self.y_min - candidate_mean) * normal.cdf(self.y_min)
                #t2 = tf.sqrt(candidate_var) * normal.prob(fmin)
                t2 = candidate_var * normal.prob(self.y_min)
                EI = tf.add(t1, t2)
            else:
                F,_,_= model.propagate(x,S=num_samples)
                EI=tf.where((F[-1]-self.y_min)<0, self.y_min-F[-1],0)
                EI = tf.math.reduce_mean(EI,axis=0)
        return -1*(EI)
    @tf.function
    def loss(self,model,x,analytic):
        """
        The Evidence Lower Bound Wrapped within a tensorflow function to create a tensorflow graph to accelerate the computation.
        """
        def closure():
                return self.run(model,x,analytic)
        return closure()
    def optimize(self,model,bounds,popsize_DE=300,popstd_DE = 1.5,iterations_DE=400, init_adam=None,iterations_adam=1000,method='DE',analytic=True):
        lw, up = bounds
        fct_optim=lambda x:self.loss(model,lw+(up-lw)*(1/(1+tf.exp(x))),analytic=analytic)
        if method == 'DE' or method == 'DE+Adam' :
            optim_results=tfp.optimizer.differential_evolution_minimize( fct_optim,initial_position=\
                                                                        tf.constant([0.]*self.d,tf.float64),\
                                                                            population_stddev=popstd_DE,population_size=popsize_DE,max_iterations=iterations_DE)
            self.x_opt=lw+(up-lw)*1/(1+np.exp((optim_results.position))).reshape(self.d,1)
            self.IC_optimized = self.run(model,self.x_opt)
        if method == 'Adam' or method == 'DE+Adam' :
            if init_adam ==None:
                if self.x_opt == None:
                    init_adam = [0.]*self.d
                else:
                    init_adam = self.x_opt
            init_ = np.log((up-init_adam+1e-3)/(init_adam-lw+1e-3))
            init_tf = tf.Variable(init_)
            optimizer = tf.optimizers.Adam(learning_rate=0.01)
            for step in range(iterations_adam):
                with tf.GradientTape(watch_accessed_variables=True) as tape:
                    tape.watch(init_tf)
                    objective = fct_optim(init_tf)
                    gradients = tape.gradient(objective, [init_tf])
                optimizer.apply_gradients(zip(gradients, [init_tf]))
            self.x_opt = lw+(up-lw)*1/(1+np.exp((init_tf.numpy()))).reshape(self.d,1)
            self.IC_optimized = objective.numpy()
        return self.x_opt
    
        # if self.method == 'Adam':
        #     init_ = np.log((up-self.x_opt+1e-3)/(self.x_opt-lw+1e-3))
        #     init_tf = tf.Variable(init_)
        #     loss = self.run(model,lw+(up-lw)*(1/(1+tf.exp(init_tf))))
        #     train_op = tf.train.AdamOptimizer(1e-3).minimize(loss,var_list=[init_tf])
            
    # @tf.function
    # def ELBO_closure(self,data):
    #     """
    #     The Evidence Lower Bound Wrapped within a tensorflow function to create a tensorflow graph to accelerate the computation.
    #     """
    #     def closure():
    #             return self.ELBO(data)
    #     return closure()
    #     return self.x_opt
    

class WB2(Infill_criteria):
    def __init__(self, y_min,d):
        super(Infill_criteria, self).__init__()
        self.name = 'WB2 criterion'
        self.y_min = y_min
        self.d = d
        self.IC_optimized = None
    def run(self,model,x):
        if model.name=='gpr':
            candidate_mean, candidate_var = model.predict_y(x)
            normal = tfp.distributions.Normal(candidate_mean, tf.sqrt(candidate_var))
            t1 = (self.y_min - candidate_mean) * normal.cdf(self.y_min)
            # t2 = tf.sqrt(candidate_var) * normal.prob(fmin)
            t2 = candidate_var * normal.prob(self.y_min)
            EI = tf.add(t1, t2)
        if model.name=='dgp':
            S = 1 / (1 + 1/tf.math.exp(x))
            # Obtain predictive distributions for candidates
            candidate_mean_, candidate_var_ = model.predict_y(x,num_samples=500)
            candidate_mean = tf.math.reduce_mean(candidate_mean_,axis=0)
            candidate_var = tf.math.reduce_mean(candidate_var_ + candidate_mean_**2,axis=0) - candidate_mean**2
            # Compute EI
            normal = tfp.distributions.Normal(candidate_mean, tf.sqrt(candidate_var))
            t1 = (self.y_min - candidate_mean) * normal.cdf(self.y_min)
            #t2 = tf.sqrt(candidate_var) * normal.prob(fmin)
            t2 = candidate_var * normal.prob(self.y_min)
            EI = tf.add(t1, t2)
        return -1*(EI-candidate_mean)
    @tf.function
    def loss(self,model,x):
        """
        The Evidence Lower Bound Wrapped within a tensorflow function to create a tensorflow graph to accelerate the computation.
        """
        def closure():
                return self.run(model,x)
        return closure()
    def optimize(self,model,bounds,popsize_DE=300,popstd_DE = 1.5,iterations_DE=400, init_adam=None,iterations_adam=1000,method='DE'):
        lw, up = bounds
        fct_optim=lambda x:self.loss(model,lw+(up-lw)*(1/(1+tf.exp(x))))
        if method == 'DE' or method == 'DE+Adam' :
            optim_results=tfp.optimizer.differential_evolution_minimize( fct_optim,initial_position=\
                                                                        tf.constant([0.]*self.d,tf.float64),\
                                                                            population_stddev=popstd_DE,population_size=popsize_DE,max_iterations=iterations_DE)
            self.x_opt=lw+(up-lw)*1/(1+np.exp((optim_results.position))).reshape(self.d,1)
            self.IC_optimized = self.run(model,self.x_opt)
        if method == 'Adam' or method == 'DE+Adam' :
            if init_adam ==None:
                if self.x_opt == None:
                    init_adam = [0.]*self.d
                else:
                    init_adam = self.x_opt
            init_ = np.log((up-init_adam+1e-3)/(init_adam-lw+1e-3))
            init_tf = tf.Variable(init_)
            optimizer = tf.optimizers.Adam(learning_rate=0.01)
            for step in range(iterations_adam):
                with tf.GradientTape(watch_accessed_variables=True) as tape:
                    tape.watch(init_tf)
                    objective = fct_optim(init_tf)
                    gradients = tape.gradient(objective, [init_tf])
                optimizer.apply_gradients(zip(gradients, [init_tf]))
            self.x_opt = lw+(up-lw)*1/(1+np.exp((init_tf.numpy()))).reshape(self.d,1)
            self.IC_optimized = objective.numpy()
        return self.x_opt
    

class WB2S(Infill_criteria):
    def __init__(self, y_min,d):
        super(Infill_criteria, self).__init__()
        self.name = 'WB2S criterion'
        self.y_min = y_min
        self.d = d
        self.IC_optimized = None
    def run(self,model,x):
        if model.name=='gpr':
            candidate_mean, candidate_var = model.predict_y(x)
            normal = tfp.distributions.Normal(candidate_mean, tf.sqrt(candidate_var))
            t1 = (self.y_min - candidate_mean) * normal.cdf(self.y_min)
            # t2 = tf.sqrt(candidate_var) * normal.prob(fmin)
            t2 = candidate_var * normal.prob(self.y_min)
            EI = tf.add(t1, t2)
        if model.name=='dgp':
            S = 1 / (1 + 1/tf.math.exp(x))
            # Obtain predictive distributions for candidates
            candidate_mean_, candidate_var_ = model.predict_y(x,num_samples=500)
            candidate_mean = tf.math.reduce_mean(candidate_mean_,axis=0)
            candidate_var = tf.math.reduce_mean(candidate_var_ + candidate_mean_**2,axis=0) - candidate_mean**2
            # Compute EI
            normal = tfp.distributions.Normal(candidate_mean, tf.sqrt(candidate_var))
            t1 = (self.y_min - candidate_mean) * normal.cdf(self.y_min)
            #t2 = tf.sqrt(candidate_var) * normal.prob(fmin)
            t2 = candidate_var * normal.prob(self.y_min)
            EI = tf.add(t1, t2)
        return -1*(S*EI-candidate_mean)
    @tf.function
    def loss(self,model,x):
        """
        The Evidence Lower Bound Wrapped within a tensorflow function to create a tensorflow graph to accelerate the computation.
        """
        def closure():
                return self.run(model,x)
        return closure()
    def optimize(self,model,bounds,popsize_DE=300,popstd_DE = 1.5,iterations_DE=400, init_adam=None,iterations_adam=1000,method='DE'):
        lw, up = bounds
        fct_optim=lambda x:self.loss(model,lw+(up-lw)*(1/(1+tf.exp(x))))
        if method == 'DE' or method == 'DE+Adam' :
            optim_results=tfp.optimizer.differential_evolution_minimize( fct_optim,initial_position=\
                                                                        tf.constant([0.]*self.d,tf.float64),\
                                                                            population_stddev=popstd_DE,population_size=popsize_DE,max_iterations=iterations_DE)
            self.x_opt=lw+(up-lw)*1/(1+np.exp((optim_results.position))).reshape(self.d,1)
            self.IC_optimized = self.run(model,self.x_opt)
        if method == 'Adam' or method == 'DE+Adam' :
            if init_adam ==None:
                if self.x_opt == None:
                    init_adam = [0.]*self.d
                else:
                    init_adam = self.x_opt
            init_ = np.log((up-init_adam+1e-3)/(init_adam-lw+1e-3))
            init_tf = tf.Variable(init_)
            optimizer = tf.optimizers.Adam(learning_rate=0.01)
            for step in range(iterations_adam):
                with tf.GradientTape(watch_accessed_variables=True) as tape:
                    tape.watch(init_tf)
                    objective = fct_optim(init_tf)
                    gradients = tape.gradient(objective, [init_tf])
                optimizer.apply_gradients(zip(gradients, [init_tf]))
            self.x_opt = lw+(up-lw)*1/(1+np.exp((init_tf.numpy()))).reshape(self.d,1)
            self.IC_optimized = objective.numpy()
        return self.x_opt

class EV_one_constraint(Infill_criteria):
    def __init__(self, zero_c,d):
        super(Infill_criteria, self).__init__()
        self.name = 'Expected Violation'
        self.zero_c = zero_c
        self.d = d
        self.IC_optimized = None
    def run(self,model,x,analytic=True,num_samples=100):
        if analytic:
            if model.name=='gpr':
                candidate_mean, candidate_var = model.predict_y(x)
                normal =  tfp.distributions.Normal(-candidate_mean, tf.sqrt(candidate_var))
                t1 = (-self.zero_c + candidate_mean) * normal.cdf(-self.zero_c)
                t2 = candidate_var * normal.prob(-self.zero_c)
            if model.name=='dgp':
                # Obtain predictive distributions for candidates
                candidate_mean_, candidate_var_ = model.predict_y(x,num_samples=500)
                candidate_mean = tf.math.reduce_mean(candidate_mean_,axis=0)
                candidate_var = tf.math.reduce_mean(candidate_var_ + candidate_mean_**2,axis=0) - candidate_mean**2
                # Compute EV
                normal =  tfp.distributions.Normal(-candidate_mean, tf.sqrt(candidate_var))
                t1 = (-self.zero_c + candidate_mean) * normal.cdf(-self.zero_c)
                t2 = candidate_var * normal.prob(-self.zero_c)
            return tf.add(t1, t2)
        else:
            F,_,_= model.propagate(x,S=num_samples)
            EV=tf.where((F[-1]-self.zero_c)<0, 0,F[-1]-self.zero_c)
            EV = tf.math.reduce_mean(EV,axis=0)
            return EV

class EV(Infill_criteria):
    def __init__(self, zero_c,d):
        super(Infill_criteria, self).__init__()
        self.name = 'Expected Violation'
        self.zero_c = zero_c
        self.d = d
        self.IC_optimized = None
    def run(self,model_C,x,analytic=True,num_samples=100):
        for i in range(len(model_C)):
            EVi = EV_one_constraint(self.zero_c[i],self.d).run(model_C[i],x,analytic=analytic,num_samples=num_samples)
            if i==0:
                EV = EVi
            else:
                EV = tf.concat([EV,EVi],1)
        return EV
    def run_with_IC(self,IC,model_Y,model_C,x,threshold=0.1,analytic=True,num_samples=100):
        EV = self.run(model_C,x,analytic=analytic,num_samples=num_samples)
        EV_max=tf.reduce_max(EV,axis=1)
        EI = IC.run(model_Y,x)
        for j in range((x.shape[0])):
            if j == 0:
                EI_EV = tf.reshape(tf.cond(EV_max[j]>threshold, lambda: tf.reduce_sum(EV[j])+10000  , lambda: EI[j] ),[1,1])
            else:
                EI_EV = tf.concat([EI_EV,tf.reshape(tf.cond(EV_max[j]>threshold, lambda: tf.reduce_sum(EV[j])+10000  , lambda: EI[j] ),[1,1])],0)
        return tf.squeeze(EI_EV[:,None],1)
    def optimize_with_IC(self,IC,model_Y,model_C,bounds,threshold=0.1,analytic=True,num_samples=100,popsize_DE=300,popstd_DE = 1.5,iterations_DE=400, init_adam=None,iterations_adam=1000,method='DE'):
        lw, up = bounds
        fct_optim=lambda x:self.run_with_IC(IC,model_Y,model_C,lw+(up-lw)*(1/(1+tf.exp(x))),threshold=0.1,analytic=analytic,num_samples=num_samples)
        if method == 'DE' or method == 'DE+Adam' :
            optim_results=tfp.optimizer.differential_evolution_minimize( fct_optim,initial_position=\
                                                                        tf.constant([0.]*self.d,tf.float64),\
                                                                            population_stddev=popstd_DE,population_size=popsize_DE,max_iterations=iterations_DE)
            self.x_opt=lw+(up-lw)*1/(1+np.exp((optim_results.position.numpy()))).reshape(self.d,1)
            self.IC_optimized = self.run_with_IC(IC,model_Y,model_C,self.x_opt,threshold)
        if method == 'Adam' or method == 'DE+Adam' :
            if init_adam ==None:
                if self.x_opt == None:
                    init_adam = [0.]*self.d
                else:
                    init_adam = self.x_opt
            init_ = np.log((up-init_adam+1e-3)/(init_adam-lw+1e-3))
            init_tf = tf.Variable(init_)
            optimizer = tf.optimizers.Adam(learning_rate=0.01)
            for step in range(iterations_adam):
                with tf.GradientTape(watch_accessed_variables=True) as tape:
                    tape.watch(init_tf)
                    objective = fct_optim(init_tf)
                    gradients = tape.gradient(objective, [init_tf])
                optimizer.apply_gradients(zip(gradients, [init_tf]))
            self.x_opt = lw+(up-lw)*1/(1+np.exp((init_tf.numpy()))).reshape(self.d,1)
            self.IC_optimized = objective.numpy()
        return self.x_opt        
        
class PoF(Infill_criteria):
    def __init__(self, zero_c,d):
        super(Infill_criteria, self).__init__()
        self.name = 'Probability of feasability'
        self.zero_c = zero_c
        self.d = d
        self.IC_optimized = None
    def run(self,model_C,x):
        if model_C.name=='gpr':
            candidate_mean, candidate_var = model_C.predict_y(x)
            normal =  tfp.distributions.Normal(candidate_mean, tf.sqrt(candidate_var))
            t1 = (self.zero_c - candidate_mean) * normal.cdf(self.zero_c)
            t2 = candidate_var * normal.prob(self.zero_c)
            # t2 = tf.sqrt(candidate_var) * normal.prob(fmin)
        if model_C.name=='dgp':
            # Obtain predictive distributions for candidates
            candidate_mean_, candidate_var_ = model_C.predict_y(x,num_samples=500)
            candidate_mean = tf.math.reduce_mean(candidate_mean_,axis=0)
            candidate_var = tf.math.reduce_mean(candidate_var_ + candidate_mean_**2,axis=0) - candidate_mean**2
            # Compute EI
            normal =  tfp.distributions.Normal(candidate_mean, tf.sqrt(candidate_var))
            t1 = (self.zero_c - candidate_mean) * normal.cdf(self.zero_c)
            # t2 = tf.sqrt(candidate_var) * normal.prob(fmin)
            t2 = candidate_var * normal.prob(self.zero_c)
    def run_with_IC(self,IC,model_Y,model_C,x):
        Pof = self.run(model_C,x)
        EI = IC.run(model_Y,x)
        return -1*EI*PoF
    def optimize_with_IC(self,IC,model_Y,model_C,bounds):
        lw, up = bounds
        fct_optim=lambda x:self.run_with_IC(IC,model_Y,model_C,lw+(up-lw)*(1/(1+tf.exp(x))))
        optim_results=tfp.optimizer.differential_evolution_minimize( fct_optim,initial_position=\
                                                                    tf.constant([0.]*self.d,tf.float64),\
                                                                        population_stddev=1.5,population_size=300,max_iterations=400)
        self.x_opt=1/(1+np.exp((optim_results.position))).reshape(self.d,1)
        self.IC_optimized = self.run_with_IC(IC,model_Y,model_C,self.x_opt)
        return self.x_opt        
       