
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp



def HV_calcul(ND,Y,bounds):
        L1,L2,U1,U2 = bounds
        a=0
        y1=Y[0]
        y2=Y[1]
        if len(ND)==0:
            return 0
        for i in range(len(ND)):
            if (y1[ND[i]]>U1) and (y2[ND[i]]>U2):
                a=1
                break
        if a==1:
            return 0
        else:
            hv=(U1-y1[ND[0]])*(U2-y2[ND[0]])
            if hv<0:
               hv=0
            for i in range(len(ND)-1):
                if (y1[ND[i+1]]>U1) or (y2[ND[i+1]]>U2):
                    a=0
                elif (y2[ND[i+1]]<=U2) and (y2[ND[i]]>U2):
                    a=(U2-y2[ND[i+1]])*(U1-y1[ND[i+1]])
                else:
                    a=(y2[ND[i]]-y2[ND[i+1]])*(U1-y1[ND[i+1]])
                hv=hv+a
            return hv

def NDC(Y,C,obj1_ascending=True):
    NDT=[]
    n=len(Y[0])
    Y=np.concatenate((Y[0],Y[1]),axis=1)
    index=[]
    for i in range(n):
        a=0
        if C[i].max()>0:
                a=1
        if a==0:
            index=np.append(index,i)
    t=len(index)
    if t==0:
        return NDT
    else:
        YF=np.zeros([t,2])
        for i in range(t):
            YF[i]=Y[int(index[i])]
        n=t
        for i in range(n):
            a= True 
            for j in range (n):
                if (YF[j][0]<YF[i][0] and YF[j][1]<=YF[i][1]):
                    a= False
                elif (YF[j][0]<=YF[i][0] and YF[j][1]<YF[i][1]):
                    a= False
            if a:
                NDT=np.append(NDT,index[i])
        swapped = True    
        while swapped:
            swapped=False
            for i in range(len(NDT)-1):
                if (Y[int(NDT[i+1])][0]<Y[int(NDT[i])][0]):
                    temp=NDT[i]
                    NDT[i]=NDT[i+1]
                    NDT[i+1]=temp
                    swapped=True
        ND=[0]*len(NDT)
        for i in range (len(NDT)):
            ND[i]=int(NDT[i])
        if obj1_ascending:
            return ND
        else:
            ND_=[0]*len(ND)
            for i in range(len(ND)):
                ND_[i] = ND[-(i+1)]
            return ND_


def ND_(ND):
    ND_=[0]*len(ND)
    for i in range(len(ND)):
        ND_[i] = ND[-(i+1)]
    return ND_

def Y_ND(Y,ND,nadir,ideal=[0,0]):
    Y0 = Y[0][ND]
    Y1 = Y[1][ND]
    Y_ = [np.zeros((len(ND)+2,1)),np.zeros((len(ND)+2,1))]
    Y_[0][1:-1] = Y0
    Y_[1][1:-1] = Y1
    Y_[0][0] = nadir[0]
    Y_[0][-1] = ideal[0]
    Y_[1][0] = ideal[1]
    Y_[1][-1] = nadir[1]
    return Y_

def psi(a,b,mu,sigma):
    normal = tfp.distributions.Normal(tf.cast(0.,tf.float64),tf.cast(1.,tf.float64))
    return sigma*normal.prob((b-mu)/sigma)+(a-mu)*normal.cdf((b-mu)/sigma)

# Change YND to ND
def EHVI(model_Y,Xcand,YND,corr=True,approximation='None',S=1000):
    normal = tfp.distributions.Normal(tf.cast(0.,tf.float64),tf.cast(1.,tf.float64))
    n = len(YND[0])
    if isinstance(model_Y,list):
        if model_Y[0].name == 'dgp':
            Fs1,Fmeans1,Fvars1 = model_Y[0].propagate(Xcand,S=S)
            # Fsamples = tf.reshape(Fs1[-1],(S,tf.shape(Xcand)[0],2))
            candidate_mean_0= tf.math.reduce_mean(Fmeans1[-1],axis=0)
            candidate_var_0 =  tf.math.reduce_mean(Fvars1[-1] + Fmeans1[-1]**2,axis=0) - candidate_mean_0**2
            Fs2,Fmeans2,Fvars2 = model_Y[1].propagate(Xcand,S=S)
            # Fsamples = tf.reshape(Fs2[-1],(S,tf.shape(Xcand)[0],2))
            candidate_mean_1= tf.math.reduce_mean(Fmeans2[-1],axis=0)
            candidate_var_1 =  tf.math.reduce_mean(Fvars2[-1] + Fmeans2[-1]**2,axis=0) - candidate_mean_1**2
        else:
            candidate_mean_0, candidate_var_0 = model_Y[0]._build_predict(Xcand)
            candidate_mean_1, candidate_var_1 = model_Y[1]._build_predict(Xcand)
    else:
        if model_Y.name == 'mo_dgp':
            Fs,Fmeans,Fvars = model_Y.model.propagate(Xcand,S=S)
            Fsamples = tf.reshape((tf.stack((Fs[-2],Fs[-1]),axis=2)),(S,tf.shape(Xcand)[0],2))
            candidate_mean_0= tf.math.reduce_mean(Fmeans[-2],axis=0)
            candidate_var_0 =  tf.math.reduce_mean(Fvars[-2] + Fmeans[-2]**2,axis=0) - candidate_mean_0**2
            candidate_mean_1 = tf.math.reduce_mean(Fmeans[-1],axis=0)
            candidate_var_1 = tf.math.reduce_mean(Fvars[-1] + Fmeans[-1]**2,axis=0) - candidate_mean_1**2
        if model_Y.name == 'coreg':
            candidate_mean_0, candidate_var_0 = model_Y._build_predict(tf.concat([Xcand, tf.zeros([tf.shape(Xcand)[0],1],dtype=tf.dtypes.float64)],axis=1))
            candidate_mean_1, candidate_var_1 = model_Y._build_predict(tf.concat([Xcand, tf.ones([tf.shape(Xcand)[0],1],dtype=tf.dtypes.float64)],axis=1))
            mu, var = model_Y._build_predict(tf.concat(([(tf.concat([Xcand, np.zeros([Xcand.shape[0],1])],axis=1)),tf.concat([Xcand, np.ones([Xcand.shape[0],1])],axis=1)]),axis=0), full_cov=True)  # N x P, # P x N x N
            jitter = tf.eye(tf.shape(mu)[0], dtype=tf.float64) * 1e-8
            samples = []
            L = tf.cholesky(var[0, :, :] + jitter)
            shape = tf.stack([tf.shape(L)[0], S])
            V = tf.random_normal(shape, dtype=tf.float64)
            samples.append(mu[:, 0:1] + tf.matmul(L, V))
            Fsamples = tf.transpose(tf.stack(samples))
            Fsamples=  tf.concat( (Fsamples[:,:tf.shape(Xcand)[0],:],Fsamples[:,tf.shape(Xcand)[0]:,:]),axis=2)
            # ////
            # Fsamples = tf.transpose((Fsamples),perm=[1,0,2])
            # Fsamples = tf.reshape((model_Y.predict_f_samples(np.concatenate(([(np.concatenate([Xcand, np.zeros([Xcand.shape[0],1])],axis=1)),np.concatenate([Xcand, np.ones([Xcand.shape[0],1])],axis=1)]),axis=0),S)),(tf.shape(Xcand)[0],S,2))
            # Fsamples = model_Y.predict_f_samples(np.concatenate(([(np.concatenate([Xcand, np.zeros([Xcand.shape[0],1])],axis=1)),np.concatenate([Xcand, np.ones([Xcand.shape[0],1])],axis=1)]),axis=0),S)
            # Fsamples=np.concatenate( (Fsamples[:,:len(Xcand),:],Fsamples[:,len(Xcand):,:]),axis=2)
            # Fsamples = tf.transpose((Fsamples),perm=[1,0,2])
            # Fsamples = tf.reshape((model_Y.predict_f_samples(np.concatenate(([(np.concatenate([Xcand, np.zeros([Xcand.shape[0],1])],axis=1)),np.concatenate([Xcand, np.ones([Xcand.shape[0],1])],axis=1)]),axis=0),S,full_output_cov= True)),(S,tf.shape(Xcand)[0],2))
    if approximation == 'None':
        if corr == True:
            print('No exact computation of the EHVI in the correlation case is implemented (yet)')
        else:
            term1 = tf.reduce_sum([(YND[0][i-1]-YND[0][i])*(normal.cdf((YND[0][i]-candidate_mean_0)/tf.math.sqrt(candidate_var_0))-normal.cdf((YND[0][-1]-candidate_mean_0)/tf.math.sqrt(candidate_var_0)))*(psi(YND[1][i],YND[1][i],candidate_mean_1,tf.math.sqrt(candidate_var_1))-psi(YND[1][i],YND[1][0],candidate_mean_1,tf.math.sqrt(candidate_var_1))) for i in range(1,n-1)],axis=0)
            term2 = tf.reduce_sum([(psi(YND[0][i-1],YND[0][i-1],candidate_mean_0,tf.math.sqrt(candidate_var_0))-psi(YND[0][i-1],YND[0][i],candidate_mean_0,tf.math.sqrt(candidate_var_0)))*(psi(YND[1][i],YND[1][i],candidate_mean_1,tf.math.sqrt(candidate_var_1))-psi(YND[1][i],YND[1][0],candidate_mean_1,tf.math.sqrt(candidate_var_1))) for i in range(1,n)],axis=0)
            EHVI_exact = term1 + term2
            return EHVI_exact
    if approximation == 'Gaussian':
        if corr == True:
            if model_Y.name == 'mo_dgp':
                Fsamples_bar = tf.math.reduce_mean(Fsamples,axis=0)
                Sigma = 1/(S)*(tf.linalg.matmul(tf.transpose((Fsamples-Fsamples_bar),perm=[1,0,2]),tf.transpose((Fsamples-Fsamples_bar),perm=[1,0,2]),transpose_a=True))
            if model_Y.name == 'coreg':
                Fsamples_bar = tf.math.reduce_mean(Fsamples,axis=0)
                # Coreg = tf.matmul(model_Y.kern.kernels[-1].W.constrained_tensor,model_Y.kern.kernels[-1].W.constrained_tensor,transpose_b=True)
                # Coreg = Coreg + tf.matrix_diag(model_Y.kern.kernels[-1].kappa.constrained_tensor)
                # Sigma = tf.matmul(Coreg,tf.matrix_diag(tf.reshape(tf.stack((candidate_var_0,candidate_var_1),axis=1),(len(Xcand),2))))
                Sigma = 1/(S)*(tf.linalg.matmul(tf.transpose((Fsamples-Fsamples_bar),perm=[1,0,2]),tf.transpose((Fsamples-Fsamples_bar),perm=[1,0,2]),transpose_a=True))
        elif corr == False:
            Sigma = tf.linalg.diag(tf.reshape(tf.stack((candidate_var_0,candidate_var_1),axis=1),(tf.shape(Xcand)[0],2)))
        term1 = 0
        for i in range(1,n-1):
            z = np.array([YND[0][i]-YND[0][-1],0.5*(YND[1][i]-YND[1][0])**2]).reshape(2,)
            lamda =np.array([0.5*(YND[0][i]+YND[0][-1]),1/3*(YND[1][i]+2*YND[1][0])]).reshape(2,)
            tau2 = np.array([1/12*(YND[0][i]-YND[0][-1])**2,1/18*(YND[1][i]-YND[1][0])**2]).reshape(2,)
            diag_tau2 = np.diag(tau2)
            multivariate = tfp.distributions.MultivariateNormalFullCovariance(tf.reshape(tf.stack((candidate_mean_0,candidate_mean_1),axis=1),(tf.shape(Xcand)[0],2)),Sigma+diag_tau2)
            res = multivariate.prob(lamda)
            res = (YND[0][i-1]-YND[0][i])*tf.reduce_prod(z)*res
            term1 = term1+res
        term2=0
        for i in range(1,n):
            z = np.array([0.5*(YND[0][i-1]-YND[0][i])**2,0.5*(YND[1][i]-YND[1][0])**2]).reshape(2,)
            lamda =np.array([1/3*(YND[0][i-1]+2*YND[0][i]),1/3*(YND[1][i]+2*YND[1][0])]).reshape(2,)
            tau2 = np.array([1/(18)*(YND[0][i-1]-YND[0][i])**2,1/(18)*(YND[1][i]-YND[1][0])**2]).reshape(2,)
            diag_tau2 = np.diag(tau2)
            multivariate = tfp.distributions.MultivariateNormalFullCovariance(tf.reshape(tf.stack((candidate_mean_0,candidate_mean_1),axis=1),(tf.shape(Xcand)[0],2)),Sigma+diag_tau2)
            res = multivariate.prob(lamda)
            res = tf.reduce_prod(z)*res
            term2=term2+res
        return tf.reshape((term1+term2),(tf.shape(Xcand)[0],1)) 
    if approximation == 'KDE':
        H = tf.linalg.diag(tf.reshape(((4/(2+2))**(1/(2+4)) * (S)**(-1/(2+4)) * tf.sqrt(tf.stack((candidate_var_0,candidate_var_1),axis=1)))**2,(tf.shape(Xcand)[0],2)))            
        term1 = tf.reduce_sum([1/S*(YND[0][i-1]-YND[0][i])*tf.reduce_sum(((normal.cdf((YND[0][i]-Fsamples[:,:,0])/tf.math.sqrt(H[:,0,0]))-normal.cdf((YND[0][-1]-Fsamples[:,:,0])/tf.math.sqrt(H[:,0,0])))*(psi(YND[1][i],YND[1][i],Fsamples[:,:,1],tf.math.sqrt(H[:,1,1]))-psi(YND[1][i],YND[1][0],Fsamples[:,:,1],tf.math.sqrt(H[:,1,1])))),axis=0) for i in range(1,n-1)],axis=0)
        term2 = tf.reduce_sum([1/S*tf.reduce_sum((psi(YND[0][i-1],YND[0][i-1],Fsamples[:,:,0],tf.math.sqrt(H[:,0,0]))-psi(YND[0][i-1],YND[0][i],Fsamples[:,:,0],tf.math.sqrt(H[:,0,0])))*(psi(YND[1][i],YND[1][i],Fsamples[:,:,1],tf.math.sqrt(H[:,1,1]))-psi(YND[1][i],YND[1][0],Fsamples[:,:,1],tf.math.sqrt(H[:,1,1]))),axis=0) for i in range(1,n)],axis=0)
        EHVI_exact = term1 + term2
        return tf.reshape((EHVI_exact),(tf.shape(Xcand)[0],1))
    
@tf.function
def loss(model_Y,Xcand,YND,corr=False,approximation='None',S=1000):
    """
    The Evidence Lower Bound Wrapped within a tensorflow function to create a tensorflow graph to accelerate the computation.
    """
    def closure():
            return EHVI(model_Y,Xcand,YND,corr=corr,approximation=approximation,S=S)
    return closure()
#### lr_adam
def optimize_EHVI(model,YND,popsize_DE=300,popstd_DE = 1.5,iterations_DE=400, init_adam=None,lr_adam=0.01,iterations_adam=1000,method='DE',corr=False,approximation='None',S=1000):
    bounds =(0,1)
    d = model._X[0].shape[1]
    lw, up = bounds
    fct_optim=lambda x:loss(model,lw+(up-lw)*(1/(1+tf.exp(x))),YND,corr=corr,S=S)
    if method == 'DE' or method == 'DE+Adam' :
        optim_results=tfp.optimizer.differential_evolution_minimize( fct_optim,initial_position=\
                                                                    tf.constant([0.]*d,tf.float64),\
                                                                        population_stddev=popsize_DE,population_size=popstd_DE,max_iterations=iterations_DE)
        x_opt=lw+(up-lw)*1/(1+np.exp((optim_results.position))).reshape(d,1)
    if method == 'Adam' or method == 'DE+Adam' :
        x_opt=np.array([[0]])
        if init_adam ==None:
            if x_opt == None:
                init_adam = [0.]*d
            else:
                init_adam = x_opt
        init_ = np.log((up-init_adam+1e-3)/(init_adam-lw+1e-3))
        init_tf = tf.Variable(init_)
        optimizer = tf.optimizers.Adam(learning_rate=lr_adam)
        for step in range(iterations_adam):
            with tf.GradientTape(watch_accessed_variables=True) as tape:
                tape.watch(init_tf)
                objective = fct_optim(init_tf)
                gradients = tape.gradient(objective, [init_tf])
            optimizer.apply_gradients(zip(gradients, [init_tf]))
        x_opt = lw+(up-lw)*1/(1+np.exp((init_tf.numpy()))).reshape(d,1)
    return x_opt