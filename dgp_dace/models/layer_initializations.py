
# import tensorflow as tf
# import numpy as np
# import pyDOE
# from gpflow.params import DataHolder, Minibatch
# from gpflow import autoflow, params_as_tensors, ParamList
# from gpflow.models.model import Model
# from gpflow.mean_functions import Identity, Linear
# from gpflow.mean_functions import Zero
# from gpflow.quadrature import mvhermgauss
# from gpflow import settings
# float_type = settings.float_type

# from layers import SVGP_Layer


  
import numpy as np
from gpflow.mean_functions import Identity, Linear, Zero
from gpflow import set_trainable
from layers import SVGP_Layer


def init_layers_linear(X, Y, Z, kernels,
                       num_units,
                       num_outputs=None,
                       mean_function=Zero(),
                       Layer=SVGP_Layer,
                       white=False):
    num_outputs = num_outputs or Y.shape[1]

    layers = []

    X_running, Z_running = X.copy(), Z.copy()
    counter=0
    for num_units_in,num_units_out, kern_in in zip(num_units[:-1],num_units[1:],kernels[:-1]):
        dim_in = num_units_in
        dim_out = num_units_out
        print(dim_in, dim_out)
        if dim_in == dim_out:
            mf = Identity()
            #mf=Zero()
        else:
            if dim_in > dim_out:  # stepping down, use the pca projection
                _, _, V = np.linalg.svd(X_running, full_matrices=False)
                W = V[:dim_out, :].T

            else: # stepping up, use identity + padding
                W = np.concatenate([np.eye(dim_in), np.zeros((dim_in, dim_out - dim_in))], 1)

            mf = Linear(W)
#            mf=Zero()
#            mf = Identity()
            set_trainable(mf,False)

        layers.append(Layer(kern_in, Z_running, dim_out, mf, white=white))

        if dim_in != dim_out:
            Z_running = Z_running.dot(W)
            X_running = X_running.dot(W)
#        Z_running = pyDOE.lhs(dim_out,dim_out*15,seed=123+counter)
#        Z_running = (Z_running-Z_running.mean())/Z_running.std()
        counter=counter+1

    # final layer
    layers.append(Layer(kernels[-1], Z_running, num_outputs, mean_function, white=white))
    return layers
