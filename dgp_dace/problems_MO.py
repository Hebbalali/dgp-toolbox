import numpy as np

class multi_obj_1D(object):
    def __init__(self):
        self.bounds = (-1., -1., 1., 3.)
        self.dim = 1
        self.hv_max = 0.47941844
    def fun(self,x):
        f1 = -x*(np.cos(15*(2*x-0.2)))
        f2 = x**2 * np.exp(np.cos(15*(2*x-0.2)))-1
        return [f1,f2]

class multi_obj_1D_2(object):
    def __init__(self):
        self.bounds = (-1., -4., 1., 1.)
        self.dim = 1
        self.hv_max = 0.47941844
    def fun(self,x):
        f1 = -np.cos(15*x)
        f2 = -x * np.exp(np.cos(15*(2*x-0.2)))-1
        return [f1,f2]


class multi_obj_1D_3(object):
    def __init__(self):
        self.bounds = (-16., -11., 6., 3.)
        self.dim = 1
        self.hv_max = 0.47941844
    def fun(self,x):
        f1 = -(6*x-2)**2* np.sin(12*x-4)
        f2 = -(0.5 *f1 + 10*(x-0.5) + 5 )
        return [f1,f2]

class multi_obj_1D_4(object):
    def __init__(self):
        self.bounds = (-16., -11., 6., 3.)
        self.dim = 1
        self.hv_max = 0.47941844
    def fun(self,x):
        f1 =  np.exp(np.cos(15*(2*x-0.2)))-1
        f2 =  -x * np.exp(np.cos(15*(2*x-0.2)))-1
        return [f1,f2]

class kursawe(object):
    def __init__(self):
        self.bounds = (-22., -14., 50., 50.)
        self.dim = 3
        self.hv_max = 0.47941844
    def fun(self,x):
        x = 10*x-5
        f1 = np.sum(-10*np.exp(-0.2*np.sqrt(x[:-1]**2+x[1:]**2)))
        f2 = np.sum(np.abs(x)**0.8+5*np.sin(x**3))
        return [f1,f2]

class kursawe_10d(object):
    def __init__(self):
        self.bounds = (-95., -45.,-60, 10.)
        self.dim = 10
        self.hv_max = 0.47941844
    def fun(self,x):
        x = 10*x-5
        f1 = np.sum(-10*np.exp(-0.2*np.sqrt(x[:-1]**2+x[1:]**2)))
        f2 = np.sum(np.abs(x)**0.8+5*np.sin(x**3))
        return [f1,f2]

class deb6(object):
    def __init__(self):
        self.bounds = (0., 0., 1., 1.)
        self.dim = 10
        self.hv_max = 0.32164096
    def fun(self,x):
        f1 = 1-np.exp(-4*x[0])*np.sin(6*np.pi*x[0])**6
        g = 1+9*((np.abs(np.sum(x[1:])))/9)**0.25
        h = 1 - (f1/g)**2
        f2 = g*h
        return [f1,f2]

class dtlz1a(object):
    def __init__(self):
        self.bounds = (-550., -550., 0., 0.)
        self.dim = 6
        self.hv_max = 0.41692852
    def fun(self,x): 
        g = 100*(5+np.sum((x[1:]-0.5)**2 - np.cos(2*np.pi*(x[1:]-0.5))))
        f1 = -0.5*x[1]*(1+g)
        f2 = -0.5*(1-x[1])*(1+g)
        return [f1,f2]
