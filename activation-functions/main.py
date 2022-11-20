import numpy as np
from scipy.special import erf
def linear(x):
    return x

def binary(x):
    return 1 if x >= 0 else 0

def sigmoid(x):
    1/(1+np.exp(-x))

def tanh(x):
    a = np.exp(x)
    b = np.exp(-x)
    return (a-b)/(a+b)

def relu(x):
    return x if x > 0 else 0

def gelu(x):
    return (0.5*x)*(1+erf(x/np.sqrt(2)))

def softplus(x):
    np.log(1+np.exp(x))

def elu(x, alpha=1.0):
    a = alpha*(np.exp(x)-1)
    return a if x <= 0 else 0

def SELU(x):
    lambdaa = 1.0507
    alpha = 1.6732
    if x >= 0:
        return lambdaa * x
    else:
        return lambdaa * alpha * (np.exp(x) - 1)

def LeakyReLU(x):
    return 0.01*x if x < 0 else x

def PReLU(x, alpha = 1.0):
    return alpha*x if x < 0 else x

def SiLU(x):
    return x/(1+np.exp(-x))

def Gaussian(x):
    return np.exp(-1*(x**2))