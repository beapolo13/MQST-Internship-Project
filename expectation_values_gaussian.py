import numpy as np
from numpy import transpose, real, sqrt, sin, cos, linalg, cosh, sinh
import scipy
import matplotlib.pyplot as plt
from itertools import combinations
from scipy import optimize
from scipy.optimize import minimize
import time
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from pprint import pprint
from scipy.linalg import block_diag
import os
from mpl_toolkits.mplot3d import Axes3D

from utils import *


#GAUSSIAN STATE

#Expectation value of N
def expvalN(sigma): #input a 2N x 2N np.array of parameters for M
    N=len(sigma)//2
    #print('sigma',np.round(sigma,3))

    #now let's calculate the tr(prod(a's)rho). The amount of ladder operators is twice the number of modes (2N)
    #the amount of destruction operators is N, and the amount of creation is also N
    sum=0
    for i in range(1,N+1):
      ops=['adag','a']
      modes=[i,i]
      sum+=expectationvalue(sigma,ops,modes)
    return sum 


#Expectation value of N^2

def N2(sigma): #dispersion of number operator on gaussian state (rho0)
    #We now compute exp(N^2):
    N=len(sigma)//2
    sum=0
    for i in range(1,N+1):
      ops= ['adag','a','adag','a']
      modes=[i,i,i,i]
      sum+=expectationvalue(sigma,ops,modes)
    for i in range(1,N+1):
      for j in range(i+1,N+1):
        ops= ['adag','a','adag','a']
        modes=[i,i,j,j]
        sum+=2*expectationvalue(sigma,ops,modes)
    return sum



def varianceN(sigma):
    return  np.sqrt(N2(sigma) - (expvalN(sigma))**2) 

def SNR_gaussian(sigma):
  N=len(sigma)//2
  return (expvalN(sigma)+N/2)/varianceN(sigma) 

def SNR_gaussian_extr(sigma,sigma0):
  return (expvalN(sigma)-expvalN(sigma0))/varianceN(sigma) 