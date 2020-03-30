#!/usr/bin/python3 -u
# ==============================================================================
# Distance Matrix
# ==============================================================================
import numpy as np
import math
import matplotlib.pyplot as plt

def distance_matrix (test, pred):
    R_dist = []
    R_dist_MSE = []
    residualmatrix = abs(test-pred)
    for  tmatrix in range (residualmatrix.shape[0]):
        #print (tmatrix)
        rpower = residualmatrix[tmatrix,:,:] * residualmatrix[tmatrix,:,:]
        rsum_residual = rpower.sum()
        rsum_residual = np.sqrt(rsum_residual)
        mse_residual = rsum_residual/(52*52)     
        R_dist.append(rsum_residual)
        R_dist_MSE.append(mse_residual)
    yield (residualmatrix,R_dist,R_dist_MSE)



