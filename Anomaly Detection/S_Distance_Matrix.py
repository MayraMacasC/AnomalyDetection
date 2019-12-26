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

def dist_euclidean(elem1, elem2):
    t_sum=0
    for i in range(len(elem1)):
        for j in range(len(elem1[0])):
            t_sum+= np.square(elem1[i][j]-elem2[i][j])
    return np.sqrt(t_sum)

def dist_euclidean_simple (elem1, elem2):
    t_sum=0
    for i in range(len(elem1)):
        r1 = elem1[i]
        r2 = elem2[i]
        t_sum+= sum([(a - b) ** 2 for a, b in zip(r1, r2)])
    return (math.sqrt(t_sum))    

def matrixdistance (test,prediction):
    R_dist = []
    for tmatrix in range (test.shape[0]): 
        #print (tmatrix)
        elem1 = test[tmatrix,:,:]
        elem2 = prediction[tmatrix,:,:]        
        dist = dist_euclidean_simple (elem1, elem2)        
        R_dist.append(dist)
    return(R_dist)


def plotmatrix1 (r_data,idx,nf,nc,size,typef):
    for i in idx:
        #print (i)
        dgt = r_data[2991].reshape(size,size)    
        plt.imshow(dgt, cmap="gray")#plt.cm.binary,interpolation='nearest'
        plt.savefig('matrix'+typef)
    
##plot matriceswinter
def plotmatrix (r_data,idx,nf,nc,size,typef):
    #Plot Reconstructed threes with the indexes chosen before
    fig, axs = plt.subplots(nf,nc, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .005, wspace=.001)
    axs = axs.ravel()
    j = 0
    for i in idx:
        dgt = r_data[i].reshape(size,size)
        axs[j].imshow(dgt, cmap="gray")#plt.cm.binary,interpolation='nearest'
        axs[j].set_yticklabels([])
        axs[j].set_xticklabels([])
        j = j +1
    plt.show()
    plt.savefig('matrix'+typef)


