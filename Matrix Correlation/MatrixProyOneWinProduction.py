#!/usr/bin/python3 -u

########################################################################################################
##Proyect TS/MM
##Correlation Matrix
########################################################################################################

from pandas import read_csv
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import math
from BayesCorrelation import process_model

def LoadData(pathX,pathY):
    """Preprocessing the dataset
    Attributes
    ----------   
    path: the path of the file

    Returns
    -------     
    Data and label (attack/normal) separated
    """
    start = datetime.now()
    print("[" + str(start) + "]" + " Starting LoadData.")
    
    #Read Data   
    X = read_csv(pathX,sep=',', header=0, index_col=0)
    Y = read_csv(pathY,sep=',', header=0, index_col=0)
    end = datetime.now()
    print("[" + str(end) + "]" + " Finished LoadData. Duration: " + str(end-start))
    
    return X,Y

def Windowing (mt,w,n,t,intv):
    """
    Attributes
    ----------     
    mt: dataset
    w = list of windows
    n = number of sensores
    t = number of time series 
    intv = interval of time 
    
    Returns
    -------       
    Return: list of tensors
    """
    start = datetime.now()
    print("[" + str(start) + "]" + " Starting Window.")
    
    listw = w
    #number of windows
    numwin = len(listw)
    #time start
    tstart = max(w)+1
    data = mt
    #list of tensors 
    MT = []
    for tiempo in range (tstart,t+1,intv):    
        mtt = []
        for x in range(numwin):
            #window
            tp = tstart
            win = listw[x]
            inwin = tp-win
            #generating the windows matrix 
            wmat = data[:, inwin-1:tp]   
            #print (wmat)
            mt = []
            #genereting the signature matrix
            for r in range (n):
                vec1 = wmat[r:r+1, :]
                #print (vec1)
                array1 = []
                n12 = np.squeeze(np.asarray(vec1))
                for c in range (n):
                    vec2 = wmat[c:c+1, :]
                    X12 = np.squeeze(np.asarray(vec2))
					data = np.array([n12,X12])
                    e = process_model(data)
                    array1.append(e)
                mt.append (array1)
            mtt.append(mt)
        tensor1= mtt[0]
        MT.append(tensor1)
        tstart = tstart+intv
    end = datetime.now()
    print("[" + str(end) + "]" + " Finished Window. Duration: " + str(end-start))
    return (MT)


def Flattensor (listtensor):
    """Flat lists of tensors
    Attributes
    ----------     
    listtensor: list of tensors
    
    Returns
    -------       
    Dataframe of tensors in 1D  
    """
    start = datetime.now()
    print("[" + str(start) + "]" + " Starting flattensor.")
    t=len(listtensor)
    ##### 0 element 
    auxtensor = listtensor[0]
    #flatten
    auxflat = [val for sublist in auxtensor for val in sublist] 
    #transofrm to dataframe
    dfxf = pd.DataFrame(auxflat)
    for x in range (1,t):
        print (x)
        tensor = listtensor[x]
        flat = [val for sublist in tensor for val in sublist]  
        dfflat= pd.DataFrame(flat)
        dfxf = pd.concat([dfxf, dfflat],axis=1)
    end = datetime.now()
    print("[" + str(end) + "]" + " Finished flattensor. Duration: " + str(end-start))
    #return(dfxf.transpose())
    return(dfxf)


def main (pathX,pathY,listw, intv):
    """Main process
    Attributes
    ----------  
    listw: list of windows
    intv: interval
    pathX = file X
    pathY = file X
    
    Returns
    -------      
    X, Y datasets    
    """    
    #Load File
    X, Y = LoadData(pathX,pathY)    
    #Parameters
    ns = X.shape[1]
    dfp = X.values  
    #transpose dataframe in order to generete the signature matrix
    XT = dfp.transpose()
    ###########parameters to generete the signature matrix###################
    #numbers of times
    t=XT.shape[1]  
    #matrix correlation
    aux= Windowing(XT,listw,ns,t,intv)    
    #Y component to the network    
    YF = Y.iloc[max(listw):]   
    #print (YF)
    YF.to_csv('Y_s2.csv')       
    #flatting the tensors
    cf_df = Flattensor(aux)
    #file 
    cf_df.to_csv('X_s2.csv')        
    
if __name__ == '__main__':
    
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-px", "--path-to-filename-x", type=str, required=True,
        help="path to csv file X")
    ap.add_argument("-py", "--path-to-filename-y", type=str, required=True,
        help="path to csv file Y")    
    ap.add_argument("-i", "--interval", type=int, default=1,
        help="interval")
    ap.add_argument("-w", "--windows", nargs=3, type=int, default=[150],
        help="lengths of three windows")
    args = vars(ap.parse_args())    
    
    #time
    s = datetime.now()
    print("[" + str(s) + "]" + " Starting Execution...")    
    #process
    main(args["path_to_filename_x"], args["path_to_filename_y"],args["windows"], args["interval"])
       
    e = datetime.now()
    print("[" + str(e) + "]" + " Finished Execution! Total script duration: " + str(e-s))
    print("I did it! I did it!")
