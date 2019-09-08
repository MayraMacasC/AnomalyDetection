#!/usr/bin/python3 -u
# ==============================================================================
# Data Preparation
# ==============================================================================
import numpy as np
########################################################################################
# We have 3 cases the first is non-overlapping without stateful = False = case_1       #
# second, non-overlapping with stateful = True, case_2, in this case the timesteps     #
# and batchsize need to divide with number with module =0                              #
# third, overlapping with time step =1, case_3  with stateful=true                     #
########################################################################################
############################Case1#######################################################
def get_data_length (datalen, batch_size,timesteps, test_percent, train,case, ban):
    """
    substract test_percent to be excluded from training, reserved for validation
    """
    #length = len(dataset)
    length = datalen
    if train:
        length *= 1-test_percent  
    data_length_values = []
    for x in range(int(length)-100,int(length)):
        modulo1 = x%batch_size
        modulo2 = x%timesteps
        if ((case==1) or (case==2 and ban==1)):
            if (modulo2==0): #timesteps
                data_length_values.append(x)
        else:
            if (modulo1==0): #batch size      
                data_length_values.append(x)
    return (max(data_length_values))

################################## Data training#######################################
def addtimesteps (data,length,timesteps,train,case):
    if (case==3):
        upper_train = length + timesteps*2
    else:
        upper_train = length
    if train:
        data = data[0:upper_train]
        setdata = data.iloc[:,0:].values
    else:
        data = data[length:]
        setdata = data.iloc[:,0:].values
    yield (setdata)

####################################General Method#####################################
def method_split_data (timesteps,length,dataset,mode,mode_process):
    """
    mode 1 = overlapping 
    mode 2 = nonoverlapping 
    """
    if (mode==1):
        print (length)
        X_data = []
        if mode_process:
            frange=length
        else:
            frange=length+timesteps
        for i in range (timesteps, frange):#timesteps=1
            X_data.append(dataset[i-timesteps:i,0:])
        yield  (np.array(X_data, dtype=np.float32))
        #(np.array(X_data, dtype=np.float32))
    else:
        samples = list()
        for i in range(0,length,timesteps):
            sample = dataset[i:i+timesteps]
            samples.append(sample)
        yield (np.array(samples, dtype=np.float32))
        #(np.array(samples, dtype=np.float32))
    