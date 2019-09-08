#!/usr/bin/python3 -u
# ==============================================================================
# DataProcessing
# ==============================================================================
import numpy as np
import pandas as pd 
import collections
from S_Parameters_ProyTS import parse_args
from S_DataPrepartion_ProyTS import get_data_length,addtimesteps,method_split_data
from sklearn.preprocessing import MinMaxScaler

def load_data (data_to_pathX, data_to_pathY, win):
    """
    #Load data
    """
    print ("Laoding the sensor data")
    data_X = pd.read_csv(data_to_pathX,engine='python',sep=",",header=0) 
    data_X = data_X.drop(data_X.columns[0],axis=1,inplace = False)
    data_Y = pd.read_csv(data_to_pathY,engine='python',sep=",",header=0, index_col=0)
    yield data_X,data_Y

def prepare_data(data_X,data_Y,numft,x,y,xzp,yzp):
    """
    #Clean and Split data into Anomaly data and Normal data 
    """
    #Dropping the labels to a different dataset which is used to train the recurrent neural network classifier    
    df_nor_X = data_X
    df_nor_Y = data_Y 
    ##zero padding
    df_nor_X = df_nor_X.values
  
    #zp    
    df_nor_X = df_nor_X.reshape(-1,x,y)
    df_nor_X = np.pad(df_nor_X, ((0,0),(1,0),(1,0)), mode='constant', constant_values=0)
    df_nor_X = df_nor_X.reshape(-1,xzp*yzp)
    
    #scaled 
    #df_nor_X = scaledata (df_nor_X);print ("SCALED")
    
    df_nor_X = pd.DataFrame(df_nor_X);print ("NOT SCALED"); df_nor_X = df_nor_X.round(4); print ("FOUR DIGITS")
    #converting input data into float which is requried in the future stage of building in the network
    df_nor_X = df_nor_X.loc[:,df_nor_X.columns[0:numft]].astype(float)
    #data collation
    split_data = collections.namedtuple('split_data', 'o_df_nor_X o_df_nor_Y')    
    datat = split_data(o_df_nor_X=df_nor_X, o_df_nor_Y=df_nor_Y)
    yield datat

def overlaping_data (xzp,yzp,inChannel,train_norm_X,train_norm_Y,datalen,
                               batch_size,timesteps, test_percent,case,mode,numft,mode_process):
    """
    mode 2 : nonoverlapping
    mode 1 : withoverlapping
    """
    #Dividing the dataset into train and test datasets
    train_data_X = pd.DataFrame(train_norm_X)
    ####################################train_X - case 1#######################
    #1.1 (datalen, batch_size,timesteps, test_percent, train=False,case)
    length_train_c1 = get_data_length (datalen, batch_size,timesteps,test_percent,True,case,1) #legth timesteps
    if (case==2):
        d = length_train_c1/timesteps
        length_train_b = get_data_length (d, batch_size,timesteps,test_percent,False,case,0) #batch size
        length_train_c1 = length_train_b * timesteps
    #1.2
    for training_set_c1 in addtimesteps(train_data_X,length_train_c1,timesteps,True,case):
        #1.3 
        for data_c1 in method_split_data (timesteps,length_train_c1,training_set_c1,mode,mode_process):
            #1.4 
            samples_train_c1 = len (data_c1)
            data_c1 = data_c1.reshape((samples_train_c1, timesteps, numft-1))
            dataspt_train_c1 = data_c1.reshape((samples_train_c1, timesteps, xzp, xzp, 1))
    ####################################train_Y - case 1#######################          
    #1.2
    for training_set_Y_c1 in addtimesteps(train_norm_Y,length_train_c1,timesteps,True,case):
        #1.3
        for data_Y_c1 in method_split_data(timesteps,length_train_c1,training_set_Y_c1,mode,mode_process):
            samples_train_Y_c1 = len (data_Y_c1)
            #1.4 reshape into [samples, timesteps, features]
            data_Y_c1 = data_Y_c1.reshape((samples_train_Y_c1, timesteps, 1))
    ###############################Validation X_case 1#########################
    #1.2
    length_val_r_c1 = len (train_data_X) - length_train_c1
    #1.3
    if (mode==1):
        length_val_r_c1 = length_val_r_c1-timesteps
    length_val_c1 = get_data_length (length_val_r_c1,batch_size,timesteps,test_percent,False,case,1)#time steps
    if (case==2):
        d = length_val_c1/timesteps
        length_val_b = get_data_length (d, batch_size,timesteps,test_percent,False,case,0)# batch size
        length_val_c1 = length_val_b * timesteps    
    #1.4 
    for validation_set_c1 in addtimesteps(train_data_X,length_train_c1,timesteps,False,case):
        #1.5
        for X_val_c1 in method_split_data(timesteps,length_val_c1,validation_set_c1,mode,mode_process):
            #1.6
            X_val_c1 = np.reshape(X_val_c1,(X_val_c1.shape[0],X_val_c1.shape[1],X_val_c1.shape[2]))
            #1.7
            X_val_SPT_c1 = X_val_c1.reshape(X_val_c1.shape[0],X_val_c1.shape[1], xzp, xzp, 1)    
    ###############################Validation Y_case 1#########################    
    #1.1
    for validation_set_Y_c1 in addtimesteps(train_norm_Y,length_train_c1,timesteps,False,case):
        #1.2
        for Y_val_c1 in method_split_data(timesteps,length_val_c1,validation_set_Y_c1,mode,mode_process):
            #1.3
            Y_val_c1 = np.reshape(Y_val_c1,(Y_val_c1.shape[0],Y_val_c1.shape[1],Y_val_c1.shape[2]))
            
            yield dataspt_train_c1,data_Y_c1,X_val_SPT_c1,Y_val_c1

def slip_normal_data_to_train (xzp,yzp,inChannel,train_norm_X,train_norm_Y,datalen,
                               batch_size,timesteps, test_percent,case,mode,numft,mode_process):
    """
    #Split normal data in order to feed the models for training the models. 
    """
    ###format to feed CNNN and ConvLSTM
    for Xtrain,Ytrain,Xval,Yval in overlaping_data (xzp,yzp,inChannel,train_norm_X,train_norm_Y,
                                   datalen,batch_size,timesteps, test_percent,case,mode,numft,mode_process):       
        ##format to feed LSTM
        Xtrain_r= np.reshape(Xtrain,(Xtrain.shape[0],Xtrain.shape[1],(Xtrain.shape[2]*Xtrain.shape[3]*Xtrain.shape[4])))
        Xval_r= np.reshape(Xval,(Xval.shape[0],Xval.shape[1],(Xval.shape[2]*Xval.shape[3]*Xval.shape[4])))  
        ##Reshape, it is useful, in mode nonoverlapping
        Xtrain_CNNr= np.reshape(Xtrain,(Xtrain.shape[0],(Xtrain.shape[1]*Xtrain.shape[2]),Xtrain.shape[3],Xtrain.shape[4]))
        Xval_CNNr= np.reshape(Xval,(Xval.shape[0],(Xval.shape[1]*Xval.shape[2]),Xval.shape[3],Xval.shape[4]))          
        ################################Model 01 SPT#####################################
        data_M01 = collections.namedtuple('data_M01', 'o_Xtrain o_Ytrain o_Xval o_Yval')
        model01_train = data_M01(o_Xtrain=Xtrain, o_Ytrain=Ytrain,o_Xval=Xval, o_Yval=Yval)
        ##############################Model 02 LSTM_AE##########################################
        data_M02 = collections.namedtuple('data_M02', 'o_Xtrain_r o_Ytrain_r o_Xval_r o_Yval_r')
        model02_train = data_M02 (o_Xtrain_r=Xtrain_r, o_Ytrain_r=Ytrain,o_Xval_r=Xval_r, o_Yval_r=Yval)
        ##############################Model 03 OC-SVM###########################################
        data_M03 = collections.namedtuple('data_M03', 'o_newtrain_M04X')
        model03_train = data_M03 (o_newtrain_M04X=train_norm_X)
        ##############################Model 04 CNN###########################################        
        data_M04 = collections.namedtuple('data_M04', 'o_Xtrain_CNNr o_Ytrain_CNNr o_Xval_CNNr o_Yval_CNNr')
        model04_train = data_M04(o_Xtrain_CNNr=Xtrain_CNNr, o_Ytrain_CNNr=Ytrain,o_Xval_CNNr=Xval_CNNr, o_Yval_CNNr=Yval)    
        ###############################Collation of data train###################################
        c_data_train_models = collections.namedtuple('c_data_train_models', 'model1 model2 model3 model4')
        data_train_models = c_data_train_models(model1=model01_train, model2=model02_train, model3=model03_train,model4=model04_train)    
        yield data_train_models


def process_data (case,mode):
    """
    #Genereting train and test data  
    """
    args = parse_args()
    data_pathX = args.data_pathX #data_pathX
    data_pathY = args.data_pathY #data_pathY
    numft = args.numft    
    win = args.win    
    mode_process = not (args.eval)    
    print (mode_process)
    x = args.x
    y = args.y    
    xzp = args.xzp
    yzp = args.yzp
    timesteps = args.time_steps    
    inChannel = args.inChannel  
    test_percent = args.test_percent
    for data_X,data_Y in load_data(data_pathX, data_pathY, win):
        for datat in prepare_data(data_X,data_Y,numft,x,y,xzp,yzp):
            datalen = len (datat.o_df_nor_X)
            batch_size = args.batch_size
            for data_train_models in slip_normal_data_to_train (xzp,yzp,inChannel,datat.o_df_nor_X,datat.o_df_nor_Y,datalen,batch_size,timesteps, test_percent,case,mode,numft,mode_process):
                yield data_train_models


    
