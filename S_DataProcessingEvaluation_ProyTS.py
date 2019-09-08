#!/usr/bin/python3 -u
# ==============================================================================
# Data Processing Evaluation
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

def scaledata (data):
    sc = MinMaxScaler(feature_range = (0,1))
    dataset_scaled = sc.fit_transform(np.float64(data))
    print (dataset_scaled.shape)
    return (np.array(dataset_scaled, dtype=np.float32))


def prepare_data(data_X,data_Y,numft,x,y,xzp,yzp):
    """
    #Clean the data and Split data into Anomaly data and testmal data 
    """
    #Dropping the labels to a different dataset which is used to train the recurrent neural network classifier    
    df_test_X = data_X
    df_test_Y = 1-data_Y ##flipping the digits#####
    ##zero padding
    df_test_X = df_test_X.values
    #zp    
    df_test_X = df_test_X.reshape(-1,x,y)
    df_test_X = np.pad(df_test_X, ((0,0),(1,0),(1,0)), mode='constant', constant_values=0)
    df_test_X = df_test_X.reshape(-1,xzp*yzp)
        
    df_test_X = pd.DataFrame(df_test_X)  
    df_test_X = df_test_X.round(4)
    #converting input data into float which is requried in the future stage of building in the network
    df_test_X = df_test_X.loc[:,df_test_X.columns[0:numft]].astype(float)
    #data collation
    split_data = collections.namedtuple('split_data', 'o_df_test_X o_df_test_Y')    
    datat = split_data(o_df_test_X=df_test_X, o_df_test_Y=df_test_Y)
    yield datat
     
def overlaping_data (xzp,yzp,inChannel,test_norm_X,test_norm_Y,datalen,
                               batch_size,timesteps, test_percent,case,mode,numft,mode_process):
    """
    mode 2 : nonoverlapping
    mode 1 : withoverlapping
    """
    #Dividing the dataset into test and test datasets
    test_data_X = pd.DataFrame(test_norm_X)
    ####################################test_X - case 1#######################
    #1.1 (datalen, batch_size,timesteps, test_percent, test=False,case)
    length_test_c1 = get_data_length (datalen, batch_size,timesteps,test_percent,True,case,1) #legth timesteps
    if (case==2):
        d = length_test_c1/timesteps
        length_test_b = get_data_length (d, batch_size,timesteps,test_percent,False,case,0) #batch size
        length_test_c1 = length_test_b * timesteps
    #1.2
    for testing_set_c1 in addtimesteps(test_data_X,length_test_c1,timesteps,True,case):
        #1.3 
        for data_c1 in method_split_data (timesteps,length_test_c1,testing_set_c1,mode,mode_process):
            #1.4 
            samples_test_c1 = len (data_c1)
            data_c1 = data_c1.reshape((samples_test_c1, timesteps, numft-1))
            dataspt_test_c1 = data_c1.reshape((samples_test_c1, timesteps, xzp, xzp, 1))
    ####################################test_Y - case 1#######################          
    #1.2
    for testing_set_Y_c1 in addtimesteps(test_norm_Y,length_test_c1,timesteps,True,case):
        #1.3
        for data_Y_c1 in method_split_data(timesteps,length_test_c1,testing_set_Y_c1,mode,mode_process):
            samples_test_Y_c1 = len (data_Y_c1)
            #1.4 reshape into [samples, timesteps, features]
            data_Y_c1 = data_Y_c1.reshape((samples_test_Y_c1, timesteps, 1))            
            yield dataspt_test_c1,data_Y_c1


def slip_normal_data_to_test (xzp,yzp,inChannel,test_norm_X,test_norm_Y,datalen,
                               batch_size,timesteps, test_percent,case,mode,numft,mode_process):
    """
    #Split normal data in order to feed the models for testing the models. 
    """
    ###format to feed CNNN and ConvLSTM
    for Xtest,Ytest in overlaping_data (xzp,yzp,inChannel,test_norm_X,test_norm_Y,
                                   datalen,batch_size,timesteps, test_percent,case,mode,numft,mode_process):       
        ##format to feed LSTM
        Xtest_r= np.reshape(Xtest,(Xtest.shape[0],Xtest.shape[1],(Xtest.shape[2]*Xtest.shape[3]*Xtest.shape[4])))
        ##Reshape, it is useful, in mode nonoverlapping
        Xtest_CNNr= np.reshape(Xtest,(Xtest.shape[0],(Xtest.shape[1]*Xtest.shape[2]),Xtest.shape[3],Xtest.shape[4]))
        ################################Model 01 SPT#####################################
        data_M01 = collections.namedtuple('data_M01', 'o_Xtest o_Ytest')
        model01_test = data_M01(o_Xtest=Xtest, o_Ytest=Ytest)
        ##############################Model 02 LSTM_AE##########################################
        data_M02 = collections.namedtuple('data_M02', 'o_Xtest_r o_Ytest_r')
        model02_test = data_M02 (o_Xtest_r=Xtest_r, o_Ytest_r=Ytest)
        ##############################Model 03 OC-SVM###########################################
        data_M03 = collections.namedtuple('data_M03', 'o_X_OCSVM o_Y_OCSVM')
        model03_test = data_M03 (o_X_OCSVM=test_norm_X, o_Y_OCSVM=test_norm_Y)
        ##############################Model 04 CNN###########################################        
        data_M04 = collections.namedtuple('data_M04', 'o_Xtest_CNNr o_Ytest_CNNr')
        model04_test = data_M04(o_Xtest_CNNr=Xtest_CNNr, o_Ytest_CNNr=Ytest)    
        ###############################Collation of data test###################################
        c_data_test_models = collections.namedtuple('c_data_test_models', 'model1 model2 model3 model4')
        data_test_models = c_data_test_models(model1=model01_test, model2=model02_test, model3=model03_test,model4=model04_test)    
        yield data_test_models


def process_data (case,mode,data_pathXE1,data_pathYE1):#
    """
    #Genereting train and test data  
    """
    args = parse_args()
    data_pathXE = data_pathXE1#data_pathXE1 #data_pathXE
    data_pathYE = data_pathYE1#data_pathYE1 #data_pathYE
    numft = args.numft    
    win = args.win    
    mode_process = args.eval    
    x = args.x
    y = args.y    
    xzp = args.xzp
    yzp = args.yzp
    timesteps = args.time_steps    
    inChannel = args.inChannel  
    test_percent = args.test_percent_eval
    for data_X,data_Y in load_data(data_pathXE, data_pathYE, win):
        for datat in prepare_data(data_X,data_Y,numft,x,y,xzp,yzp):
            datalen = len (datat.o_df_test_X);print ("yessssssssssssssssssssssssssssssssssssss")
            batch_size = args.batch_size
            for data_train_models in slip_normal_data_to_test (xzp,yzp,inChannel,datat.o_df_test_X,datat.o_df_test_Y,datalen,batch_size,timesteps, test_percent,case,mode,numft,mode_process):
                yield data_train_models
               

            