#!/usr/bin/python3 -u
# ==============================================================================
# Data Processing Evaluation
# ==============================================================================
import numpy as np
import tensorflow as tf
from keras import backend as K
from S_Distance_Matrix  import distance_matrix
import pickle
import pandas as pd
import datetime
        
K.clear_session() 
tf.keras.backend.clear_session()
keras=tf.contrib.keras
np.random.seed(42)

#save record
def saverecord (listinput,name):
    #save the list in output.bin file
    with open(name, "wb") as output:
        pickle.dump(listinput, output)
                
#data to evaluation
def datatotest (model_name,test_data_model01,test_data_model02,test_data_model04):
        if model_name in ['SpatioTemporalSimple',
                   'STAE_AD']:
            #x_test data 
            x_test = test_data_model01.o_Xtest
            #y_test data
            y_test = test_data_model01.o_Ytest            
        elif model_name == 'LSTM_AE':
            ##x_test data 
            x_test = test_data_model02.o_Xtest_r
            #y_test data
            y_test = test_data_model02.o_Ytest_r
        elif model_name == 'CNN_Noise':
            #x_test data
            x_test = test_data_model04.o_Xtest_CNNr
            #y_test data
            y_test = test_data_model04.o_Ytest_CNNr              
        else:
            raise ValueError('Unknown model_name %s was given' % model_name)
        yield (x_test,y_test)
        
#overlapping_reshape
def overlapping_reshape (dataP, dataX, dataY, model_name):
    length = dataY.shape[0]
    #####Y#####
    R_dataY = []
    for tiempo in range (0,length): 
        R_dataY.append(dataY[tiempo,:1,:1])
    Y = (np.array(R_dataY, dtype=np.float32))
    #####X######
    if model_name in ['SpatioTemporalSimple',
                   'STAE_AD']:
        numcolX = dataX.shape[2]*dataX.shape[3]*dataX.shape[4]
        numcolP = dataP.shape[2]*dataP.shape[3]*dataP.shape[4]
        dataX = dataX.reshape(dataX.shape[0],dataX.shape[1],dataX.shape[2]*dataX.shape[3]*dataX.shape[4])
        dataP = dataP.reshape(dataP.shape[0],dataP.shape[1],dataP.shape[2]*dataP.shape[3]*dataP.shape[4])

    elif model_name == 'LSTM_AE':
        numcolX = dataX.shape[2]
        numcolP = dataP.shape[2]        
        dataX = dataX.reshape(dataX.shape[0],dataX.shape[1],dataX.shape[2])
        dataP = dataP.reshape(dataP.shape[0],dataP.shape[1],dataP.shape[2])
    else:
        raise ValueError('Unknown model_name %s was given' % model_name)
    ####Taking the last value at time t####
    R_dataX = []
    for tiempo in range (0,length): 
        R_dataX.append(dataX[tiempo,:1,:numcolX])
    X = (np.array(R_dataX, dtype=np.float32))  
    
    ####Predict value#######
    R_dataP = []
    for tiempo in range (0,length): 
        R_dataP.append(dataP[tiempo,:1,:numcolP])
    Pr = (np.array(R_dataP, dtype=np.float32))  
    yield (X,Y,Pr)    
    
#data reshape
def datareshape (model_name,x_test,y_test,predictions,mode):
        if (model_name in ['SpatioTemporalSimple',
                   'STAE_AD']) and ((mode==2)):
            #reshape test data
            r_x_test = np.reshape(x_test,(x_test.shape[0]*x_test.shape[1],x_test.shape[2]*x_test.shape[3]*x_test.shape[4]))
            r_x_test_matrix = np.reshape(x_test,(x_test.shape[0]*x_test.shape[1],x_test.shape[2],x_test.shape[3]*x_test.shape[4]))
            #reshape predictions 
            r_predictions = np.reshape(predictions,(predictions.shape[0]*predictions.shape[1],predictions.shape[2]*predictions.shape[3]*predictions.shape[4]))
            r_predictions_matrix = np.reshape(predictions,(predictions.shape[0]*predictions.shape[1],predictions.shape[2], predictions.shape[3]*predictions.shape[4]))            
            #reshape test
            r_y_test = np.reshape(y_test,(y_test.shape[0]*y_test.shape[1]*y_test.shape[2]))
            
        elif (model_name == 'LSTM_AE') or ((model_name=='ModelWithNoise') and (mode==1)):
            #reshape test data
            r_x_test = np.reshape(x_test,((x_test.shape[0]*x_test.shape[1]),x_test.shape[2]))
            #reshape predictions 
            r_predictions = np.reshape(predictions,((predictions.shape[0]*predictions.shape[1]),predictions.shape[2]))
            #reshape test
            r_y_test = np.reshape(y_test,(y_test.shape[0]*y_test.shape[1]*y_test.shape[2]))
            
        elif model_name == 'CNN_Noise':
            #reshape test data
            r_x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3]));r_x_test_matrix = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],x_test.shape[2]*x_test.shape[3]))
            #reshape predictions 
            r_predictions = np.reshape(predictions,(predictions.shape[0],predictions.shape[1]*predictions.shape[2]*predictions.shape[3]));r_predictions_matrix = np.reshape(predictions,(predictions.shape[0],predictions.shape[1],predictions.shape[2]*predictions.shape[3]))            
            #reshape test
            r_y_test = np.reshape(y_test,(y_test.shape[0]*y_test.shape[1]*y_test.shape[2]))
        else:
            raise ValueError('Unknown model_name %s was given' % model_name)
        yield  (r_x_test,r_predictions,r_y_test,r_x_test_matrix,r_predictions_matrix)    
        
##adding noise
def add_noise (data):
    #Adding noise 
    noise_factor = 0.5
    x_data_noise = data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    x_data_noise = np.clip(data, 0., 1.)
    return (x_data_noise)      
  
def save_layer_info (model):
    layers_info = {}
    layer_weights = {}
    for i in model.layers:
        print (i)
        layers_info[i.name] = i.get_config()
        layer_weights[i.name] = i.get_weights()
        saverecord (layer_weights,'layer_weightsF.bin')
        saverecord (layers_info,'layers_infoF.bin')
               
class MyCustomCallback(tf.keras.callbacks.Callback):

  def on_train_batch_begin(self, batch, logs=None):
    print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_train_batch_end(self, batch, logs=None):
    print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

  def on_test_batch_begin(self, batch, logs=None):
    print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_test_batch_end(self, batch, logs=None):
    print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

from VarConvLSTM1 import AttenIConvLSTM2D
def root_mean_squared_error(actual, predicted):
  return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(actual - predicted), axis=1))
from datetime import datetime
def extract_error (data,mode,h5model,model_names,custom_object,ban):        
    countmol = 0    
    #test
    test_data_model01 = data.model1
    test_data_model02 = data.model2
    test_data_model04 = data.model4
    evaluationscores = []
    ##models    
    for model_name in model_names:
        # reshape input data according to the model's input tensor
        for x_test,y_test in datatotest (model_name,test_data_model01,test_data_model02,test_data_model04):            
            print (model_name)
            # load model
            model = tf.keras.models.load_model(h5model, custom_objects=custom_object)
            # prediction
            predictions = model.predict(x_test, verbose=2,batch_size=64)
            if (ban=='yes'):
                # evaluation
                print ("EVALUATION");start = datetime.now();print("[" + str(start) + "]" + " Starting LoadData.")
                scores = model.evaluate(x_test, x_test,batch_size=32);end = datetime.now();print("[" + str(end) + "]" + " Finished LoadData. Duration: " + str(end-start));#save_layer_info (model)
                evaluationscores.append(scores)
                listmetrics = model.metrics_names; print (model.metrics_names)
                evaluationscores.append(listmetrics)
                saverecord (evaluationscores,'evaluationscores_T.bin')
                proyectlist = pd.DataFrame(
                        {'metric': listmetrics,'value': scores});proyectlist.to_csv('ResultScores.csv', index= True)                                
                saverecord (evaluationscores,'evaluationscores.bin')
                out = proyectlist.loc[proyectlist['metric'] == 'mean_absolute_error']
                out = out.drop(['metric'], axis=1)
                listout = out['value'].values.tolist()
                saverecord (listout,'listout_T.bin')            
            #overlapping mode
            if (mode==1):
                for x_test,y_test,predictions in  overlapping_reshape (predictions, x_test, y_test, model_name):
                    print ("Overlapping Mode")
            #Reshape
            for r_x_test,r_predictions,r_y_test,r_x_test_matrix,r_predictions_matrix in datareshape (model_name,x_test,y_test,predictions,mode):
                for residualmatrix,R_dist,mse_residual in  distance_matrix (r_x_test_matrix,r_predictions_matrix):
                    print ("yes")
        countmol = countmol+1                 
        del model 
        yield R_dist,r_y_test, residualmatrix,predictions,mse_residual
                
                
                
                