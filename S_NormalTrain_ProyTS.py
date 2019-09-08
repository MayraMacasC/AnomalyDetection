#!/usr/bin/python3 -u
# ==============================================================================
# Training
# ==============================================================================
from S_ModelsProyTS_Deploy import load_model
import numpy as np
import matplotlib.pyplot as plt
from S_Parameters_ProyTS import parse_args
from S_DataProcessingProyTS import process_data
import tensorflow as tf
from keras import backend as K
from clr_callback import CyclicLR
import time
import pickle

K.clear_session() 
tf.keras.backend.clear_session()
keras=tf.contrib.keras

np.random.seed(42)

def customLoss(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true))) 
	
###time computation
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        
#save record
def saverecord (listinput,name):
    #save the list in output.bin file
    with open(name, "wb") as output:
        pickle.dump(listinput, output)
		
losses = []
###Losses
def handleLoss(loss):
        global losses
        losses+=[loss]
        print(loss)
        
class LossHistory( tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        handleLoss(logs.get('loss'))

#save images 
def savetrainingprocess (history,model_name,case,mode,stateful):
        #Evaluation --label=model_name
        #loss 
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title(model_name + 'loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right');        
        plt.savefig(model_name+str(case)+'_'+str(mode)+'_stateful_'+str(stateful) + ' loss')
        plt.clf() 
        #accurancy
        plt.plot(history['acc'])
        plt.plot(history['val_acc'])
        plt.title(model_name +' accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(model_name+str(case)+'_'+str(mode)+'_stateful_'+str(stateful) + 'accurancy')
        plt.clf() 
        #mse
        plt.plot(history['mean_squared_error'])
        plt.plot(history['val_mean_squared_error'])
        plt.title(model_name +' MSE')
        plt.ylabel('MSE')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(model_name+str(case)+'_'+str(mode)+'_stateful_'+str(stateful) + ' MSE')
        plt.clf()        
        #mae, RMSE, MAPE
        plt.plot(history['mean_absolute_error'])
        plt.plot(history['val_mean_absolute_error'])
        plt.title(model_name+str(case)+'_'+str(mode)+'_stateful_'+str(stateful) +' MAE')
        plt.ylabel('MAE')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(model_name+str(case)+'_'+str(mode)+'_stateful_'+str(stateful) + ' MAE')
        plt.clf();plt.plot(history['root_mean_squared_error']);plt.plot(history['val_root_mean_squared_error']);plt.title(model_name +' RMSE');plt.ylabel('RMSE');plt.xlabel('epoch');plt.legend(['train', 'validation'], loc='upper left');plt.savefig(model_name+str(case)+'_'+str(mode)+'_stateful_'+str(stateful) + ' RMSE');plt.clf();plt.plot(history['mean_absolute_percentage_error']);plt.plot(history['val_mean_absolute_percentage_error']);plt.title(model_name+str(case)+'_'+str(mode)+'_stateful_'+str(stateful) +' MAPE');plt.ylabel('MAPE');plt.xlabel('epoch');plt.legend(['train', 'validation'], loc='upper left');plt.savefig(model_name+str(case)+'_'+str(mode)+'_stateful_'+str(stateful) + ' MAPE');plt.clf()            

##adding noise
def add_noise (x_train_normal,x_val_nomal):
    #Adding noise 
    noise_factor = 0.5
    x_train_normal_noise = x_train_normal + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_normal.shape)
    x_train_normal_noise = np.clip(x_train_normal, 0., 1.)
    x_val_nomal_noise = x_val_nomal + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_val_nomal.shape)
    x_val_nomal_noise = np.clip(x_val_nomal, 0., 1.)
    yield (x_train_normal_noise,x_val_nomal_noise)

#data to train 
def datatotrain (model_name,train_data_model01,train_data_model02,train_data_model04):
        if model_name in ['SpatioTemporalwithAttSimple',
                   'SpatioTemporalwithAttSimpleReg',
                   'SpatioTemporalwithAttReg',
                   'SpatioTemporalwithAttBacth',
                   'SpatioTemporalwithAttBacthReg',
                   'SpatioTemporalwithSpatialAttBacth',
                   'SpatioTemporalWithSpatialAttention',
                   'SpatioTemporalwitouthAtt',
                   'SpatioTemporalwithSpatialAtt',
                   'ModelWithNoise','ModelNoiseT',
                   'SpatioTemporalwithAttSimpleModel2',
                   'SpatioTemporalwithAttSimpleModel2withNoise',
                   'SpatioTemporalwithAttSimpleRegModel2']:
            #train data 
            x_train_normal = train_data_model01.o_Xtrain
            #val data
            x_val_nomal = train_data_model01.o_Xval
            
        elif model_name == 'LSTM_AE':
            #train data
            x_train_normal = train_data_model02.o_Xtrain_r
            #val data
            x_val_nomal = train_data_model02.o_Xval_r

        elif model_name == 'CNN_Noise':
            #train data
            x_train_normal = train_data_model04.o_Xtrain_CNNr
            #val data
            x_val_nomal = train_data_model04.o_Xval_CNNr     
      
        else:
            raise ValueError('Unknown model_name %s was given' % model_name)
        yield (x_train_normal,x_val_nomal)
    
def main(args,data,case,mode,stateful):    
    ext = '.h5'
    model_names = ['ModelWithNoise','CNN_Noise','SpatioTemporalwithAttSimple','SpatioTemporalWithSpatialAttention']              
    timetraining = []
    countmol = 0    
    # set parameters
    #train
    train_data_model01 = data.model1
    train_data_model02 = data.model2
    train_data_model04 = data.model4
    ##models    
    for model_name in model_names:
        #model
		h5model = [model_name+str(case)+'_'+str(mode)+'_stateful_'+str(stateful)+ext]    
        model = load_model(model_name,stateful)
        # reshape input data according to the model's input tensor
        for x_train_normal,x_val_nomal in datatotrain (model_name,train_data_model01,train_data_model02,train_data_model04):
            # condition to implement a model with and without noise
            if (model_name == 'ModelWithNoise1') or (model_name == 'SpatioTemporalwithAttSimpleModel2withNoise') or (model_name == 'CNN_Noise1'):
                print ("NOISE1")
                for x_train_normal_ns, x_val_nomal_ns in add_noise (x_train_normal,x_val_nomal):
                    x_train_normal_n = x_train_normal
                    x_val_nomal_n = x_val_nomal 
            else: 
                print ("WITHOUT NOISE")
                x_train_normal_ns = x_train_normal
                x_train_normal_n = x_train_normal
                x_val_nomal_ns = x_val_nomal
                x_val_nomal_n = x_val_nomal       
                # compile model
            model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=1, amsgrad=True), loss=args.loss, metrics=args.metrics)#customLoss,args.loss, amsgrad=False
            model.summary()
            # train on only normal training data
            nb_epoch = args.nb_epoch
            batch_size = args.batch_size
            # time
            time_callback = TimeHistory()     
            # learning 
            clr_triangular = CyclicLR(mode='triangular')    
            #simple early stopping        
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',patience=2,verbose=1)#val_loss
            mc = tf.keras.callbacks.ModelCheckpoint(h5model[countmol], monitor='val_mean_absolute_percentage_error', mode='min', verbose=1,save_best_only=True) #mean_squared_error,val_root_mean_squared_error,val_mean_squared_error,val_mean_absolute_percentage_error              
            time_callback = TimeHistory()
            hytories = []
            if stateful:
                history = model.fit(x_train_normal_ns, x_train_normal_n,
                                    epochs=nb_epoch,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    validation_data=(x_val_nomal_ns, x_val_nomal_n),
                                    verbose=1,
                                    callbacks=[es, mc,clr_triangular,time_callback,LossHistory()]).history
                model.reset_states()
            else:   
                history = model.fit(x_train_normal_ns, x_train_normal_n,
                                    epochs=nb_epoch,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    validation_data=(x_val_nomal_ns, x_val_nomal_n),
                                    verbose=1,
                                    callbacks=[es, mc,clr_triangular,time_callback,LossHistory()]
                                    ).history            
            #save model
            model.save(filepath=h5model[countmol])
            times = time_callback.times        
            timetraining.append(times)        
            hytories.append(history)
            saverecord (hytories,'hytories.bin')
            saverecord (timetraining,'timetraining.bin')
            savetrainingprocess (history,model_name,case,mode,stateful)   
            countmol = countmol+1                
            del model 
        
if __name__ == '__main__':
    args = parse_args()
    case = args.case
    mode = args.mode 
    stateful = args.stateful
    #######################################
    ##case 1, mode 2, stateful=False      #
    #case 2, mode 2, stateful=True/False  #
    #case 3, mode 1, statful=True/False   #
    #######################################
    for data in  process_data(case,mode):
        main(args,data,case,mode,stateful)