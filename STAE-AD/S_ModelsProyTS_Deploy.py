#!/usr/bin/python3 -u
# ==============================================================================
# Models
# ==============================================================================
import tensorflow as tf
from VarConvLSTM import AttenInputConvLSTM2D
from S_Parameters_ProyTS import parse_args

ini = tf.keras.initializers.glorot_uniform(seed=None)

def SpatioTemporalSimple (batchsize,time_steps,x,y,inChannel,stf):
    seq  = tf.keras.models.Sequential()
    if stf:
        inp_layer = tf.keras.layers.Input(batch_shape=(batchsize, time_steps,x,y,inChannel))
        print ("Stateful true")
    else:
        inp_layer = tf.keras.layers.Input(shape=(time_steps,x,y,inChannel))
        print ("Stateful False")
    #Spatial encoder    
    #conv 1
    conv1 = (tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, (3, 3),strides=2,padding='SAME', activation='tanh', kernel_initializer=ini)))(inp_layer)   
    #conv 2
    conv2 = (tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3),strides=2,padding='SAME',activation='tanh',kernel_initializer=ini))) (conv1)    
    #convolutional LSTM - Temporal Encoder/Decoder
    x = (tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(3,3),strides=(1, 1), padding='SAME',kernel_initializer=ini,activation='tanh',
                          dropout=0.3,return_sequences=True,stateful = stf)) (conv2)    
    #2
    x = (tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=(3,3),strides=(1, 1), padding='SAME',kernel_initializer=ini,activation='tanh',
                          dropout=0.3,return_sequences=True,stateful = stf)) (x)    
    #3
    x = (tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(3,3),strides=(1, 1), padding='SAME',kernel_initializer=ini,activation='tanh',
                          dropout=0.3,return_sequences=True,stateful = stf)) (x)            
    #Spatial decoder
    dec1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2DTranspose(128, 3, strides=2,padding='SAME',activation='tanh',
                        kernel_initializer=ini)) (x)
    #3
    dec2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2DTranspose(1, 3, strides=2,padding='SAME',activation='tanh', 
                        kernel_initializer=ini)) (dec1)    
    # create model    
    seq = tf.keras.models.Model(inputs=inp_layer, outputs=dec2)
    return (seq)    

def STAE_AD (batchsize, time_steps,x,y,inChannel,stf):
    seq  = tf.keras.models.Sequential()
    if stf:
        inp_layer = tf.keras.layers.Input(batch_shape=(batchsize, time_steps,x,y,inChannel))
    else:
        inp_layer = tf.keras.layers.Input(shape=(time_steps,x,y,inChannel))
    #Spatial encoder    
    #conv 1
    conv1 = (tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, (3, 3),strides=2,padding='SAME', kernel_initializer=ini)))(inp_layer); conv1 = (tf.keras.layers.TimeDistributed(tf.keras.layers.Activation('tanh')))(conv1)     
    #conv 2
    conv2 = (tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3),strides=2,padding='SAME',kernel_initializer=ini))) (conv1); conv2 = (tf.keras.layers.TimeDistributed(tf.keras.layers.Activation('tanh')))(conv2)  

    #convolutional LSTM - Temporal Encoder/Decoder
    x = (AttenInputConvLSTM2D(filters=64, kernel_size=(3,3),strides=(1, 1), padding='SAME',kernel_initializer=ini,activation='tanh',dropout=0.3,return_sequences=True,stateful = stf)) (conv2)    
    #2
    x = (AttenInputConvLSTM2D(filters=32, kernel_size=(3,3),strides=(1, 1), padding='SAME',kernel_initializer=ini,activation='tanh',dropout=0.3,return_sequences=True,stateful = stf)) (x)    
    #3
    x = (tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(3,3),strides=(1, 1), padding='SAME',kernel_initializer=ini,activation='tanh',dropout=0.3,return_sequences=True,stateful = stf)) (x)            
    #Spatial decoder
    dec1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2DTranspose(128, 3, strides=2,padding='SAME',kernel_initializer=ini)) (x) ; dec1  = (tf.keras.layers.TimeDistributed(tf.keras.layers.Activation('tanh')))(dec1 )     
    #3
    dec2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2DTranspose(1, 3, strides=2,padding='SAME',kernel_initializer=ini)) (dec1); dec2 = (tf.keras.layers.TimeDistributed(tf.keras.layers.Activation('tanh')))(dec2)     
    # create model    
    seq = tf.keras.models.Model(inputs=inp_layer, outputs=dec2)   
    return (seq)

def CNN_Noise (batchsize,x,y,inChannel,stf):
    seq  = tf.keras.models.Sequential()
    if stf:
        inp_layer = tf.keras.layers.Input(batch_shape=(batchsize,x,y,inChannel))
    else:
        inp_layer = tf.keras.layers.Input(shape=(x,y,inChannel))
    #Spatial encoder    
    #conv 1
    conv1 = tf.keras.layers.Conv2D(128, (3, 3),strides=2,padding='SAME', kernel_initializer=ini)(inp_layer)   
    conv1 = tf.keras.layers.Activation('tanh')(conv1)        
    #conv 2
    conv2 = tf.keras.layers.Conv2D(64, (3, 3),strides=2,padding='SAME',kernel_initializer=ini) (conv1)    
    conv2 = tf.keras.layers.Activation('tanh')(conv2)  
    #Spatial decoder
    dec1 = tf.keras.layers.Conv2DTranspose(128, 3, strides=2,padding='SAME',kernel_initializer=ini) (conv2)
    dec1  = tf.keras.layers.Activation('tanh')(dec1 )     
    #3
    dec2 = tf.keras.layers.Conv2DTranspose(1, 3, strides=2,padding='SAME',kernel_initializer=ini) (dec1)         
    dec2= tf.keras.layers.Activation('tanh')(dec2)     
    # create model    
    seq = tf.keras.models.Model(inputs=inp_layer, outputs=dec2)   
    return (seq)  

def load_model(name,stf):
    #Parameters models
    args = parse_args()
    time_steps = args.time_steps
    x = args.xzp
    y = args.yzp
    batch_size = args.batch_size
    inChannel = args.inChannel
    num_features = args.numft-1    
    #1
    if name=='SpatioTemporalSimple':
        return SpatioTemporalSimple (batch_size,time_steps,x,y,inChannel,stf)    
    #2
    elif name=='STAE_AD':
        return STAE_AD (batch_size,time_steps,x,y,inChannel,stf)    
    #3
    elif name=='CNN_Noise':
        return CNN_Noise (batch_size,x,y,inChannel,stf)      
    else:
        raise ValueError('Unknown model name %s was given' % name)