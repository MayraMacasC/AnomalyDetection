#!/usr/bin/python3 -u
# ==============================================================================
# Parameter Settings
# ==============================================================================
import argparse
import os
from VarConvLSTM1 import AttenInputConvLSTM2D
import tensorflow as tf

#directory
curdir = os.path.dirname(os.path.abspath(__file__))

def root_mean_squared_error(actual, predicted):
  return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(actual - predicted), axis=1))
  
def parse_args():
    
    parser = argparse.ArgumentParser(description='Deep Learning Models for Anomaly Detection')    
    parser.add_argument('--data_pathX', default='XT.csv', type=str, help='path to dataset')
    parser.add_argument('--data_pathY', default='YT.csv', type=str, help='path to dataset')
    ### Validation 2 -- score, normal to extracting threshold
    parser.add_argument('--data_pathXV2', default='hX_val150.csv', type=str, help='path to dataset')
    parser.add_argument('--data_pathYV2', default='hY_val150.csv', type=str, help='path to dataset')
    ### Abnormal 1  -- F1 threshold
    parser.add_argument('--data_pathXE1', default='hX_test150.csv', type=str, help='path to dataset')
    parser.add_argument('--data_pathYE1', default='hY_test150.csv', type=str, help='path to dataset')
    ### Abnormal 2 -- testing model
    parser.add_argument('--data_pathXE2', default='FX_Test150.csv', type=str, help='path to dataset')
    parser.add_argument('--data_pathYE2', default='FY_Test150.csv', type=str, help='path to dataset')
    parser.add_argument('--normal_label', default=8, type=int, help='label defined as anomality')
    parser.add_argument('--rate_normal_train', default=0.82, type=float, help='rate of normal data to use in training')
    parser.add_argument('--rate_anomaly_test', default=0.1, type=float,
                        help='rate of abnormal data versus normal data. default is 10:1')
    parser.add_argument('--test_rep_count', default=1, type=int, help='count of repeat of test for a trained model')
    parser.add_argument('--numft', default=2705, type=float, help='number of features')
    parser.add_argument('--RANDOM_SEED', default=42, type=float, help='ramdom seed training data')
    #declare hyper parameters for the models
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--n_classes', default=2, type=int, help='number of classes (Attack and abnormal)')
    parser.add_argument('--display_step', default=100, type=int, help='display_step')
    parser.add_argument('--training_cycles', default=2, type=int, help='training cycles')
    parser.add_argument('--time_steps', default=4, type=int, help='time_steps')
    parser.add_argument('--hidden_units', default=50, type=int, help='hidden units')
    parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
    parser.add_argument('--inChannel', default=1, type=int, help='inChannel')
    parser.add_argument('--x', default=51, type=int, help='x')
    parser.add_argument('--y', default=51, type=int, help='y')
    parser.add_argument('--xzp', default=52, type=int, help='xzp')
    parser.add_argument('--yzp', default=52, type=int, help='yzp')    
    parser.add_argument('--rate', default=0.2, type=float, help='rate')
    parser.add_argument('--nb_epoch', default=500, type=int, help='number of epoch')
    parser.add_argument('--test_size', default=0.2, type=float, help='test size')
    parser.add_argument('--optimizer', choices=['adam','sgd','adagrad'], default='adam')
    parser.add_argument('--loss', choices=['mean_squared_error', 'binary_crossentropy'], default='mean_squared_error')
    parser.add_argument('--result', default=os.path.join(curdir, 'result.png'))
    parser.add_argument('--n_outputs', default=1, type=int, help='number of outputs in this case we have just one')
    parser.add_argument('--verbose', choices=[0,1,2], default=2)
    parser.add_argument('--metrics', choices=['accuracy','mse','mae', 'mape'], default=['accuracy','mse','mae', 'mape',root_mean_squared_error]);parser.add_argument('--custom_objects', choices={'AttenIConvLSTM2D': AttenIConvLSTM2D,'root_mean_squared_error':root_mean_squared_error}, default={'AttenIConvLSTM2D': AttenIConvLSTM2D,'root_mean_squared_error':root_mean_squared_error})
    parser.add_argument('--win', choices=[90,120,150], default=120)
    parser.add_argument('--test_percent',default=0.2, type=float, help='Test percent')
    parser.add_argument('--test_percent_eval',default=0, type=float, help='Test percent')     
    parser.add_argument('--case', default=2, type=int, help='case')  
    parser.add_argument('--mode', default=2, type=int, help='mode')
    parser.add_argument('--stateful', action='store_true', default=False, help='stateful')
    parser.add_argument('--modeleval', default='ModelWithNoise_2_2_stateful_False.h5', type=str, help='name model evaluation')
    parser.add_argument('--constant',default=1.0, type=float, help='Constant Value Anomaly Detection')  
    parser.add_argument('--eval', action='store_true', default=True, help='Mode Evaluation')  
	parser.add_argument('--eval', action='store_true', default=True, help='Mode Evaluation')  
    ##Evaluation
    parser.add_argument('--beta',default=1, type=float, help='BetaEval')      
    parser.add_argument('--scala',default=20, type=int, help='scala') 
    parser.add_argument('--weigth', default='ModelWithNoise_2_2_stateful_False.h5', type=str, help='name model evaluation')
    args = parser.parse_args()
    return args

