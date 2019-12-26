#!/usr/bin/python3 -u
# ==============================================================================
# Evaluation
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from S_Parameters_ProyTS import parse_args
import tensorflow as tf
from keras import backend as K
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import (precision_recall_curve, auc,
                             classification_report)
from sklearn.metrics import roc_auc_score
from S_Define_Th_Distance_ProyTS import extract_threshold, extract_score
import random
import operator
from matplotlib import pyplot 
import matplotlib.ticker as mtick

K.clear_session() 
tf.keras.backend.clear_session()
keras=tf.contrib.keras
np.random.seed(42)
sns.set(style='whitegrid', palette='muted', font_scale=1.4)
RANDOM_SEED = 42
LABELS = ["Normal", "Attacks"]
            
def analytic_reconstruction_error (error_df,model_name,mode,thf):
    #Reconstruction error without attacks
    fig = plt.figure()
    ax = fig.add_subplot(111)
    normal_error_df = error_df[(error_df['true_class']== 0) & (error_df['reconstruction_error'] < thf)]
    _ = ax.hist(normal_error_df.reconstruction_error.values, bins=10)
    #1    
    plt.title('Reconstruction error without Attacks')
    plt.ylabel('quantity')
    plt.xlabel('reconstruction_error')
    plt.savefig('1'+'_model_name_'+str(model_name)+'_mode_'+str(mode),bbox_inches="tight")
    plt.clf() 
    #2    
    #Reconstruction error with attacks
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fraud_error_df = error_df[error_df['true_class'] == 1]
    _ = ax.hist(fraud_error_df.reconstruction_error.values, bins=10)
    plt.title('Reconstruction error with Attacks')
    plt.ylabel('quantity')
    plt.xlabel('reconstruction_error')
    plt.savefig('2'+'_model_name_'+str(model_name)+'_mode_'+str(mode),bbox_inches="tight")
    plt.clf()     
   
def analytic_metric (error_df,model_name,mode):
    sns.set_style("ticks")
    # calculate AUC
    aucfinal = roc_auc_score(error_df.true_class, error_df.reconstruction_error)
    print('AUC: %.3f' % aucfinal)    
    pyplot.title('Receiver Operating Characteristic')    
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)
    # plot no skill
    pyplot.plot([0, 1], [0, 1],color='navy',linestyle='--')
    # plot the roc curve for the model
    pyplot.plot(fpr, tpr,color='darkorange', marker='.',label='ST-ED  ('+r'$\ell$'+'=150)')
    pyplot.ylabel('True Positive Rate')
    pyplot.xlabel('False Positive Rate')
    pyplot.legend(loc='best', fontsize = 14);pyplot.legend(loc=4)   
    # show the plot
    pyplot.savefig('3'+'_model_name_'+str(model_name)+'_mode_'+str(mode),bbox_inches="tight")
    pyplot.clf()     
    pyplot.show()    
    #4
    ##Precision vs Recall
    precision, recall, th = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)
    plt.plot(recall, precision, 'b', label='Precision-Recall curve')
    plt.title('Recall vs Precision')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig('4'+'_model_name_'+str(model_name)+'_mode_'+str(mode),bbox_inches="tight")
    plt.clf() 
    plt.show()
    #5    
    ##Precision for different threshold values
    plt.plot(th, precision[1:], 'b', label='Threshold-Precision curve')
    plt.title('Precision for different threshold values')
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.savefig('5'+'_model_name_'+str(model_name)+'_mode_'+str(mode))
    plt.clf() 
    plt.show()
    #6
    ##Recall for different threshold values
    plt.plot(th, recall[1:], 'b', label='Threshold-Recall curve')
    plt.title('Recall for different threshold values')
    plt.xlabel('Reconstruction error')
    plt.ylabel('Recall')
    plt.savefig('6'+'_model_name_'+str(model_name)+'_mode_'+str(mode),bbox_inches="tight")
    plt.clf() 
    plt.show()
    # calculate precision-recall AUC
    auprecisionrecall = auc(recall, precision)
    # calculate average precision score
    print('auprecisionrecall',auprecisionrecall)    
	
def point_error_segment (error_df_segment,model_name,mode,thf,model_name_c):      
    sns.set_style("ticks")
    threshold = thf
    groups = error_df_segment.groupby('true_class')
    fig, ax = plt.subplots()   
    for name, group in groups:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0));ax.xaxis.major.formatter._useMathText = True;ax.plot((group.index/1), group.reconstruction_error, marker='o', ms=2.5, linestyle='',label= "Attack" if name == 1 else "Normal", color="black" if name ==1 else "blue")
    ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, linestyle='--', label='Threshold')
    #ax.legend()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))     
    plt.title(model_name_c)
    plt.ylabel("Reconstruction error")
    plt.xlabel("Test time")
    plt.savefig('Segment'+'_model_name_'+str(model_name)+'_mode_'+str(mode),bbox_inches="tight")
    plt.clf() 
    plt.show()    
    
def analytic_error (error_df,model_name,mode,thf,model_name_c):
    #threshold
    threshold = thf
    groups = error_df.groupby('true_class')
    fig, ax = plt.subplots()   
    for name, group in groups:#
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0));ax.xaxis.major.formatter._useMathText = True;ax.plot((group.index/1), group.reconstruction_error, marker='o', ms=3.5, linestyle='',label= "Attack" if name == 1 else "Normal",color="black" if name ==1 else "blue")
    ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, linestyle='--', label='Threshold')
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(model_name_c)
    plt.ylabel("Reconstruction error")
    plt.xlabel("Test time")
    plt.savefig('7'+'_model_name_'+str(model_name)+'_mode_'+str(mode),bbox_inches="tight")
    plt.clf() 
    plt.show()    
    y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
    conf_matrix = confusion_matrix(error_df.true_class, y_pred)
    plt.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title(model_name_c)
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.savefig('8'+'_model_name_'+str(model_name)+'_mode_'+str(mode))
    plt.clf() 
    plt.show()
    yield (y_pred)

def analytic_result(case,mode,h5model,model_name,data_pathXV2,data_pathYV2,data_pathXE1,data_pathYE1,custom_object,data_pathXE2,data_pathYE2,model_name_c):
    th = extract_threshold (case,mode,h5model,model_name,data_pathXV2,data_pathYV2,data_pathXE1,data_pathYE1,custom_object)    
    print ("Final Threshold")
    print (th)
    for R_dist_testmodeling,r_y_testmodeling,predictions_testmodeling,mse in extract_score (case,mode,data_pathXE2,data_pathYE2,h5model,model_name,custom_object,'yes'):
        error_df = pd.DataFrame({'reconstruction_error': R_dist_testmodeling,'true_class': r_y_testmodeling});error_df_seg = error_df.head(3000);point_error_segment (error_df_seg,model_name,mode,th,model_name_c)
        for y_pred in analytic_error (error_df,model_name,mode,th,model_name_c):        
            print(confusion_matrix(r_y_testmodeling, y_pred))
            print(classification_report(r_y_testmodeling, y_pred))
            clsf_report = pd.DataFrame(classification_report(r_y_testmodeling, y_pred, output_dict=True)).transpose()
            clsf_report.to_csv('TESTclassification_report'+'_model_name_'+str(model_name)+'_mode_'+str(mode)+'.csv', index= True)        
            #accurancy
            accuracy = accuracy_score(r_y_testmodeling, y_pred)
            print('Accuracy: %f' % accuracy)
            #other charts
            analytic_metric (error_df,model_name,mode)            
        analytic_reconstruction_error (error_df,model_name,mode,th)
    yield (error_df,y_pred)
        
if __name__ == '__main__':
    #hyperparameters
    args = parse_args()    
    #val 2 
    data_pathXV2 = args.data_pathXV2 # validation 2
    data_pathYV2 = args.data_pathYV2 # validation 2
    #evaluation 1 
    data_pathXE1 = args.data_pathXE1 # evaluation 3
    data_pathYE1 = args.data_pathYE1 # evaluation 3 
    #test model     
    data_pathXE2 = args.data_pathXE2 # evaluation 3
    data_pathYE2 = args.data_pathYE2 # evaluation 3 
    #model name 
    model_name = args.model_name 
    case = args.case;
	model_name_c = args.model_name_c
    mode = args.mode
    h5model = args.modeleval
    custom_object = args.custom_objects
    #######################################
    ##case 1, mode 2, stateful=False      #
    #case 2, mode 2, stateful=True/False  #
    #case 3, mode 1, statful=True/False   #
    #######################################    
    for error_df,y_pred in analytic_result(case,mode,h5model,model_name,data_pathXV2,data_pathYV2,data_pathXE1,data_pathYE1,custom_object,data_pathXE2,data_pathYE2,model_name_c):
        print ("yes")
    
