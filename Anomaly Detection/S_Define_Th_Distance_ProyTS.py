#!/usr/bin/python3 -u
# ==============================================================================
# S_Define_Distance
# ==============================================================================
from S_DataProcessingEvaluation_ProyTS import process_data
from S_ProcessingErrorMatrix import extract_error
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import random
import operator

#extract_error (data,mode,h5model,model_names,custom_object)

def extract_score (case,mode,data_pathX,data_pathY,h5model,model_names,custom_object,ban):
    for v2 in process_data (case,mode,data_pathX,data_pathY):
        print ("SCORE")
        for R_dist,r_y_test, m,predictions,mse_residual in  extract_error(v2,mode,h5model,model_names,custom_object,ban):
            print ("SCORE")
            yield (R_dist,r_y_test,predictions,mse_residual)
            
def selectThreshHold(ytestth,tau,limsup,liminf,score_df_abnormal):
    F1 = 0
    bestF1 = 0
    #bestF2 = 0
    besttau = 0
    tauaux = tau     
    epsVec = epsVec = np.arange(liminf, limsup, 0.001)#0.001
    noe = len(epsVec)    
    for eps in range(noe):
        tauaux = (tauaux*epsVec[eps])
        ypred = [1 if e > tauaux else 0 for e in score_df_abnormal.reconstruction_error.values]##1 attack ##0 non attack 
        prec, rec = 0,0
        tp,fp,fn = 0,0,0
        try:
            for i in range(np.size(ytestth,0)):
                if ypred[i] == 1 and ytestth[i] == 1: ##decting attack but it is not attack
                    tp+=1
                elif ypred[i] == 1 and ytestth[i] == 0:##decting attack but it is real behaviour
                    fp+=1
                elif ypred[i] == 0 and ytestth[i] == 1:##non attack but dectecting attack
                    fn+=1
            prec = tp/(tp + fp)
            rec = tp/(tp + fn)
            F1 = (2*prec*rec)/(prec + rec)
            if F1 > bestF1:
                bestF1 = F1
                besttau = tauaux
                c = epsVec[eps]
        except ZeroDivisionError:
            print('Warning dividing by zero!!')               
    return bestF1, besttau,c

#extracting threshold
def extract_threshold (case,mode,h5model,model_name,data_pathXV2,data_pathYV2,data_pathXE1,data_pathYE1,custom_object):
    print ("extracting data")
    #validatiom
    for R_dist_val,r_y_val,predictions_val,mse_residual_val in extract_score (case,mode,data_pathXV2,data_pathYV2,h5model,model_name,custom_object,'no'):
        print ('yes validation threshold')
    #test data to extract threshold
    for R_dist_test_th,r_y_test_th,predictions_test_th,mse_residual_test_th in extract_score (case,mode,data_pathXE1,data_pathYE1,h5model,model_name,custom_object,'no'):#yes
        print ('yes evaluation threshold')        
    #threshold # max point in the validation data
    threshold = max (R_dist_val)
    print ("Max tau in validation data")
    print (threshold)    
    score_df_abnormal_th2 = pd.DataFrame({'reconstruction_error': R_dist_test_th,'true_class': r_y_test_th})
    #first confusion matrix
    y_pred = [1 if e > threshold else 0 for e in score_df_abnormal_th2.reconstruction_error.values]
    bestF1, besttau,c = selectThreshHold(r_y_test_th,threshold,2,1,score_df_abnormal_th2)
    print ("best hyperparameter")
    print ("best F1")
    print (bestF1)
    print ("best Tau")
    print (besttau)
    print ("constant")
    print (c)
    print(confusion_matrix(r_y_test_th, y_pred))
    print(classification_report(r_y_test_th, y_pred))    
    clsf_report = pd.DataFrame(classification_report(r_y_test_th, y_pred, output_dict=True)).transpose()
    clsf_report.to_csv('Validation_classification_report'+'_model_name_'+str(model_name)+'_mode_'+str(mode)+'.csv', index= True)        
    return (besttau)    

    
