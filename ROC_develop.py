# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:55:18 2022

@author: Kacper
"""
from sklearn.metrics import roc_auc_score
import time
from numba import jit
import random


#%% Implementation written by Eleonore Schiltz, Haesler Lab

def checkSignificanceROC(responseVector,predictionVector):
        # check if it is sup or inf to random vector on 5th/95th percentile
    ROCV = np.zeros((len(responseVector),2))
    for ind in range(len(responseVector)):
        ROCV[ind,0] = roc_auc_score(predictionVector,responseVector[ind])
        randomValue = np.zeros(1000)
        for rand in range(1000):
            randomValue[rand]=roc_auc_score(np.random.permutation(predictionVector),responseVector[ind])
        if ROCV[ind,0]<np.percentile(randomValue,5):
            ROCV[ind,1] = -1 #inhibited
        elif ROCV[ind,0]>np.percentile(randomValue,95):
            ROCV[ind,1] = 1 #excited
        else:
            ROCV[ind,1] = 0 #non significant
    return ROCV

#%% Version based on simpler implementation within sci-kit learn
# Usually ~100 times faster than the default version 
# Significance testing is consistent for 97-100% of trials between the versions

@jit
def checkSignificanceROC_fast(responseVector,predictionVector):
        # check if it is sup or inf to random vector on 5th/95th percentile
        
    def auc(y_true, y_score):
        """Alternative implementation to check for correctness of
        `roc_auc_score`, copied from sci-kit learn metrics/_ranking.py"""
        pos_label = np.unique(y_true)[1]

        # Count the number of times positive samples are correctly ranked above
        # negative samples.
        pos = y_score[y_true == pos_label]
        neg = y_score[y_true != pos_label]
        diff_matrix = pos.reshape(1, -1) - neg.reshape(-1, 1)
        n_correct = np.sum(diff_matrix > 0)

        return n_correct / float(len(pos) * len(neg))
        
    ROCV = np.zeros((len(responseVector),2))
    for ind in range(len(responseVector)):
        ROCV[ind,0] = auc(predictionVector,responseVector[ind])
        randomValue = np.zeros(1000)
        for rand in range(1000):
            randomValue[rand]=auc(np.random.permutation(predictionVector),responseVector[ind])
        if ROCV[ind,0]<np.percentile(randomValue,5):
            ROCV[ind,1] = -1 #inhibited
        elif ROCV[ind,0]>np.percentile(randomValue,95):
            ROCV[ind,1] = 1 #excited
        else:
            ROCV[ind,1] = 0 #non significant
    return ROCV



#%% Version based on https://stephanosterburg.gitbook.io/scrapbook
# Usually ~300 times faster than the default version 
# Significance testing is USUALLY consistent for 97-100% of trials between the versions
# But for some neurons it drops to 3% or so (!)

@jit
def checkSignificanceROC_fast2(responseVector,predictionVector):
        # check if it is sup or inf to random vector on 5th/95th percentile
        
    # copied from: https://stephanosterburg.gitbook.io/scrapbook/data-science/ds-cheatsheets/machine-learning/fast-computation-of-auc-roc-score
    def fast_auc(y_true, y_prob):
        y_true = np.asarray(y_true)
        y_true = y_true[np.argsort(y_prob)]
        nfalse = 0
        auc = 0
        n = len(y_true)
        for i in range(n):
            y_i = y_true[i]
            nfalse += (1 - y_i)
            auc += y_i * nfalse
        auc /= (nfalse * (n - nfalse))
        return auc
        
    ROCV = np.zeros((len(responseVector),2))
    for ind in range(len(responseVector)):
        ROCV[ind,0] = fast_auc(predictionVector,responseVector[ind])
        randomValue = np.zeros(1000)
        for rand in range(1000):
            randomValue[rand]=fast_auc(np.random.permutation(predictionVector),responseVector[ind])
        if ROCV[ind,0]<np.percentile(randomValue,5):
            ROCV[ind,1] = -1 #inhibited
        elif ROCV[ind,0]>np.percentile(randomValue,95):
            ROCV[ind,1] = 1 #excited
        else:
            ROCV[ind,1] = 0 #non significant
    return ROCV


#%% Prepare data etc.
s = 4
nrn = 3
roc_win_sec = 3 # window size for ROC significance (in sec, how long before and after stimulus to analyze)
roc_sr = 10 # sampling rate to which downsample neural data, in Hz (otherwise analysis is very slow)
myseed = 10


roc_from = int(pre_event*roc_sr - roc_win_sec*roc_sr)
roc_to = int(pre_event*roc_sr + roc_win_sec*roc_sr)

prediction_len  = int((roc_to-roc_from)/2)
prediction = np.concatenate([np.zeros(prediction_len), np.ones(prediction_len)])

response = all_fr[s][nrn][:,::int(ifr_sr / roc_sr)] # select neuron and downsample
response = response[:,roc_from:roc_to] # select window for analysis


#%% Test versions

#% Run default
random.seed(myseed)
start = time.time()
print("Sci-kit default version")

a = checkSignificanceROC(response, prediction)

end = time.time()
print(str(end - start) + " sec")


#% Run faster
random.seed(myseed)
start = time.time()
print("Sci-kit fast verion")

a2 = checkSignificanceROC_fast(response, prediction)

end = time.time()
print(str(end - start) + " sec")

#% Run scrapbook
random.seed(myseed)
start = time.time()
print("Scrapbook fast verion")

a3 = checkSignificanceROC_fast2(response, prediction)

end = time.time()
print(str(end - start) + " sec")

#% Compare consistency with default
print("sci-kit consistency: " + str(np.sum(a[:,1] == a2[:,1])/response.shape[0]))
print("Scrapbokk consistency: " + str(np.sum(a[:,1] == a3[:,1])/response.shape[0]))
