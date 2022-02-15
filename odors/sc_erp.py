# -*- coding: utf-8 -*-
"""
Extract evoked potentials from raw Neuropixels data.
The file paths assume it's run on syskrunch.

Created on Tue Aug 17 16:34:14 2021
@author: kkondrakiewicz
"""
#%% Imports
import sys
# Add directory with ephys module
sys.path.append(r'D:\haeslerlab\code\user\kacper\EphysAnalysis')
sys.path.append(r'D:\haeslerlab\code\user\kacper\Sniff')

import ephys as ep
import sniff_tools as st
from readSGLX import readSGLX as sglx

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from pathlib import Path
import time
import copy

#%% Set parameters for analysis
pre_event = 3
post_event = 6
ifr_sr = 50 # number of data points / sec for instantenous firing rate
    
#%% Specify files and paths
sess_ids = ['211207_KK006', '211207_KK007', '211208_KK006', '211208_KK007', \
           '211209_KK006', '211209_KK007', '211210_KK006', '211210_KK007']
    
sniff_dir = r'E:\local\users\kacper\conc\processed_data\ephys'

spks_dirs = []
for ses in sess_ids:
    tmp = "\\\\169.254.103.43\\haeslerlab\\raw_data\\ephys\\"+ses+"\\"+ses+"_g0\\"+ses+"_g0_imec0" 
    spks_dirs.append(tmp)
    
#%% Import odor trial data
sniffs = st.import_sniff_mat(sniff_dir)

#%% Get some basic variables
nses = len(sniffs)
ntrials = sniffs[0]['trial_idx'].size
npres = max(sniffs[0]['trial_occur'])

#%% Import some raw data for EPs

nChans = 384
chanList = np.arange(0, 384)
chan = 100

pre_event = 1
post_event = 4


all_ERP = []
erpAv = []

for s in range(nses):
    tmp = Path(spks_dirs[s] + "\\" + sess_ids[s] + '_g0_t0.imec0.ap.bin')
    meta = sglx.readMeta(tmp)
    sRate = sglx.SampRate(meta) 

    pre_dp = int(pre_event*sRate)
    post_dp = int(post_event*sRate)
    win_dp = int((pre_dp + post_dp)/30 + 1)
    t_vec = np.linspace(-pre_event, post_event, win_dp)

    rawData = sglx.makeMemMapRaw(tmp, meta)

    erpData = np.zeros([nChans, win_dp, ntrials], dtype='f')
    
    #plt.figure()
    TTL = sniffs[s]['ephys_onsets']
    
    start = time.time()
    print('session', s+1, 'from', nses)
    
    for trial in range(len(TTL)):
        
        event = int(round(TTL[trial]*sRate))
        selectData = rawData[chanList, event-pre_dp : event+post_dp +1 :30]

        # apply gain correction and convert to mV
        convData = 1e3*sglx.GainCorrectIM(selectData, chanList, meta)
        
        # store the relevant data fragments in the 3 dimensional array
        erpData[:,:,trial] = convData
        #erpData[:,:,trial] = selectData
        
        #plt.plot(t_vec, convData[chan,:])
        
    end = time.time()
    print('Elapsed time:', end - start)

    erpAv.append(np.mean(erpData, 2))
    all_ERP.append(erpData)

    #plt.ylabel('Voltage [mV]')
    #plt.xlabel('Time [sec]')

#%%
all_lp = copy.deepcopy(all_ERP)
lpAv = copy.deepcopy(erpAv)
 
sos = signal.butter(1, 300/1000, 'lowpass', analog = True, output = 'sos')

for s in range(nses):
    print(s)
    for chan in chanList:
        for tr in range(ntrials):
            tmp = signal.sosfilt(sos, all_ERP[s][chan, :, tr]) 
            all_lp[s][chan, :, tr] = tmp
            
        lpAv[s][chan,:] = np.mean(np.squeeze(all_lp[s][chan, :, :]), 1)


#%% Plot mean ERP on all channels (ordered by depth)
y_scale = 100
chan = 100

for s in range(nses):
    plt.figure()
    for chan in chanList:
        plt.plot(t_vec, (erpAv[s][chan, :] - np.mean(erpAv[s][chan,:])) *y_scale + chan, linewidth = 0.2)
    plt.xlabel('Time from event [sec]')
    plt.ylabel('Channel no. (0 = tip)')
    
#%% Plot grand average (mean from all channels)
grav = []
sal = [0, 1, 4, 5]

for s in range(nses): 
    bsln = np.mean(erpAv[s][:,0:1000], 1, keepdims = True)
    tmp = erpAv[s][:,:] - bsln
    grav.append(np.mean(tmp, 0))
    
for ii in sal:
    plt.figure()
    plt.plot(t_vec, grav[ii])
    plt.plot(t_vec, grav[ii+2])

#%% Plot mean ERP on all channels (ordered by depth)
y_scale = 100
chan = 100

for s in range(nses):
    plt.figure()
    for chan in chanList:
        plt.plot(t_vec, (lpAv[s][chan, :] - np.mean(lpAv[s][chan,:])) *y_scale + chan, linewidth = 0.2)
    plt.xlabel('Time from event [sec]')
    plt.ylabel('Channel no. (0 = tip)')
    
#%% Plot grand average (mean from all channels)
grav = []
sal = [0, 1, 4, 5]

for s in range(nses): 
    bsln = np.mean(lpAv[s][:,0:1000], 1, keepdims = True)
    tmp = lpAv[s][:,:] - bsln
    grav.append(np.mean(tmp, 0))
    
for ii in sal:
    plt.figure()
    plt.plot(t_vec, grav[ii])
    plt.plot(t_vec, grav[ii+2])
