# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 08:10:21 2021

@author: kkondrakiewicz
"""
#%% Import packages
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
os.chdir(r"C:\Users\kkondrakiewicz\Documents\Python Scripts")
import readSGLX as sglx

#%% Specify variables - for paths use raw strings with forward slashes (in Windows)
data_path = 'C:/Neuropixel_data/SOV_19'
CatGT_path = 'C:/CatGT'
TPrime_path = 'C:/TPrime'

run_name = 'test_sync2' # main file name
TTL_chan = 2 # number of TTL event channel (digital port)
sync_chan = 0 # number of the SYN channel 

# window size for plotting ERPs
pre_sec = 0.1
post_sec = 0.5
chan = 100 # select a single channel for plots

nChans = 384
chanOrder = np.hstack([np.arange(0,nChans,2), np.arange(1,nChans,2)]) # assume long column channel order
chanList = np.arange(0, 384)

# Construct paths - assumes there was only 1 run (g0)!
ap_path = data_path + '/' + run_name + '_g0/' + run_name + '_g0_imec0/'
ni_path = data_path + '/' + run_name + '_g0/'
binFullPath = Path(ap_path + run_name + '_g0_tcat.imec0.ap.bin')

#%% Functions for running CatGT and TPrime
def runCatGT(run_name, CatGT_path = 'D:/ProgramySzemrane/CatGT'):
    """
    Runs CatGT directly from Python: 
        a) creates .bat files for action potentials and National Instrument streams in the CatGT directory
        b) runs them and reports when the output files are ready
    Currently supports only 1 TTL event channel and assumes there is 1 run (g=0)
    """
    
    command_ap = 'CatGT -dir=' + data_path + ' -run=' + run_name + ' -g=0 -t=0,0 -ap -prb=0 -prb_fld -SY=0,384,6,500'
    command_ni = 'CatGT -dir=' + data_path + ' -run=' + run_name + ' -g=0 -t=0,0 -ni -prb=0 -XD=8,' + str(TTL_chan) + ',0' + '-XD=8,' + str(sync_chan) + ',0'

    bat_ap_path = CatGT_path + '/' + run_name + '_ap' + '.bat'
    bat_ni_path = CatGT_path + '/' + run_name + '_ni' + '.bat'

    bat_ap = open(bat_ap_path,'w+')
    bat_ap.write(command_ap)
    bat_ap.close()
    
    bat_ni = open(bat_ni_path,'w+')
    bat_ni.write(command_ni)
    bat_ni.close()
    
    print('CatGT - extracting sync timestamps (+ optional filtering)')
    subprocess.call([bat_ap_path], cwd= CatGT_path)
    print('CatGT - extracting TTL timestamps')
    subprocess.call([bat_ni_path], cwd= CatGT_path)
    
def runTPrime(run_name, TTL_chan, TPrime_path = 'D:/ProgramySzemrane/TPrime'):
    """
    Runs TPrime directly from Python: 
        a) creates .bat file in the TPrime directory
        b) runs it and returns a single .txt file with corrected timestamps of the selected TTL_channel
    Currently supports only 1 TTL event channel and assumes there is 1 run (g=0)
    """
    
    sync_ending = '_g0_tcat.imec0.SY_384_6_500.txt'
    from_ending = '_g0_tcat.nidq.XD_8_' + str(sync_chan) + '_0.txt'
    ttl_ending = '_g0_tcat.nidq.XD_8_' + str(TTL_chan) + '_0.txt'
    ttl_out = 'TTL_' + str(TTL_chan) + '.txt'
    
    command_tp = 'TPrime -syncperiod=1.0 -tostream=' + \
        ap_path + run_name + sync_ending + ' ^ ' + \
        '-fromstream=2,' + ni_path  + run_name + from_ending + ' ^ ' + \
        '-events=2,' + ni_path + run_name + ttl_ending + ',' + \
        ni_path + ttl_out
        
    bat_tp_path = TPrime_path + '/' + run_name + '_tp' + '.bat'
    bat_tp = open(bat_tp_path,'w+')
    bat_tp.write(command_tp)
    bat_tp.close()
    
    print('TPrime - correcting TTL timestamps')
    subprocess.call([bat_tp_path], cwd= TPrime_path)

#%% Run both programs to extract correct event timestamps
runCatGT(run_name, CatGT_path)
runTPrime(run_name, TTL_chan, TPrime_path)

TTL = np.loadtxt(ni_path + 'TTL_' + str(TTL_chan) + '.txt')
#TTL = np.loadtxt(ni_path + run_name + '_g0_tcat.nidq.XD_8_2_0.txt') # uncorrected time stamps
#%% Import AP data, calculate ERP average and plot all responses on single channel
meta = sglx.readMeta(binFullPath) # all meta data
sRate = sglx.SampRate(meta) 

pre_dp = int(pre_sec*sRate)
post_dp = int(post_sec*sRate)
win_dp = pre_dp + post_dp + 1
t_vec = np.linspace(-pre_sec, post_sec, win_dp)

rawData = sglx.makeMemMapRaw(binFullPath, meta)

erpData = np.zeros([nChans, win_dp, len(TTL)])
plt.figure()
for trial in range(len(TTL)):
    
    event = int(round(TTL[trial]*sRate))
    selectData = rawData[chanList, event-pre_dp : event+post_dp +1]

    # apply gain correction and convert to mV
    convData = 1e3*sglx.GainCorrectIM(selectData, chanList, meta)
    
    # store the relevant data fragments in the 3 dimensional array
    erpData[:,:,trial] = convData
    
    plt.plot(t_vec, convData[chan,:])

erpAv = np.mean(erpData, 2)

plt.ylabel('Voltage [mV]')
plt.xlabel('Time [sec]')

#%% Plot mean ERP on all channels (ordered by depth)
plt.figure()
y_scale = 3
for chan in chanOrder:
    plt.plot(t_vec, (erpAv[chan, :] - np.mean(erpAv[chan,:])) *y_scale + chan, linewidth = 0.2)
plt.xlabel('Time from event [sec]')
plt.ylabel('Channel no. (0 = tip)')
