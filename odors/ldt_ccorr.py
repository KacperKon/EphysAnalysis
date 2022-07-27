# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 08:45:28 2022

@author: Kacper
"""
import matplotlib.pyplot as plt
import numpy as np 
import sys
sys.path.append(r'C:\Users\Kacper\Documents\Python Scripts\Sniff')
import sniff_tools as st

#%%
s = 4 # session number
win_ms = 60 # window for cross-correlation histogram (one-sided)
bin_ms = 3 # bin size for cc. histogram
start_ms = 0.5 # start of data window within the trials
stop_ms = 2.5 # end of data window
bin_edges_xcorr = np.arange(-win_ms, win_ms + bin_ms, bin_ms)

units_ldt = [7, 8, 9, 19, 20, 310, 386, 387]
units_pbn = [61, 71, 378, 156, 180, 235, 239, 245, 252]

n_pbn = len(units_pbn)
n_ldt = len(units_ldt)

tr_cat, tr_incl = st.select_trials_nov(sniffs, fam_min=1, fam_max=10, nov_min=1, nov_max=5)
which_incl = sniffs[s]['trial_idx'][np.where(tr_incl[s][:,1] == 1)] - 1
ntrials = len(which_incl)

fig, axes = plt.subplots(n_pbn, n_ldt)

for idx_p, nrn_p in enumerate(units_pbn):
    ip = np.where(spks_id[s] == nrn_p)[0][0]
    l1 = cntrd_ts[s][ip] # select neuron
    l1 = [l1[i] for i in which_incl] # select trials
    
    for idx_l, nrn_l in enumerate(units_ldt):
        il = np.where(spks_id[s] == nrn_l)[0][0]
        l2 = cntrd_ts[s][il] # select neuron
        l2 = [l2[i] for i in which_incl] # select trials

        deltas = np.array([])
        for tr in range(ntrials):
            t1 = l1[tr]
            t1 = t1[np.logical_and((t1 >= start_ms), (t1 <= stop_ms))]
            t2 = l2[tr]
            t2 = t2[np.logical_and((t2 >= start_ms), (t2 <= stop_ms))]

            for ts1 in t1:
                d = t2 - ts1
                d = d*1000 # sec to msec
                d = d[np.logical_and((d >= -win_ms), (d <= win_ms))]
                deltas = np.append(deltas, d) 
                #deltas.append(d)
                
        axes[idx_p, idx_l].hist(deltas, bin_edges_xcorr)
        axes[idx_p, idx_l].axvline(0, color =  'black')
        
        axes[0, idx_l].set_title('LDT: ' + str(units_ldt[idx_l]), )
    axes[idx_p, 0].set_ylabel('PBN: ' + str(units_pbn[idx_p]))

    print('Cross-correlations: ' + str(idx_p+1) + '/' + str(n_pbn))
        
    
#%% Simple inhalation synchronization

units = units_pbn
win_ms = 250 # window for cross-correlation histogram (one-sided)
bin_ms = 10 # bin size for cc. histogram
start_ms = 0 # start of data window within the trials
stop_ms = 3 # end of data window
bin_edges_xcorr = np.arange(-win_ms, win_ms + bin_ms, bin_ms)


l2 = sniffs[s]['ml_inh_onsets']/60 - 4
which_incl = np.arange(0, 126, 1)
l2 = [l2[i] for i in which_incl]
ntrials = len(which_incl)

fig, axes = plt.subplots(3, 3)
axes = axes.flatten()


for idx_p, nrn_p in enumerate(units):
    ip = np.where(spks_id[s] == nrn_p)[0][0]
    l1 = cntrd_ts[s][ip] # select neuron
    l1 = [l1[i] for i in which_incl] # select trials
    
    deltas = np.array([])
    for tr in range(ntrials):
        t1 = l1[tr]
        t1 = t1[np.logical_and((t1 >= start_ms), (t1 <= stop_ms))]
        t2 = l2[tr]
        t2 = t2[np.logical_and((t2 >= start_ms), (t2 <= stop_ms))]

        for ts1 in t1:
            d = t2 - ts1
            d = d*1000 # sec to msec
            d = d[np.logical_and((d >= -win_ms), (d <= win_ms))]
            deltas = np.append(deltas, d) 
                
    axes[idx_p].hist(deltas, bin_edges_xcorr, density = True)
    axes[idx_p].axvline(0, linestyle = '--', color =  'gray')        
    axes[idx_p].set_title('PBN: ' + str(units[idx_p]))
    axes[idx_p].set_xlim([-win_ms, win_ms])

