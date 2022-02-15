# -*- coding: utf-8 -*-
"""
Import spiking data, calculate firing rates and data for rasters, save it. 

Created on Tue Aug 17 16:34:14 2021
@author: kkondrakiewicz
"""
#%% Imports
import sys
# Add directory with ephys module
sys.path.append(r'C:\Users\Kacper\Documents\Python Scripts\EphysAnalysis')
sys.path.append(r'C:\Users\Kacper\Documents\Python Scripts\Sniff')
sys.path.append(r'C:\Users\Kacper\Documents\Python Scripts')
import ephys as ep
import sniff_tools as st
import readSGLX as sglx

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from pathlib import Path
import time

#%% Set parameters for analysis
pre_event = 3
post_event = 6
ifr_sr = 50 # number of data points / sec for instantenous firing rate
    
#%% Specify files and paths
sess_ids = ['211207_KK006', '211207_KK007', '211208_KK006', '211208_KK007', \
           '211209_KK006', '211209_KK007', '211210_KK006', '211210_KK007']
    
sniff_dir = r'C:\Users\Kacper\Desktop\PSAM_SC\data'

spks_dirs = []
for ses in sess_ids:
    tmp = "W://preprocessed_data//ephys_sorted//catgt_"+ses+"_g0//"+ses+ "_g0_imec0//imec0_ks2//"
    spks_dirs.append(tmp)
    
#%% Import ephys data
su_ts = []; su_id = [] # for single units
mu_ts = []; mu_id = [] # for ALL, including multi units as well

for curr_dir in spks_dirs:
    tmp, tmp2 = ep.read_spikes(curr_dir, read_only = 'good')
    su_ts.append(tmp)
    su_id.append(tmp2)
    
    tmp3, tmp4 = ep.read_spikes(curr_dir, read_only = 'mua')
    mu_ts.append(tmp + tmp3)
    mu_id.append(np.concatenate([tmp2, tmp4]))

#%% Import cluster info
clst_info = []
chan_pos = []

for curr_dir in spks_dirs:
    tmp = pd.read_csv(curr_dir + "cluster_info.tsv", sep='\t')
    tmp2 = np.load(curr_dir + "channel_positions.npy")
    clst_info.append(tmp)
    chan_pos.append(tmp2)
    
#%% Import odor trial data
sniffs = st.import_sniff_mat(sniff_dir)

#%% Get some basic variables
nses = len(sniffs)
ntrials = sniffs[0]['trial_idx'].size
npres = max(sniffs[0]['trial_occur'])

#%% Calculate rasterplots % firing rates
cntrd_su = []
su_fr = []; su_mfr = []; su_efr = []; t_vec = []
cntrd_mu = []
mu_fr = []; mu_mfr = []; mu_efr = []

for s in range(nses):
    print('Calculating ' + str(s+1) + '/' + str(nses))
    
    tmp = ep.calc_rasters(su_ts[s], sniffs[s]['ephys_onsets'], pre_event, post_event)
    cntrd_su.append(tmp)
    tmp = ep.calc_rasters(mu_ts[s], sniffs[s]['ephys_onsets'], pre_event, post_event)
    cntrd_mu.append(tmp)

    tmp = ep.fr_events_binless(cntrd_su[s], 0.100, 4, ifr_sr, pre_event, post_event)
    tmp2 =ep.fr_events_binless(cntrd_mu[s], 0.100, 4, ifr_sr, pre_event, post_event)
    su_fr.append(tmp[0]); mu_fr.append(tmp2[0])
    su_mfr.append(tmp[1]); mu_mfr.append(tmp2[1])
    su_efr.append(tmp[2]); mu_efr.append(tmp2[2])
    t_vec.append(tmp[3])
    
print('Done!')

#%% Calculate zscores - only for multiunit here
all_zsc = []; mean_zsc = []; sem_zsc = []; bin_edges = []

for s in range(nses):
    tmp = ep.zscore_events(mu_fr[s], 1/ifr_sr, pre_event, post_event)
    all_zsc.append(tmp[0])
    mean_zsc.append(tmp[1])
    sem_zsc.append(tmp[2])
    bin_edges.append(tmp[3])

#%% Calculate firing rate by channel (in bins):
chan_fr = []; chan_mfr = []
for s, curr_dir in enumerate(spks_dirs):
    tmp, tmp2, bin_edges = ep.fr_by_chan(mu_ts[s], mu_id[s], curr_dir, 0.25)
    chan_fr.append(tmp)
    chan_mfr.append(tmp2)
    
#plt.figure()
#sns.heatmap(chan_fr[2], vmax = 200, cmap = 'binary')

#%% Plot how PSAM influences firing rate across locations
sal_pairs = [[0, 4], [1, 5]] # saline recordings - same mouse in 1 sublist
psam_pairs =[[2, 6], [3, 7]] # PSAM recordings

#save_dir = save_plots + 'response_locations\\'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

for pp in range(len(sal_pairs)):
    
    plt.figure(figsize = (4,8))
    
    pair = sal_pairs[pp]
    x = np.hstack([chan_pos[pair[0]][:,0], chan_pos[pair[1]][:,0]])
    y = np.hstack([chan_pos[pair[0]][:,1], chan_pos[pair[1]][:,1]])
    val = np.hstack([ chan_mfr[pair[0]],  chan_mfr[pair[1]]])
    plt.scatter(x, y, val*5, facecolors = 'none', edgecolors='b')

    pair = psam_pairs[pp]
    x = np.hstack([chan_pos[pair[0]][:,0], chan_pos[pair[1]][:,0]])
    y = np.hstack([chan_pos[pair[0]][:,1], chan_pos[pair[1]][:,1]])
    val = np.hstack([ chan_mfr[pair[0]],  chan_mfr[pair[1]]])
    plt.scatter(x, y, val*5, facecolors = 'none', edgecolors='r')


    fig_title = sess_ids[pair[0]][-5:]
    plt.title(fig_title)
    
    plt.ylabel('Position dorso-ventral [um]')
    plt.xlabel('Position anterio-posterior [um]')
    
    #plt.savefig(save_dir + fig_title + '.png', dpi = 250)

#%% Plot only bottom (2nd day) recordings and do stats

from scipy import stats

for m in range(2):
    
    plt.figure()
    
    val_s = chan_mfr[sal_pairs[m][1]]
    val_p = chan_mfr[psam_pairs[m][1]]
    valid_chans = np.logical_or(val_s > 0, val_p > 0)
    val_s = val_s[valid_chans]
    val_p = val_p[valid_chans]
    
    plt.scatter(np.ones([1, val_s.size]), val_s)
    plt.scatter(np.ones([1, val_p.size])+1, val_p)
        
    for i in range(val_s.size):
        plt.plot([1,2], [val_s[i], val_p[i]], color = 'gray', alpha=0.5)
        
    pval_t = stats.wilcoxon(val_s, val_p)[1]
    #pval_t = stats.ttest_rel(val_s, val_p)[1]
    plt.annotate(np.round(pval_t, 4), [1,20])
    

#%% Plot responses of all neurons using one heatmap

# save_dir = save_plots + 'grand_average\\'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
    
# plt.ioff()       
for s in range(nses):
    
    fig, axes = plt.subplots(2, 1, sharex = True)
    axes = axes.flatten()

    sortby = np.mean(mean_zsc[s][:, int(ifr_sr*pre_event) : int(ifr_sr*pre_event+ifr_sr*3)], 1).argsort()
    sns.heatmap(mean_zsc[s][sortby, :], vmin = -1, vmax = 3, cmap = 'inferno', \
        ax = axes[0], cbar_kws = {'location':'top', 'shrink': 0.5, 'anchor': (1.0,1.0)})
    
    axes[1].plot(np.mean(mean_zsc[s], 0))
    
    y1 = np.mean(mean_zsc[s], 0) + (np.mean(mean_zsc[s], 0) / np.sqrt(mean_zsc[s].shape[0]))
    y2 = np.mean(mean_zsc[s], 0) - (np.mean(mean_zsc[s], 0) / np.sqrt(mean_zsc[s].shape[0]))
    axes[1].fill_between(np.arange(mean_zsc[s].shape[1]), y1, y2, alpha=0.5, zorder=2)
    
    axes[0].axvline(x = pre_event*ifr_sr, linestyle = '--', color = 'gray', linewidth = 1)
    axes[1].axvline(x = pre_event*ifr_sr, linestyle = '--', color = 'gray', linewidth = 1)
    
    xlabs = np.round(t_vec[s][::ifr_sr])
    xlabs = np.linspace(-pre_event, post_event, pre_event+post_event+1)
    xticks = np.linspace(0,mean_zsc[s].shape[1],len(xlabs))
    
    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels(xlabs)
    
    axes[0].set_ylabel('Unit #')
    axes[1].set_ylabel('Mean z-score')
    
    fig.suptitle('Z-scored responses to all odors: ' + sess_ids[s])
       
#     fig.savefig(save_dir + sess_ids[s] + '.png', dpi = 250)
#     plt.close(fig)
# plt.ion()      


#%% Plot average response to novel odors

tr_cat, tr_incl = st.select_trials_nov(sniffs,1,3,1,3)

for s in range(nses):
    
    which_rows = tr_incl[s][:,1].astype(bool)
    which_trials = sniffs[s]['trial_idx'][which_rows] - 1

    tmp = mean_zsc[s][which_trials,:]
    plt.figure()
    plt.plot(np.mean(tmp, 0))

#%% Plot as line plot - fr on all chans
# for pp in range(len(sal_pairs)):
    
#     plt.figure()

#     pair = sal_pairs[pp]
#     val = np.hstack([ chan_mfr[pair[0]],  chan_mfr[pair[1]]])
#     plt.plot(val, alpha = 0.5)
#     pair = psam_pairs[pp]
#     val = np.hstack([ chan_mfr[pair[0]],  chan_mfr[pair[1]]])
#     plt.plot(val, alpha = 0.5)
