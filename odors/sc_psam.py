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
import ephys as ep
sys.path.append(r'C:\Users\Kacper\Documents\Python Scripts\Sniff')
import sniff_tools as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

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

#%% Calculate rasterplots
cntrd_su = []
su_fr = []; su_mfr = []; su_efr = []; t_vec = []
cntrd_mu = []
mu_fr = []; mu_mfr = []; mu_efr = []

for s in range(nses):
    print('Calculating ' + str(s+1) + '/' + str(nses))
    
    tmp = ep.calc_rasters(su_ts[s], sniffs[s]['ephys_onsets'], pre_event, post_event)
    cntrd_su.append(tmp)
    #tmp = ep.calc_rasters(mu_ts[s], sniffs[s]['ephys_onsets'], pre_event, post_event)
    #cntrd_mu.append(tmp)

    tmp = ep.fr_events_binless(cntrd_su[s], 0.100, 4, ifr_sr, pre_event, post_event)
    #tmp2 =ep.fr_events_binless(cntrd_mu[s], 0.100, 4, ifr_sr, pre_event, post_event)
    su_fr.append(tmp[0])#; mu_fr.append(tmp2[0])
    su_mfr.append(tmp[1])#; mu_mfr.append(tmp2[1])
    su_efr.append(tmp[2])#; mu_efr.append(tmp2[2])
    t_vec.append(tmp[3])
    
print('Done!')

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

#%% 
for pp in range(len(sal_pairs)):
    
    plt.figure()

    pair = sal_pairs[pp]
    val = np.hstack([ chan_mfr[pair[0]],  chan_mfr[pair[1]]])
    plt.plot(val, alpha = 0.5)
    pair = psam_pairs[pp]
    val = np.hstack([ chan_mfr[pair[0]],  chan_mfr[pair[1]]])
    plt.plot(val, alpha = 0.5)
 
#%%
from scipy import stats

for pp in range(len(sal_pairs)):
    
    plt.figure()
    
    pair = sal_pairs[pp]
    val = np.hstack([ chan_mfr[pair[0]],  chan_mfr[pair[1]]])
    plt.scatter(np.ones([1, val.size]), val)
    
    pair = psam_pairs[pp]
    val2 = np.hstack([ chan_mfr[pair[0]],  chan_mfr[pair[1]]])
    plt.scatter(np.ones([1, val2.size]) + 1, val2)
    
    plt.figure()
    plt.hist(val2-val, 500)
    plt.ylim([0, 20])
    
    stats.ttest_rel(val2,val)
    
    
    
