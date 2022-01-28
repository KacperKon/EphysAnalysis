# -*- coding: utf-8 -*-
"""
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
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import os

#%% Set parameters for analysis
pre_event = 3
post_event = 6
bin_size = 0.25
ifr_sr = 50 # number of data points / sec for instantenous firing rate
save_plots = 'C:\\Users\\Kacper\\Desktop\\PSAM_SC\\plots\\'

if not os.path.exists(save_plots):
    os.makedirs(save_plots)
    
#%% Specify files and paths
sess_ids = ['211207_KK006', '211207_KK007', '211208_KK006', '211208_KK007', \
           '211209_KK006', '211209_KK007', '211210_KK006', '211210_KK007']
    
sniff_dir = r'C:\Users\Kacper\Desktop\PSAM_SC\data'

spks_dirs = []
for ses in sess_ids:
    tmp = "W://preprocessed_data//ephys_sorted//catgt_"+ses+"_g0//"+ses+ "_g0_imec0//imec0_ks2//"
    spks_dirs.append(tmp)

    
#%% Import ephys data
spks_ts = []
spks_id = []

for curr_dir in spks_dirs:
    tmp, tmp2 = ep.read_spikes(curr_dir, read_only = 'good')
    spks_ts.append(tmp)
    spks_id.append(tmp2)
    
#mua_ts, mua_id = ep.read_spikes(spks_dir, read_only = 'mua')
#all_ts = spikes_ts + mua_ts
#all_id = np.concatenate([units_id, mua_id])

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
cntrd_ts = []
all_fr = []; mean_fr = []; sem_fr = []; t_vec = []

for idx, ses in enumerate(spks_ts):
    print('Calculating ' + str(idx+1) + '/' + str(len(spks_ts)))
    
    tmp = ep.calc_rasters(ses, sniffs[idx]['ephys_onsets'], pre_event, post_event)
    cntrd_ts.append(tmp)

    tmp = ep.fr_events_binless(cntrd_ts[idx], 0.100, 4, ifr_sr, pre_event, post_event)
    all_fr.append(tmp[0])
    mean_fr.append(tmp[1])
    sem_fr.append(tmp[2])
    t_vec.append(tmp[3])
    
print('Done!')

#%% Calculate zscores
all_zsc = []; mean_zsc = []; sem_zsc = []; bin_edges = []

for s in range(nses):

    tmp = ep.zscore_events(all_fr[s], 1/ifr_sr, pre_event, post_event)
    all_zsc.append(tmp[0])
    mean_zsc.append(tmp[1])
    sem_zsc.append(tmp[2])
    bin_edges.append(tmp[3])
    
#%% Save rasters, mean firing and z-score on 1 plot
save_dir = save_plots + 'rasters_all\\'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for ii in range(nses):
    tmp = save_dir + sess_ids[ii] + '\\'
    os.mkdir(tmp)
    ep.plot_resp_zsc(cntrd_ts[ii], mean_fr[ii], sem_fr[ii], mean_zsc[ii], sem_zsc[ii], t_vec[ii], tmp)
    
#%% Plot responses of all neurons using one heatmap

save_dir = save_plots + 'grand_average\\'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
plt.ioff()       
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
       
    fig.savefig(save_dir + sess_ids[s] + '.png', dpi = 250)
    plt.close(fig)
plt.ion()   
    
    
#%% Calculate 1 average response and units locations
grav = [] 
unit_pos = []
which_av = np.arange(int(ifr_sr*pre_event), int(ifr_sr*pre_event+ifr_sr*3))

for s in range(nses):
    tmp = np.mean(mean_zsc[s][:, which_av], 1)
    grav.append(tmp)

    nunits = len(spks_id[s])
    tmp2 = np.zeros([nunits ,2])
    for nrn in range(nunits):
        which_chan = int(clst_info[s]['ch'][clst_info[s]['id']==spks_id[s][nrn]])
        tmp2[nrn,:] = chan_pos[s][which_chan,:]
    unit_pos.append(tmp2)
    
#%% Plot responsivity vs. location
rec_pairs = [[0, 4], [1, 5], [2, 6], [3, 7]]
rec_des = ['saline', 'saline', 'PSEM', 'PSEM', 'saline', 'saline', 'PSEM', 'PSEM']

save_dir = save_plots + 'response_locations\\'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for pair in rec_pairs:
    
    x = np.hstack([unit_pos[pair[0]][:,0], unit_pos[pair[1]][:,0]])
    y = np.hstack([unit_pos[pair[0]][:,1], unit_pos[pair[1]][:,1]])
    val = np.hstack([ grav[pair[0]],  grav[pair[1]]])

    fig_title = sess_ids[pair[0]][-5:] + ', ' + rec_des[pair[0]]
    
    plt.figure(figsize = (4,8))
    plt.scatter(x, y, val*200)
    plt.title(fig_title)
    
    plt.ylabel('Position dorso-ventral [um]')
    plt.xlabel('Position anterio-posterior [um]')
    
    plt.savefig(save_dir + fig_title + '.png', dpi = 250)
    
    
#%% Select novel odors
tr_cat, tr_incl = st.select_trials_nov(sniffs, fam_min=5, fam_max=5, nov_min=1, nov_max=1)
ncat = tr_cat[0].shape[1]

#for m in range(nses):
for m in [4]:
    for cat in range(ncat-1):
        which_incl = sniffs[m]['trial_idx'][np.where(tr_incl[m][:,cat] == 1)] - 1 # IN MATLAB TRIAL INDEXES START FROM 1!!
        
        plt.figure()
        res_list = [cntrd_ts[m][23][i] for i in which_incl]
        plt.eventplot(res_list)
        
        mean_data = np.mean(all_fr[4][23][which_incl], 0, keepdims=True)
        sem_data = np.std(all_fr[4][23][which_incl], 0, keepdims=True) / np.sqrt(which_incl.size)

#%% Maybe restructure data to 3 dim matrix?
# If you have odor x occurence x time points, it would be easy to average by them

occur = [1,2,3,4,5,6,7,8,9,10]
npres = max(occur)
nbins = len(t_vec[0])

for s in range(5,6): # NOW ONLY 1 MOUSE
    nunits = len(all_zsc[s])
    
    for nrn in range(nunits):
        zsc_by_oc = np.zeros([npres, nbins])
        
        for oc in occur:
            tr_cat, tr_incl = st.select_trials_nov(sniffs, oc , oc, oc, oc)
            cat = 0 # novel
            which_incl = sniffs[s]['trial_idx'][np.where(tr_incl[s][:,cat] == 1)] - 1 # IN MATLAB TRIAL INDEXES START FROM 1!!

            tmp = all_zsc[s][nrn][which_incl, :]
            tmp_mean = np.mean(tmp, 0)
            #tmp_sem = np.std(tmp, 0) / np.sqrt(tmp.size[0])
            zsc_by_oc[oc-1, :] = tmp_mean
        
        plt.figure()
        sns.heatmap(zsc_by_oc, vmin = -1, vmax = 3, cmap = 'inferno')

#%%
for s in range(5,6): # NOW ONLY 1 MOUSE
    nunits = len(all_zsc[s])
    
    for nrn in range(nunits):
        zsc_by_oc = np.zeros([npres, nbins])
        
        tr_cat, tr_incl = st.select_trials_nov(sniffs, oc , oc, oc, oc)
        cat = 0 # novel
        which_incl = sniffs[s]['trial_idx'][10:24] # - 1 # IN MATLAB TRIAL INDEXES START FROM 1!!

        tmp = all_zsc[s][nrn][which_incl, :]
        tmp_mean = np.mean(tmp, 0)
        
        res_list = [cntrd_ts[s][nrn][i] for i in which_incl]
        plt.figure()
        plt.eventplot(res_list)
            
        