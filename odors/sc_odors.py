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
save_dir = 'C:\\Users\\Kacper\\Desktop\\PSAM_SC\\rasters\\'


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

#%% Import odor trial data
sniffs = st.import_sniff_mat(sniff_dir)

#%%
#mua_ts, mua_id = ep.read_spikes(spks_dir, read_only = 'mua')
#all_ts = spikes_ts + mua_ts
#all_id = np.concatenate([units_id, mua_id])

#%% Calculate rasterplots
cntrd_ts = []
all_fr = []; mean_fr = []; sem_fr = []; t_vec = []

for idx, ses in enumerate(spks_ts):
    print('Calculating ' + str(idx+1) + '/' + str(len(spks_ts)))
    
    tmp = ep.calc_rasters(ses, sniffs[idx]['ephys_onsets'], pre_event, post_event)
    cntrd_ts.append(tmp)
    
    tmp = ep.fr_events_binless(cntrd_ts[idx], 0.100, 4, 30000.0, 1000, pre_event, post_event)
    all_fr.append(tmp[0])
    mean_fr.append(tmp[1])
    sem_fr.append(tmp[2])
    t_vec.append(tmp[3])
    
print('Done!')
    
    
#%% Save raster plots + mean firing rate
for ii in range(len(cntrd_ts)):
    tmp = save_dir + sess_ids[ii] + '\\'
    os.mkdir(tmp)
    ep.plot_responses(cntrd_ts[ii], mean_fr[ii], sem_fr[ii], t_vec[ii], tmp)

#%% Select novel odors
tr_cat, tr_incl = st.select_trials_nov(sniffs, fam_min=5, fam_max=5, nov_min=1, nov_max=1)

ncat = tr_cat[0].shape[1]
nses = 8

#for m in range(nses):
for m in [4]:
    for cat in range(ncat-1):
        which_incl = sniffs[m]['trial_idx'][np.where(tr_incl[m][:,cat] == 1)] - 1 # IN MATLAB TRIAL INDEXES START FROM 1!!
        
        plt.figure()
        res_list = [cntrd_ts[m][23][i] for i in which_incl]
        plt.eventplot(res_list)
        
        mean_data = np.mean(all_fr[4][23][which_incl], 0, keepdims=True)
        sem_data = np.std(all_fr[4][23][which_incl], 0, keepdims=True) / np.sqrt(which_incl.size)

#%%
save_dir = 'C:\\Users\\Kacper\\Desktop\\PSAM_SC\\test\\'

ep.plot_responses([res_list], mean_data, sem_data, t_vec[0], save_dir)

#%% Plot responses by odor identity


#%%


#%%
#all_fr, t_vec = bfr(all_ts, 1)
#mean_fr = np.mean(all_fr, 1)
print(tmp)

#%%
#plt.figure()
#plt.plot(all_fr[15,:])

#%% Get the heights
#chan_pos = np.load(spks_dir+'channel_positions.npy')
#clst_info = pd.read_csv(spks_dir + "cluster_info.tsv", sep='\t')

#depth = clst_info[clst_info['id'].isin(all_id)]
#depth = depth.sort_values('depth')

#id_by_depth = np.array(depth['id'])

#nunits = id_by_depth.size
#which_row = np.zeros(nunits, dtype = 'int')
#for nrn in range(nunits):
#    which_row[nrn] = np.where(all_id == id_by_depth[nrn])[0]

#%% Plot responses of all neurons using one heatmap
all_fr, t_vec = fr_by_chan(all_ts, all_id, spks_dir, 1)

plt.figure()
#sortby = mean_fr.argsort()
#sortby = which_row
#sns.heatmap(all_fr[sortby,:], vmax = 100)
sns.heatmap(all_fr[:,0:3800], vmax = 200, cmap = 'binary')