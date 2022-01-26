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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
import pandas as pd


#%% Set parameters
spks_dir = "W://preprocessed_data//ephys_sorted//catgt_211207_KK006_g0//211207_KK006_g0_imec0//imec0_ks2//" # directory with spikes (Phy output)

spikes_ts, units_id = ep.read_spikes(spks_dir, read_only = 'good')
mua_ts, mua_id = ep.read_spikes(spks_dir, read_only = 'mua')


all_ts = spikes_ts + mua_ts
all_id = np.concatenate([units_id, mua_id])

#%%

def bfr(spikes_ts, bin_size):

    # Calculate how big is your data and bin edges (in sec)
    nunits = len(spikes_ts)
    max_t = np.round(np.max([np.max(x) for x in spikes_ts])) # It's rounding!!
    
    bin_edges = np.arange(0, max_t, bin_size)
    nbins = bin_edges.size - 1
    
    # Create empty list/arrays for storing results
    all_fr = np.zeros([nunits, nbins])
    
    # Do the firing rate calculation
    for nrn in range(nunits):
        spks_in_bins = np.histogram(spikes_ts[nrn], bin_edges)
        fr_in_bins = spks_in_bins[0]*1 / bin_size
        all_fr[nrn, :] = fr_in_bins[:]
    
    return all_fr, bin_edges


def fr_by_chan(spikes_ts, units_id, cluster_info_dir, bin_size):
    
    clst_info = pd.read_csv(cluster_info_dir + "cluster_info.tsv", sep='\t')

    nchans = 384
    nunits = len(spikes_ts)
    max_t = np.round(np.max([np.max(x) for x in spikes_ts])) # It's rounding!!
    
    bin_edges = np.arange(0, max_t, bin_size)
    nbins = bin_edges.size - 1
    
    all_fr = np.zeros([nchans, nbins])
        
    for nrn in range(nunits):
        spks_in_bins = np.histogram(spikes_ts[nrn], bin_edges)[0]
        
        which_chan = int(clst_info['ch'][clst_info['id']==units_id[nrn]])
        
        all_fr[which_chan, :] = all_fr[which_chan, :] + spks_in_bins
        
        #fr_in_bins = spks_in_bins[0]*1 / bin_size
        #all_fr[nrn, :] = fr_in_bins[:]    
    all_fr = all_fr / bin_size

    return all_fr, bin_edges

#%%
#all_fr, t_vec = bfr(all_ts, 1)
#mean_fr = np.mean(all_fr, 1)


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