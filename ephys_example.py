# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 16:34:14 2021

@author: kkondrakiewicz
"""
#%% Imports
import sys
# Add directory with ephys module
sys.path.append(r'C:\Users\kkondrakiewicz\Documents\Python Scripts\EphysAnalysis')
import ephys as ep
import numpy as np

#%% Set parameters
spks_dir = "D:\\buffering_np\\NP2_test\\" # directory with spikes (Phy output)
sounds = np.loadtxt("D:\\buffering_np\\NP2_test\\TTL_gcat\\TTL_3.txt") # txt file with event timestamps
save_dir = 'C:\\Users\\kkondrakiewicz\\Desktop\\ploty\\' # where to save plots

# window size for analysis
pre_event = 2.0
post_event = 5.0
bin_size = 0.250

# Read spikes from good units; in real life set also sampling rate to the one from Neuropixels .meta file
spikes_ts, units_id = ep.read_spikes(spks_dir, read_only = 'good')

# Center spikes on events within a selected time window for each trial
centered_ts = ep.calc_rasters(spikes_ts, sounds, pre_event, post_event)

# Based on these results, plot rasters for all neurons
fig, axes = ep.plot_rasters(centered_ts)
fig.suptitle('Single neurons - responses to events')

# Now calculate also firing rate for each time bin and each trial
all_fr, mean_fr, sem_fr, bin_edges = ep.fr_events(centered_ts, bin_size, pre_event, post_event)

# And also normalize the firing rate to individual baselines (pre-period during each trial) 
all_zsc, mean_zsc, sem_zsc, bin_edges = ep.zscore_events(all_fr, bin_size, pre_event, post_event)

# Now you can plot the PSTH: mean firing rate (normalized or not) together with SEM, for all neurons
fig, axes = ep.plot_psth(mean_zsc, sem_zsc, bin_edges)
fig.suptitle('Single neurons - mean responses to events')

# Or combine the raster and PSTH on a single plot: this we don't want to display for all neurons, so instead save it
ep.plot_responses(centered_ts, mean_zsc, sem_zsc, bin_edges, save_dir)

