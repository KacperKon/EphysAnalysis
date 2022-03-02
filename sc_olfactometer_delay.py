# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 17:22:03 2022

Estimate with what latency animals respond to odors - what might be olfactometer delay?

@author: Kacper
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
sys.path.append(r'C:\Users\Kacper\Documents\Python Scripts\Sniff')
import sniff_tools as st

#%% Specify paths and some global analysis parameteres
data_path = sniff_dir = r'C:\Users\Kacper\Desktop\PSAM_SC\data'
sal_ses = [0, 1, 4, 5] # sessions with saline, not DREADD
expect_files = 4 # how many files per mice you expect
nframes = 662 # how many camera frames per trial you expect
pup_nframes = 373 # the same for pupil camera
pup_sr = pup_nframes/12
sigma = 0.1
binsize = 2 # for binned analysis, bin size in seconds
odor_start = 4
odor_end = 6
bsln_start = 1
ndays = 4
sniff_the_bin = [6, 8] # concentrate on this part - from 1 sec to 3 sec after odor presentation
pup_bin = [6, 10] # this can be different for pupil, which has slower dynamics

fig_path = r'C:\Users\Kacper\Desktop\PSAM_SC\plots\behavior'

#%% Import sniffing and trial data as a list of dictionaries - 1 dictionary for each mouse or session
sniffs = st.import_sniff_mat(data_path, expect_files)

#%% Exctract some basic info from the imported data
nses = len(sniffs)
nmice = int(nses/ndays)
ntrials = sniffs[0]['trial_idx'].size
npres = max(sniffs[0]['trial_occur'])
sr = sniffs[0]['samp_freq']

#%% Restructure sniffing data into 3-dim array: trials x time point x miceand calculate breathing rate (multiple methods)
sniff_ons, sniff_list, sniff_bins, sniff_delbins, sniff_mybin = st.bin_sniff(sniffs, nframes, bsln_start, odor_start, sniff_the_bin, binsize)
sniff_gauss, sniff_delta = st.ins_sniff(sniff_ons, bsln_start, odor_start, sigma, sr)
sniff_grav = np.mean(sniff_delta, 0) # grand average from all trials

#%%
est_lat = 0.8
tvec = np.linspace(-4, 7, nframes) - est_lat

for m in range(nses):
    fig, axes = plt.subplots(2, 1, sharex = True)
    axes = axes.flatten()
    
    axes[0].eventplot(sniffs[m]['ml_inh_onsets']/sr - odor_start-est_lat)
    axes[1].plot(tvec, sniff_grav[:,m])
    axes[0].axvline(x = 0, color = 'gray', linewidth = 1)
    axes[1].axvline(x = 0, color = 'gray', linewidth = 1)
    axes[0].axvline(x = 2, color = 'gray', linewidth = 1)
    axes[1].axvline(x = 2, color = 'gray', linewidth = 1)
    
    axes[1].set_ylim([-0.5, 2])
    axes[1].set_xlabel('Estimated time from actual odor delivery [sec]')
    plt.suptitle('Sniffing aligned - ' + sniffs[m]['folder_identifier'])
    axes[0].set_ylabel('Trial #')
    axes[1].set_ylabel(u"\u0394" + ' inhalations/sec')
    
    plt.savefig(fig_path + '\\odor_latency_' + sniffs[m]['folder_identifier'] + '.png')

