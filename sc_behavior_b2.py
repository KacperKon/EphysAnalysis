# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 10:07:17 2021

@author: kkondrakiewicz
"""
import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
sys.path.append(r'C:\Users\Kacper\Documents\Python Scripts\Sniff')
import sniff_tools as st

#%% Specify paths and some global analysis parameteres
data_path = sniff_dir = r'C:\Users\Kacper\Desktop\PSAM_SC\behavior_batch2'
expect_files = 3 # how many files per mice you expect
nframes = 662 # how many camera frames per trial you expect
pup_nframes = 373 # the same for pupil camera
pup_sr = pup_nframes/12
sigma = 0.25
binsize = 2 # for binned analysis, bin size in seconds
est_lat = 0 # estimated olfactometer latency if you want to align to real delivery
odor_start = 4 + est_lat
odor_end = 6 + est_lat
bsln_start = 1
ndays = 2
sniff_the_bin = [5, 8] # concentrate on this part - from 1 sec to 3 sec after odor presentation
pup_bin = [6, 9] # this can be different for pupil, which has slower dynamics

fig_path = r'C:\Users\Kacper\Desktop\PSAM_SC\plots\behavior_batch2'

#%% Import sniffing and trial data as a list of dictionaries - 1 dictionary for each mouse or session
sniffs = st.import_sniff_mat(data_path, expect_files)

#%% Change order of data to always have control then DREADD order

#[x['unique_id'] for x in sniffs]
myorder = [4, 0, 5, 1, 6, 2, 3, 7]
sniffs = [sniffs[i] for i in myorder]

# Of course, it would be better to do it using unique ids:
#sal_ses = ['220405_KK015_1', '220405_KK016_1', '220405_KK017_1', '220404_KK018_1']
#psem_ses =['220404_KK015_1', '220404_KK016_1', '220404_KK017_1', '220405_KK018_1']

#%% Exctract some basic info from the imported data
nses = len(sniffs)
nmice = int(nses/ndays)
ntrials = sniffs[0]['trial_idx'].size
npres = max(sniffs[0]['trial_occur'])
sr = sniffs[0]['samp_freq']

#%% Import pupil dilation data and parse it into trials
pup_raw, pup_ts = st.import_pupil(data_path)
pup_raw = [pup_raw[i] for i in myorder]
pup_ts = [pup_ts[i] for i in myorder]

pup_m = st.parse_pupil(pup_raw, pup_ts, ntrials, pup_nframes, nses, smoothen=1)

#%% Normalize and average pupil data
pup_delta = pup_m.copy()
pup_mybin = np.zeros([ntrials, nses])

for m in range(nses):
    for tr in range(ntrials):
        tmp_data = pup_m[tr, :, m]
        bsl = np.nanmean(tmp_data[int(bsln_start*pup_sr) : int(odor_start*pup_sr)])
        #bsl = 0
        pup_delta[tr, 0:tmp_data.size, m] = (tmp_data - bsl)
        
        pup_mybin[tr, m] = np.nanmean(pup_delta[tr, int(pup_bin[0]*pup_sr):int(pup_bin[1]*pup_sr), m])

#%% Restructure sniffing data into 3-dim array: trials x time point x miceand calculate breathing rate (multiple methods)
sniff_ons, sniff_list, sniff_bins, sniff_delbins, sniff_mybin = st.bin_sniff(sniffs, nframes, bsln_start, odor_start, sniff_the_bin, binsize)
sniff_gauss, sniff_delta = st.ins_sniff(sniff_ons, bsln_start, odor_start, sigma, sr)

#%% Create odor category matrix, indicating for each trial which odor type is it     
incl_descr = 'First 2 odor presentations'
tr_cat, tr_incl = st.select_trials_nov(sniffs, fam_min=5, fam_max=6, nov_min=1, nov_max=2)
ncat = tr_cat[0].shape[1]

#%% Calculate mean and SEM for each occurence of a given trial type
sniff_1bin_av, sniff_1bin_n, sniff_1bin_sem = st.av_by_occur(sniffs, sniff_mybin, tr_cat)
pup_1bin_av, pup_1bin_n, pup_1bin_sem = st.av_by_occur(sniffs, pup_mybin, tr_cat)

#%% Calculate mean sniffing across time for selected presentation
sniff_av = np.zeros([nframes, nses, ncat])
sniff_n = np.sum(tr_incl[0], 0)
sniff_sem = sniff_av.copy()

for m in range(nses):
    for cat in range(ncat):
        which_incl = sniffs[m]['trial_idx'][np.where(tr_incl[m][:,cat] == 1)] - 1 # IN MATLAB TRIAL INDEXES START FROM 1!!
        
        tmp_data = sniff_delta[which_incl, :, m].T
        #tmp_data = sniff_gauss[which_incl, :, m].T
        
        sniff_av[:,m,cat] = np.mean(tmp_data, 1)
        sniff_sem[:,m,cat] = np.std(tmp_data, 1) / np.sqrt(sniff_n[cat])
             
#%% The same, but for mean pupil
pup_av = np.zeros([pup_nframes, nses, ncat])
pup_n = np.sum(tr_incl[0], 0)
pup_sem = pup_av.copy()

for m in range(nses):
    for cat in range(ncat):
        which_incl = sniffs[m]['trial_idx'][np.where(tr_incl[m][:,cat] == 1)] - 1 # IN MATLAB TRIAL INDEXES START FROM 1!!
        
        tmp_data = pup_delta[which_incl, :, m].T
        #tmp_data = pup_m[which_incl, :, m].T
        
        pup_av[:,m,cat] = np.nanmean(tmp_data, 1)
        pup_sem[:,m,cat] = np.nanstd(tmp_data, 1) / np.sqrt(pup_n[cat])
        

#%% Plot breathing across time for some selected trials
fig, axes = plt.subplots(nmice, 1, sharex = 'all', sharey='all', figsize = (6, 9))
axes = axes.flatten()
tvec = np.linspace(-4, 7, nframes) - est_lat

row = 0
for m in range(0,nmice*ndays,2):
    
    axes[row].plot(tvec, sniff_av[:,m,0], label = 'FAM SAL', color = 'C7', linestyle = '-')
    axes[row].fill_between(tvec, sniff_av[:,m,0] + sniff_sem[:,m,0], sniff_av[:,m,0] - sniff_sem[:,m,0], alpha = 0.2, color = 'C7')
    
    axes[row].plot(tvec, sniff_av[:,m+1,0], label = 'FAM PSEM', color = 'C7', linestyle = '--')
    axes[row].fill_between(tvec, sniff_av[:,m+1,0] + sniff_sem[:,m+1,0], sniff_av[:,m+1,0] - sniff_sem[:,m+1,0], alpha = 0.2, color = 'C7')

    axes[row].plot(tvec, sniff_av[:,m,1], label = 'NOV SAL', color = 'C0', linestyle = '-')
    axes[row].fill_between(tvec, sniff_av[:,m,1] + sniff_sem[:,m,1], sniff_av[:,m,1] - sniff_sem[:,m,1], alpha = 0.2, color = 'C0')

    axes[row].plot(tvec, sniff_av[:,m+1,1], label = 'NOV PSEM', color = 'C0', linestyle = '--')
    axes[row].fill_between(tvec, sniff_av[:,m+1,1] + sniff_sem[:,m+1,1], sniff_av[:,m+1,1] - sniff_sem[:,m+1,1], alpha = 0.2, color = 'C0')
    
    axes[row].axvline(x = 0, linestyle = '--', color = 'gray', linewidth = 1)
    axes[row].axvline(x = 2, linestyle = '--', color = 'gray', linewidth = 1)

    axes[row].set_ylabel(u"\u0394" + ' sniffing [inh/sec]')
    #axes[row].set_ylabel('sniffing [inh/sec]')

    
    ax2 = axes[row].twinx()
    ax2.set_yticks([])
    mouse_id = sniffs[m]['unique_id'][7:12]
    ax2.set_ylabel('Mouse ' + mouse_id)
    
    row = row+1
    
    
axes[row-1].legend(loc = 3)
axes[row-1].set_xlabel('Time from odor [sec]')
fig.suptitle(incl_descr)

plt.savefig(fig_path + '\\Sniff_' + incl_descr + '.png')


#%% Plot habituation curve for each mouse
fig, axes = plt.subplots(nmice, 1, sharex = 'all', sharey='all', figsize = (6, 9))
axes = axes.flatten()

pvec_f = np.arange(-3,11)
pvec_n = np.arange(1,11)

row = 0
for m in range(0,nmice*ndays,2):
    
    axes[row].errorbar(pvec_f, sniff_1bin_av[:,m,0], sniff_1bin_sem[:,m,0], label = 'FAM SAL', color = 'C7', linestyle = '-')
    axes[row].errorbar(pvec_f, sniff_1bin_av[:,m+1,0], sniff_1bin_sem[:,m+1,0], label = 'FAM PSEM', color = 'C7', linestyle = '--')

    axes[row].errorbar(pvec_n, sniff_1bin_av[:10,m,1], sniff_1bin_sem[:10,m,1], label = 'NOV SAL', color = 'C0', linestyle = '-')
    axes[row].errorbar(pvec_n, sniff_1bin_av[:10,m+1,1], sniff_1bin_sem[:10,m+1,1], label = 'NOV PSEM', color = 'C0', linestyle = '--')
        
    axes[row].set_ylabel(u"\u0394" + ' sniffing [inh/sec]')
    
    ax2 = axes[row].twinx()
    ax2.set_yticks([])
    mouse_id = sniffs[m]['unique_id'][7:12]
    ax2.set_ylabel('Mouse ' + mouse_id)
    row = row+1
    
axes[row-1].legend(loc = 4)
axes[row-1].set_xticklabels(['PB1', 'PB2', 'PB4', '2', '4', '6', '8', '10'])
axes[row-1].set_xlabel('Presentation number')
fig.suptitle('Habituation curve')

plt.savefig(fig_path + '\\Sniff_hab.png')

#%% Plot pupil for selected trials
fig, axes = plt.subplots(nmice, 1, sharex = 'all', figsize = (6, 9))
axes = axes.flatten()
tvec = np.linspace(-4, 8, pup_nframes) - est_lat

row = 0
for m in range(0,nmice*ndays,2):
    axes[row].plot(tvec, pup_av[:,m,0], label = 'FAM SAL', color = 'C7', linestyle = '-')
    axes[row].fill_between(tvec, pup_av[:,m,0] + pup_sem[:,m,0], pup_av[:,m,0] - pup_sem[:,m,0], alpha = 0.2, color = 'C7')
    
    axes[row].plot(tvec, pup_av[:,m+1,0], label = 'FAM PSEM', color = 'C7', linestyle = '--')
    axes[row].fill_between(tvec, pup_av[:,m+1,0] + pup_sem[:,m+1,0], pup_av[:,m+1,0] - pup_sem[:,m+1,0], alpha = 0.2, color = 'C7')

    axes[row].plot(tvec, pup_av[:,m,1], label = 'NOV SAL', color = 'C0', linestyle = '-')
    axes[row].fill_between(tvec, pup_av[:,m,1] + pup_sem[:,m,1], pup_av[:,m,1] - pup_sem[:,m,1], alpha = 0.2, color = 'C0')

    axes[row].plot(tvec, pup_av[:,m+1,1], label = 'NOV PSEM', color = 'C0', linestyle = '--')
    axes[row].fill_between(tvec, pup_av[:,m+1,1] + pup_sem[:,m+1,1], pup_av[:,m+1,1] - pup_sem[:,m+1,1], alpha = 0.2, color = 'C0')

    axes[row].axvline(x = 0, linestyle = '--', color = 'gray', linewidth = 1)
    axes[row].axvline(x = 2, linestyle = '--', color = 'gray', linewidth = 1)
    
    axes[row].set_ylabel(u"\u0394" + ' pupil dilation [au]')
    #axes[row].set_ylabel('~pupil size [mm]')
    
    ax2 = axes[row].twinx()
    ax2.set_yticks([])
    mouse_id = sniffs[m]['unique_id'][7:12]
    ax2.set_ylabel('Mouse ' + mouse_id)
    
    row = row+1
    
    
axes[row-1].legend(loc=3)
axes[row-1].set_xlabel('Time from odor [sec]')
fig.suptitle(incl_descr)

plt.savefig(fig_path + '\\Pupil_' + incl_descr + '.png')

#%% Plot pupil habituation curve for each mouse
fig, axes = plt.subplots(nmice, 1, sharex = 'all', sharey='all', figsize = (6, 9))
axes = axes.flatten()

pvec_f = np.arange(-3,11)
pvec_n = np.arange(1,11)

row = 0
for m in range(0,nmice*ndays,2):
    axes[row].errorbar(pvec_f, pup_1bin_av[:,m,0], pup_1bin_sem[:,m,0], label = 'FAM SAL', color = 'C7', linestyle = '-')
    axes[row].errorbar(pvec_f, pup_1bin_av[:,m+1,0], pup_1bin_sem[:,m+1,0], label = 'FAM PSEM', color = 'C7', linestyle = '--')

    axes[row].errorbar(pvec_n, pup_1bin_av[:10,m,1], pup_1bin_sem[:10,m,1], label = 'NOV SAL', color = 'C0', linestyle = '-')
    axes[row].errorbar(pvec_n, pup_1bin_av[:10,m+1,1], pup_1bin_sem[:10,m+1,1], label = 'NOV PSEM', color = 'C0', linestyle = '--')
        
    axes[row].set_ylabel(u"\u0394" + ' pupil dilation [au]')
    #axes[row].set_ylabel('~pupil size [mm]')
    
    ax2 = axes[row].twinx()
    ax2.set_yticks([])
    mouse_id = sniffs[m]['unique_id'][7:12]
    ax2.set_ylabel('Mouse ' + mouse_id)
    
    row = row+1
    
    
axes[row-1].legend()
axes[row-1].set_xticklabels(['PB1', 'PB2', 'PB4', '2', '4', '6', '8', '10'])
axes[row-1].set_xlabel('Presentation number')
fig.suptitle('Habituation curve')

plt.savefig(fig_path + '\\Pupil_hab.png')


#%% Rasters

# for m in range(0, nmice*ndays, 2):
#     novel_oc = sniffs[m]['trial_novelty'] == True
#     sortby = np.argsort(sniffs[m]['trial_occur'][novel_oc])

    
#     #sortby = np.argsort(sniffs[m]['trial_occur'])
#     fig, axes = plt.subplots(ndays, 1, sharex = 'all', sharey='all', figsize = (6, 9))
#     axes = axes.flatten()
#     axes[0].eventplot(sniffs[m]['ml_inh_onsets'][sortby])
#     axes[1].eventplot(sniffs[m+1]['ml_inh_onsets'][sortby])
    
#     ax2 = axes[0].twinx()
#     ax3 = axes[1].twinx()
#     ax2.set_yticks([])
#     ax3.set_yticks([])

#     ax2.set_ylabel(sniffs[m]['unique_id'][0:-2])
#     ax3.set_ylabel(sniffs[m+1]['unique_id'][0:-2])
 
