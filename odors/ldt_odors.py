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
import copy
import scipy.io
import glob
import time

#%% Set parameters for analysis
pre_event = 3
post_event = 6
bin_size = 0.25
ifr_sr = 50 # number of data points / sec for instantenous firing rate
save_plots = 'C:\\Users\\Kacper\\Desktop\\LDTg\\plots_good_sessions\\'

if not os.path.exists(save_plots):
    os.makedirs(save_plots)
    
#%% Specify files and paths
#sess_ids = ['220308_KK012', '220308_KK013', '220309_KK012', '220309_KK013', \
#           '220310_KK012', '220310_KK013', '220311_KK012', '220311_KK013']
sess_ids = ['220308_KK012', '220309_KK012', '220309_KK013', \
            '220311_KK012', '220311_KK013']
#sess_ids = ['220311_KK013']
    
sniff_dir = r'C:\Users\Kacper\Desktop\LDTg\data\only_sorted'

spks_dirs = []
for ses in sess_ids:
    tmp = "W://preprocessed_data//ephys_sorted//catgt_"+ses+"_g0//"+ses+ "_g0_imec0//imec0_ks2_manual//"
    spks_dirs.append(tmp)

#%% Import histology data and define how it maps to animals / probe tracts
h12 = r'C:\Users\Kacper\Desktop\LDTg\histology\KK012_CCF\processed\brain_structures.csv'
h13 = r'C:\Users\Kacper\Desktop\LDTg\histology\KK013_CCF\processed\brain_structures.csv'

# First col - probe # from Sharp-Track, 2nd - corresponding shank #
pr_to_sh = np.array([[1, 3], [2, 2], [3, 1], [4, 0]])    

h12 = pd.read_csv(h12); h13 = pd.read_csv(h13)
# Repeat order of sessions
hist = [copy.copy(h12), copy.copy(h12), copy.copy(h13), copy.copy(h12), copy.copy(h13)]

# Add shank mapping to the histology dataframe
nses = len(sess_ids)
for ses in range(nses):
    hist[ses].insert(0, 'Shank', 9)
    for row in range(np.size(hist[ses]['Shank'])):
        probe = hist[ses]['Probe'][row]
        shank = pr_to_sh[pr_to_sh[:,0]==probe, 1] 
        hist[ses]['Shank'][row] = shank

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

### IF DATA NOT MANUALLY SORTED< USE FILE CLUSTER_GROUP (_INFO DOES NOT EXIST) ###### 

clst_info = []
chan_pos = []
chan_map = []

for curr_dir in spks_dirs:
    tmp = pd.read_csv(curr_dir + "cluster_info.tsv", sep='\t')
    tmp['FP_label'] = 'None';  tmp['FP_abbrv'] = 'None' # add columns for anatomy labels
    tmp['CCF_label'] = 'None'; tmp['CCF_abbrv'] = 'None'
    
    tmp2 = np.load(curr_dir + "channel_positions.npy")
    
    curr_dir = os.path.dirname(os.path.dirname(curr_dir)) # go folder up for channel map
    curr_dir = glob.glob(curr_dir + '/*chanMap.mat')[0]
    tmp3 = scipy.io.loadmat(curr_dir)
    
    clst_info.append(tmp)
    chan_pos.append(tmp2)
    chan_map.append(tmp3)
    

#%% Import odor trial data
sniffs = st.import_sniff_mat(sniff_dir, expect_files = 4)

#%% Get some basic variables
ntrials = sniffs[0]['trial_idx'].size
npres = max(sniffs[0]['trial_occur'])

#%% Get anatomical labels of each unit

# First, correct shank number - channel_positions maps it wrongly!!
for ses in range(nses):
    chans = clst_info[ses].ch    
    for row in range(np.size(chans)):
        shank = int(chan_map[ses]['kcoords'][chans[row]] - 1)
        clst_info[ses].sh[row] = shank
    
    # for row in range(np.size(chans)):
    #     if ((chans[row] >= 0) and (chans[row] <= 47)) or ((chans[row] >= 96) and (chans[row] <= 143)):
    #         clst_info[ses].sh[row] = 0 # shank 0
    #     if ((chans[row] >= 48) and (chans[row] <= 95)) or ((chans[row] >= 144) and (chans[row] <= 191)):
    #         clst_info[ses].sh[row] = 1 # shank 1
    #     if ((chans[row] >= 192) and (chans[row] <= 239)) or ((chans[row] >= 288) and (chans[row] <= 335)):
    #         clst_info[ses].sh[row] = 2 # shank 2
    #     if ((chans[row] >= 240) and (chans[row] <= 287)) or ((chans[row] >= 336) and (chans[row] <= 383)):
    #         clst_info[ses].sh[row] = 3 # shank 3

for ses in range(nses):
    for row in range(np.size(clst_info[ses]['id'])):
        # match units to electrode locations
        shank = clst_info[ses]['sh'][row]
        depth = clst_info[ses]['depth'][row]
        which_hist = np.logical_and((hist[ses]['Shank'] == shank), (hist[ses]['Pos_from_tip'] == depth))
        
        # add updated FP atlas labels
        label = hist[ses]['FP_abbrv'][which_hist].values[0]
        fullname = hist[ses]['FP_name'][which_hist].values[0]
        clst_info[ses]['FP_label'][row] = fullname
        clst_info[ses]['FP_abbrv'][row] = label
        
        # add classic Allen labels
        label = hist[ses]['CCF_abbrv'][which_hist].values[0]
        fullname = hist[ses]['CCF_name'][which_hist].values[0]
        clst_info[ses]['CCF_label'][row] = fullname
        clst_info[ses]['CCF_abbrv'][row] = label

#%% Calculate rasterplots
cntrd_ts = []
all_fr = []; mean_fr = []; sem_fr = []; t_vec = []

for idx, ses in enumerate(spks_ts):
    print('Calculating ' + str(idx+1) + '/' + str(len(spks_ts)))
    
    #tmp = ep.calc_rasters(ses, sniffs[idx]['aligned_onsets'], pre_event, post_event)
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

save_dir = save_plots + 'grand_average_aligned\\'
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
    #tmp = np.argmax(mean_zsc[s][:, which_av], 1) / ifr_sr
    grav.append(tmp)

    nunits = len(spks_id[s])
    tmp2 = np.zeros([nunits ,2])
    for nrn in range(nunits):
        which_chan = int(clst_info[s]['ch'][clst_info[s]['id']==spks_id[s][nrn]])
        tmp2[nrn,:] = chan_pos[s][which_chan,:]
    unit_pos.append(tmp2)
    
#%% Plot responsivity vs. location
rec_pairs = [[1, 3], [2, 4]]

save_dir = save_plots + 'response_locations\\'
#save_dir = save_plots + 'response_latencies\\'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for pair in rec_pairs:
    
    x = np.hstack([unit_pos[pair[0]][:,0], unit_pos[pair[1]][:,0]])
    y = np.hstack([unit_pos[pair[0]][:,1], unit_pos[pair[1]][:,1]])
    val = np.hstack([ grav[pair[0]],  grav[pair[1]]])

    fig_title = sess_ids[pair[0]][-5:]
    
    plt.figure(figsize = (4,8))
    plt.scatter(x, y, val*200)
    plt.title(fig_title)
    
    plt.ylabel('Position dorso-ventral [um]')
    plt.xlabel('Position anterio-posterior [um]')
    
    plt.savefig(save_dir + fig_title + '.png', dpi = 250)


#%% Select neurons with significant responses

roc_win_sec = 3 # window size for ROC significance (in sec, how long before and after stimulus to analyze)
# set to 2 for aligned and 3 to non-aligned
roc_sr = 10 # sampling rate to which downsample neural data, in Hz (otherwise analysis is very slow)

roc_from = int(pre_event*roc_sr - roc_win_sec*roc_sr)
roc_to = int(pre_event*roc_sr + roc_win_sec*roc_sr)

prediction_len  = int((roc_to-roc_from)/2)
prediction = np.concatenate([np.zeros(prediction_len), np.ones(prediction_len)])


ROCV = []
for s in range(nses):
    start = time.time()
    print('Session ' + str(s+1) + ' / ' + str(nses))
    
    ROC_ses = []
    for nrn in range(len(cntrd_ts[s])):
        
        response = all_fr[s][nrn][:,::int(ifr_sr / roc_sr)] # select neuron and downsample
        response = response[:,roc_from:roc_to] # select window for analysis
        
        tmp = ep.checkSignificanceROC_fast(response, prediction, numIters = 1000, sigThr = 0.001)
        ROC_ses.append(tmp)
        
    ROCV.append(ROC_ses)
        
    end = time.time()
    print('Executed in: ' + str(end - start) + ' sec')


#%% Get idea on how many neurons are responsive to how many trials
count_sig = []
for s in range(nses):
    n_cells = len(cntrd_ts[s])
    tmp = np.zeros(n_cells)
    for nrn in range(n_cells):
        tmp[nrn] = np.sum(np.abs(ROCV[s][nrn][:,1]))
    count_sig.append(tmp)

#%% Check odorant selectivity
hab_win = 1 # period after odor presentation used for plotting habituation curves; set 1 for aligned
min_sig_tr = 18 # in how many trials (at least) we need responses to be significant
tr_cat = st.select_trials_nov(sniffs,1,1,1,1)[0]
unique_odors = [np.unique(sniffs[s]['trial_chem_id']) for s in range(nses)]
n_odors = [unique_odors[s].size for s in range(nses)]

save_dir = save_plots + 'odor_selectivity_sig_001_aligned\\'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
plt.ioff()
#plt.tight_layout()

for s in range(nses):
    print('Session ' + str(s+1) + ' from ' + str(nses))
    tmp_dir = save_dir + sess_ids[s] + '\\'
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    
    for nrn in range(len(cntrd_ts[s])):
        
        count_sig = np.sum(np.abs(ROCV[s][nrn][:,1]))
        if count_sig >= min_sig_tr:
        
            fig, axes = plt.subplots(3, 2, sharey = 'row')
            cmap = plt.get_cmap("tab10") 
            a = 0
            
            for cat in range(2):
                ymax = 0
            
                for o, odor in enumerate(unique_odors[s]):
                    which_rows = np.logical_and(sniffs[s]['trial_chem_id'] == odor, tr_cat[s][:,cat]) 
                    odor_idxs = sniffs[s]['trial_idx'][which_rows] - 1
                
                    if odor_idxs.size > 0:
                        tmp = [cntrd_ts[s][nrn][i] for i in odor_idxs]
                        tmp2 = all_fr[s][nrn][odor_idxs,:]
                        ymin = ymax
                        ymax = ymax + len(tmp)  
                        ypos = np.arange(ymin, ymax)
                            
                        axes[0, cat].eventplot(tmp, color = cmap(a), lineoffsets = ypos, linewidths = 0.5)
                        axes[1, cat].plot(t_vec[s], np.mean(tmp2, 0), color = cmap(a), linewidth = 1)
                        tmp3 = np.mean(tmp2[:,int(pre_event*ifr_sr) : int(pre_event*ifr_sr+hab_win*ifr_sr)], 1) # habituation curve
                        axes[2, cat].plot(tmp3, "o-", markersize = 1.5, linewidth = 0.8, color = cmap(a))
                        
                        axes[0, cat].axvline(x = 0, linestyle = '-', color = 'gray', linewidth = 0.5)
                        axes[1, cat].axvline(x = 0, linestyle = '-', color = 'gray', linewidth = 0.5)
                        
                        #### Adjust visuals & set labels #### 
                        axes[0,0].set_title('Familiar odors', size = 8)
                        axes[0,1].set_title('Novel odors', size = 8)
                        
                        ## Two upper rows ##
                        axes[0,cat].tick_params(axis="both",direction="in", labelsize = 6)
                        axes[1,cat].xaxis.set_ticklabels([])
                        axes[1,cat].tick_params(axis="both",direction="in", top = True, labelsize = 6)
                        
                        axes[1,cat].set_xlabel("Time from odor [sec]", size = 6, labelpad = 0)
                        axes[0,0].set_ylabel("Trial", size = 6, labelpad = 0)
                        axes[1,0].set_ylabel("Mean firing rate", size = 6, labelpad = 0)
                        
                        ## Bottom row ##
                        axes[2,cat].tick_params(axis="both",direction="in", labelsize = 6)
                        axes[2,0].set_ylabel("Mean response", size = 6, labelpad = 0)
                        axes[2,cat].set_xlabel("Odor occurence", size = 6, labelpad = 0)
    
                        a = a + 1
            
            axes[0,0].sharex(axes[0,1])
            axes[1,0].sharex(axes[1,1])            
            fig.subplots_adjust(wspace=0.1, hspace=0.15)
            
            unit_id = spks_id[s][nrn]
            unit_abbrv = clst_info[s]['FP_abbrv'][clst_info[s]['id'] == unit_id].values[0]
            unit_label = clst_info[s]['FP_label'][clst_info[s]['id'] == unit_id].values[0]
            shank = clst_info[s]['sh'][clst_info[s]['id'] == unit_id].values[0]
            
            fig.suptitle('sh' + str(shank) + ' , ' + unit_label + ', ' + str(unit_id))
            fig.savefig(tmp_dir + 'sh ' + str(shank) + '_' + unit_abbrv + '_' + str(unit_id) + '.png', dpi = 250)
            plt.close(fig)
        
plt.ion()

#%% Play with novelty preference score

from sklearn.linear_model import LinearRegression
hab_index = []

av_win = [pre_event*ifr_sr, pre_event*ifr_sr+3*ifr_sr]

for s in range(nses):
    #fig, axes = plt.subplots(1, 2, sharex='row', sharey='row')
    fig = plt.figure()
    cmap = plt.get_cmap("tab10") 
    a = 0
    nunits = len(cntrd_ts[s])
    hab_index.append(np.zeros([nunits, 2]))
    
    for nrn in range(nunits):
       
        for cat in range(2):
            first_pres = []
            last_pres = []
            for o, odor in enumerate(unique_odors[s]):
                which_rows = np.logical_and(sniffs[s]['trial_chem_id'] == odor, tr_cat[s][:,cat]) 
                odor_idxs = sniffs[s]['trial_idx'][which_rows] - 1
                if odor_idxs.size > 0:
                    tmp = all_fr[s][nrn][odor_idxs,:]
                    # Calculate average response in 3 first and 3 last trials
                    first_pres.append(np.mean(np.mean(tmp[:3, av_win[0]:av_win[1]], 1)))
                    last_pres.append(np.mean(np.mean(tmp[3:6, av_win[0]:av_win[1]], 1))) 
                    
                    #axes[cat].scatter(last_pres, first_pres, color = cmap(a))
            first_pres=np.array(first_pres)
            last_pres = np.array(last_pres, ndmin = 2).T
            reg = LinearRegression(fit_intercept = False).fit(last_pres, first_pres)
            
            #x = last_pres[:,np.newaxis]
            #y = first_pres
            a, _, _, _ = np.linalg.lstsq(last_pres, first_pres)
            if not reg.coef_ == a:
                print(reg.coef_, a)
            
            if not reg.coef_ == 0:
                hab_index[s][nrn, cat] = reg.coef_ - 1.0
            else:
                hab_index[s][nrn, cat] = np.nan
            
            # # PLot if modulation score is calculated properly 
            # if (s >= 4) and (s <= 5):
            #     plt.figure()
            #     plt.scatter(last_pres, first_pres)
                
            #     plt.xlabel('Response in trials 4-6')
            #     plt.ylabel('Response in trials 1-3')
                
            #     ylim = plt.gca().get_ylim()
            #     xlim = plt.gca().get_xlim()
            #     hmax = np.max([ylim[1], xlim[1]])
            #     plt.ylim(0, hmax)
            #     plt.xlim(0, hmax)
                
            #     plt.plot([0, hmax], [0, hmax])
            #     plt.plot([0, hmax], [0, hmax*(hab_index[s][nrn, cat]+1)])
            #     plt.annotate(str(hab_index[s][nrn, cat]), [0.05, 0])
        
            
    hist_edges = np.arange(-2,2,0.01)
    #plt.hist(hab_index[s][:, 0], hist_edges, histtype='step', alpha = 0.5)
    #plt.figure()
    plt.hist(hab_index[s][:, 0], hist_edges, histtype='step', cumulative = True, color = 'gray', alpha = 0.7)
    plt.hist(hab_index[s][:, 1], hist_edges, histtype='step', cumulative = True, alpha = 0.7)
    plt.legend(['Familiar', 'Novel'])
    
            # axes[cat].scatter(last_pres, first_pres)
            # axes[cat].plot([0, 1], [0, 1], transform=axes[cat].transAxes)

#%%
sal_idxs = [0,1,4,5]
psam_idxs = [2,3,6,7]
hab_sal = []
hab_psam = []
[hab_sal.append(hab_index[s]) for s in sal_idxs]
[hab_psam.append(hab_index[s]) for s in psam_idxs]

hab_sal = np.vstack(hab_sal)
hab_psam = np.vstack(hab_psam)

plt.figure()
plt.hist(hab_sal[:, 1], hist_edges, histtype='step', density = True, cumulative = True)
plt.hist(hab_sal[:, 0], hist_edges, histtype='step', density = True, cumulative = True, color = 'black')

plt.ylabel('Fraction of neurons')
plt.xlabel('Modulation index')
plt.legend(['Novel', 'Familiar'], loc = 'lower right')

plt.axvline([0], linestyle = '--', color = 'gray')
plt.xlim([-1.9, 1.9])
#plt.figure()
#plt.hist(hab_psam[:, 1], hist_edges, histtype='step')



                    