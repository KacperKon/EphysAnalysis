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
save_plots = 'C:\\Users\\Kacper\\Desktop\\PSAM_SC\\plots_aligned\\'

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
sniffs = st.import_sniff_mat(sniff_dir, expect_files = 4)

#%% Get some basic variables
nses = len(sniffs)
ntrials = sniffs[0]['trial_idx'].size
npres = max(sniffs[0]['trial_occur'])

#%% Calculate rasterplots
cntrd_ts = []
all_fr = []; mean_fr = []; sem_fr = []; t_vec = []

for idx, ses in enumerate(spks_ts):
    print('Calculating ' + str(idx+1) + '/' + str(len(spks_ts)))
    
    tmp = ep.calc_rasters(ses, sniffs[idx]['aligned_onsets'], pre_event, post_event)
    # sniffs[idx]['ephys_onsets'] # for aligned to olfactometer TTL, not 1st sniff
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
    #tmp = np.argmax(mean_zsc[s][:, which_av], 1) / ifr_sr
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
#save_dir = save_plots + 'response_latencies\\'

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


#%% Check odorant selectivity
hab_win = 3 # period after odor presentation used for plotting habituation curves
tr_cat = st.select_trials_nov(sniffs,1,1,1,1)[0]
unique_odors = [np.unique(sniffs[s]['trial_chem_id']) for s in range(nses)]
n_odors = [unique_odors[s].size for s in range(nses)]

save_dir = save_plots + 'odor_selectivity_fr\\'
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
                    tmp3 = np.mean(tmp2[:,pre_event*ifr_sr : pre_event*ifr_sr+hab_win*ifr_sr], 1) # habituation curve
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
        
        fig.savefig(tmp_dir + str(spks_id[s][nrn]) + '.png', dpi = 250)
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



                    