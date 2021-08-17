# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 16:34:14 2021

@author: kkondrakiewicz
"""
#%%
import sys
sys.path.append(r'C:\Users\kkondrakiewicz\Documents\Python Scripts\EphysAnalysis')
import ephys as ep
import numpy as np
import time

#%%
spks_dir = "D:\\buffering_np\\NP2_test\\"
sounds = np.loadtxt("D:\\buffering_np\\NP2_test\\TTL_gcat\\TTL_3.txt")

pre_event = 2.0
post_event = 5.0
bin_size = 0.250


start = time.time()
spajki, units_id = ep.read_spikes(spks_dir)
end = time.time()
print("Read spikes: ", end - start)

start = time.time()
centered_ts = ep.calc_rasters(spajki, sounds, pre_event, post_event)
end = time.time()
print("Calculate rasters: ", end - start)

start = time.time()
fig, axes = ep.plot_rasters(centered_ts)
fig.suptitle('Single neurons - responses to event')
#axes[nrn].set_title(str(units_id[nrn]), fontsize=8, fontweight='bold')
end = time.time()
print("Plot rasters: ", end - start)

start = time.time()
all_fr, mean_fr, sem_fr, bin_edges = ep.fr_events(centered_ts, bin_size, pre_event, post_event)
end = time.time()
print("Calculate firing rates: ", end - start)

#%%
start = time.time()
all_zsc, mean_zsc, sem_zsc, bin_edges = ep.zscore_events(all_fr, bin_size, pre_event, post_event)
end = time.time()
print("Calculate z-scores: ", end - start)


neuron = 5
start = time.time()
fig, axes = ep.plot_psth(mean_zsc, sem_zsc, bin_edges)
end = time.time()
print("Plot z-scores: ", end - start)

#%%
start = time.time()
save_dir = 'C:\\Users\\kkondrakiewicz\\Desktop\\ploty\\'
ep.plot_responses(centered_ts, mean_zsc, sem_zsc, bin_edges, save_dir)
end = time.time()
print("Saving plots: ", end - start)

