# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 17:23:58 2022

@author: Kacper
"""
# Easy jobs:
    # check if data looks better with spikes aligned
    # re-write the code to use pre-extracted indexes:
        # tr_idx[session][trial_type][trial_indx]
        # Select novel: spikes[s][nrn][tr_idx[s][1][:], :]
        # Select novel, only 1st pres: tr_idx[s][1][0::n_pres[1]]
        # Both fam and novel, last pres: np.vstack([ tr_idx[s][1][n_pres[1]::n_pres[1]], tr_idx[s][0][n_pres[0]::n_pres[0]] ])
    # so add also lists of int: n_pres[s][len=3], n_odors[s][len=3] 
    # quantitative comparison - fr by depth
    # Separate file for preprocess, load data with command
    # Exclude some neurons (Jennifer Colonel)
    # So new version of import

# Difficult jobs:
    # Modulation index 
        # calculate mean firing rates for each trial: 3 sec before and 3 sec after odor
            # but only fr after odor seems to be used
        # for each odor, calculate mean fr for presentations 1-3, 4-6 etc.
        # means from all novel OR fam will be used as resp(1-3) and base(4-6)
            # so modulation index for novel is calculated based on 12 numbers in your case:
            # mean fr during first pres of 6 odors vs. the same for next presentations
            
        # for calulating significance:
            # just Wilcoxon rank sum test on first vs. next firing rates
            # for calculating mod index - slope of regression line constrained on 1,0 OR 0,0
        
    # Select odor-cell pairs:
        # t-test?
        # or ROC curve
        
    # Lifetime sparseness

# Annoying jobs:
    # Can I do tests??
    # trim the gaussian window
    # Align to 1st sniff