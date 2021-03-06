# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 20:29:05 2021

A package for basic spike-trains analysis, developed in Emotions Neurobioloy Lab.
Required data format: output files from Phy2 spike sorting software.  

@authors: Kacper Kondrakiewcz, Ludwika Kondrakiewicz
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from numba import jit
import random

#%%
def read_spikes(spks_dir, sampling_rate = 30000.0, read_only = "good"):
    """
    A function to read spike times as saved by Phy2.
    
    Arguments:
        - spks_dir: directory with spike times and sorting results (a string)
        - sampling_rate: default = 30k
        - rad_only = which cluster type to read ("good" for single units, "mua" for multi unit or "noise" for shit)
    Returns: 
        - spks_ts: a list of numpy arrays, where all timestamps in each array are from one neuron
        - units_id: a list of units numbers (neuron id in Phy)
    
    """
    
    spks = np.load(spks_dir + "spike_times.npy")
    clst = np.load(spks_dir + "spike_clusters.npy")
    clst_group = pd.read_csv(spks_dir + "cluster_group.tsv", sep='\t')
    
    units_id = np.array(clst_group.cluster_id[clst_group.group == read_only]) # those clusters contain selected type of units
    spks = 1/sampling_rate * spks # convert sample numbers to time stamps

    spks_ts = []
    for nrn in units_id:
        spks_ts.append(spks[clst==nrn])
        
    return spks_ts, units_id

        

def calc_rasters(spks_ts, events_ts, pre_event = 1.0, post_event = 3.0):
    """
    A function for centering spikes' timestamps on events' timestamps.
    Returns data which is ready for creating raster plots.
    
    Arguments:
        - spks_ts: list of arrays with neuron timestamps (each array is one neuron)
        - array/list of event timestamps 
        - integers defining the window size (max time before and after event)
    
    Returns: list of lists of centered arrays, where:
        - each array contains all timestamps of single neuron during single trial (event)
        - a list groups data from all trials for single neuron
        - a higher-order list (output) groups data from all neurons
    """
    centered_ts = []
    
    for nrn in spks_ts:
        neuron_ts = []
        for evt in events_ts:
            pos = nrn - evt
            in_window = np.logical_and(pos >= -pre_event, pos <= post_event)
            pos = pos[in_window]
            neuron_ts.append(pos)
        centered_ts.append(neuron_ts)
        
    return centered_ts



def plot_rasters(centered_ts):
    """
    A function for plotting raster plots based on centered timestamps.
    Returns a figure with each neuron plotted as a separate subplot.

    Arguments:
        - centered_ts : the output of calc_rasters function (list of lists of arrays)

    Returns:
        - fig: a matplotlib figure handle with individual eventplots for each neuron
        - axes: if I get it, handles of individual subplots (flattened - only 1 indexes is needed)

    """
    nunits = len(centered_ts)
    nrows = int(round(np.sqrt(nunits) + 0.5)) # assume same no of rows and columns - not efficient
    fig, axes = plt.subplots(nrows, nrows, sharex = 'col')
    axes = axes.flatten()
    for nrn in range(nunits):
        axes[nrn].eventplot(centered_ts[nrn])
    
    return fig, axes



def fr_events(centered_ts, bin_size, pre_event = 1.0, post_event = 3.0):
    """
    A function that calculates firing rates in trials.
    
    Arguments:
    - centered_ts : list of lists of arrays (otuput of calc_raster)
    - bin_size : integer (bin size in seconds)
    - pre_event : integer (length of window before event, same as for calc_raster)
        The default is 1.0.
    - post_event : integer (length of window after event, same as for calc_raster)
        The default is 3.0.
    
    Returns:
    - all_fr: list of length n neurons; each item contains an array of size n trials x n time bins
            and contains the calculated firing rate
    mean_fr : array of size n neurons x n time bins for storing mean fr (across trials)
    sem_fr : array of size n neurons x n time bins for storing standard error (across trials)

    """
    # Calculate how big is your data and bin edges (in sec)
    nunits = len(centered_ts)
    ntrials = len(centered_ts[0])
    bin_edges = np.arange(pre_event *(-1), post_event + bin_size, bin_size)
    nbins = bin_edges.size - 1
    
    # Create empty list/arrays for storing results
    all_fr = []
    mean_fr = np.zeros([nunits, nbins])
    sem_fr = np.zeros([nunits, nbins])
    
    # Do the firing rate calculation
    for nrn in range(nunits):
        neuron_fr = np.zeros([ntrials, nbins])
        
        for trl in range(ntrials):
            spks_in_bins = np.histogram(centered_ts[nrn][trl], bin_edges)
            fr_in_bins = spks_in_bins[0]*1 / bin_size
            neuron_fr[trl, :] = fr_in_bins[:]
            
        all_fr.append(neuron_fr)
        mean_fr[nrn,:] = np.mean(neuron_fr, 0)
        sem_fr[nrn,:] = np.std(neuron_fr, 0) / np.sqrt(ntrials)
    
    return all_fr, mean_fr, sem_fr, bin_edges
    

def fr_events_binless(centered_ts, sigma_sec, trunc_gauss = 4, sampling_out = 1000, pre_event = 1.0, post_event = 3.0):
    """
    A function that calculates firing rates in trials by applying gaussian kernel (binless).
    
    Arguments:
    - centered_ts : list of lists of arrays (otuput of calc_raster)
    - bin_size : integer (bin size in seconds)
    - pre_event : integer (length of window before event, same as for calc_raster)
        The default is 1.0.
    - post_event : integer (length of window after event, same as for calc_raster)
        The default is 3.0.
    
    Returns:
    - all_fr: list of length n neurons; each item contains an array of size n trials x n time bins
            and contains the calculated firing rate
    mean_fr : array of size n neurons x n time bins for storing mean fr (across trials)
    sem_fr : array of size n neurons x n time bins for storing standard error (across trials)
    
    Note: In the current version, the function might cause problems if 1000 cannot be evenly divided
        by selected sampling_out.

    """
    
    # Calculate how big is your data etc.
    nunits = len(centered_ts)
    ntrials = len(centered_ts[0])
    nsamples = int(np.round(sampling_out*pre_event + sampling_out*post_event))
    t_vec = np.linspace(-pre_event + 1/sampling_out, post_event, nsamples)
    
    # Create empty list/arrays for storing results
    all_fr = []
    mean_fr = np.zeros([nunits, nsamples], dtype='f')
    sem_fr = np.zeros([nunits, nsamples], dtype='f')
    
    # If desired out sampling is lower than 1kHz, avoid overwriting spikes 
    if sampling_out < 1000:
        downsample = True
        sampling_out_ds = sampling_out
        sampling_out = 1000
        nsamples = int(np.round(sampling_out*pre_event + sampling_out*post_event))
        ds_by = int(sampling_out / sampling_out_ds)
        
        nsamples_ds = int(nsamples / ds_by)
        t_vec = np.linspace(-pre_event + 1/sampling_out_ds, post_event, nsamples_ds)
        
        mean_fr = np.zeros([nunits, nsamples_ds], dtype='f')
        sem_fr = np.zeros([nunits, nsamples_ds], dtype='f')
        
        print("Calculation performed with 1kHz sampling rate, but the final "  
              "output will be downsampled by the factor of", str(ds_by))
    
    else:
        downsample = False
    
    # Create the gaussian window
    sigma = sigma_sec * sampling_out
    halfwidth = trunc_gauss*sigma # half-width of gaussian window - full width is halfwidth * 2 + 1
    
    gaussian = np.arange(-halfwidth, halfwidth + 1)
    gaussian = 1 / (np.sqrt(2 * np.pi) * sigma) * np.e ** (-np.power(gaussian / sigma, 2) / 2) * sampling_out
    #gaussian = np.exp(-(gaussian/sigma)**2/2) # a simpler formula - gives some weird scaling

    # Do the firing rate calculation (convolve binarized spikes with the gaussian)
    for nrn in range(nunits):
        neuron_fr = np.zeros([ntrials, nsamples], dtype='f')
        
        for trl in range(ntrials):
            
            where_spks = centered_ts[nrn][trl] + pre_event
            where_spks = np.array(np.round(where_spks*sampling_out), int) # find spike indices with the new sampling rate
            where_spks[where_spks == nsamples] = where_spks[where_spks == nsamples] - 1 # avoid rounding timestamps to indices bigger than data length
    
            neuron_fr[trl, where_spks] = 1 # code spikes as 1
            #neuron_fr[trl, :] = np.convolve(gaussian, neuron_fr[trl, :], 'same') # do the convoloution
            neuron_fr[trl, :] = gaussian_filter1d(neuron_fr[trl, :], sigma, mode = 'reflect') * sampling_out
            
        if downsample == True:
            neuron_fr = neuron_fr[:,::ds_by]
    
        all_fr.append(neuron_fr)
        mean_fr[nrn,:] = np.mean(neuron_fr, 0)
        sem_fr[nrn,:] = np.std(neuron_fr, 0) / np.sqrt(ntrials)
        
    return all_fr, mean_fr, sem_fr, t_vec


def zscore_events(all_fr, bin_size, pre_event = 1.0, post_event = 3.0):
    """
    A function that calculates z-score in trials - where baseline is separate for each trial.
    
    Arguments:
    - all_fr : list of lists of arrays (otuput of fr_events)
    - bin_size : integer (bin size in seconds)
    - pre_event : integer (length of window before event, same as for calc_raster)
        The default is 1.0.
    - post_event : integer (length of window after event, same as for calc_raster)
        The default is 3.0.
    
    Returns:
    - all_zsc: list of length n neurons; each item contains an array of size n trials x n time bins
            and contains the calculated firing rate
    mean_zsc : array of size n neurons x n time bins for storing mean fr (across trials)
    sem_zsc : array of size n neurons x n time bins for storing standard error (across trials)

    """
    
    # Calculate how big is your data and bin edges (in sec)
    nunits = len(all_fr)
    ntrials = all_fr[0].shape[0]
    bin_edges = np.arange(pre_event *(-1), post_event + bin_size, bin_size)
    nbins = bin_edges.size - 1
    nbins_pre = int(pre_event/bin_size)
    
    # Create empty list/arrays for storing results
    all_zsc = []
    mean_zsc = np.zeros([nunits, nbins])
    sem_zsc = np.zeros([nunits, nbins])
    
    # Do the z-score calculation
    for nrn in range(nunits):
        neuron_zsc = np.ones([ntrials, nbins])
        
        baseline_mean = np.mean(all_fr[nrn][:,0:nbins_pre], 1)
        baseline_std  = np.std(all_fr[nrn][:,0:nbins_pre], 1)
        # !!! What to do if standard deviation for a given bin == 0 (it's empty) or really small?
        # -> Current solution: set these values manually to 1
        baseline_std[baseline_std < 0.1] = 1
        
        for trl in range(ntrials):
            zsc_in_bins = (all_fr[nrn][trl,:] - baseline_mean[trl]) / baseline_std[trl] 
            neuron_zsc[trl, :] = zsc_in_bins[:]
            
        all_zsc.append(neuron_zsc)
        mean_zsc[nrn,:] = np.mean(neuron_zsc, 0)
        sem_zsc[nrn,:] = np.std(neuron_zsc, 0) / np.sqrt(ntrials)
    
    return all_zsc, mean_zsc, sem_zsc, bin_edges



def plot_psth(mean_data, sem_data, t_vec):
    """
    A function for plotting peri-stimulus time histograms.
    Returns a figure with each neuron plotted on a separate subplot.

    Arguments: outputs of functions zscore_events or fr_events
        - mean_data : 2d array, mean responses of all neurons in a given time bin
        - sem_data : 2d array, SEM of responses of all neurons in a given time bin
        - t_vec : 1d array, values to display on x axis (if using bin_edges, set to bin_edges[1:])

    Returns:
        - fig: a matplotlib figure handle with individual eventplots for each neuron
        - axes: if I get it, handles of individual subplots (flattened - only 1 indexes is needed)

    """
    
    nunits = mean_data.shape[0]
    
    nrows = int(round(np.sqrt(nunits) + 0.5)) # assume same no of rows and columns - not efficient
    fig, axes = plt.subplots(nrows, nrows, sharex = 'col')
    axes = axes.flatten()
    
    for nrn in range(nunits):
        axes[nrn].plot(t_vec, mean_data[nrn,:])
        y1 = mean_data[nrn,:] + sem_data[nrn,:]
        y2 = mean_data[nrn,:] - sem_data[nrn,:]
        axes[nrn].fill_between(t_vec, y1, y2, alpha=0.5, zorder=2)
        axes[nrn].axvline(x = 0, linestyle = '--', color = 'gray', linewidth = 1)

    return fig, axes


def plot_responses(centered_ts, mean_data, sem_data, t_vec, save_dir, units_id = None, event_label = 'event'):
    
    """
    A function to plot raster & PSTH for each neuron and save as separate plots.   
    
    """
    
    # Turn off automatic display - we want to save plots as images, not show them
    plt.ioff() 
    
    nunits = len(centered_ts)
    if units_id == None:
        units_id = np.arange(nunits)
    
    for nrn in range(nunits):
        fig, axes = plt.subplots(2, 1, sharex = 'col')
        axes = axes.flatten()
        
        axes[0].eventplot(centered_ts[nrn])
        axes[0].set_ylabel('Trial #')
    
        axes[1].plot(t_vec, mean_data[nrn,:])
        y1 = mean_data[nrn,:] + sem_data[nrn,:]
        y2 = mean_data[nrn,:] - sem_data[nrn,:]
        axes[1].fill_between(t_vec, y1, y2, alpha=0.5, zorder=2)
        axes[1].axvline(x = 0, linestyle = '--', color = 'gray', linewidth = 1)
        axes[1].set_ylabel('Mean activity')

        
        fig.suptitle('Responses to ' + event_label)
        plt.xlabel('Time [sec]')
        
        fig.savefig(save_dir + str(units_id[nrn]) + '.png')
        plt.close(fig)
        
    plt.ion()
   
    
def plot_resp_zsc(centered_ts, mean_data, sem_data, mean_data2, sem_data2, t_vec, save_dir, units_id = None, event_label = 'event'):
    
    """
    A function to plot raster and 2 PSTHs (eg., raw and zscored) for each neuron and save as separate plots.   
    
    """
    
    # Turn off automatic display - we want to save plots as images, not show them
    plt.ioff() 
    
    nunits = len(centered_ts)
    if units_id == None:
        units_id = np.arange(nunits)
    
    for nrn in range(nunits):
        fig, axes = plt.subplots(3, 1, sharex = 'col')
        axes = axes.flatten()
        
        axes[0].eventplot(centered_ts[nrn])
        axes[0].set_ylabel('Trial #')
    
        axes[1].plot(t_vec, mean_data[nrn,:])
        y1 = mean_data[nrn,:] + sem_data[nrn,:]
        y2 = mean_data[nrn,:] - sem_data[nrn,:]
        axes[1].fill_between(t_vec, y1, y2, alpha=0.5, zorder=2)
        axes[1].axvline(x = 0, linestyle = '--', color = 'gray', linewidth = 1)
        axes[1].set_ylabel('Firing rate (Hz)')
        
        axes[2].plot(t_vec, mean_data2[nrn,:])
        y1 = mean_data2[nrn,:] + sem_data2[nrn,:]
        y2 = mean_data2[nrn,:] - sem_data2[nrn,:]
        axes[2].fill_between(t_vec, y1, y2, alpha=0.5, zorder=2)
        axes[2].axvline(x = 0, linestyle = '--', color = 'gray', linewidth = 1)
        axes[2].set_ylabel('Z-score')

        fig.suptitle('Responses to ' + event_label)
        plt.xlabel('Time [sec]')
        
        fig.savefig(save_dir + str(units_id[nrn]) + '.png')
        plt.close(fig)
        
    plt.ion() 


def fr_by_chan(spikes_ts, units_id, cluster_info_dir, bin_size):
    """
    Calculate overall firing rate per channel in specific time bins.
    

    Parameters
    ----------
    spikes_ts : list of arrays with spike times
    units_id : cluster id
    cluster_info_dir : path to cluster_info.tsv file
    bin_size : size of bin (in seconds)

    Returns
    -------
    all_fr : matrix size n channels (assumes 384) per n time bins
    mean_fr: vector of size n channels, with mean firing rate
    bin_edges : bin egdes in seconds
    """
    
    
    clst_info = pd.read_csv(cluster_info_dir + "cluster_info.tsv", sep='\t')

    nchans = 383
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
    mean_fr = np.mean(all_fr, 1)

    return all_fr, mean_fr, bin_edges


@jit
def checkSignificanceROC_fast(responseVector,predictionVector, numIters = 1000, sigThr = 0.05):
    """
    Verify is response to stimulus in each trial is significantly higher or 
    lower than random, based on area under ROC curve.
    Based on function written by Eleonore Schiltz, Haesler Lab 2022.
    
    *** 
    This is a simplified version using Numba, ~100 times faster than the original.
    The significance results were for ~98% trials consistent with the original
    implementation, based on roc_auc_score function from the sci-kit learn.
    However, exact values of ROC area under curve might differ.
    ***
    
    Parameters
    ----------
    responseVector : firing rate; array of size n trials x n time points
    predictionVector : is stimulus present?; binary array of size n time points
    numIters : number of permutations for significance testing; integer
    sigThr : p-value threshold (2-sided) for testing against random scores; float

    Returns
    -------
    ROCV : 2-dim array, of size n trials x 2:
        - column 0 - auROC score
        - column 1 - result of significance testing (-1: lower than baseline, 
            +1: higher, 0: non-significant)
    """
        
    def auc(y_true, y_score):
        """Simplified implementation to check for correctness of
        `roc_auc_score`, copied from sci-kit learn metrics/_ranking.py"""
        pos_label = np.unique(y_true)[1]

        # Count the number of times positive samples are correctly ranked above
        # negative samples.
        pos = y_score[y_true == pos_label]
        neg = y_score[y_true != pos_label]
        diff_matrix = pos.reshape(1, -1) - neg.reshape(-1, 1)
        n_correct = np.sum(diff_matrix > 0)

        return n_correct / float(len(pos) * len(neg))
        
    
    ROCV = np.zeros((len(responseVector),2))
    for ind in range(len(responseVector)):
        ROCV[ind,0] = auc(predictionVector,responseVector[ind])
        randomValue = np.zeros(numIters)
        for rand in range(numIters):
            randomValue[rand]=auc(np.random.permutation(predictionVector),responseVector[ind])
        if ROCV[ind,0]<np.percentile(randomValue,sigThr*100):
            ROCV[ind,1] = -1 #inhibited
        elif ROCV[ind,0]>np.percentile(randomValue,(1-sigThr)*100):
            ROCV[ind,1] = 1 #excited
        else:
            ROCV[ind,1] = 0 #non significant
    return ROCV