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
    

def fr_events_binless(centered_ts, sigma_sec, trunc_gauss = 4, sampling_rate = 30000.0, sampling_out = 1000, pre_event = 1.0, post_event = 3.0):
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

    """
    
    # Calculate how big is your data etc.
    nunits = len(centered_ts)
    ntrials = len(centered_ts[0])
    nsamples = int(np.round(sampling_out*pre_event + sampling_out*post_event))

    t_vec = np.linspace(-pre_event + 1/sampling_out, post_event, nsamples)
    

    if sampling_out < 1000:
        print("The desired output sampling rate is below 1kHz, which might cause overriding some spikes used "
              "for the computation. Instead, you might consider downsampling the output of this function afterwards.")


    # Create the gaussian window
    sigma = sigma_sec * sampling_out
    
    gaussian = np.arange(-trunc_gauss*sigma, trunc_gauss*sigma + 1)
    gaussian = 1 / (np.sqrt(2 * np.pi) * sigma) * np.e ** (-np.power(gaussian / sigma, 2) / 2) * sampling_out
    #gaussian = np.exp(-(gaussian/sigma)**2/2) # a simpler formula - gives some weird scaling

    
    # Create empty list/arrays for storing results
    all_fr = []
    mean_fr = np.zeros([nunits, nsamples])
    sem_fr = np.zeros([nunits, nsamples])
    
    # Do the firing rate calculation
    for nrn in range(nunits):
        neuron_fr = np.zeros([ntrials, nsamples])
        
        for trl in range(ntrials):
            
            where_spks = centered_ts[nrn][trl] + pre_event
            where_spks = np.array(np.round(where_spks*sampling_out), int) # find spike indices with new sampling rate
            where_spks[where_spks == nsamples] = where_spks[where_spks == nsamples] - 1
    
            neuron_fr[trl, where_spks] = 1 # set those indices in your data array to 1
            neuron_fr[trl, :] = np.convolve(gaussian, neuron_fr[trl, :], 'same')
            
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
    