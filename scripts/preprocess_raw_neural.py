""" Author: Benyamin Meschede-Krasa
preprocess raw neural features to compute spike band power and threshold crossings (-4RMS)
This code is based on [`preprocess_ns5_utils.py`](https://code.stanford.edu/fwillett/nptlrig2/-/blob/nick_dev/Analysis/nhahn7/speech-neural-dynamics/preprocess_ns5_utils.py?ref_type=heads) which was written by Nick Card and Nick Hahn. It has been updated for Gemini Hubs.
"""
import numpy as np
import scipy.signal
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns

from brpylib import NsxFile

def read_ns5_file(ns5_filename, n_channels):
    """
    Read NS5 file and extract raw voltage data.

    Parameters
    ----------
    ns5_filename : str
        Path to the NS5 file.
    n_channels : int
        Number of channels to extract.

    Returns
    -------
    np.ndarray
        Raw voltage data with shape (n_samples, n_channels).
    """
    nsx_file = NsxFile(ns5_filename)

    all_dat = nsx_file.getdata('all', 0, full_timestamps=True)  # electrode ids and start time s
    # TODO add extra lines of code to get time stamps and return time stamp of each bin. Do this by changing the full_timestamps=True parameter. 
    # 
    nsx_file.close()
    
    raw_voltage = np.hstack(all_dat['data'])[:n_channels, :]  # data sometimes chunked across segments. extract only first n_channel channels    
    return raw_voltage.T


def build_filter(filt_order, lo_cut, hi_cut, fs):
    """
    Build a bandpass filter.

    Parameters
    ----------
    filt_order : int
        Order of the Butterworth filter.
    lo_cut : float
        Lower cutoff frequency.
    hi_cut : float
        Upper cutoff frequency.
    fs : float
        Sampling frequency.

    Returns
    -------
    tuple
        Filter coefficients (b, a).
    """
    # Get filter parameters for non-causal Butterworth filter
    b, a = scipy.signal.butter(filt_order, [lo_cut, hi_cut], btype='bandpass', 
                               analog=False, output='ba', fs=fs)
        
    return b, a


def car(dat, n_arrays, n_channels):
    """
    Apply common average referencing (CAR) to data.

    Parameters
    ----------
    dat : np.ndarray
        Raw data with shape (n_samples, n_channels).
    n_arrays : int
        Number of arrays.
    n_channels : int
        Total number of channels.

    Returns
    -------
    np.ndarray
        CAR-referenced data.
    """
    n_electrodes = n_channels // n_arrays  # electrodes per array

    for array in range(n_arrays):
        this_array = np.arange(n_electrodes * array, n_electrodes * (array + 1), dtype='int16')
        this_mean = np.mean(dat[:, this_array], axis=1)
        dat[:, this_array] = dat[:, this_array] - this_mean[:, np.newaxis]

    return dat


def getLRRWeights(dat, fs, max_seconds):
    """
    Compute linear regression referencing (LRR) weights.

    Parameters
    ----------
    dat : np.ndarray
        Raw data with shape (n_samples, n_channels).
    fs : float
        Sampling frequency.
    max_seconds : int
        Maximum number of seconds to use for LRR computation.

    Returns
    -------
    np.ndarray
        LRR weight matrix.
    """
    these_chans = np.arange(dat.shape[1])
    ref_mat = np.zeros((these_chans.size, these_chans.size))  # weight matrix
    max_idx = max_seconds * fs
    up_to_idx = min(max_idx, dat.shape[0])
    rand_idx = np.random.permutation(np.arange(dat.shape[0]))
    use_idx = rand_idx[:up_to_idx]
    subsample_data = dat[use_idx, :]
    for chan_idx in these_chans:
        pred_idx = np.setdiff1d(these_chans, these_chans[chan_idx])
        filts = np.linalg.lstsq(subsample_data[:, pred_idx], subsample_data[:, chan_idx], rcond=None)[0]
        ref_mat[chan_idx, np.setdiff1d(np.arange(0, these_chans.size), chan_idx)] = filts
    return ref_mat


def lrr(dat, n_arrays, n_channels, fs, max_seconds = 45):
    """
    Apply linear regression referencing (LRR) to data.

    Parameters
    ----------
    dat : np.ndarray
        Raw data with shape (n_samples, n_channels).
    n_arrays : int
        Number of arrays.
    n_channels : int
        Total number of channels.
    fs : float
        Sampling frequency.

    Returns
    -------
    tuple
        LRR-referenced data and LRR weights.
    """
    lrr_weights = []
    n_electrodes = n_channels // n_arrays  # electrodes per array
    for array in range(n_arrays):
        print(f'starting LRR for array {array + 1} of {n_arrays}')
        this_array = np.arange(n_electrodes * array, n_electrodes * (array + 1), dtype='int16')
        this_array_coeffs = getLRRWeights(dat[:, this_array], fs, max_seconds)
        dat[:, this_array] = dat[:, this_array] - np.dot(dat[:, this_array], this_array_coeffs)
        lrr_weights.append(this_array_coeffs)
    return dat, lrr_weights


def unscrambleChans(dat):
    """
    Unscramble T12 6v channels.

    Parameters
    ----------
    dat : np.ndarray
        Raw data with shape (n_samples, n_channels).

    Returns
    -------
    np.ndarray
        Unscrambled data.
    """
    chanToElec = [63, 64, 62, 61, 59, 58, 60, 54, 57, 50, 53, 49, 52, 45, 55, 44, 56, 39, 51, 43,
                  46, 38, 48, 37, 47, 36, 42, 35, 41, 34, 40, 33, 96, 90, 95, 89, 94, 88, 93, 87,
                  92, 82, 86, 81, 91, 77, 85, 83, 84, 78, 80, 73, 79, 74, 75, 76, 71, 72, 68, 69,
                  66, 70, 65, 67, 128, 120, 127, 119, 126, 118, 125, 117, 124, 116, 123, 115, 122, 114, 121, 113,
                  112, 111, 109, 110, 107, 108, 106, 105, 104, 103, 102, 101, 100, 99, 97, 98, 32, 30, 31, 29,
                  28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 16, 17, 7, 15, 6, 14, 5, 13, 4, 12, 3, 11, 2, 10, 1, 9, 8]
    chanToElec = np.array(chanToElec).astype(np.int32) - 1
    
    unscrambledDat = dat.copy()
    for x in range(len(chanToElec)):
        unscrambledDat[:, chanToElec[x]] = dat[:, x]
        
    return unscrambledDat


def get_thresholds(dat, thresh_mult):
    """
    Compute thresholds for data based on standard deviation.

    Parameters
    ----------
    dat : np.ndarray
        Input data with shape (n_samples, n_channels).
    thresh_mult : float
        Multiplier for the standard deviation to compute the threshold.

    Returns
    -------
    np.ndarray
        Computed thresholds for each channel.
    """
    std = np.std(dat, axis=0)
    threshold = thresh_mult * std  # threshold multiplier is -4.5
    return threshold


def get_spike_bandpower(dat, clip_thresh, fs):
    """
    Compute spike band power for data.

    Parameters
    ----------
    dat : np.ndarray
        Filtered data with shape (n_samples, n_channels).
    clip_thresh : float
        Threshold for clipping spike power.
    fs : float
        Sampling frequency.

    Returns
    -------
    np.ndarray
        Spike band power computed for each 1 ms window.
    """
    # Extract spike power for every 1ms
    n_samples = int(fs / 1000)  # number of samples in each 1ms window
    spike_pow = []
    
    for i in range(0, dat.shape[0], n_samples):
        dat_1ms = dat[i:i+n_samples, :]
        
        # up the dtype to avoid overflow during squaring
        dat_1ms = np.array(dat_1ms, dtype='float32') 

        # get spike power by squaring, check the axis
        sp_pow = np.mean(np.square(dat_1ms), axis=0)

        # clip max spike power to a pre-set threshold (default 4000)
        sp_pow[sp_pow > clip_thresh] = clip_thresh
        
        spike_pow.append(sp_pow)
    
    spike_pow = np.array(spike_pow)
    return spike_pow


def compute_threshold_crossings(dat, threshold):
    """
    Compute threshold crossings for data.

    Parameters
    ----------
    dat : np.ndarray
        Input data with shape (n_samples, n_channels).
    threshold : np.ndarray
        Threshold values for each channel.

    Returns
    -------
    np.ndarray
        Binary array indicating threshold crossings.
    """
    crossings = (dat > threshold).astype(int)
    return crossings

def bin_data(dat, bin_size_ms, shift_size_ms, bin_type):
    # Input data is at 1000Hz resolution
    binned_dat = []
    
    for i in range(0, dat.shape[0], shift_size_ms):
        
        dat_for_bin = dat[i:i+bin_size_ms,:]
        
        if bin_type == 0: # bin_type == 0 for averaging the values in the bin
            bin_dat = np.mean(dat_for_bin, axis=0)
            
        elif bin_type == 1: # sum values in the bin, used for getting threshold crossing counts
            bin_dat = np.sum(dat_for_bin, axis=0)
        
        bin_dat = np.array(bin_dat, dtype='float32')
        
        binned_dat.append(bin_dat)
        
    return np.array(binned_dat)

def get_threshold_crossings(dat, threshold, fs):

    # Extract threshold crossing for every 1ms
    n_samples = int(fs/1000) # number of samples in each 1ms window
    threshold_crossings = []

    for i in range(0, dat.shape[0], n_samples):
        
        dat_1ms = dat[i:i+n_samples,:]
    
        # get bool values for spikes in all channels
        thresh_cross = np.min(dat_1ms,axis=0) <= threshold

        # convert bools to 0 and 1
        thresh_cross = np.multiply(thresh_cross,1)

        thresh_cross = np.array(thresh_cross, dtype='int16')

        threshold_crossings.append(thresh_cross)
        
    return np.array(threshold_crossings)


def bin_data(dat, bin_size_ms, shift_size_ms, bin_type):
    # Input data is at 1000Hz resolution
    binned_dat = []
    
    for i in range(0, dat.shape[0], shift_size_ms):
        
        dat_for_bin = dat[i:i+bin_size_ms,:]
        
        if bin_type == 0: # bin_type == 0 for averaging the values in the bin
            bin_dat = np.mean(dat_for_bin, axis=0)
            
        elif bin_type == 1: # sum values in the bin, used for getting threshold crossing counts
            bin_dat = np.sum(dat_for_bin, axis=0)
        
        bin_dat = np.array(bin_dat, dtype='float32')
        
        binned_dat.append(bin_dat)
        
    return np.array(binned_dat)
       

def get_mean_std(dat):
    # Get block mean and standard deviation
    
    dat_mean = np.mean(dat,axis=0)
    dat_std = np.std(dat,axis=0)
    
    dat_mean = np.array(dat_mean, dtype='float64') # float64 is required for writing to json
    dat_std = np.array(dat_std, dtype='float64')
    
    return dat_mean, dat_std

def compute_neural_features(ns5_filepath, out_filepath,
                            unscramble_vPCG=False,
                            denoise_mode='LRR',
                            max_seconds=45,  # LRR
                            n_arrays=4,
                            n_channels=256,
                            bin_size_ms=20,
                            shift_size_ms=20,  # Non-overlapping bins
                            threshold=-4.0,  # multiplied by standard dev
                            filt_order=4,  # order of the butterworth filter
                            lo_cut=250,  # lower cutoff frequency
                            hi_cut=5000,  # upper cutoff frequency
                            fs=30000,  # Sampling frequency
                            clip_thresh_sp=10000,  # threshold for clipping spike band power
                            save_plot = True, # save plot of spikepow and threshold crossings for sanity checking
                            verbose=True):
    """
    Compute neural features from raw NS5 data from gemini hubs 
    and save the results to a .mat file containing 20ms binned neural features
    of threshold crossings and spike band power. LRR thresholds computed over this block
    so this is non-causal.
    The output is similar to redis mats that are often computed online
    

    Parameters
    ----------
    ns5_filepath : str
        Path to the NS5 file containing neural data.
        For gemini this file name should start with 'Hub1-` not 'NSP`
    out_filepath : str
        Path to save the output .mat file with computed neural features
    unscramble_vPCG : bool, optional
        Flag to unscramble 6v channels to make them index 0-64 and 64-128 for 
        individual arrays, by default False.
    denoise_mode : str, optional
        Denoising mode, 'CAR' for common average referencing or 'LRR' for linear regression referencing, by default 'LRR'.
    max_seconds : int, optional
        Maximum number of seconds to use for LRR computation, by default 45.
    n_arrays : int, optional
        Number of arrays, by default 4.
    n_channels : int, optional
        Total number of channels, by default 256.
    bin_size_ms : int, optional
        Size of the bins in milliseconds, by default 20.
    shift_size_ms : int, optional
        Size of the shifts in milliseconds, by default 20 so no overlap.
    threshold : float, optional
        Multiplier for the standard deviation to compute the threshold, by default -4.0.
    filt_order : int, optional
        Order of the Butterworth filter, by default 4.
    lo_cut : float, optional
        Lower cutoff frequency for the bandpass filter, by default 250.
    hi_cut : float, optional
        Upper cutoff frequency for the bandpass filter, by default 5000.
    fs : float, optional
        Sampling frequency, by default 30000.
    clip_thresh_sp : float, optional
        Threshold for clipping spike band power, by default 10000.
    verbose : bool, optional
        Flag to enable verbose output, by default True.

    Returns
    -------
    None
    """
    raw_neural = read_ns5_file(ns5_filepath, n_channels)
    
    if unscramble_vPCG:
        if verbose: print('Unscrambling 6v Channels')
        raw_neural = unscrambleChans(raw_neural)
    if verbose:
        print('\n')
        print('1/8 Raw data loading complete ', raw_neural.shape)
    
    # 2. Filter the raw data
    filt = raw_neural.copy()  # np.zeros(raw_neural.shape)
    b, a = build_filter(filt_order, lo_cut, hi_cut, fs)
    
    for ch in range(raw_neural.shape[-1]):
        filt[:, ch] = scipy.signal.filtfilt(b, a, filt[:, ch])  # apply filter to each channel (columnwise filtering) TODO: vectorise
    del raw_neural # raw_neural can be very and we're done with it so lets remove it from memory
    if verbose: print('2/8 Bandpass filtering complete ', filt.shape)
    
    # 3. Common Average Referencing or Linear Regression Referencing
    if denoise_mode == 'CAR':
        filt = car(filt, n_arrays, n_channels)
        if verbose: print('3/8 Common average referencing complete ', filt.shape)
    elif denoise_mode == 'LRR':
        filt, lrr_coeffs = lrr(filt, n_arrays, n_channels, fs,max_seconds=max_seconds)
        if verbose: print('3/8 Linear regression referencing complete ', filt.shape)
    else:
        raise ValueError(f'{denoise_mode} not supported, choose from ["CAR", "LRR"]')
    
    # 4. Get thresholds 
    thresholds_computed = get_thresholds(filt.copy(), threshold)
    if verbose: print('4/8 Threshold computation complete ', thresholds_computed.shape)
    
    # 5. Get spike band power bins
    spike_pow = get_spike_bandpower(filt.copy(), clip_thresh_sp, fs)
    spike_pow_bin = bin_data(spike_pow.copy(), bin_size_ms, shift_size_ms, bin_type=0)  # bin by averaging the values
    if verbose: print('5/8 Spikeband power computation complete ', spike_pow.shape)
    
    # 6. Get threshold crossing bins 
    threshold_crossings = get_threshold_crossings(filt, thresholds_computed, fs)
    threshold_crossings_bin = bin_data(threshold_crossings.copy(), bin_size_ms, shift_size_ms, bin_type=1)  # bin by summing the values
    if verbose: print('6/8 Threshold crossing extraction complete ', threshold_crossings_bin.shape)
    
    # 7. Get normalization parameters for binned spike band power and threshold crossing 
    spike_pow_mean, spike_pow_std = get_mean_std(spike_pow_bin)
    thresh_cross_mean, thresh_cross_std = get_mean_std(threshold_crossings_bin)
    if verbose: print('7/8 Normalization parameters computation complete ', spike_pow_mean.shape, spike_pow_std.shape, thresh_cross_mean.shape, thresh_cross_std.shape)
    
    # 8. Save data to .mat file
    save_dict = {
        'ns5_filepath': ns5_filepath,
        'n_channels': n_channels,
        'n_arrays': n_arrays,
        'filt_order': filt_order,
        'lo_cut': lo_cut,
        'hi_cut': hi_cut,
        'fs': fs,
        'threshold_multiplier': threshold,
        'thresholds_computed': thresholds_computed,
        'clip_thresh_sp': clip_thresh_sp,
        'spike_pow': spike_pow_bin,
        'threshold_crossings': threshold_crossings_bin,
        'spike_pow_mean': spike_pow_mean,
        'spike_pow_std': spike_pow_std,
        'thresh_cross_mean': thresh_cross_mean,
        'thresh_cross_std': thresh_cross_std,
    }
    
    if denoise_mode == 'LRR':
        save_dict['lrr_coeffs'] = lrr_coeffs
    
    scipy.io.savemat(out_filepath, save_dict)
    if save_plot:
        fp_plot = out_filepath.replace('.mat','.png')
        f, [ax_sp, ax_tx] = plt.subplots(2,1)
        sns.heatmap(spike_pow_bin, ax=ax_sp)
        sns.heatmap(threshold_crossings_bin, ax=ax_tx)
        ax_tx.set_title('Threshold crossings')
        ax_sp.set_title('Spike power')
        plt.tight_layout()
        plt.savefig(fp_plot)
    
    if verbose: print('8/8 Data saved to .mat file')



if __name__ == "__main__":
    import os
    from glob import glob
    import pandas as pd
    import multiprocessing
    from joblib import Parallel, delayed, parallel_backend

    ######################
    ######  PARAMS  ######
    ######################
    DIR_NS5_RAW = '/Users/sabrasisler/Desktop/NPTL/Data/t12.2025.01.23/raw'
    DIR_SAVE_NEURAL_FEATS = '/Users/sabrasisler/Desktop/NPTL/Data/t12.2025.01.23/reprocessed'

    # List of filenames to process
    filenames = [
        'Hub1-20250123-101213-001.ns5',
        # Add more filenames as needed
    ]

    # N_JOBS = 7 # number of cores to use to parallelize preproc
    #############################
    ######  PRECONDITIONS  ######
    #############################
    assert os.path.exists(DIR_NS5_RAW)
    assert os.path.exists(DIR_SAVE_NEURAL_FEATS)

    OVERWRITE = False
    ####################
    ######  MAIN  ######
    ####################

    # preprocess all files in parallel
    ns5_filepaths = []
    save_filepaths = []
    for filename in filenames:
        ns5_filepath_search = os.path.join(DIR_NS5_RAW, filename)
        ns5_search_results = glob(ns5_filepath_search)
        if len(ns5_search_results) != 1:
            print(f"\nWARNING: couldn't find unique neural ns5 for file {filename}")
            print(f"searching in {DIR_NS5_RAW} got {ns5_search_results}")
            print(f"SKIPPING\n")
            continue
        ns5_filepath = ns5_search_results[0]
        save_name = f"ts_sp_20ms_{os.path.splitext(filename)[0]}.mat"
        save_filepath = os.path.join(DIR_SAVE_NEURAL_FEATS, save_name)

        # Make sure the directory exists
        os.makedirs(DIR_SAVE_NEURAL_FEATS, exist_ok=True)
        if os.path.exists(save_filepath) and not OVERWRITE:
            print(f"skipping file {filename}, already preprocessed")
        else:
            print(f"beginning processing for file {filename}")
            compute_neural_features(ns5_filepath, save_filepath)

    #     ns5_filepaths.append(ns5_filepath)
    #     save_filepaths.append(save_filepath)
    # with parallel_backend('multiprocessing'):
    #    Parallel(n_jobs=N_JOBS)(delayed(compute_neural_features)(ns5_filepath, save_filepath) for ns5_filepath, save_filepath in zip(ns5_filepaths, save_filepaths))

    # #############################################
    # ###########         Local         ###########
    # #############################################
    # # Params
    # FP_NS5 = os.path.expanduser('~/oak/stanford/groups/henderj/benmk/data/cvcDelayImaginedInstruct/raw/Hub1-20240425-105115-001.ns5') # block 2 of that day
    # FP_SAVE_FEATS = os.path.expanduser('~/oak/stanford/groups/henderj/benmk/data/cvcDelayImaginedInstruct/deriv/python_feats/block_2_neural_features.mat')
    # FP_ONLINE_COMPUTED_FEATS = os.path.expanduser('~/oak/stanford/groups/henderj/benmk/data/cvcDelayImaginedInstruct/deriv/RedisMat/20240425_105941_(2).mat')

    # compute_neural_features(FP_NS5, FP_SAVE_FEATS)