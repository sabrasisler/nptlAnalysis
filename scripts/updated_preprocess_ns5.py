"""
Author: Benyamin Meschede-Krasa
preprocess raw neural features to compute spike band power and threshold crossings (-4RMS)
This code is based on [`preprocess_ns5_utils.py`](https://code.stanford.edu/fwillett/nptlrig2/-/blob/nick_dev/Analysis/nhahn7/speech-neural-dynamics/preprocess_ns5_utils.py?ref_type=heads)
which was written by Nick Card and Nick Hahn. It has been updated for Gemini Hubs.
"""

import numpy as np
import scipy.signal
import scipy.io
import matplotlib.pyplot as plt
#import seaborn as sns
from pprint import pprint

from brpylib import NsxFile

def read_ns5_file(ns5_filename, n_channels):
    """
    Read NS5 file and extract raw voltage data along with header information.

    Parameters
    ----------
    ns5_filename : str
        Path to the NS5 file.
    n_channels : int
        Number of channels to extract.

    Returns
    -------
    tuple
        (raw_voltage, header) where raw_voltage is a numpy array with shape (n_samples, n_channels)
        and header is the header information from the NS5 file.
    """
    nsx_file = NsxFile(ns5_filename)
    header = nsx_file.header  # header is expected to be a list of dictionaries
    all_dat = nsx_file.getdata('all', 0)  # electrode ids and start time info
    nsx_file.close()
    
    # Data sometimes chunked across segments so we stack them.
    # Extract only the first n_channels channels.
    raw_voltage = np.hstack(all_dat['data'])[:n_channels, :]
    return raw_voltage.T, header


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
    # Get filter parameters for non-causal Butterworth filter.
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


def lrr(dat, n_arrays, n_channels, fs, max_seconds=45):
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
    max_seconds : int, optional
        Maximum number of seconds to use for LRR computation (default is 45).

    Returns
    -------
    tuple
        LRR-referenced data and LRR weights.
    """
    lrr_weights = []
    n_electrodes = n_channels // n_arrays  # electrodes per array
    for array in range(n_arrays):
        print(f'Starting LRR for array {array + 1} of {n_arrays}')
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
    threshold = thresh_mult * std  # threshold multiplier is typically -4.0 or -4.5
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
        # Convert to float32 to avoid overflow when squaring
        dat_1ms = np.array(dat_1ms, dtype='float32')
        sp_pow = np.mean(np.square(dat_1ms), axis=0)
        sp_pow[sp_pow > clip_thresh] = clip_thresh  # clip maximum values
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
    """
    Bin the data at a specified resolution.

    Parameters
    ----------
    dat : np.ndarray
        Input data at 1000Hz resolution.
    bin_size_ms : int
        The size of each bin in milliseconds.
    shift_size_ms : int
        The shift between consecutive bins in milliseconds.
    bin_type : int
        If 0, average the values in the bin; if 1, sum the values.

    Returns
    -------
    np.ndarray
        Binned data.
    """
    binned_dat = []
    for i in range(0, dat.shape[0], shift_size_ms):
        dat_for_bin = dat[i:i+bin_size_ms, :]
        if bin_type == 0:
            bin_dat = np.mean(dat_for_bin, axis=0)
        elif bin_type == 1:
            bin_dat = np.sum(dat_for_bin, axis=0)
        bin_dat = np.array(bin_dat, dtype='float32')
        binned_dat.append(bin_dat)
    return np.array(binned_dat)


def get_threshold_crossings(dat, threshold, fs):
    """
    Extract threshold crossings for every 1ms.

    Parameters
    ----------
    dat : np.ndarray
        Input data with shape (n_samples, n_channels).
    threshold : np.ndarray
        Threshold values for each channel.
    fs : float
        Sampling frequency.

    Returns
    -------
    np.ndarray
        Threshold crossing binary values for each 1ms window.
    """
    n_samples = int(fs / 1000)  # samples per 1ms
    threshold_crossings = []
    for i in range(0, dat.shape[0], n_samples):
        dat_1ms = dat[i:i+n_samples, :]
        thresh_cross = np.min(dat_1ms, axis=0) <= threshold
        thresh_cross = (thresh_cross * 1).astype('int16')
        threshold_crossings.append(thresh_cross)
    return np.array(threshold_crossings)


def get_mean_std(dat):
    """
    Compute the mean and standard deviation across samples.

    Parameters
    ----------
    dat : np.ndarray
        Input data.

    Returns
    -------
    tuple
        (mean, std) where both are numpy arrays of type float64.
    """
    dat_mean = np.mean(dat, axis=0)
    dat_std = np.std(dat, axis=0)
    return np.array(dat_mean, dtype='float64'), np.array(dat_std, dtype='float64')


def compute_neural_features(ns5_filepath, out_filepath,
                            unscramble_vPCG=False,
                            denoise_mode='LRR',
                            max_seconds=45,  # LRR computation window in seconds
                            n_arrays=4,
                            n_channels=256,
                            bin_size_ms=20,
                            shift_size_ms=20,  # Non-overlapping bins
                            threshold=-4.0,  # Multiplier for the standard deviation
                            filt_order=4,  # Order of the Butterworth filter
                            lo_cut=250,  # Lower cutoff frequency
                            hi_cut=5000,  # Upper cutoff frequency
                            fs=30000,  # Sampling frequency
                            clip_thresh_sp=10000,  # Clipping threshold for spike band power
                            save_plot=True,  # Save plot for sanity checking
                            verbose=True):
    """
    Compute neural features from raw NS5 data from Gemini Hubs and save the results to a .mat file.
    The output file will contain 20ms binned features including spike band power and threshold crossings,
    similar to the redis mats computed online.

    Parameters
    ----------
    ns5_filepath : str
        Path to the NS5 file containing neural data.
    out_filepath : str
        Path to save the output .mat file with computed neural features.
    unscramble_vPCG : bool, optional
        Whether to unscramble 6v channels (default False).
    denoise_mode : str, optional
        Denoising mode, either 'CAR' or 'LRR' (default 'LRR').
    max_seconds : int, optional
        Maximum seconds for LRR computation (default 45).
    n_arrays : int, optional
        Number of arrays (default 4).
    n_channels : int, optional
        Total number of channels (default 256).
    bin_size_ms : int, optional
        Size of the bins in ms (default 20).
    shift_size_ms : int, optional
        Shift in ms between bins (default 20).
    threshold : float, optional
        Multiplier for standard deviation to compute threshold (default -4.0).
    filt_order : int, optional
        Order of the Butterworth filter (default 4).
    lo_cut : float, optional
        Lower cutoff frequency (default 250).
    hi_cut : float, optional
        Upper cutoff frequency (default 5000).
    fs : float, optional
        Sampling frequency (default 30000).
    clip_thresh_sp : float, optional
        Threshold for clipping spike band power (default 10000).
    save_plot : bool, optional
        Whether to save a plot of the binned features (default True).
    verbose : bool, optional
        Verbose output (default True).

    Returns
    -------
    None
    """
    # Read raw neural data and header information
    raw_neural, header = read_ns5_file(ns5_filepath, n_channels)
    
    if unscramble_vPCG:
        if verbose: 
            print('Unscrambling 6v Channels')
        raw_neural = unscrambleChans(raw_neural)
    
    if verbose:
        print('\n1/8: Raw data loading complete:', raw_neural.shape)
        print('NS5 header (first entry):', header[0] if header else "No header found")
    
    # 2. Filter the raw data
    filt = raw_neural.copy()
    b, a = build_filter(filt_order, lo_cut, hi_cut, fs)
    for ch in range(filt.shape[1]):
        filt[:, ch] = scipy.signal.filtfilt(b, a, filt[:, ch])
    del raw_neural  # free memory
    if verbose: 
        print('2/8: Bandpass filtering complete:', filt.shape)
    
    # 3. Apply referencing (CAR or LRR)
    if denoise_mode == 'CAR':
        filt = car(filt, n_arrays, n_channels)
        if verbose: 
            print('3/8: Common average referencing complete:', filt.shape)
    elif denoise_mode == 'LRR':
        filt, lrr_coeffs = lrr(filt, n_arrays, n_channels, fs, max_seconds=max_seconds)
        if verbose: 
            print('3/8: Linear regression referencing complete:', filt.shape)
    else:
        raise ValueError(f'{denoise_mode} not supported, choose from ["CAR", "LRR"]')
    
    # 4. Compute thresholds
    thresholds_computed = get_thresholds(filt.copy(), threshold)
    if verbose: 
        print('4/8: Threshold computation complete:', thresholds_computed.shape)
    
    # 5. Compute spike band power and bin it
    spike_pow = get_spike_bandpower(filt.copy(), clip_thresh_sp, fs)
    spike_pow_bin = bin_data(spike_pow.copy(), bin_size_ms, shift_size_ms, bin_type=0)
    if verbose: 
        print('5/8: Spike band power computation complete:', spike_pow.shape)
    
    # 6. Compute threshold crossings and bin them
    threshold_crossings = get_threshold_crossings(filt, thresholds_computed, fs)
    threshold_crossings_bin = bin_data(threshold_crossings.copy(), bin_size_ms, shift_size_ms, bin_type=1)
    if verbose: 
        print('6/8: Threshold crossing extraction complete:', threshold_crossings_bin.shape)
    
    # 7. Compute normalization parameters
    spike_pow_mean, spike_pow_std = get_mean_std(spike_pow_bin)
    thresh_cross_mean, thresh_cross_std = get_mean_std(threshold_crossings_bin)
    if verbose: 
        print('7/8: Normalization parameters computed.')
    
    # 8. Save data to a .mat file (including header info)
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
        'ns5_header': header,  # saving entire header information
        # Also save the first timestamp from the first header dictionary (if available)
        'recording_start_timestamp': header[0]['Timestamp'][0] if header and 'Timestamp' in header[0] else None
    }
    
    if denoise_mode == 'LRR':
        save_dict['lrr_coeffs'] = lrr_coeffs
    
    scipy.io.savemat(out_filepath, save_dict)
    if save_plot:
        fp_plot = out_filepath.replace('.mat', '.png')
        # Uncomment the following two lines if you have seaborn imported
        # f, [ax_sp, ax_tx] = plt.subplots(2,1)
        # sns.heatmap(spike_pow_bin, ax=ax_sp); sns.heatmap(threshold_crossings_bin, ax=ax_tx)
        # Otherwise, a simple plot example:
        f, ax = plt.subplots(2, 1, figsize=(12, 8))
        ax[0].imshow(spike_pow_bin.T, aspect='auto')
        ax[0].set_title('Spike Power (binned)')
        ax[1].imshow(threshold_crossings_bin.T, aspect='auto')
        ax[1].set_title('Threshold Crossings (binned)')
        plt.tight_layout()
        plt.savefig(fp_plot)
    
    if verbose:
        print('8/8: Data saved to .mat file')


if __name__ == "__main__":
    import os
    import re

    # Define input/output paths
    test_data_dir = '/oak/stanford/groups/henderj/sasidhar/my_preprocessing_data/test_datasets_raw/'
    test_out_dir = '/oak/stanford/groups/henderj/sasidhar/my_preprocessing_data/test_datasets_preprocessed_0/'

    # Check if the output directory exists; if not, create it.
    if not os.path.exists(test_out_dir):
        os.makedirs(test_out_dir)
        print(f"Directory created: {test_out_dir}")
    else:
        print(f"Directory already exists: {test_out_dir}")

    # Example blocks for first pass (11/21/24)
    block_paths = [
        test_data_dir + 't12.2022.10.27/NSP Data/2_neuralProcess_Complete_bld(002)003.ns5',  # rhythm set
        test_data_dir + 't12.2024.08.29/NSP Data/7_(007)/20240829-083550/Hub1-20240829-083550-001.ns5',  # rhythm set
        test_data_dir + 't12.2024.10.15/NSP Data/7_(007)/20241015-085443/Hub1-20241015-085443-001.ns5'  # Chaofei speech production
    ]

    for ns5block_path in block_paths:
        # Define output folder based on the directory before "NSP Data"
        session = ns5block_path.split('/NSP Data')[0]
        folder_to_create = os.path.basename(session)
        save_filepath = os.path.join(test_out_dir, folder_to_create)
        if not os.path.exists(save_filepath):
            os.makedirs(save_filepath)
            print(f"Directory created: {save_filepath}")
        else:
            print(f"Directory already exists: {save_filepath}")

        # Get block number from the file path (using a regex match)
        block = re.search(r'\((\d+)\)', ns5block_path)
        save_file = os.path.join(save_filepath, f"ts_sp_20ms_block{block.group(1)}.mat")

        if os.path.isfile(save_file):
            print(f"The file exists: {save_file}")
        else:
            print('Processing session: ' + session)
            compute_neural_features(ns5block_path, save_file)