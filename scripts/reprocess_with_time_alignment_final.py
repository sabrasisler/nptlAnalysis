#!/usr/bin/env python3
"""
This script processes NS5 files by:
  1. Loading raw voltage and timestamp data.
  2. Optionally unscrambling channels.
  3. Bandpass filtering.
  4. Denoising via CAR or LRR.
  5. Computing thresholds.
  6. Aligning data with externally provided timestamps.
  7. Computing spike band power and threshold crossings.
  8. Saving the processed data to .mat files.

Adjust the file directories, parameters, and file lists as needed.
"""

import os
import numpy as np
import scipy.signal
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Import the NSx file reader from brpylib (make sure this package is installed)
from brpylib import NsxFile

###############################
# Function Definitions
###############################

def read_ns5_file(ns5_filename, n_channels):
    """
    Read NS5 file and extract raw voltage data and raw timestamps.
    
    Parameters:
      ns5_filename : str
          Path to the NS5 file.
      n_channels : int
          Number of channels to extract.
    
    Returns:
      tuple: (raw_voltage, raw_timestamps)
    """
    nsx_file = NsxFile(ns5_filename)
    # Read all data and headers (with full timestamps)
    all_dat = nsx_file.getdata('all', 0, full_timestamps=True)
    
    # Extract raw voltage. (Data may be chunked; we take only the first n_channels.)
    raw_voltage = np.hstack(all_dat['data'])[:n_channels, :].T  # shape (n_samples, n_channels)
    
    # Extract timestamp arrays from the data headers and concatenate them.
    timestamp_arrays = [entry["Timestamp"] for entry in all_dat["data_headers"]]
    raw_timestamps = np.concatenate(timestamp_arrays)
    
    nsx_file.close()
    return raw_voltage, raw_timestamps


def build_filter(filt_order, lo_cut, hi_cut, fs):
    b, a = scipy.signal.butter(filt_order, [lo_cut, hi_cut],
                                 btype='bandpass', analog=False, output='ba', fs=fs)
    return b, a


def car(dat, n_arrays, n_channels):
    n_electrodes = n_channels // n_arrays
    for array in range(n_arrays):
        this_array = np.arange(n_electrodes * array, n_electrodes * (array + 1), dtype='int16')
        this_mean = np.mean(dat[:, this_array], axis=1)
        dat[:, this_array] = dat[:, this_array] - this_mean[:, np.newaxis]
    return dat


def load_LLRWeights(llr_weights_filepath):
    """
    Loads LLR weights from a MATLAB file.
    """
    try:
        mat_contents = sio.loadmat(llr_weights_filepath)
    except Exception as e:
        raise IOError(f"Error loading MATLAB file: {e}")
    
    if "lrr_weights" not in mat_contents:
        raise KeyError("Variable 'lrr_weights' not found in the MATLAB file.")
    
    return mat_contents["lrr_weights"]


def getLRRWeights(dat, fs, max_seconds):
    print("Finding LLR weights")
    these_chans = np.arange(dat.shape[1])
    ref_mat = np.zeros((these_chans.size, these_chans.size))
    max_idx = max_seconds * fs
    up_to_idx = min(max_idx, dat.shape[0])
    rand_idx = np.random.permutation(np.arange(dat.shape[0]))
    use_idx = rand_idx[:up_to_idx]
    subsample_data = dat[use_idx, :]
    for chan_idx in these_chans:
        pred_idx = np.setdiff1d(these_chans, these_chans[chan_idx])
        filts = np.linalg.lstsq(subsample_data[:, pred_idx],
                                subsample_data[:, chan_idx],
                                rcond=None)[0]
        ref_mat[chan_idx, np.setdiff1d(np.arange(these_chans.size), chan_idx)] = filts
    return ref_mat


def lrr(dat, n_arrays, n_channels, fs, max_seconds=45, preload_llr_weights=False, llr_weights_filepath=""):
    lrr_weights = []
    n_electrodes = n_channels // n_arrays
    for array in range(n_arrays):
        print(f'Starting LRR for array {array + 1} of {n_arrays}')
        this_array = np.arange(n_electrodes * array, n_electrodes * (array + 1), dtype='int16')
        if preload_llr_weights:
            print( "Using preloaded LLR Weights")
            this_array_coeffs = load_LLRWeights(llr_weights_filepath)
        else:
            this_array_coeffs = getLRRWeights(dat[:, this_array], fs, max_seconds)
        dat[:, this_array] = dat[:, this_array] - np.dot(dat[:, this_array], this_array_coeffs)
        lrr_weights.append(this_array_coeffs)
    return dat, lrr_weights


def unscrambleChans(dat):
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
    std = np.std(dat, axis=0)
    threshold = thresh_mult * std
    return threshold


def get_mean_std(dat):
    dat_mean = np.mean(dat, axis=0).astype('float64')
    dat_std = np.std(dat, axis=0).astype('float64')
    return dat_mean, dat_std


def get_spikeband_power_aligned(dat, bins, clip_thresh):
    """
    Computes spike band power for each bin.
    
    Parameters:
      dat : np.ndarray
          Neural data (time x channels).
      bins : list of np.ndarray
          Each element is an array of indices defining a bin.
      clip_thresh : float
          Clipping threshold.
    
    Returns:
      np.ndarray: Array of spike band power values per bin.
    """
    spike_pow = []
    for bin_indices in bins:
        if bin_indices.size > 0:
            dat_bin = dat[bin_indices, :].astype('float32')
            sp_pow = np.mean(np.square(dat_bin), axis=0)
            sp_pow[sp_pow > clip_thresh] = clip_thresh
        else:
            sp_pow = np.zeros(dat.shape[1], dtype='float32')
        spike_pow.append(sp_pow)
    return np.array(spike_pow)


def get_threshold_crossings_aligned(dat, bins, threshold):
    """
    Computes binary threshold crossings for each bin.
    
    Parameters:
      dat : np.ndarray
          Neural data (time x channels).
      bins : list of np.ndarray
          Each element is an array of indices defining a bin.
      threshold : float
          Threshold value.
    
    Returns:
      np.ndarray: Binary array indicating threshold crossings per bin.
    """
    thresh_cross = []
    for bin_indices in bins:
        if bin_indices.size > 0:
            dat_bin = dat[bin_indices, :].astype('float32')
            thresh_val = (np.min(dat_bin, axis=0) <= threshold).astype(int)
        else:
            thresh_val = np.zeros(dat.shape[1], dtype='int32')
        thresh_cross.append(thresh_val)
    return np.array(thresh_cross)


def bin_data(spike_pow_nested, bin_type):
    """
    Bins the spike band power data that is grouped into sub-bins.
    
    Parameters:
      spike_pow_nested : list of list of np.ndarray
          Outer list corresponds to coarse bins, inner lists contain 1ms arrays.
      bin_type : int
          0 for mean, 1 for sum.
    
    Returns:
      np.ndarray: Aggregated data for each coarse bin.
    """
    binned_dat = []
    for coarse_bin in spike_pow_nested:
        if len(coarse_bin) == 0:
            binned_dat.append(np.zeros(256, dtype='float32'))
        else:
            arr = np.vstack(coarse_bin)
            if bin_type == 0:
                bin_val = np.mean(arr, axis=0)
            elif bin_type == 1:
                bin_val = np.sum(arr, axis=0)
            else:
                raise ValueError("Invalid bin_type. Use 0 for mean or 1 for sum.")
            binned_dat.append(bin_val.astype('float32'))
    return np.array(binned_dat)





###############################
# Main processing function
###############################

def main():

    # Parameters (adjust as necessary)
    unscramble_vPCG = True # Must be set to True is you want to preload_llr_weights for validation, otherwise it likely won't matter
    denoise_mode = 'LRR'  # Options: 'CAR' or 'LRR'
    max_seconds = 45
    n_arrays = 4
    n_channels = 256
    threshold = -3.5
    filt_order = 4
    lo_cut = 250
    hi_cut = 5000
    fs = 30000
    clip_thresh_sp = 50000
    save_plot = False
    verbose = True
    

    # Used for validation of binning. Compare the binning with the 
    preload_llr_weights = False # Default is set to false
    llr_weights_filepath = "/Users/sabrasisler/Desktop/NPTL/Data/t12.2025.01.23/t12.2025.01.23_block(3)_lrr_weights.mat"


    # Directories
    DIR_NS5_RAW = '/Users/sabrasisler/Desktop/NPTL/Data/t12.2025.01.23/raw'
    DIR_SAVE_NEURAL_FEATS = '/Users/sabrasisler/Desktop/NPTL/Data/t12.2025.01.23/reprocessed_timestamp_aligned'
    DIR_TS = '/Users/sabrasisler/Desktop/NPTL/Data/t12.2025.01.23/redisMat'
    os.makedirs(DIR_SAVE_NEURAL_FEATS, exist_ok=True)

    # Files
    #raw_filenames = ["Hub1-20250123-101213-001.ns5"] # Raw Ns5 files
    #redis_filenames = ["20250123_101632_(17).mat"] # Corresponding redisMat file
    redis_filenames = [
        "20250123_100941_(16).mat",
        "20250123_101632_(17).mat",
        "20250123_102302_(18).mat",
        "20250123_103002_(19).mat",
        "20250123_103557_(20).mat",
        "20250123_104106_(21).mat",
        "20250123_104554_(22).mat"
    ]

    raw_filenames = [
        "Hub1-20250123-100519-001.ns5",
        "Hub1-20250123-101213-001.ns5",
        "Hub1-20250123-101909-001.ns5",
        "Hub1-20250123-102524-001.ns5",
        "Hub1-20250123-103059-001.ns5",
        "Hub1-20250123-103633-001.ns5",
        "Hub1-20250123-104146-001.ns5"
    ]

    
    
    # Loop over each pair of raw and redis filenames
    for raw_filename, redis_filename in zip(raw_filenames, redis_filenames):
        ns5_filepath = os.path.join(DIR_NS5_RAW, raw_filename)
        # In the original notebook, the redis filename is adjusted; here we assume it is provided correctly.
        ts_filepath = os.path.join(DIR_TS, redis_filename)
        if preload_llr_weights:
            save_name = f"{os.path.splitext(redis_filename)[0]}_reprocessed_preloaded_weights.mat"
        else:
            save_name = f"{os.path.splitext(redis_filename)[0]}_reprocessed.mat"

        out_filepath = os.path.join(DIR_SAVE_NEURAL_FEATS, save_name)
        
        print(f"\nBeginning processing for file: {raw_filename}")

        # 1. Load raw data and timestamps from NS5 file.
        raw_neural, raw_timestamps = read_ns5_file(ns5_filepath, n_channels)
        if unscramble_vPCG:
            if verbose:
                print('Unscrambling channels...')
            raw_neural = unscrambleChans(raw_neural)
        if verbose:
            print('1/8 Raw data loading complete:', raw_neural.shape)
        
        # 2. Bandpass filtering.
        filt = raw_neural.copy()
        b, a = build_filter(filt_order, lo_cut, hi_cut, fs)
        for ch in range(filt.shape[-1]):
            filt[:, ch] = scipy.signal.filtfilt(b, a, filt[:, ch])
        del raw_neural
        if verbose:
            print('2/8 Bandpass filtering complete:', filt.shape)
        
        # 3. Denoising.
        if denoise_mode == 'CAR':
            denoised = car(filt, n_arrays, n_channels)
            if verbose:
                print('3/8 CAR denoising complete:', denoised.shape)
        elif denoise_mode == 'LRR':
            denoised, lrr_coeffs = lrr(filt, n_arrays, n_channels, fs,
                                       max_seconds=max_seconds,
                                       preload_llr_weights=preload_llr_weights,
                                       llr_weights_filepath=llr_weights_filepath)
            if verbose:
                print('3/8 LRR denoising complete:', denoised.shape)
        else:
            raise ValueError(f"{denoise_mode} not supported")
        
        # 4. Compute thresholds.
        thresholds_computed = get_thresholds(denoised.copy(), threshold)
        if verbose:
            print('4/8 Threshold computation complete:', thresholds_computed.shape)
        
        # 5. Load redis MATLAB file with nsp timestamp information.
        mat_data = sio.loadmat(ts_filepath)
        if verbose:
            print('5/8 Loaded Redis File:', ts_filepath)
        
        # Expecting keys "binned_neural_nsp_timestamp" and "binned_neural_spike_band_power"
        binned_neural_nsp_timestamp = mat_data["binned_neural_nsp_timestamp"][0]
        binned_neural_spikeband_power = mat_data["binned_neural_spike_band_power"]
        raw_timestamps = raw_timestamps.astype(np.int64)
        binned_neural_nsp_timestamp = binned_neural_nsp_timestamp.astype(np.int64)
        
        # 6. Create bins based on the provided timestamps.
        bin_width = binned_neural_nsp_timestamp[-1] - binned_neural_nsp_timestamp[-2]
        extended_bins = np.append(binned_neural_nsp_timestamp,
                                  binned_neural_nsp_timestamp[-1] + bin_width)
        interval_start_idx = np.searchsorted(raw_timestamps, extended_bins[:-1], side='left')
        interval_end_idx   = np.searchsorted(raw_timestamps, extended_bins[1:], side='left')
        indices_per_interval = [np.arange(start, end) for start, end in zip(interval_start_idx, interval_end_idx)]
        
        # 7. Compute spike band power using the 20ms bins.
        spike_pow = get_spikeband_power_aligned(denoised.copy(), indices_per_interval, clip_thresh_sp)
        if verbose:
            print('7/8 Spike band power computed:', spike_pow.shape)
        
        # 8. Compute threshold crossings.
        threshold_crossings = get_threshold_crossings_aligned(denoised.copy(), indices_per_interval, threshold)
        if verbose:
            print('6/8 Threshold crossings computed:', threshold_crossings.shape)
        
        # 9. Compute normalization parameters.
        spike_pow_mean, spike_pow_std = get_mean_std(spike_pow)
        thresh_cross_mean, thresh_cross_std = get_mean_std(threshold_crossings)
        if verbose:
            print('8/8 Normalization parameters computed:', spike_pow_mean.shape, spike_pow_std.shape)
        
        # 10. Save processed data to a .mat file.
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
            'binned_neural_spike_band_power': spike_pow,
            'binned_neural_threshold_crossings': threshold_crossings,
            'spike_pow_mean': spike_pow_mean,
            'spike_pow_std': spike_pow_std,
            'thresh_cross_mean': thresh_cross_mean,
            'thresh_cross_std': thresh_cross_std,
            'binned_neural_nsp_timestamp': binned_neural_nsp_timestamp,
        }
        if denoise_mode == 'LRR':
            save_dict['lrr_coeffs'] = lrr_coeffs

        mat_data.update()

        # update the redis file with the processed data.

        mat_data.update(save_dict)
        sio.savemat(out_filepath, mat_data)
        print(f"Updated MATLAB file saved as: {out_filepath}")

        # Optionally: if save_plot is True, create and save plots.
        if save_plot:
            fp_plot = out_filepath.replace('.mat', '.png')
            fig, (ax_sp, ax_tx) = plt.subplots(2, 1, figsize=(12, 6))
            sns.heatmap(spike_pow, ax=ax_sp)
            sns.heatmap(threshold_crossings, ax=ax_tx)
            ax_sp.set_title('Spike band power')
            ax_tx.set_title('Threshold crossings')
            plt.tight_layout()
            plt.savefig(fp_plot)
            plt.close()
            if verbose:
                print('Plots saved to:', fp_plot)

if __name__ == '__main__':
    main()