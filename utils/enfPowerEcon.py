# Author: Deeraj Nagothu
# ENF estimation from video recordings using Rolling Shutter Mechanism
# ENF estimation on the live from a long power recording. The inputs will be the window size and the shift size.
# Since the code is going to be applied on power recording, the resolution will be 1 ENF datapoint per second

# Import required packages
import numpy as np
import pyenf
import math
import librosa
import time

# from scipy.stats.stats import pearsonr

# Constants
folder = "Recordings/Sample11_parallel/"
power_rec_name = "power_recording_10min.wav"
power_filepath = folder + power_rec_name

# Live ENF estimator settings
window_size = 60
shift_size = 5

# ENF resolution settings
fs = 500  # downsampling frequency
nfft = 8192
frame_size = 1  # change it to 6 for videos with large length recording
overlap = 0

# compute the number of windows using window size and iterate over the power signal
# This variable needs pre-computing. Since the file is being read on the go. Other alternative is to setup while loop
# and exit whenever the input is not of enough size.
number_of_windows = 20

power_ENF = np.zeros((number_of_windows*window_size,1))
for win in range(0,number_of_windows):
    start = time.time()
    power_signal0, fs = librosa.load(power_filepath, sr=fs,offset=float(win*window_size),duration=float(window_size))  # loading the power ENF data
    # for above line offset is where I want to start, and duration is how long of recording starting from offset.
    # ENF extraction from power recording
    power_signal_object = pyenf.pyENF(signal0=power_signal0, fs=fs, nominal=60, harmonic_multiples=1, duration=0.1,
                                      strip_index=0, frame_size_secs=frame_size, nfft=nfft, overlap_amount_secs=overlap)
    power_spectro_strip, power_frequency_support = power_signal_object.compute_spectrogam_strips()
    power_weights = power_signal_object.compute_combining_weights_from_harmonics()
    power_OurStripCell, power_initial_frequency = power_signal_object.compute_combined_spectrum(power_spectro_strip,
                                                                                                power_weights,
                                                                                                power_frequency_support)
    power_ENF[win*window_size:win*window_size+window_size] = power_signal_object.compute_ENF_from_combined_strip(power_OurStripCell, power_initial_frequency)
    #print(power_ENF[win*window_size:win*window_size+window_size])
    print("----%s seconds -----" % (time.time() - start))

# This commented code should read the whole file first, but it was clearly causing some delays in the beginning. Now
# it can easily read on the go, and produce ENF.
"""
power_signal0, fs = librosa.load(power_filepath, sr=fs)  # loading the power ENF data

# compute the number of windows using window size and iterate over the power signal
number_of_windows = int(len(power_signal0)/(window_size*fs))

power_ENF = np.zeros((number_of_windows*window_size,1))
for win in range(0,number_of_windows):
    power_signal0_segment = power_signal0[win*window_size*fs : win*window_size*fs + window_size*fs]
    # ENF extraction from power recording
    power_signal_object = pyenf.pyENF(signal0=power_signal0_segment, fs=fs, nominal=60, harmonic_multiples=1, duration=0.1,
                                      strip_index=0, frame_size_secs=frame_size, nfft=nfft, overlap_amount_secs=overlap)
    power_spectro_strip, power_frequency_support = power_signal_object.compute_spectrogam_strips()
    power_weights = power_signal_object.compute_combining_weights_from_harmonics()
    power_OurStripCell, power_initial_frequency = power_signal_object.compute_combined_spectrum(power_spectro_strip,
                                                                                                power_weights,
                                                                                                power_frequency_support)
    power_ENF[win*window_size:win*window_size+window_size] = np.array(power_signal_object.compute_ENF_from_combined_strip(power_OurStripCell, power_initial_frequency))
    print(power_ENF[win*window_size:win*window_size+window_size])
"""
