
# Import required packages
import numpy as np
import cv2
import pickle
import pyenf
from scipy import signal, io
import math
from scipy.fftpack import fftshift
import matplotlib.pyplot as plt
import librosa
import time
from skimage.util import img_as_float
from skimage.segmentation import slic
from scipy.stats.stats import pearsonr


# creating a loop to go over all the power recordings and create their equivalent csv file. Total of 600 hours.
total_time_taken = 0

begin = time.time()

def ENF_estimator(power_signal0):
    fs = 1000  # downsampling frequency
    nfft = 8192
    frame_size = 1  # change it to 6 for videos with large length recording
    overlap = 0
    power_signal_object = pyenf.pyENF(signal0=power_signal0, fs=fs, nominal=60, harmonic_multiples=1, duration=0.1,
                                    strip_index=0, frame_size_secs=frame_size, nfft=nfft, overlap_amount_secs=overlap)
    power_spectro_strip, power_frequency_support = power_signal_object.compute_spectrogam_strips()
    power_weights = power_signal_object.compute_combining_weights_from_harmonics()
    power_OurStripCell, power_initial_frequency = power_signal_object.compute_combined_spectrum(power_spectro_strip,
                                                                                            power_weights,
                                                                                            power_frequency_support)
    power_ENF = power_signal_object.compute_ENF_from_combined_strip(power_OurStripCell, power_initial_frequency)
    
    # add a dynamic plot of the ENF here

    return power_ENF




power_signal0, fs = librosa.load(power_signal_filename, sr=fs)  # loading the power ENF data



total_time_taken = total_time_taken + (end - begin)

print("It took " + str(total_time_taken) + " sec to complete the power estimation")
"""
plt.plot(power_ENF[:-8],'g')
plt.title("Power ENF Signal", fontsize=12)
plt.ylabel("Freq (Hz)", fontsize=12)
plt.xlabel("Time (sec)", fontsize=12)
plt.show()
"""