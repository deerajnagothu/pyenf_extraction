# Author: Deeraj Nagothu
# ENF estimation from video recordings using Rolling Shutter Mechanism

# Import required packages
import numpy as np
import cv2
import pickle
import pyenf
#from scipy import signal, io
import scipy.io.wavfile
import math
from scipy.fftpack import fftshift
import matplotlib.pyplot as plt
import librosa
from skimage.util import img_as_float
from skimage.segmentation import slic
from scipy.stats.stats import pearsonr
from scipy.signal import butter, lfilter, freqz

video_folder = "/home/deeraj/Documents/Projects/pyENF_extraction_rolling_shutter/Recordings/"

power_rec_name = "power_20min.wav"
power_filepath = video_folder + power_rec_name

dup_power_rec_name = "power_20min2.wav"
dup_power_filepath = video_folder + dup_power_rec_name

def correlation_vector(ENF_signal1, ENF_signal2, window_size, shift_size):
    correlation_ENF = []
    length_of_signal = min(len(ENF_signal1), len(ENF_signal2))
    total_windows = math.ceil(( length_of_signal - window_size + 1) / shift_size)
    rho = np.zeros((1,total_windows))
    for i in range(0,total_windows):
        enf_sig1 = ENF_signal1[i * shift_size: i * shift_size + window_size]
        enf_sig2 = ENF_signal2[i * shift_size: i * shift_size + window_size]
        enf_sig1 = np.reshape(enf_sig1, (len(enf_sig1),))
        enf_sig2 = np.reshape(enf_sig2,(len(enf_sig2), ))
        r,p = pearsonr(enf_sig1, enf_sig2)
        rho[0][i] = r
    return rho,total_windows

def give_me_ENF(fs,nfft,frame_size,overlap,harmonics_mul,signal_file,nominal):
    power_signal_object = pyenf.pyENF(signal0=signal_file, fs=fs, nominal=nominal, harmonic_multiples=harmonics_mul, duration=0.1,
                                  strip_index=0, frame_size_secs=frame_size, nfft=nfft, overlap_amount_secs=overlap)
    power_spectro_strip, power_frequency_support = power_signal_object.compute_spectrogam_strips()
    power_weights = power_signal_object.compute_combining_weights_from_harmonics()
    power_OurStripCell, power_initial_frequency = power_signal_object.compute_combined_spectrum(power_spectro_strip,
                                                                                            power_weights,
                                                                                            power_frequency_support)
    ENF = power_signal_object.compute_ENF_from_combined_strip(power_OurStripCell, power_initial_frequency)
    #print(power_weights)
    #print(power_initial_frequency)
    return ENF

def normalize_this(ENF_signal):
    norm = np.linalg.norm(ENF_signal)
    norm_ENF = ENF_signal/norm
    return norm_ENF



def butter_bandstop(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    return b, a


def butter_bandstop_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandstop(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
fs = 1000  # downsampling frequency
nfft = 8192
frame_size = 3  # change it to 6 for videos with large length recording
overlap = 1

# for filtering the signal
lowcut = 55
highcut = 65

power_signal0, fs = librosa.load(power_filepath, sr=fs)  # loading the power ENF data

y2, fs = librosa.load(power_filepath, sr=fs)  # loading the power ENF data

y3, fs = librosa.load(dup_power_filepath, sr=fs)  # loading the power ENF data

y = butter_bandstop_filter(power_signal0, lowcut, highcut, fs, order=6)
y33 = butter_bandpass_filter(y3, lowcut, highcut, fs, order=6)
#print(len(y))

y[200000:350000] = y[200000:350000]

plt.subplot(131)
plt.specgram(power_signal0,Fs=fs,cmap='jet')
plt.title("Original Spectrogram", fontsize=12)
plt.ylabel("Freq (Hz)", fontsize=12)
plt.xlabel("Time (a)", fontsize=12)

plt.subplot(132)
plt.specgram(y,Fs=fs,cmap='jet')
plt.title("Bandstop Filtered", fontsize=12)
#plt.ylabel("Freq (Hz)", fontsize=12)
plt.xlabel("Time (b)", fontsize=12)

#y2[203500:348000] =  power_signal0[203500:348000]
#y2[200000:350000] =  y3[200000:350000]

plt.subplot(133)
plt.specgram(y2,Fs=fs,cmap='jet')
plt.title("Fake ENF Added", fontsize=12)
#plt.ylabel("Freq (Hz)", fontsize=12)
plt.xlabel("Time (c)", fontsize=12)


plt.show()
plt.figure(2)
plt.specgram(y2,Fs=fs,cmap='jet')
plt.title("Spectral Inconsistencies", fontsize=12)
plt.ylabel("Freq (Hz)", fontsize=12)
plt.xlabel("Time (c)", fontsize=12)


plt.show()

#plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
#plt.xlabel('time (seconds)')
#plt.hlines([-a, a], 0, T, linestyles='--')
#plt.grid(True)
#plt.axis('tight')
#plt.legend(loc='upper left')
#plt.show()


ENF60 = give_me_ENF(fs,nfft,frame_size,overlap,1,y3,60)
norm_ENF60 =  ENF60#normalize_this(ENF60)

ENF180 = give_me_ENF(fs,nfft,frame_size,overlap,1,y,180)
norm_ENF180 = ENF180 - 120 #normalize_this(ENF180)


ENF300 = give_me_ENF(fs,nfft,frame_size,overlap,1,y,300)
norm_ENF300 = ENF300 - 240 #normalize_this(ENF300)

plt.figure(3)
#plt.subplot(212)
plt.plot(norm_ENF60[:-8],'g',norm_ENF180[:-8],'b',norm_ENF300[:-8],'r')
plt.title("Normalized Multiple Harmonics ENF Signal", fontsize=12)
plt.ylabel("Freq (Hz)", fontsize=12)
plt.xlabel("Time", fontsize=12)
plt.legend(["60 Hz","180 Hz","300 Hz"])
plt.show()
