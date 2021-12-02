# Author: Deeraj Nagothu
# ENF estimation from video recordings using Rolling Shutter Mechanism

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
from skimage.util import img_as_float
from skimage.segmentation import slic
from scipy.stats.stats import pearsonr
# Constants
folder = "Recordings/Descript/"
power_rec_name_1 = "ENF_10min_power.wav"
power_rec_name_2 = "deepfakevoice_2slits_v2.wav"
#power_rec_name_2 = "voice_with_ENF.wav"

power1_filepath = folder + power_rec_name_1
power2_filepath = folder + power_rec_name_2



def correlation_vector(ENF_signal1, ENF_signal2, window_size, shift_size):
    correlation_ENF = []
    length_of_signal = min(len(ENF_signal1), len(ENF_signal2))
    total_windows = math.ceil(( length_of_signal - window_size + 1) / shift_size)
    rho = np.zeros((1,total_windows))
    for i in range(0,total_windows):
        enf_sig1 = ENF_signal1[i * shift_size: i * shift_size + window_size]
        enf_sig2 = ENF_signal2[i * shift_size: i * shift_size + window_size]
        r,p = pearsonr(enf_sig1,enf_sig2)
        rho[0][i] = r[0]
    return total_windows,rho

fs = 1000  # downsampling frequency
nfft = 8192
frame_size = 2  # change it to 6 for videos with large length recording
overlap = 0
#filename = "mediator.wav"
filename = power1_filepath

signal0, fs = librosa.load(filename, sr=fs)  # loading the video ENF data
power_signal_filename = power2_filepath
power_signal0, fs = librosa.load(power_signal_filename, sr=fs)  # loading the power ENF data


# ENF extraction from video recording
video_signal_object = pyenf.pyENF(signal0=signal0, fs=fs, nominal=60, harmonic_multiples=1, duration=1,
                                  strip_index=0, frame_size_secs=frame_size, nfft=nfft, overlap_amount_secs=overlap, width_signal=0.02, width_band=1)
spectro_strip, frequency_support = video_signal_object.compute_spectrogam_strips()
weights = video_signal_object.compute_combining_weights_from_harmonics()
OurStripCell, initial_frequency = video_signal_object.compute_combined_spectrum(spectro_strip, weights,
                                                                                frequency_support)
ENF = video_signal_object.compute_ENF_from_combined_strip(OurStripCell, initial_frequency)

# uncomment when comparing only 2 graphs.
# creating figures for 2 plots
#fig, (video, power) = plt.subplots(2, 1, sharex=True)
#video.plot( )
#video.set_title("ENF Signal 1", fontsize=12)
#video.ticklabel_format(useOffset=False)
#video.set_ylabel("Frequency (Hz)", fontsize=12)





# ENF extraction from power recording
power_signal_object = pyenf.pyENF(signal0=power_signal0, fs=fs, nominal=60, harmonic_multiples=1, duration=0.1,
                                  strip_index=0, frame_size_secs=frame_size, nfft=nfft, overlap_amount_secs=overlap, width_signal=0.02, width_band=1)
power_spectro_strip, power_frequency_support = power_signal_object.compute_spectrogam_strips()
power_weights = power_signal_object.compute_combining_weights_from_harmonics()
power_OurStripCell, power_initial_frequency = power_signal_object.compute_combined_spectrum(power_spectro_strip,
                                                                                            power_weights,
                                                                                            power_frequency_support)
power_ENF = power_signal_object.compute_ENF_from_combined_strip(power_OurStripCell, power_initial_frequency)
# add power 2 plot
plt.figure()
plt.plot(ENF[:-30],'b',power_ENF[:-10],'g')
plt.title("ENF Signal 2", fontsize=12)
plt.ylabel("Freq (Hz)", fontsize=12)
plt.xlabel("Time", fontsize=12)
plt.ticklabel_format(useOffset=False)
print("Correlating the signal")

#fig.tight_layout()
#fig.show()


# values for rho
window_size = 30
shift_size = 5

total_windows,rho = correlation_vector(ENF[3:-10], power_ENF[3:-10],window_size,shift_size)

"""
t = np.arange(0,total_windows-1,1)
plt.plot(t,rho[0][1:],'g--', label="Correlation Coefficient")
#plt.plot(t,rho_without_SSM[0][1:],'b', label="Without SSM")
plt.hlines(y=0.8, xmin=0, xmax=len(t), colors='r', linestyles='--', lw=2)
plt.ylabel('Correlation Coefficient', fontsize=12)
plt.xlabel('Number of Windows compared', fontsize=12)
plt.title('ENF fluctuations from multiple devices', fontsize=12)
plt.legend(loc="lower right")
plt.show()
"""
ENF = ENF[:-60]
power_ENF = power_ENF[:-6]
#saving the values to a csv file
counter = 0
csv_filename = folder + "power.csv"
with open(csv_filename,'w') as file:
    for each_enf in ENF:
        x = str(counter)+","+str(each_enf[0])+"\n"
        file.write(x)
        counter = counter+1

counter = 0
csv_filename = folder + "voiceENF.csv"
with open(csv_filename,'w') as file:
    for each_enf in power_ENF:
        x = str(counter)+","+str(each_enf[0])+"\n"
        file.write(x)
        counter = counter+1


rho_original = [[0.99888477, 0.99815034, 0.9969985,  0.99376432, 0.99148822, 0.99293464,  0.99343464, 0.99520847, 0.99602178, 0.99595116, 0.99544285, 0.99423483,  0.99433822, 0.9940916,  0.99450368, 0.99612189, 0.99426043, 0.99363705,  0.99579082, 0.99432844, 0.99296054, 0.98833925, 0.98700942, 0.98353817,  0.98455675, 0.99402872, 0.98879099, 0.95928161, 0.97252161, 0.99506074,  0.99735256, 0.99769417, 0.99842662, 0.99898481, 0.99914639, 0.99905237,  0.99868745, 0.99898704, 0.99914525, 0.9988579,  0.9972311,  0.9954604,  0.99592654, 0.99527461, 0.99134973, 0.9832306,  0.98276157, 0.99384387,  0.99212241, 0.9888462,  0.99050252, 0.99476399, 0.99617141]]

plt.figure()
t = np.arange(0,total_windows-1,1)
plt.plot(t,rho_original[0][:-1],'g-*', label="Original Audio")
plt.plot(t,rho[0][:-1],'b--', label="Partially Deepfake Audio")
#plt.plot(t,rho_without_SSM[0][1:],'b', label="Without SSM")
plt.hlines(y=0.8, xmin=0, xmax=len(t), colors='r', linestyles='--', lw=2)
plt.ylabel('Correlation Coefficient', fontsize=12)
plt.xlabel('Number of Windows compared', fontsize=12)
plt.title('Measure of similarity between ENF from Original and Deepfake Audio', fontsize=12)
plt.legend(loc="lower right")
plt.show()