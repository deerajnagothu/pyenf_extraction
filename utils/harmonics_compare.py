# Author: Deeraj Nagothu
# ENF estimation from video recordings using Rolling Shutter Mechanism

# Import required packages
import numpy as np
import os
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

video_folder = "./Recordings/SSA/"

power_rec_name = "3hr_ENF.wav"
power_filepath = video_folder + power_rec_name


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

def save_this_variable(variable,location,filename):
    variable_location = location + filename
    store_variable_file = open(variable_location, 'wb')
    pickle.dump(variable, store_variable_file)
    store_variable_file.close()

def load_this_variable(location):
    variable_location = location
    load_variable_file = open(variable_location, 'rb')
    variable = pickle.load(load_variable_file)
    load_variable_file.close()
    return variable

fs = 1000  # downsampling frequency
nfft = 8192
frame_size = 2  # change it to 6 for videos with large length recording
overlap = 0
window_size = 60
shift_size= 10



ENF_variable_filename = video_folder + "ENF_300.pkl"
if os.path.exists(ENF_variable_filename) is True:
    ENF60 = load_this_variable(ENF_variable_filename)
    print("Loaded Power ENF")
else:
    power_signal0, fs = librosa.load(power_filepath, sr=fs)  # loading the power ENF data
    ENF60 = give_me_ENF(fs,nfft,frame_size,overlap,1,power_signal0,300)
    save_this_variable(ENF60,video_folder,"ENF_300.pkl")
    print("Created Power ENF")

norm_ENF60 = normalize_this(ENF60)



power_rec_name2 = "3hr_ENF_in_audio.wav"
power_filepath2 = video_folder + power_rec_name2
power_signal1, fs = librosa.load(power_filepath2, sr=fs)  # loading the power ENF data

ENF60_2 = give_me_ENF(fs,nfft,frame_size,overlap,1,power_signal1,300)
norm_ENF60_2 = normalize_this(ENF60_2)
rho,total_windows = correlation_vector(ENF60,ENF60_2,window_size,shift_size)

save_this_variable(rho,video_folder,"rho.pkl")

print(rho[0])
plt.figure()

plt.plot(rho[0],'g')#,norm_ENF300[:-8],'r')
plt.title("rho", fontsize=12)
plt.ylabel("correlation", fontsize=12)
plt.xlabel("Time", fontsize=12)
#plt.legend(["Combined","60 Hz","180 Hz","300 Hz"])
plt.show()

"""

plt.subplot(211)
plt.specgram(power_signal0[:int(len(power_signal0)/frame_size)],Fs=fs)
plt.title("Spectrogram of ENF Power Recording", fontsize=12)
plt.ylabel("Freq (Hz)", fontsize=12)
#plt.xlabel("Time", fontsize=12)
#plt.show()

ENF180 = give_me_ENF(fs,nfft,frame_size,overlap,1,power_signal0,180)
norm_ENF180 = normalize_this(ENF180)


ENF300 = give_me_ENF(fs,nfft,frame_size,overlap,1,power_signal0,300)
norm_ENF300 = normalize_this(ENF300)

#plt.figure()
plt.subplot(212)
plt.plot(norm_ENF60[:-8],'g',norm_ENF180[:-8],'b',norm_ENF300[:-8],'r')
plt.title("Normalized Multiple Harmonics ENF Signal", fontsize=12)
plt.ylabel("Freq (Hz)", fontsize=12)
plt.xlabel("Time", fontsize=12)
plt.legend(["60 Hz","180 Hz","300 Hz"])
plt.show()



ENF_harmonic = give_me_ENF(fs,nfft,frame_size,overlap,2,power_signal0,60)
norm_harmonic = normalize_this(ENF_harmonic)

plt.figure()
#plt.subplot(212)
plt.plot(norm_harmonic[:-8],'y--',norm_ENF60[:-8],'g',norm_ENF180[:-8],'b')#,norm_ENF300[:-8],'r')
plt.title("Harmonic Combination of ENF Signal", fontsize=12)
plt.ylabel("Freq (Hz)", fontsize=12)
plt.xlabel("Time", fontsize=12)
plt.legend(["Combined","60 Hz","180 Hz","300 Hz"])
plt.show()

filenam = "ENF_values.csv"
counter = 0
with open(filenam,'w') as file:
    for each_enf in ENF60:
        x = str(counter)+","+str(each_enf[0])+"\n"
        file.write(x)
        counter = counter+1
"""