# Author: Deeraj Nagothu

from pylab import *
import numpy as np
import scipy as sp
from scipy.io.wavfile import read
import scipy.signal as sps
import matplotlib.pyplot as plt
import wave
# Parameters

sampling_freq = 1000

class ENF:

    def __init__(self, sampling_freq, filename, lower_freq, upper_freq, overlap):
        self.sampling_freq = sampling_freq
        self.filename = filename
        self.lower_freq = lower_freq
        self.upper_freq = upper_freq
        self.overlap = overlap

    def read_initial_data(self):
        try:
            self.original_wav = wave.open(self.filename)
            self.original_sampling_frequency = self.original_wav.getframerate() # get sampling frequency

            print("The sampling frequency of given file is ",self.original_sampling_frequency)
        except:
            print("Check File name or Path")
        return self.original_sampling_frequency
    # TODO downsampling the original audio file and create another .wav file with 1kHz sampling frequency
    def down_sample_signal(self):
        self.downsampled_filename = "Downsampled_recording.wav" # "Downsampled_"+self.filename
        self.original_nframes = self.original_wav.getnframes() # getnframes returns number of audio frames
        if self.original_wav.getsampwidth() == 1:  # getsampwidth returns sample width in bytes
            self.nptype = np.uint8
        elif self.original_wav.getsampwidth() ==2:
            self.nptype = np.uint16
        self.resampled_wav = wave.open(self.downsampled_filename,"w")
        self.resampled_wav.setframerate(self.sampling_freq) # setting it to required sampling frequency
        self.resampled_wav.setnchannels(self.original_wav.getnchannels()) # number of audio channels ( 1 for mono and 2 for stereo)
        self.resampled_wav.setsampwidth(self.original_wav.getsampwidth()) # sample width in bytes
        self.resampled_wav.setnframes(1)   # number of audio frames

        audio = self.original_wav.readframes(self.original_nframes)
        nroutsamples = round(len(audio)) * self.sampling_freq/self.original_sampling_frequency

        audio_out = sps.resample(np.fromstring(float(audio),self.nptype), nroutsamples)
        audio_out = audio_out.astype(self.nptype)

        self.resampled_wav.writeframes(audio_out.copy(order='C'))
        self.resampled_wav.close()











mysignal = ENF(sampling_freq,"Recordings/recorded_frequency.wav", 59.9, 60.1, 9)

original_sampling_frequency = mysignal.read_initial_data()
if original_sampling_frequency != sampling_freq:
    signal = mysignal.down_sample_signal()






