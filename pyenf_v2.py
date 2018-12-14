# Author Deeraj Nagothu

from scipy import signal
import wave
import numpy as np
import matplotlib.pyplot as plt
import librosa
import math

class pyENF:

    def __init__(self, filename, fs=1000, frame_size_secs=1, overlap_amount_secs=0, nfft=8192, nominal=None, harmonic_multiples=None, duration=None, width_band=1, width_signal=0.02, strip_index=1):

        self.filename = filename
        #self.signal0 = 0  # full signal
        self.fs = fs # sampling frequency required
        self.frame_size_secs = frame_size_secs  # Window size in seconds
        self.overlap_amount_secs = overlap_amount_secs
        self.nfft = nfft
        self.nominal = nominal  # the main ENF frequency ~ 60Hz for US and 50Hz for rest of the world
        self.harmonic_multiples = harmonic_multiples # multiple harmonics to combine them and get better signal estimate
        self.duration = duration # in minutes to compute weights

        # Width band is used to see what frequency range to use for SNR calculations
        self.width_band = width_band # half the width of the band about nominal values eg 1Hz for US ENF, 2 for others.

        # Width signal mentions how much does the ENF vary from its nominal value
        self.width_signal = width_signal # 0.02 for US and 0.5 for asian countries
        self.strip_index = strip_index # which harmonics dimensions should be applied to others. Normally default is 1.

        # Extract the sampling frequency of the given audio recording and return the fs
    def read_initial_data(self):
        self.orig_wav = wave.open(self.filename)
        self.original_sampling_frequency = self.orig_wav.getframerate()  # get sampling frequency
        self.signal0, self.fs = librosa.load(self.filename, sr=self.fs)
        print("The sampling frequency of original file was ", self.original_sampling_frequency)
        print("Sampling frequency Changed to ", self.fs)

        return self.signal0, self.fs

    # If the given audio file has higher sampling frequency then this function will create a new audio file by setting
    # all the traits of original audio file to new file and change the sampling frequency
    def find_closest(self, list_of_values, value):
        index = 1
        for i in range(1,len(list_of_values)+1):
            if (abs(list_of_values[i] - value) < abs(list_of_values[i-1] - value )):
                index = i
            else:
                break
        return index

    def compute_spectrogam_strips(self):
        # variables declaration
        number_of_harmonics = len(self.harmonic_multiples)
        spectro_strips = []
        frame_size = self.frame_size_secs * self.fs
        overlap_amount = self.overlap_amount_secs * self.fs
        shift_amount = frame_size - overlap_amount
        length_signal = len(self.signal0)
        number_of_frames = math.ceil( (length_signal - frame_size + 1)/shift_amount )

        # Collecting the spectrogram strips for each window in the signal and storing them in the list
        rows = int(self.nfft/2 + 1)
        starting = 0
        Pxx = np.zeros(shape=(rows,number_of_frames))
        win = signal.get_window('hamming',frame_size)
        for frame in range(number_of_frames):
            ending = starting + frame_size
            x = self.signal0[starting:ending]
            f, t, Pxx[:,frame] = signal.spectrogram(x, window=win, noverlap=self.overlap_amount_secs, nfft=self.nfft, fs=self.fs, mode='psd')
            starting = starting + shift_amount

        # choosing the strips that we need and setting up frequency support
        first_index = self.find_closest(f, self.nominal - self.width_band)
        second_index = self.find_closest(f, self.nominal + self.width_band)
        frequency_support = np.zeros(shape=(number_of_harmonics,2))

        





mysignal = pyENF(filename="recorded_frequency.wav",nominal=60, harmonic_multiples=np.arange(7), duration=2)

print(mysignal.filename)
x, fs = mysignal.read_initial_data()


