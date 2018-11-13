# Author: Deeraj Nagothu

from pylab import *
import numpy as np
import scipy as sp
from scipy.io.wavfile import read
from scipy.signal import freqz
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import wave
import librosa

# Parameters

sampling_freq = 1000
lowcut = 59.9
highcut = 60.1

class ENF:
    def __init__(self, sampling_freq, filename, lower_freq, upper_freq, overlap):
        self.sampling_freq = sampling_freq  # the sampling frequency minimum required for ananlysis
        self.filename = filename    # audio .wave recording file
        self.lower_freq = lower_freq    # lower cutoff frequency. For example, US has 60Hz so lower cutoff would be around 59.98Hz
        self.upper_freq = upper_freq    # Upper cutoff frequency. For example, US has 60Hz so upper cutoff would be around 60.02Hz
        self.overlap = overlap  # STFT window overlap between window frames

    # Extract the sampling frequency of the given audio recording and return the fs
    def read_initial_data(self):
        try:
            self.original_wav = wave.open(self.filename)
            self.original_sampling_frequency = self.original_wav.getframerate() # get sampling frequency
            self.signalData, self.new_sampling_frequency = librosa.load(self.filename,sr=self.sampling_freq)
            print("The sampling frequency of given file is ",self.original_sampling_frequency)
        except:
            print("Check File name or Path")
        return self.original_sampling_frequency, self.signalData

    # If the given audio file has higher sampling frequency then this function will create a new audio file by setting
    # all the traits of original audio file to new file and change the sampling frequency

    def down_sample_signal(self):
        self.signalData, self.new_sampling_frequency = librosa.load(self.filename, sr=self.sampling_freq)
        return self.signalData


#TODO: draw a graph of frequencies in this given file. There should be a spike at 60Hz for power files
    def plot_spectrogram(self):
        print("The sampling frequency of file in spectrogram is", self.sampling_freq)
        plt.subplot(211)
        plt.title("Spectrogram of a wav file with ENF")

        plt.plot(self.signalData)
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")

        plt.subplot(212)
        plt.specgram(self.signalData,Fs=self.new_sampling_frequency)
        plt.xlabel("Time")
        plt.ylabel("Frequency")

        plt.show()
        return 0

    def frequency_plot(self, signal, label): # plot the frequencies in the given file
        Ts = 1.0/float(self.new_sampling_frequency) # sampling interval

        n = len(signal)  # length of the signal
        T = n / float(self.new_sampling_frequency)
        t = np.arange(0,T,Ts) # Time Vector


        k = np.arange(n)
        freq = k/T # two sided frequency range
        freq = freq[range(int(n/2))] # one sided frequency range, eliminating negative frequency using Nyquist frequency

        Y = np.fft.fft(signal)/n # fft computing and normalization
        Y = Y[range(int(n/2))]

        plt.subplot(211)
        titl = "Frequency Analysis "+ label
        plt.title(titl)

        plt.plot(t,signal)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")

        plt.subplot(212)
        plt.plot(freq,abs(Y),'r') # plotting the spectrum
        plt.xlabel("Freq(Hz)")
        plt.ylabel("Y(freq)")

        plt.show()
        return 0

    def butter_bandpass(self,order=3):
        nyquist = 0.5 * self.new_sampling_frequency
        low = self.lower_freq/nyquist
        high = self.upper_freq/ nyquist
        b, a = butter(order, [low,high], btype='band')

        return b, a

    def butter_bandpass_filter(self, data, order=3):
        b,a = self.butter_bandpass(order=order)
        y = lfilter(b,a,data)
        return y


def main():
    #mysignal = ENF(sampling_freq,"Recordings/recorded_frequency.wav", 59.9, 60.1, 9)
    mysignal = ENF(sampling_freq, "Recordings/Grid_A_P1.wav", 59.9, 60.1, 9)
    original_sampling_frequency, signal = mysignal.read_initial_data()
    if original_sampling_frequency != sampling_freq:
        signal = mysignal.down_sample_signal()
        print("The given audio file has higher sampling frequency than required. So Downsampling the signal")
    else:
        print("The given audio file has the required sampling frequency, NO downsampling required")
    print("Plotting the diagram")
    # To plot the spectrogram and analyse, uncomment the following line
    #mysignal.plot_spectrogram()

    # To check the frequency analysis of the signal uncomment this line
    #mysignal.frequency_plot(signal, label="Before Filtering")

    filtered_signal = mysignal.butter_bandpass_filter(signal)

    # to check if the filtering of the signal work uncomment this line
    mysignal.frequency_plot(filtered_signal, label="After Filtering")



if __name__ == '__main__':
    main()






