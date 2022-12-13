import numpy as np
import pyenf
import math
import glob
import librosa
from scipy.stats.stats import pearsonr
import os
# Constants for file location

folder = "../ENF_data/"
csv_folder = "../ENF_data/CSV"



#function to compare the signal similarities

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

def read_filenames(folder):
    wav_filenames = glob.glob(folder+'*.wav')
    wav_filenames.sort()
    power_filenames = {}
    for file in wav_filenames:
        name = file.split('/')
        ind_filename = name[2]
        ind_filename_list = ind_filename.split('_')
        file_header = ind_filename_list[3]+'_'+ind_filename_list[4]+'_'+ind_filename_list[5]
        if file_header not in power_filenames:
            power_filenames[file_header] = []
            power_filenames[file_header].append(ind_filename)
        else:
            power_filenames[file_header].append(ind_filename)
    return power_filenames

def write_to_file(csv_filename,ENF_Data):
    counter=0
    with open(csv_filename,'w') as file:
        for each_enf in ENF_Data:
            x = str(counter)+","+str(each_enf[0])+"\n"
            file.write(x)
            counter = counter+1
            

def process_power_files(filenames_collection,power_files_location,csv_folder_location):
    dates = filenames_collection.keys()
    dates = sorted(dates)
    for date in dates:
        #print(date)
        date_list = date.split('_')
        csv_path = csv_folder_location + '/'+date_list[0]
        if os.path.exists(csv_path) is False:
            os.mkdir(csv_path)
        csv_filename = csv_path + '/' + str(date) + '.csv'
        per_day_recordings = filenames_collection[date]
        sorted(per_day_recordings)

        # For each of the power recording, the get ENF function is used to extract the power recording 
        ENF_per_day = []
        for each_file in per_day_recordings:
            each_file_path = power_files_location + str(each_file)
            power_signal0, fs = librosa.load(each_file_path, sr=1000) 
            print("Currently processing "+each_file)
            power_ENF = give_me_ENF(fs=1000,nfft=8192,frame_size=2,overlap=0,
                harmonics_mul=1,signal_file=power_signal0,nominal=60)
            
            duration = 12 # hours
            total_samples = int((12 * 60 * 60)/frame_size)
            
            ENF_Data = power_ENF[:total_samples]
            for value in ENF_Data:
                ENF_per_day.append(value)
        #print(len(ENF_per_day))
        print("Writing to file "+csv_filename)
        write_to_file(csv_filename=csv_filename,ENF_Data=ENF_per_day)
    return True

#parameters for the STFT algorithms
fs = 1000  # downsampling frequency
nfft = 8192
frame_size = 2  # change it to 6 for videos with large length recording
overlap = 0


# creating a dictionary of power files with keys as the day and the values as the recordings belonging
# to that day. 
power_filenames_dict = read_filenames(folder=folder)

#create the CSV main directory

if os.path.exists(csv_folder) is False:
    os.mkdir(csv_folder)
else:
    pass 

process = process_power_files(filenames_collection=power_filenames_dict,power_files_location=folder,csv_folder_location=csv_folder)

if process is True:
    print("All Power Recordings Processing Done...")
    