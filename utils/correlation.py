# Author: Deeraj Nagothu

# Creating a correlation matrix for all the ENF values collected in CSV file


import numpy as np
import math
import pickle
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
# Create a variable which contains all the values of ENF for each second

number_of_files = 24
hours_per_file = 25
window_size = 60 # in minutes
shift_size = 60 # in minutes

folder = "Recordings/"
file_header = "25h"
extension = ".csv"

read_values_again = False # Set this to True if you want to read the ENF values again from files

# Correlation function to evaluate the rho vector
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

total_samples = number_of_files*hours_per_file*3600
all_ENF_values = np.zeros(total_samples)
counter = 0
if read_values_again is True:
    print("[INFO] Loading ENF values from csv files..")
    for i in range(1,number_of_files+1):
        filename = folder+file_header + str(i) + extension
        for line in open(filename,"r"):
            values = line.split(",")
            all_ENF_values[counter] = values[1]
            counter += 1
    variable_location = folder + "all_enf_values.pkl"
    store_variable_file = open(variable_location,'wb')
    pickle.dump(all_ENF_values,store_variable_file)
    store_variable_file.close()
else:
    print("[INFO] Skipped reading individual csv, loading from pickle dump..")
    variable_location = folder + "all_enf_values.pkl"
    load_variable_file = open(variable_location, 'rb')
    all_ENF_values = pickle.load(load_variable_file)
    load_variable_file.close()
    print("[INFO] Loaded the variable..")

# converting the window size and shift size to seconds
sample_window_size = window_size * 60
sample_shift_size = shift_size * 60

# total number of correlation values
number_of_rho_values = math.ceil((total_samples - sample_window_size + 1) / sample_shift_size)
number_of_rho_values = 200
rho = np.zeros((number_of_rho_values,number_of_rho_values))
#creating a second copy of the master array with ENF values
duplicate_ENF_values = np.copy(all_ENF_values)

output_filename = folder + "correlation_values2.csv"
out = open(output_filename,"w")

for i in range(0,number_of_rho_values):
    signal1 = all_ENF_values[i*sample_shift_size : i*sample_shift_size + sample_window_size]
    for j in range(0,number_of_rho_values):
        signal2 = all_ENF_values[j * sample_shift_size: j * sample_shift_size + sample_window_size]
        r,p = pearsonr(signal1,signal2)
        rho[i][j] = r
        #if i==24 and j==49:
            #print(len(signal1))
        #    l = len(signal2)
        #    print("Plotting similars")
        #    plt.plot(range(1700),signal1[1800:3500])
        #    plt.plot(range(1700),signal2[1800:3500])
            #print(signal1[0:60])
            #print(signal2[0:60])
        #    print(rho[i][j])
        #    r,p = pearsonr(signal1[0:3500],signal2[0:3500])
        #    print(r)
        #    plt.show()
            
        out.write(str(rho[i][j])+",")
    out.write("\n")
out.close()
print(rho.shape)

