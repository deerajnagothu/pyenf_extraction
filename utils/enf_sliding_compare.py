
import numpy as np
import math
import pickle
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
# Create a variable which contains all the values of ENF for each second


# converting the window size and shift size to seconds
sample_window_size =  60
sample_shift_size = 1

folder = "Recordings/sliding_window/"
file_header = "ENF_3min"
extension = ".csv"

read_values_again = True # Set this to True if you want to read the ENF values again from files




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



counter = 0
all_ENF_values = np.zeros(180)

if read_values_again is True:
    print("[INFO] Loading ENF values from csv files..")
    
    filename = folder+file_header + extension
    for line in open(filename,"r"):
        values = line.split(",")
        print(values)
        all_ENF_values[counter] = values[1]
        counter += 1
    variable_location = folder + "enf_values.pkl"
    store_variable_file = open(variable_location,'wb')
    pickle.dump(all_ENF_values,store_variable_file)
    store_variable_file.close()
else:
    print("[INFO] Skipped reading individual csv, loading from pickle dump..")
    variable_location = folder + "enf_values.pkl"
    load_variable_file = open(variable_location, 'rb')
    all_ENF_values = pickle.load(load_variable_file)
    load_variable_file.close()
    print("[INFO] Loaded the variable..")


#plt.plot(all_ENF_values)
#plt.title("Original ENF values")
#plt.show()


# total number of correlation values
number_of_rho_values = math.ceil((len(all_ENF_values) - sample_window_size + 1) / sample_shift_size)

rho = np.zeros(number_of_rho_values)
#creating a second copy of the master array with ENF values
duplicate_ENF_values = np.copy(all_ENF_values)

output_filename = folder + "correlation_values.csv"
out = open(output_filename,"w")


signal2 = all_ENF_values[60:60+sample_window_size]
print(len(signal2))

for i in range(0,number_of_rho_values):
    signal1 = all_ENF_values[i*sample_shift_size : i*sample_shift_size + sample_window_size]        
    r,p = pearsonr(signal1,signal2)
    rho[i] = r
    #if r >= 0.8:
        #print("Plotting similars")
        #plt.plot(range(sample_window_size),signal1)
        #plt.plot(range(sample_window_size),signal2)
        #plt.show()
            
    out.write(str(rho[i])+",")
    out.write("\n")
out.close()
plt.plot(range(number_of_rho_values),rho, marker="*",linestyle='None')
plt.axhline(y=0.8, color='r', linestyle='-')
plt.xlabel("Number of Windows")
plt.ylabel("Correlation Coefficient")
plt.title("Window = "+str(sample_window_size)+" & Shift ="+str(sample_shift_size))
plt.show()
print(rho.shape)

