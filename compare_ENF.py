import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import math
import numpy as np


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


folder = "../ENF_data/"

csv1_filename = folder + "PCB1_power.csv"
csv2_filename = folder + "PCB2_power.csv"
csv3_filename = folder + "reference_power.csv"


pcb1_power = []
with open(csv1_filename,'r') as file:
    for each_line in file:
        x = each_line.split(',')
        y = float(x[1][:-2])
        pcb1_power.append(y)


pcb2_power = []
with open(csv2_filename,'r') as file:
    for each_line in file:
        x = each_line.split(',')
        y = float(x[1][:-2])
        pcb2_power.append(y)    

reference_power = []
with open(csv3_filename,'r') as file:
    for each_line in file:
        x = each_line.split(',')
        y = float(x[1][:-2])
        reference_power.append(y)    


window_size = 60
shift_size = 5
rho1,total_windows = correlation_vector(pcb1_power[:-5], pcb2_power[:-5],window_size,shift_size)
rho2,total_windows = correlation_vector(pcb2_power[:-5], reference_power[:-5],window_size,shift_size)


plt.figure()
t = np.arange(0,total_windows-1,1)
plt.plot(t,rho1[0][1:],'g', label="PCB1")
plt.plot(t,rho2[0][1:],'b', label="PCB2")
plt.hlines(y=0.8, xmin=0, xmax=len(t), colors='r', linestyles='--', lw=2)
plt.ylabel('Correlation Coefficient', fontsize=12)
plt.xlabel('Number of Windows compared', fontsize=12)
plt.title('ENF fluctuations compared', fontsize=12)
#plt.set_legend('With SSM','Without SSM')
plt.legend(loc="lower right")
#plt.show()

checkpoint = 500

plt.figure()
plt.plot(pcb1_power[checkpoint:checkpoint+100],'g', label="PCB1 Recording")
plt.plot(pcb2_power[checkpoint:checkpoint+100],'r', label="PCB2 Recording")
plt.plot(reference_power[checkpoint:checkpoint+100],'b', label="Reference Recording")

plt.ylabel('Frequency (Hz)', fontsize=14)
plt.xlabel('Time (sec)', fontsize=14)
plt.title('ENF Fluctuations', fontsize=14)
plt.legend(loc="lower right")
plt.show()
