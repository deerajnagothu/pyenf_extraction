


import numpy as np
import matplotlib.pyplot as plt
folder = "Recordings/"
file = folder + "correlation_values2.csv"

temp_file = folder + "partial_correlation.csv"


data = np.genfromtxt(file, delimiter=",",usecols=range(100, 100)  )
plt.imshow(data, cmap='hot', interpolation='nearest')
heatmap = plt.pcolor(data)
plt.colorbar(heatmap)
plt.title("Correlation Coefficient Heapmap for 200 hours of continuous ENF data")
plt.show()