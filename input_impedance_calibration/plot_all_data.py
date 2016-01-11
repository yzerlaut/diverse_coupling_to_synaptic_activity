import numpy as np
import matplotlib.pylab as plt
import os, sys
sys.path.append('/home/yann/work/python_library/')
from my_graph import set_plot

fig, ax = plt.subplots(figsize=(9,7))
MICE, RATS = [], []

for file in os.listdir("./intracellular_data/"):
    if file.endswith("_rat.txt"):
        freq, psd, phase = np.loadtxt("./intracellular_data/"+file, unpack=True)
        RATS = {'freq':freq, 'psd':psd, 'phase':phase}
    elif file.endswith(".txt"):
        freq, psd, phase = np.loadtxt("./intracellular_data/"+file, unpack=True)
        MICE = {'freq':freq, 'psd':psd, 'phase':phase}

        
plt.loglog(freq, psd/psd.max(), 'rD')
plt.loglog(freq, psd/psd.max(), 'bD')
plt.show()




