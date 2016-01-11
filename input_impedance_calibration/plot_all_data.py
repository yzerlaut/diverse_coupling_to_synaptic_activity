import numpy as np
import matplotlib.pylab as plt
import os, sys
sys.path.append('/home/yann/work/python_library/')
import my_graph as graph

fig, ax = plt.subplots(figsize=(9,7))
MICE, RATS = [], []

psd_boundaries = [100,1000]

for file in os.listdir("./intracellular_data/"):
    if file.endswith("_rat.txt"):
        freq, psd, phase = np.loadtxt("./intracellular_data/"+file, unpack=True)
        RATS = {'freq':freq, 'psd':psd, 'phase':phase}
    elif file.endswith(".txt"):
        freq, psd, phase = np.loadtxt("./intracellular_data/"+file, unpack=True)
        MICE = {'freq':freq, 'psd':psd, 'phase':phase}
    psd_boundaries[0] = min([psd_boundaries[0], psd.min()])
    psd_boundaries[1] = max([psd_boundaries[1], psd.max()])

mymap = graph.get_linear_colormap()
X = [100,200,500,1000]

for m in 
plt.loglog(freq, psd/psd.max(), 'rD')
plt.loglog(freq, psd/psd.max(), 'bD')
plt.show()



