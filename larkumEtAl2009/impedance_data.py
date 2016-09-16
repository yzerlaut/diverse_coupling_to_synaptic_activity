from neuron import h as nrn
# nrn.nrn_load_dll('x86_64/.libs/libnrnmech.so')
# nrnivmodl h.mod kdr2.mod kca.mod cad2.mod it2.mod SlowCa.mod hh3.mod km.mod kap.mod
nrn.xopen("build_cell.hoc")
# nrn.ca_hot_zone()

import numpy as np
import matplotlib.pylab as plt

freq = np.logspace(-1.1, np.log(500.)/np.log(10), 50)
imped, phase = 0.*freq, 0.*freq

Z = nrn.Impedance(sec=nrn.soma)
Z.loc(0.5)

for i in range(len(freq)):
    Z.compute(freq[i] , 0.5)
    imped[i] = Z.input(0.5, sec=nrn.soma)
    phase[i] = Z.input_phase(0.5, sec=nrn.soma)

fig, AX = plt.subplots(1, 2, figsize=(10,3))    
AX[0].loglog(freq, imped)
AX[1].semilogx(freq, -phase)
plt.show()

np.save('data/larkum_imped_data.npy', [freq, imped, phase])
