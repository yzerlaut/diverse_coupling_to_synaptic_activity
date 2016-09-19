from neuron import h as nrn
# nrn.nrn_load_dll('x86_64/.libs/libnrnmech.so')
# nrnivmodl h.mod kdr2.mod kca.mod cad2.mod it2.mod SlowCa.mod hh3.mod km.mod kap.mod
nrn.xopen("build_cell.hoc")
# nrn.ca_hot_zone()

import numpy as np
import matplotlib.pylab as plt

Z = nrn.Impedance(sec=nrn.soma)
Z.loc(0.5)

R_transfer, Areas = [], []
for sec in nrn.allsec():
    for seg in sec.allseg():
        Z.compute(0, seg.x, sec=sec)
        Areas.append(seg.area())
        R_transfer.append(Z.transfer(seg.x, sec=sec))

edges, bins = np.histogram(R_transfer, bins=50, weights=Areas, normed=True)
plt.bar(bins[:-1], edges, width=bins[1]-bins[0])

np.savez('data/larkum_Tf_Resist_data.npz', R_transfer=R_transfer, Areas=Areas)

plt.show()
