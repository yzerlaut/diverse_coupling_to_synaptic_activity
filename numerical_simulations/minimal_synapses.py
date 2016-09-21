from neuron import h as nrn
import numpy as np
import matplotlib.pylab as plt
from nrn_simulations import *

nrn('create soma')
nrn.soma.insert('pas')
nrn.soma.L, nrn.soma.diam = 58, 58
nrn.soma.g_pas, nrn.soma.e_pas = 5e-5, -70.
syn = nrn.my_glutamate(0.5, sec=nrn.soma)
netcon = nrn.NetCon(nrn.nil, syn)
netcon.weight[0] = 3e-9*1e6

def init_spike_train(ALL):
    netcon, spikes = ALL
    for spk in spikes:
        netcon.event(spk)

# fih = nrn.FInitializeHandler((init_spike_train, [netcon, [50.,100.,150.]]))

# t_vec = nrn.Vector()
# t_vec.record(nrn._ref_t)
# V = []
# V.append(nrn.Vector())
# exec('V[0].record(nrn.cable_0_0(0)._ref_v)')

# nrn.finitialize(-70.)
# while nrn.t<200.:
#     nrn.fadvance()

# plt.plot(np.array(t_vec), V[0], 'k-')




