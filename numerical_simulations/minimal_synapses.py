from neuron import h as nrn
import numpy as np
import matplotlib.pylab as plt
from nrn_simulations import *


for ntar, color, label in zip([1, 0], ['b', 'r'], ['NMDA', 'no-NMDA']):

    nrn('create soma')
    nrn.soma.insert('pas')
    nrn.soma.L, nrn.soma.diam = 58, 58
    nrn.soma.g_pas, nrn.soma.e_pas = 5e-5, -70.
    syn = nrn.my_glutamate(0.5, sec=nrn.soma)
    netcon = nrn.NetCon(nrn.nil, syn)
    netcon.weight[0] = 1
    syn.gmax = 4
    def init_spike_train(ALL):
        netcon, spikes = ALL
        for spk in spikes:
            netcon.event(spk)
    syn.nmda_on = ntar
    fih = nrn.FInitializeHandler((init_spike_train, [netcon, np.arange(10)*7.+50.]))
    t_vec = nrn.Vector()
    t_vec.record(nrn._ref_t)
    V = nrn.Vector()
    V.record(nrn.soma(0.5)._ref_v)
    A = nrn.Vector()
    A.record(syn._ref_gampa)

    nrn.finitialize(-70.)
    while nrn.t<400.:
        nrn.fadvance()

    plt.plot(np.array(t_vec), V, '-', color=color, label=label)
plt.legend()    
plt.ylim([-80,-30])
plt.show()



