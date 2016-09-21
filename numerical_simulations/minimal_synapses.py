from neuron import h as nrn
import numpy as np
import matplotlib.pylab as plt
from nrn_simulations import *

soma = nrn.Section()
soma.insert('pas')
soma.L, soma.diam = 58, 58
soma.g_pas, soma.e_pas = 5e-5, -70.
syn = nrn.glutamateLarkum2009(0.5, sec=soma)
netcon = nrn.NetCon(nrn.nil, syn)
netcon.weight[0] = 3e-9*1e6





