import numpy as np
import matplotlib.pylab as plt
import sys
sys.path.append('/home/yann/work/python_library')
from my_graph import set_plot
import fourier_for_real as rfft
sys.path.append('../')
from theory.analytical_calculus import *
from data_firing_response.analyze_data import get_Rm_range
from input_impedance_calibration.get_calib import adjust_model_prop

soma, stick, params = np.load('../input_impedance_calibration/mean_model.npy')

Rm = get_Rm_range()

Ke, Ki = 0*Rm, 0*Rm


for i in range(len(Rm)):
    soma1, stick1, params1 = adjust_model_prop(Rm[i], soma, stick)
    xtot1, cables1 = setup_model(soma1, stick1, params1)
    Ke[i], Ki[i] = cables1[0]['Ke_tot'], cables1[0]['Ki_tot']

print '-----------------------------------------------------------------'    
print 'mean number of synapses: ', np.mean(Ke+Ki), '+/-', np.std(Ke+Ki)
print 'max number of synapses: ', np.max(Ke+Ki)
print 'min number of synapses: ', np.min(Ke+Ki)
print 'excitatory/inhibitory : ', np.mean(Ke/Ki), '+/-', np.std(Ke/Ki)


fig, ax = plt.subplots(1, 3, figsize=(10,3))
plt.subplots_adjust(bottom=.3)
ax[0].hist(Ke, color='g');set_plot(ax[0], xlabel='exc. synapses', ylabel='cell #')
ax[1].hist(Ki, color='r');set_plot(ax[1], xlabel='inh. synapses')
ax[2].hist(np.array(Ke)/np.array(Ki), color='k');set_plot(ax[2], xlabel='ratio \n exc/inh #')
plt.show()
