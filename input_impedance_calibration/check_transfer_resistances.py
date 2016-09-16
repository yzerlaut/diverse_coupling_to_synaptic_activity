import numpy as np
import matplotlib.pylab as plt
import sys
sys.path.append('/Users/yzerlaut/work/common_libraries/')
from graphs.my_graph import set_plot
import data_analysis.fourier_for_real as rfft
sys.path.append('../')
from theory.analytical_calculus import *
from data_firing_response.analyze_data import get_Rm_range
from input_impedance_calibration.get_calib import adjust_model_prop

f=rfft.time_to_freq(1000, 1e-4)

# loading mean model
soma, stick, params = np.load('../input_impedance_calibration/mean_model.npy')

CELLS = np.load('../data_firing_response/reduced_data.npy')

Rtf_model = np.zeros(len(CELLS))
Rm_data =  np.zeros(len(CELLS))

for i in range(len(CELLS)):
    Rm_data[i] = 1e-6/CELLS[i]['Gl']
    soma1, stick1, params1 = adjust_model_prop(Rm_data[i], soma, stick)
    
    Rtf_model[i]= get_the_mean_transfer_resistance_to_soma(soma1, stick1, params1)
    
print('---------------------------------------------------')
print('Comparison between model and data for Tm')
print('MODEL, mean = ', 1e-6*Rtf_model.mean(), 'ms +/-', 1e-6*Rtf_model.std())
print('---------------------------------------------------')


fig, [ax,ax2] = plt.subplots(1, 2, figsize=(8,3))
plt.subplots_adjust(bottom=.3, wspace=.4)
# histogram
# ax.hist(1e3*Rtf_data, color='r', label='data')
ax.hist(1e-6*Rtf_model, color='b', label='model')
ax.legend(frameon=False, prop={'size':'x-small'}, loc='best')
set_plot(ax, xlabel='transfer resistance \n to soma $(M \Omega)$', ylabel='cell #')
# correl Rm-Tm
ax2.plot(Rm_data, 1e-6*Rtf_model, 'ob', label='model')
ax2.legend(prop={'size':'xx-small'}, loc='best')
set_plot(ax2, xlabel='somatic input \n resistance $(M \Omega)$', ylabel='transfer resistance \n to soma $(M \Omega)$')
plt.show()
