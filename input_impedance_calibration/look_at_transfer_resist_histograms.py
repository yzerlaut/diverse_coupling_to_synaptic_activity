import numpy as np
import matplotlib.pylab as plt
import sys
sys.path.append('/Users/yzerlaut/work/common_libraries/')
from graphs.my_graph import set_plot
sys.path.append('../')
from theory.analytical_calculus import *
from data_firing_response.analyze_data import get_Rm_range
from input_impedance_calibration.get_calib import adjust_model_prop

# loading mean model
soma, stick, params = np.load('../input_impedance_calibration/mean_model.npy')

CELLS = np.load('../data_firing_response/reduced_data.npy')

Rtf_model = np.zeros(len(CELLS))
Rm_data =  np.zeros(len(CELLS))

fig, ax = plt.subplots(figsize=(4,3))
plt.subplots_adjust(bottom=.3, left=.3)

for i in range(len(CELLS))[::5]:
    
    Rm_data[i] = 1e-6/CELLS[i]['Gl']
    soma1, stick1, params1 = adjust_model_prop(Rm_data[i], soma, stick)
    
    Rtf_model, N_synapses = get_the_transfer_resistance_to_soma(soma1, stick1, params1)

    ax.hist(1e-6*Rtf_model, label='cell'+str(i))
    
ax.legend(frameon=False, prop={'size':'x-small'}, loc='best')
set_plot(ax, xlabel='transfer resistance \n to soma $(M \Omega)$', ylabel='synapses #')
plt.show()

