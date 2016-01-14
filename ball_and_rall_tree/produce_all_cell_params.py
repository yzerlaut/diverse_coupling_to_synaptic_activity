import numpy as np
import sys
sys.path.append('../')
from input_impedance_calibration.get_calib import adjust_model_prop # where the core calculus lies
from firing_response_description.template_and_fitting import final_func

#### ================================================== ##
#### MEAN MODEL ###############################
#### ================================================== ##

soma, stick, params = np.load('../input_impedance_calibration/mean_model.npy')
stick['L_prox'] = stick['L']/2.

# --- fixing the synaptic densities !!

FACTOR = 1. # factor for the synaptic densities densities
soma['exc_density'], soma['inh_density']= 1e9, FACTOR*25.*1e-12
stick['exc_density'], stick['inh_density']= FACTOR*17*1e-12, FACTOR*100*1e-12
soma['exc_density'], soma['inh_density']= 1e9, FACTOR*25.*1e-12
stick['exc_density'], stick['inh_density']= FACTOR*17*1e-12, FACTOR*100*1e-12

# --- fixing the synaptic parameters !!
params['Qe'], params['Qi'] = 1e-9, 1.5e-9
params['Te'], params['Ti'] = 5e-3, 5e-3
params['Ee'], params['Ei'] = 0e-3, -80e-3

# data of the reduced morphologies
ALL_CELLS = np.load('../data_firing_response/reduced_data.npy')

for cell in ALL_CELLS:
    cell['Rm'] = 1e-6/cell['Gl']
    params1 = params.copy()
    soma1, stick1, params1 = adjust_model_prop(cell['Rm'], soma, stick)
    cell['soma'], cell['stick'], cell['params'] = soma1, stick1, params1
    cell['stick']['L_prox'] = cell['stick']['L']/2.

np.save('all_cell_params.npy', ALL_CELLS)
    
