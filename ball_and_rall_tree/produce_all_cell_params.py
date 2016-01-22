import numpy as np
import sys
sys.path.append('../')
from input_impedance_calibration.get_calib import adjust_model_prop # where the core calculus lies
from firing_response_description.template_and_fitting import final_func

#### ================================================== ##
#### MEAN MODEL ###############################
#### ================================================== ##

soma, stick, params = np.load('../input_impedance_calibration/mean_model.npy')

# --- fixing the synaptic densities !!

FACTOR = 1. # factor for the synaptic densities densities
soma['exc_density'], soma['inh_density']= 1e9, FACTOR*25.*1e-12
stick['exc_density'], stick['inh_density']= FACTOR*17*1e-12, FACTOR*100*1e-12
soma['exc_density'], soma['inh_density']= 1e9, FACTOR*25.*1e-12
stick['exc_density'], stick['inh_density']= FACTOR*17*1e-12, FACTOR*100*1e-12

# --- fixing the synaptic parameters !!
params['Qe'], params['Qi'] = .6e-9, 1.2e-9
params['Te'], params['Ti'] = 3e-3, 3e-3
params['Ee'], params['Ei'] = 0e-3, -80e-3
params['El'] = -60e-3#0e-3, -80e-3
params['factor_for_L_prox'] = 2./3.
params['factor_for_distal_synapses_weight'] = 3.
params['factor_for_distal_synapses_tau'] = 3.

# data of the reduced morphologies
ALL_CELLS = np.load('../data_firing_response/reduced_data.npy')

for cell in ALL_CELLS:
    cell['Rm'] = 1e-6/cell['Gl']
    soma1, stick1, params1 = adjust_model_prop(cell['Rm'], soma, stick)
    params1 = params.copy() # need to modify it AFTER :(
    cell['soma'], cell['stick'], cell['params'] = soma1, stick1, params1
    cell['stick']['L_prox'] = params1['factor_for_L_prox']*cell['stick']['L']
    

np.save('all_cell_params.npy', ALL_CELLS)
    
