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
# soma['exc_density'], soma['inh_density']= 1e9, FACTOR*25.*1e-12
# stick['exc_density'], stick['inh_density']= FACTOR*17*1e-12, FACTOR*100*1e-12
soma['exc_density'], soma['inh_density']= 1e9, (1e-5)**2/15.
stick['exc_density'], stick['inh_density']= (1e-5)**2/50., (1e-5)**2/10.

# --- fixing the synaptic parameters !!
params['Qe'], params['Qi'] = 1.e-9, 1.2e-9
params['Te'], params['Ti'] = 4e-3, 4e-3
params['Ee'], params['Ei'] = 0e-3, -80e-3
params['El'] = -60e-3#0e-3, -80e-3
params['fraction_for_L_prox'] = 2./3.
params['factor_for_distal_synapses_weight'] = 3.
params['factor_for_distal_synapses_tau'] = 3.

np.save('mean_model.npy', [soma, stick, params])

if sys.argv[-1]=='plot':

    sys.path.append('/home/yann/work/python_library/')
    from my_graph import *
    import matplotlib.pylab as plt
    
    fig, ax = plt.subplots(1,2, figsize=(6,3))
    ax[0].fill_between([0,1],[0,0],np.ones(2)*(1e-5)**2/soma['inh_density'], color='r')
    ax[0].fill_between([0,1],[0,0],np.ones(2)*(1e-5)**2/soma['exc_density'], color='g', lw=4)
    ax[1].fill_between([0,1],[0,0],np.ones(2)*(1e-5)**2/stick['exc_density'], color='g')
    ax[1].fill_between([0,1],[0,0],np.ones(2)*(1e-5)**2/stick['inh_density'], color='r')
    set_plot(ax[0], ['left'], ylabel='synaptic densities \n synapses/100$\mu m^2$', xticks=[])
    set_plot(ax[1], ['left'], ylabel='synaptic densities \n synapses/100$\mu m^2$', xticks=[])

    fig, ax = plt.subplots(1,2, figsize=(8,3))
    fig.suptitle('synaptic event')
    t = np.linspace(-1,30)*1e-3
    g = lambda Q,tau: Q*np.array([np.exp(-tt/tau) if tt>0 else 0 for tt in t])
    ax[0].plot(1e3*t, 1e9*g(params['Qe'], params['Te']), 'g-', lw=2)
    ax[0].plot(1e3*t, 1e9*g(params['Qi'], params['Ti']), 'r-', lw=2)
    ax[0].set_title('somatic and proximal')
    set_plot(ax[0], ylabel='G (nS)', xticks=[0,15,30], xlabel='time (ms)', ylim=[0,4], yticks=[0,2,4])

    ax[1].plot(1e3*t, 1e9*g(params['factor_for_distal_synapses_weight']*params['Qe'], params['factor_for_distal_synapses_tau']*params['Te']), 'g-', lw=2)
    ax[1].plot(1e3*t, 1e9*g(params['factor_for_distal_synapses_weight']*params['Qi'], params['factor_for_distal_synapses_tau']*params['Ti']), 'r-', lw=2)
    ax[1].set_title('distal')
    set_plot(ax[1], ylabel='G (nS)', xticks=[0,15,30], xlabel='time (ms)', ylim=[0,4], yticks=[0,2,4])
    plt.show()
else:
    # data of the reduced morphologies
    ALL_CELLS = np.load('../data_firing_response/reduced_data.npy')

    for cell in ALL_CELLS:
        cell['Rm'] = 1e-6/cell['Gl']
        soma1, stick1, params1 = adjust_model_prop(cell['Rm'], soma, stick, params2=params.copy())
        cell['soma'], cell['stick'], cell['params'] = soma1, stick1, params1

    np.save('all_cell_params.npy', ALL_CELLS)


